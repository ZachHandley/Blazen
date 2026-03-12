//! Workflow builder, runtime, and event loop.
//!
//! A [`Workflow`] is a named collection of steps with an event-driven
//! execution model. Events flow through an internal channel; the event loop
//! matches each event to registered step handlers and spawns them. Step
//! outputs are routed back into the queue until a [`StopEvent`] terminates
//! the loop.
//!
//! Use [`WorkflowBuilder`] to construct a validated [`Workflow`], then call
//! [`Workflow::run`] or [`Workflow::run_with_event`] to execute it.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::{broadcast, mpsc, oneshot};
use uuid::Uuid;
use zagents_events::{AnyEvent, Event, EventEnvelope, StartEvent, StopEvent};

use crate::context::Context;
use crate::error::WorkflowError;
use crate::handler::WorkflowHandler;
use crate::step::{StepOutput, StepRegistration};

/// Fluent builder for constructing a [`Workflow`].
pub struct WorkflowBuilder {
    name: String,
    steps: Vec<StepRegistration>,
    timeout: Option<Duration>,
}

impl WorkflowBuilder {
    /// Create a new builder with the given workflow name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            timeout: Some(Duration::from_secs(300)), // 5 min default
        }
    }

    /// Register a step.
    #[must_use]
    pub fn step(mut self, registration: StepRegistration) -> Self {
        self.steps.push(registration);
        self
    }

    /// Set the maximum execution time for the workflow.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Disable the execution timeout (workflow runs until `StopEvent`).
    #[must_use]
    pub fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Validate and build the workflow.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::ValidationFailed`] if no steps are registered.
    pub fn build(self) -> crate::error::Result<Workflow> {
        if self.steps.is_empty() {
            return Err(WorkflowError::ValidationFailed(
                "workflow must have at least one step".into(),
            ));
        }

        // Build the event-type -> handlers registry.
        let mut registry: HashMap<String, Vec<StepRegistration>> = HashMap::new();
        for step in self.steps {
            for &event_type in &step.accepts {
                registry
                    .entry(event_type.to_owned())
                    .or_default()
                    .push(step.clone());
            }
        }

        Ok(Workflow {
            name: self.name,
            step_registry: registry,
            timeout: self.timeout,
        })
    }
}

/// A validated, ready-to-run workflow.
pub struct Workflow {
    name: String,
    step_registry: HashMap<String, Vec<StepRegistration>>,
    timeout: Option<Duration>,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workflow")
            .field("name", &self.name)
            .field("step_count", &self.step_registry.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Workflow {
    /// Execute the workflow with a raw JSON payload wrapped in a
    /// [`StartEvent`].
    ///
    /// Returns a [`WorkflowHandler`] that can be awaited for the final result
    /// or used to stream intermediate events.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub async fn run(
        &self,
        input: serde_json::Value,
    ) -> crate::error::Result<WorkflowHandler> {
        let start_event = StartEvent { data: input };
        self.run_with_event(start_event).await
    }

    /// Execute the workflow with a custom start event.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub async fn run_with_event<E: Event + Serialize>(
        &self,
        start_event: E,
    ) -> crate::error::Result<WorkflowHandler> {
        // Internal routing channel (unbounded so steps never block).
        let (event_tx, event_rx) = mpsc::unbounded_channel::<EventEnvelope>();

        // External broadcast channel for streaming.
        let (stream_tx, _stream_rx) = broadcast::channel::<Box<dyn AnyEvent>>(256);

        // Oneshot for the final result.
        let (result_tx, result_rx) = oneshot::channel();

        // Build the shared context.
        let ctx = Context::new(event_tx.clone(), stream_tx.clone());

        // Set metadata.
        let run_id = Uuid::new_v4();
        ctx.set_metadata(
            "run_id",
            serde_json::Value::String(run_id.to_string()),
        )
        .await;
        ctx.set_metadata(
            "workflow_name",
            serde_json::Value::String(self.name.clone()),
        )
        .await;

        // Enqueue the initial event.
        let envelope = EventEnvelope::new(Box::new(start_event), None);
        event_tx
            .send(envelope)
            .map_err(|_| WorkflowError::ChannelClosed)?;

        // Spawn the event loop.
        let registry = self.step_registry.clone();
        let timeout = self.timeout;
        tokio::spawn(event_loop(event_rx, event_tx, registry, ctx, result_tx, timeout));

        Ok(WorkflowHandler::new(result_rx, stream_tx))
    }
}

// ---------------------------------------------------------------------------
// Event loop
// ---------------------------------------------------------------------------

/// Core event loop that drives workflow execution.
///
/// Runs in a spawned task. Receives events from the channel, routes them to
/// matching step handlers, and injects step outputs back into the channel.
/// Terminates when a [`StopEvent`] arrives or the timeout elapses.
async fn event_loop(
    mut event_rx: mpsc::UnboundedReceiver<EventEnvelope>,
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    registry: HashMap<String, Vec<StepRegistration>>,
    ctx: Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    timeout: Option<Duration>,
) {
    let start = Instant::now();

    // Channel for step errors -- steps run in spawned tasks and report
    // failures back here so the event loop can terminate.
    let (error_tx, mut error_rx) = mpsc::unbounded_channel::<WorkflowError>();

    loop {
        // Calculate remaining time for timeout.
        let recv_result = if let Some(timeout_dur) = timeout {
            let remaining = timeout_dur.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                let _ = result_tx.send(Err(WorkflowError::Timeout {
                    elapsed: start.elapsed(),
                }));
                return;
            }
            // Select between event channel, error channel, and timeout.
            tokio::select! {
                biased;

                err = error_rx.recv() => {
                    if let Some(workflow_err) = err {
                        let _ = result_tx.send(Err(workflow_err));
                        return;
                    }
                    continue;
                }
                maybe_envelope = event_rx.recv() => {
                    maybe_envelope.ok_or(())
                }
                () = tokio::time::sleep(remaining) => {
                    let _ = result_tx.send(Err(WorkflowError::Timeout {
                        elapsed: start.elapsed(),
                    }));
                    return;
                }
            }
        } else {
            // No timeout -- select between events and errors only.
            tokio::select! {
                biased;

                err = error_rx.recv() => {
                    if let Some(workflow_err) = err {
                        let _ = result_tx.send(Err(workflow_err));
                        return;
                    }
                    continue;
                }
                maybe_envelope = event_rx.recv() => {
                    maybe_envelope.ok_or(())
                }
            }
        };

        let Ok(envelope) = recv_result else {
            let _ = result_tx.send(Err(WorkflowError::ChannelClosed));
            return;
        };

        let event = envelope.event;
        let event_type = event.event_type_id();

        tracing::debug!(
            event_type,
            source_step = ?envelope.source_step,
            "event loop received event"
        );

        // Check for StopEvent -- terminates the loop.
        if event_type == StopEvent::event_type() {
            tracing::info!("workflow completed via StopEvent");
            let _ = result_tx.send(Ok(event));
            return;
        }

        // Look up step handlers for this event type.
        let Some(handlers) = registry.get(event_type) else {
            tracing::warn!(event_type, "no handler registered for event type");
            let _ = result_tx.send(Err(WorkflowError::NoHandler {
                event_type: event_type.to_owned(),
            }));
            return;
        };
        let handlers = handlers.clone();

        // Also push the event into the fan-in accumulator.
        ctx.push_collected(event.clone()).await;

        // Dispatch to each matching handler.
        dispatch_to_handlers(&handlers, &*event, &ctx, &event_tx, &error_tx);
    }
}

/// Spawn step handler tasks for each matching step registration.
fn dispatch_to_handlers(
    handlers: &[StepRegistration],
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
) {
    for step in handlers {
        let event_clone = event.clone_boxed();
        let ctx_clone = ctx.clone();
        let handler = step.handler.clone();
        let step_name = step.name.clone();
        let event_tx_clone = event_tx.clone();
        let error_tx_clone = error_tx.clone();

        tokio::spawn(async move {
            match handler(event_clone, ctx_clone).await {
                Ok(StepOutput::Single(output_event)) => {
                    let envelope =
                        EventEnvelope::new(output_event, Some(step_name));
                    let _ = event_tx_clone.send(envelope);
                }
                Ok(StepOutput::Multiple(events)) => {
                    for e in events {
                        let envelope =
                            EventEnvelope::new(e, Some(step_name.clone()));
                        let _ = event_tx_clone.send(envelope);
                    }
                }
                Ok(StepOutput::None) => {
                    // Side-effect only step -- nothing to route.
                }
                Err(err) => {
                    tracing::error!(
                        step = %step_name,
                        error = %err,
                        "step failed"
                    );
                    let _ = error_tx_clone.send(WorkflowError::StepFailed {
                        step_name,
                        source: Box::new(err),
                    });
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use zagents_events::{StartEvent, StopEvent};

    use crate::step::{StepFn, StepOutput, StepRegistration};

    /// Helper: a step that converts a `StartEvent` into a `StopEvent`.
    fn echo_step() -> StepRegistration {
        let handler: StepFn = Arc::new(|event, _ctx| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .expect("expected StartEvent");
                let stop = StopEvent {
                    result: start.data.clone(),
                };
                Ok(StepOutput::Single(Box::new(stop)))
            })
        });

        StepRegistration {
            name: "echo".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 0,
        }
    }

    #[tokio::test]
    async fn simple_start_to_stop() {
        let workflow = WorkflowBuilder::new("test")
            .step(echo_step())
            .build()
            .unwrap();

        let handler = workflow.run(serde_json::json!({"hello": "world"})).await.unwrap();
        let result = handler.result().await.unwrap();
        assert_eq!(result.event_type_id(), StopEvent::event_type());

        let stop = result.downcast_ref::<StopEvent>().unwrap();
        assert_eq!(stop.result, serde_json::json!({"hello": "world"}));
    }

    #[tokio::test]
    async fn empty_workflow_fails_validation() {
        let result = WorkflowBuilder::new("empty").build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, WorkflowError::ValidationFailed(_)));
    }

    #[tokio::test]
    async fn timeout_triggers() {
        // A step that never produces output.
        let handler: StepFn = Arc::new(|_event, _ctx| {
            Box::pin(async move {
                // Sleep forever.
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(StepOutput::None)
            })
        });

        let step = StepRegistration {
            name: "slow".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![],
            handler,
            max_concurrency: 0,
        };

        let workflow = WorkflowBuilder::new("timeout-test")
            .step(step)
            .timeout(Duration::from_millis(50))
            .build()
            .unwrap();

        let wf_handler = workflow.run(serde_json::json!(null)).await.unwrap();
        let result = wf_handler.result().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::Timeout { .. }
        ));
    }

    #[tokio::test]
    async fn step_error_propagates() {
        let handler: StepFn = Arc::new(|_event, _ctx| {
            Box::pin(async move {
                Err(WorkflowError::Context("test error".into()))
            })
        });

        let step = StepRegistration {
            name: "failing".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![],
            handler,
            max_concurrency: 0,
        };

        let workflow = WorkflowBuilder::new("error-test")
            .step(step)
            .build()
            .unwrap();

        let wf_handler = workflow.run(serde_json::json!(null)).await.unwrap();
        let result = wf_handler.result().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::StepFailed { .. }
        ));
    }
}
