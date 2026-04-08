//! Workflow struct and runtime methods.
//!
//! A [`Workflow`] is a named collection of steps with an event-driven
//! execution model. Events flow through an internal channel; the event loop
//! matches each event to registered step handlers and spawns them. Step
//! outputs are routed back into the queue until a [`StopEvent`] terminates
//! the loop.
//!
//! Use [`WorkflowBuilder`](crate::builder::WorkflowBuilder) to construct a
//! validated [`Workflow`], then call [`Workflow::run`] or
//! [`Workflow::run_with_event`] to execute it.
//!
//! ## Pause / Resume
//!
//! A running workflow can be paused via [`WorkflowHandler::pause`]. This
//! waits for all in-flight step tasks to complete, drains pending events from
//! the channel, snapshots the full context state, and returns a serializable
//! [`WorkflowSnapshot`](crate::snapshot::WorkflowSnapshot). The workflow can
//! later be resumed from the snapshot via [`Workflow::resume`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use blazen_events::{AnyEvent, DynamicEvent, Event, EventEnvelope, StartEvent};
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, oneshot};
use uuid::Uuid;

use crate::builder::InputHandlerFn;
use crate::context::Context;
use crate::error::WorkflowError;
#[cfg(feature = "persist")]
use crate::event_loop::CheckpointConfig;
use crate::event_loop::event_loop;
use crate::handler::{WorkflowControl, WorkflowHandler};
use crate::session_ref::SessionRefRegistry;
use crate::snapshot::WorkflowSnapshot;
use crate::step::StepRegistration;

/// A validated, ready-to-run workflow.
pub struct Workflow {
    pub(crate) name: String,
    pub(crate) step_registry: HashMap<String, Vec<StepRegistration>>,
    pub(crate) timeout: Option<Duration>,
    /// Optional inline handler for input requests (HITL without pausing).
    pub(crate) input_handler: Option<InputHandlerFn>,
    /// Whether to automatically publish lifecycle events to the broadcast stream.
    pub(crate) auto_publish_events: bool,
    /// Policy applied to live session references when the workflow is
    /// paused or snapshotted.
    pub(crate) session_pause_policy: crate::session_ref::SessionPausePolicy,
    /// Checkpoint store for durable persistence (requires `persist` feature).
    #[cfg(feature = "persist")]
    pub(crate) checkpoint_store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    /// Whether to automatically checkpoint after each step completes.
    #[cfg(feature = "persist")]
    pub(crate) checkpoint_after_step: bool,
    /// Whether to collect an append-only history of workflow events (requires `telemetry` feature).
    #[cfg(feature = "telemetry")]
    pub(crate) collect_history: bool,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workflow")
            .field("name", &self.name)
            .field("step_count", &self.step_registry.len())
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
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
    pub async fn run(&self, input: serde_json::Value) -> crate::error::Result<WorkflowHandler> {
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
        self.run_with_event_and_session_refs(start_event, None)
            .await
    }

    /// Run this workflow with an externally-supplied session-ref registry.
    ///
    /// Used by the pipeline crate to share one registry across stages so
    /// `__blazen_session_ref__` markers produced in one stage can be
    /// resolved in subsequent stages.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub async fn run_with_registry(
        &self,
        input: serde_json::Value,
        session_refs: Arc<SessionRefRegistry>,
    ) -> crate::error::Result<WorkflowHandler> {
        let start_event = StartEvent { data: input };
        self.run_with_event_and_session_refs(start_event, Some(session_refs))
            .await
    }

    /// Internal helper that runs the workflow with an optional
    /// externally-supplied session-ref registry.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub(crate) async fn run_with_event_and_session_refs<E: Event + Serialize>(
        &self,
        start_event: E,
        session_refs: Option<Arc<SessionRefRegistry>>,
    ) -> crate::error::Result<WorkflowHandler> {
        // Internal routing channel (unbounded so steps never block).
        let (event_tx, event_rx) = mpsc::unbounded_channel::<EventEnvelope>();

        // External broadcast channel for streaming.
        let (stream_tx, _stream_rx) = broadcast::channel::<Box<dyn AnyEvent>>(256);

        // Oneshot for the final result.
        let (result_tx, result_rx) = oneshot::channel();

        // Control channel (handler → event loop).
        let (control_tx, control_rx) = mpsc::unbounded_channel::<WorkflowControl>();

        // Build the shared context, optionally reusing an externally-supplied
        // session-ref registry so cross-workflow `__blazen_session_ref__`
        // markers remain resolvable.
        let ctx = match session_refs {
            Some(refs) => Context::new_with_session_refs(event_tx.clone(), stream_tx.clone(), refs),
            None => Context::new(event_tx.clone(), stream_tx.clone()),
        };

        // Apply the configured session pause policy.
        ctx.set_session_pause_policy(self.session_pause_policy)
            .await;

        // Capture an Arc to the context's session-ref registry so the
        // returned `WorkflowHandler` can resolve `__blazen_session_ref__`
        // markers in the final result even after the event loop drops
        // the original `Context`.
        let session_refs = ctx.session_refs_arc().await;

        // Set metadata.
        let run_id = Uuid::new_v4();
        ctx.set_metadata("run_id", serde_json::Value::String(run_id.to_string()))
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

        // Create history channel if telemetry is enabled and history collection is on.
        #[cfg(feature = "telemetry")]
        let (history_tx, history_rx) = if self.collect_history {
            let (tx, rx) = mpsc::unbounded_channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // Spawn the event loop.
        let registry = self.step_registry.clone();
        let timeout = self.timeout;
        let workflow_name = self.name.clone();
        let input_handler = self.input_handler.clone();
        let auto_publish = self.auto_publish_events;

        #[cfg(feature = "persist")]
        let checkpoint_config = CheckpointConfig {
            store: self.checkpoint_store.clone(),
            after_step: self.checkpoint_after_step,
        };

        let event_loop_handle = tokio::spawn(event_loop(
            event_rx,
            event_tx,
            registry,
            ctx,
            result_tx,
            timeout,
            control_rx,
            workflow_name,
            run_id,
            input_handler,
            auto_publish,
            #[cfg(feature = "persist")]
            checkpoint_config,
            #[cfg(feature = "telemetry")]
            history_tx,
        ));

        Ok(WorkflowHandler::new(
            result_rx,
            stream_tx,
            control_tx,
            event_loop_handle,
            session_refs,
            #[cfg(feature = "telemetry")]
            history_rx,
        ))
    }

    /// Resume a workflow from a previously captured snapshot.
    ///
    /// The step registry must be provided again since step handler functions
    /// are not serializable. The steps should be the same set (or a
    /// compatible superset) that were registered when the workflow was
    /// originally built.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot's pending events cannot be
    /// reinjected into the event channel.
    pub async fn resume(
        snapshot: WorkflowSnapshot,
        steps: Vec<StepRegistration>,
        timeout: Option<Duration>,
    ) -> crate::error::Result<WorkflowHandler> {
        // Rebuild the registry from the provided steps.
        let mut registry: HashMap<String, Vec<StepRegistration>> = HashMap::new();
        for step in steps {
            for &event_type in &step.accepts {
                registry
                    .entry(event_type.to_owned())
                    .or_default()
                    .push(step.clone());
            }
        }

        // Internal routing channel.
        let (event_tx, event_rx) = mpsc::unbounded_channel::<EventEnvelope>();

        // External broadcast channel.
        let (stream_tx, _stream_rx) = broadcast::channel::<Box<dyn AnyEvent>>(256);

        // Result channel.
        let (result_tx, result_rx) = oneshot::channel();

        // Control channel (handler → event loop).
        let (control_tx, control_rx) = mpsc::unbounded_channel::<WorkflowControl>();

        // Build context and restore state.
        let ctx = Context::new(event_tx.clone(), stream_tx.clone());
        ctx.restore_state(snapshot.context_state).await;
        ctx.restore_collected(snapshot.collected_events).await;
        ctx.restore_metadata(snapshot.metadata).await;

        // Capture an Arc to the resumed context's session-ref registry
        // (empty after a cross-process resume) so the handler can later
        // surface a clear error if a marker references a now-missing
        // entry.
        let session_refs = ctx.session_refs_arc().await;

        // Reinject pending events into the channel.
        // Try to reconstruct concrete event types via the deserializer
        // registry first; fall back to DynamicEvent if no deserializer is
        // registered or deserialization fails.
        for serialized in &snapshot.pending_events {
            let event: Box<dyn AnyEvent> =
                blazen_events::try_deserialize_event(&serialized.event_type, &serialized.data)
                    .unwrap_or_else(|| {
                        Box::new(DynamicEvent {
                            event_type: serialized.event_type.clone(),
                            data: serialized.data.clone(),
                        })
                    });
            let envelope = EventEnvelope::new(event, serialized.source_step.clone());
            event_tx
                .send(envelope)
                .map_err(|_| WorkflowError::ChannelClosed)?;
        }

        // Spawn the event loop.
        let workflow_name = snapshot.workflow_name;
        let run_id = snapshot.run_id;

        // Resumed workflows do not collect history (no builder config available).
        #[cfg(feature = "telemetry")]
        let history_tx: Option<mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>> = None;

        #[cfg(feature = "persist")]
        let checkpoint_config = CheckpointConfig {
            store: None,
            after_step: false,
        };

        let event_loop_handle = tokio::spawn(event_loop(
            event_rx,
            event_tx,
            registry,
            ctx,
            result_tx,
            timeout,
            control_rx,
            workflow_name,
            run_id,
            None,  // No inline input handler for resumed workflows.
            false, // No auto-publish for resumed workflows.
            #[cfg(feature = "persist")]
            checkpoint_config,
            #[cfg(feature = "telemetry")]
            history_tx,
        ));

        Ok(WorkflowHandler::new(
            result_rx,
            stream_tx,
            control_tx,
            event_loop_handle,
            session_refs,
            #[cfg(feature = "telemetry")]
            None, // No history receiver for resumed workflows.
        ))
    }

    /// Resume a workflow from a checkpoint stored in a [`CheckpointStore`].
    ///
    /// Loads the checkpoint identified by `run_id`, converts it to a
    /// [`WorkflowSnapshot`], and resumes using [`Workflow::resume`].
    ///
    /// The step registrations must be provided again since step handler
    /// functions are not serializable.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Context`] if no checkpoint exists for the
    /// given `run_id`, or propagates any storage error from the checkpoint
    /// store.
    #[cfg(feature = "persist")]
    pub async fn resume_from(
        store: Arc<dyn blazen_persist::CheckpointStore>,
        run_id: &Uuid,
        steps: Vec<StepRegistration>,
    ) -> crate::error::Result<WorkflowHandler> {
        let checkpoint = store
            .load(run_id)
            .await
            .map_err(|e| WorkflowError::Context(format!("checkpoint load failed: {e}")))?
            .ok_or_else(|| {
                WorkflowError::Context(format!("no checkpoint found for run_id {run_id}"))
            })?;

        let snapshot: WorkflowSnapshot = checkpoint.into();

        // Use a default 5-minute timeout for resumed workflows.
        Self::resume(snapshot, steps, Some(Duration::from_secs(300))).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use blazen_events::{Event, StartEvent, StopEvent};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::builder::WorkflowBuilder;
    use crate::error::WorkflowError;
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

        let handler = workflow
            .run(serde_json::json!({"hello": "world"}))
            .await
            .unwrap();
        let result = handler.result().await.unwrap().event;
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
        assert!(matches!(result.unwrap_err(), WorkflowError::Timeout { .. }));
    }

    #[tokio::test]
    async fn step_error_propagates() {
        let handler: StepFn = Arc::new(|_event, _ctx| {
            Box::pin(async move { Err(WorkflowError::Context("test error".into())) })
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
