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
//!
//! ## Pause / Resume
//!
//! A running workflow can be paused via [`WorkflowHandler::pause`]. This
//! waits for all in-flight step tasks to complete, drains pending events from
//! the channel, snapshots the full context state, and returns a serializable
//! [`WorkflowSnapshot`](crate::snapshot::WorkflowSnapshot). The workflow can
//! later be resumed from the snapshot via [`Workflow::resume`].

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use blazen_events::{
    AnyEvent, DynamicEvent, Event, EventEnvelope, InputRequestEvent, InputResponseEvent,
    StartEvent, StopEvent,
};
use chrono::Utc;
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::context::Context;
use crate::error::WorkflowError;
use crate::handler::WorkflowHandler;
use crate::snapshot::{SerializedEvent, WorkflowSnapshot};
use crate::step::{StepOutput, StepRegistration};

/// Async callback for handling input requests inline (without pausing).
///
/// When registered on a [`WorkflowBuilder`], the event loop will invoke this
/// callback instead of auto-pausing when an [`InputRequestEvent`] arrives.
/// The callback should return an [`InputResponseEvent`] which will be
/// injected back into the event queue.
pub type InputHandlerFn = Arc<
    dyn Fn(
            InputRequestEvent,
        )
            -> Pin<Box<dyn Future<Output = Result<InputResponseEvent, WorkflowError>> + Send>>
        + Send
        + Sync,
>;

/// Fluent builder for constructing a [`Workflow`].
pub struct WorkflowBuilder {
    name: String,
    steps: Vec<StepRegistration>,
    timeout: Option<Duration>,
    /// Optional inline handler for input requests (HITL without pausing).
    input_handler: Option<InputHandlerFn>,
    /// Checkpoint store for durable persistence (requires `persist` feature).
    #[cfg(feature = "persist")]
    checkpoint_store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    /// Whether to automatically checkpoint after each step completes.
    #[cfg(feature = "persist")]
    checkpoint_after_step: bool,
}

impl WorkflowBuilder {
    /// Create a new builder with the given workflow name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            timeout: Some(Duration::from_secs(300)), // 5 min default
            input_handler: None,
            #[cfg(feature = "persist")]
            checkpoint_store: None,
            #[cfg(feature = "persist")]
            checkpoint_after_step: false,
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

    /// Register an inline handler for [`InputRequestEvent`]s.
    ///
    /// When set, the event loop will call this handler instead of
    /// auto-pausing when an input request arrives. The handler should
    /// return an [`InputResponseEvent`] which is injected back into the
    /// event queue, allowing the workflow to continue without interruption.
    #[must_use]
    pub fn input_handler(mut self, handler: InputHandlerFn) -> Self {
        self.input_handler = Some(handler);
        self
    }

    /// Set the checkpoint store for durable persistence.
    ///
    /// When a checkpoint store is configured, the workflow can persist its
    /// state to durable storage for crash recovery or migration.
    ///
    /// Requires the `persist` feature.
    #[cfg(feature = "persist")]
    #[must_use]
    pub fn checkpoint_store(mut self, store: Arc<dyn blazen_persist::CheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }

    /// Enable or disable automatic checkpointing after each step completes.
    ///
    /// When enabled (and a checkpoint store is configured), the workflow will
    /// save a checkpoint after each event is dispatched to step handlers.
    /// Checkpointing is best-effort: errors are logged but do not fail the
    /// workflow.
    ///
    /// Defaults to `false`.
    ///
    /// Requires the `persist` feature.
    #[cfg(feature = "persist")]
    #[must_use]
    pub fn checkpoint_after_step(mut self, enabled: bool) -> Self {
        self.checkpoint_after_step = enabled;
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
            input_handler: self.input_handler,
            #[cfg(feature = "persist")]
            checkpoint_store: self.checkpoint_store,
            #[cfg(feature = "persist")]
            checkpoint_after_step: self.checkpoint_after_step,
        })
    }
}

/// A validated, ready-to-run workflow.
pub struct Workflow {
    name: String,
    step_registry: HashMap<String, Vec<StepRegistration>>,
    timeout: Option<Duration>,
    /// Optional inline handler for input requests (HITL without pausing).
    input_handler: Option<InputHandlerFn>,
    /// Checkpoint store for durable persistence (requires `persist` feature).
    #[cfg(feature = "persist")]
    checkpoint_store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    /// Whether to automatically checkpoint after each step completes.
    #[cfg(feature = "persist")]
    checkpoint_after_step: bool,
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
        // Internal routing channel (unbounded so steps never block).
        let (event_tx, event_rx) = mpsc::unbounded_channel::<EventEnvelope>();

        // External broadcast channel for streaming.
        let (stream_tx, _stream_rx) = broadcast::channel::<Box<dyn AnyEvent>>(256);

        // Oneshot for the final result.
        let (result_tx, result_rx) = oneshot::channel();

        // Pause/snapshot channels.
        let (pause_tx, pause_rx) = oneshot::channel::<()>();
        let (snapshot_tx, snapshot_rx) = oneshot::channel::<WorkflowSnapshot>();

        // Build the shared context.
        let ctx = Context::new(event_tx.clone(), stream_tx.clone());

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

        // Spawn the event loop.
        let registry = self.step_registry.clone();
        let timeout = self.timeout;
        let workflow_name = self.name.clone();
        let input_handler = self.input_handler.clone();

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
            pause_rx,
            snapshot_tx,
            workflow_name,
            run_id,
            input_handler,
            #[cfg(feature = "persist")]
            checkpoint_config,
        ));

        Ok(WorkflowHandler::new(
            result_rx,
            stream_tx,
            Some(pause_tx),
            Some(snapshot_rx),
            event_loop_handle,
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

        // Pause/snapshot channels for the resumed workflow.
        let (pause_tx, pause_rx) = oneshot::channel::<()>();
        let (snapshot_tx, snapshot_rx) = oneshot::channel::<WorkflowSnapshot>();

        // Build context and restore state.
        let ctx = Context::new(event_tx.clone(), stream_tx.clone());
        ctx.restore_state(snapshot.context_state).await;
        ctx.restore_collected(snapshot.collected_events).await;
        ctx.restore_metadata(snapshot.metadata).await;

        // Reinject pending events into the channel.
        for serialized in &snapshot.pending_events {
            let dynamic_event = DynamicEvent {
                event_type: serialized.event_type.clone(),
                data: serialized.data.clone(),
            };
            let envelope =
                EventEnvelope::new(Box::new(dynamic_event), serialized.source_step.clone());
            event_tx
                .send(envelope)
                .map_err(|_| WorkflowError::ChannelClosed)?;
        }

        // Spawn the event loop.
        let workflow_name = snapshot.workflow_name;
        let run_id = snapshot.run_id;

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
            pause_rx,
            snapshot_tx,
            workflow_name,
            run_id,
            None, // No inline input handler for resumed workflows.
            #[cfg(feature = "persist")]
            checkpoint_config,
        ));

        Ok(WorkflowHandler::new(
            result_rx,
            stream_tx,
            Some(pause_tx),
            Some(snapshot_rx),
            event_loop_handle,
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
    /// Returns [`WorkflowError::SnapshotNotFound`] if no checkpoint exists
    /// for the given `run_id`, or propagates any storage error from the
    /// checkpoint store.
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

    /// Resume a paused workflow, injecting a human's response.
    ///
    /// This is a convenience method for workflows that auto-paused due to an
    /// [`InputRequestEvent`]. It injects the [`InputResponseEvent`] into the
    /// snapshot's pending events and resumes execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot's pending events cannot be
    /// reinjected into the event channel.
    ///
    /// # Panics
    ///
    /// Panics if `InputResponseEvent` cannot be serialized to JSON, which
    /// should never happen for a well-formed serde type.
    pub async fn resume_with_input(
        snapshot: WorkflowSnapshot,
        response: InputResponseEvent,
        steps: Vec<StepRegistration>,
        timeout: Option<Duration>,
    ) -> crate::error::Result<WorkflowHandler> {
        let mut snapshot = snapshot;

        // Inject the response as a pending event.
        snapshot.pending_events.push(SerializedEvent {
            event_type: "blazen::InputResponseEvent".to_owned(),
            data: serde_json::to_value(&response)
                .expect("InputResponseEvent serialization should never fail"),
            source_step: Some("__human_input".to_owned()),
        });

        // Clear the input request from metadata.
        snapshot.metadata.remove("__input_request");

        Self::resume(snapshot, steps, timeout).await
    }
}

// ---------------------------------------------------------------------------
// Checkpoint configuration (persist feature)
// ---------------------------------------------------------------------------

/// Internal configuration for the checkpoint system, passed from the builder
/// to the event loop.
#[cfg(feature = "persist")]
struct CheckpointConfig {
    store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    after_step: bool,
}

/// Save a checkpoint of the current workflow state to the configured store.
///
/// This is a best-effort operation: errors are logged but do not propagate.
#[cfg(feature = "persist")]
async fn save_checkpoint(
    store: &dyn blazen_persist::CheckpointStore,
    ctx: &Context,
    workflow_name: &str,
    run_id: Uuid,
) {
    let context_state = ctx.snapshot_state().await;
    let collected_events = ctx.snapshot_collected().await;
    let metadata = ctx.snapshot_metadata().await;

    let snapshot = WorkflowSnapshot {
        workflow_name: workflow_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        context_state,
        collected_events,
        pending_events: Vec::new(), // Cannot peek at the channel non-destructively.
        metadata,
    };

    let checkpoint: blazen_persist::WorkflowCheckpoint = snapshot.into();
    if let Err(e) = store.save(&checkpoint).await {
        tracing::warn!(
            run_id = %run_id,
            error = %e,
            "auto-checkpoint failed (best-effort)"
        );
    } else {
        tracing::debug!(run_id = %run_id, "auto-checkpoint saved");
    }
}

// ---------------------------------------------------------------------------
// Event loop
// ---------------------------------------------------------------------------

/// Core event loop that drives workflow execution.
///
/// Runs in a spawned task. Receives events from the channel, routes them to
/// matching step handlers, and injects step outputs back into the channel.
/// Terminates when a [`StopEvent`] arrives, the timeout elapses, or a pause
/// signal is received.
///
/// This wrapper ensures that a `"blazen::StreamEnd"` sentinel is always sent
/// through the broadcast stream when the event loop exits, regardless of the
/// exit path. This allows stream consumers to detect completion.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn event_loop(
    event_rx: mpsc::UnboundedReceiver<EventEnvelope>,
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    registry: HashMap<String, Vec<StepRegistration>>,
    ctx: Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    timeout: Option<Duration>,
    pause_rx: oneshot::Receiver<()>,
    snapshot_tx: oneshot::Sender<WorkflowSnapshot>,
    workflow_name: String,
    run_id: Uuid,
    input_handler: Option<InputHandlerFn>,
    #[cfg(feature = "persist")] checkpoint_config: CheckpointConfig,
) {
    let stream_ctx = ctx.clone();
    event_loop_inner(
        event_rx,
        event_tx,
        registry,
        ctx,
        result_tx,
        timeout,
        pause_rx,
        snapshot_tx,
        workflow_name,
        run_id,
        input_handler,
        #[cfg(feature = "persist")]
        checkpoint_config,
    )
    .await;
    stream_ctx.signal_stream_end().await;
}

/// Inner event loop implementation. See [`event_loop`] for the public wrapper
/// that guarantees stream-end signaling.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn event_loop_inner(
    mut event_rx: mpsc::UnboundedReceiver<EventEnvelope>,
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    registry: HashMap<String, Vec<StepRegistration>>,
    ctx: Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    timeout: Option<Duration>,
    mut pause_rx: oneshot::Receiver<()>,
    snapshot_tx: oneshot::Sender<WorkflowSnapshot>,
    workflow_name: String,
    run_id: Uuid,
    input_handler: Option<InputHandlerFn>,
    #[cfg(feature = "persist")] checkpoint_config: CheckpointConfig,
) {
    let start = Instant::now();

    // Channel for step errors -- steps run in spawned tasks and report
    // failures back here so the event loop can terminate.
    let (error_tx, mut error_rx) = mpsc::unbounded_channel::<WorkflowError>();

    // Track in-flight step tasks so we can wait for them during pause.
    let mut in_flight: JoinSet<()> = JoinSet::new();

    // Counter for in-flight tasks (used for logging/diagnostics).
    let in_flight_count = Arc::new(AtomicUsize::new(0));

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
            // Select between event channel, error channel, timeout, and pause.
            // Pause is checked last (lowest priority) so the loop processes
            // all ready events/errors before honouring a pause request.
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
                // Pause signal -- lowest priority so events are drained first.
                _ = &mut pause_rx => {
                    handle_pause(
                        &mut in_flight,
                        &mut event_rx,
                        &ctx,
                        result_tx,
                        snapshot_tx,
                        &workflow_name,
                        run_id,
                    )
                    .await;
                    return;
                }
            }
        } else {
            // No timeout -- select between events, errors, and pause.
            // Pause is checked last so the loop drains ready events first.
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
                // Pause signal -- lowest priority.
                _ = &mut pause_rx => {
                    handle_pause(
                        &mut in_flight,
                        &mut event_rx,
                        &ctx,
                        result_tx,
                        snapshot_tx,
                        &workflow_name,
                        run_id,
                    )
                    .await;
                    return;
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
            // If this is a DynamicEvent (e.g. reinjected after resume),
            // reconstruct a real StopEvent so callers can downcast.
            let final_event: Box<dyn AnyEvent> =
                if event.as_any().downcast_ref::<StopEvent>().is_some() {
                    event
                } else if let Some(dynamic) = event.as_any().downcast_ref::<DynamicEvent>() {
                    // DynamicEvent.data contains the original StopEvent JSON.
                    match serde_json::from_value::<StopEvent>(dynamic.data.clone()) {
                        Ok(stop) => Box::new(stop),
                        Err(_) => {
                            // Fallback: wrap the raw data as the result.
                            Box::new(StopEvent {
                                result: dynamic.data.clone(),
                            })
                        }
                    }
                } else {
                    // Unknown wrapper -- use to_json() as a best-effort.
                    let json = event.to_json();
                    Box::new(StopEvent {
                        result: json.get("result").cloned().unwrap_or(json),
                    })
                };
            let _ = result_tx.send(Ok(final_event));
            return;
        }

        // Check for InputRequestEvent -- triggers HITL pause or callback.
        if event_type == InputRequestEvent::event_type() {
            let request = if let Some(req) = event.as_any().downcast_ref::<InputRequestEvent>() {
                req.clone()
            } else if let Some(dynamic) = event.as_any().downcast_ref::<DynamicEvent>() {
                if let Ok(req) = serde_json::from_value::<InputRequestEvent>(dynamic.data.clone()) {
                    req
                } else {
                    let _ = result_tx.send(Err(WorkflowError::Context(
                        "failed to deserialize InputRequestEvent from DynamicEvent".into(),
                    )));
                    return;
                }
            } else {
                let _ = result_tx.send(Err(WorkflowError::Context(
                    "InputRequestEvent type mismatch".into(),
                )));
                return;
            };

            // If an input handler callback is registered, call it inline.
            if let Some(ref handler) = input_handler {
                match handler(request).await {
                    Ok(response) => {
                        let envelope =
                            EventEnvelope::new(Box::new(response), Some("__input_handler".into()));
                        let _ = event_tx.send(envelope);
                        continue;
                    }
                    Err(e) => {
                        let _ = result_tx.send(Err(e));
                        return;
                    }
                }
            }

            // No callback -- auto-pause with request attached to snapshot.
            handle_input_pause(
                &mut in_flight,
                &mut event_rx,
                &ctx,
                result_tx,
                snapshot_tx,
                &workflow_name,
                run_id,
                &request,
            )
            .await;
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
        ctx.push_collected(&*event).await;

        // Dispatch to each matching handler, tracking in-flight tasks.
        dispatch_to_handlers(
            &handlers,
            &*event,
            &ctx,
            &event_tx,
            &error_tx,
            &mut in_flight,
            &in_flight_count,
        );

        // Auto-checkpoint after dispatching step handlers (best-effort).
        #[cfg(feature = "persist")]
        if checkpoint_config.after_step
            && let Some(ref store) = checkpoint_config.store
        {
            save_checkpoint(&**store, &ctx, &workflow_name, run_id).await;
        }
    }
}

/// Handle the pause sequence:
///
/// 1. Wait for all in-flight step tasks to finish.
/// 2. Drain remaining events from the channel.
/// 3. Snapshot context state.
/// 4. Send the snapshot back to the handler.
/// 5. Signal the result channel with `Paused`.
async fn handle_pause(
    in_flight: &mut JoinSet<()>,
    event_rx: &mut mpsc::UnboundedReceiver<EventEnvelope>,
    ctx: &Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    snapshot_tx: oneshot::Sender<WorkflowSnapshot>,
    workflow_name: &str,
    run_id: Uuid,
) {
    tracing::info!("pause requested -- waiting for in-flight steps to complete");

    // 1. Wait for all in-flight step tasks to finish.
    while in_flight.join_next().await.is_some() {}

    tracing::debug!("all in-flight steps completed");

    // 2. Drain remaining events from the channel.
    let mut pending_events = Vec::new();
    while let Ok(envelope) = event_rx.try_recv() {
        let serialized = SerializedEvent {
            event_type: envelope.event.event_type_id().to_owned(),
            data: envelope.event.to_json(),
            source_step: envelope.source_step,
        };
        pending_events.push(serialized);
    }

    tracing::debug!(
        pending_count = pending_events.len(),
        "drained pending events"
    );

    // 3. Snapshot context state.
    let context_state = ctx.snapshot_state().await;
    let collected_events = ctx.snapshot_collected().await;
    let metadata = ctx.snapshot_metadata().await;

    let snapshot = WorkflowSnapshot {
        workflow_name: workflow_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        context_state,
        collected_events,
        pending_events,
        metadata,
    };

    // 4. Send the snapshot back to the handler.
    let _ = snapshot_tx.send(snapshot);

    // 5. Signal the result channel with Paused.
    let _ = result_tx.send(Err(WorkflowError::Paused));
}

/// Handle the input-pause sequence (similar to [`handle_pause`]):
///
/// 1. Wait for all in-flight step tasks to finish.
/// 2. Drain remaining events from the channel.
/// 3. Store the input request in context metadata.
/// 4. Snapshot context state.
/// 5. Send the snapshot back to the handler.
/// 6. Signal the result channel with `InputRequired`.
#[allow(clippy::too_many_arguments)]
async fn handle_input_pause(
    in_flight: &mut JoinSet<()>,
    event_rx: &mut mpsc::UnboundedReceiver<EventEnvelope>,
    ctx: &Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    snapshot_tx: oneshot::Sender<WorkflowSnapshot>,
    workflow_name: &str,
    run_id: Uuid,
    request: &InputRequestEvent,
) {
    tracing::info!(
        request_id = %request.request_id,
        "input requested -- pausing for human input"
    );

    // 1. Wait for all in-flight step tasks to finish.
    while in_flight.join_next().await.is_some() {}

    // 2. Drain remaining events from the channel.
    let mut pending_events = Vec::new();
    while let Ok(envelope) = event_rx.try_recv() {
        let serialized = SerializedEvent {
            event_type: envelope.event.event_type_id().to_owned(),
            data: envelope.event.to_json(),
            source_step: envelope.source_step,
        };
        pending_events.push(serialized);
    }

    // 3. Store the input request in metadata.
    ctx.set_metadata(
        "__input_request",
        serde_json::to_value(request).expect("InputRequestEvent serialization should never fail"),
    )
    .await;

    // 4. Snapshot context state.
    let context_state = ctx.snapshot_state().await;
    let collected_events = ctx.snapshot_collected().await;
    let metadata = ctx.snapshot_metadata().await;

    let snapshot = WorkflowSnapshot {
        workflow_name: workflow_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        context_state,
        collected_events,
        pending_events,
        metadata,
    };

    // 5. Send the snapshot back to the handler.
    let _ = snapshot_tx.send(snapshot);

    // 6. Signal InputRequired.
    let _ = result_tx.send(Err(WorkflowError::InputRequired {
        request_id: request.request_id.clone(),
        prompt: request.prompt.clone(),
        metadata: request.metadata.clone(),
    }));
}

/// Spawn step handler tasks for each matching step registration.
///
/// Each spawned task is added to the `in_flight` [`JoinSet`] so the event
/// loop can wait for all of them to complete during a pause.
fn dispatch_to_handlers(
    handlers: &[StepRegistration],
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
    in_flight: &mut JoinSet<()>,
    in_flight_count: &Arc<AtomicUsize>,
) {
    for step in handlers {
        let event_clone = event.clone_boxed();
        let ctx_clone = ctx.clone();
        let handler = step.handler.clone();
        let step_name = step.name.clone();
        let event_tx_clone = event_tx.clone();
        let error_tx_clone = error_tx.clone();
        let counter = Arc::clone(in_flight_count);

        counter.fetch_add(1, Ordering::Relaxed);

        in_flight.spawn(async move {
            match handler(event_clone, ctx_clone).await {
                Ok(StepOutput::Single(output_event)) => {
                    let envelope = EventEnvelope::new(output_event, Some(step_name));
                    let _ = event_tx_clone.send(envelope);
                }
                Ok(StepOutput::Multiple(events)) => {
                    for e in events {
                        let envelope = EventEnvelope::new(e, Some(step_name.clone()));
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
            counter.fetch_sub(1, Ordering::Relaxed);
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_events::{StartEvent, StopEvent};
    use std::sync::Arc;

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
