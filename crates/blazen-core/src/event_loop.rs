//! Core event loop that drives workflow execution.
//!
//! This module contains the runtime loop that receives events, routes them to
//! registered step handlers, and manages pause/resume, checkpointing, and
//! telemetry history.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Duration;

use blazen_events::{
    AnyEvent, DynamicEvent, Event, EventEnvelope, InputRequestEvent, ProgressEvent, ProgressKind,
    StopEvent,
};
use chrono::Utc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::runtime::{Instant, JoinSet};

use tracing::Instrument;

use crate::builder::InputHandlerFn;
use crate::context::Context;
use crate::error::WorkflowError;
use crate::handler::WorkflowControl;
use crate::session_ref::{
    RefLifetime, SERIALIZED_SESSION_REFS_META_KEY, SessionPausePolicy, SessionRefRegistry,
};
use crate::snapshot::{SNAPSHOT_VERSION, WorkflowSnapshot};
use crate::step::{
    JoinStrategy, ParallelSubWorkflowsStep, StepKind, StepOutput, StepRegistration, SubWorkflowStep,
};

// ---------------------------------------------------------------------------
// Checkpoint configuration (persist feature)
// ---------------------------------------------------------------------------

/// Internal configuration for the checkpoint system, passed from the builder
/// to the event loop.
#[cfg(feature = "persist")]
pub(crate) struct CheckpointConfig {
    pub(crate) store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    pub(crate) after_step: bool,
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
    #[cfg(feature = "telemetry")] history_buffer: &[blazen_telemetry::HistoryEvent],
) {
    let context_state = ctx.snapshot_state().await;
    let collected_events = ctx.snapshot_collected().await;
    let metadata = ctx.snapshot_metadata().await;

    #[cfg(feature = "telemetry")]
    let history = history_buffer.to_vec();

    let snapshot = WorkflowSnapshot {
        version: SNAPSHOT_VERSION,
        workflow_name: workflow_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        context_state,
        collected_events,
        pending_events: Vec::new(), // Cannot peek at the channel non-destructively.
        metadata,
        #[cfg(feature = "telemetry")]
        history,
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
pub(crate) async fn event_loop(
    event_rx: mpsc::UnboundedReceiver<EventEnvelope>,
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    registry: HashMap<String, Vec<StepKind>>,
    ctx: Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    timeout: Option<Duration>,
    control_rx: mpsc::UnboundedReceiver<WorkflowControl>,
    workflow_name: String,
    run_id: Uuid,
    input_handler: Option<InputHandlerFn>,
    auto_publish_events: bool,
    #[cfg(feature = "persist")] checkpoint_config: CheckpointConfig,
    #[cfg(feature = "telemetry")] history_tx: Option<
        mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    let stream_ctx = ctx.clone();
    let span = tracing::info_span!(
        "workflow.run",
        workflow_name = %workflow_name,
        run_id = %run_id,
    );
    event_loop_inner(
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
        auto_publish_events,
        #[cfg(feature = "persist")]
        checkpoint_config,
        #[cfg(feature = "telemetry")]
        history_tx,
    )
    .instrument(span)
    .await;
    stream_ctx.signal_stream_end().await;
}

/// Record a telemetry history event: send it through the channel for external
/// consumers AND push a clone into the local buffer so snapshots capture the
/// full history without draining the channel.
#[cfg(feature = "telemetry")]
fn emit_history(
    tx: Option<&mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>>,
    buffer: &mut Vec<blazen_telemetry::HistoryEvent>,
    event: blazen_telemetry::HistoryEvent,
) {
    if let Some(tx) = tx {
        let _ = tx.send(event.clone());
    }
    buffer.push(event);
}

/// Inner event loop implementation. See [`event_loop`] for the public wrapper
/// that guarantees stream-end signaling.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn event_loop_inner(
    mut event_rx: mpsc::UnboundedReceiver<EventEnvelope>,
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    registry: HashMap<String, Vec<StepKind>>,
    ctx: Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    timeout: Option<Duration>,
    mut control_rx: mpsc::UnboundedReceiver<WorkflowControl>,
    workflow_name: String,
    run_id: Uuid,
    input_handler: Option<InputHandlerFn>,
    auto_publish_events: bool,
    #[cfg(feature = "persist")] checkpoint_config: CheckpointConfig,
    #[cfg(feature = "telemetry")] history_tx: Option<
        mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    let start = Instant::now();

    // Local buffer that mirrors every history event sent through `history_tx`.
    // `build_snapshot_in_place` clones this buffer into the snapshot so callers
    // get the full history without draining the channel.
    #[cfg(feature = "telemetry")]
    let mut history_buffer: Vec<blazen_telemetry::HistoryEvent> = Vec::new();

    // Emit WorkflowStarted history event.
    #[cfg(feature = "telemetry")]
    emit_history(
        history_tx.as_ref(),
        &mut history_buffer,
        blazen_telemetry::HistoryEvent {
            timestamp: Utc::now(),
            sequence: 0,
            kind: blazen_telemetry::HistoryEventKind::WorkflowStarted {
                input: serde_json::json!({}),
            },
        },
    );

    // Channel for step errors -- steps run in spawned tasks and report
    // failures back here so the event loop can terminate.
    let (error_tx, mut error_rx) = mpsc::unbounded_channel::<WorkflowError>();

    // Track in-flight step tasks so we can wait for them during pause.
    let mut in_flight: JoinSet<()> = JoinSet::new();

    // Counter for in-flight tasks (used for logging/diagnostics).
    let in_flight_count = Arc::new(AtomicUsize::new(0));

    // Cumulative count of steps that have completed successfully so far.
    // Used as the `current` field on the typed `ProgressEvent { kind:
    // ProgressKind::Workflow }` emitted after each step completion when
    // `auto_publish_events` is enabled. Total is unknown for dynamic
    // event-driven workflows, so the emitted progress always carries
    // `total: None` and `percent: None`.
    let completed_steps = Arc::new(AtomicU32::new(0));

    // When `true`, the event dispatch arm is disabled so the loop only
    // listens for control commands (pause/resume/snapshot/abort/input).
    let mut parked = false;

    // Helper closure for auto-publishing lifecycle events to the broadcast stream.
    let publish_lifecycle = |ctx: &Context,
                             kind: &str,
                             step_name: Option<&str>,
                             event_type_str: Option<&str>,
                             duration_ms: Option<u64>,
                             error: Option<&str>| {
        let ctx = ctx.clone();
        let kind = kind.to_owned();
        let step_name = step_name.map(ToOwned::to_owned);
        let event_type_str = event_type_str.map(ToOwned::to_owned);
        let error = error.map(ToOwned::to_owned);
        async move {
            let mut data = serde_json::Map::new();
            data.insert("kind".into(), serde_json::Value::String(kind));
            if let Some(s) = step_name {
                data.insert("step_name".into(), serde_json::Value::String(s));
            }
            if let Some(e) = event_type_str {
                data.insert("event_type".into(), serde_json::Value::String(e));
            }
            if let Some(d) = duration_ms {
                data.insert("duration_ms".into(), serde_json::Value::Number(d.into()));
            }
            if let Some(e) = error {
                data.insert("error".into(), serde_json::Value::String(e));
            }
            ctx.write_event_to_stream(DynamicEvent {
                event_type: "blazen::lifecycle".to_owned(),
                data: serde_json::Value::Object(data),
            })
            .await;
        }
    };

    loop {
        // Calculate remaining time for timeout.
        let recv_result = if let Some(timeout_dur) = timeout {
            let remaining = timeout_dur.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                #[cfg(feature = "telemetry")]
                emit_history(
                    history_tx.as_ref(),
                    &mut history_buffer,
                    blazen_telemetry::HistoryEvent {
                        timestamp: Utc::now(),
                        sequence: 0,
                        kind: blazen_telemetry::HistoryEventKind::WorkflowTimedOut {
                            elapsed_ms: u64::try_from(start.elapsed().as_millis())
                                .unwrap_or(u64::MAX),
                        },
                    },
                );
                let _ = result_tx.send(Err(WorkflowError::Timeout {
                    elapsed: start.elapsed(),
                }));
                return;
            }
            // Select between event channel, error channel, timeout, and control.
            // Control is checked last (lowest priority) so the loop processes
            // all ready events/errors before honouring a control command.
            tokio::select! {
                biased;

                err = error_rx.recv() => {
                    if let Some(workflow_err) = err {
                        #[cfg(feature = "telemetry")]
                        emit_history(
                            history_tx.as_ref(),
                            &mut history_buffer,
                            blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::WorkflowFailed {
                                    error: workflow_err.to_string(),
                                    duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                                },
                            },
                        );
                        let _ = result_tx.send(Err(workflow_err));
                        return;
                    }
                    continue;
                }
                maybe_envelope = event_rx.recv(), if !parked => {
                    maybe_envelope.ok_or(())
                }
                () = crate::runtime::sleep(remaining) => {
                    #[cfg(feature = "telemetry")]
                    emit_history(
                        history_tx.as_ref(),
                        &mut history_buffer,
                        blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::WorkflowTimedOut {
                                elapsed_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                            },
                        },
                    );
                    let _ = result_tx.send(Err(WorkflowError::Timeout {
                        elapsed: start.elapsed(),
                    }));
                    return;
                }
                // Control channel -- lowest priority so events are drained first.
                Some(control) = control_rx.recv() => {
                    match control {
                        WorkflowControl::Pause => {
                            parked = true;
                            #[cfg(feature = "telemetry")]
                            emit_history(
                                history_tx.as_ref(),
                                &mut history_buffer,
                                blazen_telemetry::HistoryEvent {
                                    timestamp: Utc::now(),
                                    sequence: 0,
                                    kind: blazen_telemetry::HistoryEventKind::WorkflowPaused {
                                        reason: blazen_telemetry::PauseReason::Manual,
                                        pending_count: 0,
                                    },
                                },
                            );
                            continue;
                        }
                        WorkflowControl::Resume => {
                            parked = false;
                            continue;
                        }
                        WorkflowControl::Snapshot { reply } => {
                            let snap = build_snapshot_in_place(
                                &ctx,
                                &workflow_name,
                                run_id,
                                #[cfg(feature = "telemetry")]
                                &history_buffer,
                            ).await;
                            let _ = reply.send(snap);
                            continue;
                        }
                        WorkflowControl::Abort => {
                            let _ = result_tx.send(Err(WorkflowError::Paused));
                            return;
                        }
                        WorkflowControl::InputResponse(response) => {
                            parked = false;
                            let envelope = EventEnvelope::new(
                                Box::new(response),
                                Some("__human_input".into()),
                            );
                            let _ = event_tx.send(envelope);
                            continue;
                        }
                    }
                }
            }
        } else {
            // No timeout -- select between events, errors, and control.
            // Control is checked last so the loop drains ready events first.
            tokio::select! {
                biased;

                err = error_rx.recv() => {
                    if let Some(workflow_err) = err {
                        #[cfg(feature = "telemetry")]
                        emit_history(
                            history_tx.as_ref(),
                            &mut history_buffer,
                            blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::WorkflowFailed {
                                    error: workflow_err.to_string(),
                                    duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                                },
                            },
                        );
                        let _ = result_tx.send(Err(workflow_err));
                        return;
                    }
                    continue;
                }
                maybe_envelope = event_rx.recv(), if !parked => {
                    maybe_envelope.ok_or(())
                }
                // Control channel -- lowest priority.
                Some(control) = control_rx.recv() => {
                    match control {
                        WorkflowControl::Pause => {
                            parked = true;
                            #[cfg(feature = "telemetry")]
                            emit_history(
                                history_tx.as_ref(),
                                &mut history_buffer,
                                blazen_telemetry::HistoryEvent {
                                    timestamp: Utc::now(),
                                    sequence: 0,
                                    kind: blazen_telemetry::HistoryEventKind::WorkflowPaused {
                                        reason: blazen_telemetry::PauseReason::Manual,
                                        pending_count: 0,
                                    },
                                },
                            );
                            continue;
                        }
                        WorkflowControl::Resume => {
                            parked = false;
                            continue;
                        }
                        WorkflowControl::Snapshot { reply } => {
                            let snap = build_snapshot_in_place(
                                &ctx,
                                &workflow_name,
                                run_id,
                                #[cfg(feature = "telemetry")]
                                &history_buffer,
                            ).await;
                            let _ = reply.send(snap);
                            continue;
                        }
                        WorkflowControl::Abort => {
                            let _ = result_tx.send(Err(WorkflowError::Paused));
                            return;
                        }
                        WorkflowControl::InputResponse(response) => {
                            parked = false;
                            let envelope = EventEnvelope::new(
                                Box::new(response),
                                Some("__human_input".into()),
                            );
                            let _ = event_tx.send(envelope);
                            continue;
                        }
                    }
                }
            }
        };

        let Ok(envelope) = recv_result else {
            let _ = result_tx.send(Err(WorkflowError::ChannelClosed));
            return;
        };

        let event = envelope.event;
        let event_type = event.event_type_id();

        // Emit EventReceived history event.
        #[cfg(feature = "telemetry")]
        emit_history(
            history_tx.as_ref(),
            &mut history_buffer,
            blazen_telemetry::HistoryEvent {
                timestamp: Utc::now(),
                sequence: 0,
                kind: blazen_telemetry::HistoryEventKind::EventReceived {
                    event_type: event_type.to_string(),
                    source_step: envelope.source_step.clone(),
                },
            },
        );

        // Auto-publish event_routed lifecycle event.
        if auto_publish_events {
            publish_lifecycle(&ctx, "event_routed", None, Some(event_type), None, None).await;
        }

        {
            let _event_span = tracing::debug_span!(
                "workflow.event",
                event_type = %event_type,
                source_step = ?envelope.source_step,
            )
            .entered();

            tracing::debug!(
                event_type,
                source_step = ?envelope.source_step,
                "event loop received event"
            );
        }

        // Check for StopEvent -- terminates the loop.
        if event_type == StopEvent::event_type() {
            tracing::info!("workflow completed via StopEvent");

            // Emit WorkflowCompleted history event.
            #[cfg(feature = "telemetry")]
            emit_history(
                history_tx.as_ref(),
                &mut history_buffer,
                blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::WorkflowCompleted {
                        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                    },
                },
            );

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

            // Emit InputRequested history event.
            #[cfg(feature = "telemetry")]
            emit_history(
                history_tx.as_ref(),
                &mut history_buffer,
                blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::InputRequested {
                        request_id: request.request_id.clone(),
                        prompt: request.prompt.clone(),
                    },
                },
            );

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

            // Emit WorkflowPaused history event (input required).
            #[cfg(feature = "telemetry")]
            emit_history(
                history_tx.as_ref(),
                &mut history_buffer,
                blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::WorkflowPaused {
                        reason: blazen_telemetry::PauseReason::InputRequired,
                        pending_count: 0,
                    },
                },
            );

            // No callback -- park the loop and let the handler deliver
            // the response via WorkflowControl::InputResponse.
            ctx.set_metadata(
                "__input_request",
                serde_json::to_value(&request)
                    .expect("InputRequestEvent serialization should never fail"),
            )
            .await;
            parked = true;
            continue;
        }

        // Look up step handlers for this event type.
        let Some(handlers) = registry.get(event_type) else {
            tracing::warn!(event_type, "no handler registered for event type");
            #[cfg(feature = "telemetry")]
            emit_history(
                history_tx.as_ref(),
                &mut history_buffer,
                blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::WorkflowFailed {
                        error: format!("no handler registered for event type: {event_type}"),
                        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                    },
                },
            );
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
            &completed_steps,
            run_id,
            auto_publish_events,
            #[cfg(feature = "telemetry")]
            history_tx.as_ref(),
        );

        // Auto-checkpoint after dispatching step handlers (best-effort).
        #[cfg(feature = "persist")]
        if checkpoint_config.after_step
            && let Some(ref store) = checkpoint_config.store
        {
            save_checkpoint(
                &**store,
                &ctx,
                &workflow_name,
                run_id,
                #[cfg(feature = "telemetry")]
                &history_buffer,
            )
            .await;
        }
    }
}

/// Build a snapshot from the current context state without draining
/// channels or waiting for in-flight steps. The event loop continues
/// running after this returns.
///
/// Enforces the [`SessionPausePolicy`] configured on the context:
/// - **`HardError`**: returns an error if any live session refs exist.
/// - **`PickleOrSerialize`**: attempts to serialize entries that opted
///   in via
///   [`crate::SessionRefRegistry::insert_serializable`], stashing the
///   captured bytes under
///   [`crate::session_ref::SERIALIZED_SESSION_REFS_META_KEY`] in
///   snapshot metadata. Non-serializable entries fall through to the
///   same drop-and-warn path as `PickleOrError`.
/// - **`WarnDrop`** / **`PickleOrError`** (without a pickler): logs a
///   warning and stores the dropped keys in snapshot metadata under
///   `"__blazen_dropped_session_refs"`.
///
/// The `history_buffer` (telemetry feature only) is cloned into the
/// snapshot so callers receive the full event history. The buffer itself
/// is not drained — the event loop keeps accumulating.
async fn build_snapshot_in_place(
    ctx: &Context,
    workflow_name: &str,
    run_id: Uuid,
    #[cfg(feature = "telemetry")] history_buffer: &[blazen_telemetry::HistoryEvent],
) -> Result<WorkflowSnapshot, WorkflowError> {
    let context_state = ctx.snapshot_state().await;
    let collected_events = ctx.snapshot_collected().await;
    let mut metadata = ctx.snapshot_metadata().await;

    apply_session_pause_policy(ctx, &mut metadata).await?;

    // ---------------------------------------------------------------
    // Build the snapshot
    // ---------------------------------------------------------------
    #[cfg(feature = "telemetry")]
    let history = history_buffer.to_vec();

    Ok(WorkflowSnapshot {
        version: SNAPSHOT_VERSION,
        workflow_name: workflow_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        context_state,
        collected_events,
        pending_events: Vec::new(), // Cannot peek at mpsc non-destructively
        metadata,
        #[cfg(feature = "telemetry")]
        history,
    })
}

/// Walk the registry and remove every entry whose [`RefLifetime`] is
/// [`RefLifetime::UntilSnapshot`]. Called before any
/// [`SessionPausePolicy`] processing so ephemeral refs are guaranteed
/// not to reach the snapshot walker.
async fn purge_until_snapshot_refs(registry: &SessionRefRegistry) {
    let keys = registry.keys().await;
    for key in keys {
        if registry.lifetime_of(key).await == Some(RefLifetime::UntilSnapshot) {
            registry.remove(key).await;
            tracing::debug!(
                key = %key,
                "purged UntilSnapshot session ref before snapshot walk"
            );
        }
    }
}

/// Apply the configured [`SessionPausePolicy`] to the live session-ref
/// registry, mutating `metadata` to reflect the outcome.
///
/// See [`build_snapshot_in_place`] for the list of policies handled and
/// their semantics. Returns
/// [`WorkflowError::SessionRefsNotSerializable`] when the policy is
/// [`SessionPausePolicy::HardError`] and live refs are present.
///
/// Per-ref [`RefLifetime`] policies are applied here as well: every
/// entry marked [`RefLifetime::UntilSnapshot`] is purged from the
/// registry **before** the configured pause policy runs, regardless
/// of which policy is active. This guarantees ephemeral refs never
/// reach the snapshot walker (so e.g. `HardError` does not trip on
/// them) and never cross a pause boundary.
async fn apply_session_pause_policy(
    ctx: &Context,
    metadata: &mut HashMap<String, serde_json::Value>,
) -> Result<(), WorkflowError> {
    let policy = ctx.session_pause_policy().await;
    let registry = ctx.session_refs_arc().await;

    // Purge `UntilSnapshot` lifetime refs unconditionally — they are
    // ephemeral and must not survive into the snapshot regardless of
    // the configured pause policy.
    purge_until_snapshot_refs(&registry).await;

    if registry.is_empty().await {
        return Ok(());
    }

    match policy {
        SessionPausePolicy::HardError => {
            let keys: Vec<String> = registry
                .keys()
                .await
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            Err(WorkflowError::SessionRefsNotSerializable { keys })
        }
        SessionPausePolicy::WarnDrop => {
            let keys = registry.keys().await;
            if !keys.is_empty() {
                let key_strs: Vec<String> =
                    keys.iter().map(std::string::ToString::to_string).collect();
                tracing::warn!(
                    count = keys.len(),
                    keys = ?key_strs,
                    "dropping live session refs from snapshot (WarnDrop policy)"
                );
                metadata.insert(
                    "__blazen_dropped_session_refs".to_owned(),
                    serde_json::to_value(&key_strs).unwrap_or_default(),
                );
            }
            Ok(())
        }
        SessionPausePolicy::PickleOrError => {
            // Without a binding-provided pickle hook, behave like WarnDrop.
            // Future: add a session_pickler callback to WorkflowBuilder.
            let keys = registry.keys().await;
            if !keys.is_empty() {
                let key_strs: Vec<String> =
                    keys.iter().map(std::string::ToString::to_string).collect();
                tracing::warn!(
                    count = keys.len(),
                    keys = ?key_strs,
                    "dropping live session refs from snapshot \
                     (PickleOrError policy, no pickler registered)"
                );
                metadata.insert(
                    "__blazen_dropped_session_refs".to_owned(),
                    serde_json::to_value(&key_strs).unwrap_or_default(),
                );
            }
            Ok(())
        }
        SessionPausePolicy::PickleOrSerialize => {
            apply_pickle_or_serialize_policy(&registry, metadata).await;
            Ok(())
        }
    }
}

/// Walk the sidecar of serializable entries for the
/// [`SessionPausePolicy::PickleOrSerialize`] policy, capturing each
/// one's binary representation into snapshot metadata and recording
/// any non-serializable keys under the existing dropped-refs field.
async fn apply_pickle_or_serialize_policy(
    registry: &SessionRefRegistry,
    metadata: &mut HashMap<String, serde_json::Value>,
) {
    let all_keys = registry.keys().await;
    let serializable = registry.serializable_entries().await;
    let mut captured: HashMap<String, serde_json::Value> =
        HashMap::with_capacity(serializable.len());

    for (key, entry) in &serializable {
        let type_tag = entry.blazen_type_tag();
        match entry.blazen_serialize() {
            Ok(bytes) => {
                let mut record = serde_json::Map::with_capacity(2);
                record.insert(
                    "type_tag".to_owned(),
                    serde_json::Value::String(type_tag.to_owned()),
                );
                // Use `serde_bytes` via a `BytesWrapper` so the
                // payload round-trips cleanly through both JSON
                // (array of numbers) and MessagePack (raw bin8)
                // without pulling in a base64 dependency.
                record.insert(
                    "data".to_owned(),
                    serde_json::to_value(crate::value::BytesWrapper(bytes))
                        .unwrap_or(serde_json::Value::Null),
                );
                captured.insert(key.to_string(), serde_json::Value::Object(record));
            }
            Err(err) => {
                tracing::warn!(
                    key = %key,
                    type_tag = %type_tag,
                    error = %err,
                    "session ref serialization failed; dropping entry from snapshot"
                );
            }
        }
    }

    if !captured.is_empty() {
        metadata.insert(
            SERIALIZED_SESSION_REFS_META_KEY.to_owned(),
            serde_json::to_value(&captured).unwrap_or_default(),
        );
    }

    // Record any non-serializable keys under the existing dropped-refs
    // metadata field so the resume side can surface a clear error if
    // someone tries to use them.
    let dropped: Vec<String> = all_keys
        .iter()
        .filter(|k| !captured.contains_key(&k.to_string()))
        .map(std::string::ToString::to_string)
        .collect();

    if !dropped.is_empty() {
        tracing::warn!(
            count = dropped.len(),
            keys = ?dropped,
            "dropping live session refs from snapshot \
             (PickleOrSerialize policy, entries did not \
              implement SessionRefSerializable)"
        );
        metadata.insert(
            "__blazen_dropped_session_refs".to_owned(),
            serde_json::to_value(&dropped).unwrap_or_default(),
        );
    }
}

/// Spawn step handler tasks for each matching step kind.
///
/// Each spawned task is added to the `in_flight` [`JoinSet`] so the event
/// loop can wait for all of them to complete during a pause.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn dispatch_to_handlers(
    handlers: &[StepKind],
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
    in_flight: &mut JoinSet<()>,
    in_flight_count: &Arc<AtomicUsize>,
    completed_steps: &Arc<AtomicU32>,
    run_id: Uuid,
    auto_publish_events: bool,
    #[cfg(feature = "telemetry")] history_tx: Option<
        &mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    for kind in handlers {
        match kind {
            StepKind::Regular(step) => dispatch_regular_step(
                step,
                event,
                ctx,
                event_tx,
                error_tx,
                in_flight,
                in_flight_count,
                completed_steps,
                run_id,
                auto_publish_events,
                #[cfg(feature = "telemetry")]
                history_tx,
            ),
            StepKind::SubWorkflow(step) => dispatch_subworkflow_step(
                step.clone(),
                event,
                ctx,
                event_tx,
                error_tx,
                in_flight,
                in_flight_count,
                completed_steps,
                run_id,
                auto_publish_events,
                #[cfg(feature = "telemetry")]
                history_tx,
            ),
            StepKind::ParallelSubWorkflows(step) => dispatch_parallel_subworkflows_step(
                step.clone(),
                event,
                ctx,
                event_tx,
                error_tx,
                in_flight,
                in_flight_count,
                completed_steps,
                run_id,
                auto_publish_events,
                #[cfg(feature = "telemetry")]
                history_tx,
            ),
        }
    }
}

/// Spawn one regular handler-backed step.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn dispatch_regular_step(
    step: &StepRegistration,
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
    in_flight: &mut JoinSet<()>,
    in_flight_count: &Arc<AtomicUsize>,
    completed_steps: &Arc<AtomicU32>,
    run_id: Uuid,
    auto_publish_events: bool,
    #[cfg(feature = "telemetry")] history_tx: Option<
        &mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    // The original loop body operated on a single `step`; keep it as-is by
    // wrapping in a single-iteration "for" so the existing diff stays minimal.
    for step in std::iter::once(step) {
        let event_clone = event.clone_boxed();
        let step_retry = step.retry_config.clone();
        let ctx_clone = ctx.clone().with_step_retry(step_retry);
        let handler = step.handler.clone();
        let step_name = step.name.clone();
        let event_tx_clone = event_tx.clone();
        let error_tx_clone = error_tx.clone();
        let counter = Arc::clone(in_flight_count);
        let progress_counter = Arc::clone(completed_steps);
        let event_type = event.event_type_id().to_owned();
        let semaphore = step.semaphore.clone();
        let step_timeout = step.timeout;

        // Emit StepDispatched history event.
        #[cfg(feature = "telemetry")]
        let htx = history_tx.cloned();
        #[cfg(feature = "telemetry")]
        if let Some(ref tx) = htx {
            let _ = tx.send(blazen_telemetry::HistoryEvent {
                timestamp: Utc::now(),
                sequence: 0,
                kind: blazen_telemetry::HistoryEventKind::StepDispatched {
                    step_name: step_name.clone(),
                    event_type: event_type.clone(),
                },
            });
        }

        // Auto-publish step_started lifecycle event.
        let stream_ctx = if auto_publish_events {
            Some(ctx.clone())
        } else {
            None
        };

        counter.fetch_add(1, Ordering::Relaxed);

        let step_span = tracing::info_span!(
            "workflow.step",
            step_name = %step_name,
            event_type = %event_type,
            otel.status_code = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
        );
        let step_span_clone = step_span.clone();

        in_flight.spawn(
            async move {
                // Acquire a concurrency permit if bounded. The permit is held
                // for the lifetime of this handler invocation.
                let _permit = match semaphore {
                    Some(ref sem) => Some(sem.acquire().await.expect("semaphore closed")),
                    None => None,
                };

                // Auto-publish step_started.
                if let Some(ref sctx) = stream_ctx {
                    let mut data = serde_json::Map::new();
                    data.insert(
                        "kind".into(),
                        serde_json::Value::String("step_started".into()),
                    );
                    data.insert(
                        "step_name".into(),
                        serde_json::Value::String(step_name.clone()),
                    );
                    data.insert(
                        "event_type".into(),
                        serde_json::Value::String(event_type.clone()),
                    );
                    sctx.write_event_to_stream(DynamicEvent {
                        event_type: "blazen::lifecycle".to_owned(),
                        data: serde_json::Value::Object(data),
                    })
                    .await;
                }

                let start = Instant::now();
                let handler_fut = handler(event_clone, ctx_clone);
                let handler_outcome = if let Some(d) = step_timeout {
                    let Ok(inner) = tokio::time::timeout(d, handler_fut).await else {
                        let elapsed_ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
                        // Emit StepFailed history event.
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = htx {
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::StepFailed {
                                    step_name: step_name.clone(),
                                    error: format!(
                                        "step '{step_name}' timed out after {elapsed_ms}ms"
                                    ),
                                    duration_ms: u64::try_from(start.elapsed().as_millis())
                                        .unwrap_or(u64::MAX),
                                },
                            });
                        }
                        let _ = error_tx_clone.send(WorkflowError::StepTimeout {
                            step_name: step_name.clone(),
                            elapsed_ms,
                        });
                        counter.fetch_sub(1, Ordering::Relaxed);
                        return;
                    };
                    inner
                } else {
                    handler_fut.await
                };
                match handler_outcome {
                    Ok(StepOutput::Single(output_event)) => {
                        let duration =
                            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                        step_span_clone.record("duration_ms", duration);
                        step_span_clone.record("otel.status_code", "OK");

                        // Emit StepCompleted history event.
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = htx {
                            let output_type = output_event.event_type_id().to_owned();
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::StepCompleted {
                                    step_name: step_name.clone(),
                                    duration_ms: duration,
                                    output_type,
                                },
                            });
                        }

                        // Auto-publish step_completed.
                        if let Some(ref sctx) = stream_ctx {
                            let mut data = serde_json::Map::new();
                            data.insert(
                                "kind".into(),
                                serde_json::Value::String("step_completed".into()),
                            );
                            data.insert(
                                "step_name".into(),
                                serde_json::Value::String(step_name.clone()),
                            );
                            data.insert(
                                "duration_ms".into(),
                                serde_json::Value::Number(duration.into()),
                            );
                            sctx.write_event_to_stream(DynamicEvent {
                                event_type: "blazen::lifecycle".to_owned(),
                                data: serde_json::Value::Object(data),
                            })
                            .await;

                            emit_workflow_progress(sctx, &progress_counter, &step_name, run_id)
                                .await;
                        }

                        let envelope = EventEnvelope::new(output_event, Some(step_name));
                        let _ = event_tx_clone.send(envelope);
                    }
                    Ok(StepOutput::Multiple(events)) => {
                        let duration =
                            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                        step_span_clone.record("duration_ms", duration);
                        step_span_clone.record("otel.status_code", "OK");

                        // Emit StepCompleted history event.
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = htx {
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::StepCompleted {
                                    step_name: step_name.clone(),
                                    duration_ms: duration,
                                    output_type: "Multiple".to_owned(),
                                },
                            });
                        }

                        // Auto-publish step_completed.
                        if let Some(ref sctx) = stream_ctx {
                            let mut data = serde_json::Map::new();
                            data.insert(
                                "kind".into(),
                                serde_json::Value::String("step_completed".into()),
                            );
                            data.insert(
                                "step_name".into(),
                                serde_json::Value::String(step_name.clone()),
                            );
                            data.insert(
                                "duration_ms".into(),
                                serde_json::Value::Number(duration.into()),
                            );
                            sctx.write_event_to_stream(DynamicEvent {
                                event_type: "blazen::lifecycle".to_owned(),
                                data: serde_json::Value::Object(data),
                            })
                            .await;

                            emit_workflow_progress(sctx, &progress_counter, &step_name, run_id)
                                .await;
                        }

                        for e in events {
                            let envelope = EventEnvelope::new(e, Some(step_name.clone()));
                            let _ = event_tx_clone.send(envelope);
                        }
                    }
                    Ok(StepOutput::None) => {
                        let duration =
                            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                        step_span_clone.record("duration_ms", duration);
                        step_span_clone.record("otel.status_code", "OK");

                        // Emit StepCompleted history event.
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = htx {
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::StepCompleted {
                                    step_name: step_name.clone(),
                                    duration_ms: duration,
                                    output_type: "None".to_owned(),
                                },
                            });
                        }

                        // Auto-publish step_completed.
                        if let Some(ref sctx) = stream_ctx {
                            let mut data = serde_json::Map::new();
                            data.insert(
                                "kind".into(),
                                serde_json::Value::String("step_completed".into()),
                            );
                            data.insert(
                                "step_name".into(),
                                serde_json::Value::String(step_name.clone()),
                            );
                            data.insert(
                                "duration_ms".into(),
                                serde_json::Value::Number(duration.into()),
                            );
                            sctx.write_event_to_stream(DynamicEvent {
                                event_type: "blazen::lifecycle".to_owned(),
                                data: serde_json::Value::Object(data),
                            })
                            .await;

                            emit_workflow_progress(sctx, &progress_counter, &step_name, run_id)
                                .await;
                        }

                        // Side-effect only step -- nothing to route.
                    }
                    Err(err) => {
                        let duration =
                            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                        step_span_clone.record("duration_ms", duration);
                        step_span_clone.record("otel.status_code", "ERROR");

                        let err_str = err.to_string();

                        // Emit StepFailed history event.
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = htx {
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::StepFailed {
                                    step_name: step_name.clone(),
                                    error: err_str.clone(),
                                    duration_ms: duration,
                                },
                            });
                        }

                        // Auto-publish step_failed.
                        if let Some(ref sctx) = stream_ctx {
                            let mut data = serde_json::Map::new();
                            data.insert(
                                "kind".into(),
                                serde_json::Value::String("step_failed".into()),
                            );
                            data.insert(
                                "step_name".into(),
                                serde_json::Value::String(step_name.clone()),
                            );
                            data.insert(
                                "duration_ms".into(),
                                serde_json::Value::Number(duration.into()),
                            );
                            data.insert("error".into(), serde_json::Value::String(err_str));
                            sctx.write_event_to_stream(DynamicEvent {
                                event_type: "blazen::lifecycle".to_owned(),
                                data: serde_json::Value::Object(data),
                            })
                            .await;
                        }

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
            }
            .instrument(step_span),
        );
    }
}

// ---------------------------------------------------------------------------
// Sub-workflow dispatch
// ---------------------------------------------------------------------------

/// Run one sub-workflow step. Honors `step.timeout` (wall-clock for the
/// entire child run) and `step.retry_config` (retry the child run on
/// failure with the resolved backoff). Emits the mapped output event on
/// success or a [`WorkflowError::SubWorkflowFailed`] on terminal failure.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn dispatch_subworkflow_step(
    step: SubWorkflowStep,
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
    in_flight: &mut JoinSet<()>,
    in_flight_count: &Arc<AtomicUsize>,
    completed_steps: &Arc<AtomicU32>,
    run_id: Uuid,
    auto_publish_events: bool,
    #[cfg(feature = "telemetry")] history_tx: Option<
        &mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    let event_clone = event.clone_boxed();
    let event_type = event.event_type_id().to_owned();
    let ctx_clone = ctx.clone();
    let event_tx_clone = event_tx.clone();
    let error_tx_clone = error_tx.clone();
    let counter = Arc::clone(in_flight_count);
    let progress_counter = Arc::clone(completed_steps);
    let stream_ctx = if auto_publish_events {
        Some(ctx.clone())
    } else {
        None
    };

    #[cfg(feature = "telemetry")]
    let htx = history_tx.cloned();

    #[cfg(feature = "telemetry")]
    if let Some(ref tx) = htx {
        let _ = tx.send(blazen_telemetry::HistoryEvent {
            timestamp: Utc::now(),
            sequence: 0,
            kind: blazen_telemetry::HistoryEventKind::StepDispatched {
                step_name: step.name.clone(),
                event_type: event_type.clone(),
            },
        });
    }

    counter.fetch_add(1, Ordering::Relaxed);

    let step_span = tracing::info_span!(
        "workflow.subworkflow",
        step_name = %step.name,
        event_type = %event_type,
        otel.status_code = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
    );
    let step_span_clone = step_span.clone();

    in_flight.spawn(
        async move {
            // step_started lifecycle event.
            if let Some(ref sctx) = stream_ctx {
                publish_lifecycle_event(
                    sctx,
                    "step_started",
                    Some(&step.name),
                    Some(&event_type),
                    None,
                    None,
                )
                .await;
            }

            let start = Instant::now();
            let outcome = run_subworkflow_with_retry(&step, &*event_clone, &ctx_clone).await;
            let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
            step_span_clone.record("duration_ms", duration);

            match outcome {
                Ok(output) => {
                    step_span_clone.record("otel.status_code", "OK");

                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = htx {
                        let output_type = output.event_type_id().to_owned();
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::StepCompleted {
                                step_name: step.name.clone(),
                                duration_ms: duration,
                                output_type,
                            },
                        });
                    }

                    if let Some(ref sctx) = stream_ctx {
                        publish_lifecycle_event(
                            sctx,
                            "step_completed",
                            Some(&step.name),
                            None,
                            Some(duration),
                            None,
                        )
                        .await;

                        emit_workflow_progress(sctx, &progress_counter, &step.name, run_id).await;
                    }

                    let envelope = EventEnvelope::new(output, Some(step.name.clone()));
                    let _ = event_tx_clone.send(envelope);
                }
                Err(err) => {
                    step_span_clone.record("otel.status_code", "ERROR");
                    let err_str = err.to_string();

                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = htx {
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::StepFailed {
                                step_name: step.name.clone(),
                                error: err_str.clone(),
                                duration_ms: duration,
                            },
                        });
                    }

                    if let Some(ref sctx) = stream_ctx {
                        publish_lifecycle_event(
                            sctx,
                            "step_failed",
                            Some(&step.name),
                            None,
                            Some(duration),
                            Some(&err_str),
                        )
                        .await;
                    }

                    tracing::error!(
                        step = %step.name,
                        error = %err,
                        "sub-workflow step failed"
                    );
                    let _ = error_tx_clone.send(WorkflowError::SubWorkflowFailed {
                        step_name: step.name.clone(),
                        message: err_str,
                    });
                }
            }
            counter.fetch_sub(1, Ordering::Relaxed);
        }
        .instrument(step_span),
    );
}

/// Fan out into multiple parallel sub-workflows and join via the
/// configured [`JoinStrategy`]. On success, emits one event per branch
/// produced by each branch's [`SubWorkflowStep::output_mapper`]. On
/// failure, emits a [`WorkflowError::SubWorkflowFailed`] naming the
/// outer step (the parent fan-out node).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn dispatch_parallel_subworkflows_step(
    step: ParallelSubWorkflowsStep,
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
    in_flight: &mut JoinSet<()>,
    in_flight_count: &Arc<AtomicUsize>,
    completed_steps: &Arc<AtomicU32>,
    run_id: Uuid,
    auto_publish_events: bool,
    #[cfg(feature = "telemetry")] history_tx: Option<
        &mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    let event_clone = event.clone_boxed();
    let event_type = event.event_type_id().to_owned();
    let ctx_clone = ctx.clone();
    let event_tx_clone = event_tx.clone();
    let error_tx_clone = error_tx.clone();
    let counter = Arc::clone(in_flight_count);
    let progress_counter = Arc::clone(completed_steps);
    let stream_ctx = if auto_publish_events {
        Some(ctx.clone())
    } else {
        None
    };

    #[cfg(feature = "telemetry")]
    let htx = history_tx.cloned();

    #[cfg(feature = "telemetry")]
    if let Some(ref tx) = htx {
        let _ = tx.send(blazen_telemetry::HistoryEvent {
            timestamp: Utc::now(),
            sequence: 0,
            kind: blazen_telemetry::HistoryEventKind::StepDispatched {
                step_name: step.name.clone(),
                event_type: event_type.clone(),
            },
        });
    }

    counter.fetch_add(1, Ordering::Relaxed);

    let step_span = tracing::info_span!(
        "workflow.parallel_subworkflows",
        step_name = %step.name,
        event_type = %event_type,
        join_strategy = ?step.join_strategy,
        branch_count = step.branches.len(),
        otel.status_code = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
    );
    let step_span_clone = step_span.clone();

    in_flight.spawn(
        async move {
            if let Some(ref sctx) = stream_ctx {
                publish_lifecycle_event(
                    sctx,
                    "step_started",
                    Some(&step.name),
                    Some(&event_type),
                    None,
                    None,
                )
                .await;
            }

            let start = Instant::now();
            let outcome = run_parallel_subworkflows(&step, &*event_clone, &ctx_clone).await;
            let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
            step_span_clone.record("duration_ms", duration);

            match outcome {
                Ok(outputs) => {
                    step_span_clone.record("otel.status_code", "OK");

                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = htx {
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::StepCompleted {
                                step_name: step.name.clone(),
                                duration_ms: duration,
                                output_type: "ParallelSubWorkflows".to_owned(),
                            },
                        });
                    }

                    if let Some(ref sctx) = stream_ctx {
                        publish_lifecycle_event(
                            sctx,
                            "step_completed",
                            Some(&step.name),
                            None,
                            Some(duration),
                            None,
                        )
                        .await;

                        emit_workflow_progress(sctx, &progress_counter, &step.name, run_id).await;
                    }

                    for output in outputs {
                        let envelope = EventEnvelope::new(output, Some(step.name.clone()));
                        let _ = event_tx_clone.send(envelope);
                    }
                }
                Err(err) => {
                    step_span_clone.record("otel.status_code", "ERROR");
                    let err_str = err.to_string();

                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = htx {
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::StepFailed {
                                step_name: step.name.clone(),
                                error: err_str.clone(),
                                duration_ms: duration,
                            },
                        });
                    }

                    if let Some(ref sctx) = stream_ctx {
                        publish_lifecycle_event(
                            sctx,
                            "step_failed",
                            Some(&step.name),
                            None,
                            Some(duration),
                            Some(&err_str),
                        )
                        .await;
                    }

                    tracing::error!(
                        step = %step.name,
                        error = %err,
                        "parallel sub-workflows step failed"
                    );
                    let _ = error_tx_clone.send(WorkflowError::SubWorkflowFailed {
                        step_name: step.name.clone(),
                        message: err_str,
                    });
                }
            }
            counter.fetch_sub(1, Ordering::Relaxed);
        }
        .instrument(step_span),
    );
}

/// Run a single sub-workflow once with the configured timeout applied.
///
/// Returns the mapped output event on success, or a `WorkflowError`
/// describing the underlying failure (timeout, child error, channel
/// closed, etc.).
async fn run_subworkflow_once(
    step: &SubWorkflowStep,
    event: &dyn AnyEvent,
) -> Result<Box<dyn AnyEvent>, WorkflowError> {
    // 1. Map parent event → child input JSON.
    let input_json = (step.input_mapper)(event);

    // 2. Run the child workflow.
    let handler = step.workflow.run(input_json).await?;

    // 3. Await completion, optionally with a wall-clock timeout that
    //    spans the whole child run.
    let wf_result = if let Some(d) = step.timeout {
        if let Ok(r) = tokio::time::timeout(d, handler.result()).await {
            r?
        } else {
            let elapsed_ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
            return Err(WorkflowError::StepTimeout {
                step_name: step.name.clone(),
                elapsed_ms,
            });
        }
    } else {
        handler.result().await?
    };

    // 4. Extract the terminal `StopEvent.result` JSON and pass it through
    //    the user-supplied output mapper to produce the parent step's
    //    emitted event.
    let result_json = if let Some(stop) = wf_result
        .event
        .as_any()
        .downcast_ref::<blazen_events::StopEvent>()
    {
        stop.result.clone()
    } else {
        // Non-StopEvent terminal — fall back to the type-erased
        // `to_json()` so the caller still sees a usable payload.
        wf_result.event.to_json()
    };

    Ok((step.output_mapper)(result_json))
}

/// Run a sub-workflow step with the resolved retry configuration applied.
///
/// Resolution precedence: per-step `retry_config` → workflow → pipeline →
/// provider. When a retry config is in effect, failures are retried up to
/// `max_retries` times with the configured exponential backoff.
async fn run_subworkflow_with_retry(
    step: &SubWorkflowStep,
    event: &dyn AnyEvent,
    ctx: &Context,
) -> Result<Box<dyn AnyEvent>, WorkflowError> {
    let stack = ctx.retry_stack();
    let resolved = blazen_llm::retry::resolve_retry(
        None,
        step.retry_config.as_ref(),
        stack.workflow.as_ref(),
        stack.pipeline.as_ref(),
        stack.provider.as_ref(),
    );

    let max_retries = resolved.max_retries;
    let mut last_err: Option<WorkflowError> = None;

    for attempt in 0..=max_retries {
        match run_subworkflow_once(step, event).await {
            Ok(out) => return Ok(out),
            Err(err) => {
                tracing::warn!(
                    step = %step.name,
                    attempt = attempt + 1,
                    max_attempts = max_retries + 1,
                    error = %err,
                    "sub-workflow attempt failed"
                );
                last_err = Some(err);
                if attempt < max_retries {
                    let delay = compute_retry_delay(&resolved, attempt);
                    crate::runtime::sleep(delay).await;
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        WorkflowError::Context("sub-workflow retry loop exited without result".to_owned())
    }))
}

/// Run every branch of a [`ParallelSubWorkflowsStep`] concurrently and
/// join via the configured [`JoinStrategy`].
///
/// `WaitAll` waits for every branch and collects their mapped outputs in
/// branch order. `FirstCompletes` resolves with a single output as soon
/// as any branch completes successfully and aborts the rest.
async fn run_parallel_subworkflows(
    step: &ParallelSubWorkflowsStep,
    event: &dyn AnyEvent,
    ctx: &Context,
) -> Result<Vec<Box<dyn AnyEvent>>, WorkflowError> {
    /// Per-branch outcome carried back from each spawned task: the
    /// branch's index in the original `branches` vec (so `WaitAll` can
    /// re-sort outputs into source order), its name (for error
    /// messages), and the [`run_subworkflow_with_retry`] result.
    type BranchOutcome = (usize, String, Result<Box<dyn AnyEvent>, WorkflowError>);

    let mut set: JoinSet<BranchOutcome> = JoinSet::new();

    for (idx, branch) in step.branches.iter().enumerate() {
        let branch_clone = branch.clone();
        let event_clone = event.clone_boxed();
        let ctx_clone = ctx.clone();
        let branch_name = branch.name.clone();
        set.spawn(async move {
            let res = run_subworkflow_with_retry(&branch_clone, &*event_clone, &ctx_clone).await;
            (idx, branch_name, res)
        });
    }

    match step.join_strategy {
        JoinStrategy::WaitAll => {
            let mut collected: Vec<(usize, Box<dyn AnyEvent>)> =
                Vec::with_capacity(step.branches.len());
            while let Some(joined) = set.join_next().await {
                let (idx, branch_name, res) = joined.map_err(|e| {
                    WorkflowError::Context(format!("parallel sub-workflow branch join failed: {e}"))
                })?;
                match res {
                    Ok(out) => collected.push((idx, out)),
                    Err(err) => {
                        // Abort the remaining branches before bubbling up.
                        set.abort_all();
                        return Err(WorkflowError::SubWorkflowFailed {
                            step_name: format!("{}::{}", step.name, branch_name),
                            message: err.to_string(),
                        });
                    }
                }
            }
            collected.sort_by_key(|(idx, _)| *idx);
            Ok(collected.into_iter().map(|(_, e)| e).collect())
        }
        JoinStrategy::FirstCompletes => {
            // Resolve as soon as the first successful branch completes.
            // Track the most recent error so we can surface it if every
            // branch fails before any succeeds.
            let mut last_err: Option<(String, WorkflowError)> = None;
            while let Some(joined) = set.join_next().await {
                let (_, branch_name, res) = joined.map_err(|e| {
                    WorkflowError::Context(format!("parallel sub-workflow branch join failed: {e}"))
                })?;
                match res {
                    Ok(out) => {
                        // Abort the remaining branches before returning.
                        set.abort_all();
                        return Ok(vec![out]);
                    }
                    Err(err) => {
                        last_err = Some((branch_name, err));
                    }
                }
            }
            let (branch_name, err) = last_err.unwrap_or_else(|| {
                (
                    String::new(),
                    WorkflowError::Context("parallel sub-workflow had no branches".to_owned()),
                )
            });
            Err(WorkflowError::SubWorkflowFailed {
                step_name: format!("{}::{}", step.name, branch_name),
                message: err.to_string(),
            })
        }
    }
}

/// Mirror of the private `RetryCompletionModel::compute_delay` so
/// sub-workflow retries honor the same exponential backoff /
/// max-delay / jitter semantics without depending on private
/// internals of `blazen-llm`. The jitter source is a cheap LCG
/// seeded from the system clock so we don't pull in `rand` (which
/// would gate the wasm build behind a feature).
fn compute_retry_delay(cfg: &blazen_llm::retry::RetryConfig, attempt: u32) -> Duration {
    let initial = Duration::from_millis(cfg.initial_delay_ms);
    let max = Duration::from_millis(cfg.max_delay_ms);
    // Saturating shift to avoid overflow on very large attempts.
    let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
    let millis = u64::try_from(
        initial
            .as_millis()
            .saturating_mul(u128::from(factor))
            .min(u128::from(u64::MAX)),
    )
    .unwrap_or(u64::MAX);
    let mut delay = Duration::from_millis(millis).min(max);

    if cfg.jitter {
        let nanos = u128::from(u64::try_from(delay.as_nanos()).unwrap_or(u64::MAX));
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // Random factor in [0, 50); subtract 25 to land in [-25, 24].
        let mixed = seed.wrapping_mul(2_862_933_555_777_941_757) % 50;
        // `mixed` is at most 49, well within `i128::MAX`, so the
        // try_from cannot fail in practice.
        let pct = i128::try_from(mixed).unwrap_or(0) - 25;
        let nanos_i = i128::try_from(nanos).unwrap_or(i128::MAX);
        let jitter_nanos = nanos_i.saturating_mul(pct) / 100;
        let new_nanos = nanos_i.saturating_add(jitter_nanos).max(0);
        let clamped = u64::try_from(new_nanos.min(i128::from(u64::MAX))).unwrap_or(u64::MAX);
        delay = Duration::from_nanos(clamped);
    }

    delay
}

/// Helper: emit a typed `ProgressEvent { kind: ProgressKind::Workflow }`
/// to the broadcast stream. Called after each step completes when
/// `auto_publish_events` is enabled. Total is unknown for dynamic
/// event-driven workflows so the emitted progress always carries
/// `total: None` and `percent: None`.
async fn emit_workflow_progress(
    ctx: &Context,
    counter: &Arc<AtomicU32>,
    step_name: &str,
    run_id: Uuid,
) {
    // `fetch_add` returns the *previous* value; add 1 for the post-increment.
    let prev = counter.fetch_add(1, Ordering::Relaxed);
    let current = prev.saturating_add(1);
    let progress = ProgressEvent {
        kind: ProgressKind::Workflow,
        current,
        total: None,
        percent: None,
        label: step_name.to_owned(),
        run_id,
    };
    ctx.write_event_to_stream(progress).await;
}

/// Helper: emit a `blazen::lifecycle` `DynamicEvent` to the broadcast
/// stream. Matches the closure inline-defined in the regular dispatch
/// path so sub-workflow steps publish identical lifecycle envelopes.
async fn publish_lifecycle_event(
    ctx: &Context,
    kind: &str,
    step_name: Option<&str>,
    event_type_str: Option<&str>,
    duration_ms: Option<u64>,
    error: Option<&str>,
) {
    let mut data = serde_json::Map::new();
    data.insert("kind".into(), serde_json::Value::String(kind.to_owned()));
    if let Some(s) = step_name {
        data.insert("step_name".into(), serde_json::Value::String(s.to_owned()));
    }
    if let Some(e) = event_type_str {
        data.insert("event_type".into(), serde_json::Value::String(e.to_owned()));
    }
    if let Some(d) = duration_ms {
        data.insert("duration_ms".into(), serde_json::Value::Number(d.into()));
    }
    if let Some(e) = error {
        data.insert("error".into(), serde_json::Value::String(e.to_owned()));
    }
    ctx.write_event_to_stream(DynamicEvent {
        event_type: "blazen::lifecycle".to_owned(),
        data: serde_json::Value::Object(data),
    })
    .await;
}
