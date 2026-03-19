//! Core event loop that drives workflow execution.
//!
//! This module contains the runtime loop that receives events, routes them to
//! registered step handlers, and manages pause/resume, checkpointing, and
//! telemetry history.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use blazen_events::{AnyEvent, DynamicEvent, Event, EventEnvelope, InputRequestEvent, StopEvent};
use chrono::Utc;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use uuid::Uuid;

use tracing::Instrument;

use crate::builder::InputHandlerFn;
use crate::context::Context;
use crate::error::WorkflowError;
use crate::snapshot::{SerializedEvent, WorkflowSnapshot};
use crate::step::{StepOutput, StepRegistration};

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
        #[cfg(feature = "telemetry")]
        history: Vec::new(),
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
    registry: HashMap<String, Vec<StepRegistration>>,
    ctx: Context,
    result_tx: oneshot::Sender<Result<Box<dyn AnyEvent>, WorkflowError>>,
    timeout: Option<Duration>,
    pause_rx: oneshot::Receiver<()>,
    snapshot_tx: oneshot::Sender<WorkflowSnapshot>,
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
        pause_rx,
        snapshot_tx,
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
    auto_publish_events: bool,
    #[cfg(feature = "persist")] checkpoint_config: CheckpointConfig,
    #[cfg(feature = "telemetry")] history_tx: Option<
        mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    let start = Instant::now();

    // Emit WorkflowStarted history event.
    #[cfg(feature = "telemetry")]
    if let Some(ref tx) = history_tx {
        let _ = tx.send(blazen_telemetry::HistoryEvent {
            timestamp: Utc::now(),
            sequence: 0,
            kind: blazen_telemetry::HistoryEventKind::WorkflowStarted {
                input: serde_json::json!({}),
            },
        });
    }

    // Channel for step errors -- steps run in spawned tasks and report
    // failures back here so the event loop can terminate.
    let (error_tx, mut error_rx) = mpsc::unbounded_channel::<WorkflowError>();

    // Track in-flight step tasks so we can wait for them during pause.
    let mut in_flight: JoinSet<()> = JoinSet::new();

    // Counter for in-flight tasks (used for logging/diagnostics).
    let in_flight_count = Arc::new(AtomicUsize::new(0));

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
                if let Some(ref tx) = history_tx {
                    let _ = tx.send(blazen_telemetry::HistoryEvent {
                        timestamp: Utc::now(),
                        sequence: 0,
                        kind: blazen_telemetry::HistoryEventKind::WorkflowTimedOut {
                            elapsed_ms: u64::try_from(start.elapsed().as_millis())
                                .unwrap_or(u64::MAX),
                        },
                    });
                }
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
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = history_tx {
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::WorkflowFailed {
                                    error: workflow_err.to_string(),
                                    duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                                },
                            });
                        }
                        let _ = result_tx.send(Err(workflow_err));
                        return;
                    }
                    continue;
                }
                maybe_envelope = event_rx.recv() => {
                    maybe_envelope.ok_or(())
                }
                () = tokio::time::sleep(remaining) => {
                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = history_tx {
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::WorkflowTimedOut {
                                elapsed_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                            },
                        });
                    }
                    let _ = result_tx.send(Err(WorkflowError::Timeout {
                        elapsed: start.elapsed(),
                    }));
                    return;
                }
                // Pause signal -- lowest priority so events are drained first.
                _ = &mut pause_rx => {
                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = history_tx {
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::WorkflowPaused {
                                reason: blazen_telemetry::PauseReason::Manual,
                                pending_count: 0,
                            },
                        });
                    }
                    handle_pause(
                        &mut in_flight,
                        &mut event_rx,
                        &ctx,
                        result_tx,
                        snapshot_tx,
                        &workflow_name,
                        run_id,
                    )
                    .instrument(tracing::info_span!("workflow.pause", pause_type = "manual"))
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
                        #[cfg(feature = "telemetry")]
                        if let Some(ref tx) = history_tx {
                            let _ = tx.send(blazen_telemetry::HistoryEvent {
                                timestamp: Utc::now(),
                                sequence: 0,
                                kind: blazen_telemetry::HistoryEventKind::WorkflowFailed {
                                    error: workflow_err.to_string(),
                                    duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                                },
                            });
                        }
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
                    #[cfg(feature = "telemetry")]
                    if let Some(ref tx) = history_tx {
                        let _ = tx.send(blazen_telemetry::HistoryEvent {
                            timestamp: Utc::now(),
                            sequence: 0,
                            kind: blazen_telemetry::HistoryEventKind::WorkflowPaused {
                                reason: blazen_telemetry::PauseReason::Manual,
                                pending_count: 0,
                            },
                        });
                    }
                    handle_pause(
                        &mut in_flight,
                        &mut event_rx,
                        &ctx,
                        result_tx,
                        snapshot_tx,
                        &workflow_name,
                        run_id,
                    )
                    .instrument(tracing::info_span!("workflow.pause", pause_type = "manual"))
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

        // Emit EventReceived history event.
        #[cfg(feature = "telemetry")]
        if let Some(ref tx) = history_tx {
            let _ = tx.send(blazen_telemetry::HistoryEvent {
                timestamp: Utc::now(),
                sequence: 0,
                kind: blazen_telemetry::HistoryEventKind::EventReceived {
                    event_type: event_type.to_string(),
                    source_step: envelope.source_step.clone(),
                },
            });
        }

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
            if let Some(ref tx) = history_tx {
                let _ = tx.send(blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::WorkflowCompleted {
                        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                    },
                });
            }

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
            if let Some(ref tx) = history_tx {
                let _ = tx.send(blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::InputRequested {
                        request_id: request.request_id.clone(),
                        prompt: request.prompt.clone(),
                    },
                });
            }

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
            if let Some(ref tx) = history_tx {
                let _ = tx.send(blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::WorkflowPaused {
                        reason: blazen_telemetry::PauseReason::InputRequired,
                        pending_count: 0,
                    },
                });
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
            .instrument(tracing::info_span!("workflow.pause", pause_type = "input"))
            .await;
            return;
        }

        // Look up step handlers for this event type.
        let Some(handlers) = registry.get(event_type) else {
            tracing::warn!(event_type, "no handler registered for event type");
            #[cfg(feature = "telemetry")]
            if let Some(ref tx) = history_tx {
                let _ = tx.send(blazen_telemetry::HistoryEvent {
                    timestamp: Utc::now(),
                    sequence: 0,
                    kind: blazen_telemetry::HistoryEventKind::WorkflowFailed {
                        error: format!("no handler registered for event type: {event_type}"),
                        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                    },
                });
            }
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
            auto_publish_events,
            #[cfg(feature = "telemetry")]
            history_tx.as_ref(),
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
        #[cfg(feature = "telemetry")]
        history: Vec::new(),
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
        #[cfg(feature = "telemetry")]
        history: Vec::new(),
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
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn dispatch_to_handlers(
    handlers: &[StepRegistration],
    event: &dyn AnyEvent,
    ctx: &Context,
    event_tx: &mpsc::UnboundedSender<EventEnvelope>,
    error_tx: &mpsc::UnboundedSender<WorkflowError>,
    in_flight: &mut JoinSet<()>,
    in_flight_count: &Arc<AtomicUsize>,
    auto_publish_events: bool,
    #[cfg(feature = "telemetry")] history_tx: Option<
        &mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>,
    >,
) {
    for step in handlers {
        let event_clone = event.clone_boxed();
        let ctx_clone = ctx.clone();
        let handler = step.handler.clone();
        let step_name = step.name.clone();
        let event_tx_clone = event_tx.clone();
        let error_tx_clone = error_tx.clone();
        let counter = Arc::clone(in_flight_count);
        let event_type = event.event_type_id().to_owned();

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
                match handler(event_clone, ctx_clone).await {
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
