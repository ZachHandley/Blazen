//! [`WorkflowHandler`] -- the handle returned after starting a workflow.
//!
//! Provides three consumption modes:
//!
//! 1. **Await the final result** -- via [`WorkflowHandler::result`].
//! 2. **Stream intermediate events** -- via [`WorkflowHandler::stream_events`]
//!    which subscribes to the broadcast channel that steps can publish to.
//! 3. **Control the workflow** -- via [`WorkflowHandler::pause`],
//!    [`WorkflowHandler::resume_in_place`], [`WorkflowHandler::snapshot`],
//!    [`WorkflowHandler::respond_to_input`], and [`WorkflowHandler::abort`].
//!
//! Modes 1 and 2 are composable: you can subscribe a stream first, then await
//! the final result. Mode 3 can be used alongside modes 1 and 2.

use std::sync::Arc;

use crate::runtime::JoinHandle;
use blazen_events::{AnyEvent, InputResponseEvent};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::error::WorkflowError;
use crate::session_ref::SessionRefRegistry;
use crate::snapshot::WorkflowSnapshot;

/// The result of a completed workflow run.
///
/// Owns both the terminal event AND the session-ref registry that backs
/// any `__blazen_session_ref__` markers carried by the event payload, so
/// markers remain resolvable for as long as the caller holds the result.
#[derive(Debug)]
pub struct WorkflowResult {
    pub event: Box<dyn AnyEvent>,
    pub session_refs: Arc<SessionRefRegistry>,
}

/// Commands sent from the handler to the event loop via the control channel.
pub(crate) enum WorkflowControl {
    /// Park the event loop. Events stop being dispatched to steps but
    /// the loop stays alive and responsive to further control commands.
    Pause,
    /// Resume a parked event loop.
    Resume,
    /// Capture a [`WorkflowSnapshot`] without stopping the loop.
    /// The snapshot is sent back via the enclosed oneshot.
    Snapshot {
        reply: oneshot::Sender<Result<WorkflowSnapshot, WorkflowError>>,
    },
    /// Tear down the event loop. The loop exits and the spawned task completes.
    Abort,
    /// Deliver a human-in-the-loop response to a workflow that auto-parked
    /// on an [`InputRequestEvent`]. The loop unparks and injects the response
    /// as a routable event.
    InputResponse(InputResponseEvent),
}

/// Handle to a running workflow.
///
/// Created by [`Workflow::run`](crate::Workflow::run) or
/// [`Workflow::run_with_event`](crate::Workflow::run_with_event).
pub struct WorkflowHandler {
    /// Receives the final result (or error) when the workflow completes.
    result_rx: Option<oneshot::Receiver<Result<Box<dyn AnyEvent>, WorkflowError>>>,
    /// Sender side of the broadcast channel -- kept alive so we can create
    /// new subscriber receivers via `subscribe()`.
    stream_tx: broadcast::Sender<Box<dyn AnyEvent>>,
    /// Control channel to the event loop (pause/resume/snapshot/abort/input).
    control_tx: mpsc::UnboundedSender<WorkflowControl>,
    /// Handle to the spawned event loop task.
    event_loop_handle: Option<JoinHandle<()>>,
    /// Live session-ref registry for this run.
    session_refs: Arc<SessionRefRegistry>,
    /// Receives history events from the event loop (requires `telemetry` feature).
    #[cfg(feature = "telemetry")]
    history_rx: Option<mpsc::UnboundedReceiver<blazen_telemetry::HistoryEvent>>,
}

impl WorkflowHandler {
    /// Create a new handler (crate-internal).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        result_rx: oneshot::Receiver<Result<Box<dyn AnyEvent>, WorkflowError>>,
        stream_tx: broadcast::Sender<Box<dyn AnyEvent>>,
        control_tx: mpsc::UnboundedSender<WorkflowControl>,
        event_loop_handle: JoinHandle<()>,
        session_refs: Arc<SessionRefRegistry>,
        #[cfg(feature = "telemetry")] history_rx: Option<
            mpsc::UnboundedReceiver<blazen_telemetry::HistoryEvent>,
        >,
    ) -> Self {
        Self {
            result_rx: Some(result_rx),
            stream_tx,
            control_tx,
            event_loop_handle: Some(event_loop_handle),
            session_refs,
            #[cfg(feature = "telemetry")]
            history_rx,
        }
    }

    /// Get a clone of the session-ref registry handle.
    ///
    /// Bindings call this after [`result`](Self::result) to resolve any
    /// `__blazen_session_ref__` markers carried by the final event,
    /// ensuring identity-preserving access to live Python / JS objects
    /// passed via event payloads.
    #[must_use]
    pub fn session_refs(&self) -> Arc<SessionRefRegistry> {
        Arc::clone(&self.session_refs)
    }

    /// Await the final workflow result.
    ///
    /// Consumes the handler. Returns the terminal event (typically a
    /// [`StopEvent`](blazen_events::StopEvent)) or a [`WorkflowError`].
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop task
    /// was dropped before sending a result.
    ///
    /// # Panics
    ///
    /// Panics if `result()` was already called on this handler (the result
    /// receiver can only be consumed once).
    pub async fn result(mut self) -> Result<WorkflowResult, WorkflowError> {
        let rx = self
            .result_rx
            .take()
            .expect("result() called after result was already consumed");
        let event = rx.await.unwrap_or(Err(WorkflowError::ChannelClosed))?;

        // Wait for the event loop task to fully exit so there are no orphaned
        // Tokio tasks keeping runtimes alive (critical for napi-rs / Node.js).
        if let Some(handle) = self.event_loop_handle.take() {
            let _ = handle.await;
        }

        let session_refs = Arc::clone(&self.session_refs);
        Ok(WorkflowResult {
            event,
            session_refs,
        })
    }

    /// Subscribe to intermediate events published by steps via
    /// [`Context::write_event_to_stream`](crate::Context::write_event_to_stream).
    ///
    /// Each call returns a fresh stream starting from the current point in
    /// time (events published before the subscription are not replayed).
    ///
    /// This method borrows `&self` so you can subscribe one or more streams
    /// and still later call [`result`](Self::result) (or `.await` the handler).
    pub fn stream_events(
        &self,
    ) -> impl tokio_stream::Stream<Item = Box<dyn AnyEvent>> + Send + Unpin + use<> {
        let rx = self.stream_tx.subscribe();
        BroadcastStream::new(rx).filter_map(std::result::Result::ok)
    }

    /// Park the event loop in place.
    ///
    /// The loop stops dispatching events to steps but stays alive and
    /// responsive to [`resume_in_place`](Self::resume_in_place),
    /// [`snapshot`](Self::snapshot), [`respond_to_input`](Self::respond_to_input),
    /// and [`abort`](Self::abort) calls.
    ///
    /// # Errors
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop has already exited.
    pub fn pause(&self) -> Result<(), WorkflowError> {
        self.control_tx
            .send(WorkflowControl::Pause)
            .map_err(|_| WorkflowError::ChannelClosed)
    }

    /// Resume a parked event loop.
    ///
    /// # Errors
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop has already exited.
    pub fn resume_in_place(&self) -> Result<(), WorkflowError> {
        self.control_tx
            .send(WorkflowControl::Resume)
            .map_err(|_| WorkflowError::ChannelClosed)
    }

    /// Capture a [`WorkflowSnapshot`] without stopping the loop.
    ///
    /// For a quiescent snapshot (no in-flight steps), call [`pause`](Self::pause)
    /// first, then `snapshot()`, then optionally [`resume_in_place`](Self::resume_in_place)
    /// or [`abort`](Self::abort).
    ///
    /// # Errors
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop has already exited.
    pub async fn snapshot(&self) -> Result<WorkflowSnapshot, WorkflowError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.control_tx
            .send(WorkflowControl::Snapshot { reply: reply_tx })
            .map_err(|_| WorkflowError::ChannelClosed)?;
        reply_rx.await.unwrap_or(Err(WorkflowError::ChannelClosed))
    }

    /// Deliver a human-in-the-loop response to a workflow that auto-parked
    /// on an [`InputRequestEvent`].
    ///
    /// The event loop unparks and injects the response as a routable event.
    ///
    /// # Errors
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop has already exited.
    pub fn respond_to_input(&self, response: InputResponseEvent) -> Result<(), WorkflowError> {
        self.control_tx
            .send(WorkflowControl::InputResponse(response))
            .map_err(|_| WorkflowError::ChannelClosed)
    }

    /// Tear down the event loop.
    ///
    /// After this call the loop exits and any pending [`result`](Self::result)
    /// will resolve with [`WorkflowError::ChannelClosed`].
    ///
    /// # Errors
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop has already exited.
    pub fn abort(&self) -> Result<(), WorkflowError> {
        self.control_tx
            .send(WorkflowControl::Abort)
            .map_err(|_| WorkflowError::ChannelClosed)
    }

    /// Collect the workflow execution history after the workflow completes.
    ///
    /// This method drains all history events from the internal channel and
    /// returns a [`WorkflowHistory`](blazen_telemetry::WorkflowHistory)
    /// with properly sequenced events.
    ///
    /// Should be called **after** [`result()`](Self::result) or
    /// [`pause()`](Self::pause) to ensure all history events have been
    /// emitted by the event loop.
    ///
    /// Returns `None` if history collection was not enabled on the
    /// [`WorkflowBuilder`](crate::WorkflowBuilder) or if the history
    /// receiver was already consumed.
    ///
    /// Requires the `telemetry` feature.
    #[cfg(feature = "telemetry")]
    pub fn collect_history(
        &mut self,
        run_id: uuid::Uuid,
        workflow_name: String,
    ) -> Option<blazen_telemetry::WorkflowHistory> {
        let mut rx = self.history_rx.take()?;
        let mut history = blazen_telemetry::WorkflowHistory::new(run_id, workflow_name);

        // Drain all events from the channel (the sender side is dropped
        // when the event loop exits, so try_recv will eventually return
        // Empty or Disconnected).
        while let Ok(mut event) = rx.try_recv() {
            event.sequence = history.events.len() as u64;
            history.events.push(event);
        }

        Some(history)
    }
}

impl Drop for WorkflowHandler {
    fn drop(&mut self) {
        // Best-effort abort so the spawned event-loop task doesn't leak.
        // Ignore errors — the loop may have already exited.
        let _ = self.control_tx.send(WorkflowControl::Abort);
    }
}
