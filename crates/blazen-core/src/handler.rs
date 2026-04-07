//! [`WorkflowHandler`] -- the handle returned after starting a workflow.
//!
//! Provides three consumption modes:
//!
//! 1. **Await the final result** -- either via [`WorkflowHandler::result`] or
//!    by using the [`IntoFuture`] implementation (`handler.await`).
//! 2. **Stream intermediate events** -- via [`WorkflowHandler::stream_events`]
//!    which subscribes to the broadcast channel that steps can publish to.
//! 3. **Pause the workflow** -- via [`WorkflowHandler::pause`] which sends a
//!    pause signal to the event loop and returns a serializable
//!    [`WorkflowSnapshot`](crate::snapshot::WorkflowSnapshot).
//!
//! Modes 1 and 2 are composable: you can subscribe a stream first, then await
//! the final result. Mode 3 consumes the handler (the workflow is stopped).

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{self, Poll};

use blazen_events::AnyEvent;
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinHandle;

#[cfg(feature = "telemetry")]
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::error::WorkflowError;
use crate::session_ref::SessionRefRegistry;
use crate::snapshot::WorkflowSnapshot;

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
    /// Sends the pause signal to the event loop (one-shot trigger).
    pause_tx: Option<oneshot::Sender<()>>,
    /// Receives the snapshot from the event loop after a pause.
    snapshot_rx: Option<oneshot::Receiver<WorkflowSnapshot>>,
    /// Handle to the spawned event loop task. Awaited during `result()` and
    /// `pause()` to ensure the task fully exits before returning, which
    /// prevents orphaned Tokio tasks from keeping runtimes alive (important
    /// for napi-rs / Node.js bindings).
    event_loop_handle: Option<JoinHandle<()>>,
    /// Live session-ref registry for this run. Cloned from the workflow's
    /// `Context` so that bindings can resolve `__blazen_session_ref__`
    /// markers carried by the final result *after* the event loop has
    /// exited and the original `Context` has been dropped.
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
        pause_tx: Option<oneshot::Sender<()>>,
        snapshot_rx: Option<oneshot::Receiver<WorkflowSnapshot>>,
        event_loop_handle: JoinHandle<()>,
        session_refs: Arc<SessionRefRegistry>,
        #[cfg(feature = "telemetry")] history_rx: Option<
            mpsc::UnboundedReceiver<blazen_telemetry::HistoryEvent>,
        >,
    ) -> Self {
        Self {
            result_rx: Some(result_rx),
            stream_tx,
            pause_tx,
            snapshot_rx,
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
    /// Panics if `result()` or `into_future()` was already called on this
    /// handler (the result receiver can only be consumed once).
    pub async fn result(mut self) -> Result<Box<dyn AnyEvent>, WorkflowError> {
        let rx = self
            .result_rx
            .take()
            .expect("result() called after result was already consumed");
        let result = rx.await.unwrap_or(Err(WorkflowError::ChannelClosed));

        // Wait for the event loop task to fully exit so there are no orphaned
        // Tokio tasks keeping runtimes alive (critical for napi-rs / Node.js).
        if let Some(handle) = self.event_loop_handle.take() {
            let _ = handle.await;
        }

        result
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

    /// Pause the running workflow and return a snapshot of its state.
    ///
    /// This method:
    ///
    /// 1. Sends a pause signal to the event loop.
    /// 2. Waits for all in-flight step tasks to complete.
    /// 3. Drains pending events from the internal channel.
    /// 4. Captures a full snapshot of context state, collected events,
    ///    pending events, and metadata.
    /// 5. Returns the [`WorkflowSnapshot`] which can be serialized and
    ///    later used with [`Workflow::resume`](crate::Workflow::resume).
    ///
    /// Consumes the handler since the workflow is no longer running after
    /// a pause.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::ChannelClosed`] if the event loop has
    /// already terminated (e.g. the workflow completed before pause was
    /// received) or if the pause/snapshot channels are unavailable.
    pub async fn pause(mut self) -> Result<WorkflowSnapshot, WorkflowError> {
        let pause_tx = self.pause_tx.take().ok_or(WorkflowError::ChannelClosed)?;

        let snapshot_rx = self
            .snapshot_rx
            .take()
            .ok_or(WorkflowError::ChannelClosed)?;

        // Send the pause signal.
        pause_tx
            .send(())
            .map_err(|()| WorkflowError::ChannelClosed)?;

        // Await the snapshot from the event loop.
        let snapshot = snapshot_rx
            .await
            .map_err(|_| WorkflowError::ChannelClosed)?;

        // Wait for the event loop task to fully exit.
        if let Some(handle) = self.event_loop_handle.take() {
            let _ = handle.await;
        }

        Ok(snapshot)
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

// ---------------------------------------------------------------------------
// IntoFuture -- allows `handler.await`
// ---------------------------------------------------------------------------

/// Future type backing the `IntoFuture` implementation for `WorkflowHandler`.
pub struct WorkflowHandlerFuture {
    rx: oneshot::Receiver<Result<Box<dyn AnyEvent>, WorkflowError>>,
    event_loop_handle: Option<JoinHandle<()>>,
    result: Option<Result<Box<dyn AnyEvent>, WorkflowError>>,
}

impl Future for WorkflowHandlerFuture {
    type Output = Result<Box<dyn AnyEvent>, WorkflowError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        // Phase 1: await the result from the oneshot channel.
        if self.result.is_none() {
            match Pin::new(&mut self.rx).poll(cx) {
                Poll::Ready(Ok(result)) => {
                    self.result = Some(result);
                }
                Poll::Ready(Err(_)) => {
                    self.result = Some(Err(WorkflowError::ChannelClosed));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Phase 2: await the event loop task to ensure clean shutdown.
        if let Some(handle) = &mut self.event_loop_handle {
            match Pin::new(handle).poll(cx) {
                Poll::Ready(_) => {
                    self.event_loop_handle = None;
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        Poll::Ready(self.result.take().expect("result was already consumed"))
    }
}

impl IntoFuture for WorkflowHandler {
    type Output = Result<Box<dyn AnyEvent>, WorkflowError>;
    type IntoFuture = WorkflowHandlerFuture;

    fn into_future(mut self) -> Self::IntoFuture {
        let rx = self
            .result_rx
            .take()
            .expect("IntoFuture: result was already consumed");
        WorkflowHandlerFuture {
            rx,
            event_loop_handle: self.event_loop_handle.take(),
            result: None,
        }
    }
}
