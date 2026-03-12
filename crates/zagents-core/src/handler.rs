//! [`WorkflowHandler`] -- the handle returned after starting a workflow.
//!
//! Provides two consumption modes:
//!
//! 1. **Await the final result** -- either via [`WorkflowHandler::result`] or
//!    by using the [`IntoFuture`] implementation (`handler.await`).
//! 2. **Stream intermediate events** -- via [`WorkflowHandler::stream_events`]
//!    which subscribes to the broadcast channel that steps can publish to.
//!
//! The two modes are composable: you can subscribe a stream first, then await
//! the final result.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::task::{self, Poll};

use tokio::sync::{broadcast, oneshot};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use zagents_events::AnyEvent;

use crate::error::WorkflowError;

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
}

impl WorkflowHandler {
    /// Create a new handler (crate-internal).
    pub(crate) fn new(
        result_rx: oneshot::Receiver<Result<Box<dyn AnyEvent>, WorkflowError>>,
        stream_tx: broadcast::Sender<Box<dyn AnyEvent>>,
    ) -> Self {
        Self {
            result_rx: Some(result_rx),
            stream_tx,
        }
    }

    /// Await the final workflow result.
    ///
    /// Consumes the handler. Returns the terminal event (typically a
    /// [`StopEvent`](zagents_events::StopEvent)) or a [`WorkflowError`].
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
        rx.await.unwrap_or(Err(WorkflowError::ChannelClosed))
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
}

// ---------------------------------------------------------------------------
// IntoFuture -- allows `handler.await`
// ---------------------------------------------------------------------------

/// Future type backing the `IntoFuture` implementation for `WorkflowHandler`.
pub struct WorkflowHandlerFuture {
    rx: oneshot::Receiver<Result<Box<dyn AnyEvent>, WorkflowError>>,
}

impl Future for WorkflowHandlerFuture {
    type Output = Result<Box<dyn AnyEvent>, WorkflowError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(WorkflowError::ChannelClosed)),
            Poll::Pending => Poll::Pending,
        }
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
        WorkflowHandlerFuture { rx }
    }
}
