//! Python wrapper for [`WorkflowHandler`](blazen_core::WorkflowHandler).
//!
//! Provides three consumption modes:
//! 1. `await handler.result()` -- get the final workflow result.
//! 2. `async for event in handler.stream_events()` -- stream intermediate events.
//! 3. `await handler.pause()` -- pause the workflow and get a JSON snapshot for later resumption.

use std::sync::Arc;

use pyo3::prelude::*;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use crate::error::BlazenPyError;
use crate::event::any_event_to_py_event;

/// Handle to a running workflow.
///
/// Use `result()` to await the final result, or `stream_events()` to
/// iterate over intermediate events published by steps.
///
/// Example:
///     >>> handler = await wf.run(prompt="Hello")
///     >>> result = await handler.result()
///     >>> print(result.to_dict())
#[pyclass(name = "WorkflowHandler")]
pub struct PyWorkflowHandler {
    /// The inner handler is wrapped in `Option` because `result()` consumes it.
    inner: Arc<Mutex<Option<blazen_core::WorkflowHandler>>>,
}

impl PyWorkflowHandler {
    /// Create a new handler wrapping a Rust `WorkflowHandler`.
    pub fn new(handler: blazen_core::WorkflowHandler) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
        }
    }
}

#[pymethods]
impl PyWorkflowHandler {
    /// Await the final workflow result.
    ///
    /// Consumes the handler. Returns the terminal event (typically a
    /// `StopEvent`). Raises `RuntimeError` if the handler was already
    /// consumed.
    ///
    /// Returns:
    ///     The final Event produced by the workflow.
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handler = {
                let mut guard = inner.lock().await;
                guard
                    .take()
                    .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?
            };

            let result = handler.result().await.map_err(BlazenPyError::from)?;

            let py_event = any_event_to_py_event(&*result);
            Ok(py_event)
        })
    }

    /// Create an async iterator over intermediate events.
    ///
    /// Steps publish events to the stream via `ctx.write_event_to_stream()`.
    /// Each call returns a fresh stream starting from the current point.
    ///
    /// Returns:
    ///     An async iterator of Events.
    ///
    /// Example:
    ///     >>> async for event in handler.stream_events():
    ///     ...     print(event.event_type, event.to_dict())
    fn stream_events(&self) -> PyResult<PyEventStream> {
        let inner_guard = self.inner.clone();

        // We need to subscribe before any events are published, so we do it
        // synchronously. The handler is NOT consumed by subscribing.
        // We use try_lock here since this is called from sync Python context.
        let guard = inner_guard
            .try_lock()
            .map_err(|_| BlazenPyError::Workflow("Handler is locked".to_owned()))?;

        let handler = guard
            .as_ref()
            .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?;

        let stream = handler.stream_events();
        Ok(PyEventStream {
            // We box and pin the stream for storage
            stream: Arc::new(Mutex::new(Box::pin(stream))),
        })
    }

    /// Pause the running workflow and return a JSON snapshot.
    ///
    /// Consumes the handler. The returned JSON string contains the full
    /// workflow state and can be passed to `Workflow.resume()` to continue
    /// execution later.
    ///
    /// Raises `RuntimeError` if the handler was already consumed.
    ///
    /// Returns:
    ///     A JSON string representing the workflow snapshot.
    ///
    /// Example:
    ///     >>> snapshot_json = await handler.pause()
    ///     >>> # ... save snapshot_json to disk / database ...
    ///     >>> handler = await Workflow.resume(snapshot_json, [step1, step2])
    fn pause<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handler = {
                let mut guard = inner.lock().await;
                guard
                    .take()
                    .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?
            };

            let snapshot = handler.pause().await.map_err(BlazenPyError::from)?;
            let json = snapshot.to_json().map_err(BlazenPyError::from)?;
            Ok(json)
        })
    }

    fn __repr__(&self) -> String {
        "WorkflowHandler(...)".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Async iterator for streaming events
// ---------------------------------------------------------------------------

type PinnedEventStream = std::pin::Pin<
    Box<dyn tokio_stream::Stream<Item = Box<dyn blazen_events::AnyEvent>> + Send + Unpin>,
>;

/// Async iterator over streamed workflow events.
///
/// Implements the Python `__aiter__` / `__anext__` protocol so it can be
/// used with `async for`.
#[pyclass(name = "_EventStream")]
pub struct PyEventStream {
    stream: Arc<Mutex<PinnedEventStream>>,
}

#[pymethods]
impl PyEventStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            match guard.next().await {
                Some(event) => {
                    let py_event = any_event_to_py_event(&*event);
                    Ok(Some(py_event))
                }
                None => {
                    // Signal end of iteration by raising StopAsyncIteration
                    Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                        "stream exhausted",
                    ))
                }
            }
        })
    }
}
