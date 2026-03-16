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
    /// Pre-subscribed stream created at handler construction time so that
    /// events published before `stream_events()` is called are not lost.
    pre_stream: Arc<Mutex<PinnedEventStream>>,
}

impl PyWorkflowHandler {
    /// Create a new handler wrapping a Rust `WorkflowHandler`.
    ///
    /// Immediately subscribes to the broadcast stream so that events
    /// published by steps are captured from the very start.
    pub fn new(handler: blazen_core::WorkflowHandler) -> Self {
        // Subscribe immediately -- the returned stream is fully owned
        // (independent of &handler) thanks to `use<>` on the core method.
        let stream = handler.stream_events();
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
            pre_stream: Arc::new(Mutex::new(Box::pin(stream))),
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
    /// The stream is pre-subscribed at handler construction time so no events
    /// are lost between `wf.run()` and this call.
    ///
    /// The stream terminates when the workflow completes (a `blazen::StreamEnd`
    /// sentinel is sent by the event loop).
    ///
    /// Returns:
    ///     An async iterator of Events.
    ///
    /// Example:
    ///     >>> async for event in handler.stream_events():
    ///     ...     print(event.event_type, event.to_dict())
    fn stream_events(&self) -> PyResult<PyEventStream> {
        Ok(PyEventStream {
            stream: self.pre_stream.clone(),
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
///
/// The stream terminates when it receives a `"blazen::StreamEnd"` sentinel
/// event from the event loop, or when the underlying broadcast channel closes.
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
                    // Check for the stream-end sentinel sent by the event loop.
                    if event.event_type_id() == "blazen::StreamEnd" {
                        return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stream exhausted",
                        ));
                    }
                    let py_event = any_event_to_py_event(&*event);
                    Ok(Some(py_event))
                }
                None => {
                    // Broadcast channel closed -- no more events.
                    Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                        "stream exhausted",
                    ))
                }
            }
        })
    }
}
