//! Python wrapper for [`WorkflowHandler`](blazen_core::WorkflowHandler).
//!
//! Provides two consumption modes plus several control methods:
//!
//! **Consumption (consumes the handler):**
//! 1. `await handler.result()` -- get the final workflow result.
//! 2. `async for event in handler.stream_events()` -- stream intermediate events.
//!
//! **Control (borrow the handler, can be called multiple times):**
//! 3. `await handler.pause()` -- pause the workflow.
//! 4. `await handler.resume_in_place()` -- resume a paused workflow.
//! 5. `await handler.snapshot()` -- capture the current state as a JSON string.
//! 6. `await handler.respond_to_input(request_id, response)` -- respond to an input request.
//! 7. `await handler.abort()` -- abort the workflow.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use blazen_core::session_ref::SessionRefRegistry;

use super::event::any_event_to_py_event;
use super::session_ref::with_session_registry;
use crate::error::BlazenPyError;

/// Handle to a running workflow.
///
/// Use `result()` to await the final result, or `stream_events()` to
/// iterate over intermediate events published by steps.
///
/// Example:
///     >>> handler = await wf.run(prompt="Hello")
///     >>> result = await handler.result()
///     >>> print(result.to_dict())
#[gen_stub_pyclass]
#[pyclass(name = "WorkflowHandler")]
pub struct PyWorkflowHandler {
    /// The inner handler is wrapped in `Option` because `result()` consumes it.
    inner: Arc<Mutex<Option<blazen_core::WorkflowHandler>>>,
    /// Pre-subscribed stream created at handler construction time so that
    /// events published before `stream_events()` is called are not lost.
    pre_stream: Arc<Mutex<PinnedEventStream>>,
    /// Live session-ref registry for this run, kept alive independently of
    /// the inner handler so streamed events and the final result can both
    /// resolve `__blazen_session_ref__` markers carried in event payloads.
    session_refs: Arc<SessionRefRegistry>,
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
        let session_refs = handler.session_refs();
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
            pre_stream: Arc::new(Mutex::new(Box::pin(stream))),
            session_refs,
        }
    }
}

#[gen_stub_pymethods]
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
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, Event]", imports = ("typing",)))]
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handler = {
                let mut guard = inner.lock().await;
                guard
                    .take()
                    .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?
            };

            let wf_result = handler.result().await.map_err(BlazenPyError::from)?;
            let session_refs = wf_result.session_refs;
            let event = wf_result.event;

            // Install the registry as the current session-ref scope while we
            // build the `PyEvent`. `any_event_to_py_event` calls
            // `current_session_registry()` to capture the Arc into the
            // returned `PyEvent`, so attribute access (`__getattr__`) keeps
            // working even after this future resolves.
            let py_event =
                with_session_registry(session_refs, async move { any_event_to_py_event(&*event) })
                    .await;
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
    fn stream_events(&self) -> PyEventStream {
        PyEventStream {
            stream: self.pre_stream.clone(),
            session_refs: Arc::clone(&self.session_refs),
        }
    }

    /// Pause the running workflow.
    ///
    /// Sends a pause signal to the workflow event loop. The workflow will
    /// park after the current step completes. Does **not** consume the
    /// handler -- you can later call `resume_in_place()`, `snapshot()`,
    /// or `abort()`.
    ///
    /// Raises `RuntimeError` if the handler was already consumed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn pause<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard
                .as_ref()
                .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?;
            handler.pause().map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Resume a paused workflow in place.
    ///
    /// Sends a resume signal so the workflow continues from where it was
    /// paused.
    ///
    /// Raises `RuntimeError` if the handler was already consumed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn resume_in_place<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard
                .as_ref()
                .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?;
            handler.resume_in_place().map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Capture a snapshot of the current workflow state.
    ///
    /// Returns a JSON string that can be passed to `Workflow.resume()` to
    /// continue execution later.
    ///
    /// Raises `RuntimeError` if the handler was already consumed.
    ///
    /// Returns:
    ///     A JSON string representing the workflow snapshot.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.str]", imports = ("typing", "builtins")))]
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard
                .as_ref()
                .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?;
            let snap = handler.snapshot().await.map_err(BlazenPyError::from)?;
            let json = snap.to_json().map_err(BlazenPyError::from)?;
            Ok(json)
        })
    }

    /// Respond to an input request from a workflow step.
    ///
    /// Args:
    ///     request_id: The ID from the `InputRequestEvent`.
    ///     response: A Python dict/value that will be converted to JSON and
    ///               delivered to the waiting step.
    ///
    /// Raises `RuntimeError` if the handler was already consumed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn respond_to_input<'py>(
        &self,
        py: Python<'py>,
        request_id: String,
        response: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let response_json = crate::convert::py_to_json(py, response)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard
                .as_ref()
                .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?;
            let input_response = blazen_events::InputResponseEvent {
                request_id,
                response: response_json,
            };
            handler
                .respond_to_input(input_response)
                .map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Abort the running workflow.
    ///
    /// Sends an abort signal. The workflow will terminate as soon as
    /// possible. Does **not** consume the handler.
    ///
    /// Raises `RuntimeError` if the handler was already consumed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn abort<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard
                .as_ref()
                .ok_or_else(|| BlazenPyError::Workflow("Handler already consumed".to_owned()))?;
            handler.abort().map_err(BlazenPyError::from)?;
            Ok(())
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
#[gen_stub_pyclass]
#[pyclass(name = "_EventStream")]
pub struct PyEventStream {
    stream: Arc<Mutex<PinnedEventStream>>,
    /// Session-ref registry kept alive across iterations so each yielded
    /// `PyEvent` can resolve `__blazen_session_ref__` markers carried in
    /// its payload.
    session_refs: Arc<SessionRefRegistry>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEventStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    // NOTE: `__anext__` is intentionally left as a manual
    // `future_into_py` wrapper. PyO3 0.28's async-iterator protocol
    // (`AsyncIterBaseTag`) uses a separate call-convention path for
    // `__anext__` / `__aiter__` and does not yet accept a native
    // `async fn` returning an awaitable. Converting this method fails
    // with `IntoPyCallbackOutput` not implemented for the returned
    // future. Keep the explicit bridge here.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, Event]", imports = ("typing",)))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        let session_refs = Arc::clone(&self.session_refs);
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
                    let py_event = with_session_registry(session_refs, async move {
                        any_event_to_py_event(&*event)
                    })
                    .await;
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
