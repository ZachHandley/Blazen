//! Python wrapper for [`blazen_pipeline::PipelineHandler`].

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use crate::events::PyProgressSnapshot;
use crate::pipeline::error::pipeline_err;
use crate::pipeline::event::PyPipelineEvent;
use crate::pipeline::snapshot::{PyPipelineResult, PyPipelineSnapshot};
use crate::types::PyTokenUsage;
use crate::workflow::event::PyInputResponseEvent;

/// Handle to a running pipeline.
///
/// Use `result()` to await the final `PipelineResult`, `stream_events()`
/// to iterate over intermediate events, `pause()` to capture a snapshot,
/// `resume_in_place()` to continue a paused pipeline, `snapshot()` to
/// capture state without stopping, or `abort()` to terminate.
#[gen_stub_pyclass]
#[pyclass(name = "PipelineHandler")]
pub struct PyPipelineHandler {
    inner: Arc<Mutex<Option<blazen_pipeline::PipelineHandler>>>,
    pre_stream: Arc<Mutex<PinnedPipelineEventStream>>,
}

type PinnedPipelineEventStream =
    std::pin::Pin<Box<dyn tokio_stream::Stream<Item = blazen_pipeline::PipelineEvent> + Send>>;

impl PyPipelineHandler {
    pub(crate) fn new(handler: blazen_pipeline::PipelineHandler) -> Self {
        let stream = handler.stream_events();
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
            pre_stream: Arc::new(Mutex::new(Box::pin(stream))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineHandler {
    /// Await the final pipeline result.
    ///
    /// Consumes the handler. Returns a `PipelineResult` containing the
    /// final output and all stage results, or raises `PipelineError`.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineResult]", imports = ("typing",)))]
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handler = {
                let mut guard = inner.lock().await;
                guard.take().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
                })?
            };
            let result = handler.result().await.map_err(pipeline_err)?;
            Ok(PyPipelineResult { inner: result })
        })
    }

    /// Create an async iterator over intermediate events from pipeline stages.
    fn stream_events(&self) -> PyPipelineEventStream {
        PyPipelineEventStream {
            stream: self.pre_stream.clone(),
        }
    }

    /// Pause the running pipeline and return a snapshot.
    ///
    /// Consumes the handler since the pipeline is no longer running after
    /// a pause. The returned `PipelineSnapshot` can be passed to
    /// `Pipeline.resume(...)`.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineSnapshot]", imports = ("typing",)))]
    fn pause<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handler = {
                let mut guard = inner.lock().await;
                guard.take().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
                })?
            };
            let snapshot = handler.pause().await.map_err(pipeline_err)?;
            Ok(PyPipelineSnapshot { inner: snapshot })
        })
    }

    /// Resume a paused pipeline in place.
    ///
    /// Forwards to the active stage's inner workflow(s) so a workflow parked
    /// on an `InputRequestEvent` (or paused) unparks and continues. A no-op
    /// between stages, where nothing is parked.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn resume_in_place<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
            })?;
            handler.resume_in_place().map_err(pipeline_err)?;
            Ok(())
        })
    }

    /// Capture a snapshot of the pipeline's current state without stopping it.
    ///
    /// Live and non-destructive: the pipeline keeps running. The returned
    /// `PipelineSnapshot` can be passed to `Pipeline.resume(...)`. Mirrors
    /// `WorkflowHandler.snapshot()`.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineSnapshot]", imports = ("typing",)))]
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
            })?;
            let snapshot = handler.snapshot().await.map_err(pipeline_err)?;
            Ok(PyPipelineSnapshot { inner: snapshot })
        })
    }

    /// Abort the running pipeline.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn abort<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
            })?;
            handler.abort().map_err(pipeline_err)?;
            Ok(())
        })
    }

    /// Deliver a human-in-the-loop response to the active stage's inner
    /// workflow(s).
    ///
    /// For a sequential stage this targets the one in-flight workflow; for a
    /// parallel stage the response is broadcast to every live branch, where
    /// the workflow that requested input consumes it and the others ignore a
    /// response they did not request.
    ///
    /// Args:
    ///     request_id: The ID from the `InputRequestEvent`, or an
    ///         `InputResponseEvent` carrying both id and response.
    ///     response: A Python dict/value that will be converted to JSON and
    ///         delivered to the waiting step. Required when `request_id`
    ///         is a string; ignored when an `InputResponseEvent` is passed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    #[pyo3(signature = (request_id, response=None))]
    fn respond_to_input<'py>(
        &self,
        py: Python<'py>,
        request_id: &Bound<'py, PyAny>,
        response: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input_response =
            if let Ok(event) = request_id.extract::<PyRef<'_, PyInputResponseEvent>>() {
                event.to_rust()
            } else {
                let id: String = request_id.extract()?;
                let resp_obj = response.ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "respond_to_input: 'response' is required when 'request_id' is a string",
                    )
                })?;
                blazen_events::InputResponseEvent {
                    request_id: id,
                    response: crate::convert::py_to_json(py, resp_obj)?,
                }
            };
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
            })?;
            handler
                .respond_to_input(input_response)
                .map_err(pipeline_err)?;
            Ok(())
        })
    }

    /// Snapshot the running aggregate `TokenUsage` for this pipeline run.
    ///
    /// Safe to call at any point during or after the run; the value matches
    /// `PipelineResult.usage_total` once `result()` has been awaited.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, TokenUsage]", imports = ("typing",)))]
    fn usage_total<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
            })?;
            let usage = handler.usage_total().await;
            Ok(PyTokenUsage::from(usage))
        })
    }

    /// Snapshot the running aggregate cost in USD for this pipeline run.
    ///
    /// Sums `UsageEvent::cost_usd` across every emitted usage event. After
    /// `result()` has returned, the value matches
    /// `PipelineResult.cost_total_usd`.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.float]", imports = ("typing", "builtins")))]
    fn cost_total_usd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let handler = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Handler already consumed")
            })?;
            Ok(handler.cost_total_usd().await)
        })
    }

    /// Snapshot the pipeline's current progress without affecting execution.
    ///
    /// Reads are best-effort and may briefly be one stage stale relative to
    /// the executor task. Returns `None` once the handler has been consumed
    /// by `result()` or `pause()`.
    fn progress(&self, py: Python<'_>) -> Option<PyProgressSnapshot> {
        let guard = py.detach(|| {
            let rt = pyo3_async_runtimes::tokio::get_runtime();
            rt.block_on(self.inner.lock())
        });
        guard
            .as_ref()
            .map(|h| PyProgressSnapshot::from(h.progress()))
    }

    fn __repr__(&self) -> String {
        "PipelineHandler(...)".to_owned()
    }
}

/// Async iterator over streamed pipeline events.
#[gen_stub_pyclass]
#[pyclass(name = "_PipelineEventStream")]
pub struct PyPipelineEventStream {
    stream: Arc<Mutex<PinnedPipelineEventStream>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineEventStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineEvent]", imports = ("typing",)))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            match guard.next().await {
                Some(event) => Ok(Some(PyPipelineEvent::from_inner(event))),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream exhausted",
                )),
            }
        })
    }
}
