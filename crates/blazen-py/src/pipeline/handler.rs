//! Python wrapper for [`blazen_pipeline::PipelineHandler`].

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use crate::pipeline::error::pipeline_err;
use crate::pipeline::event::PyPipelineEvent;
use crate::pipeline::snapshot::{PyPipelineResult, PyPipelineSnapshot};

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

    /// Capture a snapshot without stopping the pipeline.
    ///
    /// Note: this is a stub in the underlying engine and currently raises
    /// `PipelineError`.
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
