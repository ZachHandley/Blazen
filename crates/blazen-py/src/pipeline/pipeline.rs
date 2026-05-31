//! Python wrapper for [`blazen_pipeline::Pipeline`].
//!
//! Holds the deferred stage definitions accumulated by `PipelineBuilder`.
//! `start()` and `resume()` materialize them into a `blazen_pipeline::Pipeline`
//! at call time using the current Python task locals so that step closures
//! capture the right asyncio loop.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::retry::RetryConfig;

use crate::convert::{dict_to_json, py_to_json};
use crate::pipeline::error::pipeline_err;
use crate::pipeline::handler::PyPipelineHandler;
use crate::pipeline::snapshot::{PyPipelineResult, PyPipelineSnapshot};
use crate::pipeline::stage::PendingStage;

/// A validated, ready-to-run pipeline.
///
/// Returned by [`PipelineBuilder.build`](super::builder::PyPipelineBuilder).
/// Execute via [`start`](Self::start). Resume from a saved snapshot via
/// [`resume`](Self::resume).
#[gen_stub_pyclass]
#[pyclass(name = "Pipeline")]
#[allow(clippy::struct_excessive_bools)]
pub struct PyPipeline {
    pub(crate) name: String,
    pub(crate) pending_stages: Vec<PendingStage>,
    pub(crate) on_persist: Option<Py<PyAny>>,
    pub(crate) on_persist_json: Option<Py<PyAny>>,
    pub(crate) timeout_per_stage: Option<Duration>,
    pub(crate) total_timeout: Option<Duration>,
    pub(crate) total_timeout_set: bool,
    pub(crate) retry_config: Option<Arc<RetryConfig>>,
    pub(crate) retry_config_set: bool,
}

impl PyPipeline {
    /// Materialize a `blazen_pipeline::Pipeline` from the deferred stage
    /// definitions using the supplied Python task locals so each step
    /// closure captures the active asyncio loop.
    ///
    /// Shared by `start`, `run`, `run_with_handler`, `resume`, and the
    /// `SubPipelineStep` materialization path. Non-consuming: the
    /// `on_persist` / `on_persist_json` callbacks are cloned (not taken),
    /// so the pipeline can be materialized repeatedly (e.g. once per
    /// invocation of an embedded sub-pipeline).
    pub(crate) fn build_pipeline_with_locals(
        &self,
        py: Python<'_>,
        locals: &pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<blazen_pipeline::Pipeline> {
        let mut builder = blazen_pipeline::PipelineBuilder::new(self.name.clone());
        for pending in &self.pending_stages {
            let stage_kind = pending.materialize(py, locals)?;
            builder = match stage_kind {
                blazen_pipeline::StageKind::Sequential(s) => builder.stage(s),
                blazen_pipeline::StageKind::Parallel(p) => builder.parallel(p),
                blazen_pipeline::StageKind::Loop(l) => builder.loop_stage(l),
            };
        }
        if let Some(t) = self.timeout_per_stage {
            builder = builder.timeout_per_stage(t);
        }
        if self.total_timeout_set {
            match self.total_timeout {
                Some(t) => builder = builder.total_timeout(t),
                None => builder = builder.no_total_timeout(),
            }
        }
        if self.retry_config_set {
            match self.retry_config.as_ref() {
                Some(cfg) => builder = builder.retry_config((**cfg).clone()),
                None => builder = builder.clear_retry_config(),
            }
        }
        if let Some(cb) = self.on_persist.as_ref() {
            builder = builder.on_persist(build_persist_fn(cb.clone_ref(py)));
        }
        if let Some(cb) = self.on_persist_json.as_ref() {
            builder = builder.on_persist_json(build_persist_json_fn(cb.clone_ref(py)));
        }
        builder.build().map_err(pipeline_err)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipeline {
    /// Execute the pipeline and await its final result.
    ///
    /// Args:
    ///     **kwargs: Initial input as keyword arguments. Wrapped into a
    ///         dict and passed as the pipeline input.
    ///
    /// Returns:
    ///     The :class:`PipelineResult` produced by the pipeline.
    ///
    /// Note:
    ///     This awaits the pipeline to completion and resolves directly to
    ///     the result. To pause, resume, stream events, or respond to input
    ///     requests, use :meth:`start` / :meth:`run_with_handler` instead,
    ///     which return a :class:`PipelineHandler`.
    #[pyo3(signature = (**kwargs))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineResult]", imports = ("typing",)))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let pipeline = self.build_pipeline_with_locals(py, &locals)?;

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let result = pipeline.run(input).await.map_err(pipeline_err)?;
            Ok(PyPipelineResult { inner: result })
        })
    }

    /// Execute the pipeline.
    ///
    /// Args:
    ///     **kwargs: Initial input as keyword arguments. Wrapped into a
    ///         dict and passed as the pipeline input.
    ///
    /// Returns:
    ///     A `PipelineHandler` for awaiting the result, streaming events,
    ///     or pausing.
    #[pyo3(signature = (**kwargs))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineHandler]", imports = ("typing",)))]
    fn start<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let pipeline = self.build_pipeline_with_locals(py, &locals)?;

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = pipeline.start(input);
            Ok(PyPipelineHandler::new(handler))
        })
    }

    /// Execute the pipeline and return a :class:`PipelineHandler` for
    /// pausing, resuming, streaming events, or responding to input requests.
    ///
    /// Alias of :meth:`start`, mirroring `Workflow.run_with_handler`. Use
    /// this instead of :meth:`run` when you need control over the running
    /// pipeline rather than just its final result.
    ///
    /// Args:
    ///     **kwargs: Initial input as keyword arguments.
    ///
    /// Returns:
    ///     A `PipelineHandler`.
    #[pyo3(signature = (**kwargs))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineHandler]", imports = ("typing",)))]
    fn run_with_handler<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.start(py, kwargs)
    }

    /// Resume a pipeline from a previously captured snapshot.
    ///
    /// Args:
    ///     snapshot: A `PipelineSnapshot` produced by `PipelineHandler.pause()`.
    ///
    /// Returns:
    ///     A new `PipelineHandler` for the resumed pipeline.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, PipelineHandler]", imports = ("typing",)))]
    fn resume<'py>(
        &self,
        py: Python<'py>,
        snapshot: PyRef<'_, PyPipelineSnapshot>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let pipeline = self.build_pipeline_with_locals(py, &locals)?;
        let snap = snapshot.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = pipeline.resume(snap).map_err(pipeline_err)?;
            Ok(PyPipelineHandler::new(handler))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Pipeline(name={:?}, stages={})",
            self.name,
            self.pending_stages.len()
        )
    }
}

fn build_persist_fn(cb: Py<PyAny>) -> blazen_pipeline::PersistFn {
    let cb = Arc::new(cb);
    Arc::new(
        move |snapshot: blazen_pipeline::PipelineSnapshot| -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<(), blazen_pipeline::PipelineError>> + Send,
            >,
        > {
            let cb = Arc::clone(&cb);
            Box::pin(async move {
                let py_snapshot = PyPipelineSnapshot { inner: snapshot };
                Python::attach(|py| -> Result<(), blazen_pipeline::PipelineError> {
                    let py_snap = Py::new(py, py_snapshot).map_err(|e| {
                        blazen_pipeline::PipelineError::PersistFailed(e.to_string())
                    })?;
                    let result = cb.call1(py, (py_snap,)).map_err(|e| {
                        blazen_pipeline::PipelineError::PersistFailed(e.to_string())
                    })?;
                    let _ = result;
                    Ok(())
                })
            })
        },
    )
}

fn build_persist_json_fn(cb: Py<PyAny>) -> blazen_pipeline::PersistJsonFn {
    let cb = Arc::new(cb);
    Arc::new(
        move |json: String| -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<(), blazen_pipeline::PipelineError>> + Send,
            >,
        > {
            let cb = Arc::clone(&cb);
            Box::pin(async move {
                Python::attach(|py| -> Result<(), blazen_pipeline::PipelineError> {
                    cb.call1(py, (json,)).map_err(|e| {
                        blazen_pipeline::PipelineError::PersistFailed(e.to_string())
                    })?;
                    Ok(())
                })
            })
        },
    )
}

// Suppress unused-import warning on `py_to_json`.
const _: fn() = || {
    let _ = py_to_json;
};
