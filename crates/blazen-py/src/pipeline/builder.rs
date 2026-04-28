//! Python wrapper for [`blazen_pipeline::PipelineBuilder`].
//!
//! Constructs a [`PyPipeline`](super::pipeline::PyPipeline) from accumulated
//! sequential and parallel stage definitions. Stages are stored as deferred
//! references to `PyStage` / `PyParallelStage`; the actual `blazen_core::Workflow`
//! inside each stage is materialized at `Pipeline.start()` time so the current
//! Python task locals are captured.

use std::time::Duration;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::pipeline::pipeline::PyPipeline;
use crate::pipeline::stage::{PendingStage, PyParallelStage, PyStage};

/// Fluent builder for constructing a [`Pipeline`].
///
/// Example:
///     >>> ingest = Stage(name="ingest", workflow=ingest_wf)
///     >>> enrich = Stage(name="enrich", workflow=enrich_wf)
///     >>> pipeline = (
///     ...     PipelineBuilder("etl")
///     ...     .stage(ingest)
///     ...     .stage(enrich)
///     ...     .timeout_per_stage(60.0)
///     ...     .build()
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "PipelineBuilder")]
pub struct PyPipelineBuilder {
    pub(crate) name: String,
    pub(crate) pending_stages: Vec<PendingStage>,
    pub(crate) on_persist: Option<Py<PyAny>>,
    pub(crate) on_persist_json: Option<Py<PyAny>>,
    pub(crate) timeout_per_stage: Option<Duration>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineBuilder {
    /// Create a new builder with the given pipeline name.
    #[new]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            pending_stages: Vec::new(),
            on_persist: None,
            on_persist_json: None,
            timeout_per_stage: None,
        }
    }

    /// Append a sequential `Stage` to the pipeline.
    fn stage(mut slf: PyRefMut<'_, Self>, stage: Py<PyStage>) -> PyRefMut<'_, Self> {
        slf.pending_stages.push(PendingStage::Sequential(stage));
        slf
    }

    /// Append a `ParallelStage` (multiple branches) to the pipeline.
    fn parallel(mut slf: PyRefMut<'_, Self>, parallel: Py<PyParallelStage>) -> PyRefMut<'_, Self> {
        slf.pending_stages.push(PendingStage::Parallel(parallel));
        slf
    }

    /// Set a persist callback that receives a typed `PipelineSnapshot`
    /// after each stage completes. The callback may be sync or async.
    fn on_persist(mut slf: PyRefMut<'_, Self>, callback: Py<PyAny>) -> PyRefMut<'_, Self> {
        slf.on_persist = Some(callback);
        slf
    }

    /// Set a persist callback that receives the snapshot as a JSON string
    /// after each stage completes. The callback may be sync or async.
    fn on_persist_json(mut slf: PyRefMut<'_, Self>, callback: Py<PyAny>) -> PyRefMut<'_, Self> {
        slf.on_persist_json = Some(callback);
        slf
    }

    /// Set a per-stage timeout in seconds. Each stage's workflow will be
    /// given this duration before being considered timed out.
    fn timeout_per_stage(mut slf: PyRefMut<'_, Self>, seconds: f64) -> PyRefMut<'_, Self> {
        slf.timeout_per_stage = Some(Duration::from_secs_f64(seconds));
        slf
    }

    /// Validate and build the pipeline.
    ///
    /// Raises `PipelineError` if the pipeline has no stages or if any
    /// stage names are duplicated.
    fn build(&mut self) -> PyResult<PyPipeline> {
        if self.pending_stages.is_empty() {
            return Err(crate::pipeline::error::pipeline_err(
                blazen_pipeline::PipelineError::ValidationFailed(
                    "pipeline must have at least one stage".into(),
                ),
            ));
        }

        let pending = std::mem::take(&mut self.pending_stages);

        Ok(PyPipeline {
            name: self.name.clone(),
            pending_stages: pending,
            on_persist: self.on_persist.take(),
            on_persist_json: self.on_persist_json.take(),
            timeout_per_stage: self.timeout_per_stage,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineBuilder(name={:?}, stages={})",
            self.name,
            self.pending_stages.len()
        )
    }
}
