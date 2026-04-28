//! Python wrappers for pipeline snapshot/result types.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_pipeline::{ActiveWorkflowSnapshot, PipelineResult, PipelineSnapshot, StageResult};

use crate::convert::json_to_py;
use crate::pipeline::error::pipeline_err;

/// The outcome of a single completed stage.
#[gen_stub_pyclass]
#[pyclass(name = "StageResult", frozen)]
pub struct PyStageResult {
    pub(crate) inner: StageResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStageResult {
    /// The name of the stage.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The output produced by the stage's workflow.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.output)
    }

    /// Whether the stage was skipped due to its condition returning `False`.
    #[getter]
    fn skipped(&self) -> bool {
        self.inner.skipped
    }

    /// How long the stage took to execute, in milliseconds.
    #[getter]
    fn duration_ms(&self) -> u64 {
        self.inner.duration_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "StageResult(name={:?}, skipped={}, duration_ms={})",
            self.inner.name, self.inner.skipped, self.inner.duration_ms
        )
    }
}

/// A snapshot of an in-progress workflow within a stage, captured when
/// a pipeline is paused mid-stage.
#[gen_stub_pyclass]
#[pyclass(name = "ActiveWorkflowSnapshot", frozen)]
pub struct PyActiveWorkflowSnapshot {
    pub(crate) inner: ActiveWorkflowSnapshot,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyActiveWorkflowSnapshot {
    /// The name of the stage that owns this workflow.
    #[getter]
    fn stage_name(&self) -> &str {
        &self.inner.stage_name
    }

    /// For parallel stages, the name of the specific branch.
    /// `None` for sequential stages.
    #[getter]
    fn branch_name(&self) -> Option<String> {
        self.inner.branch_name.clone()
    }

    /// Workflow snapshot serialized as a JSON string.
    fn workflow_snapshot_json(&self) -> PyResult<String> {
        self.inner
            .workflow_snapshot
            .to_json()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ActiveWorkflowSnapshot(stage_name={:?}, branch_name={:?})",
            self.inner.stage_name, self.inner.branch_name
        )
    }
}

/// Complete snapshot of a pipeline's state at the moment it was paused.
///
/// Pass this to [`Pipeline.resume`](crate::pipeline::pipeline::PyPipeline::resume)
/// to continue the pipeline from where it left off.
#[gen_stub_pyclass]
#[pyclass(name = "PipelineSnapshot", frozen)]
pub struct PyPipelineSnapshot {
    pub(crate) inner: PipelineSnapshot,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineSnapshot {
    /// The name of the pipeline.
    #[getter]
    fn pipeline_name(&self) -> &str {
        &self.inner.pipeline_name
    }

    /// Unique identifier for this pipeline run.
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// When the snapshot was captured (ISO 8601 string).
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    /// Index of the stage that was executing (or about to execute) when
    /// the pipeline was paused.
    #[getter]
    fn current_stage_index(&self) -> usize {
        self.inner.current_stage_index
    }

    /// Results from stages that completed before the pause.
    #[getter]
    fn completed_stages(&self) -> Vec<PyStageResult> {
        self.inner
            .completed_stages
            .iter()
            .map(|s| PyStageResult { inner: s.clone() })
            .collect()
    }

    /// Snapshots of workflows that were actively executing when the pause
    /// signal arrived.
    #[getter]
    fn active_snapshots(&self) -> Vec<PyActiveWorkflowSnapshot> {
        self.inner
            .active_snapshots
            .iter()
            .map(|s| PyActiveWorkflowSnapshot { inner: s.clone() })
            .collect()
    }

    /// The original pipeline input.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.input)
    }

    /// The shared key/value state at pause time, as a Python dict.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn shared_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let map: serde_json::Map<String, serde_json::Value> = self
            .inner
            .shared_state
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        json_to_py(py, &serde_json::Value::Object(map))
    }

    /// Serialize the snapshot to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(pipeline_err)
    }

    /// Serialize the snapshot to a pretty-printed JSON string.
    fn to_json_pretty(&self) -> PyResult<String> {
        self.inner.to_json_pretty().map_err(pipeline_err)
    }

    /// Deserialize a snapshot from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PipelineSnapshot::from_json(json).map_err(pipeline_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineSnapshot(pipeline_name={:?}, current_stage_index={}, completed_stages={})",
            self.inner.pipeline_name,
            self.inner.current_stage_index,
            self.inner.completed_stages.len()
        )
    }
}

/// The final output of a successfully completed pipeline run.
#[gen_stub_pyclass]
#[pyclass(name = "PipelineResult", frozen)]
pub struct PyPipelineResult {
    pub(crate) inner: PipelineResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineResult {
    /// The name of the pipeline.
    #[getter]
    fn pipeline_name(&self) -> &str {
        &self.inner.pipeline_name
    }

    /// Unique identifier for this pipeline run.
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// The output of the last stage (the pipeline's final result).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn final_output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.final_output)
    }

    /// Results from every stage, in execution order.
    #[getter]
    fn stage_results(&self) -> Vec<PyStageResult> {
        self.inner
            .stage_results
            .iter()
            .map(|s| PyStageResult { inner: s.clone() })
            .collect()
    }

    /// The shared key/value state at completion time, as a Python dict.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn shared_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let map: serde_json::Map<String, serde_json::Value> = self
            .inner
            .shared_state
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        json_to_py(py, &serde_json::Value::Object(map))
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineResult(pipeline_name={:?}, run_id={}, stage_count={})",
            self.inner.pipeline_name,
            self.inner.run_id,
            self.inner.stage_results.len()
        )
    }
}
