//! Python wrapper for [`blazen_pipeline::PipelineEvent`].

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::workflow::event::{PyEvent, any_event_to_py_event};

/// An event emitted by a pipeline stage, tagged with stage/branch
/// provenance.
#[gen_stub_pyclass]
#[pyclass(name = "PipelineEvent")]
pub struct PyPipelineEvent {
    pub(crate) stage_name: String,
    pub(crate) branch_name: Option<String>,
    pub(crate) workflow_run_id: String,
    pub(crate) event: PyEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineEvent {
    /// The name of the stage that produced this event.
    #[getter]
    fn stage_name(&self) -> &str {
        &self.stage_name
    }

    /// For parallel stages, the name of the specific branch.
    /// `None` for sequential stages.
    #[getter]
    fn branch_name(&self) -> Option<String> {
        self.branch_name.clone()
    }

    /// The workflow run ID that produced this event.
    #[getter]
    fn workflow_run_id(&self) -> &str {
        &self.workflow_run_id
    }

    /// The underlying workflow event.
    #[getter]
    fn event(&self) -> PyEvent {
        self.event.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineEvent(stage_name={:?}, branch_name={:?})",
            self.stage_name, self.branch_name
        )
    }
}

impl PyPipelineEvent {
    pub(crate) fn from_inner(inner: blazen_pipeline::PipelineEvent) -> Self {
        let event = any_event_to_py_event(&*inner.event);
        Self {
            stage_name: inner.stage_name,
            branch_name: inner.branch_name,
            workflow_run_id: inner.workflow_run_id.to_string(),
            event,
        }
    }
}
