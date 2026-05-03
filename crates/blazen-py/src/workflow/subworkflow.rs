//! Python wrappers for [`blazen_core::SubWorkflowStep`] and
//! [`blazen_core::ParallelSubWorkflowsStep`].
//!
//! `SubWorkflowStep` lets a parent workflow delegate to a child `Workflow` as a
//! step. `ParallelSubWorkflowsStep` fans out into multiple child workflows
//! concurrently and joins them via a [`JoinStrategy`].
//!
//! Both classes store a [`PyWorkflow`] reference and lazily materialize the
//! Rust [`blazen_core::Workflow`] inside
//! [`PyWorkflow::build_workflow_with_locals`] so the asyncio task locals
//! captured at run time flow through to each child workflow's step handlers.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::{JoinStrategy as CoreJoinStrategy, ParallelSubWorkflowsStep, SubWorkflowStep};
use blazen_events::intern_event_type;
use blazen_llm::retry::RetryConfig;

use crate::pipeline::stage::PyJoinStrategy;
use crate::providers::config::PyRetryConfig;
use crate::workflow::workflow::PyWorkflow;

// ---------------------------------------------------------------------------
// PySubWorkflowStep
// ---------------------------------------------------------------------------

/// A workflow step that delegates to another `Workflow`.
///
/// The parent workflow's event loop spawns the child via `Workflow.run()`,
/// converts the parent event to JSON for the child's input, and wraps the
/// child's terminal `StopEvent.result` into a `DynamicEvent` named
/// `"<step_name>::output"` for the parent.
///
/// Example:
///     >>> child = Workflow("enrich", [enrich_step])
///     >>> step = SubWorkflowStep(
///     ...     name="enrich",
///     ...     accepts=["StartEvent"],
///     ...     emits=["enrich::output"],
///     ...     workflow=child,
///     ... )
///     >>> wf = Workflow.builder("parent").add_subworkflow_step(step).build()
#[gen_stub_pyclass]
#[pyclass(name = "SubWorkflowStep")]
pub struct PySubWorkflowStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) workflow: Py<PyWorkflow>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) retry_config: Option<Arc<RetryConfig>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySubWorkflowStep {
    /// Create a sub-workflow step.
    ///
    /// Args:
    ///     name: Human-readable name for this step.
    ///     accepts: Event type identifiers this step accepts.
    ///     emits: Event type identifiers this step may emit.
    ///     workflow: The child `Workflow` to run as this step.
    ///     timeout: Optional per-step wall-clock timeout (seconds) for the
    ///         entire child run. ``None`` inherits the child's own timeout.
    ///     retry_config: Optional per-step `RetryConfig` applied to the
    ///         child run as a whole.
    #[new]
    #[pyo3(signature = (name, accepts, emits, workflow, timeout=None, retry_config=None))]
    fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        workflow: Py<PyWorkflow>,
        timeout: Option<f64>,
        retry_config: Option<PyRef<'_, PyRetryConfig>>,
    ) -> Self {
        Self {
            name,
            accepts,
            emits,
            workflow,
            timeout: timeout.map(Duration::from_secs_f64),
            retry_config: retry_config.map(|c| Arc::new(c.inner.clone())),
        }
    }

    /// The step name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Event type identifiers this step accepts.
    #[getter]
    fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[getter]
    fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SubWorkflowStep(name='{}', accepts={:?}, emits={:?})",
            self.name, self.accepts, self.emits
        )
    }
}

impl PySubWorkflowStep {
    /// Materialize a [`SubWorkflowStep`] using the supplied task locals so the
    /// child workflow's Python step handlers run on the active asyncio loop.
    pub(crate) fn to_core(
        &self,
        py: Python<'_>,
        locals: pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<SubWorkflowStep> {
        // intern_event_type returns &'static str; build owned vectors.
        let accepts: Vec<&'static str> =
            self.accepts.iter().map(|s| intern_event_type(s)).collect();
        let emits: Vec<&'static str> = self.emits.iter().map(|s| intern_event_type(s)).collect();

        let py_wf = self.workflow.borrow(py);
        let core_wf = py_wf.build_workflow_with_locals(py, locals)?;
        let mut step = SubWorkflowStep::with_json_mappers(
            self.name.clone(),
            accepts,
            emits,
            Arc::new(core_wf),
        );
        if let Some(t) = self.timeout {
            step = step.with_timeout(t);
        }
        if let Some(cfg) = self.retry_config.as_ref() {
            step = step.with_retry_config((**cfg).clone());
        }
        Ok(step)
    }
}

// ---------------------------------------------------------------------------
// PyParallelSubWorkflowsStep
// ---------------------------------------------------------------------------

/// Fan out into multiple parallel sub-workflow branches.
///
/// Each branch is a `SubWorkflowStep` that runs concurrently. The
/// `join_strategy` controls whether the parent waits for all branches
/// (`JoinStrategy.WaitAll`) or only the first to complete
/// (`JoinStrategy.FirstCompletes`).
///
/// Example:
///     >>> step = ParallelSubWorkflowsStep(
///     ...     name="enrich-fanout",
///     ...     accepts=["StartEvent"],
///     ...     emits=["enrich-fanout::output"],
///     ...     branches=[step_a, step_b, step_c],
///     ...     join_strategy=JoinStrategy.WaitAll,
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "ParallelSubWorkflowsStep")]
pub struct PyParallelSubWorkflowsStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) branches: Vec<Py<PySubWorkflowStep>>,
    pub(crate) join_strategy: PyJoinStrategy,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyParallelSubWorkflowsStep {
    /// Create a parallel sub-workflow fan-out step.
    ///
    /// Args:
    ///     name: Human-readable name for this fan-out step.
    ///     accepts: Event type identifiers this step accepts.
    ///     emits: Event type identifiers this step may emit.
    ///     branches: List of `SubWorkflowStep` instances to run concurrently.
    ///     join_strategy: How to join the branch results. Defaults to
    ///         `JoinStrategy.WaitAll`.
    #[new]
    #[pyo3(signature = (name, accepts, emits, branches, join_strategy=None))]
    fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        branches: Vec<Py<PySubWorkflowStep>>,
        join_strategy: Option<PyJoinStrategy>,
    ) -> Self {
        Self {
            name,
            accepts,
            emits,
            branches,
            join_strategy: join_strategy.unwrap_or(PyJoinStrategy::WaitAll),
        }
    }

    /// The step name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Event type identifiers this step accepts.
    #[getter]
    fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[getter]
    fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }

    /// The join strategy used to combine branch results.
    #[getter]
    fn join_strategy(&self) -> PyJoinStrategy {
        self.join_strategy
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelSubWorkflowsStep(name='{}', branches={}, join_strategy={:?})",
            self.name,
            self.branches.len(),
            self.join_strategy
        )
    }
}

impl PyParallelSubWorkflowsStep {
    /// Materialize a [`ParallelSubWorkflowsStep`] using the supplied task
    /// locals so every branch's child workflow step handlers run on the
    /// active asyncio loop.
    pub(crate) fn to_core(
        &self,
        py: Python<'_>,
        locals: pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<ParallelSubWorkflowsStep> {
        let accepts: Vec<&'static str> =
            self.accepts.iter().map(|s| intern_event_type(s)).collect();
        let emits: Vec<&'static str> = self.emits.iter().map(|s| intern_event_type(s)).collect();

        let mut branches = Vec::with_capacity(self.branches.len());
        for branch_py in &self.branches {
            let branch = branch_py.borrow(py);
            branches.push(branch.to_core(py, locals.clone())?);
        }

        // PyJoinStrategy converts to blazen_pipeline::JoinStrategy. The
        // `ParallelSubWorkflowsStep` field uses blazen_core's own
        // JoinStrategy (a separate but identically-shaped enum), so map
        // explicitly here.
        let join_strategy = match self.join_strategy {
            PyJoinStrategy::WaitAll => CoreJoinStrategy::WaitAll,
            PyJoinStrategy::FirstCompletes => CoreJoinStrategy::FirstCompletes,
        };
        Ok(ParallelSubWorkflowsStep {
            name: self.name.clone(),
            accepts,
            emits,
            branches,
            join_strategy,
        })
    }
}
