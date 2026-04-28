//! Python wrapper for [`blazen_core::WorkflowBuilder`].
//!
//! Mirrors the Node.js / wasm-sdk surface: a fluent builder that registers
//! steps, configures timeouts and policies, and produces a [`Workflow`].
//! Python users can already build a `Workflow` directly via
//! `Workflow(name, steps, timeout=..., session_pause_policy=...)`; the
//! builder is the typed alternative that exposes every knob (auto-publish,
//! history collection, checkpointing).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::workflow::{PySessionPausePolicy, PyWorkflow};

/// Fluent builder for a [`Workflow`].
///
/// Each configuration method returns ``self`` so calls can be chained.
/// Call ``build()`` to produce a ready-to-run [`Workflow`].
///
/// Example:
///     >>> wf = (Workflow.builder("my-wf")
///     ...      .step(my_step)
///     ...      .timeout(60.0)
///     ...      .auto_publish_events(True)
///     ...      .build())
#[gen_stub_pyclass]
#[pyclass(name = "WorkflowBuilder")]
#[allow(dead_code)]
#[allow(clippy::struct_excessive_bools)]
pub struct PyWorkflowBuilder {
    pub(crate) name: String,
    pub(crate) steps: Vec<Py<super::step::PyStepWrapper>>,
    pub(crate) timeout: Option<f64>,
    pub(crate) timeout_set: bool,
    pub(crate) session_pause_policy: PySessionPausePolicy,
    pub(crate) auto_publish_events: bool,
    pub(crate) checkpoint_after_step: bool,
    pub(crate) collect_history: bool,
    pub(crate) checkpoint_store: Option<Py<PyAny>>,
}

impl PyWorkflowBuilder {
    /// Construct a fresh builder for the given workflow name. Used by
    /// the Rust-side `Workflow.builder(name)` static method, which
    /// cannot call the `#[new]` PyO3 constructor as a regular Rust fn.
    pub(crate) fn with_name(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            steps: Vec::new(),
            timeout: Some(300.0),
            timeout_set: false,
            session_pause_policy: PySessionPausePolicy::PickleOrError,
            auto_publish_events: false,
            checkpoint_after_step: false,
            collect_history: false,
            checkpoint_store: None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWorkflowBuilder {
    /// Create a new builder with the given workflow name.
    #[new]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            steps: Vec::new(),
            timeout: Some(300.0),
            timeout_set: false,
            session_pause_policy: PySessionPausePolicy::PickleOrError,
            auto_publish_events: false,
            checkpoint_after_step: false,
            collect_history: false,
            checkpoint_store: None,
        }
    }

    /// Register a step (an `@step`-decorated function).
    fn step(mut slf: PyRefMut<'_, Self>, registration: Py<super::step::PyStepWrapper>) -> Py<Self> {
        slf.steps.push(registration);
        slf.into()
    }

    /// Set the maximum execution time for the workflow in seconds.
    fn timeout(mut slf: PyRefMut<'_, Self>, seconds: f64) -> Py<Self> {
        slf.timeout = Some(seconds);
        slf.timeout_set = true;
        slf.into()
    }

    /// Disable the execution timeout (workflow runs until `StopEvent`).
    fn no_timeout(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.timeout = None;
        slf.timeout_set = true;
        slf.into()
    }

    /// Enable automatic publishing of lifecycle events to the broadcast
    /// stream. Defaults to ``False``.
    fn auto_publish_events(mut slf: PyRefMut<'_, Self>, enabled: bool) -> Py<Self> {
        slf.auto_publish_events = enabled;
        slf.into()
    }

    /// Set the session-pause policy for live session references.
    fn session_pause_policy(mut slf: PyRefMut<'_, Self>, policy: PySessionPausePolicy) -> Py<Self> {
        slf.session_pause_policy = policy;
        slf.into()
    }

    /// Enable collection of an append-only history of workflow events.
    fn with_history(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.collect_history = true;
        slf.into()
    }

    /// Configure the checkpoint store backing durable persistence.
    fn checkpoint_store(mut slf: PyRefMut<'_, Self>, store: Py<PyAny>) -> Py<Self> {
        slf.checkpoint_store = Some(store);
        slf.into()
    }

    /// Enable or disable automatic checkpointing after each step.
    fn checkpoint_after_step(mut slf: PyRefMut<'_, Self>, enabled: bool) -> Py<Self> {
        slf.checkpoint_after_step = enabled;
        slf.into()
    }

    /// Build the [`Workflow`].
    ///
    /// Validation: the workflow must have at least one registered step.
    fn build(&self, py: Python<'_>) -> PyResult<PyWorkflow> {
        if self.steps.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "workflow must have at least one step",
            ));
        }

        // Clone the per-step `Py` handles so the builder can be reused
        // and the resulting workflow owns its own references.
        let steps: Vec<Py<super::step::PyStepWrapper>> =
            self.steps.iter().map(|s| s.clone_ref(py)).collect();

        Ok(PyWorkflow::from_parts(
            self.name.clone(),
            steps,
            self.timeout,
            self.session_pause_policy,
        ))
    }

    /// The workflow name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkflowBuilder(name={:?}, steps={}, timeout={:?})",
            self.name,
            self.steps.len(),
            self.timeout
        )
    }
}
