//! Python wrapper for [`blazen_core::SubPipelineStep`].
//!
//! `SubPipelineStep` lets a parent `Workflow` delegate to a child
//! `Pipeline` as a single step — the pipeline analogue of
//! [`SubWorkflowStep`](crate::workflow::subworkflow::PySubWorkflowStep).
//! The parent workflow's event loop maps the triggering event to JSON,
//! runs the embedded pipeline to completion, and wraps the pipeline's
//! `final_output` into a `DynamicEvent` named `"<step_name>::output"`
//! for the parent.
//!
//! The wrapper stores a [`PyPipeline`] reference and lazily materializes
//! the Rust [`blazen_pipeline::Pipeline`] inside [`PySubPipelineStep::to_core`]
//! so the asyncio task locals captured at workflow-build time flow through
//! to every stage's step handlers.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::SubPipelineStep;
use blazen_events::intern_event_type;
use blazen_llm::retry::RetryConfig;

use crate::core_types::sub_executable::PySubExecutableAdapter;
use crate::pipeline::pipeline::PyPipeline;
use crate::providers::config::PyRetryConfig;

/// The child runner backing a [`PySubPipelineStep`] -- either an embedded
/// `Pipeline` or an arbitrary Python `SubExecutable` subclass.
pub(crate) enum SubExecutableSource {
    /// An embedded child pipeline.
    Pipeline(Py<PyPipeline>),
    /// A Python object implementing the `SubExecutable` ABC.
    Executable(Py<PyAny>),
}

/// A workflow step that delegates to a child `Pipeline`.
///
/// The parent workflow's event loop runs the embedded pipeline via
/// `Pipeline.run()`, converts the triggering event to JSON for the
/// pipeline's input, and wraps the pipeline's `final_output` into a
/// `DynamicEvent` named `"<step_name>::output"` for the parent. This is
/// the pipeline analogue of `SubWorkflowStep`.
///
/// Example:
/// ```text
///  >>> child = Pipeline.builder("enrich").stage(stage).build()
///  >>> step = SubPipelineStep(
///  ...     name="enrich",
///  ...     accepts=["StartEvent"],
///  ...     emits=["enrich::output"],
///  ...     pipeline=child,
///  ... )
///  >>> wf = Workflow.builder("parent").add_subpipeline_step(step).build()
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "SubPipelineStep")]
pub struct PySubPipelineStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) source: SubExecutableSource,
    pub(crate) timeout: Option<Duration>,
    pub(crate) retry_config: Option<Arc<RetryConfig>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySubPipelineStep {
    /// Create a sub-pipeline step.
    ///
    /// Args:
    ///     name: Human-readable name for this step.
    ///     accepts: Event type identifiers this step accepts.
    ///     emits: Event type identifiers this step may emit.
    ///     pipeline: The child `Pipeline` to run as this step.
    ///     timeout: Optional per-step wall-clock timeout (seconds) for the
    ///         entire child run. ``None`` inherits the child's own timeout.
    ///     retry_config: Optional per-step `RetryConfig` applied to the
    ///         child run as a whole.
    #[new]
    #[pyo3(signature = (name, accepts, emits, pipeline, timeout=None, retry_config=None))]
    fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        pipeline: Py<PyPipeline>,
        timeout: Option<f64>,
        retry_config: Option<PyRef<'_, PyRetryConfig>>,
    ) -> Self {
        Self {
            name,
            accepts,
            emits,
            source: SubExecutableSource::Pipeline(pipeline),
            timeout: timeout.map(Duration::from_secs_f64),
            retry_config: retry_config.map(|c| Arc::new(c.inner.clone())),
        }
    }

    /// Build a sub-pipeline step from any `SubExecutable` instance.
    ///
    /// Use this to embed a custom Python child runner (a `SubExecutable`
    /// subclass overriding ``async def execute(self, input, ctx)``) inside a
    /// parent workflow, instead of a concrete `Pipeline`.
    ///
    /// Args:
    ///     name: Human-readable name for this step.
    ///     accepts: Event type identifiers this step accepts.
    ///     emits: Event type identifiers this step may emit.
    ///     executable: A `SubExecutable` subclass instance to run as this step.
    ///     timeout: Optional per-step wall-clock timeout (seconds).
    ///     retry_config: Optional per-step `RetryConfig`.
    #[classmethod]
    #[pyo3(signature = (name, accepts, emits, executable, timeout=None, retry_config=None))]
    fn from_executable(
        _cls: &Bound<'_, pyo3::types::PyType>,
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        executable: Py<PyAny>,
        timeout: Option<f64>,
        retry_config: Option<PyRef<'_, PyRetryConfig>>,
    ) -> Self {
        Self {
            name,
            accepts,
            emits,
            source: SubExecutableSource::Executable(executable),
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
            "SubPipelineStep(name='{}', accepts={:?}, emits={:?})",
            self.name, self.accepts, self.emits
        )
    }
}

impl PySubPipelineStep {
    /// Materialize a [`SubPipelineStep`] using the supplied task locals so the
    /// child pipeline's Python stage step handlers run on the active asyncio
    /// loop.
    ///
    /// The embedded [`blazen_pipeline::Pipeline`] implements
    /// [`blazen_core::SubExecutable`]; each parent dispatch clones the
    /// blueprint (cheap — `Arc`-backed) and runs it to completion, surfacing
    /// the `final_output` JSON back to the parent.
    pub(crate) fn to_core(
        &self,
        py: Python<'_>,
        locals: pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<SubPipelineStep> {
        let accepts: Vec<&'static str> =
            self.accepts.iter().map(|s| intern_event_type(s)).collect();
        let emits: Vec<&'static str> = self.emits.iter().map(|s| intern_event_type(s)).collect();

        let executable: Arc<dyn blazen_core::SubExecutable> = match &self.source {
            SubExecutableSource::Pipeline(pipeline) => {
                let py_pipeline = pipeline.borrow(py);
                let core_pipeline = py_pipeline.build_pipeline_with_locals(py, &locals)?;
                Arc::new(core_pipeline)
            }
            SubExecutableSource::Executable(instance) => {
                Arc::new(PySubExecutableAdapter::new(instance.clone_ref(py)))
            }
        };

        let mut step =
            SubPipelineStep::with_json_mappers(self.name.clone(), accepts, emits, executable);
        if let Some(t) = self.timeout {
            step = step.with_timeout(t);
        }
        if let Some(cfg) = self.retry_config.as_ref() {
            step = step.with_retry_config((**cfg).clone());
        }
        Ok(step)
    }
}
