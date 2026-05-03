//! Python wrapper for [`Workflow`](blazen_core::Workflow) and
//! [`WorkflowBuilder`](blazen_core::WorkflowBuilder).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_core::SessionRefDeserializerFn;
use blazen_core::session_ref::{
    SERIALIZED_SESSION_REFS_META_KEY, SessionPausePolicy as CoreSessionPausePolicy,
};
use blazen_core::snapshot::WorkflowSnapshot;
use blazen_llm::retry::RetryConfig;

use super::handler::PyWorkflowHandler;
use super::step::PyStepWrapper;
use super::subworkflow::{PyParallelSubWorkflowsStep, PySubWorkflowStep};
use crate::convert::dict_to_json;
use crate::error::{BlazenPyError, to_py_result};
use crate::session_ref_serializable::{DESERIALIZER_FN, intern_type_tag};

// ---------------------------------------------------------------------------
// SessionPausePolicy enum
// ---------------------------------------------------------------------------

/// Policy applied to live session references when a workflow is paused
/// or snapshotted.
///
/// Mirrors the Rust-side
/// [`SessionPausePolicy`](blazen_core::session_ref::SessionPausePolicy)
/// enum. Configure it on a [`Workflow`] by passing
/// `session_pause_policy=SessionPausePolicy.PickleOrSerialize` to the
/// constructor.
///
/// Variants:
///     PickleOrError: Best-effort pickle each live ref; fail the pause
///         with a descriptive error if a ref is not picklable. This is
///         the default.
///     PickleOrSerialize: Same as `PickleOrError` but also honours the
///         `__blazen_serialize__` / `__blazen_deserialize__` protocol.
///         Values whose class implements those dunders are captured as
///         opaque bytes in snapshot metadata and reconstructed on
///         resume via `Workflow.resume_with_session_refs`.
///     WarnDrop: Log a warning and drop each live ref. The snapshot
///         will no longer contain any references to the dropped
///         values, and downstream `__blazen_session_ref__` markers
///         become unresolved.
///     HardError: Fail the pause immediately if any live refs exist.
#[gen_stub_pyclass_enum]
#[pyclass(name = "SessionPausePolicy", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySessionPausePolicy {
    PickleOrError,
    PickleOrSerialize,
    WarnDrop,
    HardError,
}

impl From<PySessionPausePolicy> for CoreSessionPausePolicy {
    fn from(p: PySessionPausePolicy) -> Self {
        match p {
            PySessionPausePolicy::PickleOrError => Self::PickleOrError,
            PySessionPausePolicy::PickleOrSerialize => Self::PickleOrSerialize,
            PySessionPausePolicy::WarnDrop => Self::WarnDrop,
            PySessionPausePolicy::HardError => Self::HardError,
        }
    }
}

impl From<CoreSessionPausePolicy> for PySessionPausePolicy {
    fn from(p: CoreSessionPausePolicy) -> Self {
        match p {
            CoreSessionPausePolicy::PickleOrError => Self::PickleOrError,
            CoreSessionPausePolicy::PickleOrSerialize => Self::PickleOrSerialize,
            CoreSessionPausePolicy::WarnDrop => Self::WarnDrop,
            CoreSessionPausePolicy::HardError => Self::HardError,
        }
    }
}

/// A validated, ready-to-run workflow.
///
/// Construct a `Workflow` by providing a name and a list of step wrappers
/// (created with the `@step` decorator).
///
/// Example:
///     >>> @step
///     ... async def echo(ctx, ev):
///     ...     return StopEvent(result=ev.to_dict())
///     >>>
///     >>> wf = Workflow("echo-wf", [echo])
///     >>> handler = await wf.run({"message": "hello"})
///     >>> result = await handler.result()
#[gen_stub_pyclass]
#[pyclass(name = "Workflow")]
pub struct PyWorkflow {
    name: String,
    steps: Vec<Py<PyStepWrapper>>,
    /// Sub-workflow steps registered via
    /// [`PyWorkflowBuilder::add_subworkflow_step`]. Materialized into
    /// `blazen_core::SubWorkflowStep` lazily inside
    /// `build_workflow_with_locals` so each child workflow's Python step
    /// handlers run on the active asyncio loop.
    subworkflow_steps: Vec<Py<PySubWorkflowStep>>,
    /// Parallel sub-workflow fan-out steps registered via
    /// [`PyWorkflowBuilder::add_parallel_subworkflows`].
    parallel_subworkflow_steps: Vec<Py<PyParallelSubWorkflowsStep>>,
    timeout: Option<f64>,
    session_pause_policy: PySessionPausePolicy,
    /// Whether to publish lifecycle events on the broadcast stream.
    /// Defaults to `true` (matches the canonical Rust default flip from
    /// Wave 7).
    auto_publish_events: bool,
    /// Workflow-level default `RetryConfig`. Layered into the
    /// `Context.retry_stack` at the workflow scope so every LLM call
    /// inherits unless a step or per-call override fires.
    retry_config: Option<Arc<RetryConfig>>,
    /// Optional per-step default timeout applied to every step that
    /// hasn't set its own. Currently a passthrough placeholder — wiring
    /// it through the canonical `WorkflowBuilder::step_timeout` requires
    /// per-step access, so for now we store it on the Python wrapper for
    /// introspection and bake it into individual steps when materializing
    /// the workflow if the canonical builder grows a "default-step-timeout"
    /// helper (tracked sibling-crate tweak).
    step_timeout_default: Option<Duration>,
}

impl PyWorkflow {
    /// Construct a `PyWorkflow` directly from its parts. Used by the
    /// Rust-side `WorkflowBuilder::build` path which has already
    /// validated the inputs and just needs to assemble the struct
    /// without going through the `#[new]` PyO3 constructor.
    pub(crate) fn from_parts(
        name: String,
        steps: Vec<Py<PyStepWrapper>>,
        timeout: Option<f64>,
        session_pause_policy: PySessionPausePolicy,
    ) -> Self {
        Self {
            name,
            steps,
            subworkflow_steps: Vec::new(),
            parallel_subworkflow_steps: Vec::new(),
            timeout,
            session_pause_policy,
            auto_publish_events: true,
            retry_config: None,
            step_timeout_default: None,
        }
    }

    /// Replace the workflow's sub-workflow step list. Called from
    /// [`PyWorkflowBuilder::build`].
    pub(crate) fn set_subworkflow_steps(&mut self, steps: Vec<Py<PySubWorkflowStep>>) {
        self.subworkflow_steps = steps;
    }

    /// Replace the workflow's parallel-sub-workflow step list.
    pub(crate) fn set_parallel_subworkflow_steps(
        &mut self,
        steps: Vec<Py<PyParallelSubWorkflowsStep>>,
    ) {
        self.parallel_subworkflow_steps = steps;
    }

    /// Set whether lifecycle events are auto-published. Used by the
    /// `WorkflowBuilder` plumbing.
    pub(crate) fn set_auto_publish_events(&mut self, enabled: bool) {
        self.auto_publish_events = enabled;
    }

    /// Set the workflow-level default retry config (or clear with `None`).
    pub(crate) fn set_retry_config(&mut self, cfg: Option<Arc<RetryConfig>>) {
        self.retry_config = cfg;
    }

    /// Set the workflow-level default step timeout (or clear with `None`).
    pub(crate) fn set_step_timeout_default(&mut self, t: Option<Duration>) {
        self.step_timeout_default = t;
    }

    /// Build a `blazen_core::Workflow` from this `PyWorkflow` using the
    /// provided Python task locals. Used by `Pipeline` to materialize
    /// stage workflows at run time.
    pub(crate) fn build_workflow_with_locals(
        &self,
        py: Python<'_>,
        locals: pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<blazen_core::Workflow> {
        let mut builder = blazen_core::WorkflowBuilder::new(&self.name);
        for step in &self.steps {
            let step_ref = step.borrow(py);
            let mut registration = step_ref.to_registration_with_locals(locals.clone())?;
            if let Some(t) = self.step_timeout_default
                && registration.timeout.is_none()
            {
                registration = registration.with_timeout(t);
            }
            builder = builder.step(registration);
        }
        for sub in &self.subworkflow_steps {
            let sub_ref = sub.borrow(py);
            let core_step = sub_ref.to_core(py, locals.clone())?;
            builder = builder.add_subworkflow_step(core_step);
        }
        for par in &self.parallel_subworkflow_steps {
            let par_ref = par.borrow(py);
            let core_step = par_ref.to_core(py, locals.clone())?;
            builder = builder.add_parallel_subworkflows(core_step);
        }
        if let Some(t) = self.timeout {
            builder = builder.timeout(Duration::from_secs_f64(t));
        }
        builder = builder.session_pause_policy(self.session_pause_policy.into());
        builder = builder.auto_publish_events(self.auto_publish_events);
        if let Some(cfg) = self.retry_config.as_ref() {
            builder = builder.retry_config((**cfg).clone());
        }
        builder.build().map_err(|e| BlazenPyError::from(e).into())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWorkflow {
    /// Create a new workflow.
    ///
    /// Args:
    ///     name: A human-readable name for the workflow.
    ///     steps: A list of `_StepWrapper` objects created by `@step`.
    ///     timeout: Optional timeout in seconds (default: 300).
    ///     session_pause_policy: How to treat live session refs when the
    ///         workflow is paused or snapshotted. Defaults to
    ///         `SessionPausePolicy.PickleOrError`. Use
    ///         `SessionPausePolicy.PickleOrSerialize` to honour the
    ///         `__blazen_serialize__` / `__blazen_deserialize__`
    ///         protocol on value classes.
    #[new]
    #[pyo3(signature = (
        name,
        steps,
        timeout=None,
        session_pause_policy=None,
        auto_publish_events=None,
        retry_config=None,
    ))]
    fn new(
        name: &str,
        steps: Vec<PyRef<'_, PyStepWrapper>>,
        timeout: Option<f64>,
        session_pause_policy: Option<PySessionPausePolicy>,
        auto_publish_events: Option<bool>,
        retry_config: Option<PyRef<'_, crate::providers::config::PyRetryConfig>>,
    ) -> Self {
        let step_refs: Vec<Py<PyStepWrapper>> = steps.into_iter().map(Into::into).collect();

        Self {
            name: name.to_string(),
            steps: step_refs,
            subworkflow_steps: Vec::new(),
            parallel_subworkflow_steps: Vec::new(),
            timeout,
            session_pause_policy: session_pause_policy
                .unwrap_or(PySessionPausePolicy::PickleOrError),
            auto_publish_events: auto_publish_events.unwrap_or(true),
            retry_config: retry_config.map(|c| Arc::new(c.inner.clone())),
            step_timeout_default: None,
        }
    }

    /// Execute the workflow with a JSON payload.
    ///
    /// The payload is wrapped in a `StartEvent` and dispatched to the
    /// workflow engine.
    ///
    /// Args:
    ///     input: A dict of data to pass as the start event payload.
    ///
    /// Returns:
    ///     A `WorkflowHandler` for awaiting results or streaming events.
    ///
    /// Example:
    ///     >>> handler = await wf.run({"prompt": "Hello!"})
    ///     >>> result = await handler.result()
    #[pyo3(signature = (**kwargs))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, WorkflowHandler]", imports = ("typing",)))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;

        let workflow = self.build_workflow_with_locals(py, locals.clone())?;

        // Phase 0.4: capture the parent session-ref registry BEFORE
        // entering the async block. This call must run on the Python
        // asyncio thread (where `py` is bound), because the Python
        // ContextVar that holds the active registry is only visible
        // from inside the current asyncio Task. Capturing up front
        // means: if the caller is inside a parent `@step` body, we
        // read the parent's registry here and thread it into
        // `run_with_registry` so the child inherits it.
        let parent_registry = crate::workflow::session_ref::current_session_registry();

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = if let Some(parent_registry) = parent_registry {
                workflow
                    .run_with_registry(data, parent_registry)
                    .await
                    .map_err(BlazenPyError::from)?
            } else {
                workflow.run(data).await.map_err(BlazenPyError::from)?
            };
            to_py_result(Ok(PyWorkflowHandler::new(handler)))
        })
    }

    /// Resume a workflow from a previously captured snapshot.
    ///
    /// This is a static method -- call it as `Workflow.resume(...)`.
    ///
    /// The steps must be the same set (or a compatible superset) of steps
    /// that were registered when the workflow was originally created.
    ///
    /// Args:
    ///     snapshot_json: A JSON string produced by `WorkflowHandler.pause()`.
    ///     steps: A list of `_StepWrapper` objects created by `@step`.
    ///     timeout: Optional timeout in seconds (default: 300).
    ///
    /// Returns:
    ///     A new `WorkflowHandler` for the resumed workflow.
    ///
    /// Example:
    ///     >>> handler = await Workflow.resume(snapshot_json, [step1, step2])
    ///     >>> result = await handler.result()
    #[staticmethod]
    #[pyo3(signature = (snapshot_json, steps, timeout=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, WorkflowHandler]", imports = ("typing",)))]
    fn resume<'py>(
        py: Python<'py>,
        snapshot_json: &str,
        steps: Vec<PyRef<'_, PyStepWrapper>>,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let snapshot = WorkflowSnapshot::from_json(snapshot_json).map_err(BlazenPyError::from)?;

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;

        let mut registrations = Vec::with_capacity(steps.len());
        for step in &steps {
            registrations.push(step.to_registration_with_locals(locals.clone())?);
        }

        let timeout_dur = timeout.map(Duration::from_secs_f64);

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = blazen_core::Workflow::resume(snapshot, registrations, timeout_dur)
                .await
                .map_err(BlazenPyError::from)?;
            to_py_result(Ok(PyWorkflowHandler::new(handler)))
        })
    }

    /// Resume a workflow from a snapshot, reconstructing every
    /// `SessionPausePolicy.PickleOrSerialize` session-ref entry via
    /// the Python `__blazen_deserialize__` classmethod on each value's
    /// original class.
    ///
    /// This is the resume-side counterpart of setting
    /// `session_pause_policy=SessionPausePolicy.PickleOrSerialize` on
    /// the workflow that produced the snapshot. For every entry stored
    /// under the snapshot's serialized-session-refs metadata, the
    /// class `module.qualname` baked into the payload is imported and
    /// `cls.__blazen_deserialize__(data)` is invoked to produce a
    /// fresh instance; the rebuilt instance is re-inserted into the
    /// resumed workflow's session-ref registry under the *original*
    /// [`RegistryKey`] so `__blazen_session_ref__` markers carried in
    /// reinjected events keep resolving.
    ///
    /// Snapshots that do NOT contain any serialized session refs work
    /// fine with the plain `Workflow.resume` entrypoint. Use this
    /// method only when you need the serializable rehydration path.
    ///
    /// Args:
    ///     snapshot_json: A JSON string produced by
    ///         `WorkflowHandler.snapshot()` from a workflow configured
    ///         with `SessionPausePolicy.PickleOrSerialize`.
    ///     steps: A list of `_StepWrapper` objects created by `@step`,
    ///         equivalent to the set registered on the original
    ///         workflow.
    ///     timeout: Optional timeout in seconds (default: no timeout).
    ///
    /// Returns:
    ///     A new `WorkflowHandler` for the resumed workflow.
    ///
    /// Example:
    ///     >>> class Blob:
    ///     ...     def __init__(self, n: int) -> None:
    ///     ...         self.n = n
    ///     ...     def __blazen_serialize__(self) -> bytes:
    ///     ...         return self.n.to_bytes(4, "big")
    ///     ...     @classmethod
    ///     ...     def __blazen_deserialize__(cls, data: bytes) -> "Blob":
    ///     ...         return cls(int.from_bytes(data, "big"))
    ///     >>>
    ///     >>> handler = await Workflow.resume_with_session_refs(snap, [s])
    #[staticmethod]
    #[pyo3(signature = (snapshot_json, steps, timeout=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, WorkflowHandler]", imports = ("typing",)))]
    fn resume_with_session_refs<'py>(
        py: Python<'py>,
        snapshot_json: &str,
        steps: Vec<PyRef<'_, PyStepWrapper>>,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let snapshot = WorkflowSnapshot::from_json(snapshot_json).map_err(BlazenPyError::from)?;

        // Walk the snapshot metadata and intern every type tag referenced
        // in the serialized session refs sidecar. Each unique tag gets a
        // `&'static str` pointer via the global intern pool, which is the
        // form required by `HashMap<&'static str, SessionRefDeserializerFn>`.
        let mut deserializers: HashMap<&'static str, SessionRefDeserializerFn> = HashMap::new();
        if let Some(serde_json::Value::Object(entries)) =
            snapshot.metadata.get(SERIALIZED_SESSION_REFS_META_KEY)
        {
            for record in entries.values() {
                if let Some(type_tag) = record.get("type_tag").and_then(serde_json::Value::as_str) {
                    let interned = intern_type_tag(type_tag);
                    deserializers.insert(interned, DESERIALIZER_FN);
                }
            }
        }

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;

        let mut registrations = Vec::with_capacity(steps.len());
        for step in &steps {
            registrations.push(step.to_registration_with_locals(locals.clone())?);
        }

        let timeout_dur = timeout.map(Duration::from_secs_f64);

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = blazen_core::Workflow::resume_with_deserializers(
                snapshot,
                registrations,
                deserializers,
                timeout_dur,
            )
            .await
            .map_err(BlazenPyError::from)?;
            to_py_result(Ok(PyWorkflowHandler::new(handler)))
        })
    }

    /// Create a fluent [`WorkflowBuilder`] for the given workflow name.
    ///
    /// Mirrors the Node.js / wasm-sdk surface. Use this when you want to
    /// configure the workflow piecemeal rather than passing every option
    /// to ``Workflow(...)`` directly.
    ///
    /// Example:
    ///     >>> wf = Workflow.builder("my-wf").step(my_step).timeout(60.0).build()
    #[staticmethod]
    fn builder(name: &str) -> super::builder::PyWorkflowBuilder {
        super::builder::PyWorkflowBuilder::with_name(name)
    }

    fn __repr__(&self) -> String {
        "Workflow(...)".to_owned()
    }
}
