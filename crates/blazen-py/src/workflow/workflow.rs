//! Python wrapper for [`Workflow`](blazen_core::Workflow) and
//! [`WorkflowBuilder`](blazen_core::WorkflowBuilder).

use std::collections::HashMap;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_core::SessionRefDeserializerFn;
use blazen_core::session_ref::{
    SERIALIZED_SESSION_REFS_META_KEY, SessionPausePolicy as CoreSessionPausePolicy,
};
use blazen_core::snapshot::WorkflowSnapshot;

use super::handler::PyWorkflowHandler;
use super::step::PyStepWrapper;
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
    timeout: Option<f64>,
    session_pause_policy: PySessionPausePolicy,
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
    #[pyo3(signature = (name, steps, timeout=None, session_pause_policy=None))]
    fn new(
        name: &str,
        steps: Vec<PyRef<'_, PyStepWrapper>>,
        timeout: Option<f64>,
        session_pause_policy: Option<PySessionPausePolicy>,
    ) -> Self {
        let step_refs: Vec<Py<PyStepWrapper>> = steps.into_iter().map(Into::into).collect();

        Self {
            name: name.to_string(),
            steps: step_refs,
            timeout,
            session_pause_policy: session_pause_policy
                .unwrap_or(PySessionPausePolicy::PickleOrError),
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

        // Build the workflow here with task locals
        let mut builder = blazen_core::WorkflowBuilder::new(&self.name);
        for step in &self.steps {
            let step_ref = step.borrow(py);
            let registration = step_ref.to_registration_with_locals(locals.clone())?;
            builder = builder.step(registration);
        }
        if let Some(t) = self.timeout {
            builder = builder.timeout(Duration::from_secs_f64(t));
        }
        builder = builder.session_pause_policy(self.session_pause_policy.into());
        let workflow = builder.build().map_err(BlazenPyError::from)?;

        // Phase 0.4: capture the parent session-ref registry BEFORE
        // entering the async block. This call must run on the Python
        // asyncio thread (where `py` is bound), because the Python
        // ContextVar that holds the active registry is only visible
        // from inside the current asyncio Task — once we move into the
        // Tokio-scheduled `async move` block below, that context is
        // gone. Capturing up front means: if the caller is inside a
        // parent `@step` body, we read the parent's registry here and
        // thread it into `run_with_registry` so the child inherits it.
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

    fn __repr__(&self) -> String {
        "Workflow(...)".to_owned()
    }
}
