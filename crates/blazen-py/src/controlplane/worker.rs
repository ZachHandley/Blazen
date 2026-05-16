//! Python wrappers for the worker-side control-plane bindings.
//!
//! Exposes:
//!
//! - [`PyControlPlaneWorkerConfig`] — chainable builder mirroring
//!   [`blazen_controlplane::WorkerConfig`].
//! - [`PyAssignmentContext`] — handed to user handlers; lets them emit
//!   non-terminal events back to the control plane.
//! - [`PyAssignmentHandler`] — abstract base class subclassed from
//!   Python. The user overrides `handle` (required) and optionally
//!   `on_cancel`, `on_drain`, `evaluate_offer`. The Rust adapter
//!   ([`PyAssignmentHandlerAdapter`]) re-enters Python under the GIL on
//!   each call.
//! - [`PyControlPlaneWorker`] — `connect` (synchronous validation) and
//!   `run` (async driver) for the worker session.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use uuid::Uuid;

use blazen_controlplane::protocol::{Assignment, DeclineReason, Offer, OfferOutcome};
use blazen_controlplane::{
    AssignmentContext as CoreAssignmentContext, AssignmentFailure, AssignmentHandler, Worker,
    WorkerConfig,
};

use crate::convert::{json_to_py, py_to_json};
use crate::error::BlazenException;

use super::types::{PyAdmissionMode, PyControlPlaneWorkerCapability};

// ===========================================================================
// Exception helpers
// ===========================================================================

pyo3::create_exception!(blazen, ControlPlaneException, BlazenException);

pub(crate) fn cp_err(err: blazen_controlplane::ControlPlaneError) -> PyErr {
    ControlPlaneException::new_err(err.to_string())
}

pub(crate) fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "ControlPlaneError",
        m.py().get_type::<ControlPlaneException>(),
    )?;
    Ok(())
}

// ===========================================================================
// PyControlPlaneWorkerConfig
// ===========================================================================

/// Chainable builder for a [`Worker`]'s configuration. Mirrors
/// [`blazen_controlplane::WorkerConfig`].
///
/// Construct via the `__init__` (endpoint + node_id) and chain the
/// builder methods. Each method returns a new instance so the chain
/// composes cleanly.
#[gen_stub_pyclass]
#[pyclass(name = "ControlPlaneWorkerConfig", from_py_object)]
#[derive(Clone)]
pub struct PyControlPlaneWorkerConfig {
    pub(crate) inner: WorkerConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyControlPlaneWorkerConfig {
    /// Base config with sensible defaults: ``Fixed { max_in_flight: 1 }``
    /// admission, 5s heartbeat interval, no TLS, default exponential
    /// retry policy.
    #[new]
    fn new(endpoint: String, node_id: String) -> Self {
        Self {
            inner: WorkerConfig::new(endpoint, node_id),
        }
    }

    /// Append a capability the worker advertises at handshake.
    fn with_capability(&self, cap: PyControlPlaneWorkerCapability) -> Self {
        Self {
            inner: self.inner.clone().with_capability(cap.inner),
        }
    }

    /// Insert a single ``key=value`` tag.
    fn with_tag(&self, key: String, value: String) -> Self {
        Self {
            inner: self.inner.clone().with_tag(key, value),
        }
    }

    /// Override the admission mode.
    fn with_admission(&self, admission: PyAdmissionMode) -> Self {
        Self {
            inner: self.inner.clone().with_admission(admission.inner),
        }
    }

    /// Override the heartbeat cadence in milliseconds.
    fn with_heartbeat_interval_ms(&self, ms: u64) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_heartbeat_interval(Duration::from_millis(ms)),
        }
    }

    /// Enable mTLS by loading a client identity + CA from PEM files.
    ///
    /// Raises:
    ///     ControlPlaneError: If any PEM file cannot be read or the
    ///         resulting TLS config is rejected.
    fn with_mtls(&self, cert_path: String, key_path: String, ca_path: String) -> PyResult<Self> {
        let cert = PathBuf::from(cert_path);
        let key = PathBuf::from(key_path);
        let ca = PathBuf::from(ca_path);
        let inner = self
            .inner
            .clone()
            .with_mtls(&cert, &key, &ca)
            .map_err(cp_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn endpoint(&self) -> &str {
        &self.inner.endpoint
    }

    #[getter]
    fn node_id(&self) -> &str {
        &self.inner.node_id
    }

    fn __repr__(&self) -> String {
        format!(
            "ControlPlaneWorkerConfig(endpoint={:?}, node_id={:?}, capabilities={}, tags={})",
            self.inner.endpoint,
            self.inner.node_id,
            self.inner.capabilities.len(),
            self.inner.tags.len(),
        )
    }
}

// ===========================================================================
// PyAssignmentContext
// ===========================================================================

/// Per-assignment context handed to a Python handler's `handle`
/// implementation.
///
/// Holds an `Arc` to the underlying Rust [`CoreAssignmentContext`] so
/// `emit_event` can forward non-terminal events back to the control
/// plane while the assignment is running.
#[gen_stub_pyclass]
#[pyclass(name = "AssignmentContext")]
pub struct PyAssignmentContext {
    inner: Arc<CoreAssignmentContext>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAssignmentContext {
    /// UUID of the run this context belongs to, as a string.
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id().to_string()
    }

    /// Emit a non-terminal event from the running assignment. The
    /// returned coroutine completes once the frame is queued for send.
    ///
    /// Args:
    ///     event_type: Free-form event kind (e.g. ``"step.start"``,
    ///         ``"progress"``, ``"workflow.error"``).
    ///     data: A JSON-serializable payload. ``None`` is allowed.
    ///
    /// Raises:
    ///     ControlPlaneError: If the worker's outbound channel is closed
    ///         (the session has disconnected).
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, None]",
        imports = ("typing",)
    ))]
    fn emit_event<'py>(
        &self,
        py: Python<'py>,
        event_type: String,
        data: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let data_json = match data {
            Some(value) if !value.is_none() => py_to_json(py, value)?,
            _ => serde_json::Value::Null,
        };
        let ctx = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            ctx.emit_event(&event_type, data_json).await.map_err(cp_err)
        })
    }

    fn __repr__(&self) -> String {
        format!("AssignmentContext(run_id={})", self.inner.run_id())
    }
}

// ===========================================================================
// PyAssignmentHandler -- abstract base
// ===========================================================================

/// Abstract base class for worker assignment handlers.
///
/// Subclass this from Python and override `handle` (required) and
/// optionally `on_cancel`, `on_drain`, `evaluate_offer`. Pass the
/// instance to [`PyControlPlaneWorker::run`].
///
/// The methods may be `async def` (preferred) or plain `def`; the
/// adapter awaits any returned coroutine via
/// `pyo3_async_runtimes::into_future_with_locals`.
#[gen_stub_pyclass]
#[pyclass(name = "AssignmentHandler", subclass)]
pub struct PyAssignmentHandler {
    _private: (),
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAssignmentHandler {
    #[new]
    fn new() -> Self {
        Self { _private: () }
    }

    /// Run an assignment to completion. The default implementation
    /// raises ``NotImplementedError`` — override in a subclass.
    ///
    /// Args:
    ///     assignment: A dict carrying the workflow name, JSON-decoded
    ///         input, attempt counter, and optional deadline.
    ///     ctx: Per-assignment context (see `AssignmentContext`).
    ///
    /// Returns:
    ///     A JSON-serializable value reported back to the control plane
    ///     as the run's terminal output.
    #[pyo3(signature = (assignment, ctx))]
    fn handle(&self, assignment: &Bound<'_, PyAny>, ctx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let _ = (assignment, ctx);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AssignmentHandler.handle must be overridden by a subclass",
        ))
    }

    /// Hook fired when the control plane cancels an in-flight run.
    /// Defaults to a no-op.
    #[pyo3(signature = (run_id))]
    fn on_cancel(&self, run_id: String) {
        let _ = run_id;
    }

    /// Hook fired when the control plane sends a drain instruction.
    /// Defaults to a no-op.
    #[pyo3(signature = (immediate))]
    fn on_drain(&self, immediate: bool) {
        let _ = immediate;
    }

    /// Decide whether to claim a reactive offer. Default: always claim.
    /// Return ``"claim"`` to take the assignment or ``"decline"`` to
    /// let the control plane try the next candidate.
    #[pyo3(signature = (offer))]
    fn evaluate_offer(&self, offer: &Bound<'_, PyAny>) -> String {
        let _ = offer;
        "claim".to_owned()
    }
}

// ===========================================================================
// PyAssignmentHandlerAdapter -- Python ↔ Rust trait bridge
// ===========================================================================

/// Adapter that lets a Python-side [`PyAssignmentHandler`] subclass
/// satisfy the Rust [`AssignmentHandler`] trait. Holds the Python
/// instance and a snapshot of the current asyncio `TaskLocals` so it
/// can await coroutines on the same event loop the caller used to
/// invoke `Worker.run`.
pub(crate) struct PyAssignmentHandlerAdapter {
    py_obj: Py<PyAny>,
    locals: TaskLocals,
}

impl PyAssignmentHandlerAdapter {
    pub(crate) fn new(py_obj: Py<PyAny>, locals: TaskLocals) -> Self {
        Self { py_obj, locals }
    }
}

/// One of: a value already produced under the GIL, or a future that
/// will produce one once awaited on the asyncio loop.
enum CallPrep {
    Value(Py<PyAny>),
    Future(std::pin::Pin<Box<dyn std::future::Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Call `instance.method_name(args...)` and await any returned
/// coroutine on the supplied asyncio locals. If the return value is
/// not a coroutine it is returned directly. Result is the final
/// Python value (unbound).
async fn call_handler_method(
    instance: &Py<PyAny>,
    locals: &TaskLocals,
    method: &str,
    build_args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Py<PyAny>>> + Send,
) -> PyResult<Py<PyAny>> {
    let prep: PyResult<CallPrep> = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<CallPrep> {
            let bound = instance.bind(py);
            let args_py = build_args(py)?;
            let bound_args: Vec<Bound<'_, PyAny>> =
                args_py.into_iter().map(|p| p.into_bound(py)).collect();
            let py_args = pyo3::types::PyTuple::new(py, bound_args.iter())?;
            let result = bound.call_method1(method, py_args)?;
            // Detect coroutine: if it has an `__await__` attribute,
            // forward to the asyncio event loop. Otherwise return as-is.
            let is_awaitable = result.hasattr("__await__").unwrap_or(false);
            if is_awaitable {
                let fut = pyo3_async_runtimes::into_future_with_locals(locals, result)?;
                Ok(CallPrep::Future(Box::pin(fut)))
            } else {
                Ok(CallPrep::Value(result.unbind()))
            }
        })
    });

    match prep? {
        CallPrep::Value(v) => Ok(v),
        CallPrep::Future(fut) => pyo3_async_runtimes::tokio::scope(locals.clone(), fut).await,
    }
}

#[async_trait]
impl AssignmentHandler for PyAssignmentHandlerAdapter {
    async fn handle(
        &self,
        assignment: Assignment,
        ctx: CoreAssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        let ctx_arc = Arc::new(ctx);
        let ctx_for_call = Arc::clone(&ctx_arc);

        let result = call_handler_method(
            &self.py_obj,
            &self.locals,
            "handle",
            move |py| -> PyResult<Vec<Py<PyAny>>> {
                let assignment_dict = assignment_to_pydict(py, &assignment)?;
                let py_ctx = Py::new(
                    py,
                    PyAssignmentContext {
                        inner: ctx_for_call,
                    },
                )?;
                Ok(vec![assignment_dict.into_any().unbind(), py_ctx.into_any()])
            },
        )
        .await;

        match result {
            Ok(value) => Python::attach(|py| -> Result<serde_json::Value, AssignmentFailure> {
                let bound = value.bind(py);
                if bound.is_none() {
                    return Ok(serde_json::Value::Null);
                }
                py_to_json(py, bound).map_err(|e| {
                    AssignmentFailure::new(format!(
                        "AssignmentHandler.handle returned a non-JSON-serializable value: {e}"
                    ))
                })
            }),
            Err(e) => Err(AssignmentFailure::new(format!(
                "AssignmentHandler.handle raised: {e}"
            ))),
        }
    }

    async fn on_cancel(&self, run_id: Uuid) {
        let res = call_handler_method(
            &self.py_obj,
            &self.locals,
            "on_cancel",
            move |py| -> PyResult<Vec<Py<PyAny>>> {
                let s = pyo3::types::PyString::new(py, &run_id.to_string());
                Ok(vec![s.into_any().unbind()])
            },
        )
        .await;
        if let Err(e) = res {
            tracing::warn!(error = %e, %run_id, "AssignmentHandler.on_cancel raised");
        }
    }

    async fn on_drain(&self, immediate: bool) {
        let res = call_handler_method(
            &self.py_obj,
            &self.locals,
            "on_drain",
            move |py| -> PyResult<Vec<Py<PyAny>>> {
                let b = pyo3::types::PyBool::new(py, immediate);
                Ok(vec![b.to_owned().into_any().unbind()])
            },
        )
        .await;
        if let Err(e) = res {
            tracing::warn!(error = %e, immediate, "AssignmentHandler.on_drain raised");
        }
    }

    async fn evaluate_offer(&self, offer: &Offer) -> OfferOutcome {
        let offer_clone = offer.clone();
        let res = call_handler_method(
            &self.py_obj,
            &self.locals,
            "evaluate_offer",
            move |py| -> PyResult<Vec<Py<PyAny>>> {
                let dict = offer_to_pydict(py, &offer_clone)?;
                Ok(vec![dict.into_any().unbind()])
            },
        )
        .await;
        match res {
            Ok(value) => {
                let decision: PyResult<String> = Python::attach(|py| value.extract(py));
                match decision {
                    Ok(s) if s.eq_ignore_ascii_case("decline") => OfferOutcome::Decline {
                        reason: DeclineReason::Other("python handler declined".to_owned()),
                    },
                    _ => OfferOutcome::Claim,
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "AssignmentHandler.evaluate_offer raised; defaulting to Claim",
                );
                OfferOutcome::Claim
            }
        }
    }
}

// ===========================================================================
// Conversions: Assignment / Offer → Python dicts
// ===========================================================================

fn assignment_to_pydict<'py>(
    py: Python<'py>,
    assignment: &Assignment,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("run_id", assignment.run_id.to_string())?;
    dict.set_item(
        "parent_run_id",
        assignment.parent_run_id.map(|u| u.to_string()),
    )?;
    dict.set_item("workflow_name", assignment.workflow_name.clone())?;
    dict.set_item("workflow_version", assignment.workflow_version)?;
    let input_value = assignment.input_value().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "assignment.input_json was not valid JSON: {e}"
        ))
    })?;
    dict.set_item("input", json_to_py(py, &input_value)?)?;
    dict.set_item("deadline_ms", assignment.deadline_ms)?;
    dict.set_item("attempt", assignment.attempt)?;
    if let Some(hint) = &assignment.resource_hint {
        let hint_dict = PyDict::new(py);
        hint_dict.set_item("vram_mb", hint.vram_mb)?;
        hint_dict.set_item("cpu_cores", hint.cpu_cores)?;
        hint_dict.set_item("expected_seconds", hint.expected_seconds)?;
        dict.set_item("resource_hint", hint_dict)?;
    } else {
        dict.set_item("resource_hint", py.None())?;
    }
    Ok(dict)
}

fn offer_to_pydict<'py>(py: Python<'py>, offer: &Offer) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("assignment", assignment_to_pydict(py, &offer.assignment)?)?;
    Ok(dict)
}

// ===========================================================================
// PyControlPlaneWorker
// ===========================================================================

/// Worker handle that drives a bidi control-plane session.
///
/// `connect` validates the configured endpoint without opening a
/// connection (the first network attempt happens inside `run` so the
/// retry policy covers the initial connect uniformly with reconnects).
///
/// Calling `run()` consumes the underlying Rust `Worker` by value (the
/// upstream API requires `self` by move so it can spawn the bidi
/// session task and assignment workers tied to the same lifetime).
/// `shutdown()` is therefore best-effort: it works while `run()` has
/// not yet been called. Once `run()` is in flight, the canonical way
/// to terminate the worker is to cancel the awaiting Python task — the
/// inner select loop on each session honors the cancellation cleanly.
#[gen_stub_pyclass]
#[pyclass(name = "ControlPlaneWorker")]
pub struct PyControlPlaneWorker {
    /// `Worker::run` consumes the value by move, so the slot empties
    /// on the first successful `run()` call. Wrapped in `Mutex<Option>`
    /// so `run()` can stay `&self` and `shutdown()` can still inspect
    /// the slot to decide whether shutdown is meaningful.
    inner: Arc<Mutex<Option<Worker>>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyControlPlaneWorker {
    /// Validate the config and prepare a worker handle. Does NOT open
    /// a network connection — the first connect happens inside `run`.
    ///
    /// Raises:
    ///     ControlPlaneError: If the configured endpoint URI cannot be
    ///         parsed.
    #[staticmethod]
    fn connect(config: PyControlPlaneWorkerConfig) -> PyResult<Self> {
        let worker = Worker::connect(config.inner).map_err(cp_err)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(worker))),
        })
    }

    /// Drive the worker until shutdown, drain, or the retry policy is
    /// exhausted. Returns a coroutine that resolves when the session
    /// ends.
    ///
    /// Args:
    ///     handler: A subclass of `AssignmentHandler`. The current
    ///         asyncio task locals are captured here so the handler's
    ///         coroutines run on the caller's event loop.
    ///
    /// Raises:
    ///     ControlPlaneError: When the retry policy is exhausted or the
    ///         initial endpoint is fundamentally broken; also raised
    ///         if `run()` is invoked more than once on the same
    ///         instance.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, None]",
        imports = ("typing",)
    ))]
    fn run<'py>(&self, py: Python<'py>, handler: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // Capture the asyncio task locals so the adapter can dispatch
        // Python coroutines on the caller's event loop.
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let worker = {
                let mut guard = inner.lock().await;
                guard.take().ok_or_else(|| {
                    ControlPlaneException::new_err(
                        "ControlPlaneWorker.run already called; create a fresh worker",
                    )
                })?
            };
            let adapter = PyAssignmentHandlerAdapter::new(handler, locals);
            worker.run(adapter).await.map_err(cp_err)
        })
    }

    /// Signal the worker to stop. Idempotent.
    ///
    /// This is best-effort: it has effect only while `run()` has not
    /// yet been called. Once `run()` has consumed the underlying
    /// worker by value, the way to stop the worker is to cancel the
    /// asyncio task awaiting the `run()` coroutine — the bidi session
    /// loop honors cancellation at every iteration.
    fn shutdown(&self) {
        if let Ok(guard) = self.inner.try_lock()
            && let Some(worker) = guard.as_ref()
        {
            worker.shutdown();
        }
    }

    fn __repr__(&self) -> String {
        "ControlPlaneWorker(...)".to_owned()
    }
}
