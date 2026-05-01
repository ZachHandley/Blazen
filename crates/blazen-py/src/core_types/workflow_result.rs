//! Python wrapper for [`blazen_core::handler::WorkflowResult`].
//!
//! ``WorkflowResult`` bundles the terminal event from a completed
//! workflow run with the live session-ref registry that backs any
//! ``__blazen_session_ref__`` markers carried on that event's payload.
//! Owning the registry alongside the event keeps such markers
//! resolvable for as long as the result is held.
//!
//! The current high-level [`PyWorkflowHandler.result`] coroutine eagerly
//! unwraps the event into a ``PyEvent`` for ergonomics. This module
//! provides the typed [`PyWorkflowResult`] container for callers that
//! drive the runtime themselves (custom transports, bridge layers) and
//! need the (event, registry) pair as a single value.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::session_ref::SessionRefRegistry;
use blazen_llm::types::TokenUsage;

use crate::types::PyTokenUsage;
use crate::workflow::event::PyEvent;

use super::session_ref::PySessionRefRegistry;

// ---------------------------------------------------------------------------
// PyWorkflowResult
// ---------------------------------------------------------------------------

/// Final result of a completed workflow run.
///
/// Mirrors [`blazen_core::WorkflowResult`]: a terminal
/// :class:`Event` plus the :class:`SessionRefRegistry` that owns the
/// in-process objects any session-ref markers on that event refer to.
///
/// Construct one via the static factory or by reading the components
/// off an existing :class:`WorkflowHandler` result.
#[gen_stub_pyclass]
#[pyclass(name = "WorkflowResult", frozen)]
pub struct PyWorkflowResult {
    event: Py<PyEvent>,
    session_refs: Arc<SessionRefRegistry>,
    usage_total: TokenUsage,
    cost_total_usd: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWorkflowResult {
    /// Build a [`WorkflowResult`] from its parts.
    ///
    /// Args:
    ///     event: The terminal :class:`Event` produced by the workflow
    ///         (typically a ``StopEvent``).
    ///     session_refs: Optional :class:`SessionRefRegistry` that owns
    ///         the in-process objects referenced by any session-ref
    ///         markers on the event's payload. Pass ``None`` when the
    ///         event does not carry session refs.
    #[new]
    #[pyo3(signature = (event, session_refs=None, usage_total=None, cost_total_usd=0.0))]
    fn new(
        event: Py<PyEvent>,
        session_refs: Option<&PySessionRefRegistry>,
        usage_total: Option<PyTokenUsage>,
        cost_total_usd: f64,
    ) -> Self {
        let registry =
            session_refs.map_or_else(|| Arc::new(SessionRefRegistry::new()), |r| r.inner.clone());
        Self {
            event,
            session_refs: registry,
            usage_total: usage_total.map(|u| u.inner).unwrap_or_default(),
            cost_total_usd,
        }
    }

    /// The terminal event produced by the workflow.
    #[getter]
    fn event(&self, py: Python<'_>) -> Py<PyEvent> {
        self.event.clone_ref(py)
    }

    /// The session-ref registry owning the in-process objects that any
    /// session-ref markers on the event refer to.
    #[getter]
    fn session_refs(&self) -> PySessionRefRegistry {
        PySessionRefRegistry {
            inner: Arc::clone(&self.session_refs),
        }
    }

    /// Aggregated token usage across the workflow run.
    #[getter]
    fn usage_total(&self) -> PyTokenUsage {
        PyTokenUsage::from(&self.usage_total)
    }

    /// Aggregated USD cost across the workflow run.
    #[getter]
    fn cost_total_usd(&self) -> f64 {
        self.cost_total_usd
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkflowResult(event=..., usage_total={:?}, cost_total_usd={})",
            self.usage_total, self.cost_total_usd,
        )
    }
}

impl PyWorkflowResult {
    /// Construct a [`PyWorkflowResult`] from its component parts.
    ///
    /// Used by adapters that build a result from a Rust
    /// [`blazen_core::WorkflowResult`] -- they convert the boxed
    /// `dyn AnyEvent` into a [`PyEvent`] via
    /// [`crate::workflow::event::any_event_to_py_event`] and then call
    /// this constructor to attach the registry.
    pub fn new_from_parts(event: Py<PyEvent>, session_refs: Arc<SessionRefRegistry>) -> Self {
        Self {
            event,
            session_refs,
            usage_total: TokenUsage::default(),
            cost_total_usd: 0.0,
        }
    }

    /// Build a [`PyWorkflowResult`] including aggregated usage / cost.
    ///
    /// Used by adapters that have already extracted `usage_total` and
    /// `cost_total_usd` from a Rust [`blazen_core::WorkflowResult`].
    pub fn new_with_usage(
        event: Py<PyEvent>,
        session_refs: Arc<SessionRefRegistry>,
        usage_total: TokenUsage,
        cost_total_usd: f64,
    ) -> Self {
        Self {
            event,
            session_refs,
            usage_total,
            cost_total_usd,
        }
    }
}
