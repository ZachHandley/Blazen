//! Python wrappers for [`blazen_core::step::StepOutput`] and
//! [`blazen_core::step::StepRegistration`].
//!
//! These types live in the workflow runtime and are normally consumed
//! purely on the Rust side (the public Python step decorator builds
//! them implicitly behind the scenes). The bindings here exist so
//! callers writing custom orchestrators in Python can introspect the
//! step graph -- e.g. read the registered step names, accepted/emitted
//! event types, and concurrency limits -- without falling out into
//! Rust-only territory.
//!
//! `PyStepOutput` is exposed as a typed enum-style class with three
//! constructors mirroring the Rust variants: `StepOutput.single(event)`,
//! `StepOutput.multiple([event, ...])`, `StepOutput.none()`. Steps
//! authored via the standard Python decorator never need to construct
//! one explicitly -- they just `return event` -- but custom transports
//! that drive the Rust event loop manually do.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::step::StepRegistration;

use crate::workflow::event::PyEvent;

// ---------------------------------------------------------------------------
// PyStepOutput
// ---------------------------------------------------------------------------

/// One of three terminal results from a step body.
///
/// Mirrors the Rust [`blazen_core::step::StepOutput`] enum:
///
/// * ``Single`` -- one outbound event, routed normally.
/// * ``Multiple`` -- a fan-out of events. Every entry is dispatched.
/// * ``None`` -- side-effect only, the workflow loop produces no
///   downstream events from this step.
///
/// Construct via the static helpers (``StepOutput.single(event)``,
/// ``StepOutput.multiple([...])``, ``StepOutput.none()``); the inner
/// ``kind`` and ``events`` getters let callers introspect after the
/// fact.
#[gen_stub_pyclass]
#[pyclass(name = "StepOutput", frozen)]
pub struct PyStepOutput {
    /// Discriminator -- ``"single"``, ``"multiple"``, or ``"none"``.
    kind: StepOutputKind,
    /// Carried events. Empty for ``"none"``, length 1 for ``"single"``,
    /// length N for ``"multiple"``.
    events: Vec<Py<PyEvent>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepOutputKind {
    Single,
    Multiple,
    None_,
}

impl StepOutputKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Multiple => "multiple",
            Self::None_ => "none",
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStepOutput {
    /// Construct a single-event output.
    #[staticmethod]
    fn single(event: Py<PyEvent>) -> Self {
        Self {
            kind: StepOutputKind::Single,
            events: vec![event],
        }
    }

    /// Construct a fan-out output from a list of events.
    #[staticmethod]
    fn multiple(events: Vec<Py<PyEvent>>) -> Self {
        Self {
            kind: StepOutputKind::Multiple,
            events,
        }
    }

    /// Construct a no-output result (side-effect only).
    #[staticmethod]
    fn none() -> Self {
        Self {
            kind: StepOutputKind::None_,
            events: Vec::new(),
        }
    }

    /// Discriminator: ``"single"``, ``"multiple"``, or ``"none"``.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Whether this output has no events (the ``None`` variant).
    fn is_none(&self) -> bool {
        self.kind == StepOutputKind::None_
    }

    /// Return all events carried by this output.
    ///
    /// Empty for ``"none"``; length 1 for ``"single"``; length N for
    /// ``"multiple"``.
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for ev in &self.events {
            list.append(ev.clone_ref(py))?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "StepOutput(kind={:?}, count={})",
            self.kind.as_str(),
            self.events.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyStepRegistration
// ---------------------------------------------------------------------------

/// Read-only metadata for a step registered on a [`Workflow`].
///
/// Wraps [`blazen_core::step::StepRegistration`]. Returned by the
/// workflow introspection helpers so callers can list the steps that
/// will run, their accepted/emitted event types, and the configured
/// max-concurrency.
///
/// The handler closure itself is not exposed -- it is a Rust closure
/// type that cannot cross the FFI boundary -- but every other field is
/// available as a getter.
#[gen_stub_pyclass]
#[pyclass(name = "StepRegistration", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStepRegistration {
    pub(crate) inner: StepRegistration,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStepRegistration {
    /// Human-readable name for this step.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Event type identifiers this step accepts (matches
    /// ``Event.event_type``).
    #[getter]
    fn accepts(&self) -> Vec<String> {
        self.inner.accepts.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Event type identifiers this step may emit (informational only;
    /// the runtime does not enforce this).
    #[getter]
    fn emits(&self) -> Vec<String> {
        self.inner.emits.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Maximum number of concurrent invocations of this step. ``0``
    /// means unlimited.
    #[getter]
    fn max_concurrency(&self) -> usize {
        self.inner.max_concurrency
    }

    fn __repr__(&self) -> String {
        format!(
            "StepRegistration(name={:?}, accepts={:?}, emits={:?}, max_concurrency={})",
            self.inner.name, self.inner.accepts, self.inner.emits, self.inner.max_concurrency,
        )
    }
}

impl From<StepRegistration> for PyStepRegistration {
    fn from(inner: StepRegistration) -> Self {
        Self { inner }
    }
}
