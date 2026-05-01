//! Python wrapper for [`blazen_pipeline::PipelineState`].
//!
//! Read-only view passed to Python `input_mapper` and `condition` callbacks
//! defined on `Stage`. The wrapper holds a raw pointer to the borrowed Rust
//! `PipelineState` and is only valid for the duration of the synchronous
//! callback invocation. The Python side cannot escape the pointer because
//! the wrapper is consumed by the callback and not stored anywhere by the
//! Rust closure that bridges the call.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_pipeline::PipelineState;

use crate::convert::json_to_py;
use crate::events::PyUsageEvent;
use crate::types::PyTokenUsage;

/// Read-only view of pipeline state.
///
/// Passed to user-supplied `input_mapper(state)` and `condition(state)`
/// callbacks. The view exposes the original pipeline input, the shared
/// key/value store, and per-stage results.
///
/// Note: the view is only valid during the callback call. Storing it
/// outside of the callback (e.g. on `self`) is not supported.
#[gen_stub_pyclass]
#[pyclass(name = "PipelineState", frozen)]
pub struct PyPipelineState {
    /// Raw pointer to a borrowed Rust `PipelineState`. The pointer is valid
    /// for the lifetime of the closure that constructed the wrapper. Wrapped
    /// in an `AtomicUsize` only to satisfy `Sync`; semantically it is
    /// constant-after-construction.
    state_ptr: usize,
}

impl PyPipelineState {
    /// Construct a wrapper around the borrowed pipeline state.
    ///
    /// # Safety
    /// The caller must ensure the returned `PyPipelineState` is dropped
    /// before the borrowed `&PipelineState` goes out of scope. In practice
    /// this means: build the wrapper, hand it to a Python call, drop it
    /// when the call returns. Do not let Python side store it.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn from_ref(state: &PipelineState) -> Self {
        Self {
            state_ptr: std::ptr::from_ref(state) as usize,
        }
    }

    #[allow(unsafe_code)]
    fn state(&self) -> &PipelineState {
        // SAFETY: caller of `from_ref` upholds the lifetime invariant.
        unsafe { &*(self.state_ptr as *const PipelineState) }
    }
}

// SAFETY: a raw pointer reading an immutably borrowed `PipelineState` is
// safe to share across threads for the lifetime guaranteed by `from_ref`.
#[allow(unsafe_code)]
unsafe impl Send for PyPipelineState {}
#[allow(unsafe_code)]
unsafe impl Sync for PyPipelineState {}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineState {
    /// The original pipeline input.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, self.state().input())
    }

    /// Get a value from the shared key/value store. Returns `None` if not set.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn get(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match self.state().get(key) {
            Some(v) => json_to_py(py, v),
            None => Ok(py.None()),
        }
    }

    /// Get the output of a specific completed stage by name.
    /// Returns `None` if the stage has not yet completed.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn stage_result(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        match self.state().stage_result(name) {
            Some(v) => json_to_py(py, v),
            None => Ok(py.None()),
        }
    }

    /// The output of the most recently completed stage, or the original
    /// pipeline input if no stages have completed yet.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn last_result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, self.state().last_result())
    }

    /// Token-usage rollup for the pipeline run so far.
    ///
    /// Sums every `UsageEvent` emitted on the workflow event streams
    /// across every stage that has completed at the moment of the read.
    #[getter]
    fn usage_total(&self) -> PyTokenUsage {
        PyTokenUsage::from(self.state().usage_total())
    }

    /// USD cost rollup for the pipeline run so far.
    ///
    /// Sums `UsageEvent::cost_usd` across every stage that has completed
    /// at the moment of the read.
    #[getter]
    fn cost_total_usd(&self) -> f64 {
        self.state().cost_total_usd()
    }

    fn __repr__(&self) -> String {
        "PipelineState(...)".to_owned()
    }
}

// Suppress unused-import warning — PyUsageEvent is referenced through
// the rollup machinery internally and from sibling modules.
const _: fn() = || {
    let _ = std::marker::PhantomData::<PyUsageEvent>;
};
