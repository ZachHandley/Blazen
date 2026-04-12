use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::types::RequestTiming;

// ---------------------------------------------------------------------------
// RequestTiming
// ---------------------------------------------------------------------------

/// Breakdown of request timing (queue, execution, total) in milliseconds.
#[gen_stub_pyclass]
#[pyclass(name = "RequestTiming", frozen)]
pub struct PyRequestTiming {
    pub(crate) inner: RequestTiming,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRequestTiming {
    /// Time spent in the provider queue, in milliseconds.
    #[getter]
    fn queue_ms(&self) -> Option<u64> {
        self.inner.queue_ms
    }

    /// Time spent executing the request, in milliseconds.
    #[getter]
    fn execution_ms(&self) -> Option<u64> {
        self.inner.execution_ms
    }

    /// Total wall-clock time for the request, in milliseconds.
    #[getter]
    fn total_ms(&self) -> Option<u64> {
        self.inner.total_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestTiming(queue_ms={:?}, execution_ms={:?}, total_ms={:?})",
            self.inner.queue_ms, self.inner.execution_ms, self.inner.total_ms
        )
    }
}
