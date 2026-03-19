//! Python wrappers for token usage and request timing types.

use pyo3::prelude::*;

use blazen_llm::{RequestTiming, TokenUsage};

// ---------------------------------------------------------------------------
// PyTokenUsage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion.
#[pyclass(name = "TokenUsage", from_py_object)]
#[derive(Clone)]
pub struct PyTokenUsage {
    pub(crate) inner: TokenUsage,
}

#[pymethods]
impl PyTokenUsage {
    #[getter]
    fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    #[getter]
    fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    #[getter]
    fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "prompt_tokens" => Ok(self
                .inner
                .prompt_tokens
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "completion_tokens" => Ok(self
                .inner
                .completion_tokens
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "total_tokens" => Ok(self
                .inner
                .total_tokens
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenUsage(prompt_tokens={}, completion_tokens={}, total_tokens={})",
            self.inner.prompt_tokens, self.inner.completion_tokens, self.inner.total_tokens
        )
    }
}

// ---------------------------------------------------------------------------
// PyRequestTiming
// ---------------------------------------------------------------------------

/// Request timing metadata.
#[pyclass(name = "RequestTiming", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyRequestTiming {
    pub(crate) inner: RequestTiming,
}

#[pymethods]
impl PyRequestTiming {
    #[getter]
    fn queue_ms(&self) -> Option<u64> {
        self.inner.queue_ms
    }

    #[getter]
    fn execution_ms(&self) -> Option<u64> {
        self.inner.execution_ms
    }

    #[getter]
    fn total_ms(&self) -> Option<u64> {
        self.inner.total_ms
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "queue_ms" => Ok(self.inner.queue_ms.into_pyobject(py)?.into_any().unbind()),
            "execution_ms" => Ok(self
                .inner
                .execution_ms
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "total_ms" => Ok(self.inner.total_ms.into_pyobject(py)?.into_any().unbind()),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestTiming(queue_ms={:?}, execution_ms={:?}, total_ms={:?})",
            self.inner.queue_ms, self.inner.execution_ms, self.inner.total_ms
        )
    }
}
