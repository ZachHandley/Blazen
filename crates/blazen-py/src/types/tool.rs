//! Python wrapper for tool call types.

use pyo3::prelude::*;

use blazen_llm::ToolCall;

// ---------------------------------------------------------------------------
// PyToolCall
// ---------------------------------------------------------------------------

/// A tool invocation requested by the model.
#[pyclass(name = "ToolCall", from_py_object)]
#[derive(Clone)]
pub struct PyToolCall {
    pub(crate) inner: ToolCall,
}

#[pymethods]
impl PyToolCall {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.arguments)
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "id" => Ok(self.inner.id.clone().into_pyobject(py)?.into_any().unbind()),
            "name" => Ok(self
                .inner
                .name
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "arguments" => crate::convert::json_to_py(py, &self.inner.arguments),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolCall(id='{}', name='{}')",
            self.inner.id, self.inner.name
        )
    }
}
