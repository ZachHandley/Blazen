//! Python wrapper for streaming completion chunks.

use pyo3::prelude::*;

use blazen_llm::StreamChunk;

use crate::types::tool::PyToolCall;

// ---------------------------------------------------------------------------
// PyStreamChunk
// ---------------------------------------------------------------------------

/// A single chunk from a streaming completion response.
///
/// Attributes:
///     delta: Incremental text content, or None.
///     finish_reason: Present only in the final chunk.
///     tool_calls: Tool invocations completed in this chunk.
///
/// Example:
///     >>> async def on_chunk(chunk):
///     ...     if chunk.delta:
///     ...         print(chunk.delta, end="")
///     ...     if chunk.finish_reason:
///     ...         print(f"\n[done: {chunk.finish_reason}]")
#[pyclass(name = "StreamChunk", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStreamChunk {
    pub(crate) inner: StreamChunk,
}

#[pymethods]
impl PyStreamChunk {
    /// Incremental text content, if any.
    #[getter]
    fn delta(&self) -> Option<&str> {
        self.inner.delta.as_deref()
    }

    /// Present in the final chunk to indicate why generation stopped.
    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    /// Tool invocations completed in this chunk.
    #[getter]
    fn tool_calls(&self) -> Vec<PyToolCall> {
        self.inner
            .tool_calls
            .iter()
            .map(|tc| PyToolCall { inner: tc.clone() })
            .collect()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "delta" => match &self.inner.delta {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "finish_reason" => match &self.inner.finish_reason {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "tool_calls" => {
                let tool_calls: Vec<serde_json::Value> = self
                    .inner
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        })
                    })
                    .collect();
                crate::convert::json_to_py(py, &serde_json::Value::Array(tool_calls))
            }
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamChunk(delta={:?}, finish_reason={:?}, tool_calls={})",
            self.inner.delta,
            self.inner.finish_reason,
            self.inner.tool_calls.len()
        )
    }
}
