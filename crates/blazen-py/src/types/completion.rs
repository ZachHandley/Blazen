//! Python wrapper for completion response types.

use pyo3::prelude::*;

use blazen_llm::CompletionResponse;

use crate::types::tool::PyToolCall;
use crate::types::usage::{PyRequestTiming, PyTokenUsage};

// ---------------------------------------------------------------------------
// PyCompletionResponse
// ---------------------------------------------------------------------------

/// The result of a chat completion.
///
/// Supports both attribute access and dict-style access for backwards
/// compatibility:
///     >>> response.content        # attribute
///     >>> response["content"]     # dict-style
#[pyclass(name = "CompletionResponse", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionResponse {
    pub(crate) inner: CompletionResponse,
}

#[pymethods]
impl PyCompletionResponse {
    #[getter]
    fn content(&self) -> Option<&str> {
        self.inner.content.as_deref()
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    #[getter]
    fn tool_calls(&self) -> Vec<PyToolCall> {
        self.inner
            .tool_calls
            .iter()
            .map(|tc| PyToolCall { inner: tc.clone() })
            .collect()
    }

    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.clone().map(|u| PyTokenUsage { inner: u })
    }

    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    #[getter]
    fn timing(&self) -> Option<PyRequestTiming> {
        self.inner
            .timing
            .clone()
            .map(|t| PyRequestTiming { inner: t })
    }

    #[getter]
    fn images(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let val = serde_json::to_value(&self.inner.images).unwrap_or_default();
        crate::convert::json_to_py(py, &val)
    }

    #[getter]
    fn audio(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let val = serde_json::to_value(&self.inner.audio).unwrap_or_default();
        crate::convert::json_to_py(py, &val)
    }

    #[getter]
    fn videos(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let val = serde_json::to_value(&self.inner.videos).unwrap_or_default();
        crate::convert::json_to_py(py, &val)
    }

    #[getter]
    fn metadata_extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "content" => match &self.inner.content {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "model" => Ok(self
                .inner
                .model
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "finish_reason" => match &self.inner.finish_reason {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "tool_calls" => {
                // Return as list of dicts for backwards compat
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
            "usage" => {
                if let Some(usage) = &self.inner.usage {
                    let val = serde_json::json!({
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    });
                    crate::convert::json_to_py(py, &val)
                } else {
                    Ok(py.None())
                }
            }
            "cost" => match self.inner.cost {
                Some(c) => Ok(c.into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "timing" => match &self.inner.timing {
                Some(t) => {
                    let val = serde_json::json!({
                        "queue_ms": t.queue_ms,
                        "execution_ms": t.execution_ms,
                        "total_ms": t.total_ms,
                    });
                    crate::convert::json_to_py(py, &val)
                }
                None => Ok(py.None()),
            },
            "images" => {
                let val = serde_json::to_value(&self.inner.images).unwrap_or_default();
                crate::convert::json_to_py(py, &val)
            }
            "audio" => {
                let val = serde_json::to_value(&self.inner.audio).unwrap_or_default();
                crate::convert::json_to_py(py, &val)
            }
            "videos" => {
                let val = serde_json::to_value(&self.inner.videos).unwrap_or_default();
                crate::convert::json_to_py(py, &val)
            }
            "metadata" => crate::convert::json_to_py(py, &self.inner.metadata),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn keys(&self) -> Vec<&str> {
        vec![
            "content",
            "model",
            "finish_reason",
            "tool_calls",
            "usage",
            "cost",
            "timing",
            "images",
            "audio",
            "videos",
            "metadata",
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletionResponse(model='{}', content='{}')",
            self.inner.model,
            self.inner.content.as_deref().unwrap_or(""),
        )
    }
}
