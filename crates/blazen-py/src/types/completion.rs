//! Python wrapper for completion response types.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::CompletionResponse;

use super::{
    PyArtifact, PyCitation, PyGeneratedAudio, PyGeneratedImage, PyGeneratedVideo, PyReasoningTrace,
    PyRequestTiming, PyTokenUsage, PyToolCall,
};

// ---------------------------------------------------------------------------
// PyCompletionResponse
// ---------------------------------------------------------------------------

/// The result of a chat completion.
///
/// Supports both attribute access and dict-style access for backwards
/// compatibility:
///     >>> response.content        # attribute
///     >>> response["content"]     # dict-style
#[gen_stub_pyclass]
#[pyclass(name = "CompletionResponse", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionResponse {
    pub(crate) inner: CompletionResponse,
}

#[gen_stub_pymethods]
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
        self.inner.tool_calls.iter().map(PyToolCall::from).collect()
    }

    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.as_ref().map(PyTokenUsage::from)
    }

    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    #[getter]
    fn timing(&self) -> Option<PyRequestTiming> {
        self.inner
            .timing
            .as_ref()
            .map(|t| PyRequestTiming { inner: t.clone() })
    }

    #[getter]
    fn images(&self) -> Vec<PyGeneratedImage> {
        self.inner
            .images
            .iter()
            .map(|i| PyGeneratedImage { inner: i.clone() })
            .collect()
    }

    #[getter]
    fn audio(&self) -> Vec<PyGeneratedAudio> {
        self.inner
            .audio
            .iter()
            .map(|a| PyGeneratedAudio { inner: a.clone() })
            .collect()
    }

    #[getter]
    fn videos(&self) -> Vec<PyGeneratedVideo> {
        self.inner
            .videos
            .iter()
            .map(|v| PyGeneratedVideo { inner: v.clone() })
            .collect()
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata_extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    /// Reasoning trace from models that expose one (Anthropic extended thinking,
    /// DeepSeek R1, OpenAI o-series, xAI Grok, Gemini thoughts).
    #[getter]
    fn reasoning(&self) -> Option<PyReasoningTrace> {
        self.inner.reasoning.as_ref().map(PyReasoningTrace::from)
    }

    /// Web/document citations backing the model's statement (Perplexity,
    /// Gemini grounding, Anthropic web search).
    #[getter]
    fn citations(&self) -> Vec<PyCitation> {
        self.inner.citations.iter().map(PyCitation::from).collect()
    }

    /// Typed inline artifacts extracted from the response (SVG, code blocks,
    /// markdown, mermaid, html, latex, json, custom).
    #[getter]
    fn artifacts(&self) -> Vec<PyArtifact> {
        self.inner.artifacts.iter().map(PyArtifact::from).collect()
    }

    /// Lazily map the raw provider finish-reason string into a normalized
    /// [`FinishReason`].
    fn finish_reason_normalized(&self) -> Option<crate::types::PyFinishReason> {
        self.inner
            .finish_reason_normalized()
            .map(crate::types::PyFinishReason::from)
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
                Some(t) => Ok(PyRequestTiming { inner: t.clone() }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()),
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
            "reasoning" => match &self.inner.reasoning {
                Some(r) => {
                    let val = serde_json::to_value(r).unwrap_or_default();
                    crate::convert::json_to_py(py, &val)
                }
                None => Ok(py.None()),
            },
            "citations" => {
                let val = serde_json::to_value(&self.inner.citations).unwrap_or_default();
                crate::convert::json_to_py(py, &val)
            }
            "artifacts" => {
                let val = serde_json::to_value(&self.inner.artifacts).unwrap_or_default();
                crate::convert::json_to_py(py, &val)
            }
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
            "reasoning",
            "citations",
            "artifacts",
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
