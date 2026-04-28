//! Python wrapper for [`blazen_llm::CompletionRequest`].

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::CompletionRequest;
use blazen_llm::types::{ChatMessage, ToolDefinition};

use super::{PyChatMessage, PyToolDefinition};

/// A provider-agnostic chat-completion request.
///
/// Mirrors [`blazen_llm::CompletionRequest`]. Most code paths build the
/// request inline inside ``CompletionModel.complete(messages, options)``;
/// `CompletionRequest` is the typed alternative for callers that want to
/// inspect, serialize, or hand off the full request body explicitly.
///
/// Example:
///     >>> req = CompletionRequest(
///     ...     messages=[ChatMessage.user("hi")],
///     ...     temperature=0.0,
///     ...     max_tokens=100,
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "CompletionRequest", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionRequest {
    pub(crate) inner: CompletionRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompletionRequest {
    /// Construct a completion request.
    #[new]
    #[pyo3(signature = (
        *,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        response_format=None,
        model=None,
        modalities=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        messages: Vec<PyRef<'_, PyChatMessage>>,
        tools: Option<Vec<PyRef<'_, PyToolDefinition>>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        top_p: Option<f32>,
        response_format: Option<&Bound<'_, PyAny>>,
        model: Option<String>,
        modalities: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let rust_tools: Vec<ToolDefinition> = tools
            .map(|v| v.iter().map(|t| t.inner.clone()).collect())
            .unwrap_or_default();
        let rf_value = response_format
            .map(|rf| crate::convert::py_to_json(py, rf))
            .transpose()?;
        Ok(Self {
            inner: CompletionRequest {
                messages: rust_messages,
                tools: rust_tools,
                temperature,
                max_tokens,
                top_p,
                response_format: rf_value,
                model,
                modalities,
                image_config: None,
                audio_config: None,
            },
        })
    }

    /// The conversation messages.
    #[getter]
    fn messages(&self) -> Vec<PyChatMessage> {
        self.inner
            .messages
            .iter()
            .map(|m| PyChatMessage { inner: m.clone() })
            .collect()
    }

    /// Tool definitions available to the model.
    #[getter]
    fn tools(&self) -> Vec<PyToolDefinition> {
        self.inner
            .tools
            .iter()
            .map(|t| PyToolDefinition { inner: t.clone() })
            .collect()
    }

    /// Sampling temperature, if set.
    #[getter]
    fn temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Maximum output tokens, if set.
    #[getter]
    fn max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }

    /// Nucleus sampling parameter, if set.
    #[getter]
    fn top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Model override for this request, if set.
    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    /// Output modalities (`["text"]`, `["image", "text"]`, ...), if set.
    #[getter]
    fn modalities(&self) -> Option<Vec<String>> {
        self.inner.modalities.clone()
    }

    /// JSON schema / response-format hint, returned as a Python dict.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing",)))]
    fn response_format(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.response_format {
            Some(v) => Ok(Some(crate::convert::json_to_py(py, v)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletionRequest(messages={}, tools={}, temperature={:?}, max_tokens={:?})",
            self.inner.messages.len(),
            self.inner.tools.len(),
            self.inner.temperature,
            self.inner.max_tokens,
        )
    }
}

impl From<CompletionRequest> for PyCompletionRequest {
    fn from(inner: CompletionRequest) -> Self {
        Self { inner }
    }
}
