//! Python wrapper for tool call types.
//!
//! Exposes:
//! - [`PyToolOutput`] — the two-channel return value from a tool
//!   (`data` for callers, optional `llm_override` for what the LLM sees).
//! - [`PyLlmPayload`] — the override payload, constructed via classmethod
//!   factories (`text`, `json`, `provider_raw`).
//!
//! Re-exports the upstream [`ToolCall`] for callers that need the raw
//! struct (e.g., introspecting an assistant message).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::types::{LlmPayload, ProviderId, ToolOutput};

pub use blazen_llm::types::ToolCall;

// ---------------------------------------------------------------------------
// PyLlmPayload
// ---------------------------------------------------------------------------

/// Provider-aware override for what the LLM sees as a tool result.
///
/// Construct via the classmethod factories — there is no public `__init__`:
///
/// - ``LlmPayload.text("hello")`` — plain text, works on every provider.
/// - ``LlmPayload.json({"k": "v"})`` — structured JSON; Anthropic and
///   Gemini consume natively, OpenAI/Responses stringify on the wire.
/// - ``LlmPayload.provider_raw(provider="anthropic", value={...})`` —
///   provider-specific escape hatch. The named provider receives ``value``
///   verbatim; every other provider falls back to the default conversion
///   from ``ToolOutput.data``.
///
/// Inspect the variant via ``payload.kind`` (``"text"``, ``"json"``, or
/// ``"provider_raw"``). The ``text``, ``value``, and ``provider`` getters
/// return ``None`` for variants that don't carry the corresponding field.
#[gen_stub_pyclass]
#[pyclass(name = "LlmPayload", from_py_object)]
#[derive(Clone)]
pub struct PyLlmPayload {
    pub(crate) inner: LlmPayload,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlmPayload {
    /// Create a plain-text payload.
    #[staticmethod]
    fn text(text: String) -> Self {
        Self {
            inner: LlmPayload::Text { text },
        }
    }

    /// Create a structured-JSON payload from any JSON-serializable Python value.
    #[staticmethod]
    fn json(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Self> {
        let value = crate::convert::py_to_json(py, &value)?;
        Ok(Self {
            inner: LlmPayload::Json { value },
        })
    }

    /// Create a provider-specific payload.
    ///
    /// `provider` must be one of: `"openai"`, `"openai_compat"`, `"azure"`,
    /// `"anthropic"`, `"gemini"`, `"responses"`, `"fal"`.
    #[staticmethod]
    #[pyo3(signature = (*, provider, value))]
    fn provider_raw(py: Python<'_>, provider: &str, value: Bound<'_, PyAny>) -> PyResult<Self> {
        let provider_id = match provider {
            "openai" => ProviderId::OpenAi,
            "openai_compat" => ProviderId::OpenAiCompat,
            "azure" => ProviderId::Azure,
            "anthropic" => ProviderId::Anthropic,
            "gemini" => ProviderId::Gemini,
            "responses" => ProviderId::Responses,
            "fal" => ProviderId::Fal,
            other => {
                return Err(crate::error::BlazenPyError::InvalidArgument(format!(
                    "unknown provider: '{other}' \
                     (expected one of: openai, openai_compat, azure, anthropic, \
                     gemini, responses, fal)"
                ))
                .into());
            }
        };
        let value = crate::convert::py_to_json(py, &value)?;
        Ok(Self {
            inner: LlmPayload::ProviderRaw {
                provider: provider_id,
                value,
            },
        })
    }

    /// The variant tag: `"text"`, `"json"`, `"parts"`, or `"provider_raw"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            LlmPayload::Text { .. } => "text",
            LlmPayload::Json { .. } => "json",
            LlmPayload::Parts { .. } => "parts",
            LlmPayload::ProviderRaw { .. } => "provider_raw",
        }
    }

    /// The text body for `Text` payloads. `None` for other variants.
    #[getter]
    fn text_value(&self) -> Option<String> {
        if let LlmPayload::Text { text } = &self.inner {
            Some(text.clone())
        } else {
            None
        }
    }

    /// The structured value for `Json` and `ProviderRaw` payloads.
    /// `None` for `Text` and `Parts`.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            LlmPayload::Json { value } | LlmPayload::ProviderRaw { value, .. } => {
                Ok(Some(crate::convert::json_to_py(py, value)?))
            }
            _ => Ok(None),
        }
    }

    /// The provider name for `ProviderRaw` payloads. `None` otherwise.
    #[getter]
    fn provider(&self) -> Option<&'static str> {
        if let LlmPayload::ProviderRaw { provider, .. } = &self.inner {
            Some(match provider {
                ProviderId::OpenAi => "openai",
                ProviderId::OpenAiCompat => "openai_compat",
                ProviderId::Azure => "azure",
                ProviderId::Anthropic => "anthropic",
                ProviderId::Gemini => "gemini",
                ProviderId::Responses => "responses",
                ProviderId::Fal => "fal",
            })
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            LlmPayload::Text { text } => {
                let preview: String = text.chars().take(40).collect();
                format!("LlmPayload.text({preview:?})")
            }
            LlmPayload::Json { .. } => "LlmPayload.json(...)".to_owned(),
            LlmPayload::Parts { parts } => format!("LlmPayload.parts(<{} parts>)", parts.len()),
            LlmPayload::ProviderRaw { provider, .. } => {
                let name = match provider {
                    ProviderId::OpenAi => "openai",
                    ProviderId::OpenAiCompat => "openai_compat",
                    ProviderId::Azure => "azure",
                    ProviderId::Anthropic => "anthropic",
                    ProviderId::Gemini => "gemini",
                    ProviderId::Responses => "responses",
                    ProviderId::Fal => "fal",
                };
                format!("LlmPayload.provider_raw(provider='{name}', ...)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PyToolOutput
// ---------------------------------------------------------------------------

/// Two-channel tool result: structured `data` for callers, optional
/// `llm_override` for what the LLM sees on the next turn.
///
/// Returning a bare `dict`, `list`, or `str` from a tool handler is
/// equivalent to wrapping it in `ToolOutput(data=value)` — the `llm_override`
/// is left unset and each provider applies its default conversion.
///
/// Construct an explicit override when the structured payload should differ
/// from what the LLM sees::
///
///     return ToolOutput(
///         data={"items": [...], "total": 1234},
///         llm_override=LlmPayload.text("Found 1234 items."),
///     )
#[gen_stub_pyclass]
#[pyclass(name = "ToolOutput", from_py_object)]
#[derive(Clone)]
pub struct PyToolOutput {
    pub(crate) inner: ToolOutput<serde_json::Value>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyToolOutput {
    /// Create a tool output.
    ///
    /// Args:
    ///     data: Any JSON-serializable Python value (the structured result
    ///         visible to callers).
    ///     llm_override: Optional [`LlmPayload`] to override what the LLM
    ///         sees on the next turn. Defaults to ``None`` (provider applies
    ///         its default conversion from ``data``).
    #[new]
    #[pyo3(signature = (data, llm_override = None))]
    fn new(
        py: Python<'_>,
        data: Bound<'_, PyAny>,
        llm_override: Option<PyLlmPayload>,
    ) -> PyResult<Self> {
        let data = crate::convert::py_to_json(py, &data)?;
        Ok(Self {
            inner: ToolOutput {
                data,
                llm_override: llm_override.map(|p| p.inner),
            },
        })
    }

    /// The structured payload visible to callers.
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.data)
    }

    /// The optional LLM-visible override payload.
    #[getter]
    fn llm_override(&self) -> Option<PyLlmPayload> {
        self.inner
            .llm_override
            .as_ref()
            .map(|p| PyLlmPayload { inner: p.clone() })
    }

    fn __repr__(&self) -> String {
        if self.inner.llm_override.is_some() {
            "ToolOutput(data=..., llm_override=...)".to_owned()
        } else {
            "ToolOutput(data=...)".to_owned()
        }
    }
}

impl PyToolOutput {
    /// Consume the wrapper and return the underlying Rust value.
    pub(crate) fn into_inner(self) -> ToolOutput<serde_json::Value> {
        self.inner
    }
}
