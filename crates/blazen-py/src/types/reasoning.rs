//! Python wrapper for `ReasoningTrace`.

use pyo3::prelude::*;

use blazen_llm::ReasoningTrace;

/// Chain-of-thought / extended-thinking trace from a model that exposes one.
///
/// Populated by Anthropic extended thinking, DeepSeek R1 reasoning_content,
/// OpenAI o-series, xAI Grok reasoning, and Gemini thoughts.
///
/// Example:
///     >>> response = await model.complete(messages)
///     >>> if response.reasoning is not None:
///     ...     print(response.reasoning.text)
#[pyclass(name = "ReasoningTrace", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyReasoningTrace {
    pub(crate) inner: ReasoningTrace,
}

#[pymethods]
impl PyReasoningTrace {
    /// Plain-text rendering of the reasoning content.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Provider-specific signature/redaction handle, if any (Anthropic).
    #[getter]
    fn signature(&self) -> Option<&str> {
        self.inner.signature.as_deref()
    }

    /// Whether the trace was redacted by the provider.
    #[getter]
    fn redacted(&self) -> bool {
        self.inner.redacted
    }

    /// Reasoning effort level if the provider exposes one
    /// (e.g. ``"low"``, ``"medium"``, ``"high"``).
    #[getter]
    fn effort(&self) -> Option<&str> {
        self.inner.effort.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReasoningTrace(text={:?}, signature={:?}, redacted={}, effort={:?})",
            self.inner.text, self.inner.signature, self.inner.redacted, self.inner.effort
        )
    }
}

impl From<ReasoningTrace> for PyReasoningTrace {
    fn from(inner: ReasoningTrace) -> Self {
        Self { inner }
    }
}

impl From<&ReasoningTrace> for PyReasoningTrace {
    fn from(inner: &ReasoningTrace) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
