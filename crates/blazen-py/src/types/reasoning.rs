//! Python wrapper for `ReasoningTrace` (chain-of-thought from a model).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::types::ReasoningTrace;

/// Chain-of-thought / extended-thinking trace from models that expose one
/// (Anthropic extended thinking, DeepSeek R1, OpenAI o-series, xAI Grok,
/// Gemini thoughts).
///
/// Example:
///     >>> trace = response.reasoning
///     >>> if trace is not None:
///     ...     print(trace.text, trace.effort, trace.redacted)
#[gen_stub_pyclass]
#[pyclass(name = "ReasoningTrace", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyReasoningTrace {
    pub(crate) inner: ReasoningTrace,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReasoningTrace {
    /// Construct a reasoning trace explicitly.
    #[new]
    #[pyo3(signature = (*, text, signature=None, redacted=false, effort=None))]
    fn new(
        text: String,
        signature: Option<String>,
        redacted: bool,
        effort: Option<String>,
    ) -> Self {
        Self {
            inner: ReasoningTrace {
                text,
                signature,
                redacted,
                effort,
            },
        }
    }

    /// Plain-text rendering of the reasoning content.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Provider-specific signature/redaction handle (Anthropic), if any.
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
    /// (`"low"` / `"medium"` / `"high"` / `"max"` / ...).
    #[getter]
    fn effort(&self) -> Option<&str> {
        self.inner.effort.as_deref()
    }

    fn __repr__(&self) -> String {
        let preview: String = self.inner.text.chars().take(40).collect();
        format!(
            "ReasoningTrace(text={preview:?}, redacted={}, effort={:?})",
            self.inner.redacted, self.inner.effort
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
