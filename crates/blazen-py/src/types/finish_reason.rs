//! Python wrapper for `FinishReason`.

use pyo3::prelude::*;

use blazen_llm::FinishReason;

/// Normalized finish reason across providers.
///
/// Maps provider-specific finish-reason strings (``"stop"``, ``"end_turn"``,
/// ``"STOP"``, ``"length"``, ``"tool_calls"``, ``"max_tokens"``, etc.) into a
/// canonical set. Unknown values fall through to ``kind == "other"``.
///
/// Use ``kind`` for the canonical category and ``value`` for the original
/// provider string.
///
/// Example:
///     >>> normalized = response.finish_reason_normalized()
///     >>> if normalized is not None and normalized.kind == "tool_calls":
///     ...     handle_tool_calls(response.tool_calls)
#[pyclass(name = "FinishReason", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFinishReason {
    pub(crate) inner: FinishReason,
}

#[pymethods]
impl PyFinishReason {
    /// Canonical category.
    ///
    /// One of: ``"stop"``, ``"length"``, ``"tool_calls"``, ``"content_filter"``,
    /// ``"safety"``, ``"end_turn"``, ``"stop_sequence"``, ``"max_tokens"``,
    /// ``"error"``, or ``"other"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Safety => "safety",
            FinishReason::EndTurn => "end_turn",
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::MaxTokens => "max_tokens",
            FinishReason::Error => "error",
            FinishReason::Other(_) => "other",
        }
    }

    /// The raw provider string.
    ///
    /// For canonical variants this equals ``kind``. For ``Other(s)`` it
    /// returns the original string the provider sent.
    #[getter]
    fn value(&self) -> String {
        match &self.inner {
            FinishReason::Other(s) => s.clone(),
            _ => self.kind().to_owned(),
        }
    }

    /// Build a [`FinishReason`] from a raw provider string.
    #[staticmethod]
    fn from_provider_string(s: &str) -> Self {
        Self {
            inner: FinishReason::from_provider_string(s),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FinishReason(kind={:?}, value={:?})",
            self.kind(),
            self.value()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl From<FinishReason> for PyFinishReason {
    fn from(inner: FinishReason) -> Self {
        Self { inner }
    }
}

impl From<&FinishReason> for PyFinishReason {
    fn from(inner: &FinishReason) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
