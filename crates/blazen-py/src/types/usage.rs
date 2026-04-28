//! Python wrappers for token usage and request timing types.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::types::TokenUsage;

// Re-export RequestTiming so existing imports (`crate::types::usage::RequestTiming`)
// keep working.
pub use blazen_llm::types::RequestTiming;

/// Token usage statistics for a completion request.
///
/// Example:
///     >>> usage = response.usage
///     >>> if usage is not None:
///     ...     print(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
#[gen_stub_pyclass]
#[pyclass(name = "TokenUsage", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTokenUsage {
    pub(crate) inner: TokenUsage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTokenUsage {
    /// Construct a token-usage record explicitly.
    #[new]
    #[pyo3(signature = (
        *,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        reasoning_tokens=0,
        cached_input_tokens=0,
        audio_input_tokens=0,
        audio_output_tokens=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        reasoning_tokens: u32,
        cached_input_tokens: u32,
        audio_input_tokens: u32,
        audio_output_tokens: u32,
    ) -> Self {
        Self {
            inner: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens,
                cached_input_tokens,
                audio_input_tokens,
                audio_output_tokens,
            },
        }
    }

    /// Number of tokens in the prompt / input.
    #[getter]
    fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Number of tokens in the completion / output.
    #[getter]
    fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    /// Total tokens consumed (prompt + completion).
    #[getter]
    fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    /// Tokens spent on hidden reasoning (o-series, R1, Anthropic thinking).
    #[getter]
    fn reasoning_tokens(&self) -> u32 {
        self.inner.reasoning_tokens
    }

    /// Tokens served from a prompt cache, if the provider reports them.
    #[getter]
    fn cached_input_tokens(&self) -> u32 {
        self.inner.cached_input_tokens
    }

    /// Tokens consumed by audio input.
    #[getter]
    fn audio_input_tokens(&self) -> u32 {
        self.inner.audio_input_tokens
    }

    /// Tokens consumed by audio output.
    #[getter]
    fn audio_output_tokens(&self) -> u32 {
        self.inner.audio_output_tokens
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenUsage(prompt={}, completion={}, total={}, reasoning={})",
            self.inner.prompt_tokens,
            self.inner.completion_tokens,
            self.inner.total_tokens,
            self.inner.reasoning_tokens,
        )
    }
}

impl From<TokenUsage> for PyTokenUsage {
    fn from(inner: TokenUsage) -> Self {
        Self { inner }
    }
}

impl From<&TokenUsage> for PyTokenUsage {
    fn from(inner: &TokenUsage) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
