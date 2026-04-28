//! Python wrappers for token counting utilities.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_llm::ChatMessage;
use blazen_llm::tokens::{EstimateCounter, TokenCounter};

use crate::types::message::PyChatMessage;

// ---------------------------------------------------------------------------
// PyTokenCounter (subclassable abstract base)
// ---------------------------------------------------------------------------

/// Abstract base class for token counters.
///
/// Subclass and override ``count_tokens``, ``count_message_tokens``, and
/// ``context_size`` to plug in a custom counting strategy. The default
/// implementations raise ``NotImplementedError``.
///
/// Example:
///     >>> class MyCounter(TokenCounter):
///     ...     def count_tokens(self, text: str) -> int:
///     ...         return len(text) // 4
///     ...     def count_message_tokens(self, messages):
///     ...         return sum(self.count_tokens(m.content or "") for m in messages)
///     ...     def context_size(self) -> int:
///     ...         return 128_000
#[gen_stub_pyclass]
#[pyclass(name = "TokenCounter", subclass)]
pub struct PyTokenCounter;

#[gen_stub_pymethods]
#[pymethods]
impl PyTokenCounter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Count tokens in a raw text string.
    fn count_tokens(&self, _text: &str) -> PyResult<usize> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override count_tokens()",
        ))
    }

    /// Count tokens for a list of chat messages, including per-message overhead.
    fn count_message_tokens(&self, _messages: Vec<PyRef<'_, PyChatMessage>>) -> PyResult<usize> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override count_message_tokens()",
        ))
    }

    /// Maximum context window size (tokens) for the model this counter targets.
    fn context_size(&self) -> PyResult<usize> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override context_size()",
        ))
    }

    /// Number of tokens remaining after the given prompt.
    fn remaining_tokens(&self, messages: Vec<PyRef<'_, PyChatMessage>>) -> PyResult<usize> {
        let used = self.count_message_tokens(messages)?;
        Ok(self.context_size()?.saturating_sub(used))
    }
}

// ---------------------------------------------------------------------------
// PyEstimateCounter
// ---------------------------------------------------------------------------

/// Lightweight heuristic token counter (chars-per-token ratio).
///
/// Default ratio is 3.5 chars/token, a reasonable approximation for English
/// BPE tokenisation. Requires no external data files and works everywhere.
#[gen_stub_pyclass]
#[pyclass(name = "EstimateCounter", extends = PyTokenCounter)]
pub struct PyEstimateCounter {
    pub(crate) inner: EstimateCounter,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEstimateCounter {
    /// Build a counter with the default 3.5 chars/token ratio.
    ///
    /// Args:
    ///     context_size: The model's context window in tokens.
    ///     chars_per_token: Optional override for the chars-per-token ratio.
    #[new]
    #[pyo3(signature = (context_size=128_000, chars_per_token=None))]
    fn new(
        context_size: usize,
        chars_per_token: Option<f64>,
    ) -> (PyEstimateCounter, PyTokenCounter) {
        let inner = match chars_per_token {
            Some(r) => EstimateCounter::with_ratio(context_size, r),
            None => EstimateCounter::new(context_size),
        };
        (Self { inner }, PyTokenCounter)
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }

    fn count_message_tokens(&self, messages: Vec<PyRef<'_, PyChatMessage>>) -> usize {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.count_message_tokens(&rust_messages)
    }

    fn context_size(&self) -> usize {
        self.inner.context_size()
    }

    fn remaining_tokens(&self, messages: Vec<PyRef<'_, PyChatMessage>>) -> usize {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.remaining_tokens(&rust_messages)
    }
}

// ---------------------------------------------------------------------------
// PyTiktokenCounter (feature-gated)
// ---------------------------------------------------------------------------

/// Exact BPE token counter backed by ``tiktoken-rs`` (feature: ``tiktoken``).
#[cfg(feature = "tiktoken")]
#[gen_stub_pyclass]
#[pyclass(name = "TiktokenCounter", extends = PyTokenCounter)]
pub struct PyTiktokenCounter {
    pub(crate) inner: blazen_llm::tokens::TiktokenCounter,
}

#[cfg(feature = "tiktoken")]
#[gen_stub_pymethods]
#[pymethods]
impl PyTiktokenCounter {
    /// Build a counter for a specific model identifier.
    ///
    /// Raises ``ValueError`` if ``tiktoken-rs`` does not recognise the model.
    #[new]
    #[pyo3(signature = (*, model))]
    fn new(model: &str) -> PyResult<(PyTiktokenCounter, PyTokenCounter)> {
        let inner = blazen_llm::tokens::TiktokenCounter::for_model(model)
            .map_err(crate::error::blazen_error_to_pyerr)?;
        Ok((Self { inner }, PyTokenCounter))
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }

    fn count_message_tokens(&self, messages: Vec<PyRef<'_, PyChatMessage>>) -> usize {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.count_message_tokens(&rust_messages)
    }

    fn context_size(&self) -> usize {
        self.inner.context_size()
    }

    fn remaining_tokens(&self, messages: Vec<PyRef<'_, PyChatMessage>>) -> usize {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.remaining_tokens(&rust_messages)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text string.
///
/// Uses a lightweight heuristic (approximately 3.5 characters per token)
/// that works everywhere without external data files.
///
/// Args:
///     text: The text to estimate tokens for.
///     context_size: The context window size (default: 128000).
///
/// Returns:
///     The estimated token count.
///
/// Example:
///     >>> tokens = estimate_tokens("Hello, world!")
///     >>> print(tokens)  # 4
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (text, context_size=128_000))]
pub fn estimate_tokens(text: &str, context_size: usize) -> usize {
    let counter = EstimateCounter::new(context_size);
    counter.count_tokens(text)
}

/// Estimate the number of tokens for a list of chat messages.
///
/// Includes per-message overhead tokens (role markers, separators, etc.)
/// in addition to the content tokens.
///
/// Args:
///     messages: A list of ChatMessage objects.
///     context_size: The context window size (default: 128000).
///
/// Returns:
///     The estimated token count for the full message list.
///
/// Example:
///     >>> tokens = count_message_tokens([
///     ...     ChatMessage.system("You are helpful."),
///     ...     ChatMessage.user("Hello!"),
///     ... ])
///     >>> print(tokens)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (messages, context_size=128_000))]
pub fn count_message_tokens(messages: Vec<PyRef<'_, PyChatMessage>>, context_size: usize) -> usize {
    let counter = EstimateCounter::new(context_size);
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    counter.count_message_tokens(&rust_messages)
}
