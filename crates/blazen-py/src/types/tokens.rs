//! Python wrappers for token counting utilities.

use pyo3::prelude::*;

use blazen_llm::ChatMessage;
use blazen_llm::tokens::{EstimateCounter, TokenCounter};

use crate::types::message::PyChatMessage;

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
#[pyfunction]
#[pyo3(signature = (messages, context_size=128_000))]
pub fn count_message_tokens(messages: Vec<PyRef<'_, PyChatMessage>>, context_size: usize) -> usize {
    let counter = EstimateCounter::new(context_size);
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    counter.count_message_tokens(&rust_messages)
}
