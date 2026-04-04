//! Python wrapper for the token-windowed conversation memory.

use pyo3::prelude::*;

use blazen_llm::chat_window::ChatWindow;

use crate::types::message::PyChatMessage;

// ---------------------------------------------------------------------------
// PyChatWindow
// ---------------------------------------------------------------------------

/// A token-windowed conversation memory.
///
/// Holds a sequence of ChatMessage objects and automatically evicts the
/// oldest messages (preserving system messages) when the token count
/// exceeds the configured budget.
///
/// Example:
///     >>> window = ChatWindow(max_tokens=4096)
///     >>> window.add(ChatMessage.system("You are helpful."))
///     >>> window.add(ChatMessage.user("Hello!"))
///     >>> print(window.token_count())
///     >>> print(window.remaining_tokens())
#[pyclass(name = "ChatWindow")]
pub struct PyChatWindow {
    inner: ChatWindow,
}

#[pymethods]
impl PyChatWindow {
    /// Create a new chat window with the given token budget.
    ///
    /// Args:
    ///     max_tokens: The maximum number of tokens to allow in the window.
    #[new]
    fn new(max_tokens: usize) -> Self {
        Self {
            inner: ChatWindow::new(max_tokens),
        }
    }

    /// Append a message to the window.
    ///
    /// If the token count exceeds the budget after adding the message,
    /// the oldest non-system messages are evicted until within budget.
    /// System messages are never evicted.
    ///
    /// Args:
    ///     message: A ChatMessage to add.
    fn add(&mut self, message: PyRef<'_, PyChatMessage>) {
        self.inner.add(message.inner.clone());
    }

    /// Get the current messages in the window.
    ///
    /// Returns:
    ///     A list of ChatMessage objects.
    fn messages(&self) -> Vec<PyChatMessage> {
        self.inner
            .messages()
            .iter()
            .map(|m| PyChatMessage { inner: m.clone() })
            .collect()
    }

    /// Remove all messages from the window.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Estimate the total token count of all messages in the window.
    ///
    /// Returns:
    ///     The estimated number of tokens.
    fn token_count(&self) -> usize {
        self.inner.token_count()
    }

    /// Return the number of tokens remaining before the budget is reached.
    ///
    /// Returns:
    ///     The number of tokens remaining (0 if already at or over budget).
    fn remaining_tokens(&self) -> usize {
        self.inner.remaining_tokens()
    }

    /// Return the number of messages in the window.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatWindow(messages={}, tokens={})",
            self.inner.len(),
            self.inner.token_count()
        )
    }
}
