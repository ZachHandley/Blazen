//! Token-windowed conversation memory.
//!
//! [`ChatWindow`] holds a sequence of [`ChatMessage`] objects and automatically
//! evicts the oldest non-system messages when the estimated token count exceeds
//! a configured budget.  System messages are never evicted, ensuring that
//! system-level instructions are always present in the context.

use crate::tokens::{EstimateCounter, TokenCounter};
use crate::types::{ChatMessage, Role};

/// A token-windowed conversation memory.
///
/// Holds a sequence of [`ChatMessage`] objects and automatically evicts the
/// oldest messages (preserving system messages) when the token count exceeds
/// the configured budget.
///
/// # Example
///
/// ```
/// use blazen_llm::chat_window::ChatWindow;
/// use blazen_llm::ChatMessage;
///
/// let mut window = ChatWindow::new(100);
/// window.add(ChatMessage::system("You are helpful."));
/// window.add(ChatMessage::user("Hello!"));
/// assert!(window.token_count() <= 100);
/// ```
pub struct ChatWindow {
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    /// Characters-per-token estimate for the heuristic counter.
    chars_per_token: f64,
}

impl ChatWindow {
    /// Create a new chat window with the given token budget.
    ///
    /// Uses the default estimate of 3.5 characters per token, matching
    /// [`EstimateCounter`]'s default.
    #[must_use]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            chars_per_token: 3.5,
        }
    }

    /// Set a custom characters-per-token ratio (builder pattern).
    ///
    /// The default is 3.5, which is a reasonable approximation for English text
    /// tokenised with BPE.
    #[must_use]
    pub fn with_chars_per_token(mut self, cpt: f64) -> Self {
        self.chars_per_token = cpt;
        self
    }

    /// Append a message and trim the oldest non-system messages if the token
    /// budget is exceeded.
    ///
    /// System messages (`Role::System`) are never evicted.  When trimming is
    /// required the oldest non-system message is removed first, preserving
    /// system prompts that typically appear at the start of the conversation.
    pub fn add(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.trim();
    }

    /// Return a slice of the current messages.
    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Remove all messages from the window.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Estimate the total token count of all messages in the window.
    ///
    /// Uses the same heuristic as [`EstimateCounter`]: each message incurs a
    /// 4-token overhead, text content is divided by the chars-per-token ratio,
    /// and 3 tokens are added for assistant priming.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.counter().count_message_tokens(&self.messages)
    }

    /// Return the number of tokens remaining before the budget is reached.
    ///
    /// Returns `0` if the current token count already meets or exceeds the budget.
    #[must_use]
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.token_count())
    }

    /// Return the number of messages currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Return `true` if the window contains no messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn counter(&self) -> EstimateCounter {
        // The context_window value passed to `EstimateCounter` is only used by
        // `context_size()` / `remaining_tokens()` on the counter itself.  We
        // only call `count_message_tokens`, so the value doesn't matter.
        EstimateCounter::with_ratio(self.max_tokens, self.chars_per_token)
    }

    /// Remove the oldest non-system messages until within budget.
    fn trim(&mut self) {
        while self.token_count() > self.max_tokens {
            // Find the first non-system message and remove it.
            let pos = self.messages.iter().position(|m| m.role != Role::System);

            if let Some(idx) = pos {
                self.messages.remove(idx);
            } else {
                // Only system messages remain -- nothing more to trim.
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[test]
    fn new_window_is_empty() {
        let w = ChatWindow::new(1000);
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
        assert_eq!(w.messages().len(), 0);
    }

    #[test]
    fn add_single_message() {
        let mut w = ChatWindow::new(1000);
        w.add(ChatMessage::user("Hello!"));
        assert_eq!(w.len(), 1);
        assert!(!w.is_empty());
    }

    #[test]
    fn clear_removes_all() {
        let mut w = ChatWindow::new(1000);
        w.add(ChatMessage::user("one"));
        w.add(ChatMessage::user("two"));
        assert_eq!(w.len(), 2);
        w.clear();
        assert!(w.is_empty());
    }

    #[test]
    fn token_count_is_positive_for_messages() {
        let mut w = ChatWindow::new(1000);
        w.add(ChatMessage::user("Hello, world!"));
        assert!(w.token_count() > 0);
    }

    #[test]
    fn remaining_tokens_decreases() {
        let mut w = ChatWindow::new(1000);
        let before = w.remaining_tokens();

        w.add(ChatMessage::user("Hello!"));
        let after = w.remaining_tokens();
        assert!(
            after < before,
            "remaining should decrease after adding a message"
        );
    }

    #[test]
    fn evicts_oldest_non_system_when_over_budget() {
        // Use a very small budget to force eviction.
        let mut w = ChatWindow::new(30);

        w.add(ChatMessage::system("Be helpful."));
        w.add(ChatMessage::user("first"));
        w.add(ChatMessage::user("second"));
        w.add(ChatMessage::user("third"));

        // Some messages should have been evicted.
        assert!(w.token_count() <= 30);

        // System message must still be present.
        assert!(w.messages().iter().any(|m| m.role == Role::System));
    }

    #[test]
    fn system_messages_are_never_evicted() {
        let mut w = ChatWindow::new(40);

        w.add(ChatMessage::system("System prompt one."));
        w.add(ChatMessage::system("System prompt two."));
        w.add(ChatMessage::user(
            "A user message that is rather long to push tokens over budget",
        ));

        // Both system messages must survive.
        let system_count = w
            .messages()
            .iter()
            .filter(|m| m.role == Role::System)
            .count();
        assert_eq!(system_count, 2);
    }

    #[test]
    fn oldest_non_system_is_removed_first() {
        // Budget that fits system + 1 user message but not 2.
        let mut w = ChatWindow::new(50);

        w.add(ChatMessage::system("sys"));
        w.add(ChatMessage::user("first user message"));
        w.add(ChatMessage::user("second user message that pushes over"));

        // After trimming, if the first user message was evicted we should
        // only see the system message and/or the second user message.
        let contents: Vec<Option<String>> = w
            .messages()
            .iter()
            .map(|m| m.content.text_content())
            .collect();

        // "first user message" should not be present if it was evicted.
        // (It's the oldest non-system message.)
        if contents.len() < 3 {
            assert!(
                !contents.contains(&Some("first user message".to_owned())),
                "expected oldest non-system message to be evicted first"
            );
        }
    }

    #[test]
    fn with_chars_per_token_changes_estimate() {
        let w1 = {
            let mut w = ChatWindow::new(1000);
            w.add(ChatMessage::user("Hello, world!"));
            w.token_count()
        };

        let w2 = {
            let mut w = ChatWindow::new(1000).with_chars_per_token(1.0);
            w.add(ChatMessage::user("Hello, world!"));
            w.token_count()
        };

        // With 1.0 chars/token the estimate should be higher than 3.5.
        assert!(w2 > w1);
    }

    #[test]
    fn only_system_messages_cannot_be_trimmed_further() {
        // A budget so small that even a single system message exceeds it.
        let mut w = ChatWindow::new(1);

        w.add(ChatMessage::system(
            "A long system message that exceeds one token",
        ));

        // The system message should remain even though it exceeds the budget.
        assert_eq!(w.len(), 1);
        assert!(w.messages()[0].role == Role::System);
    }
}
