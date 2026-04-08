//! Token-windowed conversation memory bindings for the Node.js SDK.
//!
//! Wraps `blazen_llm::chat_window::ChatWindow` for use from JavaScript/TypeScript.

use napi_derive::napi;

use blazen_llm::chat_window::ChatWindow;

use super::message::JsChatMessage;

// ---------------------------------------------------------------------------
// JsChatWindow
// ---------------------------------------------------------------------------

/// A token-windowed conversation memory.
///
/// Holds a sequence of `ChatMessage` objects and automatically evicts the
/// oldest messages (preserving system messages) when the token count exceeds
/// the configured budget.
///
/// ```javascript
/// const window = new ChatWindow(4096);
/// window.add(ChatMessage.system("You are helpful."));
/// window.add(ChatMessage.user("Hello!"));
/// console.log(window.tokenCount());
/// console.log(window.remainingTokens());
/// ```
#[napi(js_name = "ChatWindow")]
pub struct JsChatWindow {
    inner: ChatWindow,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::new_without_default)]
impl JsChatWindow {
    /// Create a new chat window with the given token budget.
    ///
    /// @param maxTokens - The maximum number of tokens to allow in the window.
    #[napi(constructor)]
    pub fn new(max_tokens: u32) -> Self {
        Self {
            inner: ChatWindow::new(max_tokens as usize),
        }
    }

    /// Append a message to the window.
    ///
    /// If the token count exceeds the budget after adding the message,
    /// the oldest non-system messages are evicted until within budget.
    /// System messages are never evicted.
    ///
    /// @param message - A `ChatMessage` to add.
    #[napi]
    pub fn add(&mut self, message: &JsChatMessage) {
        self.inner.add(message.inner.clone());
    }

    /// Get the current messages in the window.
    ///
    /// @returns An array of `ChatMessage` objects.
    #[napi]
    pub fn messages(&self) -> Vec<JsChatMessage> {
        self.inner
            .messages()
            .iter()
            .map(|m| JsChatMessage { inner: m.clone() })
            .collect()
    }

    /// Remove all messages from the window.
    #[napi]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Estimate the total token count of all messages in the window.
    ///
    /// @returns The estimated number of tokens.
    #[napi(js_name = "tokenCount")]
    pub fn token_count(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.token_count() as u32
        }
    }

    /// Return the number of tokens remaining before the budget is reached.
    ///
    /// @returns The number of tokens remaining (0 if already at or over budget).
    #[napi(js_name = "remainingTokens")]
    pub fn remaining_tokens(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.remaining_tokens() as u32
        }
    }

    /// Return the number of messages in the window.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.len() as u32
        }
    }
}
