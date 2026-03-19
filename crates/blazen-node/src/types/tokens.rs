//! Token counting utilities for the Node.js bindings.

use napi_derive::napi;

use blazen_llm::tokens::{EstimateCounter, TokenCounter};

use crate::types::JsChatMessage;

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text string.
///
/// Uses a heuristic (3.5 characters per token) that works everywhere without
/// external data files.  Good enough for budget checks when exact counts are
/// not critical.
///
/// `contextSize` defaults to 128 000 if not provided.
///
/// ```javascript
/// const tokens = estimateTokens("Hello world");
/// ```
#[napi(js_name = "estimateTokens")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn estimate_tokens(text: String, context_size: Option<u32>) -> u32 {
    let ctx = context_size.unwrap_or(128_000) as usize;
    let counter = EstimateCounter::new(ctx);
    #[allow(clippy::cast_possible_truncation)]
    {
        counter.count_tokens(&text) as u32
    }
}

/// Estimate the total token count for an array of chat messages.
///
/// Includes per-message overhead (role markers, separators) and assistant
/// priming tokens, matching the overhead model used by `OpenAI`.
///
/// `contextSize` defaults to 128 000 if not provided.
///
/// ```javascript
/// const tokens = countMessageTokens([
///   ChatMessage.system("You are helpful."),
///   ChatMessage.user("Hi"),
/// ]);
/// ```
#[napi(js_name = "countMessageTokens")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn count_message_tokens(messages: Vec<&JsChatMessage>, context_size: Option<u32>) -> u32 {
    let ctx = context_size.unwrap_or(128_000) as usize;
    let counter = EstimateCounter::new(ctx);
    let chat_messages: Vec<blazen_llm::ChatMessage> =
        messages.iter().map(|m| m.inner.clone()).collect();
    #[allow(clippy::cast_possible_truncation)]
    {
        counter.count_message_tokens(&chat_messages) as u32
    }
}
