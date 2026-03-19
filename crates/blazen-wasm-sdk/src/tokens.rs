//! Token counting utilities for WASM.
//!
//! Exposes the heuristic [`EstimateCounter`] from `blazen-llm` as standalone
//! JavaScript functions. The tiktoken-based counter is **not** available in
//! WASM (it relies on native data files that don't compile to `wasm32`).
//!
//! ```js
//! import { estimateTokens, countMessageTokens } from '@blazen/sdk';
//!
//! const count = estimateTokens('Hello, world!');
//! console.log(count); // 4
//!
//! const msgCount = countMessageTokens([
//!   { role: 'system', content: 'You are helpful.' },
//!   { role: 'user', content: 'Hello!' },
//! ]);
//! console.log(msgCount); // ~21
//! ```

use wasm_bindgen::prelude::*;

use blazen_llm::tokens::{EstimateCounter, TokenCounter};
use blazen_llm::types::ChatMessage;

/// Estimate the number of tokens in a text string.
///
/// Uses a heuristic (~3.5 characters per token) which is a reasonable
/// approximation for English text tokenised with BPE.
///
/// `contextSize` optionally sets the context window size (defaults to
/// 128 000). This does not affect the count itself but is stored for use
/// by `remainingTokens()` if you use the counter directly.
///
/// Returns the estimated token count.
#[wasm_bindgen(js_name = "estimateTokens")]
pub fn estimate_tokens(text: &str, context_size: Option<u32>) -> u32 {
    let counter = EstimateCounter::new(context_size.unwrap_or(128_000) as usize);
    #[allow(clippy::cast_possible_truncation)]
    {
        counter.count_tokens(text) as u32
    }
}

/// Count the estimated tokens for an array of chat messages.
///
/// Each message in the array should be a plain JS object with at least
/// `role` (string) and `content` (string) fields.
///
/// Includes per-message overhead and assistant priming tokens, matching
/// the heuristic used by the Blazen `EstimateCounter`.
///
/// Returns the estimated total token count, or throws on parse failure.
#[wasm_bindgen(js_name = "countMessageTokens")]
pub fn count_message_tokens(messages: JsValue, context_size: Option<u32>) -> Result<u32, JsError> {
    let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages)
        .map_err(|e| JsError::new(&format!("Failed to parse messages: {e}")))?;

    let counter = EstimateCounter::new(context_size.unwrap_or(128_000) as usize);
    #[allow(clippy::cast_possible_truncation)]
    {
        Ok(counter.count_message_tokens(&msgs) as u32)
    }
}
