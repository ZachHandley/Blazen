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

use std::sync::Arc;

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

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_TOKEN_COUNTER_HANDLER: &str = r#"
/**
 * A JavaScript function that counts tokens in a text string and returns the
 * count as a non-negative integer. Used by `TokenCounter` to bridge
 * tokenisers implemented in JS (e.g. `tiktoken-wasm`) into the Blazen
 * runtime.
 */
export type CountTokensHandler = (text: string) => number;
"#;

// ---------------------------------------------------------------------------
// WasmTokenCounter
// ---------------------------------------------------------------------------

/// A JavaScript-backed implementation of [`blazen_llm::tokens::TokenCounter`].
///
/// Wraps a JS callback that counts tokens for a single string. Per-message
/// overhead is computed in Rust using the same heuristic as
/// [`EstimateCounter`] so the counter behaves predictably on chat-message
/// arrays even when the underlying tokeniser only handles raw text.
///
/// ```js
/// import { TokenCounter } from '@blazen/sdk';
/// import { encode } from 'gpt-tokenizer';
///
/// const counter = new TokenCounter((text) => encode(text).length, 128000);
/// const n = counter.countTokens('Hello, world!');
/// ```
#[wasm_bindgen(js_name = "TokenCounter")]
pub struct WasmTokenCounter {
    inner: Arc<JsTokenCounter>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmTokenCounter {}
unsafe impl Sync for WasmTokenCounter {}

#[wasm_bindgen(js_class = "TokenCounter")]
impl WasmTokenCounter {
    /// Create a new token counter from a JS callback.
    ///
    /// @param handler      - Function that returns the token count for a string.
    /// @param contextSize  - Optional context window size (defaults to 128 000).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(handler: js_sys::Function, context_size: Option<u32>) -> Self {
        Self {
            inner: Arc::new(JsTokenCounter {
                handler,
                context_size: context_size.unwrap_or(128_000) as usize,
            }),
        }
    }

    /// Count tokens in a single string.
    #[wasm_bindgen(js_name = "countTokens")]
    #[must_use]
    pub fn count_tokens(&self, text: &str) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.count_tokens(text) as u32
        }
    }

    /// Count tokens for an array of chat messages.
    ///
    /// Each message should be a plain JS object with `role` and `content`
    /// fields; per-message overhead matches the
    /// [`blazen_llm::tokens::EstimateCounter`] heuristic.
    ///
    /// # Errors
    ///
    /// Throws if the message array fails to deserialise.
    #[wasm_bindgen(js_name = "countMessageTokens")]
    pub fn count_message_tokens(&self, messages: JsValue) -> Result<u32, JsError> {
        let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages)
            .map_err(|e| JsError::new(&format!("Failed to parse messages: {e}")))?;
        #[allow(clippy::cast_possible_truncation)]
        {
            Ok(self.inner.count_message_tokens(&msgs) as u32)
        }
    }

    /// Context window size (the value passed to the constructor).
    #[wasm_bindgen(getter, js_name = "contextSize")]
    #[must_use]
    pub fn context_size(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.context_size() as u32
        }
    }

    /// Compute the remaining tokens left in the context window after the
    /// given message array has been accounted for.
    ///
    /// # Errors
    ///
    /// Throws if the message array fails to deserialise.
    #[wasm_bindgen(js_name = "remainingTokens")]
    pub fn remaining_tokens(&self, messages: JsValue) -> Result<u32, JsError> {
        let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages)
            .map_err(|e| JsError::new(&format!("Failed to parse messages: {e}")))?;
        #[allow(clippy::cast_possible_truncation)]
        {
            Ok(self.inner.remaining_tokens(&msgs) as u32)
        }
    }
}

impl WasmTokenCounter {
    /// Borrow the inner `Arc<dyn TokenCounter>` for use by other crate
    /// modules that need to plug a JS-backed counter into a Blazen pipeline.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn inner_arc(&self) -> Arc<dyn TokenCounter> {
        Arc::clone(&self.inner) as Arc<dyn TokenCounter>
    }
}

/// Internal `TokenCounter` impl that calls into the JS handler.
struct JsTokenCounter {
    handler: js_sys::Function,
    context_size: usize,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsTokenCounter {}
unsafe impl Sync for JsTokenCounter {}

impl TokenCounter for JsTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        let arg = JsValue::from_str(text);
        let Ok(result) = self.handler.call1(&JsValue::NULL, &arg) else {
            return EstimateCounter::new(self.context_size).count_tokens(text);
        };
        match result.as_f64() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Some(n) if n.is_finite() && n >= 0.0 => n as usize,
            _ => EstimateCounter::new(self.context_size).count_tokens(text),
        }
    }

    fn count_message_tokens(&self, messages: &[ChatMessage]) -> usize {
        let mut total: usize = 0;
        for msg in messages {
            total = total.saturating_add(4);
            if let Some(text) = msg.content.text_content() {
                total = total.saturating_add(self.count_tokens(&text));
            }
            if let Some(name) = &msg.name {
                total = total.saturating_add(self.count_tokens(name));
            }
        }
        total.saturating_add(2)
    }

    fn context_size(&self) -> usize {
        self.context_size
    }
}
