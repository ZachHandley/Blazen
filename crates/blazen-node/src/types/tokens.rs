//! Token counting utilities for the Node.js bindings.
//!
//! Exposes the [`TokenCounter`] trait as a JS abstract base class plus two
//! concrete implementations:
//!
//! - [`JsEstimateCounter`] (`EstimateCounter`) — heuristic, always available.
//! - [`JsTiktokenCounter`] (`TiktokenCounter`) — exact BPE counting via
//!   `tiktoken-rs`, gated behind the `tiktoken` feature.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::tokens::{EstimateCounter, TokenCounter};

use crate::types::JsChatMessage;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Hold a `Send + Sync` boxed `TokenCounter` so it can be shared between
/// concrete subclasses and the abstract base.
type CounterArc = Arc<dyn TokenCounter>;

// ---------------------------------------------------------------------------
// JsTokenCounter (subclassable abstract base)
// ---------------------------------------------------------------------------

/// Abstract base class for token counters.
///
/// Concrete implementations are [`EstimateCounter`](JsEstimateCounter) and
/// (when built with the `tiktoken` feature) [`TiktokenCounter`](JsTiktokenCounter).
///
/// JavaScript subclasses **must** override `countTokens`,
/// `countMessageTokens`, and `contextSize`.
///
/// ```javascript
/// class MyCounter extends TokenCounter {
///   countTokens(text) { return Math.ceil(text.length / 4); }
///   countMessageTokens(messages) { /* ... */ }
///   contextSize() { return 32_000; }
/// }
/// ```
#[napi(js_name = "TokenCounter")]
pub struct JsTokenCounter {
    /// Concrete inner counter. `None` for JS subclasses.
    pub(crate) inner: Option<CounterArc>,
}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsTokenCounter {
    /// Construct a base `TokenCounter`.
    ///
    /// Called by JavaScript subclasses via `super()`. Instances created this
    /// way have no inner Rust counter — overriding the methods in the
    /// subclass is required.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self { inner: None }
    }

    /// Count tokens in a raw text string.
    ///
    /// Subclasses with no inner counter **must** override this method.
    #[napi(js_name = "countTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_tokens(&self, text: String) -> Result<u32> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("subclass must override countTokens()"))?;
        Ok(inner.count_tokens(&text) as u32)
    }

    /// Count tokens for an array of chat messages, including per-message
    /// overhead.
    ///
    /// Subclasses with no inner counter **must** override this method.
    #[napi(js_name = "countMessageTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_message_tokens(&self, messages: Vec<&JsChatMessage>) -> Result<u32> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason("subclass must override countMessageTokens()")
        })?;
        let chat_messages: Vec<blazen_llm::ChatMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        Ok(inner.count_message_tokens(&chat_messages) as u32)
    }

    /// The model's context window size in tokens.
    #[napi(js_name = "contextSize")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn context_size(&self) -> Result<u32> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("subclass must override contextSize()"))?;
        Ok(inner.context_size() as u32)
    }

    /// Tokens remaining after the given prompt, saturating at zero.
    #[napi(js_name = "remainingTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn remaining_tokens(&self, messages: Vec<&JsChatMessage>) -> Result<u32> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("subclass must override remainingTokens()"))?;
        let chat_messages: Vec<blazen_llm::ChatMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        Ok(inner.remaining_tokens(&chat_messages) as u32)
    }
}

// ---------------------------------------------------------------------------
// JsEstimateCounter (heuristic; always available)
// ---------------------------------------------------------------------------

/// Heuristic token counter that uses a characters-per-token ratio.
///
/// Default ratio is 3.5 characters per token, a reasonable approximation for
/// English text tokenised with BPE. Requires no external data files and
/// works on any platform.
///
/// ```javascript
/// const counter = new EstimateCounter(128_000);
/// const n = counter.countTokens("Hello, world!");
/// ```
#[napi(js_name = "EstimateCounter")]
pub struct JsEstimateCounter {
    pub(crate) inner: Arc<EstimateCounter>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsEstimateCounter {
    /// Create an estimate counter with the default 3.5 chars-per-token ratio.
    #[napi(constructor)]
    pub fn new(context_window: Option<u32>) -> Self {
        let ctx = context_window.unwrap_or(128_000) as usize;
        Self {
            inner: Arc::new(EstimateCounter::new(ctx)),
        }
    }

    /// Create an estimate counter with a custom chars-per-token ratio.
    #[napi(factory, js_name = "withRatio")]
    pub fn with_ratio(context_window: u32, chars_per_token: f64) -> Self {
        Self {
            inner: Arc::new(EstimateCounter::with_ratio(
                context_window as usize,
                chars_per_token,
            )),
        }
    }

    /// Count tokens in a raw text string.
    #[napi(js_name = "countTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_tokens(&self, text: String) -> u32 {
        self.inner.count_tokens(&text) as u32
    }

    /// Count tokens for an array of chat messages, including per-message overhead.
    #[napi(js_name = "countMessageTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_message_tokens(&self, messages: Vec<&JsChatMessage>) -> u32 {
        let chat_messages: Vec<blazen_llm::ChatMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.count_message_tokens(&chat_messages) as u32
    }

    /// The model's context window size in tokens.
    #[napi(js_name = "contextSize")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn context_size(&self) -> u32 {
        self.inner.context_size() as u32
    }

    /// Tokens remaining after the given prompt, saturating at zero.
    #[napi(js_name = "remainingTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn remaining_tokens(&self, messages: Vec<&JsChatMessage>) -> u32 {
        let chat_messages: Vec<blazen_llm::ChatMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.remaining_tokens(&chat_messages) as u32
    }
}

// ---------------------------------------------------------------------------
// JsTiktokenCounter (feature-gated)
// ---------------------------------------------------------------------------

/// Exact BPE token counter backed by `tiktoken-rs`.
///
/// Mirrors the per-message overhead rules documented by `OpenAI` for GPT-3.5,
/// GPT-4, GPT-4.1, and o-series models. Unknown model names return an error.
///
/// ```javascript
/// const counter = TiktokenCounter.forModel("gpt-4o");
/// const n = counter.countTokens("Hello, world!");
/// ```
#[cfg(feature = "tiktoken")]
#[napi(js_name = "TiktokenCounter")]
pub struct JsTiktokenCounter {
    pub(crate) inner: Arc<blazen_llm::tokens::TiktokenCounter>,
}

#[cfg(feature = "tiktoken")]
#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsTiktokenCounter {
    /// Create a counter tuned for the given model name.
    ///
    /// Throws if the model is unknown to `tiktoken-rs`.
    #[napi(factory, js_name = "forModel")]
    pub fn for_model(model: String) -> Result<Self> {
        let counter = blazen_llm::tokens::TiktokenCounter::for_model(&model)
            .map_err(crate::error::blazen_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(counter),
        })
    }

    /// Count tokens in a raw text string.
    #[napi(js_name = "countTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_tokens(&self, text: String) -> u32 {
        self.inner.count_tokens(&text) as u32
    }

    /// Count tokens for an array of chat messages, including per-message overhead.
    #[napi(js_name = "countMessageTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_message_tokens(&self, messages: Vec<&JsChatMessage>) -> u32 {
        let chat_messages: Vec<blazen_llm::ChatMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.count_message_tokens(&chat_messages) as u32
    }

    /// The model's context window size in tokens.
    #[napi(js_name = "contextSize")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn context_size(&self) -> u32 {
        self.inner.context_size() as u32
    }

    /// Tokens remaining after the given prompt, saturating at zero.
    #[napi(js_name = "remainingTokens")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn remaining_tokens(&self, messages: Vec<&JsChatMessage>) -> u32 {
        let chat_messages: Vec<blazen_llm::ChatMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        self.inner.remaining_tokens(&chat_messages) as u32
    }
}

// ---------------------------------------------------------------------------
// Public free functions
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text string.
///
/// Uses a heuristic (3.5 characters per token) that works everywhere without
/// external data files. Good enough for budget checks when exact counts are
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
