//! Standalone `wasm-bindgen` wrappers for the [`blazen_llm`] decorator
//! types: [`blazen_llm::fallback::FallbackModel`],
//! [`blazen_llm::retry::RetryCompletionModel`], and
//! [`blazen_llm::cache::CachedCompletionModel`].
//!
//! These classes provide an explicit, constructor-style alternative to the
//! decorator builder methods (`withRetry`, `withFallback`, `withCache`) on
//! [`crate::completion_model::WasmCompletionModel`]. They are particularly
//! useful when the configuration is built up dynamically or when callers
//! prefer to name the wrapper type explicitly.
//!
//! Each class is itself a `CompletionModel` -- call `.toCompletionModel()`
//! to obtain a generic [`crate::completion_model::WasmCompletionModel`]
//! that can be passed to `runAgent`, `batchComplete`, etc.

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::cache::{CacheConfig, CacheStrategy, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::CompletionRequest;

use crate::chat_message::js_messages_to_vec;
use crate::completion_model::WasmCompletionModel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read an optional unsigned integer field from a JS options object.
fn read_u32(obj: &JsValue, key: &str) -> Option<u32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|n| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                n as u32
            }
        })
}

/// Read an optional unsigned 64-bit integer field from a JS options object.
fn read_u64(obj: &JsValue, key: &str) -> Option<u64> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|n| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                n as u64
            }
        })
}

/// Read an optional boolean field from a JS options object.
fn read_bool(obj: &JsValue, key: &str) -> Option<bool> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_bool())
}

/// Build a [`RetryConfig`] from a JS options object.
///
/// Recognised keys (all optional): `maxRetries`, `initialDelayMs`,
/// `maxDelayMs`, `honorRetryAfter`, `jitter`.
pub(crate) fn build_retry_config(options: &JsValue) -> RetryConfig {
    let mut config = RetryConfig::default();
    if !options.is_object() {
        return config;
    }
    if let Some(v) = read_u32(options, "maxRetries") {
        config.max_retries = v;
    }
    if let Some(v) = read_u64(options, "initialDelayMs") {
        config.initial_delay_ms = v;
    }
    if let Some(v) = read_u64(options, "maxDelayMs") {
        config.max_delay_ms = v;
    }
    if let Some(v) = read_bool(options, "honorRetryAfter") {
        config.honor_retry_after = v;
    }
    if let Some(v) = read_bool(options, "jitter") {
        config.jitter = v;
    }
    config
}

/// Build a [`CacheConfig`] from a JS options object.
///
/// Recognised keys (all optional): `strategy` (`"none"`, `"contentHash"`,
/// `"anthropicEphemeral"`, `"auto"`), `ttlSeconds`, `maxEntries`.
pub(crate) fn build_cache_config(options: &JsValue) -> CacheConfig {
    let mut config = CacheConfig::default();
    if !options.is_object() {
        return config;
    }
    if let Some(s) = js_sys::Reflect::get(options, &JsValue::from_str("strategy"))
        .ok()
        .and_then(|v| v.as_string())
    {
        config.strategy = match s.as_str() {
            "none" | "None" => CacheStrategy::None,
            "anthropicEphemeral" | "anthropic_ephemeral" | "AnthropicEphemeral" => {
                CacheStrategy::AnthropicEphemeral
            }
            "auto" | "Auto" => CacheStrategy::Auto,
            _ => CacheStrategy::ContentHash,
        };
    }
    if let Some(v) = read_u64(options, "ttlSeconds") {
        config.ttl_seconds = v;
    }
    if let Some(v) = read_u32(options, "maxEntries") {
        config.max_entries = v as usize;
    }
    config
}

// ---------------------------------------------------------------------------
// FallbackModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` that tries multiple providers in order, falling back
/// on retryable failures.
///
/// Non-retryable errors (auth, validation, content policy) short-circuit
/// immediately so that broken credentials are not masked by a fallback
/// attempt.
///
/// ```js
/// const primary = CompletionModel.openai();
/// const backup = CompletionModel.groq();
/// const resilient = new FallbackModel([primary, backup]);
/// const response = await resilient.complete([ChatMessage.user('Hi')]);
/// ```
#[wasm_bindgen(js_name = "FallbackModel")]
pub struct WasmFallbackModel {
    inner: Arc<dyn CompletionModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmFallbackModel {}
unsafe impl Sync for WasmFallbackModel {}

#[wasm_bindgen(js_class = "FallbackModel")]
impl WasmFallbackModel {
    /// Create a fallback model from an ordered list of `CompletionModel`s.
    ///
    /// The first provider is tried first; subsequent providers are tried
    /// only when the previous one fails with a retryable error.
    ///
    /// Throws if `models` is empty.
    #[wasm_bindgen(constructor)]
    pub fn new(models: Vec<WasmCompletionModel>) -> Result<WasmFallbackModel, JsValue> {
        if models.is_empty() {
            return Err(JsValue::from_str(
                "FallbackModel requires at least one provider",
            ));
        }
        let providers: Vec<Arc<dyn CompletionModel>> =
            models.into_iter().map(|m| m.inner_arc()).collect();
        Ok(Self {
            inner: Arc::new(FallbackModel::new(providers)),
        })
    }

    /// The default model identifier of the primary provider.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Convert this fallback wrapper into a generic `CompletionModel` so it
    /// can be passed to `runAgent`, `batchComplete`, or further decorators.
    #[wasm_bindgen(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> WasmCompletionModel {
        WasmCompletionModel::from_arc(Arc::clone(&self.inner))
    }

    /// Perform a non-streaming chat completion through the fallback chain.
    #[wasm_bindgen]
    pub fn complete(&self, messages: JsValue) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let request = CompletionRequest::new(msgs);
            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Perform a streaming chat completion through the fallback chain.
    #[wasm_bindgen]
    pub fn stream(&self, messages: JsValue, callback: js_sys::Function) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let request = CompletionRequest::new(msgs);
            let mut stream = model
                .stream(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            use futures_util::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(|e| JsValue::from_str(&e.to_string()))?;
                let js_chunk = serde_wasm_bindgen::to_value(&chunk)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let _ = callback.call1(&JsValue::NULL, &js_chunk);
            }
            Ok(JsValue::UNDEFINED)
        })
    }
}

// ---------------------------------------------------------------------------
// RetryCompletionModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` that retries transient failures with exponential
/// backoff and optional `Retry-After` honouring.
///
/// ```js
/// const model = CompletionModel.openai();
/// const resilient = new RetryCompletionModel(model, {
///   maxRetries: 5,
///   initialDelayMs: 500,
///   jitter: true,
/// });
/// ```
#[wasm_bindgen(js_name = "RetryCompletionModel")]
pub struct WasmRetryCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmRetryCompletionModel {}
unsafe impl Sync for WasmRetryCompletionModel {}

#[wasm_bindgen(js_class = "RetryCompletionModel")]
impl WasmRetryCompletionModel {
    /// Wrap `model` with the given retry options.
    ///
    /// `options` is an optional plain JS object with the following fields
    /// (all optional, snake-case is also accepted by the underlying serde
    /// derive but the camel-case spelling shown here is preferred):
    /// - `maxRetries` (number) -- default 3
    /// - `initialDelayMs` (number) -- default 1000
    /// - `maxDelayMs` (number) -- default 30000
    /// - `honorRetryAfter` (boolean) -- default true
    /// - `jitter` (boolean) -- default true
    #[wasm_bindgen(constructor)]
    pub fn new(model: &WasmCompletionModel, options: JsValue) -> WasmRetryCompletionModel {
        let config = build_retry_config(&options);
        Self {
            inner: Arc::new(RetryCompletionModel::from_arc(model.inner_arc(), config)),
        }
    }

    /// The default model identifier of the wrapped provider.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Convert this retry wrapper into a generic `CompletionModel` so it
    /// can be passed to `runAgent`, `batchComplete`, or further decorators.
    #[wasm_bindgen(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> WasmCompletionModel {
        WasmCompletionModel::from_arc(Arc::clone(&self.inner))
    }

    /// Perform a non-streaming chat completion with retry.
    #[wasm_bindgen]
    pub fn complete(&self, messages: JsValue) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let request = CompletionRequest::new(msgs);
            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Perform a streaming chat completion. Only the initial connection
    /// attempt is retried -- mid-stream failures propagate to the caller.
    #[wasm_bindgen]
    pub fn stream(&self, messages: JsValue, callback: js_sys::Function) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let request = CompletionRequest::new(msgs);
            let mut stream = model
                .stream(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            use futures_util::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(|e| JsValue::from_str(&e.to_string()))?;
                let js_chunk = serde_wasm_bindgen::to_value(&chunk)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let _ = callback.call1(&JsValue::NULL, &js_chunk);
            }
            Ok(JsValue::UNDEFINED)
        })
    }
}

// ---------------------------------------------------------------------------
// CachedCompletionModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` decorator that caches non-streaming responses keyed
/// on a hash of the full request (messages, parameters, tools, model).
///
/// Streaming requests are never cached and always delegate directly to the
/// inner model.
///
/// ```js
/// const cached = new CachedCompletionModel(model, {
///   strategy: 'contentHash',
///   ttlSeconds: 600,
///   maxEntries: 500,
/// });
/// ```
#[wasm_bindgen(js_name = "CachedCompletionModel")]
pub struct WasmCachedCompletionModel {
    inner: Arc<CachedCompletionModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmCachedCompletionModel {}
unsafe impl Sync for WasmCachedCompletionModel {}

#[wasm_bindgen(js_class = "CachedCompletionModel")]
impl WasmCachedCompletionModel {
    /// Wrap `model` with the given cache options.
    ///
    /// `options` is an optional plain JS object with the following fields:
    /// - `strategy` (string) -- `"contentHash"` (default), `"none"`,
    ///   `"anthropicEphemeral"`, or `"auto"`
    /// - `ttlSeconds` (number) -- default 300
    /// - `maxEntries` (number) -- default 1000
    #[wasm_bindgen(constructor)]
    pub fn new(model: &WasmCompletionModel, options: JsValue) -> WasmCachedCompletionModel {
        let config = build_cache_config(&options);
        Self {
            inner: Arc::new(CachedCompletionModel::from_arc(model.inner_arc(), config)),
        }
    }

    /// The default model identifier of the wrapped provider.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// The number of entries currently in the cache.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> u32 {
        u32::try_from(self.inner.len()).unwrap_or(u32::MAX)
    }

    /// Whether the cache is currently empty.
    #[wasm_bindgen(getter, js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Remove all entries from the cache.
    #[wasm_bindgen]
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Convert this cache wrapper into a generic `CompletionModel` so it
    /// can be passed to `runAgent`, `batchComplete`, or further decorators.
    #[wasm_bindgen(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> WasmCompletionModel {
        let inner: Arc<dyn CompletionModel> = self.inner.clone();
        WasmCompletionModel::from_arc(inner)
    }

    /// Perform a non-streaming chat completion through the cache.
    #[wasm_bindgen]
    pub fn complete(&self, messages: JsValue) -> js_sys::Promise {
        let model: Arc<dyn CompletionModel> = self.inner.clone();
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let request = CompletionRequest::new(msgs);
            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Perform a streaming chat completion. Streaming bypasses the cache
    /// entirely and always delegates to the underlying model.
    #[wasm_bindgen]
    pub fn stream(&self, messages: JsValue, callback: js_sys::Function) -> js_sys::Promise {
        let model: Arc<dyn CompletionModel> = self.inner.clone();
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let request = CompletionRequest::new(msgs);
            let mut stream = model
                .stream(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            use futures_util::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(|e| JsValue::from_str(&e.to_string()))?;
                let js_chunk = serde_wasm_bindgen::to_value(&chunk)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let _ = callback.call1(&JsValue::NULL, &js_chunk);
            }
            Ok(JsValue::UNDEFINED)
        })
    }
}

