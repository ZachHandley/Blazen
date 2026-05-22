//! Standalone wrapper classes for the retry / fallback / cache decorators.
//!
//! These mirror the inline `withRetry` / `withFallback` / `withCache`
//! decorator methods on [`JsModel`] but expose each wrapper as
//! a first-class JavaScript class that can be constructed directly:
//!
//! ```javascript
//! const primary = Model.openai();
//! const backup  = Model.anthropic();
//!
//! const fb       = new FallbackModel([primary, backup]);
//! const retried  = new RetryModel(primary, { maxRetries: 5 });
//! const cached   = new CachedModel(primary, { ttlSeconds: 600 });
//! ```
//!
//! All three classes expose the same `complete` / `stream` /
//! `completeWithOptions` / `streamWithOptions` / `modelId` surface as
//! [`JsModel`]. They hold an `Arc<dyn Model>` and
//! delegate every call to it.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::Model;
use blazen_llm::cache::{CacheConfig, CachedModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryConfig, RetryModel};
use blazen_llm::types::{ChatMessage, ModelRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::generated::{JsCacheConfig, JsRetryConfig};
use crate::providers::JsModel;
use crate::providers::model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsModelOptions, JsModelResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Apply [`JsModelOptions`] to a freshly-built [`ModelRequest`].
///
/// Centralised so the per-class `complete_with_options` / `stream_with_options`
/// implementations stay in lock-step with the canonical version on
/// [`JsModel`].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn apply_options(mut request: ModelRequest, options: JsModelOptions) -> ModelRequest {
    if let Some(temp) = options.temperature {
        request.temperature = Some(temp as f32);
    }
    if let Some(max) = options.max_tokens {
        request.max_tokens = Some(max as u32);
    }
    if let Some(top_p) = options.top_p {
        request.top_p = Some(top_p as f32);
    }
    if let Some(model) = options.model {
        request.model = Some(model);
    }
    if let Some(tools) = options.tools {
        request.tools = tools
            .into_iter()
            .map(|t| ToolDefinition {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
            })
            .collect();
    }
    if let Some(fmt) = options.response_format {
        request = request.with_response_format(fmt);
    }
    request
}

/// Resolve a `&JsModel` argument to its inner provider, rejecting
/// subclassed instances that have no concrete Rust provider.
fn require_inner(model: &JsModel, wrapper: &'static str) -> Result<Arc<dyn Model>> {
    model.inner.clone().ok_or_else(|| {
        napi::Error::from_reason(format!(
            "{wrapper} cannot wrap a subclassed Model that has no concrete provider",
        ))
    })
}

/// Run a non-streaming completion against an `Arc<dyn Model>`.
async fn run_complete(
    inner: Arc<dyn Model>,
    messages: Vec<&JsChatMessage>,
    options: Option<JsModelOptions>,
) -> Result<JsModelResponse> {
    let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    let mut request = ModelRequest::new(chat_messages);
    if let Some(opts) = options {
        request = apply_options(request, opts);
    }
    let response = inner.complete(request).await.map_err(llm_error_to_napi)?;
    Ok(build_response(response))
}

/// Run a streaming completion against an `Arc<dyn Model>`,
/// forwarding each chunk to the JavaScript callback.
async fn run_stream(
    inner: Arc<dyn Model>,
    messages: Vec<&JsChatMessage>,
    on_chunk: StreamChunkCallbackTsfn,
    options: Option<JsModelOptions>,
) -> Result<()> {
    let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    let mut request = ModelRequest::new(chat_messages);
    if let Some(opts) = options {
        request = apply_options(request, opts);
    }
    let stream = inner.stream(request).await.map_err(llm_error_to_napi)?;
    let mut stream = std::pin::pin!(stream);
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                on_chunk.call(
                    build_stream_chunk(chunk),
                    ThreadsafeFunctionCallMode::Blocking,
                );
            }
            Err(e) => return Err(napi::Error::from_reason(e.to_string())),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// JsFallbackModel
// ---------------------------------------------------------------------------

/// A completion model that tries multiple providers in order, falling
/// back to the next on transient (retryable) failures.
///
/// Non-retryable errors (auth, validation, content policy) short-circuit
/// immediately so that broken credentials are not masked by the fallback
/// attempt.
///
/// ```javascript
/// const model = new FallbackModel([primary, backup]);
/// const response = await model.complete([ChatMessage.user("hi")]);
/// ```
#[napi(js_name = "FallbackModel")]
pub struct JsFallbackModel {
    inner: Arc<dyn Model>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsFallbackModel {
    /// Create a new fallback model from two or more providers.
    ///
    /// Throws if `models` is empty. A single-element list is allowed but
    /// degenerate -- prefer using the underlying model directly in that
    /// case.
    #[napi(constructor)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(models: Vec<&JsModel>) -> Result<Self> {
        if models.is_empty() {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "FallbackModel requires at least one model",
            ));
        }
        let mut providers: Vec<Arc<dyn Model>> = Vec::with_capacity(models.len());
        for m in &models {
            providers.push(require_inner(m, "FallbackModel")?);
        }
        Ok(Self {
            inner: Arc::new(FallbackModel::new(providers)),
        })
    }

    /// The model id of the primary provider.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion, falling back across providers.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsModelResponse> {
        run_complete(Arc::clone(&self.inner), messages, None).await
    }

    /// Perform a chat completion with options, falling back across providers.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsModelOptions,
    ) -> Result<JsModelResponse> {
        run_complete(Arc::clone(&self.inner), messages, Some(options)).await
    }

    /// Stream a chat completion, falling back across providers on
    /// retryable initial-stream failures.
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, None).await
    }

    /// Stream a chat completion with options.
    #[napi(js_name = "streamWithOptions")]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsModelOptions,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, Some(options)).await
    }

    /// Convert this fallback wrapper into a plain [`JsModel`]
    /// so it can be passed to APIs that expect the base type
    /// (`Agent`, `Batch`, further wrappers, etc.).
    #[napi(js_name = "toModel")]
    pub fn to_model(&self) -> JsModel {
        JsModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// JsRetryModel
// ---------------------------------------------------------------------------

/// A completion model that retries transient failures with exponential
/// backoff.
///
/// ```javascript
/// const model = new RetryModel(
///     Model.openrouter(),
///     { maxRetries: 5, initialDelayMs: 500 },
/// );
/// const response = await model.complete([ChatMessage.user("hi")]);
/// ```
#[napi(js_name = "RetryModel")]
pub struct JsRetryModel {
    inner: Arc<dyn Model>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsRetryModel {
    /// Wrap `model` with retry-on-transient-error behaviour.
    ///
    /// `config` defaults to [`RetryConfig::default()`] (3 retries, 1s
    /// initial delay, 30s cap, jitter on, `Retry-After` honoured) when
    /// omitted.
    #[napi(constructor)]
    pub fn new(model: &JsModel, config: Option<JsRetryConfig>) -> Result<Self> {
        let inner = require_inner(model, "RetryModel")?;
        let retry_config: RetryConfig = config.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(RetryModel::from_arc(inner, retry_config)),
        })
    }

    /// The wrapped model's id.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion with automatic retries.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsModelResponse> {
        run_complete(Arc::clone(&self.inner), messages, None).await
    }

    /// Perform a chat completion with options and automatic retries.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsModelOptions,
    ) -> Result<JsModelResponse> {
        run_complete(Arc::clone(&self.inner), messages, Some(options)).await
    }

    /// Stream a chat completion. Retries the initial request on transient
    /// failures; mid-stream errors are not retried.
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, None).await
    }

    /// Stream a chat completion with options.
    #[napi(js_name = "streamWithOptions")]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsModelOptions,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, Some(options)).await
    }

    /// Convert this retry wrapper into a plain [`JsModel`] so
    /// it can be passed to APIs that expect the base type.
    #[napi(js_name = "toModel")]
    pub fn to_model(&self) -> JsModel {
        JsModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// JsCachedModel
// ---------------------------------------------------------------------------

/// A completion model that caches non-streaming responses keyed on a
/// hash of the full request.
///
/// Streaming requests are never cached and pass straight through to the
/// inner model.
///
/// ```javascript
/// const cached = new CachedModel(
///     Model.openai(),
///     { ttlSeconds: 300, maxEntries: 1000 },
/// );
/// ```
#[napi(js_name = "CachedModel")]
pub struct JsCachedModel {
    inner: Arc<dyn Model>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCachedModel {
    /// Wrap `model` with an in-memory response cache.
    #[napi(constructor)]
    pub fn new(model: &JsModel, config: Option<JsCacheConfig>) -> Result<Self> {
        let inner = require_inner(model, "CachedModel")?;
        let cache_config: CacheConfig = config.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(CachedModel::from_arc(inner, cache_config)),
        })
    }

    /// The wrapped model's id.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion, returning a cached response on a
    /// hit and otherwise delegating to the inner model.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsModelResponse> {
        run_complete(Arc::clone(&self.inner), messages, None).await
    }

    /// Perform a chat completion with options. The full options object
    /// is included in the cache key.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsModelOptions,
    ) -> Result<JsModelResponse> {
        run_complete(Arc::clone(&self.inner), messages, Some(options)).await
    }

    /// Stream a chat completion. Streaming requests bypass the cache
    /// entirely.
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, None).await
    }

    /// Stream a chat completion with options. Streaming requests bypass
    /// the cache entirely.
    #[napi(js_name = "streamWithOptions")]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsModelOptions,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, Some(options)).await
    }

    /// Convert this cache wrapper into a plain [`JsModel`] so
    /// it can be passed to APIs that expect the base type.
    #[napi(js_name = "toModel")]
    pub fn to_model(&self) -> JsModel {
        JsModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}
