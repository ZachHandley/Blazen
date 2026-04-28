//! Standalone wrapper classes for the retry / fallback / cache decorators.
//!
//! These mirror the inline `withRetry` / `withFallback` / `withCache`
//! decorator methods on [`JsCompletionModel`] but expose each wrapper as
//! a first-class JavaScript class that can be constructed directly:
//!
//! ```javascript
//! const primary = CompletionModel.openai();
//! const backup  = CompletionModel.anthropic();
//!
//! const fb       = new FallbackModel([primary, backup]);
//! const retried  = new RetryCompletionModel(primary, { maxRetries: 5 });
//! const cached   = new CachedCompletionModel(primary, { ttlSeconds: 600 });
//! ```
//!
//! All three classes expose the same `complete` / `stream` /
//! `completeWithOptions` / `streamWithOptions` / `modelId` surface as
//! [`JsCompletionModel`]. They hold an `Arc<dyn CompletionModel>` and
//! delegate every call to it.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::CompletionModel;
use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::generated::{JsCacheConfig, JsRetryConfig};
use crate::providers::JsCompletionModel;
use crate::providers::completion_model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Apply [`JsCompletionOptions`] to a freshly-built [`CompletionRequest`].
///
/// Centralised so the per-class `complete_with_options` / `stream_with_options`
/// implementations stay in lock-step with the canonical version on
/// [`JsCompletionModel`].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn apply_options(
    mut request: CompletionRequest,
    options: JsCompletionOptions,
) -> CompletionRequest {
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

/// Resolve a `&JsCompletionModel` argument to its inner provider, rejecting
/// subclassed instances that have no concrete Rust provider.
fn require_inner(
    model: &JsCompletionModel,
    wrapper: &'static str,
) -> Result<Arc<dyn CompletionModel>> {
    model.inner.clone().ok_or_else(|| {
        napi::Error::from_reason(format!(
            "{wrapper} cannot wrap a subclassed CompletionModel that has no concrete provider",
        ))
    })
}

/// Run a non-streaming completion against an `Arc<dyn CompletionModel>`.
async fn run_complete(
    inner: Arc<dyn CompletionModel>,
    messages: Vec<&JsChatMessage>,
    options: Option<JsCompletionOptions>,
) -> Result<JsCompletionResponse> {
    let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    let mut request = CompletionRequest::new(chat_messages);
    if let Some(opts) = options {
        request = apply_options(request, opts);
    }
    let response = inner.complete(request).await.map_err(llm_error_to_napi)?;
    Ok(build_response(response))
}

/// Run a streaming completion against an `Arc<dyn CompletionModel>`,
/// forwarding each chunk to the JavaScript callback.
async fn run_stream(
    inner: Arc<dyn CompletionModel>,
    messages: Vec<&JsChatMessage>,
    on_chunk: StreamChunkCallbackTsfn,
    options: Option<JsCompletionOptions>,
) -> Result<()> {
    let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    let mut request = CompletionRequest::new(chat_messages);
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
    inner: Arc<dyn CompletionModel>,
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
    pub fn new(models: Vec<&JsCompletionModel>) -> Result<Self> {
        if models.is_empty() {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "FallbackModel requires at least one model",
            ));
        }
        let mut providers: Vec<Arc<dyn CompletionModel>> = Vec::with_capacity(models.len());
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
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        run_complete(Arc::clone(&self.inner), messages, None).await
    }

    /// Perform a chat completion with options, falling back across providers.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
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
        options: JsCompletionOptions,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, Some(options)).await
    }

    /// Convert this fallback wrapper into a plain [`JsCompletionModel`]
    /// so it can be passed to APIs that expect the base type
    /// (`Agent`, `Batch`, further wrappers, etc.).
    #[napi(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> JsCompletionModel {
        JsCompletionModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// JsRetryCompletionModel
// ---------------------------------------------------------------------------

/// A completion model that retries transient failures with exponential
/// backoff.
///
/// ```javascript
/// const model = new RetryCompletionModel(
///     CompletionModel.openrouter(),
///     { maxRetries: 5, initialDelayMs: 500 },
/// );
/// const response = await model.complete([ChatMessage.user("hi")]);
/// ```
#[napi(js_name = "RetryCompletionModel")]
pub struct JsRetryCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsRetryCompletionModel {
    /// Wrap `model` with retry-on-transient-error behaviour.
    ///
    /// `config` defaults to [`RetryConfig::default()`] (3 retries, 1s
    /// initial delay, 30s cap, jitter on, `Retry-After` honoured) when
    /// omitted.
    #[napi(constructor)]
    pub fn new(model: &JsCompletionModel, config: Option<JsRetryConfig>) -> Result<Self> {
        let inner = require_inner(model, "RetryCompletionModel")?;
        let retry_config: RetryConfig = config.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(RetryCompletionModel::from_arc(inner, retry_config)),
        })
    }

    /// The wrapped model's id.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion with automatic retries.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        run_complete(Arc::clone(&self.inner), messages, None).await
    }

    /// Perform a chat completion with options and automatic retries.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
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
        options: JsCompletionOptions,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, Some(options)).await
    }

    /// Convert this retry wrapper into a plain [`JsCompletionModel`] so
    /// it can be passed to APIs that expect the base type.
    #[napi(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> JsCompletionModel {
        JsCompletionModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// JsCachedCompletionModel
// ---------------------------------------------------------------------------

/// A completion model that caches non-streaming responses keyed on a
/// hash of the full request.
///
/// Streaming requests are never cached and pass straight through to the
/// inner model.
///
/// ```javascript
/// const cached = new CachedCompletionModel(
///     CompletionModel.openai(),
///     { ttlSeconds: 300, maxEntries: 1000 },
/// );
/// ```
#[napi(js_name = "CachedCompletionModel")]
pub struct JsCachedCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCachedCompletionModel {
    /// Wrap `model` with an in-memory response cache.
    #[napi(constructor)]
    pub fn new(model: &JsCompletionModel, config: Option<JsCacheConfig>) -> Result<Self> {
        let inner = require_inner(model, "CachedCompletionModel")?;
        let cache_config: CacheConfig = config.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(CachedCompletionModel::from_arc(inner, cache_config)),
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
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        run_complete(Arc::clone(&self.inner), messages, None).await
    }

    /// Perform a chat completion with options. The full options object
    /// is included in the cache key.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
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
        options: JsCompletionOptions,
    ) -> Result<()> {
        run_stream(Arc::clone(&self.inner), messages, on_chunk, Some(options)).await
    }

    /// Convert this cache wrapper into a plain [`JsCompletionModel`] so
    /// it can be passed to APIs that expect the base type.
    #[napi(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> JsCompletionModel {
        JsCompletionModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}
