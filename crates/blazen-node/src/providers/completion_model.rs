//! JavaScript wrapper for LLM completion models.
//!
//! Provides [`JsCompletionModel`] with factory constructors for each
//! supported provider (`OpenAI`, Anthropic, Gemini, etc.), plus decorator
//! methods for retry, fallback, and caching.

use std::sync::Arc;
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::CompletionModel;
use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::generated::JsFalOptions;
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, JsStreamChunk, build_response,
    build_stream_chunk,
};

/// Stream callback: takes a typed `JsStreamChunk`, returns nothing meaningful.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
pub(crate) type StreamChunkCallbackTsfn =
    ThreadsafeFunction<JsStreamChunk, Unknown<'static>, JsStreamChunk, napi::Status, false, true>;

// ---------------------------------------------------------------------------
// Config objects for decorator methods
// ---------------------------------------------------------------------------

/// Configuration for the `withRetry` decorator.
#[napi(object)]
pub struct JsRetryConfig {
    /// Maximum number of retry attempts (total calls = `maxRetries + 1`).
    #[napi(js_name = "maxRetries")]
    pub max_retries: Option<u32>,
    /// Delay before the first retry, in milliseconds.
    #[napi(js_name = "initialDelayMs")]
    pub initial_delay_ms: Option<u32>,
    /// Upper bound on the computed backoff delay, in milliseconds.
    #[napi(js_name = "maxDelayMs")]
    pub max_delay_ms: Option<u32>,
}

/// Configuration for the `withCache` decorator.
#[napi(object)]
pub struct JsCacheConfig {
    /// How long a cached response remains valid, in seconds.
    #[napi(js_name = "ttlSeconds")]
    pub ttl_seconds: Option<u32>,
    /// Maximum number of entries to keep in the cache.
    #[napi(js_name = "maxEntries")]
    pub max_entries: Option<u32>,
}

// ---------------------------------------------------------------------------
// CompletionModel wrapper
// ---------------------------------------------------------------------------

/// A chat completion model.
///
/// Use the static factory methods to create an instance for your provider:
///
/// ```javascript
/// const model = CompletionModel.openai("sk-...");
/// const response = await model.complete([
///   ChatMessage.user("What is 2 + 2?")
/// ]);
/// ```
#[napi(js_name = "CompletionModel")]
pub struct JsCompletionModel {
    pub(crate) inner: Arc<dyn CompletionModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompletionModel {
    // -----------------------------------------------------------------
    // Provider factory methods
    // -----------------------------------------------------------------

    /// Create an `OpenAI` completion model.
    #[napi(factory)]
    pub fn openai(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::openai::OpenAiProvider::new(api_key);
        if let Some(opts) = options {
            if let Some(m) = opts.model {
                provider = provider.with_model(m);
            }
            if let Some(url) = opts.base_url {
                provider = provider.with_base_url(url);
            }
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an Anthropic completion model.
    #[napi(factory)]
    pub fn anthropic(
        api_key: String,
        options: Option<crate::generated::JsProviderOptions>,
    ) -> Self {
        let mut provider = blazen_llm::providers::anthropic::AnthropicProvider::new(api_key);
        if let Some(opts) = options {
            if let Some(m) = opts.model {
                provider = provider.with_model(m);
            }
            if let Some(url) = opts.base_url {
                provider = provider.with_base_url(url);
            }
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Google Gemini completion model.
    #[napi(factory)]
    pub fn gemini(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::gemini::GeminiProvider::new(api_key);
        if let Some(opts) = options {
            if let Some(m) = opts.model {
                provider = provider.with_model(m);
            }
            if let Some(url) = opts.base_url {
                provider = provider.with_base_url(url);
            }
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an Azure `OpenAI` completion model.
    #[napi(factory)]
    pub fn azure(api_key: String, options: crate::generated::JsAzureOptions) -> Self {
        let mut provider = blazen_llm::providers::azure::AzureOpenAiProvider::new(
            api_key,
            options.resource_name,
            options.deployment_name,
        );
        if let Some(v) = options.api_version {
            provider = provider.with_api_version(v);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a fal.ai completion model.
    ///
    /// `options` optionally configures the LLM model, endpoint family,
    /// enterprise tier, and modality auto-routing. Defaults to the
    /// OpenAI-compatible chat-completions endpoint.
    #[napi(factory)]
    pub fn fal(api_key: String, options: Option<JsFalOptions>) -> Self {
        use blazen_llm::providers::fal::FalLlmEndpoint;
        use blazen_llm::types::provider_options::FalLlmEndpointKind;

        let mut provider = blazen_llm::providers::fal::FalProvider::new(api_key);
        if let Some(opts) = options {
            if let Some(model) = opts.model {
                provider = provider.with_llm_model(model);
            }
            if let Some(base_url) = opts.base_url {
                provider = provider.with_base_url(base_url);
            }
            let enterprise = opts.enterprise;
            if let Some(ep) = opts.endpoint {
                let kind: FalLlmEndpointKind = ep.into();
                let endpoint = match kind {
                    FalLlmEndpointKind::OpenAiChat => FalLlmEndpoint::OpenAiChat,
                    FalLlmEndpointKind::OpenAiResponses => FalLlmEndpoint::OpenAiResponses,
                    FalLlmEndpointKind::OpenAiEmbeddings => FalLlmEndpoint::OpenAiEmbeddings,
                    FalLlmEndpointKind::OpenRouter => FalLlmEndpoint::OpenRouter { enterprise },
                    FalLlmEndpointKind::AnyLlm => FalLlmEndpoint::AnyLlm { enterprise },
                };
                provider = provider.with_llm_endpoint(endpoint);
            } else if enterprise {
                provider = provider.with_enterprise();
            }
            provider = provider.with_auto_route_modality(opts.auto_route_modality);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an `OpenRouter` completion model.
    #[napi(factory)]
    pub fn openrouter(
        api_key: String,
        options: Option<crate::generated::JsProviderOptions>,
    ) -> Self {
        let mut provider = blazen_llm::providers::openrouter::OpenRouterProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Groq completion model.
    #[napi(factory)]
    pub fn groq(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::groq::GroqProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Together AI completion model.
    #[napi(factory)]
    pub fn together(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::together::TogetherProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Mistral AI completion model.
    #[napi(factory)]
    pub fn mistral(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::mistral::MistralProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a `DeepSeek` completion model.
    #[napi(factory)]
    pub fn deepseek(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::deepseek::DeepSeekProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Fireworks AI completion model.
    #[napi(factory)]
    pub fn fireworks(
        api_key: String,
        options: Option<crate::generated::JsProviderOptions>,
    ) -> Self {
        let mut provider = blazen_llm::providers::fireworks::FireworksProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Perplexity completion model.
    #[napi(factory)]
    pub fn perplexity(
        api_key: String,
        options: Option<crate::generated::JsProviderOptions>,
    ) -> Self {
        let mut provider = blazen_llm::providers::perplexity::PerplexityProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an xAI (Grok) completion model.
    #[napi(factory)]
    pub fn xai(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::xai::XaiProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Cohere completion model.
    #[napi(factory)]
    pub fn cohere(api_key: String, options: Option<crate::generated::JsProviderOptions>) -> Self {
        let mut provider = blazen_llm::providers::cohere::CohereProvider::new(api_key);
        if let Some(opts) = options
            && let Some(m) = opts.model
        {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an AWS Bedrock completion model.
    #[napi(factory)]
    pub fn bedrock(api_key: String, options: crate::generated::JsBedrockOptions) -> Self {
        let mut provider =
            blazen_llm::providers::bedrock::BedrockProvider::new(api_key, &options.region);
        if let Some(m) = options.model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    // -----------------------------------------------------------------
    // Model configuration
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Decorator methods (retry, cache, fallback)
    // -----------------------------------------------------------------

    /// Wrap this model with automatic retry on transient failures.
    ///
    /// ```javascript
    /// const model = CompletionModel.openrouter(key);
    /// const withRetry = model.withRetry({ maxRetries: 3, initialDelayMs: 1000 });
    /// ```
    #[napi(js_name = "withRetry")]
    #[must_use]
    pub fn with_retry(&self, config: Option<JsRetryConfig>) -> JsCompletionModel {
        let cfg = config.unwrap_or(JsRetryConfig {
            max_retries: None,
            initial_delay_ms: None,
            max_delay_ms: None,
        });
        let retry_config = RetryConfig {
            max_retries: cfg.max_retries.unwrap_or(3),
            initial_delay: Duration::from_millis(u64::from(cfg.initial_delay_ms.unwrap_or(1000))),
            max_delay: Duration::from_millis(u64::from(cfg.max_delay_ms.unwrap_or(30_000))),
            honor_retry_after: true,
            jitter: true,
        };
        JsCompletionModel {
            inner: Arc::new(RetryCompletionModel::from_arc(
                Arc::clone(&self.inner),
                retry_config,
            )),
        }
    }

    /// Wrap this model with an in-memory response cache.
    ///
    /// Streaming requests are never cached and always delegate directly to the
    /// underlying model.
    ///
    /// ```javascript
    /// const cached = model.withCache({ ttlSeconds: 300, maxEntries: 1000 });
    /// ```
    #[napi(js_name = "withCache")]
    #[must_use]
    pub fn with_cache(&self, config: Option<JsCacheConfig>) -> JsCompletionModel {
        let cfg = config.unwrap_or(JsCacheConfig {
            ttl_seconds: None,
            max_entries: None,
        });
        let cache_config = CacheConfig {
            ttl: Duration::from_secs(u64::from(cfg.ttl_seconds.unwrap_or(300))),
            max_entries: cfg.max_entries.unwrap_or(1000) as usize,
            ..CacheConfig::default()
        };
        JsCompletionModel {
            inner: Arc::new(CachedCompletionModel::from_arc(
                Arc::clone(&self.inner),
                cache_config,
            )),
        }
    }

    /// Create a fallback model that tries multiple providers in order.
    ///
    /// When the primary provider fails with a transient error (rate limit,
    /// timeout, server error) the request is automatically forwarded to the
    /// next provider. Non-retryable errors short-circuit immediately.
    ///
    /// ```javascript
    /// const model = CompletionModel.withFallback([modelA, modelB]);
    /// ```
    #[napi(factory, js_name = "withFallback")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn with_fallback(models: Vec<&JsCompletionModel>) -> Result<JsCompletionModel> {
        if models.is_empty() {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "withFallback requires at least one model",
            ));
        }
        let providers: Vec<Arc<dyn CompletionModel>> =
            models.iter().map(|m| Arc::clone(&m.inner)).collect();
        Ok(JsCompletionModel {
            inner: Arc::new(FallbackModel::new(providers)),
        })
    }

    // -----------------------------------------------------------------
    // Completion methods
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    ///
    /// Messages should be an array of `ChatMessage` instances.
    ///
    /// Returns a typed response with `content`, `toolCalls`, `usage`, `model`,
    /// and `finishReason` fields.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        Ok(build_response(response))
    }

    /// Perform a chat completion with additional options.
    ///
    /// Options object may include:
    /// - `temperature` (number): Sampling temperature (0.0 - 2.0)
    /// - `maxTokens` (number): Maximum tokens to generate
    /// - `topP` (number): Nucleus sampling parameter
    /// - `model` (string): Override the default model
    /// - `tools` (array): Tool definitions for function calling
    #[napi(js_name = "completeWithOptions")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);

        // Apply options.
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

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        Ok(build_response(response))
    }

    /// Stream a chat completion.
    ///
    /// The `onChunk` callback receives each chunk as a typed `StreamChunk` with
    /// `delta`, `finishReason`, and `toolCalls` fields.
    ///
    /// ```javascript
    /// await model.stream(
    ///   [ChatMessage.user("Tell me a story")],
    ///   (chunk) => { if (chunk.delta) process.stdout.write(chunk.delta); }
    /// );
    /// ```
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let stream = self
            .inner
            .stream(request)
            .await
            .map_err(llm_error_to_napi)?;

        let mut stream = std::pin::pin!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    on_chunk.call(
                        build_stream_chunk(chunk),
                        ThreadsafeFunctionCallMode::Blocking,
                    );
                }
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }

    /// Stream a chat completion with additional options.
    ///
    /// Options object may include:
    /// - `temperature` (number): Sampling temperature (0.0 - 2.0)
    /// - `maxTokens` (number): Maximum tokens to generate
    /// - `topP` (number): Nucleus sampling parameter
    /// - `model` (string): Override the default model
    /// - `tools` (array): Tool definitions for function calling
    #[napi(js_name = "streamWithOptions")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsCompletionOptions,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);

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

        let stream = self
            .inner
            .stream(request)
            .await
            .map_err(llm_error_to_napi)?;

        let mut stream = std::pin::pin!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    on_chunk.call(
                        build_stream_chunk(chunk),
                        ThreadsafeFunctionCallMode::Blocking,
                    );
                }
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }
}
