//! JavaScript wrapper for LLM completion models.
//!
//! Provides [`JsCompletionModel`] with factory constructors for each
//! supported provider (`OpenAI`, Anthropic, Gemini, etc.), plus decorator
//! methods for retry, fallback, and caching.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::CompletionModel;
use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::types::provider_options::ProviderOptions;
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::generated::{
    JsAzureOptions, JsBedrockOptions, JsCacheConfig, JsFalOptions, JsProviderOptions, JsRetryConfig,
};
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, JsStreamChunk, build_response,
    build_stream_chunk,
};

/// Convert an optional generated `JsProviderOptions` into the typed core
/// [`ProviderOptions`]. Returns the default (all-`None`) options when no
/// dict is provided.
fn js_to_provider_options(options: Option<JsProviderOptions>) -> ProviderOptions {
    options.map(Into::into).unwrap_or_default()
}

/// Stream callback: takes a typed `JsStreamChunk`, returns nothing meaningful.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
pub(crate) type StreamChunkCallbackTsfn =
    ThreadsafeFunction<JsStreamChunk, Unknown<'static>, JsStreamChunk, napi::Status, false, true>;

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
    //
    // Each factory converts the generated NAPI options struct into the typed
    // core options struct (via the auto-generated `From` impls in build.rs)
    // and delegates to the provider's `from_options()` method. The actual
    // construction logic lives in `blazen-llm` (see
    // `crates/blazen-llm/src/providers/mod.rs::impl_simple_from_options`).

    /// Create an `OpenAI` completion model.
    #[napi(factory)]
    pub fn openai(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::openai::OpenAiProvider::from_options(
                api_key,
                js_to_provider_options(options),
            )),
        }
    }

    /// Create an Anthropic completion model.
    #[napi(factory)]
    pub fn anthropic(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::anthropic::AnthropicProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create a Google Gemini completion model.
    #[napi(factory)]
    pub fn gemini(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::gemini::GeminiProvider::from_options(
                api_key,
                js_to_provider_options(options),
            )),
        }
    }

    /// Create an Azure `OpenAI` completion model.
    #[napi(factory)]
    pub fn azure(api_key: String, options: JsAzureOptions) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::azure::AzureOpenAiProvider::from_options(
                    api_key,
                    options.into(),
                ),
            ),
        }
    }

    /// Create a fal.ai completion model.
    ///
    /// `options` optionally configures the LLM model, endpoint family,
    /// enterprise tier, and modality auto-routing. Defaults to the
    /// OpenAI-compatible chat-completions endpoint.
    #[napi(factory)]
    pub fn fal(api_key: String, options: Option<JsFalOptions>) -> Self {
        let opts: blazen_llm::types::provider_options::FalOptions =
            options.map(Into::into).unwrap_or_default();
        Self {
            inner: Arc::new(blazen_llm::providers::fal::FalProvider::from_options(
                api_key, opts,
            )),
        }
    }

    /// Create an `OpenRouter` completion model.
    #[napi(factory)]
    pub fn openrouter(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openrouter::OpenRouterProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create a Groq completion model.
    #[napi(factory)]
    pub fn groq(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::groq::GroqProvider::from_options(
                api_key,
                js_to_provider_options(options),
            )),
        }
    }

    /// Create a Together AI completion model.
    #[napi(factory)]
    pub fn together(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::together::TogetherProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create a Mistral AI completion model.
    #[napi(factory)]
    pub fn mistral(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::mistral::MistralProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create a `DeepSeek` completion model.
    #[napi(factory)]
    pub fn deepseek(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::deepseek::DeepSeekProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create a Fireworks AI completion model.
    #[napi(factory)]
    pub fn fireworks(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::fireworks::FireworksProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create a Perplexity completion model.
    #[napi(factory)]
    pub fn perplexity(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::perplexity::PerplexityProvider::from_options(
                    api_key,
                    js_to_provider_options(options),
                ),
            ),
        }
    }

    /// Create an xAI (Grok) completion model.
    #[napi(factory)]
    pub fn xai(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::xai::XaiProvider::from_options(
                api_key,
                js_to_provider_options(options),
            )),
        }
    }

    /// Create a Cohere completion model.
    #[napi(factory)]
    pub fn cohere(api_key: String, options: Option<JsProviderOptions>) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::cohere::CohereProvider::from_options(
                api_key,
                js_to_provider_options(options),
            )),
        }
    }

    /// Create an AWS Bedrock completion model.
    #[napi(factory)]
    pub fn bedrock(api_key: String, options: JsBedrockOptions) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::bedrock::BedrockProvider::from_options(
                    api_key,
                    options.into(),
                ),
            ),
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
        // `config.map(Into::into)` uses the auto-generated `From<JsRetryConfig>`
        // impl for explicit configs, and falls back to `RetryConfig::default()`
        // when no config was supplied. This avoids requiring `Default` on the
        // generated `JsRetryConfig` type.
        let retry_config: RetryConfig = config.map(Into::into).unwrap_or_default();
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
        // See `with_retry` for why we use `config.map(Into::into).unwrap_or_default()`.
        let cache_config: CacheConfig = config.map(Into::into).unwrap_or_default();
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
