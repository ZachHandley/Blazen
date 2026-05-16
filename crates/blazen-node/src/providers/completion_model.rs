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
use blazen_llm::CustomProvider as CustomProviderTrait;
use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::providers::custom::CustomProviderHandle;
use blazen_llm::providers::openai_compat::OpenAiCompatConfig;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::types::provider_options::ProviderOptions;
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::{blazen_error_to_napi, llm_error_to_napi};
use crate::generated::{
    JsAzureOptions, JsBedrockOptions, JsCacheConfig, JsFalOptions, JsProviderOptions, JsRetryConfig,
};
use crate::providers::custom::JsCustomProviderAdapter;
use crate::providers::openai_compat::JsOpenAiCompatConfig;
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

// ---------------------------------------------------------------------------
// MistralRs options (manual mirror -- feature-gated)
// ---------------------------------------------------------------------------

/// Options for the local mistral.rs LLM backend.
///
/// `modelId` is required (`HuggingFace` model ID or local GGUF path).
/// All other fields are optional.
///
/// ```javascript
/// const model = CompletionModel.mistralrs({
///   modelId: "mistralai/Mistral-7B-Instruct-v0.3",
///   device: "cuda:0",
///   quantization: "q4_k_m",
/// });
/// ```
#[cfg(feature = "mistralrs")]
#[napi(object)]
pub struct JsMistralRsOptions {
    /// `HuggingFace` model ID or local GGUF path.
    #[napi(js_name = "modelId")]
    pub model_id: String,
    /// Quantization format string (e.g. `"q4_k_m"`, `"f16"`, `"gptq-4bit"`).
    pub quantization: Option<String>,
    /// Hardware device string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub device: Option<String>,
    /// Maximum context length in tokens.
    #[napi(js_name = "contextLength")]
    pub context_length: Option<u32>,
    /// Maximum batch size for concurrent requests.
    #[napi(js_name = "maxBatchSize")]
    pub max_batch_size: Option<u32>,
    /// Jinja2 chat template override.
    #[napi(js_name = "chatTemplate")]
    pub chat_template: Option<String>,
    /// Path to cache downloaded models.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

#[cfg(feature = "mistralrs")]
impl From<JsMistralRsOptions> for blazen_llm::MistralRsOptions {
    fn from(val: JsMistralRsOptions) -> Self {
        Self {
            model_id: val.model_id,
            quantization: val.quantization,
            device: val.device,
            context_length: val.context_length.map(|v| v as usize),
            max_batch_size: val.max_batch_size.map(|v| v as usize),
            chat_template: val.chat_template,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
            // Vision input is not yet surfaced through the Node binding.
            // Users must construct `MistralRsOptions` directly in Rust to
            // enable vision mode.
            vision: false,
        }
    }
}

/// Stream callback: takes a typed `JsStreamChunk`, returns nothing meaningful.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
pub(crate) type StreamChunkCallbackTsfn =
    ThreadsafeFunction<JsStreamChunk, Unknown<'static>, JsStreamChunk, napi::Status, false, true>;

// ---------------------------------------------------------------------------
// CompletionModel wrapper
// ---------------------------------------------------------------------------

/// Configuration for subclassed `CompletionModel` instances.
///
/// When extending `CompletionModel` from JavaScript/TypeScript, pass this
/// to `super()` so the base class can report `modelId` and other metadata
/// without a concrete provider.
///
/// ```javascript
/// class MyLLM extends CompletionModel {
///   constructor() {
///     super({ modelId: "my-custom-model", contextLength: 8192 });
///   }
/// }
/// ```
#[napi(object)]
pub struct CompletionModelConfig {
    /// Model identifier (e.g. `"my-org/custom-llama"`).
    #[napi(js_name = "modelId")]
    pub model_id: Option<String>,
    /// Maximum context window in tokens.
    #[napi(js_name = "contextLength")]
    pub context_length: Option<u32>,
    /// Base URL for HTTP-based providers.
    #[napi(js_name = "baseUrl")]
    pub base_url: Option<String>,
    /// Estimated memory footprint in bytes when loaded (host RAM if
    /// the provider targets the CPU, GPU VRAM otherwise).
    #[napi(js_name = "memoryEstimateBytes")]
    pub memory_estimate_bytes: Option<u32>,
    /// Maximum output tokens the model supports.
    #[napi(js_name = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
}

/// A chat completion model.
///
/// Use the static factory methods to create an instance for your provider:
///
/// ```javascript
/// const model = CompletionModel.openai({ apiKey: "sk-..." });
/// const response = await model.complete([
///   ChatMessage.user("What is 2 + 2?")
/// ]);
/// ```
///
/// Or extend the class to implement a custom provider:
///
/// ```javascript
/// class MyLLM extends CompletionModel {
///   constructor() {
///     super({ modelId: "my-custom-model" });
///   }
///   async complete(messages) { /* ... */ }
/// }
/// ```
#[napi(js_name = "CompletionModel")]
pub struct JsCompletionModel {
    /// The underlying Rust `CompletionModel` implementation.
    /// `None` when the instance was constructed by a JavaScript subclass
    /// (via `new CompletionModel(config)` / `super(config)`).
    pub(crate) inner: Option<Arc<dyn CompletionModel>>,
    /// Present iff the underlying provider is a local in-process model
    /// (mistral.rs, llama.cpp, candle) that implements
    /// [`blazen_llm::LocalModel`]. `None` for remote HTTP providers and
    /// subclassed instances.
    pub(crate) local_model: Option<Arc<dyn blazen_llm::LocalModel>>,
    /// Provider configuration metadata, used by subclassed instances to
    /// report `modelId` and other properties without a concrete inner
    /// provider.
    pub(crate) config: Option<blazen_llm::ProviderConfig>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompletionModel {
    // -----------------------------------------------------------------
    // Constructor (for JavaScript subclasses)
    // -----------------------------------------------------------------

    /// Construct a base `CompletionModel`.
    ///
    /// Called by JavaScript subclasses via `super(config)`. The `config`
    /// parameter is optional and carries metadata such as `modelId`.
    ///
    /// Instances created this way have no inner Rust provider -- calling
    /// `complete()` or `stream()` without overriding them in the subclass
    /// will throw.
    #[napi(constructor)]
    pub fn new(config: Option<CompletionModelConfig>) -> Self {
        Self {
            inner: None,
            local_model: None,
            config: config.map(|c| blazen_llm::ProviderConfig {
                model_id: c.model_id,
                context_length: c.context_length.map(u64::from),
                base_url: c.base_url,
                memory_estimate_bytes: c.memory_estimate_bytes.map(u64::from),
                max_output_tokens: c.max_output_tokens.map(u64::from),
                ..Default::default()
            }),
        }
    }

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
    pub fn openai(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::openai::OpenAiProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an Anthropic completion model.
    #[napi(factory)]
    pub fn anthropic(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::anthropic::AnthropicProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Google Gemini completion model.
    #[napi(factory)]
    pub fn gemini(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::gemini::GeminiProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an Azure `OpenAI` completion model.
    #[napi(factory)]
    pub fn azure(options: JsAzureOptions) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::azure::AzureOpenAiProvider::from_options(options.into())
                    .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a fal.ai completion model.
    ///
    /// `options` optionally configures the LLM model, endpoint family,
    /// enterprise tier, and modality auto-routing. Defaults to the
    /// OpenAI-compatible chat-completions endpoint.
    #[napi(factory)]
    pub fn fal(options: Option<JsFalOptions>) -> Result<Self> {
        let opts: blazen_llm::types::provider_options::FalOptions =
            options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::fal::FalProvider::from_options(opts)
                    .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an `OpenRouter` completion model.
    #[napi(factory)]
    pub fn openrouter(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::openrouter::OpenRouterProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Groq completion model.
    #[napi(factory)]
    pub fn groq(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::groq::GroqProvider::from_options(js_to_provider_options(
                    options,
                ))
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Together AI completion model.
    #[napi(factory)]
    pub fn together(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::together::TogetherProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Mistral AI completion model.
    #[napi(factory)]
    pub fn mistral(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::mistral::MistralProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a `DeepSeek` completion model.
    #[napi(factory)]
    pub fn deepseek(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::deepseek::DeepSeekProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Fireworks AI completion model.
    #[napi(factory)]
    pub fn fireworks(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::fireworks::FireworksProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Perplexity completion model.
    #[napi(factory)]
    pub fn perplexity(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::perplexity::PerplexityProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an xAI (Grok) completion model.
    #[napi(factory)]
    pub fn xai(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::xai::XaiProvider::from_options(js_to_provider_options(
                    options,
                ))
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Cohere completion model.
    #[napi(factory)]
    pub fn cohere(options: Option<JsProviderOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::cohere::CohereProvider::from_options(
                    js_to_provider_options(options),
                )
                .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an AWS Bedrock completion model.
    #[napi(factory)]
    pub fn bedrock(options: JsBedrockOptions) -> Result<Self> {
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::bedrock::BedrockProvider::from_options(options.into())
                    .map_err(blazen_error_to_napi)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a local Ollama completion model.
    ///
    /// Talks to a running Ollama server (defaults to `http://host:port/v1`).
    /// No API key is required.
    ///
    /// ```javascript
    /// const model = CompletionModel.ollama("localhost", 11434, "llama3.1:8b");
    /// ```
    #[napi(factory)]
    pub fn ollama(host: String, port: u16, model: String) -> Result<Self> {
        let provider = blazen_llm::ollama(host, port, model);
        Ok(Self {
            inner: Some(Arc::new(provider)),
            local_model: None,
            config: None,
        })
    }

    /// Create a local LM Studio completion model.
    ///
    /// Talks to a running LM Studio server's OpenAI-compatible endpoint.
    ///
    /// ```javascript
    /// const model = CompletionModel.lmStudio("localhost", 1234, "my-model");
    /// ```
    #[napi(factory, js_name = "lmStudio")]
    pub fn lm_studio(host: String, port: u16, model: String) -> Result<Self> {
        let provider = blazen_llm::lm_studio(host, port, model);
        Ok(Self {
            inner: Some(Arc::new(provider)),
            local_model: None,
            config: None,
        })
    }

    /// Create a generic OpenAI-compatible completion model.
    ///
    /// Drives any OpenAI-compatible chat-completions endpoint with the
    /// supplied [`JsOpenAiCompatConfig`].
    ///
    /// ```javascript
    /// const model = CompletionModel.openaiCompat("my-host", {
    ///   providerName: "my-host",
    ///   baseUrl: "https://api.example.com/v1",
    ///   apiKey: "sk-...",
    ///   defaultModel: "my-model",
    /// });
    /// ```
    #[napi(factory, js_name = "openaiCompat")]
    pub fn openai_compat(provider_id: String, config: JsOpenAiCompatConfig) -> Result<Self> {
        let cfg: OpenAiCompatConfig = config.into();
        let provider = blazen_llm::openai_compat(provider_id, cfg);
        Ok(Self {
            inner: Some(Arc::new(provider)),
            local_model: None,
            config: None,
        })
    }

    /// Create a fully user-defined completion model backed by a JavaScript
    /// host object.
    ///
    /// `hostObject` must expose Blazen capability methods (e.g.
    /// `complete`, `stream`) using the camelCase trait-method names. The
    /// optional `providerId` is used for logging; defaults to `"custom"`.
    ///
    /// ```javascript
    /// class MyProvider {
    ///   async complete(request) { /* ... */ }
    /// }
    /// const model = CompletionModel.custom(new MyProvider(), "my-provider");
    /// ```
    #[napi(factory)]
    pub fn custom(host_object: Object<'_>, provider_id: Option<String>) -> Result<Self> {
        let id = provider_id.unwrap_or_else(|| "custom".to_owned());
        let adapter: Arc<dyn CustomProviderTrait> =
            Arc::new(JsCustomProviderAdapter::from_host_object(id, &host_object)?);
        let handle = CustomProviderHandle::new(adapter);
        Ok(Self {
            inner: Some(Arc::new(handle)),
            local_model: None,
            config: None,
        })
    }

    // -----------------------------------------------------------------
    // Model configuration
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        if let Some(ref inner) = self.inner {
            inner.model_id().to_owned()
        } else {
            self.config
                .as_ref()
                .and_then(|c| c.model_id.clone())
                .unwrap_or_default()
        }
    }

    // -----------------------------------------------------------------
    // Decorator methods (retry, cache, fallback)
    // -----------------------------------------------------------------

    /// Wrap this model with automatic retry on transient failures.
    ///
    /// ```javascript
    /// const model = CompletionModel.openrouter({ apiKey: key });
    /// const withRetry = model.withRetry({ maxRetries: 3, initialDelayMs: 1000 });
    /// ```
    #[napi(js_name = "withRetry")]
    pub fn with_retry(&self, config: Option<JsRetryConfig>) -> Result<JsCompletionModel> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "withRetry() is not supported on subclassed CompletionModel instances",
            )
        })?;
        // `config.map(Into::into)` uses the auto-generated `From<JsRetryConfig>`
        // impl for explicit configs, and falls back to `RetryConfig::default()`
        // when no config was supplied. This avoids requiring `Default` on the
        // generated `JsRetryConfig` type.
        let retry_config: RetryConfig = config.map(Into::into).unwrap_or_default();
        Ok(JsCompletionModel {
            inner: Some(Arc::new(RetryCompletionModel::from_arc(
                Arc::clone(inner),
                retry_config,
            ))),
            // Retry wraps a single provider; forwarding load/unload to the
            // underlying local model is still meaningful.
            local_model: self.local_model.clone(),
            config: None,
        })
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
    pub fn with_cache(&self, config: Option<JsCacheConfig>) -> Result<JsCompletionModel> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "withCache() is not supported on subclassed CompletionModel instances",
            )
        })?;
        // See `with_retry` for why we use `config.map(Into::into).unwrap_or_default()`.
        let cache_config: CacheConfig = config.map(Into::into).unwrap_or_default();
        Ok(JsCompletionModel {
            inner: Some(Arc::new(CachedCompletionModel::from_arc(
                Arc::clone(inner),
                cache_config,
            ))),
            // Cache wraps a single provider; forwarding load/unload to the
            // underlying local model is still meaningful.
            local_model: self.local_model.clone(),
            config: None,
        })
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
        let mut providers: Vec<Arc<dyn CompletionModel>> = Vec::with_capacity(models.len());
        for m in &models {
            let inner = m.inner.as_ref().ok_or_else(|| {
                napi::Error::from_reason(
                    "withFallback() is not supported on subclassed CompletionModel instances",
                )
            })?;
            providers.push(Arc::clone(inner));
        }
        Ok(JsCompletionModel {
            inner: Some(Arc::new(FallbackModel::new(providers))),
            // Fallback combines heterogeneous providers (potentially a mix of
            // local and remote). There is no single `LocalModel` to forward
            // `load`/`unload` to, so callers must manage lifecycle on the
            // component models before combining them via `withFallback`.
            local_model: None,
            config: None,
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
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("subclass must override complete()"))?
            .clone();

        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let response = inner.complete(request).await.map_err(llm_error_to_napi)?;

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
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| {
                napi::Error::from_reason("subclass must override completeWithOptions()")
            })?
            .clone();

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

        let response = inner.complete(request).await.map_err(llm_error_to_napi)?;

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
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("subclass must override stream()"))?
            .clone();

        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

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
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("subclass must override streamWithOptions()"))?
            .clone();

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
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Local-model control (only meaningful for in-process providers)
    // -----------------------------------------------------------------

    /// Explicitly load the model weights into memory / `VRAM`.
    ///
    /// For remote providers (`OpenAI`, Anthropic, fal, etc.) this throws --
    /// there is no local model to load. For local providers (mistral.rs,
    /// llama.cpp, candle) this triggers the download + load synchronously,
    /// so the next inference call does not pay the startup cost.
    ///
    /// Idempotent: calling `load` on an already-loaded model is a no-op
    /// that resolves immediately.
    #[napi]
    pub async fn load(&self) -> Result<()> {
        match &self.local_model {
            Some(lm) => lm.load().await.map_err(blazen_error_to_napi),
            None => Err(napi::Error::from_reason(
                "load() is only supported for local in-process providers (mistral.rs, llama.cpp, candle)",
            )),
        }
    }

    /// Drop the loaded model and free its memory / `VRAM`.
    ///
    /// For remote providers this throws. For local providers this frees
    /// `GPU` memory so the process can load a different model. Idempotent.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        match &self.local_model {
            Some(lm) => lm.unload().await.map_err(blazen_error_to_napi),
            None => Err(napi::Error::from_reason(
                "unload() is only supported for local in-process providers",
            )),
        }
    }

    /// Whether the model is currently loaded in memory / `VRAM`.
    ///
    /// Always returns `false` for remote providers (they have no local
    /// model to load). Returns the real state for local providers.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> Result<bool> {
        Ok(match &self.local_model {
            Some(lm) => lm.is_loaded().await,
            None => false,
        })
    }

    /// Approximate memory footprint in bytes (host RAM if the
    /// provider targets the CPU, GPU VRAM otherwise), if the
    /// implementation can report it. Returns `null` for remote
    /// providers or for local providers that do not expose memory
    /// usage.
    ///
    /// Note: napi-rs exposes this as a JS `number`. The underlying
    /// [`blazen_llm::LocalModel::memory_bytes`] returns `u64`; we clamp
    /// to `i64::MAX` (~9.2 exabytes) when surfacing through
    /// `JSON`-compatible types, which is effectively lossless for any
    /// realistic footprint.
    #[napi(js_name = "memoryBytes")]
    #[allow(clippy::cast_possible_wrap)]
    pub async fn memory_bytes(&self) -> Result<Option<i64>> {
        Ok(match &self.local_model {
            Some(lm) => lm
                .memory_bytes()
                .await
                .map(|b| i64::try_from(b).unwrap_or(i64::MAX)),
            None => None,
        })
    }
}

// ---------------------------------------------------------------------------
// Feature-gated mistralrs factory (separate impl block required by napi-derive)
// ---------------------------------------------------------------------------

#[cfg(feature = "mistralrs")]
#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompletionModel {
    /// Create a local mistral.rs completion model.
    ///
    /// Runs LLM inference entirely on-device -- no API key required.
    ///
    /// ```javascript
    /// const model = CompletionModel.mistralrs({
    ///   modelId: "mistralai/Mistral-7B-Instruct-v0.3",
    /// });
    /// ```
    #[napi(factory)]
    pub fn mistralrs(options: JsMistralRsOptions) -> Result<Self> {
        let opts: blazen_llm::MistralRsOptions = options.into();
        // `MistralRsProvider` implements both `CompletionModel` and
        // `LocalModel`, so we construct a single concrete `Arc` and clone
        // it into both trait-object storages. Both clones share the same
        // allocation.
        let concrete = Arc::new(
            blazen_llm::MistralRsProvider::from_options(opts)
                .map_err(|e| napi::Error::from_reason(e.to_string()))?,
        );
        Ok(Self {
            inner: Some(concrete.clone()),
            local_model: Some(concrete),
            config: None,
        })
    }
}

// ---------------------------------------------------------------------------
// arc_from_js_model — bridge for run_agent and friends
// ---------------------------------------------------------------------------

/// Extract an `Arc<dyn CompletionModel>` from a [`JsCompletionModel`].
///
/// - If the model carries a Rust-side `inner` (built via one of the
///   factory constructors), the inner `Arc` is cloned and returned.
/// - Otherwise the model was constructed as a JavaScript subclass via
///   `new CompletionModel(config)` / `super(config)` — the
///   subclass-as-CompletionModel path is not wired through to the Rust
///   side here because the JavaScript instance handle is not retained on
///   `JsCompletionModel`. Callers should use the
///   [`JsCompletionModel::custom`] factory which accepts a host object
///   directly, or subclass [`crate::providers::custom::JsCustomProvider`]
///   instead.
pub(crate) fn arc_from_js_model(
    model: &JsCompletionModel,
) -> napi::Result<Arc<dyn CompletionModel>> {
    if let Some(inner) = &model.inner {
        return Ok(Arc::clone(inner));
    }
    Err(napi::Error::from_reason(
        "CompletionModel subclass support in Node is not yet implemented. \
         Use CompletionModel.custom(hostObject) factory instead.",
    ))
}
