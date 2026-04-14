//! `wasm-bindgen` wrapper for [`blazen_llm::CompletionModel`].
//!
//! Exposes factory methods for each provider and `complete()` / `stream()`
//! as async methods that return JavaScript `Promise`s.

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::http::HttpClient;
use blazen_llm::providers::anthropic::AnthropicProvider;
use blazen_llm::providers::azure::AzureOpenAiProvider;
use blazen_llm::providers::fal::FalProvider;
use blazen_llm::providers::gemini::GeminiProvider;
use blazen_llm::providers::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::CompletionRequest;

use crate::chat_message::js_messages_to_vec;
use crate::js_completion::JsCompletionHandler;
use blazen_llm::FetchHttpClient;

// ---------------------------------------------------------------------------
// WebLlmOptions
// ---------------------------------------------------------------------------

/// Options for the `CompletionModel.webLlm()` convenience factory.
///
/// Controls default temperature and max-token limits for models loaded via
/// `@mlc-ai/web-llm`.
#[wasm_bindgen(js_name = "WebLlmOptions")]
pub struct WebLlmOptions {
    /// Sampling temperature. Defaults to `0.7`.
    pub temperature: f32,

    /// Maximum number of tokens to generate. Defaults to `512`.
    #[wasm_bindgen(js_name = "maxTokens")]
    pub max_tokens: u32,
}

#[wasm_bindgen(js_class = "WebLlmOptions")]
impl WebLlmOptions {
    /// Create default options (`temperature: 0.7`, `maxTokens: 512`).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 512,
        }
    }
}

impl Default for WebLlmOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WasmCompletionModel
// ---------------------------------------------------------------------------

/// A provider-agnostic LLM completion model.
///
/// Use the static factory methods to create a model for a specific provider:
///
/// ```js
/// const model = CompletionModel.openai();
/// const model = CompletionModel.openrouter();
/// ```
///
/// All async methods (`complete`, `stream`) return native JavaScript `Promise`s.
#[wasm_bindgen(js_name = "CompletionModel")]
pub struct WasmCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmCompletionModel {}
unsafe impl Sync for WasmCompletionModel {}

impl WasmCompletionModel {
    /// Access the inner `Arc<dyn CompletionModel>` for use by other crate modules.
    pub(crate) fn inner_arc(&self) -> Arc<dyn CompletionModel> {
        Arc::clone(&self.inner)
    }
}

/// Create an `OpenAiCompatProvider` backed by the fetch HTTP client.
fn compat_with_fetch(provider: OpenAiCompatProvider) -> OpenAiCompatProvider {
    let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
    provider.with_http_client(client)
}

#[wasm_bindgen(js_class = "CompletionModel")]
impl WasmCompletionModel {
    // -----------------------------------------------------------------------
    // Factory methods (each provider)
    // -----------------------------------------------------------------------

    /// OpenAI (`gpt-4.1`).
    #[wasm_bindgen]
    pub fn openai() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("openai", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key,
            default_model: "gpt-4.1".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// OpenRouter (400+ models, default `openai/gpt-4.1`).
    #[wasm_bindgen]
    pub fn openrouter() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("openrouter", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "openrouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            api_key,
            default_model: "openai/gpt-4.1".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Groq (fast inference, default `llama-3.3-70b-versatile`).
    #[wasm_bindgen]
    pub fn groq() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("groq", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "groq".into(),
            base_url: "https://api.groq.com/openai/v1".into(),
            api_key,
            default_model: "llama-3.3-70b-versatile".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Together AI (default `meta-llama/Llama-3.3-70B-Instruct-Turbo`).
    #[wasm_bindgen]
    pub fn together() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("together", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "together".into(),
            base_url: "https://api.together.xyz/v1".into(),
            api_key,
            default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Mistral AI (default `mistral-large-latest`).
    #[wasm_bindgen]
    pub fn mistral() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("mistral", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "mistral".into(),
            base_url: "https://api.mistral.ai/v1".into(),
            api_key,
            default_model: "mistral-large-latest".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// DeepSeek (default `deepseek-chat`).
    #[wasm_bindgen]
    pub fn deepseek() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("deepseek", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "deepseek".into(),
            base_url: "https://api.deepseek.com".into(),
            api_key,
            default_model: "deepseek-chat".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: false,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Fireworks AI.
    #[wasm_bindgen]
    pub fn fireworks() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("fireworks", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "fireworks".into(),
            base_url: "https://api.fireworks.ai/inference/v1".into(),
            api_key,
            default_model: "accounts/fireworks/models/llama-v3p3-70b-instruct".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Perplexity (default `sonar-pro`).
    #[wasm_bindgen]
    pub fn perplexity() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("perplexity", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "perplexity".into(),
            base_url: "https://api.perplexity.ai".into(),
            api_key,
            default_model: "sonar-pro".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: false,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// xAI / Grok (default `grok-3`).
    #[wasm_bindgen]
    pub fn xai() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("xai", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "xai".into(),
            base_url: "https://api.x.ai/v1".into(),
            api_key,
            default_model: "grok-3".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Cohere (default `command-a-08-2025`).
    #[wasm_bindgen]
    pub fn cohere() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("cohere", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "cohere".into(),
            base_url: "https://api.cohere.ai/compatibility/v1".into(),
            api_key,
            default_model: "command-a-08-2025".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: false,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// AWS Bedrock via Mantle endpoint.
    #[wasm_bindgen]
    pub fn bedrock(region: &str) -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("bedrock", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let provider = compat_with_fetch(OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "bedrock".into(),
            base_url: format!("https://bedrock-mantle.{region}.api.aws/v1"),
            api_key,
            default_model: "anthropic.claude-sonnet-4-20250514-v1:0".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        }));
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Anthropic (default `claude-sonnet-4-5-20250929`).
    ///
    /// Reads the API key from the `ANTHROPIC_API_KEY` environment variable.
    #[wasm_bindgen]
    pub fn anthropic() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("anthropic", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
        let provider = AnthropicProvider::new_with_client(api_key, client);
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Google Gemini (default `gemini-2.5-flash`).
    ///
    /// Reads the API key from the `GEMINI_API_KEY` environment variable.
    #[wasm_bindgen]
    pub fn gemini() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("gemini", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
        let provider = GeminiProvider::new_with_client(api_key, client);
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Azure OpenAI.
    ///
    /// Reads the API key from the `AZURE_OPENAI_API_KEY` environment variable.
    /// Requires `resourceName` and `deploymentName` — Azure does not have a
    /// single global endpoint.
    #[wasm_bindgen(js_name = "azure")]
    pub fn azure(
        resource_name: &str,
        deployment_name: &str,
    ) -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("azure", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
        let provider =
            AzureOpenAiProvider::new_with_client(api_key, resource_name, deployment_name, client);
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// fal.ai (default `anthropic/claude-sonnet-4.5` via the OpenAI-compatible
    /// chat-completions endpoint).
    ///
    /// Reads the API key from the `FAL_KEY` environment variable.
    #[wasm_bindgen]
    pub fn fal() -> Result<WasmCompletionModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("fal", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
        let provider = FalProvider::new_with_client(api_key, client);
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    // -----------------------------------------------------------------------
    // JS callback handler
    // -----------------------------------------------------------------------

    /// Create a `CompletionModel` backed by JavaScript callback functions.
    ///
    /// This enables local/custom models (e.g. `transformers.js`, WebLLM, or
    /// any JS-side inference library) to participate in the Blazen completion
    /// pipeline.
    ///
    /// - `model_id` identifies the model for logging and routing.
    /// - `complete_handler` is called for non-streaming completions with a
    ///   `CompletionRequest`-shaped JS object; must return (or resolve to)
    ///   a `CompletionResponse`-shaped object.
    /// - `stream_handler` (optional) is called for streaming completions with
    ///   `(request, onChunk)` where `onChunk` is a callback that accepts
    ///   `StreamChunk`-shaped objects. If omitted, `stream()` falls back to
    ///   calling the `complete_handler` and yielding a single chunk.
    /// - `config` (optional) is a plain JS object with provider configuration:
    ///   - `contextLength` (number) -- context window size in tokens
    ///   - `maxOutputTokens` (number) -- max output tokens
    ///   - `vramEstimateBytes` (number) -- estimated VRAM footprint in bytes
    ///   - `pricing` (object) -- `{ inputPerMillion, outputPerMillion }` in USD
    ///
    ///   When `pricing` is provided, it is automatically registered in the
    ///   global pricing registry so that `lookupPricing(modelId)` works.
    ///
    /// ```js
    /// const model = CompletionModel.fromJsHandler(
    ///   'my-local-model',
    ///   async (request) => {
    ///     const text = await myLocalModel.generate(request.messages);
    ///     return {
    ///       content: text, toolCalls: [], citations: [],
    ///       artifacts: [], images: [], audio: [], videos: [],
    ///       model: 'my-local-model', metadata: {},
    ///     };
    ///   },
    ///   undefined,
    ///   { contextLength: 4096, pricing: { inputPerMillion: 0, outputPerMillion: 0 } },
    /// );
    /// ```
    #[wasm_bindgen(js_name = "fromJsHandler")]
    pub fn from_js_handler(
        model_id: String,
        complete_handler: js_sys::Function,
        stream_handler: Option<js_sys::Function>,
        config: JsValue,
    ) -> Self {
        // Extract config fields from the JS object if present.
        if config.is_object() {
            // Register pricing if provided.
            if let Ok(pricing_val) = js_sys::Reflect::get(&config, &JsValue::from_str("pricing")) {
                if pricing_val.is_object() {
                    let input = js_sys::Reflect::get(&pricing_val, &JsValue::from_str("inputPerMillion"))
                        .ok()
                        .and_then(|v| v.as_f64());
                    let output = js_sys::Reflect::get(&pricing_val, &JsValue::from_str("outputPerMillion"))
                        .ok()
                        .and_then(|v| v.as_f64());
                    if let (Some(inp), Some(out)) = (input, output) {
                        blazen_llm::register_pricing(
                            &model_id,
                            blazen_llm::PricingEntry {
                                input_per_million: inp,
                                output_per_million: out,
                            },
                        );
                    }
                }
            }
        }

        let handler = JsCompletionHandler::new(model_id, complete_handler, stream_handler);
        Self {
            inner: Arc::new(handler),
        }
    }

    /// Create a completion model backed by `@mlc-ai/web-llm`.
    ///
    /// The library is **dynamically imported** on the first `complete()` call,
    /// so there is no top-level `await` and the WASM init path stays
    /// synchronous.  If the package is not installed the first call fails
    /// with a clear import error.
    ///
    /// **Requires:** `npm install @mlc-ai/web-llm`
    ///
    /// # Errors
    ///
    /// This factory itself is infallible in practice; errors from the dynamic
    /// import surface at `complete()` call time, not here.
    ///
    /// ```js
    /// const model = CompletionModel.webLlm('Llama-3.1-8B-Instruct-q4f32_1-MLC');
    /// const res = await model.complete([ChatMessage.user('Hello!')]);
    /// console.log(res.content);
    /// ```
    #[allow(clippy::needless_pass_by_value)] // wasm_bindgen requires owned args
    #[wasm_bindgen(js_name = "webLlm")]
    pub fn web_llm(
        model_id: String,
        options: Option<WebLlmOptions>,
    ) -> Result<WasmCompletionModel, JsValue> {
        let temperature = options.as_ref().map_or(0.7, |o| o.temperature);
        let max_tokens = options.as_ref().map_or(512, |o| o.max_tokens);

        // Build a JS function body that lazily imports @mlc-ai/web-llm,
        // creates an MLC engine (cached on globalThis), and performs a
        // chat completion.
        //
        // The cache key includes the model id so different models can coexist.
        let cache_key = format!(
            "__blazen_llm_engine_{}",
            model_id.replace(['/', '-', '.'], "_")
        );

        // Note: we use named `format!` args rather than inline captures because
        // the `{{` / `}}` escaping for JS object literals conflicts with inline
        // capture syntax and makes the template less readable.
        #[allow(clippy::uninlined_format_args)]
        let handler_body = format!(
            r"
            const webllm = await import('@mlc-ai/web-llm');
            if (!globalThis['{cache_key}']) {{
                globalThis['{cache_key}'] = await webllm.CreateMLCEngine('{model_id}');
            }}
            const engine = globalThis['{cache_key}'];
            const messages = request.messages || [];
            const response = await engine.chat.completions.create({{
                messages: messages,
                temperature: request.temperature != null ? request.temperature : {temperature},
                max_tokens: request.max_tokens != null ? request.max_tokens : {max_tokens},
            }});
            const choice = response.choices && response.choices[0];
            return {{
                content: choice && choice.message ? choice.message.content || '' : '',
                toolCalls: [],
                citations: [],
                artifacts: [],
                images: [],
                audio: [],
                videos: [],
                model: '{model_id}',
                metadata: {{}},
                usage: response.usage || null,
            }};
            ",
            cache_key = cache_key,
            model_id = model_id,
            temperature = temperature,
            max_tokens = max_tokens,
        );

        // `new Function("request", body)` creates `function anonymous(request) { body }`.
        let complete_handler =
            js_sys::Function::new_with_args("request", &handler_body);

        let handler = JsCompletionHandler::new(
            model_id,
            complete_handler,
            None, // streaming falls back to single-chunk via complete handler
        );
        Ok(Self {
            inner: Arc::new(handler),
        })
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Override the default model for this provider instance.
    ///
    /// Returns a new `CompletionModel` -- does not mutate in place (WASM
    /// limitation with wasm-bindgen ownership).
    #[wasm_bindgen(js_name = "withModel")]
    pub fn with_model(&self, model: &str) -> Self {
        // We cannot mutate the inner Arc'd provider, so we reconstruct.
        // For now, we store the model override and apply it per-request.
        Self {
            inner: Arc::new(ModelOverride {
                inner: Arc::clone(&self.inner),
                model: model.to_owned(),
            }),
        }
    }

    /// Wrap this model with automatic retry on transient failures.
    ///
    /// Retries rate-limit, timeout, and server errors with exponential
    /// backoff. `maxRetries` defaults to 3 if not specified.
    ///
    /// Returns a new `CompletionModel`.
    ///
    /// ```js
    /// const resilient = model.withRetry(5);
    /// const response = await resilient.complete([ChatMessage.user('Hi')]);
    /// ```
    #[wasm_bindgen(js_name = "withRetry")]
    pub fn with_retry(&self, max_retries: Option<u32>) -> WasmCompletionModel {
        let config = RetryConfig {
            max_retries: max_retries.unwrap_or(3),
            ..RetryConfig::default()
        };
        WasmCompletionModel {
            inner: Arc::new(RetryCompletionModel::from_arc(
                Arc::clone(&self.inner),
                config,
            )),
        }
    }

    /// Create a fallback model that tries multiple providers in order.
    ///
    /// When one provider fails with a retryable error, the next is tried.
    /// Non-retryable errors (e.g. auth) short-circuit immediately.
    ///
    /// `models` is an array of `CompletionModel` instances to try in order.
    ///
    /// ```js
    /// const primary = CompletionModel.openai();
    /// const backup = CompletionModel.groq();
    /// const resilient = CompletionModel.withFallback([primary, backup]);
    /// ```
    #[wasm_bindgen(js_name = "withFallback")]
    pub fn with_fallback(models: Vec<WasmCompletionModel>) -> WasmCompletionModel {
        let providers: Vec<Arc<dyn CompletionModel>> =
            models.into_iter().map(|m| m.inner).collect();
        WasmCompletionModel {
            inner: Arc::new(FallbackModel::new(providers)),
        }
    }

    /// Wrap this model with a response cache.
    ///
    /// Identical non-streaming requests will be served from memory for
    /// `ttlSeconds` (default 300 = 5 minutes). Streaming is never cached.
    ///
    /// `maxEntries` caps the cache size (default 1000); the oldest entry
    /// is evicted when the limit is reached.
    ///
    /// Returns a new `CompletionModel`.
    ///
    /// ```js
    /// const cached = model.withCache(600, 500);
    /// ```
    #[wasm_bindgen(js_name = "withCache")]
    pub fn with_cache(
        &self,
        ttl_seconds: Option<u32>,
        max_entries: Option<u32>,
    ) -> WasmCompletionModel {
        let config = CacheConfig {
            ttl_seconds: u64::from(ttl_seconds.unwrap_or(300)),
            max_entries: max_entries.unwrap_or(1000) as usize,
            ..CacheConfig::default()
        };
        WasmCompletionModel {
            inner: Arc::new(CachedCompletionModel::from_arc(
                Arc::clone(&self.inner),
                config,
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// The default model identifier for this provider.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------------
    // Completion
    // -----------------------------------------------------------------------

    /// Perform a non-streaming chat completion.
    ///
    /// `messages` should be an array of `ChatMessage` objects or plain JSON
    /// objects with `role` and `content` fields.
    ///
    /// Returns a `Promise<CompletionResponse>`.
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
            let js_resp = serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(js_resp)
        })
    }

    /// Perform a non-streaming completion with additional options.
    ///
    /// `options` is a plain JS object that may contain:
    /// - `temperature` (number)
    /// - `maxTokens` (number)
    /// - `topP` (number)
    /// - `model` (string) -- override the default model for this request
    /// - `tools` (array of tool definition objects)
    /// - `responseFormat` (JSON schema object)
    ///
    /// Returns a `Promise<CompletionResponse>`.
    #[wasm_bindgen(js_name = "completeWithOptions")]
    pub fn complete_with_options(
        &self,
        messages: JsValue,
        options: JsValue,
    ) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let msgs = js_messages_to_vec(&messages)?;
            let mut request = CompletionRequest::new(msgs);

            // Apply options from the JS object.
            if let Ok(opts) = serde_wasm_bindgen::from_value::<serde_json::Value>(options) {
                if let Some(temp) = opts.get("temperature").and_then(|v| v.as_f64()) {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        request = request.with_temperature(temp as f32);
                    }
                }
                if let Some(max) = opts.get("maxTokens").and_then(|v| v.as_u64()) {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        request = request.with_max_tokens(max as u32);
                    }
                }
                if let Some(top_p) = opts.get("topP").and_then(|v| v.as_f64()) {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        request = request.with_top_p(top_p as f32);
                    }
                }
                if let Some(model_override) = opts.get("model").and_then(|v| v.as_str()) {
                    request = request.with_model(model_override);
                }
                if let Some(fmt) = opts.get("responseFormat") {
                    request = request.with_response_format(fmt.clone());
                }
                if let Some(tools) = opts.get("tools").and_then(|v| v.as_array()) {
                    let mut tool_defs = Vec::with_capacity(tools.len());
                    for (i, t) in tools.iter().enumerate() {
                        let def: blazen_llm::types::ToolDefinition =
                            serde_json::from_value(t.clone()).map_err(|e| {
                                JsValue::from_str(&format!(
                                    "invalid tool definition at index {i}: {e}"
                                ))
                            })?;
                        tool_defs.push(def);
                    }
                    if !tool_defs.is_empty() {
                        request = request.with_tools(tool_defs);
                    }
                }
            }

            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let js_resp = serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(js_resp)
        })
    }

    /// Perform a streaming chat completion.
    ///
    /// The `callback` function is invoked for each streaming chunk with a
    /// `StreamChunk` object. Returns a `Promise<void>` that resolves when
    /// streaming is complete.
    ///
    /// ```js
    /// await model.stream([ChatMessage.user('Count to 5')], (chunk) => {
    ///   if (chunk.delta) process.stdout.write(chunk.delta);
    /// });
    /// ```
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
                let chunk = chunk_result
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let js_chunk = serde_wasm_bindgen::to_value(&chunk)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                // Invoke the JS callback with the chunk.
                let _ = callback.call1(&JsValue::NULL, &js_chunk);
            }

            Ok(JsValue::UNDEFINED)
        })
    }
}

// ---------------------------------------------------------------------------
// Model override wrapper
// ---------------------------------------------------------------------------

/// Internal wrapper that overrides the model ID on every request.
struct ModelOverride {
    inner: Arc<dyn CompletionModel>,
    model: String,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for ModelOverride {}
unsafe impl Sync for ModelOverride {}

#[async_trait::async_trait]
impl CompletionModel for ModelOverride {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<blazen_llm::CompletionResponse, blazen_llm::BlazenError> {
        let request = request.with_model(&self.model);
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures_util::Stream<
                        Item = Result<blazen_llm::StreamChunk, blazen_llm::BlazenError>,
                    > + Send,
            >,
        >,
        blazen_llm::BlazenError,
    > {
        let request = request.with_model(&self.model);
        self.inner.stream(request).await
    }
}
