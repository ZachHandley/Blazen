//! Generic OpenAI-compatible chat completion provider.
//!
//! Most hosted LLM APIs follow the `OpenAI` chat completions wire format. This
//! module provides [`OpenAiCompatProvider`] -- a single implementation of
//! [`CompletionModel`] that works with any OpenAI-compatible endpoint by
//! configuring the base URL, auth method, and extra headers.
//!
//! For specific providers, use the dedicated provider modules (e.g.
//! `GroqProvider`, `OpenRouterProvider`, etc.) which wrap this type with
//! pre-configured defaults.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use tracing::debug;

use super::openai_format::{content_to_openai_value, parse_retry_after};
use super::sse::{OaiResponse, SseParser};
use super::{provider_http_error, provider_http_error_parts};
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest, HttpResponse};
use crate::traits::{ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry};
use crate::types::{
    Citation, CompletionRequest, CompletionResponse, EmbeddingResponse, ReasoningTrace, Role,
    StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Auth method
// ---------------------------------------------------------------------------

/// How to authenticate with the provider API.
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// `Authorization: Bearer <key>` (`OpenAI`, `OpenRouter`, Groq, etc.)
    Bearer,
    /// A custom header name for the API key (e.g. `x-api-key`).
    ApiKeyHeader(String),
    /// `api-key: <key>` (Azure `OpenAI`).
    AzureApiKey,
    /// `Authorization: Key <key>` (fal.ai).
    KeyPrefix,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    /// A human-readable name for this provider (used in logs and model info).
    pub provider_name: String,
    /// The base URL for the API (e.g. `https://api.openai.com/v1`).
    pub base_url: String,
    /// The API key.
    pub api_key: String,
    /// The default model to use if the request does not override it.
    pub default_model: String,
    /// How to send the API key.
    pub auth_method: AuthMethod,
    /// Extra headers to include in every request.
    pub extra_headers: Vec<(String, String)>,
    /// Query parameters to include in every request (e.g. Azure `api-version`).
    pub query_params: Vec<(String, String)>,
    /// Whether this provider supports the `/models` listing endpoint.
    pub supports_model_listing: bool,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A generic OpenAI-compatible chat completion provider.
///
/// Build a custom configuration with [`Self::new`] or use a dedicated
/// provider module for popular services.
pub struct OpenAiCompatProvider {
    config: OpenAiCompatConfig,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Clone for OpenAiCompatProvider {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: Arc::clone(&self.client),
        }
    }
}

impl OpenAiCompatProvider {
    /// Create a provider from a fully-specified configuration.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(config: OpenAiCompatConfig) -> Self {
        Self {
            config,
            client: crate::default_http_client(),
        }
    }

    /// Create a provider from a configuration with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(config: OpenAiCompatConfig, client: Arc<dyn HttpClient>) -> Self {
        Self { config, client }
    }
}

impl OpenAiCompatProvider {
    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Override the default model for this provider instance.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Add an extra header to include in every request.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.extra_headers.push((key.into(), value.into()));
        self
    }

    /// Override the base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Build the JSON request body for the chat completions endpoint.
    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let model = request
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let content = content_to_openai_value(&m.content);
                let mut msg = serde_json::json!({ "role": role, "content": content });

                // Tool result messages must include the tool_call_id.
                if let Some(ref id) = m.tool_call_id {
                    msg["tool_call_id"] = serde_json::json!(id);
                }

                // Assistant messages with tool calls must include the tool_calls
                // array and may have null content.
                if !m.tool_calls.is_empty() {
                    let tc_arr: Vec<serde_json::Value> = m
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    msg["tool_calls"] = serde_json::json!(tc_arr);
                    if m.content.as_text().is_none_or(str::is_empty) {
                        msg["content"] = serde_json::Value::Null;
                    }
                }

                msg
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": stream,
        });

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(max) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(ref fmt) = request.response_format {
            if fmt.get("type").and_then(|v| v.as_str()) == Some("json_schema") {
                body["response_format"] = fmt.clone();
            } else {
                body["response_format"] = serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": fmt,
                        "strict": true,
                    }
                });
            }
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        // Multimodal output modalities
        if let Some(modalities) = &request.modalities {
            body["modalities"] = serde_json::to_value(modalities).unwrap_or_default();
        }

        // Image generation configuration
        if let Some(image_config) = &request.image_config {
            body["image_config"] = image_config.clone();
        }

        // Audio output configuration
        if let Some(audio_config) = &request.audio_config {
            body["audio"] = audio_config.clone();
        }

        body
    }

    /// Apply authentication and extra headers/query params to an [`HttpRequest`].
    fn apply_config(&self, mut request: HttpRequest) -> HttpRequest {
        // Auth
        match &self.config.auth_method {
            AuthMethod::Bearer => {
                request.headers.push((
                    "Authorization".to_owned(),
                    format!("Bearer {}", self.config.api_key),
                ));
            }
            AuthMethod::ApiKeyHeader(header_name) => {
                request
                    .headers
                    .push((header_name.clone(), self.config.api_key.clone()));
            }
            AuthMethod::AzureApiKey => {
                request
                    .headers
                    .push(("api-key".to_owned(), self.config.api_key.clone()));
            }
            AuthMethod::KeyPrefix => {
                request.headers.push((
                    "Authorization".to_owned(),
                    format!("Key {}", self.config.api_key),
                ));
            }
        }

        // Extra headers
        for (key, value) in &self.config.extra_headers {
            request.headers.push((key.clone(), value.clone()));
        }

        // Query params
        for (key, value) in &self.config.query_params {
            request.query_params.push((key.clone(), value.clone()));
        }

        request
    }

    /// Build an [`HttpRequest`] for the chat completions endpoint.
    fn build_http_request(&self, body: &serde_json::Value) -> Result<HttpRequest, BlazenError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let request = HttpRequest::post(url).json_body(body)?;
        Ok(self.apply_config(request))
    }

    /// Send a request and return the raw response, handling common HTTP errors.
    async fn send_request(&self, body: &serde_json::Value) -> Result<HttpResponse, BlazenError> {
        let request = self.build_http_request(body)?;
        let response = self.client.send(request).await?;

        if response.is_success() {
            return Ok(response);
        }

        match response.status {
            401 => Err(BlazenError::auth("authentication failed")),
            404 => Err(BlazenError::model_not_found(response.text())),
            429 => Err(BlazenError::RateLimit {
                retry_after_ms: parse_retry_after(&response.headers),
            }),
            _ => {
                let url = format!("{}/chat/completions", self.config.base_url);
                Err(provider_http_error(
                    self.config.provider_name.clone(),
                    &url,
                    &response,
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Response parsing helpers
// ---------------------------------------------------------------------------

/// Build a [`ReasoningTrace`] from the two possible OpenAI-compat fields.
///
/// Prefers `reasoning_content` (`DeepSeek` R1 family) over `reasoning` (Grok).
fn build_reasoning_trace(
    reasoning_content: Option<String>,
    reasoning: Option<String>,
) -> Option<ReasoningTrace> {
    match (reasoning_content, reasoning) {
        (Some(text), _) | (None, Some(text)) => Some(ReasoningTrace {
            text,
            signature: None,
            redacted: false,
            effort: None,
        }),
        (None, None) => None,
    }
}

/// Convert a wire-format `citations` array (used by Perplexity and friends)
/// into typed [`Citation`] values. Each entry's original JSON is preserved in
/// `Citation::metadata`.
fn parse_citations(entries: Vec<serde_json::Value>) -> Vec<Citation> {
    entries
        .into_iter()
        .map(|entry| {
            let (url, title, snippet) = if let Some(obj) = entry.as_object() {
                let url = obj
                    .get("url")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_default();
                let title = obj
                    .get("title")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                let snippet = obj
                    .get("snippet")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                (url, title, snippet)
            } else if let Some(s) = entry.as_str() {
                (s.to_owned(), None, None)
            } else {
                (entry.to_string(), None, None)
            };
            Citation {
                url,
                title,
                snippet,
                start: None,
                end: None,
                document_id: None,
                metadata: entry,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for OpenAiCompatProvider {
    fn model_id(&self) -> &str {
        &self.config.default_model
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model_id = request
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let provider_name = self.config.provider_name.as_str();
        let span = tracing::info_span!(
            "llm.complete",
            provider = %provider_name,
            model = %model_id,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let body = self.build_body(&request, false);
        debug!(
            provider = %self.config.provider_name,
            model = %body["model"],
            "OpenAI-compat completion request"
        );

        let response = self.send_request(&body).await?;
        let oai: OaiResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let choice = oai
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| BlazenError::invalid_response("empty choices array"))?;

        let message = choice.message;
        let finish_reason = choice.finish_reason;

        let tool_calls = message
            .tool_calls
            .into_iter()
            .map(|tc| {
                let args = serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: args,
                }
            })
            .collect();

        let reasoning = build_reasoning_trace(message.reasoning_content, message.reasoning);
        let citations = parse_citations(message.citations);

        let usage = oai.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .map_or(0, |d| d.reasoning_tokens),
            ..Default::default()
        });

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        if let Some(ref u) = usage {
            span.record("prompt_tokens", u.prompt_tokens);
            span.record("completion_tokens", u.completion_tokens);
            span.record("total_tokens", u.total_tokens);
        }
        if let Some(ref reason) = finish_reason {
            span.record("finish_reason", reason.as_str());
        }

        let cost = usage
            .as_ref()
            .and_then(|u| crate::pricing::compute_cost(&oai.model, u));

        Ok(CompletionResponse {
            content: message.content,
            tool_calls,
            reasoning,
            citations,
            artifacts: vec![],
            usage,
            model: oai.model,
            finish_reason,
            cost,
            timing: None,
            images: vec![],
            audio: vec![],
            videos: vec![],
            metadata: serde_json::Value::Null,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let model_id = request
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let provider_name = self.config.provider_name.as_str();
        let span = tracing::info_span!(
            "llm.stream",
            provider = %provider_name,
            model = %model_id,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let body = self.build_body(&request, true);
        debug!(
            provider = %self.config.provider_name,
            model = %body["model"],
            "OpenAI-compat streaming request"
        );

        let http_request = self.build_http_request(&body)?;
        let (status, headers, byte_stream) = self.client.send_streaming(http_request).await?;

        if !(200..300).contains(&status) {
            match status {
                401 => return Err(BlazenError::auth("authentication failed")),
                404 => return Err(BlazenError::model_not_found("model not found")),
                429 => {
                    return Err(BlazenError::RateLimit {
                        retry_after_ms: parse_retry_after(&headers),
                    });
                }
                _ => {
                    let url = format!("{}/chat/completions", self.config.base_url);
                    return Err(provider_http_error_parts(
                        self.config.provider_name.clone(),
                        &url,
                        status,
                        &headers,
                        "",
                    ));
                }
            }
        }

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );

        let stream = SseParser::new(byte_stream);
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry implementation
// ---------------------------------------------------------------------------

/// Wire format for the standard `/models` endpoint response.
#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<ModelEntry>,
}

/// A model entry from the `/models` endpoint.
#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
    // OpenRouter pricing format
    #[serde(default)]
    pricing: Option<OpenRouterPricing>,
}

/// OpenRouter-specific pricing within a model entry.
#[derive(Debug, Deserialize)]
struct OpenRouterPricing {
    /// USD per token (input).
    prompt: Option<String>,
    /// USD per token (output).
    completion: Option<String>,
}

/// Together AI model entry (bare array format).
#[derive(Debug, Deserialize)]
struct TogetherModelEntry {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    pricing: Option<TogetherPricing>,
}

/// Together AI pricing within a model entry.
#[derive(Debug, Deserialize)]
struct TogetherPricing {
    /// USD per million input tokens.
    input: Option<f64>,
    /// USD per million output tokens.
    output: Option<f64>,
}

#[async_trait]
impl ModelRegistry for OpenAiCompatProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError> {
        if !self.config.supports_model_listing {
            return Ok(Vec::new());
        }

        let url = format!("{}/models", self.config.base_url);
        let request = self.apply_config(HttpRequest::get(&url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            return Err(provider_http_error(
                self.config.provider_name.clone(),
                &url,
                &response,
            ));
        }

        let body_text = response.text();

        // Try the standard `{ "data": [...] }` format first.
        if let Ok(list) = serde_json::from_str::<ModelsListResponse>(&body_text) {
            return Ok(list
                .data
                .into_iter()
                .map(|entry| self.model_entry_to_info(entry))
                .collect());
        }

        // Together AI returns a bare array `[...]`.
        if let Ok(models) = serde_json::from_str::<Vec<TogetherModelEntry>>(&body_text) {
            return Ok(models
                .into_iter()
                .map(|entry| self.together_entry_to_info(entry))
                .collect());
        }

        Err(BlazenError::invalid_response(
            "unexpected model listing response format",
        ))
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, BlazenError> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}

impl OpenAiCompatProvider {
    /// Convert a standard model entry to [`ModelInfo`].
    fn model_entry_to_info(&self, entry: ModelEntry) -> ModelInfo {
        let pricing = entry.pricing.map(|p| {
            // OpenRouter gives USD per token; convert to per-million.
            let input = p
                .prompt
                .and_then(|s| s.parse::<f64>().ok())
                .map(|v| v * 1_000_000.0);
            let output = p
                .completion
                .and_then(|s| s.parse::<f64>().ok())
                .map(|v| v * 1_000_000.0);
            ModelPricing {
                input_per_million: input,
                output_per_million: output,
                ..Default::default()
            }
        });

        let info = ModelInfo {
            id: entry.id,
            name: entry.name,
            provider: self.config.provider_name.clone(),
            context_length: entry.context_length,
            pricing,
            capabilities: ModelCapabilities {
                chat: true,
                streaming: true,
                ..Default::default()
            },
        };
        crate::pricing::register_from_model_info(&info);
        info
    }

    /// Convert a Together AI model entry to [`ModelInfo`].
    fn together_entry_to_info(&self, entry: TogetherModelEntry) -> ModelInfo {
        let pricing = entry.pricing.map(|p| ModelPricing {
            input_per_million: p.input,
            output_per_million: p.output,
            ..Default::default()
        });

        let info = ModelInfo {
            id: entry.id,
            name: entry.display_name,
            provider: self.config.provider_name.clone(),
            context_length: entry.context_length,
            pricing,
            capabilities: ModelCapabilities {
                chat: true,
                streaming: true,
                ..Default::default()
            },
        };
        crate::pricing::register_from_model_info(&info);
        info
    }
}

// ---------------------------------------------------------------------------
// Embedding model (OpenAI-compatible)
// ---------------------------------------------------------------------------

/// An OpenAI-compatible embedding model.
///
/// Works with any provider that implements the `OpenAI` embeddings API
/// (`POST /embeddings`). For specific providers, use the dedicated provider
/// modules.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::openai_compat::{
///     AuthMethod, OpenAiCompatConfig, OpenAiCompatEmbeddingModel,
/// };
///
/// let embedder = OpenAiCompatEmbeddingModel::new(
///     OpenAiCompatConfig {
///         provider_name: "my-provider".into(),
///         base_url: "https://api.example.com/v1".into(),
///         api_key: "your-key".into(),
///         default_model: String::new(),
///         auth_method: AuthMethod::Bearer,
///         extra_headers: Vec::new(),
///         query_params: Vec::new(),
///         supports_model_listing: false,
///     },
///     "my-embedding-model",
///     768,
/// );
/// ```
pub struct OpenAiCompatEmbeddingModel {
    config: OpenAiCompatConfig,
    client: Arc<dyn HttpClient>,
    model: String,
    dimensions: usize,
}

impl std::fmt::Debug for OpenAiCompatEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatEmbeddingModel")
            .field("provider", &self.config.provider_name)
            .field("base_url", &self.config.base_url)
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl Clone for OpenAiCompatEmbeddingModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: Arc::clone(&self.client),
            model: self.model.clone(),
            dimensions: self.dimensions,
        }
    }
}

impl OpenAiCompatEmbeddingModel {
    /// Create an embedding model from a fully-specified configuration.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(config: OpenAiCompatConfig, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            config,
            client: crate::default_http_client(),
            model: model.into(),
            dimensions,
        }
    }

    /// Create an embedding model with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(
        config: OpenAiCompatConfig,
        model: impl Into<String>,
        dimensions: usize,
        client: Arc<dyn HttpClient>,
    ) -> Self {
        Self {
            config,
            client,
            model: model.into(),
            dimensions,
        }
    }
}

/// Default embedding configurations for known providers.
const EMBEDDING_DEFAULTS: &[(&str, &str, &str, usize)] = &[
    // (provider_name, base_url, default_model, dimensions)
    (
        "together",
        "https://api.together.xyz/v1",
        "togethercomputer/m2-bert-80M-8k-retrieval",
        768,
    ),
    (
        "cohere",
        "https://api.cohere.ai/compatibility/v1",
        "embed-v4.0",
        1024,
    ),
    (
        "fireworks",
        "https://api.fireworks.ai/inference/v1",
        "nomic-ai/nomic-embed-text-v1.5",
        768,
    ),
];

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest"
))]
impl OpenAiCompatEmbeddingModel {
    /// Construct an embedding model for a known provider from
    /// [`ProviderOptions`](crate::types::provider_options::ProviderOptions).
    ///
    /// Resolves the API key from `opts.api_key` or the provider's
    /// environment variable, then applies default base URL, model, and
    /// dimensions for the provider (overridable via `opts`).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Auth`] if the API key cannot be resolved, or
    /// [`BlazenError::InvalidArg`] if the provider name is unknown.
    pub fn embedding_from_options(
        provider: &str,
        opts: crate::types::provider_options::ProviderOptions,
    ) -> Result<Self, crate::BlazenError> {
        let (_, default_url, default_model, default_dims) = EMBEDDING_DEFAULTS
            .iter()
            .find(|(name, _, _, _)| *name == provider)
            .ok_or_else(|| crate::BlazenError::Unsupported {
                message: format!("unknown embedding provider: {provider}"),
            })?;

        let api_key = crate::keys::resolve_api_key(provider, opts.api_key)?;
        let base_url = opts.base_url.unwrap_or_else(|| (*default_url).to_owned());
        let model = opts.model.unwrap_or_else(|| (*default_model).to_owned());

        Ok(Self::new_with_client(
            OpenAiCompatConfig {
                provider_name: provider.into(),
                base_url,
                api_key,
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            model,
            *default_dims,
            crate::default_http_client(),
        ))
    }
}

impl OpenAiCompatEmbeddingModel {
    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Override the embedding model and dimensionality.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>, dimensions: usize) -> Self {
        self.model = model.into();
        self.dimensions = dimensions;
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Apply authentication and extra headers/query params to an [`HttpRequest`].
    fn apply_config(&self, mut request: HttpRequest) -> HttpRequest {
        match &self.config.auth_method {
            AuthMethod::Bearer => {
                request.headers.push((
                    "Authorization".to_owned(),
                    format!("Bearer {}", self.config.api_key),
                ));
            }
            AuthMethod::ApiKeyHeader(header_name) => {
                request
                    .headers
                    .push((header_name.clone(), self.config.api_key.clone()));
            }
            AuthMethod::AzureApiKey => {
                request
                    .headers
                    .push(("api-key".to_owned(), self.config.api_key.clone()));
            }
            AuthMethod::KeyPrefix => {
                request.headers.push((
                    "Authorization".to_owned(),
                    format!("Key {}", self.config.api_key),
                ));
            }
        }

        for (key, value) in &self.config.extra_headers {
            request.headers.push((key.clone(), value.clone()));
        }

        for (key, value) in &self.config.query_params {
            request.query_params.push((key.clone(), value.clone()));
        }

        request
    }
}

/// Wire format for an OpenAI-compatible embeddings API response.
#[derive(Debug, Deserialize)]
struct OaiCompatEmbeddingResponse {
    data: Vec<OaiCompatEmbeddingData>,
    model: String,
    usage: Option<OaiCompatEmbeddingUsage>,
}

/// A single embedding vector from the response.
#[derive(Debug, Deserialize)]
struct OaiCompatEmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

/// Token usage from the embeddings response.
#[derive(Debug, Deserialize)]
struct OaiCompatEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[async_trait]
impl crate::traits::EmbeddingModel for OpenAiCompatEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let url = format!("{}/embeddings", self.config.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let request = HttpRequest::post(&url).json_body(&body)?;
        let request = self.apply_config(request);

        let response = self.client.send(request).await?;

        if !response.is_success() {
            return match response.status {
                401 => Err(BlazenError::auth("authentication failed")),
                404 => Err(BlazenError::model_not_found(response.text())),
                429 => Err(BlazenError::RateLimit {
                    retry_after_ms: parse_retry_after(&response.headers),
                }),
                _ => Err(provider_http_error(
                    self.config.provider_name.clone(),
                    &url,
                    &response,
                )),
            };
        }

        let oai: OaiCompatEmbeddingResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let mut embeddings: Vec<(usize, Vec<f32>)> = oai
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        embeddings.sort_by_key(|(idx, _)| *idx);
        let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, v)| v).collect();

        let usage = oai.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: 0,
            total_tokens: u.total_tokens,
            ..Default::default()
        });

        Ok(EmbeddingResponse {
            embeddings,
            model: oai.model,
            usage,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl crate::traits::ProviderInfo for OpenAiCompatProvider {
    fn provider_name(&self) -> &str {
        &self.config.provider_name
    }

    fn base_url(&self) -> &str {
        &self.config.base_url
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: true,
            model_listing: self.config.supports_model_listing,
            embeddings: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ToolDefinition};

    /// Helper to create a test provider with sensible defaults.
    fn test_provider(model: &str) -> OpenAiCompatProvider {
        OpenAiCompatProvider::new(OpenAiCompatConfig {
            provider_name: "test".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: "test-key".into(),
            default_model: model.into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    #[test]
    fn new_creates_provider_with_config() {
        let provider = test_provider("gpt-4.1");
        assert_eq!(provider.config.base_url, "https://api.openai.com/v1");
        assert_eq!(provider.config.default_model, "gpt-4.1");
        assert!(matches!(provider.config.auth_method, AuthMethod::Bearer));
        assert!(provider.config.supports_model_listing);
    }

    #[test]
    fn with_model_override() {
        let provider = test_provider("gpt-4.1").with_model("gpt-4.1-mini");
        assert_eq!(provider.config.default_model, "gpt-4.1-mini");
    }

    #[test]
    fn with_header_appends() {
        let provider = test_provider("gpt-4.1")
            .with_header("HTTP-Referer", "https://myapp.com")
            .with_header("X-Title", "My App");
        assert_eq!(provider.config.extra_headers.len(), 2);
    }

    #[test]
    fn build_body_minimal() {
        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest {
            messages: vec![ChatMessage::user("Hello")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            response_format: None,
            model: None,
            modalities: None,
            image_config: None,
            audio_config: None,
        };

        let body = provider.build_body(&request, false);
        assert_eq!(body["model"], "gpt-4.1");
        assert_eq!(body["stream"], false);
        assert!(body.get("temperature").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn build_body_with_options() {
        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")])
            .with_temperature(0.5)
            .with_max_tokens(100)
            .with_model("gpt-4.1-mini");

        let body = provider.build_body(&request, true);
        assert_eq!(body["model"], "gpt-4.1-mini");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn build_body_with_tools() {
        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]).with_tools(vec![
            ToolDefinition {
                name: "get_weather".to_owned(),
                description: "Get current weather".to_owned(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    }
                }),
            },
        ]);

        let body = provider.build_body(&request, false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn test_text_backward_compat() {
        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_body_image_url() {
        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "What is this?",
            "https://example.com/cat.jpg",
            Some("image/jpeg"),
        )]);

        let body = provider.build_body(&request, false);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "What is this?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "https://example.com/cat.jpg"
        );
    }

    #[test]
    fn test_build_body_base64_image() {
        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_base64(
            "Describe this",
            "abc123base64data",
            "image/png",
        )]);

        let body = provider.build_body(&request, false);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[1]["type"], "image_url");
        assert!(
            content[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn test_build_body_multipart() {
        use crate::types::{ContentPart, ImageContent, ImageSource};

        let provider = test_provider("gpt-4.1");
        let request = CompletionRequest::new(vec![ChatMessage::user_parts(vec![
            ContentPart::Text {
                text: "First".into(),
            },
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/a.png".into(),
                },
                media_type: None,
            }),
            ContentPart::Text {
                text: "Second".into(),
            },
        ])]);

        let body = provider.build_body(&request, false);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["text"], "First");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[2]["text"], "Second");
    }

    #[test]
    fn test_parse_response_with_reasoning_content() {
        let json_body = r#"{
            "id": "x",
            "model": "test-r1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42",
                    "reasoning_content": "I considered..."
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        }"#;
        let parsed: OaiResponse = serde_json::from_str(json_body).unwrap();
        let msg = &parsed.choices[0].message;
        assert_eq!(msg.reasoning_content.as_deref(), Some("I considered..."));
    }

    #[test]
    fn test_parse_response_with_citations() {
        let json_body = r#"{
            "id": "x",
            "model": "test-perplexity",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Per source A...",
                    "citations": [
                        {"url": "https://example.com/a", "title": "Source A"},
                        {"url": "https://example.com/b"}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        }"#;
        let parsed: OaiResponse = serde_json::from_str(json_body).unwrap();
        let msg = &parsed.choices[0].message;
        assert_eq!(msg.citations.len(), 2);
        assert_eq!(msg.citations[0]["url"], "https://example.com/a");
    }

    #[test]
    fn test_parse_response_with_completion_tokens_details() {
        let json_body = r#"{
            "id": "x",
            "model": "test-o1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 100,
                "total_tokens": 110,
                "completion_tokens_details": { "reasoning_tokens": 80 }
            }
        }"#;
        let parsed: OaiResponse = serde_json::from_str(json_body).unwrap();
        assert_eq!(
            parsed
                .usage
                .unwrap()
                .completion_tokens_details
                .unwrap()
                .reasoning_tokens,
            80
        );
    }

    #[test]
    fn parse_standard_model_list() {
        let provider = test_provider("gpt-4.1");
        let json = r#"{"data":[{"id":"gpt-4o","context_length":128000}]}"#;

        let list: ModelsListResponse = serde_json::from_str(json).unwrap();
        let models: Vec<ModelInfo> = list
            .data
            .into_iter()
            .map(|e| provider.model_entry_to_info(e))
            .collect();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[0].context_length, Some(128_000));
        assert_eq!(models[0].provider, "test");
    }

    #[test]
    fn parse_openrouter_model_with_pricing() {
        let provider = test_provider("gpt-4.1");
        let json = r#"{"data":[{"id":"openai/gpt-4o","context_length":128000,"pricing":{"prompt":"0.000005","completion":"0.000015"}}]}"#;

        let list: ModelsListResponse = serde_json::from_str(json).unwrap();
        let models: Vec<ModelInfo> = list
            .data
            .into_iter()
            .map(|e| provider.model_entry_to_info(e))
            .collect();

        assert_eq!(models.len(), 1);
        let pricing = models[0].pricing.as_ref().unwrap();
        // 0.000005 per token * 1M = 5.0 per million
        assert!((pricing.input_per_million.unwrap() - 5.0).abs() < 0.001);
        assert!((pricing.output_per_million.unwrap() - 15.0).abs() < 0.001);
    }

    #[test]
    fn parse_together_model_list() {
        let provider = test_provider("gpt-4.1");
        let json = r#"[{"id":"meta-llama/Llama-3-70b","display_name":"Llama 3 70B","context_length":8192,"pricing":{"input":0.9,"output":0.9}}]"#;

        let models: Vec<TogetherModelEntry> = serde_json::from_str(json).unwrap();
        let infos: Vec<ModelInfo> = models
            .into_iter()
            .map(|e| provider.together_entry_to_info(e))
            .collect();

        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, "meta-llama/Llama-3-70b");
        assert_eq!(infos[0].name.as_deref(), Some("Llama 3 70B"));
        let pricing = infos[0].pricing.as_ref().unwrap();
        assert!((pricing.input_per_million.unwrap() - 0.9).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // Embedding model tests
    // -----------------------------------------------------------------------

    /// Helper to create a test embedding model.
    fn test_embedding_model(model: &str, dimensions: usize) -> OpenAiCompatEmbeddingModel {
        OpenAiCompatEmbeddingModel::new(
            OpenAiCompatConfig {
                provider_name: "test".into(),
                base_url: "https://api.example.com/v1".into(),
                api_key: "test-key".into(),
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            model,
            dimensions,
        )
    }

    #[test]
    fn compat_embedding_new() {
        use crate::traits::EmbeddingModel;

        let embedder = test_embedding_model("text-embedding-3-small", 768);
        assert_eq!(embedder.model_id(), "text-embedding-3-small");
        assert_eq!(embedder.dimensions(), 768);
        assert_eq!(embedder.config.base_url, "https://api.example.com/v1");
        assert_eq!(embedder.config.provider_name, "test");
    }

    #[test]
    fn compat_embedding_with_model_override() {
        use crate::traits::EmbeddingModel;

        let embedder = test_embedding_model("text-embedding-3-small", 768)
            .with_model("BAAI/bge-large-en-v1.5", 1024);
        assert_eq!(embedder.model_id(), "BAAI/bge-large-en-v1.5");
        assert_eq!(embedder.dimensions(), 1024);
    }

    #[test]
    fn compat_embedding_response_parsing() {
        let json = r#"{
            "data": [
                {"embedding": [0.7, 0.8, 0.9], "index": 2},
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "togethercomputer/m2-bert-80M-8k-retrieval",
            "usage": {"prompt_tokens": 25, "total_tokens": 25}
        }"#;

        let oai: OaiCompatEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(oai.data.len(), 3);
        assert_eq!(oai.model, "togethercomputer/m2-bert-80M-8k-retrieval");

        // Verify reordering by index
        let mut embeddings: Vec<(usize, Vec<f32>)> = oai
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        embeddings.sort_by_key(|(idx, _)| *idx);
        let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, v)| v).collect();

        assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]); // index 0
        assert_eq!(embeddings[1], vec![0.4, 0.5, 0.6]); // index 1
        assert_eq!(embeddings[2], vec![0.7, 0.8, 0.9]); // index 2

        let usage = oai.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 25);
        assert_eq!(usage.total_tokens, 25);
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider (stub -- required as a supertrait for AudioGeneration)
// ---------------------------------------------------------------------------

// OpenAI-compatible services use plain POST/response, not a queue-based
// compute job, so the queue-lifecycle methods return Unsupported.

#[async_trait]
impl crate::compute::traits::ComputeProvider for OpenAiCompatProvider {
    fn provider_id(&self) -> &str {
        &self.config.provider_name
    }

    async fn submit(
        &self,
        _request: crate::compute::job::ComputeRequest,
    ) -> Result<crate::compute::job::JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI-compatible provider does not expose a queue-based compute API",
        ))
    }

    async fn status(
        &self,
        _job: &crate::compute::job::JobHandle,
    ) -> Result<crate::compute::job::JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI-compatible provider does not expose a queue-based compute API",
        ))
    }

    async fn result(
        &self,
        _job: crate::compute::job::JobHandle,
    ) -> Result<crate::compute::job::ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI-compatible provider does not expose a queue-based compute API",
        ))
    }

    async fn cancel(&self, _job: &crate::compute::job::JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI-compatible provider does not expose a queue-based compute API",
        ))
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration (text-to-speech via /v1/audio/speech)
// ---------------------------------------------------------------------------

// Only Bearer auth is currently supported for TTS because the underlying
// helper (`openai_audio::text_to_speech_request`) applies bearer auth
// unconditionally. Non-Bearer compat services (Azure, custom header,
// KeyPrefix) return Unsupported until the helper gains multi-auth support.

#[async_trait]
impl crate::compute::traits::AudioGeneration for OpenAiCompatProvider {
    async fn text_to_speech(
        &self,
        request: crate::compute::requests::SpeechRequest,
    ) -> Result<crate::compute::results::AudioResult, BlazenError> {
        match &self.config.auth_method {
            AuthMethod::Bearer => {
                super::openai_audio::text_to_speech_request(
                    self.client.as_ref(),
                    &self.config.base_url,
                    &self.config.api_key,
                    request,
                )
                .await
            }
            _ => Err(BlazenError::unsupported(
                "OpenAI-compatible TTS currently supports Bearer auth only",
            )),
        }
    }
    // generate_music and generate_sfx intentionally NOT overridden --
    // they fall through to the default `Err(Unsupported)` impls in the trait.
}
