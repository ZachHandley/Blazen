//! `OpenAI` chat completion provider.
//!
//! This module provides the original [`OpenAiProvider`] for direct use with the
//! `OpenAI` API. For connecting to other `OpenAI`-compatible services (`OpenRouter`,
//! Groq, Together AI, etc.), see [`super::openai_compat::OpenAiCompatProvider`].

use std::pin::Pin;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use tracing::debug;

use serde::Deserialize;

use super::openai_format::{
    content_to_openai_value, parse_retry_after, tool_result_to_openai_string,
};
use super::sse::{OaiResponse, SseParser};
use super::{provider_http_error, provider_http_error_parts};
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest, HttpResponse};
use crate::retry::RetryConfig;
use crate::traits::{ModelCapabilities, ModelInfo, ModelRegistry};
use crate::types::{
    CompletionRequest, CompletionResponse, EmbeddingResponse, Role, StreamChunk, TokenUsage,
    ToolCall,
};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// An `OpenAI` chat completion provider.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::openai::OpenAiProvider;
///
/// let provider = OpenAiProvider::new("sk-...")
///     .with_model("gpt-4.1-mini");
/// ```
pub struct OpenAiProvider {
    client: Arc<dyn HttpClient>,
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiProvider")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

impl Clone for OpenAiProvider {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            retry_config: self.retry_config.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            default_model: self.default_model.clone(),
        }
    }
}

impl OpenAiProvider {
    /// Create a new provider with the given API key, targeting the official
    /// `OpenAI` endpoint.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            retry_config: None,
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_owned(),
            default_model: "gpt-4.1".to_owned(),
        }
    }

    /// Create a new provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            retry_config: None,
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_owned(),
            default_model: "gpt-4.1".to_owned(),
        }
    }

    /// Use a custom base URL (e.g. for local proxies).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the default model identifier.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// Return a clone of the underlying HTTP client.
    ///
    /// Escape hatch for power users who need to issue raw HTTP requests
    /// (custom headers, endpoints not yet covered by Blazen's typed
    /// surface, debugging) while reusing the same connection pool, TLS
    /// config, and timeouts as this provider.
    #[must_use]
    pub fn http_client(&self) -> Arc<dyn HttpClient> {
        Arc::clone(&self.client)
    }

    /// Set the provider-level default retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Build the JSON request body for the `OpenAI` chat completions endpoint.
    #[allow(clippy::too_many_lines)]
    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let model = request.model.as_deref().unwrap_or(&self.default_model);

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
                let content = if m.role == Role::Tool {
                    serde_json::Value::String(tool_result_to_openai_string(
                        m,
                        crate::types::ProviderId::OpenAi,
                    ))
                } else {
                    content_to_openai_value(&m.content)
                };
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
                    // OpenAI expects content to be null when tool_calls are present
                    // and there is no meaningful text.
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
            body["max_completion_tokens"] = serde_json::json!(max);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(ref fmt) = request.response_format {
            // If the caller already provided the full OpenAI envelope
            // (with "type": "json_schema"), pass it through verbatim.
            // Otherwise wrap a bare schema in the standard envelope.
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

    /// Build an [`HttpRequest`] for the chat completions endpoint.
    fn build_http_request(&self, body: &serde_json::Value) -> Result<HttpRequest, BlazenError> {
        let url = format!("{}/chat/completions", self.base_url);
        HttpRequest::post(url)
            .bearer_auth(&self.api_key)
            .json_body(body)
    }

    /// Send a request and return the raw response, handling common errors.
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
                let url = format!("{}/chat/completions", self.base_url);
                Err(provider_http_error("openai", &url, &response))
            }
        }
    }
}

super::impl_simple_from_options!(OpenAiProvider, "openai");

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for OpenAiProvider {
    fn model_id(&self) -> &str {
        &self.default_model
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config.as_ref()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        Some(Self::http_client(self))
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model_id = request.model.as_deref().unwrap_or(&self.default_model);
        let span = tracing::info_span!(
            "llm.complete",
            provider = "openai",
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
        debug!(model = %body["model"], "OpenAI completion request");

        let response = self.send_request(&body).await?;
        let oai: OaiResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let choice = oai
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| BlazenError::invalid_response("empty choices array"))?;

        let tool_calls = choice
            .message
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

        let usage = oai.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
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
        if let Some(ref reason) = choice.finish_reason {
            span.record("finish_reason", reason.as_str());
        }

        let cost = usage
            .as_ref()
            .and_then(|u| crate::pricing::compute_cost(&oai.model, u));

        // TODO(phase-4.2): OpenAI o-series exposes `reasoning_summary` only via the
        // Responses API (`/v1/responses`), not chat completions. Once the fal.rs
        // Responses-API body builder lands in Phase 4.2, the parser there will
        // populate `CompletionResponse.reasoning` for o-series models. This direct
        // chat-completions code path can stay as-is — o-series users should call the
        // Responses endpoint via fal.
        Ok(CompletionResponse {
            content: choice.message.content,
            tool_calls,
            reasoning: None,
            citations: vec![],
            artifacts: vec![],
            usage,
            model: oai.model,
            finish_reason: choice.finish_reason,
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
        let model_id = request.model.as_deref().unwrap_or(&self.default_model);
        let span = tracing::info_span!(
            "llm.stream",
            provider = "openai",
            model = %model_id,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let body = self.build_body(&request, true);
        debug!(model = %body["model"], "OpenAI streaming request");

        let http_request = self.build_http_request(&body)?;
        let (status, headers, byte_stream) = self.client.send_streaming(http_request).await?;

        if !(200..300).contains(&status) {
            // For streaming, we need to handle errors before we start parsing.
            // Read the error from the stream is not practical; use status + headers.
            match status {
                401 => return Err(BlazenError::auth("authentication failed")),
                404 => return Err(BlazenError::model_not_found("model not found")),
                429 => {
                    return Err(BlazenError::RateLimit {
                        retry_after_ms: parse_retry_after(&headers),
                    });
                }
                _ => {
                    let url = format!("{}/chat/completions", self.base_url);
                    return Err(provider_http_error_parts(
                        "openai", &url, status, &headers, "",
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

/// Wire format for the `OpenAI` `/models` endpoint.
#[derive(Debug, Deserialize)]
struct OaiModelsResponse {
    data: Vec<OaiModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OaiModelEntry {
    id: String,
}

#[async_trait]
impl ModelRegistry for OpenAiProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError> {
        let url = format!("{}/models", self.base_url);
        let request = HttpRequest::get(&url).bearer_auth(&self.api_key);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            return Err(provider_http_error("openai", &url, &response));
        }

        let list: OaiModelsResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let models = list
            .data
            .into_iter()
            .map(|entry| {
                let is_chat = entry.id.starts_with("gpt-")
                    || entry.id.starts_with("o3")
                    || entry.id.starts_with("o4")
                    || entry.id.starts_with("chatgpt");
                let is_embedding = entry.id.contains("embedding");

                ModelInfo {
                    id: entry.id,
                    name: None,
                    provider: "openai".into(),
                    context_length: None,
                    pricing: None, // OpenAI /models does not include pricing.
                    capabilities: ModelCapabilities {
                        chat: is_chat,
                        streaming: is_chat,
                        tool_use: is_chat,
                        structured_output: is_chat,
                        vision: is_chat,
                        embeddings: is_embedding,
                        ..Default::default()
                    },
                }
            })
            .collect();

        Ok(models)
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, BlazenError> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}

// ---------------------------------------------------------------------------
// Embedding model
// ---------------------------------------------------------------------------

/// An `OpenAI` embedding model.
///
/// This is a separate struct from [`OpenAiProvider`] because embedding models
/// and chat completion models are fundamentally different endpoints with
/// different capabilities.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::openai::OpenAiEmbeddingModel;
///
/// // Defaults to text-embedding-3-small (1536 dimensions)
/// let embedder = OpenAiEmbeddingModel::new("sk-...");
///
/// // Or use a larger model
/// let embedder = OpenAiEmbeddingModel::new("sk-...")
///     .with_model("text-embedding-3-large", 3072);
/// ```
pub struct OpenAiEmbeddingModel {
    client: Arc<dyn HttpClient>,
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
    api_key: String,
    base_url: String,
    model: String,
    dimensions: usize,
}

impl std::fmt::Debug for OpenAiEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbeddingModel")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl Clone for OpenAiEmbeddingModel {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            retry_config: self.retry_config.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            dimensions: self.dimensions,
        }
    }
}

impl OpenAiEmbeddingModel {
    /// Create a new embedding model targeting the official `OpenAI` endpoint.
    ///
    /// Defaults to `text-embedding-3-small` with 1536 dimensions.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            retry_config: None,
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_owned(),
            model: "text-embedding-3-small".to_owned(),
            dimensions: 1536,
        }
    }

    /// Create a new embedding model with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            retry_config: None,
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_owned(),
            model: "text-embedding-3-small".to_owned(),
            dimensions: 1536,
        }
    }

    /// Set the embedding model and its output dimensionality.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>, dimensions: usize) -> Self {
        self.model = model.into();
        self.dimensions = dimensions;
        self
    }

    /// Use a custom base URL (e.g. for local proxies).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// Return a clone of the underlying HTTP client.
    ///
    /// Escape hatch for power users who need to issue raw HTTP requests
    /// while reusing the same connection pool, TLS config, and timeouts
    /// as this embedding model.
    #[must_use]
    pub fn http_client(&self) -> Arc<dyn HttpClient> {
        Arc::clone(&self.client)
    }

    /// Set the provider-level default retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest"
))]
impl OpenAiEmbeddingModel {
    /// Construct from typed [`ProviderOptions`](crate::types::provider_options::ProviderOptions).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Auth`] if no API key is provided and
    /// `OPENAI_API_KEY` is not set.
    pub fn from_options(
        opts: crate::types::provider_options::ProviderOptions,
    ) -> Result<Self, crate::BlazenError> {
        let api_key = crate::keys::resolve_api_key("openai", opts.api_key)?;
        let mut em = Self::new_with_client(api_key, crate::default_http_client());
        if let Some(url) = opts.base_url {
            em = em.with_base_url(url);
        }
        if let Some(m) = opts.model {
            let dims = em.dimensions;
            em = em.with_model(m, dims);
        }
        Ok(em)
    }
}

/// Wire format for the `OpenAI` embeddings API response.
#[derive(Debug, Deserialize)]
struct OaiEmbeddingResponse {
    data: Vec<OaiEmbeddingData>,
    model: String,
    usage: Option<OaiEmbeddingUsage>,
}

/// A single embedding vector from the `OpenAI` embeddings API.
#[derive(Debug, Deserialize)]
struct OaiEmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

/// Token usage from the `OpenAI` embeddings API.
#[derive(Debug, Deserialize)]
struct OaiEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[async_trait]
impl crate::traits::EmbeddingModel for OpenAiEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config.as_ref()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        Some(Self::http_client(self))
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let request = HttpRequest::post(&url)
            .bearer_auth(&self.api_key)
            .json_body(&body)?;

        let response = self.client.send(request).await?;

        if !response.is_success() {
            return match response.status {
                401 => Err(BlazenError::auth("authentication failed")),
                404 => Err(BlazenError::model_not_found(response.text())),
                429 => Err(BlazenError::RateLimit {
                    retry_after_ms: parse_retry_after(&response.headers),
                }),
                _ => Err(provider_http_error("openai", &url, &response)),
            };
        }

        let oai: OaiEmbeddingResponse = response
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

impl crate::traits::ProviderInfo for OpenAiProvider {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: true,
            model_listing: true,
            embeddings: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider (stub — required as a supertrait for AudioGeneration)
// ---------------------------------------------------------------------------

// OpenAI's TTS is a plain POST/response, not a queue-based compute job, so
// the queue-lifecycle methods return Unsupported. The only reason this impl
// exists is that `AudioGeneration: ComputeProvider` is a trait bound; any
// queue-oriented code calling into the OpenAI provider should return a
// clear "not supported" error rather than compiling at all is unhelpful.

#[async_trait]
impl crate::compute::traits::ComputeProvider for OpenAiProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        "openai"
    }

    async fn submit(
        &self,
        _request: crate::compute::job::ComputeRequest,
    ) -> Result<crate::compute::job::JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI provider does not expose a queue-based compute API",
        ))
    }

    async fn status(
        &self,
        _job: &crate::compute::job::JobHandle,
    ) -> Result<crate::compute::job::JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI provider does not expose a queue-based compute API",
        ))
    }

    async fn result(
        &self,
        _job: crate::compute::job::JobHandle,
    ) -> Result<crate::compute::job::ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI provider does not expose a queue-based compute API",
        ))
    }

    async fn cancel(&self, _job: &crate::compute::job::JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "OpenAI provider does not expose a queue-based compute API",
        ))
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration (text-to-speech via /v1/audio/speech)
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::compute::traits::AudioGeneration for OpenAiProvider {
    async fn text_to_speech(
        &self,
        request: crate::compute::requests::SpeechRequest,
    ) -> Result<crate::compute::results::AudioResult, BlazenError> {
        super::openai_audio::text_to_speech_request(
            self.client.as_ref(),
            &self.base_url,
            &self.api_key,
            request,
        )
        .await
    }
    // generate_music and generate_sfx intentionally NOT overridden —
    // they fall through to the default `Err(Unsupported)` impls in the trait.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ToolDefinition};

    // SSE parsing tests are now in the shared `sse` module.

    #[test]
    fn build_body_minimal() {
        let provider = OpenAiProvider::new("test-key");
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
        let provider = OpenAiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")])
            .with_temperature(0.5)
            .with_max_tokens(100)
            .with_model("gpt-4.1-mini");

        let body = provider.build_body(&request, true);
        assert_eq!(body["model"], "gpt-4.1-mini");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_completion_tokens"], 100);
    }

    #[test]
    fn build_body_with_tools() {
        let provider = OpenAiProvider::new("test-key");
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
        let provider = OpenAiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        // Text messages should produce a plain string, not an array.
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_body_image_url() {
        let provider = OpenAiProvider::new("test-key");
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
        let provider = OpenAiProvider::new("test-key");
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

        let provider = OpenAiProvider::new("test-key");
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

    // Verify SSE tests still pass through shared module
    #[test]
    fn shared_sse_parse_text_delta() {
        use crate::providers::sse::parse_next_event;

        let mut buf = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}],\"model\":\"gpt-4o\"}\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert_eq!(result.delta.as_deref(), Some("Hello"));
        assert!(result.finish_reason.is_none());
    }

    #[test]
    fn shared_sse_parse_done() {
        use crate::providers::sse::parse_next_event;

        let mut buf = "data: [DONE]\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert!(result.delta.is_none());
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    }

    // -----------------------------------------------------------------------
    // Embedding model tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_openai_embedding_default() {
        use crate::traits::EmbeddingModel;

        let embedder = OpenAiEmbeddingModel::new("sk-test");
        assert_eq!(embedder.model_id(), "text-embedding-3-small");
        assert_eq!(embedder.dimensions(), 1536);
    }

    #[test]
    fn test_openai_embedding_with_model() {
        use crate::traits::EmbeddingModel;

        let embedder =
            OpenAiEmbeddingModel::new("sk-test").with_model("text-embedding-3-large", 3072);
        assert_eq!(embedder.model_id(), "text-embedding-3-large");
        assert_eq!(embedder.dimensions(), 3072);
    }

    #[test]
    fn test_openai_embedding_with_base_url() {
        let embedder =
            OpenAiEmbeddingModel::new("sk-test").with_base_url("https://custom.proxy.com/v1");
        assert_eq!(embedder.base_url, "https://custom.proxy.com/v1");
    }

    #[test]
    fn test_openai_embedding_response_parsing() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 1},
                {"embedding": [0.4, 0.5, 0.6], "index": 0}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }"#;

        let oai: OaiEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(oai.data.len(), 2);
        assert_eq!(oai.model, "text-embedding-3-small");

        // Verify reordering by index works
        let mut embeddings: Vec<(usize, Vec<f32>)> = oai
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        embeddings.sort_by_key(|(idx, _)| *idx);
        let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, v)| v).collect();

        assert_eq!(embeddings[0], vec![0.4, 0.5, 0.6]); // index 0
        assert_eq!(embeddings[1], vec![0.1, 0.2, 0.3]); // index 1

        let usage = oai.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.total_tokens, 10);
    }
}
