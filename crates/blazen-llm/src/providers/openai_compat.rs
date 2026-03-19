//! Generic OpenAI-compatible chat completion provider.
//!
//! Most hosted LLM APIs follow the `OpenAI` chat completions wire format. This
//! module provides [`OpenAiCompatProvider`] -- a single implementation of
//! [`CompletionModel`] that works with any OpenAI-compatible endpoint by
//! configuring the base URL, auth method, and extra headers.
//!
//! Convenience constructors are provided for popular providers:
//!
//! | Constructor | Provider | Default model |
//! |-------------|----------|---------------|
//! | [`OpenAiCompatProvider::openai`] | OpenAI | `gpt-4o` |
//! | [`OpenAiCompatProvider::openrouter`] | OpenRouter | `openai/gpt-4o` |
//! | [`OpenAiCompatProvider::groq`] | Groq | `llama-3.3-70b-versatile` |
//! | [`OpenAiCompatProvider::together`] | Together AI | `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
//! | [`OpenAiCompatProvider::mistral`] | Mistral AI | `mistral-large-latest` |
//! | [`OpenAiCompatProvider::deepseek`] | DeepSeek | `deepseek-chat` |
//! | [`OpenAiCompatProvider::fireworks`] | Fireworks AI | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
//! | [`OpenAiCompatProvider::perplexity`] | Perplexity | `sonar-pro` |
//! | [`OpenAiCompatProvider::xai`] | xAI (Grok) | `grok-3` |
//! | [`OpenAiCompatProvider::cohere`] | Cohere | `command-a-03-2025` |
//! | [`OpenAiCompatProvider::bedrock`] | AWS Bedrock (Mantle) | `anthropic.claude-sonnet-4-20250514-v1:0` |

use std::pin::Pin;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use reqwest::Client;
use serde::Deserialize;
use tracing::debug;

use super::openai_format::{content_to_openai_value, parse_retry_after};
use super::sse::{OaiResponse, SseParser};
use crate::error::BlazenError;
use crate::traits::{ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry};
use crate::types::{
    CompletionRequest, CompletionResponse, Role, StreamChunk, TokenUsage, ToolCall,
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
/// Use the convenience constructors ([`Self::openai`], [`Self::openrouter`],
/// etc.) or build a custom configuration with [`Self::new`].
#[derive(Debug, Clone)]
pub struct OpenAiCompatProvider {
    config: OpenAiCompatConfig,
    client: Client,
}

impl OpenAiCompatProvider {
    /// Create a provider from a fully-specified configuration.
    #[must_use]
    pub fn new(config: OpenAiCompatConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Convenience constructors
    // -----------------------------------------------------------------------

    /// `OpenAI` (`https://api.openai.com/v1`, default model `gpt-4o`).
    #[must_use]
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: api_key.into(),
            default_model: "gpt-4o".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// `OpenRouter` (`https://openrouter.ai/api/v1`, default model `openai/gpt-4o`).
    #[must_use]
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "openrouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            api_key: api_key.into(),
            default_model: "openai/gpt-4o".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// Groq (`https://api.groq.com/openai/v1`, default model `llama-3.3-70b-versatile`).
    #[must_use]
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "groq".into(),
            base_url: "https://api.groq.com/openai/v1".into(),
            api_key: api_key.into(),
            default_model: "llama-3.3-70b-versatile".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// Together AI (`https://api.together.xyz/v1`, default model `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`).
    #[must_use]
    pub fn together(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "together".into(),
            base_url: "https://api.together.xyz/v1".into(),
            api_key: api_key.into(),
            default_model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// Mistral AI (`https://api.mistral.ai/v1`, default model `mistral-large-latest`).
    #[must_use]
    pub fn mistral(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "mistral".into(),
            base_url: "https://api.mistral.ai/v1".into(),
            api_key: api_key.into(),
            default_model: "mistral-large-latest".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// `DeepSeek` (`https://api.deepseek.com`, default model `deepseek-chat`).
    #[must_use]
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "deepseek".into(),
            base_url: "https://api.deepseek.com".into(),
            api_key: api_key.into(),
            default_model: "deepseek-chat".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: false,
        })
    }

    /// Fireworks AI (`https://api.fireworks.ai/inference/v1`, default model `accounts/fireworks/models/llama-v3p1-70b-instruct`).
    #[must_use]
    pub fn fireworks(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "fireworks".into(),
            base_url: "https://api.fireworks.ai/inference/v1".into(),
            api_key: api_key.into(),
            default_model: "accounts/fireworks/models/llama-v3p1-70b-instruct".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// Perplexity (`https://api.perplexity.ai`, default model `sonar-pro`).
    #[must_use]
    pub fn perplexity(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "perplexity".into(),
            base_url: "https://api.perplexity.ai".into(),
            api_key: api_key.into(),
            default_model: "sonar-pro".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: false,
        })
    }

    /// xAI / Grok (`https://api.x.ai/v1`, default model `grok-3`).
    #[must_use]
    pub fn xai(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "xai".into(),
            base_url: "https://api.x.ai/v1".into(),
            api_key: api_key.into(),
            default_model: "grok-3".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

    /// Cohere (via compatibility endpoint, default model `command-a-03-2025`).
    #[must_use]
    pub fn cohere(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiCompatConfig {
            provider_name: "cohere".into(),
            base_url: "https://api.cohere.ai/compatibility/v1".into(),
            api_key: api_key.into(),
            default_model: "command-a-03-2025".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: false,
        })
    }

    /// AWS Bedrock via Mantle OpenAI-compatible endpoint.
    ///
    /// URL pattern: `https://bedrock-mantle.{region}.api.aws/v1`
    #[must_use]
    pub fn bedrock(api_key: impl Into<String>, region: impl Into<String>) -> Self {
        let region = region.into();
        Self::new(OpenAiCompatConfig {
            provider_name: "bedrock".into(),
            base_url: format!("https://bedrock-mantle.{region}.api.aws/v1"),
            api_key: api_key.into(),
            default_model: "anthropic.claude-sonnet-4-20250514-v1:0".into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        })
    }

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
                serde_json::json!({ "role": role, "content": content })
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
            body["response_format"] = serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": fmt,
                    "strict": true,
                }
            });
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

    /// Apply authentication to a request builder.
    fn apply_auth(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.config.auth_method {
            AuthMethod::Bearer => {
                builder = builder.bearer_auth(&self.config.api_key);
            }
            AuthMethod::ApiKeyHeader(header_name) => {
                builder = builder.header(header_name.as_str(), &self.config.api_key);
            }
            AuthMethod::AzureApiKey => {
                builder = builder.header("api-key", &self.config.api_key);
            }
            AuthMethod::KeyPrefix => {
                builder = builder.header("Authorization", format!("Key {}", self.config.api_key));
            }
        }

        // Apply extra headers
        for (key, value) in &self.config.extra_headers {
            builder = builder.header(key.as_str(), value.as_str());
        }

        builder
    }

    /// Apply query parameters to a request builder.
    fn apply_query_params(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (key, value) in &self.config.query_params {
            builder = builder.query(&[(key.as_str(), value.as_str())]);
        }
        builder
    }

    /// Send a request and return the raw response, handling common HTTP errors.
    async fn send_request(
        &self,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, BlazenError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let builder = self.client.post(&url);
        let builder = self.apply_auth(builder);
        let builder = self.apply_query_params(builder);

        let response = builder
            .json(body)
            .send()
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        // Extract Retry-After before consuming the body.
        let retry_after_ms = parse_retry_after(response.headers());

        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| String::from("<unable to read error body>"));

        match status.as_u16() {
            401 => Err(BlazenError::auth("authentication failed")),
            404 => Err(BlazenError::model_not_found(error_body)),
            429 => Err(BlazenError::RateLimit { retry_after_ms }),
            _ => Err(BlazenError::request(format!("HTTP {status}: {error_body}"))),
        }
    }
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
            .await
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

        Ok(CompletionResponse {
            content: choice.message.content,
            tool_calls,
            usage,
            model: oai.model,
            finish_reason: choice.finish_reason,
            cost: None,
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

        let response = self.send_request(&body).await?;
        let byte_stream = response.bytes_stream();

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
///
/// This struct is intentionally permissive (all `Option`) to handle the wide
/// variety of response shapes across providers.
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

        let builder = self.client.get(&url);
        let builder = self.apply_auth(builder);
        let builder = self.apply_query_params(builder);

        let response = builder
            .send()
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unable to read error body>"));
            return Err(BlazenError::request(format!("HTTP {status}: {error_body}")));
        }

        let body_text = response
            .text()
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

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

        ModelInfo {
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
        }
    }

    /// Convert a Together AI model entry to [`ModelInfo`].
    fn together_entry_to_info(&self, entry: TogetherModelEntry) -> ModelInfo {
        let pricing = entry.pricing.map(|p| ModelPricing {
            input_per_million: p.input,
            output_per_million: p.output,
            ..Default::default()
        });

        ModelInfo {
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

    #[test]
    fn openai_defaults() {
        let provider = OpenAiCompatProvider::openai("sk-test");
        assert_eq!(provider.config.base_url, "https://api.openai.com/v1");
        assert_eq!(provider.config.default_model, "gpt-4o");
        assert!(matches!(provider.config.auth_method, AuthMethod::Bearer));
        assert!(provider.config.supports_model_listing);
    }

    #[test]
    fn openrouter_defaults() {
        let provider = OpenAiCompatProvider::openrouter("or-test");
        assert_eq!(provider.config.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(provider.config.default_model, "openai/gpt-4o");
    }

    #[test]
    fn groq_defaults() {
        let provider = OpenAiCompatProvider::groq("gsk-test");
        assert_eq!(provider.config.base_url, "https://api.groq.com/openai/v1");
        assert_eq!(provider.config.default_model, "llama-3.3-70b-versatile");
    }

    #[test]
    fn together_defaults() {
        let provider = OpenAiCompatProvider::together("tog-test");
        assert_eq!(provider.config.base_url, "https://api.together.xyz/v1");
        assert_eq!(
            provider.config.default_model,
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        );
    }

    #[test]
    fn mistral_defaults() {
        let provider = OpenAiCompatProvider::mistral("ms-test");
        assert_eq!(provider.config.base_url, "https://api.mistral.ai/v1");
        assert_eq!(provider.config.default_model, "mistral-large-latest");
    }

    #[test]
    fn deepseek_defaults() {
        let provider = OpenAiCompatProvider::deepseek("ds-test");
        assert_eq!(provider.config.base_url, "https://api.deepseek.com");
        assert_eq!(provider.config.default_model, "deepseek-chat");
        assert!(!provider.config.supports_model_listing);
    }

    #[test]
    fn fireworks_defaults() {
        let provider = OpenAiCompatProvider::fireworks("fw-test");
        assert_eq!(
            provider.config.base_url,
            "https://api.fireworks.ai/inference/v1"
        );
    }

    #[test]
    fn perplexity_defaults() {
        let provider = OpenAiCompatProvider::perplexity("pplx-test");
        assert_eq!(provider.config.base_url, "https://api.perplexity.ai");
        assert_eq!(provider.config.default_model, "sonar-pro");
        assert!(!provider.config.supports_model_listing);
    }

    #[test]
    fn xai_defaults() {
        let provider = OpenAiCompatProvider::xai("xai-test");
        assert_eq!(provider.config.base_url, "https://api.x.ai/v1");
        assert_eq!(provider.config.default_model, "grok-3");
    }

    #[test]
    fn cohere_defaults() {
        let provider = OpenAiCompatProvider::cohere("co-test");
        assert_eq!(
            provider.config.base_url,
            "https://api.cohere.ai/compatibility/v1"
        );
        assert_eq!(provider.config.default_model, "command-a-03-2025");
        assert!(!provider.config.supports_model_listing);
    }

    #[test]
    fn bedrock_defaults() {
        let provider = OpenAiCompatProvider::bedrock("bk-test", "us-east-1");
        assert_eq!(
            provider.config.base_url,
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        );
        assert_eq!(
            provider.config.default_model,
            "anthropic.claude-sonnet-4-20250514-v1:0"
        );
    }

    #[test]
    fn with_model_override() {
        let provider = OpenAiCompatProvider::openai("sk-test").with_model("gpt-4o-mini");
        assert_eq!(provider.config.default_model, "gpt-4o-mini");
    }

    #[test]
    fn with_header_appends() {
        let provider = OpenAiCompatProvider::openrouter("or-test")
            .with_header("HTTP-Referer", "https://myapp.com")
            .with_header("X-Title", "My App");
        assert_eq!(provider.config.extra_headers.len(), 2);
    }

    #[test]
    fn build_body_minimal() {
        let provider = OpenAiCompatProvider::openai("test-key");
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
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        assert!(body.get("temperature").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn build_body_with_options() {
        let provider = OpenAiCompatProvider::openai("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")])
            .with_temperature(0.5)
            .with_max_tokens(100)
            .with_model("gpt-4o-mini");

        let body = provider.build_body(&request, true);
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn build_body_with_tools() {
        let provider = OpenAiCompatProvider::groq("test-key");
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
        let provider = OpenAiCompatProvider::openai("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_body_image_url() {
        let provider = OpenAiCompatProvider::openai("test-key");
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
        let provider = OpenAiCompatProvider::openai("test-key");
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

        let provider = OpenAiCompatProvider::openrouter("test-key");
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
    fn parse_standard_model_list() {
        let provider = OpenAiCompatProvider::openai("test-key");
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
        assert_eq!(models[0].provider, "openai");
    }

    #[test]
    fn parse_openrouter_model_with_pricing() {
        let provider = OpenAiCompatProvider::openrouter("test-key");
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
        let provider = OpenAiCompatProvider::together("test-key");
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
}
