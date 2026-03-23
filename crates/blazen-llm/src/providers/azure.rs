//! Azure `OpenAI` chat completion provider.
//!
//! Azure `OpenAI` uses the same wire format as `OpenAI` but with different URL
//! structure and authentication:
//!
//! - URL: `https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}`
//! - Auth: `api-key: <key>` header (not Bearer)
//!
//! The SSE streaming and request/response formats are identical to `OpenAI`.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use tracing::debug;

use super::openai_format::{content_to_openai_value, parse_retry_after};
use super::sse::{OaiResponse, SseParser};
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest, HttpResponse};
use crate::traits::{ModelCapabilities, ModelInfo, ModelRegistry};
use crate::types::{
    CompletionRequest, CompletionResponse, Role, StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default Azure `OpenAI` API version.
const DEFAULT_API_VERSION: &str = "2025-04-01-preview";

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// An Azure `OpenAI` chat completion provider.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::azure::AzureOpenAiProvider;
///
/// let provider = AzureOpenAiProvider::new(
///     "my-api-key",
///     "my-resource",
///     "gpt-4o-deployment",
/// );
/// ```
pub struct AzureOpenAiProvider {
    client: Arc<dyn HttpClient>,
    api_key: String,
    resource_name: String,
    deployment_name: String,
    api_version: String,
}

impl std::fmt::Debug for AzureOpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAiProvider")
            .field("resource_name", &self.resource_name)
            .field("deployment_name", &self.deployment_name)
            .field("api_version", &self.api_version)
            .finish_non_exhaustive()
    }
}

impl Clone for AzureOpenAiProvider {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            api_key: self.api_key.clone(),
            resource_name: self.resource_name.clone(),
            deployment_name: self.deployment_name.clone(),
            api_version: self.api_version.clone(),
        }
    }
}

impl AzureOpenAiProvider {
    /// Create a new Azure `OpenAI` provider.
    ///
    /// - `api_key`: The Azure API key.
    /// - `resource_name`: The Azure `OpenAI` resource name (the subdomain).
    /// - `deployment_name`: The model deployment name.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        resource_name: impl Into<String>,
        deployment_name: impl Into<String>,
    ) -> Self {
        Self {
            client: crate::default_http_client(),
            api_key: api_key.into(),
            resource_name: resource_name.into(),
            deployment_name: deployment_name.into(),
            api_version: DEFAULT_API_VERSION.to_owned(),
        }
    }

    /// Create a new Azure `OpenAI` provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(
        api_key: impl Into<String>,
        resource_name: impl Into<String>,
        deployment_name: impl Into<String>,
        client: Arc<dyn HttpClient>,
    ) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            resource_name: resource_name.into(),
            deployment_name: deployment_name.into(),
            api_version: DEFAULT_API_VERSION.to_owned(),
        }
    }

    /// Override the API version.
    #[must_use]
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// Get the full endpoint URL for chat completions.
    fn completions_url(&self) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource_name, self.deployment_name, self.api_version
        )
    }

    /// Build the JSON request body (same format as `OpenAI`, but without the
    /// `model` field since the deployment determines the model).
    #[allow(clippy::unused_self)]
    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
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

        body
    }

    /// Build an [`HttpRequest`] for the Azure chat completions endpoint.
    fn build_http_request(&self, body: &serde_json::Value) -> Result<HttpRequest, BlazenError> {
        let url = self.completions_url();
        HttpRequest::post(url)
            .header("api-key", &self.api_key)
            .json_body(body)
    }

    /// Send a request and return the raw response, handling common errors.
    async fn send_request(&self, body: &serde_json::Value) -> Result<HttpResponse, BlazenError> {
        let request = self.build_http_request(body)?;
        let response = self.client.send(request).await?;

        if response.is_success() {
            return Ok(response);
        }

        // Extract Retry-After before inspecting the body.
        let retry_after_ms = parse_retry_after(&response.headers);
        let error_body = response.text();

        match response.status {
            401 => Err(BlazenError::auth("authentication failed")),
            404 => Err(BlazenError::model_not_found(error_body)),
            429 => Err(BlazenError::RateLimit { retry_after_ms }),
            status => Err(BlazenError::request(format!("HTTP {status}: {error_body}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for AzureOpenAiProvider {
    fn model_id(&self) -> &str {
        &self.deployment_name
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model_id = request.model.as_deref().unwrap_or(&self.deployment_name);
        let span = tracing::info_span!(
            "llm.complete",
            provider = "azure",
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
        debug!(deployment = %self.deployment_name, "Azure OpenAI completion request");

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

        Ok(CompletionResponse {
            content: choice.message.content,
            tool_calls,
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
        let model_id = request.model.as_deref().unwrap_or(&self.deployment_name);
        let span = tracing::info_span!(
            "llm.stream",
            provider = "azure",
            model = %model_id,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let body = self.build_body(&request, true);
        debug!(deployment = %self.deployment_name, "Azure OpenAI streaming request");

        let http_request = self.build_http_request(&body)?;
        let (status, headers, byte_stream) = self.client.send_streaming(http_request).await?;

        if !(200..300).contains(&status) {
            let retry_after_ms = parse_retry_after(&headers);
            match status {
                401 => return Err(BlazenError::auth("authentication failed")),
                404 => return Err(BlazenError::model_not_found("model not found")),
                429 => return Err(BlazenError::RateLimit { retry_after_ms }),
                _ => return Err(BlazenError::request(format!("HTTP {status}"))),
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

/// Wire format for the Azure `OpenAI` `/models` endpoint.
#[derive(Debug, Deserialize)]
struct AzureModelsResponse {
    data: Vec<AzureModelEntry>,
}

#[derive(Debug, Deserialize)]
struct AzureModelEntry {
    id: String,
}

#[async_trait]
impl ModelRegistry for AzureOpenAiProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError> {
        let url = format!(
            "https://{}.openai.azure.com/openai/models?api-version={}",
            self.resource_name, self.api_version
        );
        let request = HttpRequest::get(&url).header("api-key", &self.api_key);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "HTTP {}: {error_body}",
                response.status
            )));
        }

        let list: AzureModelsResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let models = list
            .data
            .into_iter()
            .map(|entry| ModelInfo {
                id: entry.id,
                name: None,
                provider: "azure".into(),
                context_length: None,
                pricing: None, // Azure pricing is per-deployment/region, not in API.
                capabilities: ModelCapabilities {
                    chat: true,
                    streaming: true,
                    tool_use: true,
                    structured_output: true,
                    ..Default::default()
                },
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
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl crate::traits::ProviderInfo for AzureOpenAiProvider {
    fn provider_name(&self) -> &str {
        "azure"
    }

    fn base_url(&self) -> &str {
        // Azure uses a dynamic URL based on resource + deployment, not a static base URL.
        // Return the resource-level base.
        "https://openai.azure.com"
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: true,
            model_listing: true,
            embeddings: false,
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
    fn completions_url_format() {
        let provider = AzureOpenAiProvider::new("key", "my-resource", "gpt-4o");
        let url = provider.completions_url();
        assert_eq!(
            url,
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-04-01-preview"
        );
    }

    #[test]
    fn completions_url_custom_version() {
        let provider =
            AzureOpenAiProvider::new("key", "my-resource", "gpt-4o").with_api_version("2025-01-01");
        let url = provider.completions_url();
        assert!(url.contains("api-version=2025-01-01"));
    }

    #[test]
    fn model_id_is_deployment() {
        let provider = AzureOpenAiProvider::new("key", "res", "my-deployment");
        assert_eq!(
            crate::traits::CompletionModel::model_id(&provider),
            "my-deployment"
        );
    }

    #[test]
    fn build_body_no_model_field() {
        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        // Azure doesn't need a "model" field since the deployment determines it.
        assert!(body.get("model").is_none());
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn build_body_with_options() {
        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")])
            .with_temperature(0.5)
            .with_max_tokens(100);

        let body = provider.build_body(&request, true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_completion_tokens"], 100);
    }

    #[test]
    fn build_body_with_tools() {
        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]).with_tools(vec![
            ToolDefinition {
                name: "search".to_owned(),
                description: "Search".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ]);

        let body = provider.build_body(&request, false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "search");
    }

    #[test]
    fn test_text_backward_compat() {
        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_body_image_url() {
        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
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
        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
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

        let provider = AzureOpenAiProvider::new("key", "res", "deploy");
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
}
