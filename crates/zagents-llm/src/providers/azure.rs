//! Azure OpenAI chat completion provider.
//!
//! Azure OpenAI uses the same wire format as OpenAI but with different URL
//! structure and authentication:
//!
//! - URL: `https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}`
//! - Auth: `api-key: <key>` header (not Bearer)
//!
//! The SSE streaming and request/response formats are identical to OpenAI.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use reqwest::Client;
use tracing::debug;

use super::sse::{OaiResponse, SseParser};
use crate::error::LlmError;
use crate::types::{
    CompletionRequest, CompletionResponse, MessageContent, Role, StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default Azure OpenAI API version.
const DEFAULT_API_VERSION: &str = "2024-10-21";

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// An Azure OpenAI chat completion provider.
///
/// # Examples
///
/// ```rust,no_run
/// use zagents_llm::providers::azure::AzureOpenAiProvider;
///
/// let provider = AzureOpenAiProvider::new(
///     "my-api-key",
///     "my-resource",
///     "gpt-4o-deployment",
/// );
/// ```
#[derive(Debug, Clone)]
pub struct AzureOpenAiProvider {
    client: Client,
    api_key: String,
    resource_name: String,
    deployment_name: String,
    api_version: String,
}

impl AzureOpenAiProvider {
    /// Create a new Azure OpenAI provider.
    ///
    /// - `api_key`: The Azure API key.
    /// - `resource_name`: The Azure OpenAI resource name (the subdomain).
    /// - `deployment_name`: The model deployment name.
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        resource_name: impl Into<String>,
        deployment_name: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
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

    /// Get the full endpoint URL for chat completions.
    fn completions_url(&self) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource_name, self.deployment_name, self.api_version
        )
    }

    /// Build the JSON request body (same format as OpenAI, but without the
    /// `model` field since the deployment determines the model).
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
                let content = match &m.content {
                    MessageContent::Text(t) => serde_json::Value::String(t.clone()),
                };
                serde_json::json!({ "role": role, "content": content })
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

        body
    }

    /// Send a request and return the raw response, handling common errors.
    async fn send_request(
        &self,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        let url = self.completions_url();

        let response = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| String::from("<unable to read error body>"));

        match status.as_u16() {
            401 => Err(LlmError::AuthFailed),
            404 => Err(LlmError::ModelNotFound(error_body)),
            429 => Err(LlmError::RateLimited { retry_after: None }),
            _ => Err(LlmError::RequestFailed(format!(
                "HTTP {status}: {error_body}"
            ))),
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

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = self.build_body(&request, false);
        debug!(deployment = %self.deployment_name, "Azure OpenAI completion request");

        let response = self.send_request(&body).await?;
        let oai: OaiResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let choice = oai
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::InvalidResponse("empty choices array".into()))?;

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

        Ok(CompletionResponse {
            content: choice.message.content,
            tool_calls,
            usage,
            model: oai.model,
            finish_reason: choice.finish_reason,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError> {
        let body = self.build_body(&request, true);
        debug!(deployment = %self.deployment_name, "Azure OpenAI streaming request");

        let response = self.send_request(&body).await?;
        let byte_stream = response.bytes_stream();

        let stream = SseParser::new(byte_stream);
        Ok(Box::pin(stream))
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
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn completions_url_custom_version() {
        let provider =
            AzureOpenAiProvider::new("key", "my-resource", "gpt-4o")
                .with_api_version("2025-01-01");
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
        assert_eq!(body["max_tokens"], 100);
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
}
