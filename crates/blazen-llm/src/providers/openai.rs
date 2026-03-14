//! `OpenAI` chat completion provider.
//!
//! This module provides the original [`OpenAiProvider`] for direct use with the
//! `OpenAI` API. For connecting to other `OpenAI`-compatible services (`OpenRouter`,
//! Groq, Together AI, etc.), see [`super::openai_compat::OpenAiCompatProvider`].

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use reqwest::Client;
use tracing::debug;

use super::sse::{OaiResponse, SseParser};
use crate::error::LlmError;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource, MessageContent,
    Role, StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Multimodal helpers
// ---------------------------------------------------------------------------

/// Convert an [`ImageContent`] to the OpenAI `image_url` content-part format.
fn image_content_to_openai(img: &ImageContent) -> serde_json::Value {
    let url = match &img.source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { data } => {
            let media_type = img.media_type.as_deref().unwrap_or("image/png");
            format!("data:{media_type};base64,{data}")
        }
    };
    serde_json::json!({
        "type": "image_url",
        "image_url": { "url": url }
    })
}

/// Convert a single [`ContentPart`] to an OpenAI content-array element.
fn content_part_to_openai(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => {
            serde_json::json!({ "type": "text", "text": text })
        }
        ContentPart::Image(img) => image_content_to_openai(img),
        ContentPart::File(file) => {
            // Files are sent as image_url with a data URI (best-effort for
            // OpenAI-compatible endpoints).
            let url = match &file.source {
                ImageSource::Url { url } => url.clone(),
                ImageSource::Base64 { data } => {
                    format!("data:{};base64,{data}", file.media_type)
                }
            };
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": url }
            })
        }
    }
}

/// Convert [`MessageContent`] to a `serde_json::Value` suitable for the
/// OpenAI `content` field.
///
/// - `Text` -> a plain JSON string (backward-compatible).
/// - `Image` / `Parts` -> a JSON array of content parts.
fn content_to_openai_value(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text(t) => serde_json::Value::String(t.clone()),
        MessageContent::Image(img) => {
            serde_json::json!([image_content_to_openai(img)])
        }
        MessageContent::Parts(parts) => {
            let arr: Vec<serde_json::Value> = parts.iter().map(content_part_to_openai).collect();
            serde_json::json!(arr)
        }
    }
}

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
///     .with_model("gpt-4o-mini");
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl OpenAiProvider {
    /// Create a new provider with the given API key, targeting the official
    /// `OpenAI` endpoint.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_owned(),
            default_model: "gpt-4o".to_owned(),
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

    /// Build the JSON request body for the `OpenAI` chat completions endpoint.
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

        body
    }

    /// Send a request and return the raw response, handling common errors.
    async fn send_request(&self, body: &serde_json::Value) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        // Try to read the error body for better diagnostics.
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| String::from("<unable to read error body>"));

        match status.as_u16() {
            401 => Err(LlmError::AuthFailed),
            404 => Err(LlmError::ModelNotFound(error_body)),
            429 => {
                // TODO: parse Retry-After header if present.
                Err(LlmError::RateLimited { retry_after: None })
            }
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
impl crate::traits::CompletionModel for OpenAiProvider {
    fn model_id(&self) -> &str {
        &self.default_model
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = self.build_body(&request, false);
        debug!(model = %body["model"], "OpenAI completion request");

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
        debug!(model = %body["model"], "OpenAI streaming request");

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
        };

        let body = provider.build_body(&request, false);
        assert_eq!(body["model"], "gpt-4o");
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
            .with_model("gpt-4o-mini");

        let body = provider.build_body(&request, true);
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_tokens"], 100);
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
}
