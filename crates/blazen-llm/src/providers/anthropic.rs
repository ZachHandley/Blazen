//! Anthropic Messages API provider.
//!
//! This module implements [`CompletionModel`] for the Anthropic Messages API.
//! The Anthropic API differs from `OpenAI` in several important ways:
//!
//! - Authentication uses the `x-api-key` header (not Bearer auth).
//! - A `anthropic-version` header is required.
//! - The `system` prompt is a top-level field, not part of the messages array.
//! - `max_tokens` is required.
//! - The response format uses content blocks instead of a single string.
//! - Streaming uses typed SSE events (`message_start`, `content_block_delta`,
//!   etc.) rather than generic `data:` lines.

use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::LlmError;
use crate::types::{
    CompletionRequest, CompletionResponse, MessageContent, Role, StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The required Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default max tokens when the caller does not specify one.
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// An Anthropic Messages API provider.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::anthropic::AnthropicProvider;
///
/// let provider = AnthropicProvider::new("sk-ant-...")
///     .with_model("claude-sonnet-4-20250514");
/// ```
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl AnthropicProvider {
    /// Create a new provider with the given API key, targeting the official
    /// Anthropic endpoint.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_owned(),
            default_model: "claude-sonnet-4-20250514".to_owned(),
        }
    }

    /// Use a custom base URL.
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

    /// Build the JSON request body for the Anthropic Messages endpoint.
    ///
    /// Key differences from `OpenAI`:
    /// - System messages are extracted into the top-level `system` field.
    /// - `max_tokens` is always present (required by the API).
    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let model = request.model.as_deref().unwrap_or(&self.default_model);

        // Separate system messages from the conversation messages.
        let mut system_parts: Vec<String> = Vec::new();
        let mut messages: Vec<serde_json::Value> = Vec::new();

        for msg in &request.messages {
            if msg.role == Role::System {
                let MessageContent::Text(t) = &msg.content;
                system_parts.push(t.clone());
            } else {
                let role = match msg.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "assistant",
                    Role::System => unreachable!(),
                };
                let content = match &msg.content {
                    MessageContent::Text(t) => t.clone(),
                };
                messages.push(serde_json::json!({
                    "role": role,
                    "content": content,
                }));
            }
        }

        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        });

        if !system_parts.is_empty() {
            body["system"] = serde_json::Value::String(system_parts.join("\n\n"));
        }

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        body
    }

    /// Send a request to the Messages endpoint, handling common HTTP errors.
    async fn send_request(&self, body: &serde_json::Value) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}/messages", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
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
// Wire types (response deserialization)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// Streaming wire types
//
// These structs mirror the Anthropic SSE wire format. Some fields are present
// for correct deserialization but are not read by the mapping logic, hence the
// `#[allow(dead_code)]` annotations.

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartPayload },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStartPayload,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDeltaPayload,
    },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaPayload,
        usage: Option<MessageDeltaUsage>,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: StreamErrorPayload },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageStartPayload {
    model: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ContentBlockStartPayload {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlockDeltaPayload {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaPayload {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageDeltaUsage {
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct StreamErrorPayload {
    message: String,
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for AnthropicProvider {
    fn model_id(&self) -> &str {
        &self.default_model
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = self.build_body(&request, false);
        debug!(model = %body["model"], "Anthropic completion request");

        let response = self.send_request(&body).await?;
        let anthropic: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        // Extract text content and tool calls from content blocks.
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in anthropic.content {
            match block {
                ContentBlock::Text { text } => {
                    text_parts.push(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
            }
        }

        let content = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        let usage = anthropic.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        Ok(CompletionResponse {
            content,
            tool_calls,
            usage,
            model: anthropic.model,
            finish_reason: anthropic.stop_reason,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError> {
        let body = self.build_body(&request, true);
        debug!(model = %body["model"], "Anthropic streaming request");

        let response = self.send_request(&body).await?;
        let byte_stream = response.bytes_stream();

        let stream = AnthropicSseParser::new(byte_stream);
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// SSE stream parser
// ---------------------------------------------------------------------------

/// Parses an SSE byte stream from the Anthropic Messages API into
/// [`StreamChunk`]s.
///
/// Anthropic SSE uses typed events. Each SSE frame has an `event:` line
/// followed by a `data:` line. We parse these pairs into [`StreamEvent`]
/// variants and then map them to our provider-agnostic [`StreamChunk`].
struct AnthropicSseParser<S> {
    inner: S,
    buffer: String,
    /// Track in-progress tool use blocks by index.
    tool_blocks: Vec<ToolBlockState>,
}

/// State for a `tool_use` content block being streamed.
#[derive(Debug, Clone)]
struct ToolBlockState {
    id: String,
    name: String,
    json_buf: String,
}

impl<S> AnthropicSseParser<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            tool_blocks: Vec::new(),
        }
    }
}

impl<S> Stream for AnthropicSseParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
{
    type Item = Result<StreamChunk, LlmError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Try to extract a complete SSE event pair from the buffer.
            if let Some(result) = parse_anthropic_event(&mut this.buffer, &mut this.tool_blocks) {
                return std::task::Poll::Ready(Some(result));
            }

            match Pin::new(&mut this.inner).poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes);
                    this.buffer.push_str(&text);
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(LlmError::Stream(e.to_string()))));
                }
                std::task::Poll::Ready(None) => {
                    if !this.buffer.is_empty()
                        && let Some(result) =
                            parse_anthropic_event(&mut this.buffer, &mut this.tool_blocks)
                    {
                        return std::task::Poll::Ready(Some(result));
                    }
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

/// Try to extract the next SSE event from the Anthropic buffer.
///
/// Anthropic SSE frames look like:
/// ```text
/// event: content_block_delta
/// data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
///
/// ```
///
/// We need both the `event:` and `data:` lines to form a complete frame.
#[allow(clippy::too_many_lines)]
fn parse_anthropic_event(
    buffer: &mut String,
    tool_blocks: &mut Vec<ToolBlockState>,
) -> Option<Result<StreamChunk, LlmError>> {
    loop {
        // Find the data line - Anthropic SSE always has event: then data:
        // We just need the data: line to parse the typed JSON.
        let data_prefix = "data: ";

        // Scan for a complete data line.
        let mut search_start = 0;
        let data_line_start;
        let data_line_end;

        loop {
            let remaining = &buffer[search_start..];
            let newline_pos = remaining.find('\n')?;
            let line_end = search_start + newline_pos;
            let line = buffer[search_start..line_end].trim();

            if line.starts_with(data_prefix) {
                data_line_start = search_start;
                data_line_end = line_end;
                break;
            }

            // Skip non-data lines (event:, empty, comments).
            search_start = line_end + 1;
        }

        // Extract the data payload into an owned String before mutating the buffer.
        let data = {
            let line = buffer[data_line_start..data_line_end].trim();
            let raw = line.strip_prefix(data_prefix).unwrap_or("").trim();
            raw.to_owned()
        };

        // Consume everything up to and including the data line's newline.
        buffer.drain(..=data_line_end);

        // Also drain any trailing blank line that separates SSE frames.
        if buffer.starts_with('\n') {
            buffer.drain(..1);
        } else if buffer.starts_with("\r\n") {
            buffer.drain(..2);
        }

        if data.is_empty() {
            continue;
        }

        let event: StreamEvent = match serde_json::from_str(&data) {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, %data, "failed to parse Anthropic SSE event");
                return Some(Err(LlmError::Stream(format!(
                    "failed to parse SSE event: {e}"
                ))));
            }
        };

        match event {
            StreamEvent::ContentBlockDelta { index: _, delta } => match delta {
                ContentBlockDeltaPayload::TextDelta { text } => {
                    return Some(Ok(StreamChunk {
                        delta: Some(text),
                        tool_calls: Vec::new(),
                        finish_reason: None,
                    }));
                }
                ContentBlockDeltaPayload::InputJsonDelta { partial_json } => {
                    // Accumulate partial JSON for the current tool block.
                    if let Some(block) = tool_blocks.last_mut() {
                        block.json_buf.push_str(&partial_json);
                    }
                    // Don't emit a chunk yet; wait for content_block_stop.
                }
            },
            StreamEvent::ContentBlockStart {
                index: _,
                content_block,
            } => {
                match content_block {
                    ContentBlockStartPayload::Text { .. } => {
                        // Text block start — nothing to emit yet.
                    }
                    ContentBlockStartPayload::ToolUse { id, name } => {
                        tool_blocks.push(ToolBlockState {
                            id,
                            name,
                            json_buf: String::new(),
                        });
                    }
                }
            }
            StreamEvent::ContentBlockStop { index: _ } => {
                // If we have a completed tool block, emit it.
                if let Some(block) = tool_blocks.pop() {
                    let arguments =
                        serde_json::from_str(&block.json_buf).unwrap_or(serde_json::Value::Null);
                    return Some(Ok(StreamChunk {
                        delta: None,
                        tool_calls: vec![ToolCall {
                            id: block.id,
                            name: block.name,
                            arguments,
                        }],
                        finish_reason: None,
                    }));
                }
            }
            StreamEvent::MessageDelta { delta, .. } => {
                if delta.stop_reason.is_some() {
                    return Some(Ok(StreamChunk {
                        delta: None,
                        tool_calls: Vec::new(),
                        finish_reason: delta.stop_reason,
                    }));
                }
            }
            StreamEvent::MessageStop => {
                return Some(Ok(StreamChunk {
                    delta: None,
                    tool_calls: Vec::new(),
                    finish_reason: Some("end_turn".to_owned()),
                }));
            }
            StreamEvent::Error { error } => {
                return Some(Err(LlmError::Stream(error.message)));
            }
            // Ping, MessageStart — skip.
            _ => {}
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
    fn build_body_extracts_system() {
        let provider = AnthropicProvider::new("test-key");
        let request = CompletionRequest {
            messages: vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hello"),
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            response_format: None,
            model: None,
        };

        let body = provider.build_body(&request, false);
        assert_eq!(body["system"], "You are helpful.");
        // Only the user message should be in the messages array.
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn build_body_requires_max_tokens() {
        let provider = AnthropicProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn build_body_with_tools() {
        let provider = AnthropicProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]).with_tools(vec![
            ToolDefinition {
                name: "search".to_owned(),
                description: "Search the web".to_owned(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    }
                }),
            },
        ]);

        let body = provider.build_body(&request, false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        // Anthropic uses "input_schema" not "parameters".
        assert!(tools[0].get("input_schema").is_some());
        assert_eq!(tools[0]["name"], "search");
    }

    #[test]
    fn parse_text_delta_event() {
        let mut buf = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n",
            "\n",
        ).to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks)
            .unwrap()
            .unwrap();
        assert_eq!(result.delta.as_deref(), Some("Hello"));
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn parse_message_stop_event() {
        let mut buf = concat!(
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        )
        .to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks)
            .unwrap()
            .unwrap();
        assert!(result.delta.is_none());
        assert_eq!(result.finish_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn parse_message_delta_stop_reason() {
        let mut buf = concat!(
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":15}}\n",
            "\n",
        ).to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks)
            .unwrap()
            .unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn parse_ping_is_skipped() {
        let mut buf = concat!(
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"World\"}}\n",
            "\n",
        ).to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks)
            .unwrap()
            .unwrap();
        // Should skip ping and return the text delta.
        assert_eq!(result.delta.as_deref(), Some("World"));
    }

    #[test]
    fn parse_tool_use_streaming() {
        let mut buf = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_123\",\"name\":\"get_weather\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\": \\\"SF\\\"}\"}}\n",
            "\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "\n",
        ).to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks)
            .unwrap()
            .unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "toolu_123");
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].arguments["city"], "SF");
    }

    #[test]
    fn parse_error_event() {
        let mut buf = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n",
            "\n",
        ).to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks).unwrap();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Overloaded"));
    }

    #[test]
    fn incomplete_event_returns_none() {
        let mut buf = "event: content_block_delta\n".to_owned();
        let mut tool_blocks = Vec::new();

        let result = parse_anthropic_event(&mut buf, &mut tool_blocks);
        assert!(result.is_none());
    }
}
