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
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use tracing::{debug, warn};

use super::openai_format::parse_retry_after;
use crate::error::BlazenError;
use crate::http::{ByteStream, HttpClient, HttpRequest, HttpResponse};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource, MessageContent,
    Role, StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The required Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default max tokens when the caller does not specify one.
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ---------------------------------------------------------------------------
// Multimodal helpers
// ---------------------------------------------------------------------------

/// Convert an [`ImageContent`] to the Anthropic `image` content block format.
fn image_content_to_anthropic(img: &ImageContent) -> serde_json::Value {
    match &img.source {
        ImageSource::Base64 { data } => {
            let media_type = img.media_type.as_deref().unwrap_or("image/png");
            serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                }
            })
        }
        ImageSource::Url { url } => {
            serde_json::json!({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": url,
                }
            })
        }
    }
}

/// Convert a single [`ContentPart`] to an Anthropic content block.
fn content_part_to_anthropic(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => {
            serde_json::json!({ "type": "text", "text": text })
        }
        ContentPart::Image(img) => image_content_to_anthropic(img),
        ContentPart::File(file) => {
            // Anthropic supports document types via the `document` block type.
            match &file.source {
                ImageSource::Base64 { data } => {
                    serde_json::json!({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": file.media_type,
                            "data": data,
                        }
                    })
                }
                ImageSource::Url { url } => {
                    serde_json::json!({
                        "type": "document",
                        "source": {
                            "type": "url",
                            "url": url,
                        }
                    })
                }
            }
        }
    }
}

/// Convert [`MessageContent`] to a `serde_json::Value` suitable for the
/// Anthropic `content` field.
///
/// - `Text` -> a plain JSON string (backward-compatible).
/// - `Image` / `Parts` -> a JSON array of content blocks.
fn content_to_anthropic_value(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text(t) => serde_json::Value::String(t.clone()),
        MessageContent::Image(img) => {
            serde_json::json!([image_content_to_anthropic(img)])
        }
        MessageContent::Parts(parts) => {
            let arr: Vec<serde_json::Value> = parts.iter().map(content_part_to_anthropic).collect();
            serde_json::json!(arr)
        }
    }
}

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
pub struct AnthropicProvider {
    client: Arc<dyn HttpClient>,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

impl Clone for AnthropicProvider {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            default_model: self.default_model.clone(),
        }
    }
}

impl AnthropicProvider {
    /// Create a new provider with the given API key, targeting the official
    /// Anthropic endpoint.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::ReqwestHttpClient::new().into_arc(),
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

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
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
                if let Some(text) = msg.content.text_content() {
                    system_parts.push(text);
                }
            } else if msg.role == Role::Tool {
                // Anthropic expects tool results as a user message with
                // tool_result content blocks.
                if let Some(ref call_id) = msg.tool_call_id {
                    let text = msg.content.text_content().unwrap_or_default();
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": text,
                        }],
                    }));
                } else {
                    // Legacy path: no tool_call_id, send as plain user message.
                    let content = content_to_anthropic_value(&msg.content);
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
            } else if msg.role == Role::Assistant {
                // If the assistant message carries tool_calls, serialize them
                // as tool_use content blocks (Anthropic format).
                if msg.tool_calls.is_empty() {
                    let content = content_to_anthropic_value(&msg.content);
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                } else {
                    let mut content_blocks: Vec<serde_json::Value> = Vec::new();
                    // Include text content if present.
                    if let Some(text) = msg.content.text_content() {
                        if !text.is_empty() {
                            content_blocks.push(serde_json::json!({"type": "text", "text": text}));
                        }
                    }
                    for tc in &msg.tool_calls {
                        content_blocks.push(serde_json::json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }));
                    }
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content_blocks,
                    }));
                }
            } else {
                let role = match msg.role {
                    Role::User => "user",
                    _ => continue,
                };
                let content = content_to_anthropic_value(&msg.content);
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

    /// Build an [`HttpRequest`] for the Anthropic Messages endpoint.
    fn build_http_request(&self, body: &serde_json::Value) -> Result<HttpRequest, BlazenError> {
        let url = format!("{}/messages", self.base_url);
        HttpRequest::post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json_body(body)
    }

    /// Send a request to the Messages endpoint, handling common HTTP errors.
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

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model_id = request.model.as_deref().unwrap_or(&self.default_model);
        let span = tracing::info_span!(
            "llm.complete",
            provider = "anthropic",
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
        debug!(model = %body["model"], "Anthropic completion request");

        let response = self.send_request(&body).await?;
        let anthropic: AnthropicResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

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

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        if let Some(ref u) = usage {
            span.record("prompt_tokens", u.prompt_tokens);
            span.record("completion_tokens", u.completion_tokens);
            span.record("total_tokens", u.total_tokens);
        }
        if let Some(ref reason) = anthropic.stop_reason {
            span.record("finish_reason", reason.as_str());
        }

        Ok(CompletionResponse {
            content,
            tool_calls,
            usage,
            model: anthropic.model,
            finish_reason: anthropic.stop_reason,
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
        let model_id = request.model.as_deref().unwrap_or(&self.default_model);
        let span = tracing::info_span!(
            "llm.stream",
            provider = "anthropic",
            model = %model_id,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let body = self.build_body(&request, true);
        debug!(model = %body["model"], "Anthropic streaming request");

        let http_request = self.build_http_request(&body)?;
        let (status, headers, byte_stream) = self.client.send_streaming(http_request).await?;

        if !(200..300).contains(&status) {
            let retry_after_ms = parse_retry_after(&headers);
            let error_body = String::from("streaming error");
            match status {
                401 => return Err(BlazenError::auth("authentication failed")),
                404 => return Err(BlazenError::model_not_found(error_body)),
                429 => return Err(BlazenError::RateLimit { retry_after_ms }),
                _ => return Err(BlazenError::request(format!("HTTP {status}: {error_body}"))),
            }
        }

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );

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
struct AnthropicSseParser {
    inner: ByteStream,
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

impl AnthropicSseParser {
    fn new(inner: ByteStream) -> Self {
        Self {
            inner,
            buffer: String::new(),
            tool_blocks: Vec::new(),
        }
    }
}

impl Stream for AnthropicSseParser {
    type Item = Result<StreamChunk, BlazenError>;

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
                    return std::task::Poll::Ready(Some(Err(BlazenError::stream_error(
                        e.to_string(),
                    ))));
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
) -> Option<Result<StreamChunk, BlazenError>> {
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
                return Some(Err(BlazenError::stream_error(format!(
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
                        // Text block start -- nothing to emit yet.
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
                return Some(Err(BlazenError::stream_error(error.message)));
            }
            // Ping, MessageStart -- skip.
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
            modalities: None,
            image_config: None,
            audio_config: None,
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
    fn test_text_backward_compat() {
        let provider = AnthropicProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, false);
        // Text messages should produce a plain string, not an array.
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_body_image_url() {
        let provider = AnthropicProvider::new("test-key");
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
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/cat.jpg");
    }

    #[test]
    fn test_build_body_base64_image() {
        let provider = AnthropicProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_base64(
            "Describe this",
            "abc123base64data",
            "image/png",
        )]);

        let body = provider.build_body(&request, false);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "abc123base64data");
    }

    #[test]
    fn test_build_body_multipart() {
        use crate::types::{ContentPart, ImageContent, ImageSource};

        let provider = AnthropicProvider::new("test-key");
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
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[2]["text"], "Second");
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
