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
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use tracing::{debug, warn};

use super::openai_format::parse_retry_after;
use super::{provider_http_error, provider_http_error_parts};
use crate::error::BlazenError;
use crate::http::{ByteStream, HttpClient, HttpRequest, HttpResponse};
use crate::retry::RetryConfig;
use crate::traits::{ModelCapabilities, ModelInfo, ModelRegistry};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource, MessageContent,
    ReasoningTrace, Role, StreamChunk, TokenUsage, ToolCall,
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
        ImageSource::File { .. } => {
            tracing::warn!(
                "anthropic: local file source is not supported — use a URL or base64 source \
                 instead; image content dropped."
            );
            serde_json::Value::Null
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(provider, crate::types::ProviderId::Anthropic) {
                // Anthropic natively accepts file IDs from its Files API.
                serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "file",
                        "file_id": id,
                    }
                })
            } else {
                crate::content::render::warn_provider_file_mismatch(
                    crate::types::ProviderId::Anthropic,
                    *provider,
                    id,
                    crate::content::render::MediaKindLabel::Image,
                );
                serde_json::Value::Null
            }
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Anthropic,
                handle,
                crate::content::render::MediaKindLabel::Image,
            );
            serde_json::Value::Null
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
                ImageSource::File { .. } => {
                    tracing::warn!(
                        "anthropic: local file source is not supported — use a URL or base64 \
                         source instead; file content dropped."
                    );
                    serde_json::Value::Null
                }
                ImageSource::ProviderFile { provider, id } => {
                    if matches!(provider, crate::types::ProviderId::Anthropic) {
                        serde_json::json!({
                            "type": "document",
                            "source": {
                                "type": "file",
                                "file_id": id,
                            }
                        })
                    } else {
                        crate::content::render::warn_provider_file_mismatch(
                            crate::types::ProviderId::Anthropic,
                            *provider,
                            id,
                            crate::content::render::MediaKindLabel::Document,
                        );
                        serde_json::Value::Null
                    }
                }
                ImageSource::Handle { handle } => {
                    crate::content::render::warn_handle_unresolved(
                        crate::types::ProviderId::Anthropic,
                        handle,
                        crate::content::render::MediaKindLabel::Document,
                    );
                    serde_json::Value::Null
                }
            }
        }
        ContentPart::Audio(_) => {
            tracing::warn!("anthropic: audio chat input is not supported; audio content dropped.");
            serde_json::Value::Null
        }
        ContentPart::Video(_) => {
            tracing::warn!("anthropic: video chat input is not supported; video content dropped.");
            serde_json::Value::Null
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
            // Drop `Null` entries that the part converter emits for content
            // Anthropic does not natively support (audio, video).
            let arr: Vec<serde_json::Value> = parts
                .iter()
                .map(content_part_to_anthropic)
                .filter(|v| !v.is_null())
                .collect();
            serde_json::json!(arr)
        }
    }
}

/// Combine extracted system messages with an optional response-format
/// instruction into a single `system` field value.
fn resolve_system_text(
    system_parts: &[String],
    response_format: Option<&serde_json::Value>,
) -> Option<String> {
    let base = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    match (base, response_format) {
        (Some(sys), Some(rf)) => Some(format!(
            "{}\n\n{sys}",
            build_json_schema_system_instruction(rf)
        )),
        (None, Some(rf)) => Some(build_json_schema_system_instruction(rf)),
        (sys, None) => sys,
    }
}

/// Build a synthetic system instruction asking the model to emit JSON matching
/// a given schema. Supports both a raw JSON Schema value and an OpenAI-style
/// `{"type":"json_schema","json_schema":{"schema":...}}` envelope.
fn build_json_schema_system_instruction(response_format: &serde_json::Value) -> String {
    let schema = if response_format
        .get("type")
        .and_then(serde_json::Value::as_str)
        == Some("json_schema")
    {
        response_format
            .get("json_schema")
            .and_then(|js| js.get("schema"))
            .unwrap_or(response_format)
    } else {
        response_format
    };
    let pretty = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
    format!(
        "You must respond with a JSON object that matches this exact JSON Schema:\n\n{pretty}\n\nRespond with ONLY the JSON object — no other text, no markdown, no code fences."
    )
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
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
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
            retry_config: self.retry_config.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            default_model: self.default_model.clone(),
        }
    }
}

impl AnthropicProvider {
    /// Create a new provider with the given API key, targeting the official
    /// Anthropic endpoint.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            retry_config: None,
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_owned(),
            default_model: "claude-sonnet-4-5-20250929".to_owned(),
        }
    }

    /// Create a new provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            retry_config: None,
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_owned(),
            default_model: "claude-sonnet-4-5-20250929".to_owned(),
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
                    let content = tool_result_to_anthropic_content(msg);
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": content,
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
                    if let Some(text) = msg.content.text_content()
                        && !text.is_empty()
                    {
                        content_blocks.push(serde_json::json!({"type": "text", "text": text}));
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

        if let Some(text) = resolve_system_text(&system_parts, request.response_format.as_ref()) {
            body["system"] = serde_json::Value::String(text);
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

        match response.status {
            401 => Err(BlazenError::auth("authentication failed")),
            404 => Err(BlazenError::model_not_found(response.text())),
            429 => Err(BlazenError::RateLimit {
                retry_after_ms: parse_retry_after(&response.headers),
            }),
            _ => {
                let url = format!("{}/messages", self.base_url);
                Err(provider_http_error("anthropic", &url, &response))
            }
        }
    }
}

super::impl_simple_from_options!(AnthropicProvider, "anthropic");

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
    /// Extended-thinking block emitted by Anthropic models with reasoning enabled.
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    /// Redacted reasoning block: the model produced reasoning but the provider
    /// withheld the plain text and returned an opaque `data` handle instead.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[allow(dead_code)]
        data: String,
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
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
// Every variant maps to an Anthropic SSE `*_delta` payload type, so the
// shared "Delta" suffix is intentional and matches the wire vocabulary.
#[allow(clippy::enum_variant_names)]
enum ContentBlockDeltaPayload {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    /// Streaming chunk of an extended-thinking block.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    /// Streaming signature attached to a thinking block. We discard this in
    /// streaming mode (the non-streaming `complete()` path is the only place
    /// signatures are exposed today), but we must accept the variant so the
    /// SSE parser does not error on it.
    #[serde(rename = "signature_delta")]
    #[allow(dead_code)]
    SignatureDelta {
        #[allow(dead_code)]
        signature: String,
    },
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
// Content block parsing helpers
// ---------------------------------------------------------------------------

/// Output of [`parse_content_blocks`]: the assistant text, any tool calls,
/// and any extended-thinking trace recovered from the response.
struct ParsedContent {
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
    reasoning: Option<ReasoningTrace>,
}

/// Walk a list of Anthropic content blocks and split them into the
/// provider-agnostic pieces consumed by [`CompletionResponse`].
///
/// Anthropic returns the assistant turn as an array of typed blocks. This
/// helper concatenates `text` blocks, lifts `tool_use` blocks into
/// [`ToolCall`]s, and accumulates `thinking` / `redacted_thinking` blocks
/// into a single [`ReasoningTrace`]. Multiple thinking blocks in one
/// response are concatenated; the last non-`None` signature wins.
fn parse_content_blocks(blocks: Vec<ContentBlock>) -> ParsedContent {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut thinking_text = String::new();
    let mut last_signature: Option<String> = None;
    let mut had_redacted = false;

    for block in blocks {
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
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                thinking_text.push_str(&thinking);
                if signature.is_some() {
                    last_signature = signature;
                }
            }
            ContentBlock::RedactedThinking { .. } => {
                had_redacted = true;
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let reasoning = if thinking_text.is_empty() && !had_redacted {
        None
    } else {
        Some(ReasoningTrace {
            text: thinking_text,
            signature: last_signature,
            redacted: had_redacted,
            effort: None,
        })
    };

    ParsedContent {
        content,
        tool_calls,
        reasoning,
    }
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for AnthropicProvider {
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

        let ParsedContent {
            content,
            tool_calls,
            reasoning,
        } = parse_content_blocks(anthropic.content);

        let usage = anthropic.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
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
        if let Some(ref reason) = anthropic.stop_reason {
            span.record("finish_reason", reason.as_str());
        }

        let cost = usage
            .as_ref()
            .and_then(|u| crate::pricing::compute_cost(&anthropic.model, u));

        Ok(CompletionResponse {
            content,
            tool_calls,
            reasoning,
            citations: vec![],
            artifacts: vec![],
            usage,
            model: anthropic.model,
            finish_reason: anthropic.stop_reason,
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
            match status {
                401 => return Err(BlazenError::auth("authentication failed")),
                404 => return Err(BlazenError::model_not_found("streaming error")),
                429 => {
                    return Err(BlazenError::RateLimit {
                        retry_after_ms: parse_retry_after(&headers),
                    });
                }
                _ => {
                    let url = format!("{}/messages", self.base_url);
                    return Err(provider_http_error_parts(
                        "anthropic",
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

        let stream = AnthropicSseParser::new(byte_stream);
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry implementation
// ---------------------------------------------------------------------------

/// Wire format for the Anthropic `/models` endpoint.
#[derive(Debug, Deserialize)]
struct AnthropicModelsResponse {
    data: Vec<AnthropicModelEntry>,
}

#[derive(Debug, Deserialize)]
struct AnthropicModelEntry {
    id: String,
    display_name: Option<String>,
}

#[async_trait]
impl ModelRegistry for AnthropicProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError> {
        let url = format!("{}/models", self.base_url);
        let request = HttpRequest::get(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            return Err(provider_http_error("anthropic", &url, &response));
        }

        let list: AnthropicModelsResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let models = list
            .data
            .into_iter()
            .map(|entry| ModelInfo {
                id: entry.id,
                name: entry.display_name,
                provider: "anthropic".into(),
                context_length: None,
                pricing: None, // Anthropic /models does not include pricing.
                capabilities: ModelCapabilities {
                    chat: true,
                    streaming: true,
                    tool_use: true,
                    structured_output: true,
                    vision: true,
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
                        ..Default::default()
                    }));
                }
                ContentBlockDeltaPayload::InputJsonDelta { partial_json } => {
                    // Accumulate partial JSON for the current tool block.
                    if let Some(block) = tool_blocks.last_mut() {
                        block.json_buf.push_str(&partial_json);
                    }
                    // Don't emit a chunk yet; wait for content_block_stop.
                }
                ContentBlockDeltaPayload::ThinkingDelta { thinking } => {
                    // Surface extended-thinking text as a reasoning delta.
                    return Some(Ok(StreamChunk {
                        reasoning_delta: Some(thinking),
                        ..Default::default()
                    }));
                }
                ContentBlockDeltaPayload::SignatureDelta { .. } => {
                    // Signature deltas are not surfaced through StreamChunk;
                    // they only matter for the non-streaming `complete()`
                    // path, which captures them off the final content block.
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
                    ContentBlockStartPayload::Thinking { thinking } => {
                        // Thinking blocks usually start empty and stream their
                        // text via `thinking_delta` events. If the start frame
                        // already carries text, emit it immediately so callers
                        // never lose it.
                        if !thinking.is_empty() {
                            return Some(Ok(StreamChunk {
                                reasoning_delta: Some(thinking),
                                ..Default::default()
                            }));
                        }
                    }
                    ContentBlockStartPayload::RedactedThinking { .. } => {
                        // The provider withheld the thinking text. Emit a
                        // single sentinel chunk so downstream consumers know
                        // a redacted reasoning block was present without
                        // having to special-case the absence of deltas.
                        return Some(Ok(StreamChunk {
                            reasoning_delta: Some("[redacted]".to_owned()),
                            ..Default::default()
                        }));
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
                        ..Default::default()
                    }));
                }
            }
            StreamEvent::MessageDelta { delta, .. } if delta.stop_reason.is_some() => {
                return Some(Ok(StreamChunk {
                    delta: None,
                    tool_calls: Vec::new(),
                    finish_reason: delta.stop_reason,
                    ..Default::default()
                }));
            }
            StreamEvent::MessageStop => {
                return Some(Ok(StreamChunk {
                    delta: None,
                    tool_calls: Vec::new(),
                    finish_reason: Some("end_turn".to_owned()),
                    ..Default::default()
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
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl crate::traits::ProviderInfo for AnthropicProvider {
    fn provider_name(&self) -> &'static str {
        "anthropic"
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
            embeddings: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool-result rendering
// ---------------------------------------------------------------------------

/// Render a tool-result payload for Anthropic's `tool_result.content` field.
///
/// Anthropic accepts either a plain string OR an array of content blocks.
/// This function honours `LlmPayload::Text/Json/Parts/ProviderRaw` overrides
/// when present, else falls back to a default conversion from `data`:
/// strings pass through, structured values become a single `text` block
/// containing the JSON-stringified value.
fn tool_result_to_anthropic_content(msg: &crate::types::ChatMessage) -> serde_json::Value {
    let Some((data, override_payload)) = msg.tool_result_view() else {
        return serde_json::json!("");
    };

    if let Some(payload) = override_payload {
        return match payload {
            crate::types::LlmPayload::Text { text } => serde_json::json!(text),
            crate::types::LlmPayload::Json { value } => match value {
                serde_json::Value::String(s) => serde_json::json!(s),
                other => serde_json::json!([{
                    "type": "text",
                    "text": serde_json::to_string(other).unwrap_or_default(),
                }]),
            },
            crate::types::LlmPayload::Parts { parts } => {
                // Anthropic accepts native multimodal content blocks here.
                serde_json::to_value(parts).unwrap_or_else(|_| serde_json::json!(""))
            }
            crate::types::LlmPayload::ProviderRaw { provider, value }
                if *provider == crate::types::ProviderId::Anthropic =>
            {
                value.clone()
            }
            crate::types::LlmPayload::ProviderRaw { .. } => default_anthropic_content(&data),
        };
    }

    default_anthropic_content(&data)
}

fn default_anthropic_content(data: &serde_json::Value) -> serde_json::Value {
    match data {
        serde_json::Value::String(s) => serde_json::json!(s),
        other => serde_json::json!([{
            "type": "text",
            "text": serde_json::to_string(other).unwrap_or_default(),
        }]),
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
    fn test_response_format_injects_schema_into_system_prompt() {
        let provider = AnthropicProvider::new("test-key");
        let request = CompletionRequest {
            model: None,
            messages: vec![ChatMessage::user("hi")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            response_format: Some(serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            })),
            modalities: None,
            image_config: None,
            audio_config: None,
        };
        let body = provider.build_body(&request, false);
        let system = body["system"].as_str().expect("system field should be set");
        assert!(system.contains("You must respond with a JSON object"));
        assert!(system.contains("answer"));
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

    #[test]
    fn test_parse_thinking_block() {
        // Fixture mirrors a real Anthropic `messages.create` response with
        // extended thinking enabled: a `thinking` block followed by a `text`
        // block. We deserialize via the same `AnthropicResponse` wire type
        // the live HTTP path uses, then run the shared `parse_content_blocks`
        // helper that `complete()` invokes.
        let body = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-5-20250101",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "step 1: consider...",
                    "signature": "sig123"
                },
                {
                    "type": "text",
                    "text": "the answer is 42"
                }
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 20 }
        });

        let anthropic: AnthropicResponse = serde_json::from_value(body).unwrap();

        // Build a CompletionResponse the same way `complete()` does, minus
        // the HTTP / span / pricing plumbing.
        let ParsedContent {
            content,
            tool_calls,
            reasoning,
        } = parse_content_blocks(anthropic.content);

        let response = CompletionResponse {
            content,
            tool_calls,
            reasoning,
            citations: vec![],
            artifacts: vec![],
            usage: None,
            model: anthropic.model,
            finish_reason: anthropic.stop_reason,
            cost: None,
            timing: None,
            images: vec![],
            audio: vec![],
            videos: vec![],
            metadata: serde_json::Value::Null,
        };

        assert_eq!(response.content.as_deref(), Some("the answer is 42"));
        assert!(response.reasoning.is_some());
        let trace = response.reasoning.as_ref().unwrap();
        assert_eq!(trace.text, "step 1: consider...");
        assert_eq!(trace.signature.as_deref(), Some("sig123"));
        assert!(!trace.redacted);
        assert!(trace.effort.is_none());
    }

    // -----------------------------------------------------------------------
    // tool_result_to_anthropic_content helper
    // -----------------------------------------------------------------------

    #[test]
    fn anthropic_helper_string_data_passes_through() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!("hello"));
        assert_eq!(
            super::tool_result_to_anthropic_content(&msg),
            serde_json::json!("hello")
        );
    }

    #[test]
    fn anthropic_helper_structured_data_becomes_text_block() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!({"k":"v"}));
        assert_eq!(
            super::tool_result_to_anthropic_content(&msg),
            serde_json::json!([{"type":"text","text":"{\"k\":\"v\"}"}])
        );
    }

    #[test]
    fn anthropic_helper_text_override_wins() {
        use crate::types::{LlmPayload, ToolOutput};
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({"k":"v"}),
                LlmPayload::Text {
                    text: "summary".into(),
                },
            ),
        );
        assert_eq!(
            super::tool_result_to_anthropic_content(&msg),
            serde_json::json!("summary")
        );
    }

    #[test]
    fn anthropic_helper_provider_raw_for_anthropic_passes_through() {
        use crate::types::{LlmPayload, ProviderId, ToolOutput};
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({"k":"v"}),
                LlmPayload::ProviderRaw {
                    provider: ProviderId::Anthropic,
                    value: serde_json::json!([{"type":"text","text":"raw block"}]),
                },
            ),
        );
        assert_eq!(
            super::tool_result_to_anthropic_content(&msg),
            serde_json::json!([{"type":"text","text":"raw block"}])
        );
    }

    #[test]
    fn anthropic_helper_parts_with_image_passes_native_blocks() {
        // Anthropic's `tool_result.content` natively accepts a content-block
        // array including text and image blocks. `LlmPayload::Parts` should
        // pass through as such an array, preserving multimodal payloads.
        use crate::types::{ContentPart, LlmPayload, ToolOutput};
        let msg = ChatMessage::tool_result(
            "call_1",
            "render",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![
                        ContentPart::text("rendered"),
                        ContentPart::image_base64("AAAA", "image/png"),
                    ],
                },
            ),
        );
        let value = super::tool_result_to_anthropic_content(&msg);
        let arr = value.as_array().expect("expected content-block array");
        assert_eq!(arr.len(), 2);
        // First: text part.
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "rendered");
        // Second: image part. ContentPart::Image serializes as
        // {"type":"image","source":{"type":"base64","data":"AAAA"},...}
        assert_eq!(arr[1]["type"], "image");
        assert_eq!(arr[1]["source"]["type"], "base64");
        assert_eq!(arr[1]["source"]["data"], "AAAA");
    }
}
