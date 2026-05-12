//! LLM completion + embedding surface for the UniFFI bindings.
//!
//! Mirrors `blazen-py`'s `CompletionModel` / `EmbeddingModel` shape, but using
//! UniFFI's value-record + opaque-object idiom instead of PyO3 classes.
//!
//! ## Wire-format shape
//!
//! Upstream `blazen_llm` carries rich enum-shaped types
//! (`MessageContent::{Text, Image, Parts}`, `ContentPart::{Text, Image, Audio,
//! Video, File}`, `ImageSource::{Url, Base64, File, ProviderFile, Handle}`)
//! that don't translate cleanly to the lowest-common-denominator FFI of UDL.
//! This module flattens that down to:
//!
//! - [`ChatMessage`] — `role: String`, `content: String` (text), plus an
//!   optional [`Media`] list for multimodal inputs (URL or base64 only).
//! - [`Tool`] / [`ToolCall`] — JSON-Schema parameters and arguments as
//!   strings on the wire (foreign callers marshal to/from their native JSON
//!   type just outside this module).
//! - [`CompletionResponse`] — `content` is always a `String` (empty when the
//!   provider returned no text); `finish_reason` is also a plain `String`
//!   (`""` when the provider didn't report one).
//! - [`TokenUsage`] — every counter is `u64` to keep the binding ergonomic
//!   (Swift / Kotlin / Ruby don't have a u32 type at all). Upstream `u32`
//!   values widen losslessly.
//! - [`EmbeddingResponse::embeddings`] — `Vec<Vec<f64>>`. UniFFI doesn't
//!   expose `f32` cleanly; upstream `f32` vectors widen losslessly.
//!
//! ## Provider construction
//!
//! [`CompletionModel`] and [`EmbeddingModel`] are *opaque* — there are no
//! foreign-side constructors here. Per-provider factories live in
//! `providers.rs`. This module only handles the dispatch surface (i.e.
//! "given an Arc<dyn CompletionModel>, here is how `complete` / `embed`
//! cross the FFI").

use std::sync::Arc;

use blazen_llm::types::{
    AudioContent as CoreAudioContent, ChatMessage as CoreChatMessage,
    ContentPart as CoreContentPart, EmbeddingResponse as CoreEmbeddingResponse,
    ImageContent as CoreImageContent, ImageSource as CoreImageSource,
    MessageContent as CoreMessageContent, Role as CoreRole, TokenUsage as CoreTokenUsage,
    ToolCall as CoreToolCall, ToolDefinition as CoreToolDefinition,
    VideoContent as CoreVideoContent,
};
use blazen_llm::{
    CompletionModel as CoreCompletionModel, CompletionRequest as CoreCompletionRequest,
    CompletionResponse as CoreCompletionResponse, EmbeddingModel as CoreEmbeddingModel,
};

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// Multimodal media attached to a [`ChatMessage`].
///
/// `kind` selects the part type and is one of `"image"`, `"audio"`, `"video"`.
/// `mime_type` is the IANA MIME (`"image/png"`, `"audio/mp3"`, ...).
/// `data_base64` carries the raw bytes base64-encoded; for URL-backed media,
/// set `data_base64` to the empty string and put the URL in `mime_type` is
/// **not** correct — instead, callers wanting URL inputs should base64-encode
/// the fetched bytes. (URL passthrough is intentionally elided here to keep
/// the FFI surface single-shape; providers that need URL fidelity can be
/// reached via `providers.rs`.)
#[derive(Debug, Clone, uniffi::Record)]
pub struct Media {
    pub kind: String,
    pub mime_type: String,
    pub data_base64: String,
}

/// A single message in a chat conversation.
///
/// `role` is one of `"system"`, `"user"`, `"assistant"`, `"tool"`.
/// `content` is the text payload (empty string when the message carries only
/// tool calls or media). Multimodal media attaches via [`media_parts`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub media_parts: Vec<Media>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments object. Foreign callers parse this with their
    /// native JSON library to access the tool's input parameters.
    pub arguments_json: String,
}

/// A tool that the model may invoke during a completion.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's input parameters. Stored as a string
    /// on the wire; foreign callers serialize their native schema dict/struct
    /// to JSON just before constructing the [`Tool`].
    pub parameters_json: String,
}

/// Token usage statistics for a completion or embedding request.
///
/// Every counter is `u64` for FFI uniformity. Upstream `u32` values widen
/// losslessly. Zero means either "the provider didn't report this counter"
/// or "the counter is genuinely zero" — the wire format does not distinguish.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
}

/// A provider-agnostic chat completion request.
///
/// `system`, when set, is prepended as a `Role::System` message — equivalent
/// to building the message list with a leading system entry. Provided as a
/// convenience because most foreign callers think of the system prompt as a
/// request-level field, not a message.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<Tool>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    /// Optional model identifier to use for this request, overriding the
    /// provider's default model.
    pub model: Option<String>,
    /// Optional JSON Schema (encoded as a string) constraining the model's
    /// output. Passed through to upstream's `response_format` slot.
    pub response_format_json: Option<String>,
    /// Optional system prompt, prepended as a `system`-role message.
    pub system: Option<String>,
}

/// The result of a non-streaming chat completion.
///
/// `content` is the empty string when the provider returned no text (e.g.
/// the model emitted only tool calls). `finish_reason` is the empty string
/// when the provider didn't report one.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CompletionResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub model: String,
    pub usage: TokenUsage,
}

/// Response from an embedding model.
///
/// `embeddings[i]` is the vector for the `i`-th input string. Vectors are
/// `f64` for FFI uniformity (UniFFI doesn't expose `f32`); upstream `f32`
/// values widen losslessly.
#[derive(Debug, Clone, uniffi::Record)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f64>>,
    pub model: String,
    pub usage: TokenUsage,
}

// ---------------------------------------------------------------------------
// Conversions: wire-format <-> blazen_llm internal types
// ---------------------------------------------------------------------------

impl From<CoreTokenUsage> for TokenUsage {
    fn from(u: CoreTokenUsage) -> Self {
        Self {
            prompt_tokens: u64::from(u.prompt_tokens),
            completion_tokens: u64::from(u.completion_tokens),
            total_tokens: u64::from(u.total_tokens),
            cached_input_tokens: u64::from(u.cached_input_tokens),
            reasoning_tokens: u64::from(u.reasoning_tokens),
        }
    }
}

impl From<CoreToolCall> for ToolCall {
    fn from(call: CoreToolCall) -> Self {
        Self {
            id: call.id,
            name: call.name,
            arguments_json: call.arguments.to_string(),
        }
    }
}

impl TryFrom<ToolCall> for CoreToolCall {
    type Error = BlazenError;
    fn try_from(call: ToolCall) -> Result<Self, Self::Error> {
        let arguments = if call.arguments_json.is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&call.arguments_json)?
        };
        Ok(Self {
            id: call.id,
            name: call.name,
            arguments,
        })
    }
}

impl From<CoreToolDefinition> for Tool {
    fn from(def: CoreToolDefinition) -> Self {
        Self {
            name: def.name,
            description: def.description,
            parameters_json: def.parameters.to_string(),
        }
    }
}

impl TryFrom<Tool> for CoreToolDefinition {
    type Error = BlazenError;
    fn try_from(tool: Tool) -> Result<Self, Self::Error> {
        let parameters = if tool.parameters_json.is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&tool.parameters_json)?
        };
        Ok(Self {
            name: tool.name,
            description: tool.description,
            parameters,
        })
    }
}

/// Render a wire [`Media`] entry as an upstream [`CoreContentPart`].
///
/// Unknown `kind` strings collapse to an image part by default — this matches
/// the most permissive provider behaviour (image inputs are the widest-
/// supported modality) and avoids a hard error in callers that fat-finger
/// the kind string.
fn media_to_part(media: Media) -> CoreContentPart {
    let source = CoreImageSource::Base64 {
        data: media.data_base64,
    };
    let media_type = if media.mime_type.is_empty() {
        None
    } else {
        Some(media.mime_type.clone())
    };
    match media.kind.as_str() {
        "audio" => CoreContentPart::Audio(CoreAudioContent {
            source,
            media_type,
            duration_seconds: None,
        }),
        "video" => CoreContentPart::Video(CoreVideoContent {
            source,
            media_type,
            duration_seconds: None,
        }),
        _ => CoreContentPart::Image(CoreImageContent { source, media_type }),
    }
}

/// Render an upstream [`CoreContentPart`] back as a wire [`Media`] entry.
///
/// URL-backed sources are base64-of-the-URL-bytes is wrong, so we serialise
/// the source enum to JSON in `data_base64` when the source isn't already a
/// raw base64 payload. This preserves round-tripping for outbound responses
/// at the cost of foreign callers needing to inspect the prefix —
/// providers that emit multimodal output are rare today, so this is a
/// pragmatic loss.
fn part_to_media(part: &CoreContentPart) -> Option<Media> {
    let (kind, media_type, source) = match part {
        CoreContentPart::Image(img) => ("image", img.media_type.clone(), &img.source),
        CoreContentPart::Audio(a) => ("audio", a.media_type.clone(), &a.source),
        CoreContentPart::Video(v) => ("video", v.media_type.clone(), &v.source),
        CoreContentPart::Text { .. } | CoreContentPart::File(_) => return None,
    };
    let data_base64 = match source {
        CoreImageSource::Base64 { data } => data.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };
    Some(Media {
        kind: kind.to_owned(),
        mime_type: media_type.unwrap_or_default(),
        data_base64,
    })
}

fn role_from_str(role: &str) -> CoreRole {
    match role {
        "system" => CoreRole::System,
        "assistant" => CoreRole::Assistant,
        "tool" => CoreRole::Tool,
        _ => CoreRole::User,
    }
}

fn role_to_str(role: &CoreRole) -> &'static str {
    match role {
        CoreRole::System => "system",
        CoreRole::User => "user",
        CoreRole::Assistant => "assistant",
        CoreRole::Tool => "tool",
    }
}

impl TryFrom<ChatMessage> for CoreChatMessage {
    type Error = BlazenError;
    fn try_from(msg: ChatMessage) -> Result<Self, Self::Error> {
        let role = role_from_str(&msg.role);
        let content = if msg.media_parts.is_empty() {
            CoreMessageContent::Text(msg.content)
        } else {
            let mut parts = Vec::with_capacity(1 + msg.media_parts.len());
            if !msg.content.is_empty() {
                parts.push(CoreContentPart::Text { text: msg.content });
            }
            for media in msg.media_parts {
                parts.push(media_to_part(media));
            }
            CoreMessageContent::Parts(parts)
        };
        let tool_calls = msg
            .tool_calls
            .into_iter()
            .map(CoreToolCall::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            role,
            content,
            tool_call_id: msg.tool_call_id,
            name: msg.name,
            tool_calls,
            tool_result: None,
        })
    }
}

impl From<CoreChatMessage> for ChatMessage {
    fn from(msg: CoreChatMessage) -> Self {
        let role = role_to_str(&msg.role).to_owned();
        let (content, media_parts) = match msg.content {
            CoreMessageContent::Text(s) => (s, Vec::new()),
            CoreMessageContent::Image(img) => (
                String::new(),
                part_to_media(&CoreContentPart::Image(img))
                    .map(|m| vec![m])
                    .unwrap_or_default(),
            ),
            CoreMessageContent::Parts(parts) => {
                let mut text = String::new();
                let mut media = Vec::new();
                for part in parts {
                    match &part {
                        CoreContentPart::Text { text: t } => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                        _ => {
                            if let Some(m) = part_to_media(&part) {
                                media.push(m);
                            }
                        }
                    }
                }
                (text, media)
            }
        };
        Self {
            role,
            content,
            media_parts,
            tool_calls: msg.tool_calls.into_iter().map(ToolCall::from).collect(),
            tool_call_id: msg.tool_call_id,
            name: msg.name,
        }
    }
}

impl TryFrom<CompletionRequest> for CoreCompletionRequest {
    type Error = BlazenError;
    fn try_from(req: CompletionRequest) -> Result<Self, Self::Error> {
        let mut messages: Vec<CoreChatMessage> = Vec::with_capacity(req.messages.len() + 1);
        if let Some(system) = req.system
            && !system.is_empty()
        {
            messages.push(CoreChatMessage::system(system));
        }
        for m in req.messages {
            messages.push(CoreChatMessage::try_from(m)?);
        }
        let tools = req
            .tools
            .into_iter()
            .map(CoreToolDefinition::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let response_format = match req.response_format_json {
            Some(s) if !s.is_empty() => Some(serde_json::from_str(&s)?),
            _ => None,
        };
        // f64 -> f32 narrowing is fine for sampling params (callers think in
        // 0.0-2.0 / 0.0-1.0 ranges); the cast preserves enough precision.
        Ok(Self {
            messages,
            tools,
            temperature: req.temperature.map(|t| t as f32),
            max_tokens: req.max_tokens,
            top_p: req.top_p.map(|p| p as f32),
            response_format,
            model: req.model,
            modalities: None,
            image_config: None,
            audio_config: None,
        })
    }
}

impl From<CoreCompletionResponse> for CompletionResponse {
    fn from(resp: CoreCompletionResponse) -> Self {
        Self {
            content: resp.content.unwrap_or_default(),
            tool_calls: resp.tool_calls.into_iter().map(ToolCall::from).collect(),
            finish_reason: resp.finish_reason.unwrap_or_default(),
            model: resp.model,
            usage: resp.usage.map(TokenUsage::from).unwrap_or_default(),
        }
    }
}

impl From<CoreEmbeddingResponse> for EmbeddingResponse {
    fn from(resp: CoreEmbeddingResponse) -> Self {
        let embeddings = resp
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect())
            .collect();
        Self {
            embeddings,
            model: resp.model,
            usage: resp.usage.map(TokenUsage::from).unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque model handles
// ---------------------------------------------------------------------------

/// A chat completion model.
///
/// Construct one via the per-provider factories in `providers.rs` (e.g.
/// `CompletionModel::openai(options)` from the foreign-language side).
/// Once obtained, call [`complete`](Self::complete) (async) or
/// [`complete_blocking`](Self::complete_blocking) (sync) to generate
/// responses.
#[derive(uniffi::Object)]
pub struct CompletionModel {
    pub(crate) inner: Arc<dyn CoreCompletionModel>,
}

impl CompletionModel {
    /// Wrap a `blazen_llm` completion model in the FFI handle.
    ///
    /// Used by `providers.rs` factories; not exposed across the FFI.
    pub(crate) fn from_arc(inner: Arc<dyn CoreCompletionModel>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl CompletionModel {
    /// Perform a chat completion. Async on Swift / Kotlin; blocking on Go
    /// (UniFFI's Go bindgen wraps the future in a goroutine-friendly call).
    pub async fn complete(
        self: Arc<Self>,
        request: CompletionRequest,
    ) -> BlazenResult<CompletionResponse> {
        let core_request = CoreCompletionRequest::try_from(request)?;
        let response = self
            .inner
            .complete(core_request)
            .await
            .map_err(BlazenError::from)?;
        Ok(CompletionResponse::from(response))
    }

    /// The model's identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
    #[must_use]
    pub fn model_id(self: Arc<Self>) -> String {
        self.inner.model_id().to_owned()
    }
}

#[uniffi::export]
impl CompletionModel {
    /// Synchronous variant of [`complete`](Self::complete) — blocks the
    /// current thread on the shared Tokio runtime. Handy for Ruby scripts
    /// and quick Go `main` functions where async machinery is overkill.
    /// Prefer the async [`complete`](Self::complete) in long-running services.
    pub fn complete_blocking(
        self: Arc<Self>,
        request: CompletionRequest,
    ) -> BlazenResult<CompletionResponse> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.complete(request).await })
    }
}

/// An embedding model that produces vector embeddings for text inputs.
///
/// Construct one via the per-provider factories in `providers.rs`.
#[derive(uniffi::Object)]
pub struct EmbeddingModel {
    pub(crate) inner: Arc<dyn CoreEmbeddingModel>,
}

impl EmbeddingModel {
    pub(crate) fn from_arc(inner: Arc<dyn CoreEmbeddingModel>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl EmbeddingModel {
    /// Embed one or more text strings, returning one vector per input.
    pub async fn embed(self: Arc<Self>, inputs: Vec<String>) -> BlazenResult<EmbeddingResponse> {
        let response = self.inner.embed(&inputs).await.map_err(BlazenError::from)?;
        Ok(EmbeddingResponse::from(response))
    }

    /// The model's identifier (e.g. `"text-embedding-3-small"`).
    #[must_use]
    pub fn model_id(self: Arc<Self>) -> String {
        self.inner.model_id().to_owned()
    }

    /// The dimensionality of vectors produced by this model.
    #[must_use]
    pub fn dimensions(self: Arc<Self>) -> u32 {
        // `usize` doesn't cross FFI cleanly; widen to u32 (no embedding model
        // produces vectors longer than 2^32 dimensions).
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

#[uniffi::export]
impl EmbeddingModel {
    /// Synchronous variant of [`embed`](Self::embed).
    pub fn embed_blocking(self: Arc<Self>, inputs: Vec<String>) -> BlazenResult<EmbeddingResponse> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.embed(inputs).await })
    }
}
