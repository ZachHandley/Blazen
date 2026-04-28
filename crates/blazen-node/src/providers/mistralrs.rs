//! JavaScript bindings for the local mistral.rs LLM provider.
//!
//! Exposes [`JsMistralRsProvider`] as a NAPI class with a factory
//! constructor, async completion / streaming methods, and `LocalModel`
//! lifecycle controls (`load`, `unload`, `isLoaded`, `vramBytes`).
//!
//! Runs LLM inference entirely on-device using the mistral.rs engine.
//! No API key is required.

#![cfg(feature = "mistralrs")]

use std::path::PathBuf;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};
use blazen_llm::{
    ChatMessageInput, ChatRole, InferenceChunk, InferenceChunkStream, InferenceImage,
    InferenceImageSource, InferenceResult, InferenceToolCall, InferenceUsage, MistralRsOptions,
    MistralRsProvider,
};

use crate::error::{llm_error_to_napi, mistralrs_error_to_napi, to_napi_error};
use crate::providers::completion_model::{JsMistralRsOptions, StreamChunkCallbackTsfn};
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// JsMistralRsProvider NAPI class
// ---------------------------------------------------------------------------

/// A local mistral.rs LLM provider with completion and streaming.
///
/// ```javascript
/// const provider = MistralRsProvider.create({
///   modelId: "mistralai/Mistral-7B-Instruct-v0.3",
/// });
/// await provider.load();
/// const response = await provider.complete([
///   ChatMessage.user("What is 2+2?"),
/// ]);
/// ```
#[napi(js_name = "MistralRsProvider")]
pub struct JsMistralRsProvider {
    inner: Arc<MistralRsProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
impl JsMistralRsProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new mistral.rs provider.
    #[napi(factory)]
    pub fn create(options: JsMistralRsOptions) -> Result<Self> {
        let opts: MistralRsOptions = options.into();
        Ok(Self {
            inner: Arc::new(
                MistralRsProvider::from_options(opts).map_err(mistralrs_error_to_napi)?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Completion
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        Ok(build_response(response))
    }

    /// Perform a chat completion with additional options.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);

        if let Some(temp) = options.temperature {
            request.temperature = Some(temp as f32);
        }
        if let Some(max) = options.max_tokens {
            request.max_tokens = Some(max as u32);
        }
        if let Some(top_p) = options.top_p {
            request.top_p = Some(top_p as f32);
        }
        if let Some(model) = options.model {
            request.model = Some(model);
        }
        if let Some(tools) = options.tools {
            request.tools = tools
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect();
        }
        if let Some(fmt) = options.response_format {
            request = request.with_response_format(fmt);
        }

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        Ok(build_response(response))
    }

    /// Stream a chat completion.
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let stream = self
            .inner
            .stream(request)
            .await
            .map_err(llm_error_to_napi)?;

        let mut stream = std::pin::pin!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    on_chunk.call(
                        build_stream_chunk(chunk),
                        ThreadsafeFunctionCallMode::Blocking,
                    );
                }
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // LocalModel lifecycle
    // -----------------------------------------------------------------

    /// Explicitly load the model weights into memory / `VRAM`.
    #[napi]
    pub async fn load(&self) -> Result<()> {
        self.inner.load().await.map_err(to_napi_error)
    }

    /// Drop the loaded model and free its memory / `VRAM`.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        self.inner.unload().await.map_err(to_napi_error)
    }

    /// Whether the model is currently loaded in memory / `VRAM`.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> bool {
        self.inner.is_loaded().await
    }

    /// Approximate `VRAM` footprint in bytes.
    #[napi(js_name = "vramBytes")]
    pub async fn vram_bytes(&self) -> Option<i64> {
        self.inner
            .vram_bytes()
            .await
            .map(|b| i64::try_from(b).unwrap_or(i64::MAX))
    }
}

// ---------------------------------------------------------------------------
// ChatRole -- string enum mirror of `blazen_llm::ChatRole`
// ---------------------------------------------------------------------------

/// Role of a chat message in the local mistral.rs / llama.cpp inference path.
#[napi(string_enum, js_name = "ChatRole")]
pub enum JsChatRole {
    #[napi(value = "system")]
    System,
    #[napi(value = "user")]
    User,
    #[napi(value = "assistant")]
    Assistant,
    #[napi(value = "tool")]
    Tool,
}

impl JsChatRole {
    /// Build the JS-facing role from the underlying Rust enum.
    #[must_use]
    pub fn from_rust(role: ChatRole) -> Self {
        match role {
            ChatRole::System => Self::System,
            ChatRole::User => Self::User,
            ChatRole::Assistant => Self::Assistant,
            ChatRole::Tool => Self::Tool,
        }
    }

    /// Convert this JS-facing role into the underlying Rust enum.
    #[must_use]
    pub fn into_rust(self) -> ChatRole {
        match self {
            Self::System => ChatRole::System,
            Self::User => ChatRole::User,
            Self::Assistant => ChatRole::Assistant,
            Self::Tool => ChatRole::Tool,
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceImageSource -- typed class mirror of `blazen_llm::InferenceImageSource`
// ---------------------------------------------------------------------------

/// Source of an image payload attached to a chat message.
///
/// Construct with [`InferenceImageSource.bytes`] or
/// [`InferenceImageSource.path`]. Inspect with the `kind` getter
/// (`"bytes"` or `"path"`) and the appropriate value getter.
#[napi(js_name = "InferenceImageSource")]
pub struct JsInferenceImageSource {
    inner: InferenceImageSource,
}

#[napi]
impl JsInferenceImageSource {
    /// Build an image source from raw encoded image bytes (PNG/JPEG/WebP).
    #[must_use]
    #[napi(factory)]
    pub fn bytes(data: Buffer) -> Self {
        Self {
            inner: InferenceImageSource::Bytes(data.into()),
        }
    }

    /// Build an image source from a local file path.
    #[must_use]
    #[napi(factory)]
    pub fn path(path: String) -> Self {
        Self {
            inner: InferenceImageSource::Path(PathBuf::from(path)),
        }
    }

    /// Discriminant: `"bytes"` or `"path"`.
    #[must_use]
    #[napi(getter)]
    pub fn kind(&self) -> &'static str {
        match &self.inner {
            InferenceImageSource::Bytes(_) => "bytes",
            InferenceImageSource::Path(_) => "path",
        }
    }

    /// The raw image bytes, if this source is a `bytes` variant.
    #[must_use]
    #[napi(getter)]
    pub fn data(&self) -> Option<Buffer> {
        match &self.inner {
            InferenceImageSource::Bytes(b) => Some(Buffer::from(b.clone())),
            InferenceImageSource::Path(_) => None,
        }
    }

    /// The file path, if this source is a `path` variant.
    #[must_use]
    #[napi(getter, js_name = "filePath")]
    pub fn file_path(&self) -> Option<String> {
        match &self.inner {
            InferenceImageSource::Path(p) => Some(p.to_string_lossy().into_owned()),
            InferenceImageSource::Bytes(_) => None,
        }
    }
}

impl JsInferenceImageSource {
    /// Wrap an existing Rust [`InferenceImageSource`].
    #[must_use]
    pub fn from_rust(inner: InferenceImageSource) -> Self {
        Self { inner }
    }

    /// Unwrap into the underlying Rust [`InferenceImageSource`], consuming this wrapper.
    #[must_use]
    pub fn into_rust(self) -> InferenceImageSource {
        self.inner
    }

    /// Clone out the underlying Rust [`InferenceImageSource`] without consuming this wrapper.
    #[must_use]
    pub fn clone_rust(&self) -> InferenceImageSource {
        self.inner.clone()
    }
}

// ---------------------------------------------------------------------------
// InferenceImage -- typed class mirror of `blazen_llm::InferenceImage`
// ---------------------------------------------------------------------------

/// An image payload attached to a chat message.
#[napi(js_name = "InferenceImage")]
pub struct JsInferenceImage {
    inner: InferenceImage,
}

#[napi]
impl JsInferenceImage {
    /// Build an image from raw encoded bytes (PNG/JPEG/WebP).
    #[must_use]
    #[napi(factory, js_name = "fromBytes")]
    pub fn from_bytes(bytes: Buffer) -> Self {
        Self {
            inner: InferenceImage::from_bytes(bytes.into()),
        }
    }

    /// Build an image from a local file path.
    #[must_use]
    #[napi(factory, js_name = "fromPath")]
    pub fn from_path(path: String) -> Self {
        Self {
            inner: InferenceImage::from_path(path),
        }
    }

    /// Build an image from an existing [`InferenceImageSource`].
    #[must_use]
    #[napi(factory, js_name = "fromSource")]
    pub fn from_source(source: &JsInferenceImageSource) -> Self {
        Self {
            inner: InferenceImage {
                source: source.clone_rust(),
            },
        }
    }

    /// The image source.
    #[must_use]
    #[napi(getter)]
    pub fn source(&self) -> JsInferenceImageSource {
        JsInferenceImageSource::from_rust(self.inner.source.clone())
    }
}

impl JsInferenceImage {
    /// Wrap an existing Rust [`InferenceImage`].
    #[must_use]
    pub fn from_rust(inner: InferenceImage) -> Self {
        Self { inner }
    }

    /// Unwrap into the underlying Rust [`InferenceImage`], consuming this wrapper.
    #[must_use]
    pub fn into_rust(self) -> InferenceImage {
        self.inner
    }

    /// Clone out the underlying Rust [`InferenceImage`] without consuming this wrapper.
    #[must_use]
    pub fn clone_rust(&self) -> InferenceImage {
        self.inner.clone()
    }
}

// ---------------------------------------------------------------------------
// ChatMessageInput -- typed class mirror of `blazen_llm::ChatMessageInput`
// ---------------------------------------------------------------------------

/// A single chat message for the local mistral.rs / llama.cpp inference path,
/// optionally carrying image attachments for vision-capable models.
#[napi(js_name = "ChatMessageInput")]
pub struct JsChatMessageInput {
    inner: ChatMessageInput,
}

#[napi]
impl JsChatMessageInput {
    /// Build a chat message with text and optional image attachments.
    #[must_use]
    #[napi(constructor)]
    pub fn new(role: JsChatRole, text: String, images: Option<Vec<&JsInferenceImage>>) -> Self {
        let images = images
            .unwrap_or_default()
            .into_iter()
            .map(JsInferenceImage::clone_rust)
            .collect();
        Self {
            inner: ChatMessageInput::with_images(role.into_rust(), text, images),
        }
    }

    /// Build a text-only chat message.
    #[must_use]
    #[napi(factory, js_name = "fromText")]
    pub fn from_text(role: JsChatRole, text: String) -> Self {
        Self {
            inner: ChatMessageInput::text(role.into_rust(), text),
        }
    }

    /// The role of the message author.
    #[must_use]
    #[napi(getter)]
    pub fn role(&self) -> JsChatRole {
        JsChatRole::from_rust(self.inner.role)
    }

    /// The textual content of the message.
    #[must_use]
    #[napi(getter)]
    pub fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// The image attachments on this message.
    #[must_use]
    #[napi(getter)]
    pub fn images(&self) -> Vec<JsInferenceImage> {
        self.inner
            .images
            .iter()
            .cloned()
            .map(JsInferenceImage::from_rust)
            .collect()
    }

    /// Whether this message has any image attachments.
    #[must_use]
    #[napi(getter, js_name = "hasImages")]
    pub fn has_images(&self) -> bool {
        self.inner.has_images()
    }
}

impl JsChatMessageInput {
    /// Wrap an existing Rust [`ChatMessageInput`].
    #[must_use]
    pub fn from_rust(inner: ChatMessageInput) -> Self {
        Self { inner }
    }

    /// Unwrap into the underlying Rust [`ChatMessageInput`], consuming this wrapper.
    #[must_use]
    pub fn into_rust(self) -> ChatMessageInput {
        self.inner
    }

    /// Clone out the underlying Rust [`ChatMessageInput`] without consuming this wrapper.
    #[must_use]
    pub fn clone_rust(&self) -> ChatMessageInput {
        self.inner.clone()
    }
}

// ---------------------------------------------------------------------------
// InferenceUsage -- typed class mirror of `blazen_llm::InferenceUsage`
// ---------------------------------------------------------------------------

/// Token usage statistics from a local inference call.
#[napi(js_name = "InferenceUsage")]
pub struct JsInferenceUsage {
    inner: InferenceUsage,
}

#[napi]
impl JsInferenceUsage {
    /// Number of prompt tokens consumed.
    #[must_use]
    #[napi(getter, js_name = "promptTokens")]
    pub fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Number of completion tokens generated.
    #[must_use]
    #[napi(getter, js_name = "completionTokens")]
    pub fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    /// Total tokens (prompt + completion).
    #[must_use]
    #[napi(getter, js_name = "totalTokens")]
    pub fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    /// Total wall-clock inference time in seconds.
    #[must_use]
    #[napi(getter, js_name = "totalTimeSec")]
    pub fn total_time_sec(&self) -> f64 {
        f64::from(self.inner.total_time_sec)
    }
}

impl JsInferenceUsage {
    /// Wrap an existing Rust [`InferenceUsage`].
    #[must_use]
    pub fn from_rust(inner: InferenceUsage) -> Self {
        Self { inner }
    }

    /// Unwrap into the underlying Rust [`InferenceUsage`], consuming this wrapper.
    #[must_use]
    pub fn into_rust(self) -> InferenceUsage {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// InferenceToolCall -- typed class mirror of `blazen_llm::InferenceToolCall`
// ---------------------------------------------------------------------------

/// A tool call requested by the model during local inference.
#[napi(js_name = "InferenceToolCall")]
pub struct JsInferenceToolCall {
    inner: InferenceToolCall,
}

#[napi]
impl JsInferenceToolCall {
    /// Build a tool call explicitly. Mainly useful for tests / replays.
    #[must_use]
    #[napi(constructor)]
    pub fn new(id: String, name: String, arguments: String) -> Self {
        Self {
            inner: InferenceToolCall {
                id,
                name,
                arguments,
            },
        }
    }

    /// Provider-assigned call identifier.
    #[must_use]
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Function name to invoke.
    #[must_use]
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// JSON-encoded arguments string.
    #[must_use]
    #[napi(getter)]
    pub fn arguments(&self) -> String {
        self.inner.arguments.clone()
    }
}

impl JsInferenceToolCall {
    /// Wrap an existing Rust [`InferenceToolCall`].
    #[must_use]
    pub fn from_rust(inner: InferenceToolCall) -> Self {
        Self { inner }
    }

    /// Unwrap into the underlying Rust [`InferenceToolCall`], consuming this wrapper.
    #[must_use]
    pub fn into_rust(self) -> InferenceToolCall {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// InferenceResult -- typed class mirror of `blazen_llm::InferenceResult`
// ---------------------------------------------------------------------------

/// Result of a non-streaming local inference call.
#[napi(js_name = "InferenceResult")]
pub struct JsInferenceResult {
    inner: InferenceResult,
}

#[napi]
impl JsInferenceResult {
    /// The generated text content, if any.
    #[must_use]
    #[napi(getter)]
    pub fn content(&self) -> Option<String> {
        self.inner.content.clone()
    }

    /// Reasoning / chain-of-thought content, if the model exposes it.
    #[must_use]
    #[napi(getter, js_name = "reasoningContent")]
    pub fn reasoning_content(&self) -> Option<String> {
        self.inner.reasoning_content.clone()
    }

    /// Tool calls requested by the model.
    #[must_use]
    #[napi(getter, js_name = "toolCalls")]
    pub fn tool_calls(&self) -> Vec<JsInferenceToolCall> {
        self.inner
            .tool_calls
            .iter()
            .cloned()
            .map(JsInferenceToolCall::from_rust)
            .collect()
    }

    /// Why the model stopped generating.
    #[must_use]
    #[napi(getter, js_name = "finishReason")]
    pub fn finish_reason(&self) -> String {
        self.inner.finish_reason.clone()
    }

    /// The model identifier that produced this result.
    #[must_use]
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// Token usage statistics for this call.
    #[must_use]
    #[napi(getter)]
    pub fn usage(&self) -> JsInferenceUsage {
        JsInferenceUsage::from_rust(self.inner.usage.clone())
    }
}

impl JsInferenceResult {
    /// Wrap an existing Rust [`InferenceResult`].
    #[must_use]
    pub fn from_rust(inner: InferenceResult) -> Self {
        Self { inner }
    }
}

// `InferenceResult` is not `Clone`, so we hand-implement a `Clone`-equivalent
// builder that clones the underlying fields explicitly.
impl Clone for JsInferenceResult {
    fn clone(&self) -> Self {
        Self {
            inner: InferenceResult {
                content: self.inner.content.clone(),
                reasoning_content: self.inner.reasoning_content.clone(),
                tool_calls: self.inner.tool_calls.clone(),
                finish_reason: self.inner.finish_reason.clone(),
                model: self.inner.model.clone(),
                usage: self.inner.usage.clone(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceChunk -- typed class mirror of `blazen_llm::InferenceChunk`
// ---------------------------------------------------------------------------

/// A single chunk from a streaming local inference call.
#[napi(js_name = "InferenceChunk")]
pub struct JsInferenceChunk {
    inner: InferenceChunk,
}

#[napi]
impl JsInferenceChunk {
    /// Incremental text content for this chunk, if any.
    #[must_use]
    #[napi(getter)]
    pub fn delta(&self) -> Option<String> {
        self.inner.delta.clone()
    }

    /// Incremental reasoning content for this chunk, if any.
    #[must_use]
    #[napi(getter, js_name = "reasoningDelta")]
    pub fn reasoning_delta(&self) -> Option<String> {
        self.inner.reasoning_delta.clone()
    }

    /// Tool calls completed in this chunk.
    #[must_use]
    #[napi(getter, js_name = "toolCalls")]
    pub fn tool_calls(&self) -> Vec<JsInferenceToolCall> {
        self.inner
            .tool_calls
            .iter()
            .cloned()
            .map(JsInferenceToolCall::from_rust)
            .collect()
    }

    /// Present in the final chunk when generation stops.
    #[must_use]
    #[napi(getter, js_name = "finishReason")]
    pub fn finish_reason(&self) -> Option<String> {
        self.inner.finish_reason.clone()
    }
}

impl JsInferenceChunk {
    /// Wrap an existing Rust [`InferenceChunk`].
    #[must_use]
    pub fn from_rust(inner: InferenceChunk) -> Self {
        Self { inner }
    }

    /// Unwrap into the underlying Rust [`InferenceChunk`], consuming this wrapper.
    #[must_use]
    pub fn into_rust(self) -> InferenceChunk {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// InferenceChunkStream -- typed class wrapper around `blazen_llm::InferenceChunkStream`
// ---------------------------------------------------------------------------

/// Async stream of [`InferenceChunk`] values.
///
/// Drive the stream by repeatedly awaiting [`InferenceChunkStream.next`].
/// Each call returns the next chunk, or `null` once the stream is exhausted.
/// Errors from the underlying engine are surfaced as awaited rejections.
///
/// ```javascript
/// const stream = await provider.inferStream(messages);
/// while (true) {
///   const chunk = await stream.next();
///   if (chunk === null) break;
///   process.stdout.write(chunk.delta ?? "");
/// }
/// ```
#[napi(js_name = "InferenceChunkStream")]
pub struct JsInferenceChunkStream {
    inner: Arc<Mutex<Option<InferenceChunkStream>>>,
}

#[napi]
impl JsInferenceChunkStream {
    /// Pull the next chunk from the stream. Returns `null` once the stream
    /// is exhausted. Any engine error is raised as a thrown exception.
    ///
    /// # Errors
    ///
    /// Returns a JS error if the underlying mistral.rs engine reports an
    /// inference failure on this chunk.
    #[napi]
    pub async fn next(&self) -> Result<Option<JsInferenceChunk>> {
        let mut guard = self.inner.lock().await;
        let Some(stream) = guard.as_mut() else {
            return Ok(None);
        };
        match stream.next().await {
            Some(Ok(chunk)) => Ok(Some(JsInferenceChunk::from_rust(chunk))),
            Some(Err(e)) => Err(mistralrs_error_to_napi(e)),
            None => {
                // Drop the underlying stream so subsequent calls short-circuit
                // without re-polling an exhausted iterator.
                *guard = None;
                Ok(None)
            }
        }
    }
}

impl JsInferenceChunkStream {
    /// Wrap an existing Rust [`InferenceChunkStream`].
    #[must_use]
    pub fn from_rust(stream: InferenceChunkStream) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(stream))),
        }
    }
}
