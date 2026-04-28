//! JavaScript bindings for the local llama.cpp LLM provider.
//!
//! Exposes [`JsLlamaCppProvider`] as a NAPI class with an async factory
//! constructor, async completion / streaming methods, and `LocalModel`
//! lifecycle controls (`load`, `unload`, `isLoaded`, `vramBytes`).
//!
//! Runs LLM inference entirely on-device using the llama.cpp engine.
//! No API key is required.

#![cfg(feature = "llamacpp")]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};
use blazen_llm::{
    LlamaCppChatMessageInput, LlamaCppChatRole, LlamaCppInferenceChunk,
    LlamaCppInferenceChunkStream, LlamaCppInferenceResult, LlamaCppInferenceUsage, LlamaCppOptions,
    LlamaCppProvider,
};
use tokio::sync::Mutex;

use crate::error::{llm_error_to_napi, to_napi_error};
use crate::providers::completion_model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// JsLlamaCppOptions
// ---------------------------------------------------------------------------

/// Options for the local llama.cpp LLM backend.
///
/// All fields are optional. `modelPath` defaults to a sensible local path
/// resolution if omitted, but most callers will set it explicitly.
///
/// ```javascript
/// const provider = await LlamaCppProvider.create({
///   modelPath: "/models/llama-3.2-1b-q4_k_m.gguf",
///   nGpuLayers: 32,
/// });
/// ```
#[napi(object)]
pub struct JsLlamaCppOptions {
    /// Path to the GGUF model file, or a `HuggingFace` model ID.
    #[napi(js_name = "modelPath")]
    pub model_path: Option<String>,
    /// Hardware device specifier (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub device: Option<String>,
    /// Quantization format string (e.g. `"q4_k_m"`).
    pub quantization: Option<String>,
    /// Maximum context length in tokens.
    #[napi(js_name = "contextLength")]
    pub context_length: Option<u32>,
    /// Number of layers to offload to GPU.
    #[napi(js_name = "nGpuLayers")]
    pub n_gpu_layers: Option<u32>,
    /// Path to cache downloaded models.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

impl From<JsLlamaCppOptions> for LlamaCppOptions {
    fn from(val: JsLlamaCppOptions) -> Self {
        Self {
            model_path: val.model_path,
            device: val.device,
            quantization: val.quantization,
            context_length: val.context_length.map(|v| v as usize),
            n_gpu_layers: val.n_gpu_layers,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
        }
    }
}

// ---------------------------------------------------------------------------
// JsLlamaCppProvider NAPI class
// ---------------------------------------------------------------------------

/// A local llama.cpp LLM provider with completion and streaming.
///
/// ```javascript
/// const provider = await LlamaCppProvider.create({
///   modelPath: "/models/llama.gguf",
/// });
/// const response = await provider.complete([
///   ChatMessage.user("What is 2+2?"),
/// ]);
/// ```
#[napi(js_name = "LlamaCppProvider")]
pub struct JsLlamaCppProvider {
    inner: Arc<LlamaCppProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
impl JsLlamaCppProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new llama.cpp provider.
    ///
    /// This is async because llama.cpp may download the GGUF model file
    /// from `HuggingFace` on first use.
    #[napi(factory)]
    pub async fn create(options: Option<JsLlamaCppOptions>) -> Result<Self> {
        let opts: LlamaCppOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                LlamaCppProvider::from_options(opts)
                    .await
                    .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?,
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
// JsLlamaCppChatRole
// ---------------------------------------------------------------------------

/// Simplified chat role for the llama.cpp bridge layer.
///
/// Mirrors `blazen_llm::LlamaCppChatRole` (re-exported from
/// `blazen-llm-llamacpp`). Distinct from the framework-wide `JsRole` because
/// the llama.cpp bridge layer uses its own simplified four-variant enum.
#[napi(string_enum, js_name = "LlamaCppChatRole")]
pub enum JsLlamaCppChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl From<JsLlamaCppChatRole> for LlamaCppChatRole {
    fn from(role: JsLlamaCppChatRole) -> Self {
        match role {
            JsLlamaCppChatRole::System => Self::System,
            JsLlamaCppChatRole::User => Self::User,
            JsLlamaCppChatRole::Assistant => Self::Assistant,
            JsLlamaCppChatRole::Tool => Self::Tool,
        }
    }
}

impl From<LlamaCppChatRole> for JsLlamaCppChatRole {
    fn from(role: LlamaCppChatRole) -> Self {
        match role {
            LlamaCppChatRole::System => Self::System,
            LlamaCppChatRole::User => Self::User,
            LlamaCppChatRole::Assistant => Self::Assistant,
            LlamaCppChatRole::Tool => Self::Tool,
        }
    }
}

// ---------------------------------------------------------------------------
// JsLlamaCppChatMessageInput
// ---------------------------------------------------------------------------

/// A single chat message for input to the llama.cpp provider.
///
/// ```javascript
/// const msg = new LlamaCppChatMessageInput(LlamaCppChatRole.User, "Hello");
/// ```
#[napi(js_name = "LlamaCppChatMessageInput")]
pub struct JsLlamaCppChatMessageInput {
    inner: LlamaCppChatMessageInput,
}

#[napi]
impl JsLlamaCppChatMessageInput {
    /// Build a chat message from a role and text body.
    #[napi(constructor)]
    #[must_use]
    pub fn new(role: JsLlamaCppChatRole, text: String) -> Self {
        Self {
            inner: LlamaCppChatMessageInput::new(role.into(), text),
        }
    }

    /// Who produced this message.
    #[napi(getter)]
    #[must_use]
    pub fn role(&self) -> JsLlamaCppChatRole {
        self.inner.role.into()
    }

    /// The textual content of the message.
    #[napi(getter)]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.inner.text
    }
}

// ---------------------------------------------------------------------------
// JsLlamaCppInferenceUsage
// ---------------------------------------------------------------------------

/// Token usage statistics from a llama.cpp inference call.
#[napi(js_name = "LlamaCppInferenceUsage")]
pub struct JsLlamaCppInferenceUsage {
    inner: LlamaCppInferenceUsage,
}

#[napi]
impl JsLlamaCppInferenceUsage {
    /// Tokens in the prompt.
    #[napi(getter, js_name = "promptTokens")]
    #[must_use]
    pub fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Tokens generated.
    #[napi(getter, js_name = "completionTokens")]
    #[must_use]
    pub fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    /// Total tokens (prompt + completion).
    #[napi(getter, js_name = "totalTokens")]
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    /// Total wall-clock inference time in seconds.
    #[napi(getter, js_name = "totalTimeSec")]
    #[must_use]
    pub fn total_time_sec(&self) -> f64 {
        f64::from(self.inner.total_time_sec)
    }
}

impl From<LlamaCppInferenceUsage> for JsLlamaCppInferenceUsage {
    fn from(inner: LlamaCppInferenceUsage) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// JsLlamaCppInferenceResult
// ---------------------------------------------------------------------------

/// Result of a single non-streaming llama.cpp inference call.
#[napi(js_name = "LlamaCppInferenceResult")]
pub struct JsLlamaCppInferenceResult {
    inner: LlamaCppInferenceResult,
}

#[napi]
impl JsLlamaCppInferenceResult {
    /// The generated text content, if any.
    #[napi(getter)]
    #[must_use]
    pub fn content(&self) -> Option<String> {
        self.inner.content.clone()
    }

    /// Why the model stopped generating (e.g. `"stop"`, `"length"`).
    #[napi(getter, js_name = "finishReason")]
    #[must_use]
    pub fn finish_reason(&self) -> &str {
        &self.inner.finish_reason
    }

    /// The model identifier that produced this result.
    #[napi(getter)]
    #[must_use]
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Token usage statistics.
    #[napi(getter)]
    #[must_use]
    pub fn usage(&self) -> JsLlamaCppInferenceUsage {
        JsLlamaCppInferenceUsage {
            inner: self.inner.usage.clone(),
        }
    }
}

impl From<LlamaCppInferenceResult> for JsLlamaCppInferenceResult {
    fn from(inner: LlamaCppInferenceResult) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// JsLlamaCppInferenceChunk
// ---------------------------------------------------------------------------

/// A single chunk from a streaming llama.cpp inference call.
#[napi(js_name = "LlamaCppInferenceChunk")]
pub struct JsLlamaCppInferenceChunk {
    inner: LlamaCppInferenceChunk,
}

#[napi]
impl JsLlamaCppInferenceChunk {
    /// Incremental text content, if any.
    #[napi(getter)]
    #[must_use]
    pub fn delta(&self) -> Option<String> {
        self.inner.delta.clone()
    }

    /// Present in the final chunk when generation stops (e.g. `"stop"`,
    /// `"length"`).
    #[napi(getter, js_name = "finishReason")]
    #[must_use]
    pub fn finish_reason(&self) -> Option<String> {
        self.inner.finish_reason.clone()
    }
}

impl From<LlamaCppInferenceChunk> for JsLlamaCppInferenceChunk {
    fn from(inner: LlamaCppInferenceChunk) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// JsLlamaCppInferenceChunkStream
// ---------------------------------------------------------------------------

/// An async iterator over llama.cpp streaming inference chunks.
///
/// Wraps the underlying `Pin<Box<dyn Stream<Item = Result<...>> + Send>>`
/// behind a `tokio::sync::Mutex` so it can be polled from JavaScript via
/// `next()`. Each successful poll yields the next `LlamaCppInferenceChunk`
/// (or `null` when the stream ends); a stream-level error is thrown.
///
/// ```javascript
/// for (let chunk = await stream.next(); chunk !== null; chunk = await stream.next()) {
///   process.stdout.write(chunk.delta ?? "");
/// }
/// ```
#[napi(js_name = "LlamaCppInferenceChunkStream")]
pub struct JsLlamaCppInferenceChunkStream {
    inner: Arc<Mutex<LlamaCppInferenceChunkStream>>,
}

#[napi]
impl JsLlamaCppInferenceChunkStream {
    /// Pull the next chunk from the stream.
    ///
    /// Returns `null` once the stream is exhausted.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error if the underlying llama.cpp inference fails
    /// partway through generation (e.g. tokenization failure or a context
    /// decode error).
    #[napi]
    pub async fn next(&self) -> Result<Option<JsLlamaCppInferenceChunk>> {
        let mut guard = self.inner.lock().await;
        match guard.next().await {
            Some(Ok(chunk)) => Ok(Some(chunk.into())),
            Some(Err(err)) => Err(napi::Error::from_reason(err.to_string())),
            None => Ok(None),
        }
    }
}

impl From<LlamaCppInferenceChunkStream> for JsLlamaCppInferenceChunkStream {
    fn from(stream: LlamaCppInferenceChunkStream) -> Self {
        Self {
            inner: Arc::new(Mutex::new(stream)),
        }
    }
}
