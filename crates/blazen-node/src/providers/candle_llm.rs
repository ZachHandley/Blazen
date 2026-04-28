//! JavaScript bindings for the local candle LLM provider.
//!
//! Exposes [`JsCandleLlmProvider`] as a NAPI class with a factory
//! constructor, async completion / streaming methods, and `LocalModel`
//! lifecycle controls (`load`, `unload`, `isLoaded`, `vramBytes`).
//!
//! Runs LLM inference entirely on-device using the candle engine.
//! No API key is required.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};
use blazen_llm::{
    CandleInferenceResult, CandleLlmCompletionModel, CandleLlmOptions, CandleLlmProvider,
};

use crate::error::{llm_error_to_napi, to_napi_error};
use crate::providers::completion_model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// JsCandleLlmOptions
// ---------------------------------------------------------------------------

/// Options for the local candle LLM backend.
///
/// All fields are optional. `modelId` defaults to a sensible model when
/// omitted, but most callers will set it explicitly.
///
/// ```javascript
/// const provider = CandleLlmProvider.create({
///   modelId: "meta-llama/Llama-3.2-1B",
///   device: "cuda:0",
/// });
/// ```
#[napi(object)]
pub struct JsCandleLlmOptions {
    /// `HuggingFace` model ID or local path to model weights.
    #[napi(js_name = "modelId")]
    pub model_id: Option<String>,
    /// Hardware device specifier (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub device: Option<String>,
    /// Quantization format string (e.g. `"q4_k_m"` for GGUF).
    pub quantization: Option<String>,
    /// `HuggingFace` revision / branch (e.g. `"main"`).
    pub revision: Option<String>,
    /// Maximum context length in tokens.
    #[napi(js_name = "contextLength")]
    pub context_length: Option<u32>,
    /// Path to cache downloaded models.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

impl From<JsCandleLlmOptions> for CandleLlmOptions {
    fn from(val: JsCandleLlmOptions) -> Self {
        Self {
            model_id: val.model_id,
            device: val.device,
            quantization: val.quantization,
            revision: val.revision,
            context_length: val.context_length.map(|v| v as usize),
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
        }
    }
}

// ---------------------------------------------------------------------------
// JsCandleLlmProvider NAPI class
// ---------------------------------------------------------------------------

/// A local candle LLM provider with completion and streaming.
///
/// ```javascript
/// const provider = CandleLlmProvider.create({
///   modelId: "meta-llama/Llama-3.2-1B",
/// });
/// await provider.load();
/// const response = await provider.complete([
///   ChatMessage.user("What is 2+2?"),
/// ]);
/// ```
#[napi(js_name = "CandleLlmProvider")]
pub struct JsCandleLlmProvider {
    /// The completion-capable wrapper -- `CandleLlmProvider` itself does
    /// not implement `CompletionModel`; `CandleLlmCompletionModel` does.
    completion: Arc<CandleLlmCompletionModel>,
    /// A second instance of the underlying `CandleLlmProvider` used for
    /// `LocalModel` lifecycle controls. `CandleLlmCompletionModel` owns
    /// its provider by value and exposes no accessor, so we construct two
    /// providers from the same options at factory time and let candle's
    /// internal model cache deduplicate the weights download.
    lifecycle: Arc<CandleLlmProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
impl JsCandleLlmProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new candle LLM provider.
    #[napi(factory)]
    pub fn create(options: Option<JsCandleLlmOptions>) -> Result<Self> {
        let opts: CandleLlmOptions = options.map(Into::into).unwrap_or_default();
        let provider = CandleLlmProvider::from_options(opts.clone())
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        let completion_provider = CandleLlmProvider::from_options(opts)
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(Self {
            completion: Arc::new(CandleLlmCompletionModel::new(completion_provider)),
            lifecycle: Arc::new(provider),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.completion.model_id().to_owned()
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
            .completion
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
            .completion
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
            .completion
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
        self.lifecycle.load().await.map_err(to_napi_error)
    }

    /// Drop the loaded model and free its memory / `VRAM`.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        self.lifecycle.unload().await.map_err(to_napi_error)
    }

    /// Whether the model is currently loaded in memory / `VRAM`.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> bool {
        self.lifecycle.is_loaded().await
    }

    /// Approximate `VRAM` footprint in bytes.
    #[napi(js_name = "vramBytes")]
    pub async fn vram_bytes(&self) -> Option<i64> {
        self.lifecycle
            .vram_bytes()
            .await
            .map(|b| i64::try_from(b).unwrap_or(i64::MAX))
    }
}

// ---------------------------------------------------------------------------
// JsCandleInferenceResult NAPI class
// ---------------------------------------------------------------------------

/// Result from a non-streaming candle inference call.
///
/// Mirrors the underlying `blazen_llm::CandleInferenceResult` struct and
/// exposes the generated text alongside token-count and timing metadata.
///
/// ```javascript
/// const result = new CandleInferenceResult("hello", 12, 4, 0.42);
/// console.log(result.content, result.promptTokens, result.completionTokens);
/// ```
#[napi(js_name = "CandleInferenceResult")]
pub struct JsCandleInferenceResult {
    /// The generated text content.
    content: String,
    /// Number of prompt tokens consumed.
    prompt_tokens: u32,
    /// Number of completion tokens generated.
    completion_tokens: u32,
    /// Wall-clock time for the inference in seconds.
    total_time_secs: f64,
}

#[napi]
impl JsCandleInferenceResult {
    /// Construct a new `CandleInferenceResult`.
    ///
    /// `promptTokens` and `completionTokens` are token counts; pass `0`
    /// when unknown. `totalTimeSecs` is the wall-clock duration of the
    /// inference in seconds.
    #[napi(constructor)]
    #[must_use]
    pub fn new(
        content: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_time_secs: f64,
    ) -> Self {
        Self {
            content,
            prompt_tokens,
            completion_tokens,
            total_time_secs,
        }
    }

    /// The generated text content.
    #[napi(getter)]
    #[must_use]
    pub fn content(&self) -> String {
        self.content.clone()
    }

    /// Number of prompt tokens consumed.
    #[napi(getter, js_name = "promptTokens")]
    #[must_use]
    pub const fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    /// Number of completion tokens generated.
    #[napi(getter, js_name = "completionTokens")]
    #[must_use]
    pub const fn completion_tokens(&self) -> u32 {
        self.completion_tokens
    }

    /// Wall-clock time for the inference in seconds.
    #[napi(getter, js_name = "totalTimeSecs")]
    #[must_use]
    pub const fn total_time_secs(&self) -> f64 {
        self.total_time_secs
    }
}

impl JsCandleInferenceResult {
    /// Convert from the underlying `blazen_llm::CandleInferenceResult`.
    ///
    /// `usize` token counts are saturated to `u32::MAX` on the (highly
    /// unlikely) overflow path; JavaScript consumers see a `number` either
    /// way and would not be able to represent values that large precisely.
    #[must_use]
    pub fn from_rust(inner: CandleInferenceResult) -> Self {
        Self {
            content: inner.content,
            prompt_tokens: u32::try_from(inner.prompt_tokens).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(inner.completion_tokens).unwrap_or(u32::MAX),
            total_time_secs: inner.total_time_secs,
        }
    }
}
