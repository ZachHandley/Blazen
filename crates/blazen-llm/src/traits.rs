//! Provider traits for LLM and media model implementations.
//!
//! These traits define the interface that all providers must implement.
//! You can create custom providers by implementing these traits on your
//! own structs:
//!
//! ```rust,ignore
//! use blazen_llm::{CompletionModel, CompletionRequest, CompletionResponse, BlazenError};
//!
//! struct MyCustomProvider { /* ... */ }
//!
//! #[async_trait::async_trait]
//! impl CompletionModel for MyCustomProvider {
//!     fn model_id(&self) -> &str { "my-model" }
//!     async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, BlazenError> {
//!         // Your implementation here
//!         todo!()
//!     }
//!     // stream() must also be implemented
//!     async fn stream(
//!         &self,
//!         request: CompletionRequest,
//!     ) -> Result<
//!         std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<blazen_llm::StreamChunk, BlazenError>> + Send>>,
//!         BlazenError,
//!     > {
//!         // Your streaming implementation here
//!         todo!()
//!     }
//! }
//! ```
//!
//! [`CompletionModel`] is the central trait that every provider must implement.
//! [`StructuredOutput`] and [`EmbeddingModel`] extend the surface area with
//! schema-constrained extraction and vector embeddings respectively.
//! [`ModelRegistry`] allows providers to advertise their available models.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::BlazenError;
use crate::types::{
    ChatMessage, CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk,
    StructuredResponse, ToolDefinition,
};

// ---------------------------------------------------------------------------
// CompletionModel
// ---------------------------------------------------------------------------

/// A chat completion model capable of generating text and invoking tools.
///
/// Implementors must handle both one-shot and streaming completions.
#[async_trait]
pub trait CompletionModel: Send + Sync {
    /// The identifier of the default model used by this provider.
    fn model_id(&self) -> &str;

    /// Perform a non-streaming chat completion.
    async fn complete(&self, request: CompletionRequest)
    -> Result<CompletionResponse, BlazenError>;

    /// Perform a streaming chat completion, returning an async stream of chunks.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>;

    /// Optional configuration metadata for this provider.
    ///
    /// Custom providers return their stored config; built-in providers may
    /// return `None` (the default) or construct one from internal state.
    fn provider_config(&self) -> Option<&ProviderConfig> {
        None
    }
}

// ---------------------------------------------------------------------------
// StructuredOutput
// ---------------------------------------------------------------------------

/// Extract structured data from a model by providing a JSON Schema.
///
/// This trait has a blanket implementation for every [`CompletionModel`], so
/// providers do not need to implement it explicitly. It works by injecting the
/// schema derived from `T` via `schemars` into the `response_format` field of
/// the completion request.
#[async_trait]
pub trait StructuredOutput: CompletionModel {
    /// Extract a value of type `T` from the model's response.
    ///
    /// The conversation in `messages` is sent to the model with a JSON Schema
    /// constraint derived from `T`. The response is then deserialized into `T`.
    async fn extract<T: JsonSchema + DeserializeOwned + Send>(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<StructuredResponse<T>, BlazenError> {
        let schema = schemars::schema_for!(T);
        let schema_json = serde_json::to_value(&schema)?;

        let request = CompletionRequest {
            messages,
            tools: vec![],
            temperature: Some(0.0),
            max_tokens: None,
            top_p: None,
            response_format: Some(schema_json),
            model: None,
            modalities: None,
            image_config: None,
            audio_config: None,
        };

        let response = self.complete(request).await?;
        let content = response.content.ok_or_else(BlazenError::no_content)?;
        let data: T = serde_json::from_str(&content)?;
        Ok(StructuredResponse {
            data,
            usage: response.usage,
            model: response.model,
            cost: response.cost,
            timing: response.timing,
            metadata: response.metadata,
            reasoning: response.reasoning,
            citations: response.citations,
            artifacts: response.artifacts,
        })
    }
}

/// Blanket implementation: every [`CompletionModel`] automatically supports
/// structured output extraction.
impl<M: CompletionModel> StructuredOutput for M {}

// ---------------------------------------------------------------------------
// EmbeddingModel
// ---------------------------------------------------------------------------

/// A model that produces vector embeddings for text inputs.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// The identifier of the embedding model.
    fn model_id(&self) -> &str;

    /// The dimensionality of the vectors produced by this model.
    fn dimensions(&self) -> usize;

    /// Embed one or more texts, returning one vector per input text.
    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError>;

    /// Optional configuration metadata for this provider.
    fn provider_config(&self) -> Option<&ProviderConfig> {
        None
    }
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// A callable tool that can be invoked by an LLM during a conversation.
///
/// Implementations describe their schema via [`Tool::definition`] and handle
/// invocations via [`Tool::execute`].
#[async_trait]
pub trait Tool: Send + Sync {
    /// Return the JSON Schema definition of this tool.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool. Returns a [`ToolOutput`] carrying both the
    /// caller-visible `data` and an optional `llm_override` controlling
    /// what is sent to the model on the next turn.
    ///
    /// For the common case (no override), use `Ok(value.into())` —
    /// `From<Value> for ToolOutput<Value>` constructs a default-shaped output.
    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<crate::types::ToolOutput<serde_json::Value>, BlazenError>;
}

// ---------------------------------------------------------------------------
// Model information and registry
// ---------------------------------------------------------------------------

/// Information about a model offered by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ModelInfo {
    /// The model identifier used in API requests (e.g. `"gpt-4o"`).
    pub id: String,
    /// A human-readable display name, if different from the id.
    pub name: Option<String>,
    /// The provider that serves this model.
    pub provider: String,
    /// Maximum context window length in tokens.
    pub context_length: Option<u64>,
    /// Pricing information, if available.
    pub pricing: Option<ModelPricing>,
    /// What this model can do.
    pub capabilities: ModelCapabilities,
}

/// Pricing information for a model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ModelPricing {
    /// Cost per million input tokens in USD.
    pub input_per_million: Option<f64>,
    /// Cost per million output tokens in USD.
    pub output_per_million: Option<f64>,
    /// Cost per image (for image generation models).
    pub per_image: Option<f64>,
    /// Cost per second of compute (for fal.ai style pricing).
    pub per_second: Option<f64>,
}

/// Capabilities that a model may support.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[allow(clippy::struct_excessive_bools)]
pub struct ModelCapabilities {
    /// Supports chat completions.
    pub chat: bool,
    /// Supports streaming responses.
    pub streaming: bool,
    /// Supports tool/function calling.
    pub tool_use: bool,
    /// Supports structured output (JSON schema constraints).
    pub structured_output: bool,
    /// Supports vision / image inputs.
    pub vision: bool,
    /// Supports image generation.
    pub image_generation: bool,
    /// Supports text embeddings.
    pub embeddings: bool,
    /// Video generation support (text-to-video, image-to-video).
    pub video_generation: bool,
    /// Text-to-speech synthesis.
    pub text_to_speech: bool,
    /// Speech-to-text transcription.
    pub speech_to_text: bool,
    /// Audio generation (music, sound effects).
    pub audio_generation: bool,
    /// 3D model generation.
    pub three_d_generation: bool,
}

/// A provider that can list its available models.
#[async_trait]
pub trait ModelRegistry: Send + Sync {
    /// List all models available from this provider.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError>;

    /// Look up a specific model by its identifier.
    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, BlazenError>;
}

// ---------------------------------------------------------------------------
// ProviderInfo
// ---------------------------------------------------------------------------

/// Capabilities advertised by a provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[allow(clippy::struct_excessive_bools)]
pub struct ProviderCapabilities {
    /// Whether the provider supports streaming responses.
    pub streaming: bool,
    /// Whether the provider supports tool/function calling.
    pub tool_calling: bool,
    /// Whether the provider supports structured output (JSON mode).
    pub structured_output: bool,
    /// Whether the provider supports vision/image inputs.
    pub vision: bool,
    /// Whether the provider supports the /models listing endpoint.
    pub model_listing: bool,
    /// Whether the provider supports embeddings.
    pub embeddings: bool,
}

/// Configuration metadata for a provider instance.
///
/// Carries identity, endpoint, pricing, and resource information that
/// custom providers set at construction time. Built-in providers populate
/// this from their own internal state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ProviderConfig {
    /// A human-readable name for this provider instance.
    pub name: Option<String>,
    /// The model identifier (e.g. `"my-org/llama-3-8b"`).
    pub model_id: Option<String>,
    /// A provider identifier (e.g. `"elevenlabs"`, `"fal"`).
    pub provider_id: Option<String>,
    /// Base URL for HTTP-based providers.
    pub base_url: Option<String>,
    /// Context window size in tokens.
    pub context_length: Option<u64>,
    /// Maximum output tokens the model supports.
    pub max_output_tokens: Option<u64>,
    /// Estimated VRAM footprint in bytes when loaded.
    pub vram_estimate_bytes: Option<u64>,
    /// Pricing information for automatic cost tracking.
    pub pricing: Option<ModelPricing>,
    /// Capability flags.
    pub capabilities: Option<ModelCapabilities>,
}

/// Information about a provider's identity, endpoint, and capabilities.
///
/// Implemented by each dedicated provider to expose its configuration
/// for discovery, registry, and routing purposes.
pub trait ProviderInfo {
    /// The provider's canonical name (e.g. "groq", "openai", "anthropic").
    fn provider_name(&self) -> &str;

    /// The provider's base API URL.
    fn base_url(&self) -> &str;

    /// The provider's capabilities.
    fn capabilities(&self) -> ProviderCapabilities;
}

// ---------------------------------------------------------------------------
// LocalModel -- explicit load/unload for in-process model providers
// ---------------------------------------------------------------------------

/// A model that is loaded into memory / `VRAM` on the current process, as
/// opposed to being reached over an HTTP API.
///
/// Providers that hold actual model weights (mistral.rs, llama.cpp, candle,
/// whisper.cpp, etc.) implement this trait so callers can:
///
/// 1. Explicitly trigger loading (`load`) -- avoiding the "lazy load on
///    first call" latency spike during a workflow step that needs
///    predictable timing.
/// 2. Explicitly free `GPU` memory (`unload`) -- letting a single Blazen
///    process swap models in and out, or release `VRAM` when idle.
/// 3. Query load state (`is_loaded`) and an approximate `VRAM` footprint
///    (`vram_bytes`) for monitoring or budget-aware scheduling.
///
/// Remote providers (`OpenAI`, Anthropic, Gemini, fal.ai, etc.) do NOT
/// implement this trait -- there is no local model to load or unload.
///
/// # Implementor guidance
///
/// - `load` and `unload` must both be **idempotent**. Calling `load` on
///   an already-loaded model is a no-op success; calling `unload` on an
///   already-unloaded model is also a no-op success.
/// - Inference methods (on [`CompletionModel`], [`EmbeddingModel`], etc.)
///   should auto-load on first call to preserve today's lazy-init
///   behavior -- this trait only adds explicit control on top.
/// - After `unload` returns, the provider may be re-loaded; the struct
///   itself must not be invalidated.
///
/// # Example
///
/// ```rust,ignore
/// use blazen_llm::traits::LocalModel;
///
/// async fn swap_models(
///     model_a: &impl LocalModel,
///     model_b: &impl LocalModel,
/// ) -> Result<(), blazen_llm::BlazenError> {
///     model_a.unload().await?;  // free VRAM
///     model_b.load().await?;    // load the other one
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait LocalModel: Send + Sync {
    /// Load the model into memory / `VRAM`. Idempotent -- if the model is
    /// already loaded, this is a no-op that returns `Ok(())`.
    async fn load(&self) -> Result<(), crate::error::BlazenError>;

    /// Drop the loaded model and free its memory / `VRAM`. Idempotent --
    /// if the model is already unloaded, this is a no-op that returns
    /// `Ok(())`.
    async fn unload(&self) -> Result<(), crate::error::BlazenError>;

    /// Whether the model is currently loaded in memory / `VRAM`.
    async fn is_loaded(&self) -> bool;

    /// Approximate memory footprint in bytes, if the implementation can
    /// report it. Returns `None` for implementations that have no way to
    /// measure (e.g. because the underlying runtime does not expose it).
    ///
    /// The default implementation returns `None`.
    async fn vram_bytes(&self) -> Option<u64> {
        None
    }
}
