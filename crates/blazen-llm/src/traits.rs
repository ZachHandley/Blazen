//! Provider traits for LLM and media model implementations.
//!
//! These traits define the interface that all providers must implement.
//! You can create custom providers by implementing these traits on your
//! own structs:
//!
//! ```rust,ignore
//! use blazen_llm::{Model, ModelRequest, ModelResponse, BlazenError};
//!
//! struct MyCustomProvider { /* ... */ }
//!
//! #[async_trait::async_trait]
//! impl Model for MyCustomProvider {
//!     fn model_id(&self) -> &str { "my-model" }
//!     async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
//!         // Your implementation here
//!         todo!()
//!     }
//!     // stream() must also be implemented
//!     async fn stream(
//!         &self,
//!         request: ModelRequest,
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
//! [`Model`] is the central trait that every provider must implement.
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
    ChatMessage, EmbeddingResponse, ModelRequest, ModelResponse, StreamChunk, StructuredResponse,
    ToolDefinition,
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A chat completion model capable of generating text and invoking tools.
///
/// Implementors must handle both one-shot and streaming completions.
#[async_trait]
pub trait Model: Send + Sync {
    /// The identifier of the default model used by this provider.
    fn model_id(&self) -> &str;

    /// Perform a non-streaming chat completion.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError>;

    /// Perform a streaming chat completion, returning an async stream of chunks.
    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>;

    /// Optional configuration metadata for this provider.
    ///
    /// Custom providers return their stored config; built-in providers may
    /// return `None` (the default) or construct one from internal state.
    fn provider_config(&self) -> Option<&ProviderConfig> {
        None
    }

    /// Provider-level default retry configuration, if any.
    ///
    /// When the host runtime (Pipeline / Workflow / Step) sets up a
    /// [`RetryStack`](crate::retry::RetryStack), this provider default is
    /// the lowest-priority entry. Returning `None` (the default) means the
    /// provider has no opinion and the runtime falls back to
    /// [`RetryConfig::default`](crate::retry::RetryConfig::default).
    fn retry_config(&self) -> Option<&std::sync::Arc<crate::retry::RetryConfig>> {
        None
    }

    /// Escape hatch returning the underlying HTTP client this provider
    /// uses for wire-level I/O, if any.
    ///
    /// Power users can use this to issue raw requests for debugging,
    /// custom headers, or endpoints not yet covered by Blazen's typed
    /// surface area. Returns `None` for providers that have no
    /// HTTP client (e.g. local-inference providers, or
    /// [`CustomProvider`](crate::providers::custom::CustomProvider) where
    /// dispatch happens entirely in the host language).
    ///
    /// HTTP-based built-in providers override this to return
    /// `Some(self.http_client())` — the inherent method is the more
    /// ergonomic accessor when the concrete type is known.
    fn http_client(&self) -> Option<std::sync::Arc<dyn crate::http::HttpClient>> {
        None
    }
}

// ---------------------------------------------------------------------------
// StructuredOutput
// ---------------------------------------------------------------------------

/// Extract structured data from a model by providing a JSON Schema.
///
/// This trait has a blanket implementation for every [`Model`], so
/// providers do not need to implement it explicitly. It works by injecting the
/// schema derived from `T` via `schemars` into the `response_format` field of
/// the completion request.
#[async_trait]
pub trait StructuredOutput: Model {
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

        let request = ModelRequest {
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

/// Blanket implementation: every [`Model`] automatically supports
/// structured output extraction.
impl<M: Model> StructuredOutput for M {}

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

    /// Provider-level default retry configuration, if any.
    ///
    /// When the host runtime (Pipeline / Workflow / Step) sets up a
    /// [`RetryStack`](crate::retry::RetryStack), this provider default is
    /// the lowest-priority entry. Returning `None` (the default) means the
    /// provider has no opinion and the runtime falls back to
    /// [`RetryConfig::default`](crate::retry::RetryConfig::default).
    fn retry_config(&self) -> Option<&std::sync::Arc<crate::retry::RetryConfig>> {
        None
    }

    /// Escape hatch returning the underlying HTTP client this provider
    /// uses for wire-level I/O, if any.
    ///
    /// Power users can use this to issue raw requests for debugging,
    /// custom headers, or endpoints not yet covered by Blazen's typed
    /// surface area. Returns `None` for providers that have no HTTP
    /// client (e.g. local-inference providers).
    ///
    /// HTTP-based built-in providers override this to return
    /// `Some(self.http_client())` — the inherent method is the more
    /// ergonomic accessor when the concrete type is known.
    fn http_client(&self) -> Option<std::sync::Arc<dyn crate::http::HttpClient>> {
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

    /// Whether this tool is an *exit tool*. When the LLM invokes a tool whose
    /// `is_exit()` returns `true`, the agent loop returns immediately and the
    /// tool's arguments become the final result. Defaults to `false`.
    fn is_exit(&self) -> bool {
        false
    }
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
    /// Estimated memory footprint in bytes when loaded (host RAM if the model
    /// runs on CPU, GPU VRAM otherwise).
    pub memory_estimate_bytes: Option<u64>,
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
///    (`memory_bytes`) for monitoring or budget-aware scheduling.
///
/// Remote providers (`OpenAI`, Anthropic, Gemini, fal.ai, etc.) do NOT
/// implement this trait -- there is no local model to load or unload.
///
/// # Implementor guidance
///
/// - `load` and `unload` must both be **idempotent**. Calling `load` on
///   an already-loaded model is a no-op success; calling `unload` on an
///   already-unloaded model is also a no-op success.
/// - Inference methods (on [`Model`], [`EmbeddingModel`], etc.)
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
///     model_a.unload().await?;  // free memory
///     model_b.load().await?;    // load the other one
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait LocalModel: Send + Sync {
    /// Load the model into memory. Idempotent -- if the model is already
    /// loaded, this is a no-op that returns `Ok(())`.
    async fn load(&self) -> Result<(), crate::error::BlazenError>;

    /// Drop the loaded model and free its memory. Idempotent -- if the model
    /// is already unloaded, this is a no-op that returns `Ok(())`.
    async fn unload(&self) -> Result<(), crate::error::BlazenError>;

    /// Whether the model is currently loaded in memory.
    async fn is_loaded(&self) -> bool;

    /// Which device the model is configured to run on. Determines which
    /// memory pool the [`ModelManager`](../../blazen_manager/struct.ModelManager.html)
    /// charges this model against. Defaults to [`Device::Cpu`](crate::device::Device::Cpu)
    /// for backwards compatibility with implementors that have not yet
    /// declared a target.
    fn device(&self) -> crate::device::Device {
        crate::device::Device::Cpu
    }

    /// Approximate memory footprint in bytes (host RAM if [`Self::device`]
    /// returns [`Device::Cpu`](crate::device::Device::Cpu), GPU VRAM
    /// otherwise). Returns `None` for implementations that can't measure.
    async fn memory_bytes(&self) -> Option<u64> {
        None
    }

    /// Mount a `PEFT`-format `LoRA` adapter onto the loaded base model.
    ///
    /// `adapter_dir` must contain PEFT canonical layout:
    /// `adapter_model.safetensors` + `adapter_config.json`. The base model
    /// must already be loaded (`is_loaded()` returns `true`); the standard
    /// caller (`ModelManager::load_adapter`) guarantees this.
    ///
    /// Backends that cannot mount adapters return
    /// [`BlazenError::unsupported`](crate::error::BlazenError::unsupported)
    /// with a backend-specific diagnostic message. Backends that have to
    /// rebuild the underlying engine to honor the verb (e.g. mistral.rs,
    /// whose upstream API is `LoraModelBuilder` only) report
    /// [`AdapterMountStrategy::Rebuilt`] on the returned handle so callers
    /// know they paid full reload cost.
    ///
    /// The default implementation returns `Unsupported` so existing
    /// `LocalModel` implementors compile unchanged.
    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: AdapterOptions,
    ) -> Result<AdapterHandle, crate::error::BlazenError> {
        Err(crate::error::BlazenError::unsupported(
            "this backend does not support LoRA adapters",
        ))
    }

    /// Remove a previously-mounted adapter. Default impl returns
    /// [`BlazenError::unsupported`](crate::error::BlazenError::unsupported)
    /// for the same reason as [`Self::load_adapter`].
    async fn unload_adapter(
        &self,
        _handle: &AdapterHandle,
    ) -> Result<(), crate::error::BlazenError> {
        Err(crate::error::BlazenError::unsupported(
            "this backend does not support LoRA adapters",
        ))
    }

    /// List currently-mounted adapters. The default returns an empty
    /// `Vec` because "no adapters mounted" is a truthful state for any
    /// backend — adapter capability is probed via [`Self::load_adapter`].
    async fn list_adapters(&self) -> Vec<AdapterStatus> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// LoRA adapter types (used by LocalModel::load_adapter and ModelManager)
// ---------------------------------------------------------------------------

/// Caller-supplied parameters for [`LocalModel::load_adapter`].
#[derive(Debug, Clone)]
pub struct AdapterOptions {
    /// Caller-chosen identifier. Used as the `adapter_id` on the returned
    /// handle, surfaced in [`LocalModel::list_adapters`], and accepted by
    /// [`LocalModel::unload_adapter`]. Must be unique per (model, adapter)
    /// pair within a manager; [`ModelManager::load_adapter`] enforces this
    /// uniqueness (individual backends may not).
    pub adapter_id: String,

    /// Scaling factor applied to the adapter's delta-weights. `1.0` = full
    /// strength (PEFT convention). Backends that cannot honor non-`1.0`
    /// scales return
    /// [`BlazenError::unsupported`](crate::error::BlazenError::unsupported).
    pub scale: f32,
}

impl AdapterOptions {
    /// Construct options with the default scale (`1.0`).
    pub fn new(adapter_id: impl Into<String>) -> Self {
        Self {
            adapter_id: adapter_id.into(),
            scale: 1.0,
        }
    }
}

/// Token returned by [`LocalModel::load_adapter`], passed back to
/// [`LocalModel::unload_adapter`] to remove this specific adapter.
#[derive(Debug, Clone)]
pub struct AdapterHandle {
    /// Echoes [`AdapterOptions::adapter_id`].
    pub adapter_id: String,
    /// Bytes the adapter occupies on top of the base model, as reported by
    /// the backend. Used by [`ModelManager`] to update its pool accounting.
    pub memory_bytes: u64,
    /// What the backend actually did to honor the request. Forensic only —
    /// the manager treats every strategy identically.
    pub mount_strategy: AdapterMountStrategy,
}

/// How a backend honored a [`LocalModel::load_adapter`] request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterMountStrategy {
    /// Adapter was hot-attached to the live model (no engine rebuild).
    Attached,
    /// Engine was torn down and rebuilt with the adapter. Caller paid
    /// full base-weights reload cost.
    Rebuilt,
    /// Adapter weights were merged into the base in place; cannot be
    /// removed without reloading the base. Reserved for future backends.
    Merged,
}

/// Snapshot of a single mounted adapter, returned by
/// [`LocalModel::list_adapters`] and [`ModelManager::list_adapters`].
#[derive(Debug, Clone)]
pub struct AdapterStatus {
    pub adapter_id: String,
    pub scale: f32,
    pub source_dir: std::path::PathBuf,
    pub memory_bytes: u64,
}

/// How an adapter on the Blazen-orchestrator side gets handed to a *remote*
/// inference engine.
///
/// Required by external-engine proxy backends (`blazen-llm-vllm`,
/// `blazen-llm-ollama`, ...) when the engine runs on a different host than
/// Blazen so the adapter directory on the Blazen filesystem isn't directly
/// visible to the engine. In-process backends (mistral.rs, llama.cpp,
/// candle) ignore this — they always read straight from
/// [`AdapterOptions::adapter_id`]'s source path.
///
/// Defaults to [`AdapterTransport::LocalFs`] for backwards compatibility:
/// existing callers that don't set a transport keep the "engine reads
/// from a shared filesystem" behaviour they always had.
#[derive(Debug, Clone)]
pub enum AdapterTransport {
    /// Adapter directory is reachable on the engine host's filesystem at
    /// the given path. Wraps a path the engine can read directly — works
    /// for single-host deployments, `NFS` / `CephFS` / k8s PVC mounts where
    /// Blazen and the engine see the same disk.
    LocalFs(std::path::PathBuf),

    /// Adapter weights have been read into memory on the Blazen side and
    /// should be pushed to the engine over HTTP. The backend is responsible
    /// for choosing the right upload endpoint (vLLM has no first-class push
    /// API today; this variant is reserved for sidecar-style uploaders).
    HttpPush(Vec<u8>),

    /// Adapter lives on Hugging Face Hub; the engine pulls it itself.
    /// `repo` is the canonical `org/name` slug; `revision` pins a specific
    /// commit, branch, or tag (default: the engine's idea of "latest").
    HfHub {
        repo: String,
        revision: Option<String>,
    },
}

impl Default for AdapterTransport {
    /// Defaults to `LocalFs("")` — a back-compat sentinel saying "the
    /// caller didn't specify a transport; the adapter directory passed to
    /// [`LocalModel::load_adapter`] is itself the path the engine sees."
    /// Proxy providers should treat an empty path here as "fall through
    /// to the `adapter_dir` argument".
    fn default() -> Self {
        Self::LocalFs(std::path::PathBuf::new())
    }
}
