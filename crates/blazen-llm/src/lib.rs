//! # `Blazen` LLM
//!
//! Provides traits and implementations for large language model providers
//! (`OpenAI`, Anthropic, Gemini, Azure, fal.ai, and many `OpenAI`-compatible
//! services) with streaming support, tool calling, and structured output
//! via JSON Schema.
//!
//! ## Feature flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `reqwest` | Enables the native HTTP client (for non-WASM targets) |
//! | `tiktoken` | Enables exact BPE token counting |
//!
//! All providers are always compiled -- no feature flags needed.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use blazen_llm::{Model, ModelRequest, ChatMessage};
//! use blazen_llm::providers::openai::OpenAiProvider;
//!
//! # async fn example() -> Result<(), blazen_llm::BlazenError> {
//! let model = OpenAiProvider::new("sk-...");
//! let request = ModelRequest::new(vec![
//!     ChatMessage::user("What is 2 + 2?"),
//! ]);
//! let response = model.complete(request).await?;
//! println!("{}", response.content.unwrap_or_default());
//! # Ok(())
//! # }
//! ```
//!
//! ## Multi-provider support
//!
//! Use [`providers::openai_compat::OpenAiCompatProvider`] to connect to any
//! OpenAI-compatible service:
//!
//! ```rust,no_run
//! use blazen_llm::providers::groq::GroqProvider;
//! use blazen_llm::providers::openrouter::OpenRouterProvider;
//! use blazen_llm::providers::together::TogetherProvider;
//!
//! # fn example() {
//! // Groq (fast inference)
//! let groq = GroqProvider::new("gsk-...");
//!
//! // OpenRouter (400+ models)
//! let openrouter = OpenRouterProvider::new("sk-or-...");
//!
//! // Together AI
//! let together = TogetherProvider::new("...");
//! # }
//! ```

pub mod agent;
pub mod artifacts;
pub mod batch;
pub mod cache;
pub mod chat_window;
pub mod compute;
pub mod content;
pub mod device;
pub mod error;
pub mod events;
pub mod fallback;
pub mod http;
#[cfg(all(feature = "reqwest", not(target_arch = "wasm32")))]
mod http_reqwest;
pub mod keys;
pub(crate) mod sleep;
#[cfg(all(feature = "reqwest", not(target_arch = "wasm32")))]
pub use http_reqwest::ReqwestHttpClient;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
mod http_fetch;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub use http_fetch::FetchHttpClient;
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub mod http_napi_wasi;

/// Returns the platform-appropriate default HTTP client.
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub(crate) fn default_http_client() -> std::sync::Arc<dyn http::HttpClient> {
    FetchHttpClient::new().into_arc()
}

/// Returns the platform-appropriate default HTTP client.
#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest"))]
pub(crate) fn default_http_client() -> std::sync::Arc<dyn http::HttpClient> {
    ReqwestHttpClient::new().into_arc()
}

/// Returns the platform-appropriate default HTTP client.
///
/// On wasi, this is a [`http_napi_wasi::LazyHttpClient`] proxy that forwards
/// to whatever client the host registered via
/// [`http_napi_wasi::register_default`] (typically the napi
/// `setDefaultHttpClient` binding wrapping `globalThis.fetch`).
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub(crate) fn default_http_client() -> std::sync::Arc<dyn http::HttpClient> {
    http_napi_wasi::LazyHttpClient::new().into_arc()
}

pub mod media;
pub mod middleware;
pub mod pricing;
pub mod pricing_fetcher;
pub mod providers;
pub mod quantization;
pub mod retry;
pub mod tokens;
pub mod traits;
pub mod typed_tool;
pub mod types;
pub mod usage_recording;

pub use providers::base::LlmProviderDefaults;
pub use providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};
pub use providers::capabilities::{
    BackgroundRemovalProvider, CodecProvider, EmbeddingProvider, ImageGenProvider, LLMProvider,
    MusicProvider, SttProvider, ThreeDProvider, TtsProvider, VcProvider, VideoProvider,
};
pub use providers::custom::{ApiProtocol, CustomProvider, CustomProviderHandle};
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
pub use providers::custom::{lm_studio, ollama, openai_compat};
pub use providers::defaults::{
    AudioMusicProviderDefaults, AudioSpeechProviderDefaults, BackgroundRemovalProviderDefaults,
    BaseProviderDefaults, BeforeBackgroundRemovalRequestHook, BeforeImageRequestHook,
    BeforeModelRequestHook, BeforeMusicRequestHook, BeforeRequestHook, BeforeSpeechRequestHook,
    BeforeThreeDRequestHook, BeforeTranscriptionRequestHook, BeforeUpscaleRequestHook,
    BeforeVideoRequestHook, BeforeVoiceCloneRequestHook, EmbeddingProviderDefaults,
    ImageGenerationProviderDefaults, ImageUpscaleProviderDefaults, ProviderDefaults,
    ThreeDProviderDefaults, TranscriptionProviderDefaults, VideoProviderDefaults,
    VoiceCloningProviderDefaults,
};

// Re-export primary types at crate root for ergonomic imports.
pub use agent::{
    AgentConfig, AgentEvent, AgentResult, FINISH_WORKFLOW_TOOL_NAME, finish_workflow_tool,
    run_agent, run_agent_with_callback,
};
pub use artifacts::extract_inline_artifacts;
pub use batch::{BatchConfig, BatchResult, complete_batch};
pub use cache::{CacheConfig, CacheStrategy, CachedModel};
pub use chat_window::ChatWindow;
pub use compute::{
    // Audio
    AudioGeneration,
    AudioResult,
    // Background removal
    BackgroundRemoval,
    BackgroundRemovalRequest,
    // Core compute
    ComputeProvider,
    ComputeRequest,
    ComputeResult,
    // Image
    ImageGeneration,
    ImageModel,
    ImageRequest,
    ImageResult,
    JobHandle,
    JobStatus,
    MusicRequest,
    SpeechRequest,
    // 3D
    ThreeDGeneration,
    ThreeDRequest,
    ThreeDResult,
    // Transcription
    Transcription,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
    UpscaleRequest,
    // Video
    VideoGeneration,
    VideoRequest,
    VideoResult,
};
pub use device::{Device, Pool};
#[allow(deprecated)]
pub use error::LlmError;
pub use error::{BlazenError, ComputeErrorKind, MediaErrorKind, ModelErrorKind};
pub use events::{StreamChunkEvent, StreamCompleteEvent};
pub use fallback::FallbackModel;
pub use media::{
    Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput, MediaType,
};
pub use middleware::{CacheMiddleware, Middleware, MiddlewareStack, RetryMiddleware};
pub use pricing::{PricingEntry, compute_cost, lookup_pricing, register_pricing};
pub use pricing_fetcher::{
    DEFAULT_MODEL_PRICING_URL_BASE, DEFAULT_PRICING_URL, fetch_one_from_url, refresh_from_url,
};
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "reqwest"),
    target_arch = "wasm32"
))]
pub use pricing_fetcher::{
    fetch_one_default, fetch_one_default_with_url_base, refresh_default, refresh_default_with_url,
};
pub use quantization::Quantization;
pub use retry::{RetryConfig, RetryModel};
#[cfg(feature = "tiktoken")]
pub use tokens::TiktokenCounter;
pub use tokens::{EstimateCounter, TokenCounter};
pub use traits::{
    AdapterHandle, AdapterMountStrategy, AdapterOptions, AdapterStatus, AdapterTransport,
    EmbeddingModel, LocalModel, Model, ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry,
    ProviderCapabilities, ProviderConfig, ProviderInfo, StructuredOutput, Tool,
};
pub use typed_tool::{TypedTool, typed_tool_simple};
pub use types::{
    Artifact, AudioContent, ChatMessage, Citation, ContentPart, EmbeddingResponse, FileContent,
    FinishReason, ImageContent, ImageSource, LlmPayload, MediaSource, MessageContent, ModelRequest,
    ModelResponse, ProviderId, ReasoningTrace, RequestTiming, ResponseFormat, Role, StreamChunk,
    StructuredResponse, TokenUsage, ToolCall, ToolDefinition, ToolOutput, VideoContent,
};
pub use usage_recording::{
    NoopUsageEmitter, UsageEmitter, UsageRecordingEmbeddingModel, UsageRecordingModel,
};

// ---------------------------------------------------------------------------
// Local inference backends (gated behind feature flags)
// ---------------------------------------------------------------------------

mod backends;

#[cfg(feature = "embed")]
pub use blazen_embed::{EmbedError, EmbedModel, EmbedOptions, EmbedResponse};

#[cfg(feature = "mistralrs")]
pub use blazen_llm_mistralrs::{
    ChatMessageInput, ChatRole, InferenceChunk, InferenceChunkStream, InferenceImage,
    InferenceImageSource, InferenceResult, InferenceToolCall, InferenceUsage, MistralRsError,
    MistralRsOptions, MistralRsProvider, MountedAdapter,
};

#[cfg(feature = "candle-embed")]
pub use blazen_embed_candle::{CandleEmbedError, CandleEmbedModel, CandleEmbedOptions};

#[cfg(feature = "audio-stt")]
pub use blazen_audio_stt::{DynSttProvider, SttBackendHandle, SttError, SttOptions};

// Backwards-compat aliases for the pre-PR-AUDIO names. The bindings still
// reference these — they will be migrated to the canonical
// `WhisperCppBackend` / `WhisperCppOptions` / `SttError` names in a
// follow-up sweep.
#[cfg(feature = "audio-stt-whispercpp")]
pub use crate::compat::whisper::WhisperCppProvider;
#[cfg(feature = "audio-stt-whispercpp")]
pub use blazen_audio_stt::SttError as WhisperError;
#[cfg(feature = "audio-stt-whispercpp")]
pub use blazen_audio_stt::backends::whispercpp::{
    WhisperCppOptions as WhisperOptions, WhisperModel,
};

#[cfg(feature = "audio-stt-faster-whisper")]
pub use blazen_audio_stt::backends::faster_whisper::{FasterWhisperBackend, FasterWhisperConfig};

/// Backwards-compatibility shims for legacy crate names that were
/// dissolved in the PR-AUDIO restructure. Slated for removal in two
/// releases — new code should reach for the canonical types directly.
#[cfg(feature = "audio-stt-whispercpp")]
pub mod compat;

#[cfg(feature = "diffusion")]
pub use blazen_image_diffusion::{
    DiffusionError, DiffusionOptions, DiffusionProvider, DiffusionScheduler,
};

#[cfg(feature = "candle-llm")]
pub use backends::candle_llm::CandleLlmModel;
#[cfg(feature = "candle-llm")]
pub use blazen_llm_candle::{
    CandleInferenceResult, CandleLlmError, CandleLlmOptions, CandleLlmProvider,
};

#[cfg(feature = "llamacpp")]
pub use blazen_llm_llamacpp::{
    ChatMessageInput as LlamaCppChatMessageInput, ChatRole as LlamaCppChatRole,
    InferenceChunk as LlamaCppInferenceChunk, InferenceChunkStream as LlamaCppInferenceChunkStream,
    InferenceResult as LlamaCppInferenceResult, InferenceUsage as LlamaCppInferenceUsage,
    LlamaCppError, LlamaCppOptions, LlamaCppProvider,
};

#[cfg(feature = "audio-tts")]
pub use blazen_audio_tts::{DynTtsProvider, TtsBackendHandle, TtsError, TtsModel, TtsOptions};

#[cfg(feature = "audio-tts-anytts")]
pub use blazen_audio_tts::AnyTtsBackend;

#[cfg(feature = "audio-tts-spark")]
pub use blazen_audio_tts::backends::spark::{SparkTtsBackend, SparkTtsConfig};

#[cfg(feature = "audio-music")]
pub use blazen_audio::{AudioFormat as AudioMusicFormat, GeneratedAudio as MusicGeneratedAudio};
#[cfg(feature = "audio-music-audiogen")]
pub use blazen_audio_music::backends::audiogen::{
    AUDIOGEN_FRAME_RATE, AUDIOGEN_MAX_DURATION_HARD_LIMIT, AUDIOGEN_SAMPLE_RATE, AudioGenBackend,
    AudioGenConfig,
};
#[cfg(feature = "audio-music-musicgen")]
pub use blazen_audio_music::backends::musicgen::{
    MUSICGEN_MAX_DURATION_HARD_LIMIT, MusicgenBackend, MusicgenConfig, MusicgenVariant,
};
#[cfg(feature = "audio-music-stable-audio")]
pub use blazen_audio_music::backends::stable_audio::{
    StableAudioBackend, StableAudioConfig, StableAudioVariant,
};
#[cfg(feature = "audio-music")]
pub use blazen_audio_music::{
    DynMusicProvider, MusicBackend, MusicBackendHandle, MusicChunk, MusicError,
};

#[cfg(feature = "audio-codec")]
pub use blazen_audio_codec::{CodecBackendHandle, CodecError, DynCodecProvider};

#[cfg(feature = "audio-vc-rvc")]
pub use blazen_audio_vc::RvcBackend;
#[cfg(feature = "audio-vc")]
pub use blazen_audio_vc::{TargetVoice, VcError, VoiceConversionBackend};
