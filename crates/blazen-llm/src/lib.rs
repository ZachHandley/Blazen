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
//! use blazen_llm::{CompletionModel, CompletionRequest, ChatMessage};
//! use blazen_llm::providers::openai::OpenAiProvider;
//!
//! # async fn example() -> Result<(), blazen_llm::BlazenError> {
//! let model = OpenAiProvider::new("sk-...");
//! let request = CompletionRequest::new(vec![
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
pub mod cache;
pub mod chat_window;
pub mod compute;
pub mod error;
pub mod events;
pub mod fallback;
pub mod http;
#[cfg(all(feature = "reqwest", not(target_arch = "wasm32")))]
mod http_reqwest;
pub(crate) mod sleep;
#[cfg(all(feature = "reqwest", not(target_arch = "wasm32")))]
pub use http_reqwest::ReqwestHttpClient;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
mod http_fetch;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub use http_fetch::FetchHttpClient;

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

pub mod media;
pub mod middleware;
pub mod pricing;
pub mod providers;
pub mod retry;
pub mod tokens;
pub mod traits;
pub mod types;

// Re-export primary types at crate root for ergonomic imports.
pub use agent::{AgentConfig, AgentEvent, AgentResult, run_agent, run_agent_with_callback};
pub use cache::{CacheConfig, CacheStrategy, CachedCompletionModel};
pub use chat_window::ChatWindow;
pub use compute::{
    // Audio
    AudioGeneration,
    AudioResult,
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
#[allow(deprecated)]
pub use error::LlmError;
pub use error::{BlazenError, CompletionErrorKind, ComputeErrorKind, MediaErrorKind};
pub use events::{StreamChunkEvent, StreamCompleteEvent};
pub use fallback::FallbackModel;
pub use media::{
    Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput, MediaType,
};
pub use middleware::{CacheMiddleware, Middleware, MiddlewareStack, RetryMiddleware};
pub use retry::{RetryCompletionModel, RetryConfig};
#[cfg(feature = "tiktoken")]
pub use tokens::TiktokenCounter;
pub use tokens::{EstimateCounter, TokenCounter};
pub use traits::{
    CompletionModel, EmbeddingModel, ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry,
    ProviderCapabilities, ProviderInfo, StructuredOutput, Tool,
};
pub use types::{
    ChatMessage, CompletionRequest, CompletionResponse, ContentPart, EmbeddingResponse,
    FileContent, ImageContent, ImageSource, MessageContent, RequestTiming, Role, StreamChunk,
    StructuredResponse, TokenUsage, ToolCall, ToolDefinition,
};
