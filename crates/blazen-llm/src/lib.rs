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
//! | `openai` | Enables [`providers::openai::OpenAiProvider`] and [`providers::openai_compat::OpenAiCompatProvider`] (covers `OpenRouter`, Groq, Together, Mistral, `DeepSeek`, Fireworks, Perplexity, xAI, Cohere, Bedrock) |
//! | `anthropic` | Enables [`providers::anthropic::AnthropicProvider`] |
//! | `gemini` | Enables [`providers::gemini::GeminiProvider`] |
//! | `fal` | Enables [`providers::fal::FalProvider`] |
//! | `azure` | Enables [`providers::azure::AzureOpenAiProvider`] |
//! | `all-providers` | Enables all provider implementations |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use blazen_llm::{CompletionModel, CompletionRequest, ChatMessage};
//! # #[cfg(feature = "openai")]
//! use blazen_llm::providers::openai::OpenAiProvider;
//!
//! # #[cfg(feature = "openai")]
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
//! # #[cfg(feature = "openai")]
//! use blazen_llm::providers::openai_compat::OpenAiCompatProvider;
//!
//! # #[cfg(feature = "openai")]
//! # fn example() {
//! // Groq (fast inference)
//! let groq = OpenAiCompatProvider::groq("gsk-...");
//!
//! // OpenRouter (400+ models)
//! let openrouter = OpenAiCompatProvider::openrouter("sk-or-...");
//!
//! // Together AI
//! let together = OpenAiCompatProvider::together("...");
//! # }
//! ```

pub mod agent;
pub mod compute;
pub mod error;
pub mod events;
pub mod media;
pub mod providers;
pub mod traits;
pub mod types;

// Re-export primary types at crate root for ergonomic imports.
pub use agent::{AgentConfig, AgentEvent, AgentResult, run_agent, run_agent_with_callback};
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
pub use media::{
    Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput, MediaType,
};
pub use traits::{
    CompletionModel, EmbeddingModel, ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry,
    StructuredOutput, Tool,
};
pub use types::{
    ChatMessage, CompletionRequest, CompletionResponse, ContentPart, EmbeddingResponse,
    FileContent, ImageContent, ImageSource, MessageContent, RequestTiming, Role, StreamChunk,
    StructuredResponse, TokenUsage, ToolCall, ToolDefinition,
};
