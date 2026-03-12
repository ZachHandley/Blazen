//! # `ZAgents` LLM
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
//! use zagents_llm::{CompletionModel, CompletionRequest, ChatMessage};
//! # #[cfg(feature = "openai")]
//! use zagents_llm::providers::openai::OpenAiProvider;
//!
//! # #[cfg(feature = "openai")]
//! # async fn example() -> Result<(), zagents_llm::LlmError> {
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
//! use zagents_llm::providers::openai_compat::OpenAiCompatProvider;
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

pub mod error;
pub mod events;
pub mod providers;
pub mod traits;
pub mod types;

// Re-export primary types at crate root for ergonomic imports.
pub use error::LlmError;
pub use events::{StreamChunkEvent, StreamCompleteEvent};
pub use traits::{
    CompletionModel, EmbeddingModel, ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry,
    StructuredOutput, Tool,
};
pub use types::{
    ChatMessage, CompletionRequest, CompletionResponse, MessageContent, Role, StreamChunk,
    TokenUsage, ToolCall, ToolDefinition,
};
