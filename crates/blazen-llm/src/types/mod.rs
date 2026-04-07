//! Core data types for LLM request/response modelling.
//!
//! These types are provider-agnostic. Each provider implementation is
//! responsible for converting between these types and its wire format.

mod completion;
mod message;
pub mod provider_options;
mod tool;
mod usage;

pub use completion::{
    Artifact, Citation, CompletionRequest, CompletionResponse, EmbeddingResponse, FinishReason,
    ReasoningTrace, ResponseFormat, StreamChunk, StructuredResponse,
};
pub use message::{
    AudioContent, ChatMessage, ContentPart, FileContent, ImageContent, ImageSource, MediaSource,
    MessageContent, Role, VideoContent,
};
pub use tool::{ToolCall, ToolDefinition};
pub use usage::{RequestTiming, TokenUsage};
