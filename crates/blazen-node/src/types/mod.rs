//! Shared type definitions for the Node.js bindings.

pub mod artifact;
pub mod chat_window;
pub mod citation;
pub mod completion;
pub mod embedding;
pub mod finish_reason;
pub mod media;
pub mod memory;
pub mod message;
pub mod pricing;
pub mod prompts;
pub mod reasoning;
pub mod response_format;
pub mod stream;
pub mod tokens;
pub mod tool;
pub mod tool_output;
pub mod usage;

// Re-export all public types for convenient access.
pub use crate::generated::{
    JsAudioResult, JsBackgroundRemovalRequest, JsComputeResult, JsGenerated3DModel,
    JsGeneratedAudio, JsGeneratedImage, JsGeneratedVideo, JsImageResult, JsJobHandle,
    JsMediaOutput, JsThreeDResult, JsTranscriptionResult, JsTranscriptionSegment, JsVideoResult,
};
pub use artifact::JsArtifact;
pub use chat_window::JsChatWindow;
pub use citation::JsCitation;
pub(crate) use completion::build_response;
pub use completion::{JsCompletionOptions, JsCompletionResponse};
pub use embedding::{JsEmbeddingModel, JsEmbeddingResponse};
pub use finish_reason::JsFinishReason;
pub use media::{Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput};
pub use memory::{
    JsAddEntry, JsInMemoryBackend, JsJsonlBackend, JsMemory, JsMemoryBackend, JsMemoryEntry,
    JsMemoryResult, JsValkeyBackend,
};
pub use message::{
    ChatMessageOptions, JsChatMessage, JsContentPart, JsImageContent, JsImageSource, JsRole,
};
pub use pricing::{JsModelPricing, lookup_pricing, register_pricing};
pub use prompts::{JsPromptRegistry, JsPromptTemplate};
pub use reasoning::JsReasoningTrace;
pub use response_format::JsResponseFormat;
pub use stream::JsStreamChunk;
pub(crate) use stream::build_stream_chunk;
pub use tokens::{count_message_tokens, estimate_tokens};
pub use tool::{ToolCall, ToolDefinition};
pub use tool_output::{LlmPayload, ToolOutput};
pub use usage::{RequestTiming, TokenUsage};
