//! Shared type definitions for the Node.js bindings.

pub mod artifact;
pub mod citation;
pub mod completion;
pub mod embedding;
pub mod finish_reason;
pub mod media;
pub mod memory;
pub mod message;
pub mod reasoning;
pub mod response_format;
pub mod stream;
pub mod tokens;
pub mod tool;
pub mod usage;

// Re-export all public types for convenient access.
pub use artifact::JsArtifact;
pub use citation::JsCitation;
pub(crate) use completion::build_response;
pub use completion::{JsCompletionOptions, JsCompletionResponse};
pub use embedding::{JsEmbeddingModel, JsEmbeddingResponse};
pub use finish_reason::JsFinishReason;
pub use media::{
    JsGenerated3DModel, JsGeneratedAudio, JsGeneratedImage, JsGeneratedVideo, JsMediaOutput,
    JsMediaTypeMap, media_types,
};
pub use memory::{
    JsAddEntry, JsInMemoryBackend, JsJsonlBackend, JsMemory, JsMemoryEntry, JsMemoryResult,
    JsValkeyBackend,
};
pub use message::{
    ChatMessageOptions, JsChatMessage, JsContentPart, JsImageContent, JsImageSource, JsRole,
};
pub use reasoning::JsReasoningTrace;
pub use response_format::JsResponseFormat;
pub use stream::JsStreamChunk;
pub(crate) use stream::build_stream_chunk;
pub use tokens::{count_message_tokens, estimate_tokens};
pub use tool::{JsToolCall, JsToolDefinition};
pub use usage::{JsRequestTiming, JsTokenUsage};
