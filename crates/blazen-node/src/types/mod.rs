//! Shared type definitions for the Node.js bindings.

pub mod completion;
pub mod media;
pub mod message;
pub mod tool;
pub mod usage;

// Re-export all public types for convenient access.
pub(crate) use completion::build_response;
pub use completion::{JsCompletionOptions, JsCompletionResponse};
pub use media::{
    JsGenerated3DModel, JsGeneratedAudio, JsGeneratedImage, JsGeneratedVideo, JsMediaOutput,
    JsMediaTypeMap, media_types,
};
pub use message::{
    ChatMessageOptions, JsChatMessage, JsContentPart, JsImageContent, JsImageSource, JsRole,
};
pub use tool::{JsToolCall, JsToolDefinition};
pub use usage::{JsRequestTiming, JsTokenUsage};
