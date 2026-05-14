//! Shared type definitions for the Node.js bindings.

pub mod abc;
pub mod artifact;
pub mod chat_window;
pub mod citation;
pub mod completion;
pub mod completion_request;
#[cfg(feature = "embed")]
pub mod embed_facade;
pub mod embedding;
pub mod events;
pub mod events_wave;
pub mod finish_reason;
pub mod http_client;
pub mod http_client_config;
pub mod media;
pub mod memory;
pub mod message;
pub mod pricing;
pub mod prompts;
pub mod provider_info;
pub mod reasoning;
pub mod response_format;
pub mod retry_stack;
pub mod stream;
pub mod tokens;
pub mod tool;
pub mod tool_output;
pub mod usage;
pub mod usage_recording;

// Re-export all public types for convenient access.
pub use crate::generated::{
    JsAudioResult, JsBackgroundRemovalRequest, JsComputeResult, JsGenerated3DModel,
    JsGeneratedAudio, JsGeneratedImage, JsGeneratedVideo, JsImageResult, JsJobHandle,
    JsMediaOutput, JsThreeDResult, JsTranscriptionResult, JsTranscriptionSegment, JsVideoResult,
};
pub use abc::{JsLocalModel, JsModelRegistry, JsStructuredOutput, JsTool};
pub use artifact::JsArtifact;
pub use chat_window::JsChatWindow;
pub use citation::{CitationOptions, JsCitation, JsCitationClass};
pub(crate) use completion::build_response;
pub use completion::{JsCompletionOptions, JsCompletionResponse};
pub use completion_request::{
    JsCompletionRequest, JsFileContent, JsMessageContent, JsStructuredResponse,
};
#[cfg(feature = "embed")]
pub use embed_facade::JsEmbedResponse;
pub use embedding::{JsEmbeddingModel, JsEmbeddingResponse};
pub use events::{
    JsAgentConfig, JsAgentEvent, JsStartEventClass, JsStopEventClass, JsStreamChunkEvent,
    JsStreamCompleteEvent, StartEventOptions, StopEventOptions,
};
pub use events_wave::{
    JsModality, JsProgressEvent, JsProgressKind, JsUsageEvent, add_usage_to_token_usage,
    new_usage_event,
};
pub use finish_reason::JsFinishReason;
pub use http_client::{JsHttpClient, JsHttpRequest, JsHttpResponse};
pub use http_client_config::{
    JsHttpClientConfig, default_http_client_config, unlimited_http_client_config,
};
pub use media::{Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput};
#[cfg(target_os = "wasi")]
pub use memory::JsUpstashBackend;
pub use memory::{
    JsAddEntry, JsInMemoryBackend, JsMemory, JsMemoryBackend, JsMemoryEntry, JsMemoryResult,
    JsRetryMemoryBackend,
};
#[cfg(not(target_os = "wasi"))]
pub use memory::{JsJsonlBackend, JsValkeyBackend};
pub use message::{
    ChatMessageOptions, JsChatMessage, JsContentPart, JsImageContent, JsImageSource, JsRole,
};
pub use pricing::{
    JsModelPricing, compute_audio_cost, compute_image_cost, compute_video_cost, lookup_pricing,
    register_pricing,
};
pub use prompts::{JsPromptFile, JsPromptRegistry, JsPromptTemplate, JsTemplateRole};
pub use provider_info::{
    JsModelCapabilities, JsModelInfo, JsPricingEntry, JsProviderCapabilities, JsProviderConfig,
    JsProviderId,
};
pub use reasoning::{JsReasoningTrace, JsReasoningTraceClass, ReasoningTraceOptions};
pub use response_format::JsResponseFormat;
pub use retry_stack::{JsRetryStack, new_retry_stack, resolve_retry_stack};
pub use stream::JsStreamChunk;
pub(crate) use stream::build_stream_chunk;
#[cfg(feature = "tiktoken")]
pub use tokens::JsTiktokenCounter;
pub use tokens::{JsEstimateCounter, JsTokenCounter, count_message_tokens, estimate_tokens};
pub use tool::{
    JsToolCallClass, JsToolDefinitionClass, ToolCall, ToolCallOptions, ToolDefinition,
    ToolDefinitionOptions,
};
pub use tool_output::{LlmPayload, ToolOutput};
pub use usage::{
    JsRequestTimingClass, JsTokenUsageClass, RequestTiming, RequestTimingOptions, TokenUsage,
    TokenUsageOptions,
};
pub use usage_recording::{
    JsNoopUsageEmitter, JsUsageEmitter, JsUsageRecordingCompletionModel,
    JsUsageRecordingEmbeddingModel,
};
