//! Shared Python-facing types for LLM messages, completions, tools, usage, and media.

pub mod abc;
pub mod artifact;
pub mod chat_window;
pub mod citation;
pub mod completion;
pub mod completion_request;
pub mod embedding;
pub mod finish_reason;
pub mod http;
pub mod http_client_handle;
pub mod media;
pub mod memory;
pub mod memory_entry;
pub mod message;
pub mod message_content;
pub mod pricing;
pub mod pricing_entry;
pub mod prompts;
pub mod provider_id;
pub mod provider_info;
pub mod reasoning;
pub mod request_timing;
pub mod response_format;
pub mod stream;
pub mod tokens;
pub mod tool;
pub mod transcription;
pub mod typed_tool;
pub mod usage;
pub mod usage_recording;

pub use abc::{
    PyHostDispatchAbc, PyImageModel, PyLocalModel, PyModelRegistry, PyStructuredOutput,
    PyStructuredResponse, PyTool,
};
pub use artifact::PyArtifact;
pub use chat_window::PyChatWindow;
pub use citation::PyCitation;
pub use completion::PyCompletionResponse;
pub use completion_request::PyCompletionRequest;
pub use embedding::{PyEmbeddingModel, PyEmbeddingResponse};
pub use finish_reason::PyFinishReason;
pub use http::{PyHttpClient, PyHttpClientConfig};
pub use http_client_handle::PyHttpClientHandle;
pub use media::{
    PyGenerated3DModel, PyGeneratedAudio, PyGeneratedImage, PyGeneratedVideo, PyMediaOutput,
    PyMediaType,
};
pub use memory::{
    PyInMemoryBackend, PyJsonlBackend, PyMemory, PyMemoryBackend, PyMemoryResult, PyMemoryStore,
    PyRetryMemoryBackend, PyStoredEntry, PyValkeyBackend, compute_elid_similarity,
    compute_embedding_simhash_similarity, compute_text_simhash_similarity, simhash_from_hex,
    simhash_to_hex,
};
pub use memory_entry::PyMemoryEntry;
pub use message::{PyChatMessage, PyContentPart, PyRole};
pub use message_content::{
    PyAudioContent, PyFileContent, PyImageContent, PyImageSource, PyMessageContent, PyVideoContent,
};
pub use pricing::PyModelPricing;
pub use pricing_entry::PyPricingEntry;
pub use prompts::{PyPromptFile, PyPromptRegistry, PyPromptTemplate, PyTemplateRole};
pub use provider_id::PyProviderId;
pub use provider_info::{
    PyModelCapabilities, PyProviderCapabilities, PyProviderConfig, PyProviderInfo,
};
pub use reasoning::PyReasoningTrace;
pub use request_timing::PyRequestTiming;
pub use response_format::PyResponseFormat;
pub use stream::{PyStreamChunk, PyStreamChunkEvent, PyStreamCompleteEvent, StreamChunk};
#[cfg(feature = "tiktoken")]
pub use tokens::PyTiktokenCounter;
pub use tokens::{PyEstimateCounter, PyTokenCounter, count_message_tokens, estimate_tokens};
pub use tool::{PyLlmPayload, PyToolCall, PyToolDefinition, PyToolOutput};
pub use transcription::PyTranscription;
pub use usage::{PyTokenUsage, RequestTiming};
pub use usage_recording::{
    PyNoopUsageEmitter, PyUsageEmitter, PyUsageRecordingCompletionModel,
    PyUsageRecordingEmbeddingModel,
};
