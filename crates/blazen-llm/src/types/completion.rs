//! Completion request/response types, structured output, embeddings, and streaming.

use crate::media::{GeneratedAudio, GeneratedImage, GeneratedVideo};

use super::message::ChatMessage;
use super::tool::{ToolCall, ToolDefinition};
use super::usage::{RequestTiming, TokenUsage};

// ---------------------------------------------------------------------------
// Completion request
// ---------------------------------------------------------------------------

/// A provider-agnostic request for a chat completion.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// The conversation history.
    pub messages: Vec<ChatMessage>,
    /// Tools available for the model to invoke.
    pub tools: Vec<ToolDefinition>,
    /// Sampling temperature (0.0 = deterministic, 2.0 = very random).
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// A JSON Schema that the model's output should conform to.
    pub response_format: Option<serde_json::Value>,
    /// Override the provider's default model for this request.
    pub model: Option<String>,
    /// Output modalities to request (e.g., \["text"\], \["image", "text"\]).
    pub modalities: Option<Vec<String>>,
    /// Image generation configuration (model-specific).
    pub image_config: Option<serde_json::Value>,
    /// Audio output configuration (voice, format, etc.).
    pub audio_config: Option<serde_json::Value>,
}

impl CompletionRequest {
    /// Create a new request from a list of messages.
    #[must_use]
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            response_format: None,
            model: None,
            modalities: None,
            image_config: None,
            audio_config: None,
        }
    }

    /// Add tools that the model may invoke.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the nucleus sampling parameter.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set a JSON Schema for structured output.
    #[must_use]
    pub fn with_response_format(mut self, schema: serde_json::Value) -> Self {
        self.response_format = Some(schema);
        self
    }

    /// Override the default model for this request.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set output modalities (e.g., `["text"]`, `["image", "text"]`).
    #[must_use]
    pub fn with_modalities(mut self, modalities: Vec<String>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    /// Set image generation configuration (model-specific).
    #[must_use]
    pub fn with_image_config(mut self, config: serde_json::Value) -> Self {
        self.image_config = Some(config);
        self
    }

    /// Set audio output configuration (voice, format, etc.).
    #[must_use]
    pub fn with_audio_config(mut self, config: serde_json::Value) -> Self {
        self.audio_config = Some(config);
        self
    }
}

// ---------------------------------------------------------------------------
// Completion response
// ---------------------------------------------------------------------------

/// The result of a non-streaming chat completion.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The text content of the assistant's reply, if any.
    pub content: Option<String>,
    /// Tool invocations requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage statistics, if provided by the API.
    pub usage: Option<TokenUsage>,
    /// The model that produced this response.
    pub model: String,
    /// The reason the model stopped generating (e.g. "stop", "`tool_use`").
    pub finish_reason: Option<String>,
    /// Estimated cost for this request in USD, if available.
    pub cost: Option<f64>,
    /// Request timing breakdown, if available.
    pub timing: Option<RequestTiming>,
    /// Generated images (for multimodal models).
    pub images: Vec<GeneratedImage>,
    /// Generated audio (for TTS or multimodal models).
    pub audio: Vec<GeneratedAudio>,
    /// Generated videos (for video generation models).
    pub videos: Vec<GeneratedVideo>,
    /// Provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Structured response
// ---------------------------------------------------------------------------

/// Response from structured output extraction, preserving metadata.
#[derive(Debug, Clone)]
pub struct StructuredResponse<T> {
    /// The extracted structured data.
    pub data: T,
    /// Token usage statistics.
    pub usage: Option<TokenUsage>,
    /// The model that produced this response.
    pub model: String,
    /// Estimated cost in USD.
    pub cost: Option<f64>,
    /// Request timing.
    pub timing: Option<RequestTiming>,
    /// Provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Embedding response
// ---------------------------------------------------------------------------

/// Response from an embedding operation.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// The embedding vectors.
    pub embeddings: Vec<Vec<f32>>,
    /// The model used.
    pub model: String,
    /// Token usage statistics.
    pub usage: Option<TokenUsage>,
    /// Estimated cost in USD.
    pub cost: Option<f64>,
    /// Request timing.
    pub timing: Option<RequestTiming>,
    /// Provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Streaming types
// ---------------------------------------------------------------------------

/// A single chunk from a streaming completion response.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Incremental text content, if any.
    pub delta: Option<String>,
    /// Tool invocations completed in this chunk.
    pub tool_calls: Vec<ToolCall>,
    /// Present in the final chunk to indicate why generation stopped.
    pub finish_reason: Option<String>,
}
