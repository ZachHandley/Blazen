//! Completion request/response types, structured output, embeddings, and streaming.

use serde::{Deserialize, Serialize};

use crate::media::{GeneratedAudio, GeneratedImage, GeneratedVideo};

use super::message::ChatMessage;
use super::tool::{ToolCall, ToolDefinition};
use super::usage::{RequestTiming, TokenUsage};

// ---------------------------------------------------------------------------
// Reasoning trace
// ---------------------------------------------------------------------------

/// Chain-of-thought / extended-thinking trace from a model that exposes one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ReasoningTrace {
    /// Plain-text rendering of the reasoning content.
    pub text: String,
    /// Provider-specific signature/redaction handle, if any (Anthropic).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Whether the trace was redacted by the provider.
    pub redacted: bool,
    /// Reasoning effort level if the provider exposes one ("low"/"medium"/"high"/"max"/...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

/// A web/document citation backing a model statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct Citation {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Byte offsets in the response text that this citation backs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
    /// Optional document id (for retrieval-augmented citations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    /// Provider-specific extra fields preserved as JSON.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A typed artifact extracted from or returned by a model.
///
/// SVG / code blocks / markdown documents / mermaid diagrams / latex / html
/// can be returned inline as text by an LLM. The artifact surface lets
/// providers (or post-processors) lift them into typed values that callers
/// can dispatch on without re-parsing the assistant content string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Artifact {
    Svg {
        content: String,
        title: Option<String>,
    },
    CodeBlock {
        language: Option<String>,
        content: String,
        filename: Option<String>,
    },
    Markdown {
        content: String,
    },
    Mermaid {
        content: String,
    },
    Html {
        content: String,
    },
    Latex {
        content: String,
    },
    Json {
        content: serde_json::Value,
    },
    /// Escape hatch: unknown artifact kind.
    Custom {
        #[serde(rename = "name")]
        kind: String,
        content: String,
        #[serde(default)]
        metadata: serde_json::Value,
    },
}

/// Normalized finish reason across providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Safety,
    EndTurn,
    StopSequence,
    MaxTokens,
    Error,
    /// Provider returned a value not in the canonical set.
    Other(String),
}

impl FinishReason {
    /// Map a provider-specific finish-reason string into the canonical enum.
    ///
    /// Recognizes both `OpenAI` (`"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`,
    /// `"function_call"`), Anthropic (`"end_turn"`, `"stop_sequence"`, `"max_tokens"`,
    /// `"tool_use"`), and Gemini (`"STOP"`, `"MAX_TOKENS"`, `"SAFETY"`, `"RECITATION"`)
    /// variants. Unknown values fall through to [`FinishReason::Other`].
    #[must_use]
    pub fn from_provider_string(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" | "tool_use" | "function_call" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            "safety" | "recitation" => Self::Safety,
            "end_turn" => Self::EndTurn,
            "stop_sequence" => Self::StopSequence,
            "max_tokens" => Self::MaxTokens,
            "error" => Self::Error,
            other => Self::Other(other.to_owned()),
        }
    }
}

/// Typed response-format hint passed to providers that support structured output.
///
/// The on-the-wire JSON shape (returned by `From<ResponseFormat> for serde_json::Value`)
/// matches `OpenAI`'s chat completions `response_format` field. The existing
/// `CompletionRequest::response_format: Option<serde_json::Value>` keeps
/// raw JSON for backwards compatibility; the typed enum is opt-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text — no structural constraint.
    Text,
    /// JSON object mode (any valid JSON object).
    JsonObject,
    /// JSON Schema mode with a named schema.
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default)]
        strict: bool,
    },
}

impl ResponseFormat {
    /// Build a [`ResponseFormat::JsonSchema`] from a Rust type that implements
    /// [`schemars::JsonSchema`]. The schema name is derived from the type name.
    #[must_use]
    pub fn json_schema<T: schemars::JsonSchema>() -> Self {
        let schema = schemars::schema_for!(T);
        Self::JsonSchema {
            name: std::any::type_name::<T>()
                .split("::")
                .last()
                .unwrap_or("Schema")
                .to_owned(),
            schema: serde_json::to_value(&schema).unwrap_or(serde_json::Value::Null),
            strict: true,
        }
    }
}

impl From<ResponseFormat> for serde_json::Value {
    fn from(rf: ResponseFormat) -> Self {
        serde_json::to_value(rf).unwrap_or(serde_json::Value::Null)
    }
}

// ---------------------------------------------------------------------------
// Completion request
// ---------------------------------------------------------------------------

/// A provider-agnostic request for a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct CompletionRequest {
    /// The conversation history.
    pub messages: Vec<ChatMessage>,
    /// Tools available for the model to invoke.
    pub tools: Vec<ToolDefinition>,
    /// Sampling temperature (0.0 = deterministic, 2.0 = very random).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// A JSON Schema that the model's output should conform to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    /// Override the provider's default model for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Output modalities to request (e.g., \["text"\], \["image", "text"\]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    /// Image generation configuration (model-specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<serde_json::Value>,
    /// Audio output configuration (voice, format, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct CompletionResponse {
    /// The text content of the assistant's reply, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool invocations requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Chain-of-thought / extended-thinking trace, if exposed by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningTrace>,
    pub citations: Vec<Citation>,
    pub artifacts: Vec<Artifact>,
    /// Token usage statistics, if provided by the API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// The model that produced this response.
    pub model: String,
    /// The reason the model stopped generating (e.g. "stop", "`tool_use`").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Estimated cost for this request in USD, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Request timing breakdown, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
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

impl CompletionResponse {
    /// Lazily map the raw `finish_reason` string into a normalized [`FinishReason`].
    ///
    /// Returns `None` if the response carries no finish reason.
    #[must_use]
    pub fn finish_reason_normalized(&self) -> Option<FinishReason> {
        self.finish_reason
            .as_deref()
            .map(FinishReason::from_provider_string)
    }
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
    /// Chain-of-thought / extended-thinking trace, if exposed by the provider.
    pub reasoning: Option<ReasoningTrace>,
    /// Citations backing the model's response.
    pub citations: Vec<Citation>,
    /// Typed artifacts extracted from or returned by the model.
    pub artifacts: Vec<Artifact>,
}

// ---------------------------------------------------------------------------
// Embedding response
// ---------------------------------------------------------------------------

/// Response from an embedding operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct StreamChunk {
    /// Incremental text content, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    /// Tool invocations completed in this chunk.
    pub tool_calls: Vec<ToolCall>,
    /// Present in the final chunk to indicate why generation stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Reasoning text delta (Anthropic thinking, R1 `reasoning_content`, o-series).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_delta: Option<String>,
    /// Citations completed in this chunk.
    pub citations: Vec<Citation>,
    /// Artifacts completed in this chunk.
    pub artifacts: Vec<Artifact>,
}

#[cfg(test)]
mod finish_reason_tests {
    use super::*;

    #[test]
    fn from_provider_string_handles_anthropic_end_turn() {
        assert_eq!(
            FinishReason::from_provider_string("end_turn"),
            FinishReason::EndTurn
        );
    }

    #[test]
    fn from_provider_string_handles_gemini_uppercase_stop() {
        assert_eq!(
            FinishReason::from_provider_string("STOP"),
            FinishReason::Stop
        );
    }

    #[test]
    fn from_provider_string_normalizes_tool_call_aliases() {
        assert_eq!(
            FinishReason::from_provider_string("tool_calls"),
            FinishReason::ToolCalls
        );
        assert_eq!(
            FinishReason::from_provider_string("tool_use"),
            FinishReason::ToolCalls
        );
        assert_eq!(
            FinishReason::from_provider_string("function_call"),
            FinishReason::ToolCalls
        );
    }

    #[test]
    fn unknown_value_falls_through_to_other() {
        let result = FinishReason::from_provider_string("xyz_unknown");
        assert!(matches!(result, FinishReason::Other(s) if s == "xyz_unknown"));
    }
}

#[cfg(test)]
mod response_format_tests {
    use super::*;

    #[test]
    fn text_serializes_with_type_tag() {
        let rf = ResponseFormat::Text;
        let v: serde_json::Value = rf.into();
        assert_eq!(v["type"], "text");
    }

    #[test]
    fn json_object_serializes_with_type_tag() {
        let rf = ResponseFormat::JsonObject;
        let v: serde_json::Value = rf.into();
        assert_eq!(v["type"], "json_object");
    }

    #[test]
    fn json_schema_round_trip() {
        let rf = ResponseFormat::JsonSchema {
            name: "MyType".into(),
            schema: serde_json::json!({"type":"object"}),
            strict: true,
        };
        let v: serde_json::Value = rf.into();
        assert_eq!(v["type"], "json_schema");
        assert_eq!(v["name"], "MyType");
        assert_eq!(v["strict"], true);
    }
}
