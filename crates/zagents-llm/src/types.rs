//! Core data types for LLM request/response modelling.
//!
//! These types are provider-agnostic. Each provider implementation is
//! responsible for converting between these types and its wire format.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Role & message content
// ---------------------------------------------------------------------------

/// The role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A system-level instruction that guides assistant behaviour.
    System,
    /// A message from the human user.
    User,
    /// A message produced by the assistant / model.
    Assistant,
    /// The result of a tool invocation.
    Tool,
}

/// The content payload of a [`ChatMessage`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    // Future variants: Image, ToolResult, MultiPart, etc.
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl MessageContent {
    /// Return the text content, if this is a [`MessageContent::Text`].
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
        }
    }
}

// ---------------------------------------------------------------------------
// Chat message
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Who produced this message.
    pub role: Role,
    /// The message payload.
    pub content: MessageContent,
}

impl ChatMessage {
    /// Create a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a tool result message.
    #[must_use]
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool definitions and calls
// ---------------------------------------------------------------------------

/// Describes a tool that the model may invoke during a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The unique name of the tool.
    pub name: String,
    /// A human-readable description of what the tool does.
    pub description: String,
    /// A JSON Schema object describing the tool's input parameters.
    pub parameters: serde_json::Value,
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned identifier for this specific invocation.
    pub id: String,
    /// The name of the tool to invoke.
    pub name: String,
    /// The arguments to pass, as a JSON value.
    pub arguments: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Token usage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Number of tokens in the prompt / input.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion / output.
    pub completion_tokens: u32,
    /// Total tokens consumed (prompt + completion).
    pub total_tokens: u32,
}

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
    /// The reason the model stopped generating (e.g. "stop", "tool_use").
    pub finish_reason: Option<String>,
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
