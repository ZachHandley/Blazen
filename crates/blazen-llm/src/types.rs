//! Core data types for LLM request/response modelling.
//!
//! These types are provider-agnostic. Each provider implementation is
//! responsible for converting between these types and its wire format.

use serde::{Deserialize, Serialize};

use crate::media::{GeneratedAudio, GeneratedImage, GeneratedVideo};

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

/// How an image or file is provided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// An image provided via URL.
    Url { url: String },
    /// An image provided as base64-encoded data.
    Base64 { data: String },
}

/// Image content for multimodal messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageContent {
    /// The source of the image data.
    pub source: ImageSource,
    /// The MIME type of the image (e.g. `"image/jpeg"`, `"image/png"`).
    pub media_type: Option<String>,
}

/// File/document content (PDF, video, audio, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileContent {
    /// The source of the file data.
    pub source: ImageSource,
    /// The MIME type of the file (e.g. `"application/pdf"`, `"video/mp4"`).
    pub media_type: String,
    /// An optional filename for display purposes.
    pub filename: Option<String>,
}

/// A single part in a multi-part message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text segment.
    Text { text: String },
    /// An image segment.
    Image(ImageContent),
    /// A file/document segment.
    File(FileContent),
}

/// The content payload of a [`ChatMessage`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// A single image.
    Image(ImageContent),
    /// A multi-part message containing text, images, and/or files.
    Parts(Vec<ContentPart>),
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
            _ => None,
        }
    }

    /// Convert any variant into a `Vec<ContentPart>`.
    #[must_use]
    pub fn as_parts(&self) -> Vec<ContentPart> {
        match self {
            Self::Text(s) => vec![ContentPart::Text { text: s.clone() }],
            Self::Image(img) => vec![ContentPart::Image(img.clone())],
            Self::Parts(parts) => parts.clone(),
        }
    }

    /// Extract only text content, concatenating all text parts.
    ///
    /// Returns `None` if there is no text content at all.
    #[must_use]
    pub fn text_content(&self) -> Option<String> {
        match self {
            Self::Text(s) => Some(s.clone()),
            Self::Image(_) => None,
            Self::Parts(parts) => {
                let texts: Vec<&str> = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join("\n"))
                }
            }
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

    /// Create a user message containing text and an image from a URL.
    #[must_use]
    pub fn user_image_url(
        text: impl Into<String>,
        url: impl Into<String>,
        media_type: Option<&str>,
    ) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text { text: text.into() },
                ContentPart::Image(ImageContent {
                    source: ImageSource::Url { url: url.into() },
                    media_type: media_type.map(String::from),
                }),
            ]),
        }
    }

    /// Create a user message containing text and a base64-encoded image.
    #[must_use]
    pub fn user_image_base64(
        text: impl Into<String>,
        data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text { text: text.into() },
                ContentPart::Image(ImageContent {
                    source: ImageSource::Base64 { data: data.into() },
                    media_type: Some(media_type.into()),
                }),
            ]),
        }
    }

    /// Create a user message from an explicit list of content parts.
    #[must_use]
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(parts),
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
// Request timing
// ---------------------------------------------------------------------------

/// Timing metadata for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTiming {
    /// Time spent waiting in queue (ms), if applicable.
    pub queue_ms: Option<u64>,
    /// Time spent executing the request (ms).
    pub execution_ms: Option<u64>,
    /// Total wall-clock time from submit to response (ms).
    pub total_ms: Option<u64>,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MessageContent serialization roundtrips
    // -----------------------------------------------------------------------

    #[test]
    fn text_serialization_roundtrip() {
        let content = MessageContent::Text("hello".into());
        let json = serde_json::to_string(&content).unwrap();
        // Text should serialize as a plain JSON string.
        assert_eq!(json, "\"hello\"");

        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, content);
    }

    #[test]
    fn parts_serialization_roundtrip() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "hello".into(),
            },
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/img.png".into(),
                },
                media_type: Some("image/png".into()),
            }),
        ]);
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, content);
    }

    #[test]
    fn image_source_url_serde() {
        let source = ImageSource::Url {
            url: "https://example.com/img.jpg".into(),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"type\":\"url\""));
        let deserialized: ImageSource = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, source);
    }

    #[test]
    fn image_source_base64_serde() {
        let source = ImageSource::Base64 {
            data: "abc123".into(),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"type\":\"base64\""));
        let deserialized: ImageSource = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, source);
    }

    // -----------------------------------------------------------------------
    // as_parts() conversion
    // -----------------------------------------------------------------------

    #[test]
    fn as_parts_from_text() {
        let content = MessageContent::Text("hello".into());
        let parts = content.as_parts();
        assert_eq!(parts.len(), 1);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "hello"));
    }

    #[test]
    fn as_parts_from_image() {
        let content = MessageContent::Image(ImageContent {
            source: ImageSource::Url {
                url: "https://example.com/img.png".into(),
            },
            media_type: None,
        });
        let parts = content.as_parts();
        assert_eq!(parts.len(), 1);
        assert!(matches!(&parts[0], ContentPart::Image(_)));
    }

    #[test]
    fn as_parts_from_parts() {
        let original = vec![
            ContentPart::Text { text: "a".into() },
            ContentPart::Text { text: "b".into() },
        ];
        let content = MessageContent::Parts(original.clone());
        let parts = content.as_parts();
        assert_eq!(parts, original);
    }

    // -----------------------------------------------------------------------
    // text_content() extraction
    // -----------------------------------------------------------------------

    #[test]
    fn text_content_from_text() {
        let content = MessageContent::Text("hello".into());
        assert_eq!(content.text_content(), Some("hello".into()));
    }

    #[test]
    fn text_content_from_image() {
        let content = MessageContent::Image(ImageContent {
            source: ImageSource::Url {
                url: "https://example.com/img.png".into(),
            },
            media_type: None,
        });
        assert_eq!(content.text_content(), None);
    }

    #[test]
    fn text_content_from_parts_mixed() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "first".into(),
            },
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/img.png".into(),
                },
                media_type: None,
            }),
            ContentPart::Text {
                text: "second".into(),
            },
        ]);
        assert_eq!(content.text_content(), Some("first\nsecond".into()));
    }

    #[test]
    fn text_content_from_parts_no_text() {
        let content = MessageContent::Parts(vec![ContentPart::Image(ImageContent {
            source: ImageSource::Url {
                url: "https://example.com/img.png".into(),
            },
            media_type: None,
        })]);
        assert_eq!(content.text_content(), None);
    }

    // -----------------------------------------------------------------------
    // ChatMessage convenience constructors
    // -----------------------------------------------------------------------

    #[test]
    fn user_image_url_constructor() {
        let msg =
            ChatMessage::user_image_url("Describe", "https://img.com/a.png", Some("image/png"));
        assert_eq!(msg.role, Role::User);
        let parts = msg.content.as_parts();
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "Describe"));
        assert!(
            matches!(&parts[1], ContentPart::Image(img) if matches!(&img.source, ImageSource::Url { url } if url == "https://img.com/a.png"))
        );
    }

    #[test]
    fn user_image_base64_constructor() {
        let msg = ChatMessage::user_image_base64("What", "base64data", "image/jpeg");
        assert_eq!(msg.role, Role::User);
        let parts = msg.content.as_parts();
        assert_eq!(parts.len(), 2);
        assert!(
            matches!(&parts[1], ContentPart::Image(img) if img.media_type == Some("image/jpeg".into()))
        );
    }

    #[test]
    fn user_parts_constructor() {
        let msg = ChatMessage::user_parts(vec![ContentPart::Text { text: "hi".into() }]);
        assert_eq!(msg.role, Role::User);
        assert!(matches!(msg.content, MessageContent::Parts(ref p) if p.len() == 1));
    }

    // -----------------------------------------------------------------------
    // Backward compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn as_text_still_works_for_text() {
        let content = MessageContent::Text("hello".into());
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn as_text_none_for_non_text() {
        let content = MessageContent::Image(ImageContent {
            source: ImageSource::Url { url: "u".into() },
            media_type: None,
        });
        assert_eq!(content.as_text(), None);

        let content2 = MessageContent::Parts(vec![]);
        assert_eq!(content2.as_text(), None);
    }

    #[test]
    fn from_str_still_works() {
        let content: MessageContent = "hello".into();
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn from_string_still_works() {
        let content: MessageContent = String::from("hello").into();
        assert_eq!(content.as_text(), Some("hello"));
    }
}
