//! Message-related types: roles, content variants, and chat messages.

use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

// ---------------------------------------------------------------------------
// Role & message content
// ---------------------------------------------------------------------------

/// The role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
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

/// How a piece of media is provided.
///
/// Used by [`ImageContent`], [`AudioContent`], [`VideoContent`], and
/// [`FileContent`] — all media kinds share this URL-or-base64 envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Media provided via URL.
    Url { url: String },
    /// Media provided as base64-encoded data.
    Base64 { data: String },
}

/// Generic media source alias used by audio, video, and document content.
///
/// `MediaSource` is a re-export of [`ImageSource`] under a more inclusive
/// name. Adopt `MediaSource` in new code that handles non-image modalities;
/// existing image code can keep using `ImageSource` indefinitely — they are
/// the same type.
pub type MediaSource = ImageSource;

/// Image content for multimodal messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ImageContent {
    /// The source of the image data.
    pub source: ImageSource,
    /// The MIME type of the image (e.g. `"image/jpeg"`, `"image/png"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// File/document content (PDF, generic file, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct FileContent {
    /// The source of the file data.
    pub source: MediaSource,
    /// The MIME type of the file (e.g. `"application/pdf"`).
    pub media_type: String,
    /// An optional filename for display purposes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Audio content for chat-message input (multimodal models, ASR feeds).
///
/// Audio chat input is supported natively by Gemini (via `inlineData`/`fileData`)
/// and `OpenAI`'s `gpt-4o-audio-preview` (via `input_audio` blocks). Providers
/// that do not support audio input drop the content with a warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct AudioContent {
    /// The source of the audio data.
    pub source: MediaSource,
    /// MIME type, e.g. `"audio/mp3"`, `"audio/wav"`, `"audio/ogg"`, `"audio/flac"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Optional duration in seconds, populated when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
}

impl AudioContent {
    /// Build an audio content from a public URL.
    #[must_use]
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: MediaSource::Url { url: url.into() },
            media_type: None,
            duration_seconds: None,
        }
    }

    /// Build an audio content from base64-encoded data with an explicit MIME.
    #[must_use]
    pub fn from_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            source: MediaSource::Base64 { data: data.into() },
            media_type: Some(media_type.into()),
            duration_seconds: None,
        }
    }

    /// Builder: attach a known duration in seconds.
    #[must_use]
    pub fn with_duration(mut self, seconds: f32) -> Self {
        self.duration_seconds = Some(seconds);
        self
    }
}

/// Video content for chat-message input (multimodal models).
///
/// Video chat input is supported natively by Gemini. Providers that do not
/// support video input drop the content with a warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct VideoContent {
    /// The source of the video data.
    pub source: MediaSource,
    /// MIME type, e.g. `"video/mp4"`, `"video/mov"`, `"video/webm"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Optional duration in seconds, populated when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
}

impl VideoContent {
    /// Build a video content from a public URL.
    #[must_use]
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: MediaSource::Url { url: url.into() },
            media_type: None,
            duration_seconds: None,
        }
    }

    /// Build a video content from base64-encoded data with an explicit MIME.
    #[must_use]
    pub fn from_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            source: MediaSource::Base64 { data: data.into() },
            media_type: Some(media_type.into()),
            duration_seconds: None,
        }
    }

    /// Builder: attach a known duration in seconds.
    #[must_use]
    pub fn with_duration(mut self, seconds: f32) -> Self {
        self.duration_seconds = Some(seconds);
        self
    }
}

/// A single part in a multi-part message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text segment.
    Text { text: String },
    /// An image segment.
    Image(ImageContent),
    /// A file/document segment.
    File(FileContent),
    /// An audio segment.
    Audio(AudioContent),
    /// A video segment.
    Video(VideoContent),
}

impl ContentPart {
    /// Build a text content part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Build an image content part from a URL.
    #[must_use]
    pub fn image_url(url: impl Into<String>, media_type: Option<String>) -> Self {
        Self::Image(ImageContent {
            source: ImageSource::Url { url: url.into() },
            media_type,
        })
    }

    /// Build an image content part from base64-encoded data.
    #[must_use]
    pub fn image_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Image(ImageContent {
            source: ImageSource::Base64 { data: data.into() },
            media_type: Some(media_type.into()),
        })
    }

    /// Build an audio content part from a URL.
    #[must_use]
    pub fn audio_url(url: impl Into<String>) -> Self {
        Self::Audio(AudioContent::from_url(url))
    }

    /// Build an audio content part from base64-encoded data.
    #[must_use]
    pub fn audio_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Audio(AudioContent::from_base64(data, media_type))
    }

    /// Build a video content part from a URL.
    #[must_use]
    pub fn video_url(url: impl Into<String>) -> Self {
        Self::Video(VideoContent::from_url(url))
    }

    /// Build a video content part from base64-encoded data.
    #[must_use]
    pub fn video_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Video(VideoContent::from_base64(data, media_type))
    }

    /// Build a file content part from a URL with an explicit MIME type.
    #[must_use]
    pub fn file_url(
        url: impl Into<String>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File(FileContent {
            source: MediaSource::Url { url: url.into() },
            media_type: media_type.into(),
            filename,
        })
    }

    /// Build a file content part from base64-encoded data.
    #[must_use]
    pub fn file_base64(
        data: impl Into<String>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File(FileContent {
            source: MediaSource::Base64 { data: data.into() },
            media_type: media_type.into(),
            filename,
        })
    }
}

/// The content payload of a [`ChatMessage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
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

    /// Borrow every image part inside this message.
    ///
    /// Walks both the top-level [`MessageContent::Image`] shorthand and any
    /// [`ContentPart::Image`] parts inside [`MessageContent::Parts`].
    #[must_use]
    pub fn image_parts(&self) -> Vec<&ImageContent> {
        match self {
            Self::Text(_) => Vec::new(),
            Self::Image(img) => vec![img],
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Image(img) => Some(img),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Borrow every audio part inside this message.
    #[must_use]
    pub fn audio_parts(&self) -> Vec<&AudioContent> {
        match self {
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Audio(a) => Some(a),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Borrow every video part inside this message.
    #[must_use]
    pub fn video_parts(&self) -> Vec<&VideoContent> {
        match self {
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Video(v) => Some(v),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Borrow every file part inside this message.
    #[must_use]
    pub fn file_parts(&self) -> Vec<&FileContent> {
        match self {
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::File(f) => Some(f),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Whether this message carries any image content.
    #[must_use]
    pub fn has_images(&self) -> bool {
        !self.image_parts().is_empty()
    }

    /// Whether this message carries any audio content.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        !self.audio_parts().is_empty()
    }

    /// Whether this message carries any video content.
    #[must_use]
    pub fn has_video(&self) -> bool {
        !self.video_parts().is_empty()
    }

    /// Whether this message carries any file/document content.
    #[must_use]
    pub fn has_files(&self) -> bool {
        !self.file_parts().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Chat message
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ChatMessage {
    /// Who produced this message.
    pub role: Role,
    /// The message payload.
    pub content: MessageContent,
    /// The ID of the tool call this message is a response to (for `Role::Tool` messages).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    /// The name of the tool/function that produced this result (for `Role::Tool` messages).
    ///
    /// Some providers (e.g. Gemini) require the function name alongside the tool
    /// result, rather than a synthetic call ID.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Tool calls requested by the assistant in this message.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    /// Create a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create an assistant message that includes tool calls.
    ///
    /// When the model responds with tool invocations, use this constructor to
    /// preserve the `tool_calls` array so that subsequent `tool` result messages
    /// can be matched by their `tool_call_id`.
    #[must_use]
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.map_or(MessageContent::Text(String::new()), MessageContent::Text),
            tool_call_id: None,
            name: None,
            tool_calls,
        }
    }

    /// Create a tool result message.
    #[must_use]
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create a tool result message with an associated tool call ID and function name.
    ///
    /// OpenAI-compatible APIs require each tool result message to reference the
    /// `tool_call_id` of the invocation it responds to.  Gemini requires the
    /// original function `name` in the `functionResponse` payload.
    #[must_use]
    pub fn tool_result(
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
            tool_call_id: Some(call_id.into()),
            name: Some(name.into()),
            tool_calls: Vec::new(),
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
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
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
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create a user message from an explicit list of content parts.
    #[must_use]
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(parts),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create a user message containing text and an audio clip from a URL.
    #[must_use]
    pub fn user_audio(prompt: impl Into<String>, audio_url: impl Into<String>) -> Self {
        Self::user_parts(vec![
            ContentPart::text(prompt),
            ContentPart::audio_url(audio_url),
        ])
    }

    /// Create a user message containing text and a base64-encoded audio clip.
    #[must_use]
    pub fn user_audio_base64(
        prompt: impl Into<String>,
        data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self::user_parts(vec![
            ContentPart::text(prompt),
            ContentPart::audio_base64(data, media_type),
        ])
    }

    /// Create a user message containing text and a video from a URL.
    #[must_use]
    pub fn user_video(prompt: impl Into<String>, video_url: impl Into<String>) -> Self {
        Self::user_parts(vec![
            ContentPart::text(prompt),
            ContentPart::video_url(video_url),
        ])
    }

    /// Create a user message containing text and a base64-encoded video.
    #[must_use]
    pub fn user_video_base64(
        prompt: impl Into<String>,
        data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self::user_parts(vec![
            ContentPart::text(prompt),
            ContentPart::video_base64(data, media_type),
        ])
    }
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

    // -----------------------------------------------------------------------
    // Audio / Video content variants
    // -----------------------------------------------------------------------

    #[test]
    fn audio_content_from_url() {
        let audio = AudioContent::from_url("https://example.com/clip.mp3").with_duration(12.5);
        assert!(
            matches!(&audio.source, MediaSource::Url { url } if url == "https://example.com/clip.mp3")
        );
        assert_eq!(audio.duration_seconds, Some(12.5));
    }

    #[test]
    fn audio_content_from_base64() {
        let audio = AudioContent::from_base64("YWJjMTIz", "audio/wav");
        assert!(matches!(&audio.source, MediaSource::Base64 { data } if data == "YWJjMTIz"));
        assert_eq!(audio.media_type.as_deref(), Some("audio/wav"));
    }

    #[test]
    fn video_content_from_url() {
        let video = VideoContent::from_url("https://example.com/clip.mp4");
        assert!(
            matches!(&video.source, MediaSource::Url { url } if url == "https://example.com/clip.mp4")
        );
    }

    #[test]
    fn content_part_constructors() {
        let text = ContentPart::text("hi");
        assert!(matches!(&text, ContentPart::Text { text } if text == "hi"));

        let img = ContentPart::image_url("https://i.com/a.png", Some("image/png".into()));
        assert!(matches!(&img, ContentPart::Image(_)));

        let aud = ContentPart::audio_url("https://a.com/c.mp3");
        assert!(matches!(&aud, ContentPart::Audio(_)));

        let vid = ContentPart::video_url("https://v.com/c.mp4");
        assert!(matches!(&vid, ContentPart::Video(_)));

        let file = ContentPart::file_url("https://f.com/d.pdf", "application/pdf", None);
        assert!(matches!(&file, ContentPart::File(_)));
    }

    #[test]
    fn user_audio_constructor() {
        let msg = ChatMessage::user_audio("Transcribe this", "https://a.com/c.mp3");
        assert_eq!(msg.role, Role::User);
        let parts = msg.content.as_parts();
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "Transcribe this"));
        assert!(matches!(&parts[1], ContentPart::Audio(_)));
    }

    #[test]
    fn user_video_constructor() {
        let msg = ChatMessage::user_video("Describe this", "https://v.com/c.mp4");
        assert_eq!(msg.role, Role::User);
        let parts = msg.content.as_parts();
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[1], ContentPart::Video(_)));
    }

    #[test]
    fn message_content_helpers_image() {
        let img = ImageContent {
            source: ImageSource::Url { url: "u".into() },
            media_type: None,
        };
        let content = MessageContent::Image(img.clone());
        assert!(content.has_images());
        assert_eq!(content.image_parts().len(), 1);
        assert!(!content.has_audio());
        assert!(!content.has_video());
    }

    #[test]
    fn message_content_helpers_parts_audio_video() {
        let content = MessageContent::Parts(vec![
            ContentPart::text("describe"),
            ContentPart::audio_url("https://a.com/c.mp3"),
            ContentPart::video_url("https://v.com/c.mp4"),
        ]);
        assert!(content.has_audio());
        assert!(content.has_video());
        assert!(!content.has_images());
        assert_eq!(content.audio_parts().len(), 1);
        assert_eq!(content.video_parts().len(), 1);
    }

    #[test]
    fn audio_part_serialization_roundtrip() {
        let part = ContentPart::audio_url("https://a.com/c.mp3");
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"type\":\"audio\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(back, part);
    }

    #[test]
    fn video_part_serialization_roundtrip() {
        let part = ContentPart::video_base64("YWJj", "video/mp4");
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"type\":\"video\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(back, part);
    }
}
