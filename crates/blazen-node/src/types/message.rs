//! Chat message types and content part conversion helpers.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::types::{
    AudioContent, ChatMessage, ContentPart, ImageContent, ImageSource, MediaSource, MessageContent,
    Role, VideoContent,
};

// ---------------------------------------------------------------------------
// Role string enum
// ---------------------------------------------------------------------------

/// The role of a participant in a chat conversation.
#[napi(string_enum)]
pub enum JsRole {
    #[napi(value = "system")]
    System,
    #[napi(value = "user")]
    User,
    #[napi(value = "assistant")]
    Assistant,
    #[napi(value = "tool")]
    Tool,
}

// ---------------------------------------------------------------------------
// Content part types for multimodal messages
// ---------------------------------------------------------------------------

/// How an image is provided (URL or base64).
#[napi(object)]
pub struct JsImageSource {
    #[napi(js_name = "sourceType")]
    pub source_type: String,
    pub url: Option<String>,
    pub data: Option<String>,
}

/// Image content for multimodal messages.
#[napi(object)]
pub struct JsImageContent {
    pub source: JsImageSource,
    #[napi(js_name = "mediaType")]
    pub media_type: Option<String>,
}

/// Audio content for multimodal messages.
#[napi(object)]
pub struct JsAudioContent {
    pub source: JsImageSource,
    #[napi(js_name = "mediaType")]
    pub media_type: Option<String>,
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
}

/// Video content for multimodal messages.
#[napi(object)]
pub struct JsVideoContent {
    pub source: JsImageSource,
    #[napi(js_name = "mediaType")]
    pub media_type: Option<String>,
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
}

/// A single part in a multi-part message.
///
/// `partType` is one of `"text"`, `"image"`, `"audio"`, `"video"`. Set the
/// matching field (`text`, `image`, `audio`, `video`) accordingly.
#[napi(object)]
pub struct JsContentPart {
    #[napi(js_name = "partType")]
    pub part_type: String,
    pub text: Option<String>,
    pub image: Option<JsImageContent>,
    pub audio: Option<JsAudioContent>,
    pub video: Option<JsVideoContent>,
}

// ---------------------------------------------------------------------------
// ChatMessage class
// ---------------------------------------------------------------------------

/// Options for creating a `ChatMessage`.
#[napi(object)]
pub struct ChatMessageOptions {
    /// Role: "system", "user", "assistant", or "tool". Defaults to "user".
    pub role: Option<String>,
    /// Text content.
    pub content: Option<String>,
    /// Multimodal content parts (alternative to content).
    pub parts: Option<Vec<JsContentPart>>,
}

/// A single message in a chat conversation.
#[napi(js_name = "ChatMessage")]
pub struct JsChatMessage {
    pub(crate) inner: ChatMessage,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsChatMessage {
    /// Create a new chat message from an options object.
    ///
    /// `role` defaults to `"user"` if not provided.
    /// Supply either `content` (text) or `parts` (multimodal).
    #[napi(constructor)]
    pub fn new(options: ChatMessageOptions) -> Result<Self> {
        let role_str = options.role.as_deref().unwrap_or("user");
        let role = parse_role(role_str)?;

        let content = if let Some(parts) = options.parts {
            let rust_parts = convert_js_parts(parts)?;
            MessageContent::Parts(rust_parts)
        } else {
            MessageContent::Text(options.content.unwrap_or_default())
        };

        Ok(Self {
            inner: ChatMessage {
                role,
                content,
                tool_call_id: None,
                name: None,
                tool_calls: Vec::new(),
                tool_result: None,
            },
        })
    }

    /// Create a system message.
    #[napi(factory)]
    pub fn system(content: String) -> Self {
        Self {
            inner: ChatMessage::system(content),
        }
    }

    /// Create a user message.
    #[napi(factory)]
    pub fn user(content: String) -> Self {
        Self {
            inner: ChatMessage::user(content),
        }
    }

    /// Create an assistant message.
    #[napi(factory)]
    pub fn assistant(content: String) -> Self {
        Self {
            inner: ChatMessage::assistant(content),
        }
    }

    /// Create a tool result message.
    #[napi(factory)]
    pub fn tool(content: String) -> Self {
        Self {
            inner: ChatMessage::tool(content),
        }
    }

    /// Create a user message containing text and an image from a URL.
    #[napi(factory, js_name = "userImageUrl")]
    pub fn user_image_url(text: String, url: String, media_type: Option<String>) -> Self {
        Self {
            inner: ChatMessage::user_image_url(text, url, media_type.as_deref()),
        }
    }

    /// Create a user message containing text and a base64-encoded image.
    #[napi(factory, js_name = "userImageBase64")]
    pub fn user_image_base64(text: String, data: String, media_type: String) -> Self {
        Self {
            inner: ChatMessage::user_image_base64(text, data, media_type),
        }
    }

    /// Create a user message containing text and an audio clip from a URL.
    #[napi(factory, js_name = "userAudio")]
    pub fn user_audio(text: String, url: String) -> Self {
        Self {
            inner: ChatMessage::user_audio(text, url),
        }
    }

    /// Create a user message containing text and base64-encoded audio data.
    #[napi(factory, js_name = "userAudioBase64")]
    pub fn user_audio_base64(text: String, data: String, media_type: String) -> Self {
        Self {
            inner: ChatMessage::user_audio_base64(text, data, media_type),
        }
    }

    /// Create a user message containing text and a video clip from a URL.
    #[napi(factory, js_name = "userVideo")]
    pub fn user_video(text: String, url: String) -> Self {
        Self {
            inner: ChatMessage::user_video(text, url),
        }
    }

    /// Create a user message containing text and base64-encoded video data.
    #[napi(factory, js_name = "userVideoBase64")]
    pub fn user_video_base64(text: String, data: String, media_type: String) -> Self {
        Self {
            inner: ChatMessage::user_video_base64(text, data, media_type),
        }
    }

    /// Create a user message from an explicit list of content parts.
    #[napi(factory, js_name = "userParts")]
    pub fn user_parts(parts: Vec<JsContentPart>) -> Result<Self> {
        let content_parts = convert_js_parts(parts)?;
        Ok(Self {
            inner: ChatMessage::user_parts(content_parts),
        })
    }

    /// Create a tool result message whose payload is a list of multimodal
    /// content parts (text, image, audio, video).
    ///
    /// Useful for tools that return images or other media. The parts live in
    /// the message's content; the structured `toolResult` field is left
    /// unset.
    #[napi(factory, js_name = "toolResultParts")]
    pub fn tool_result_parts(
        call_id: String,
        name: String,
        parts: Vec<JsContentPart>,
    ) -> Result<Self> {
        let content_parts = convert_js_parts(parts)?;
        Ok(Self {
            inner: ChatMessage::tool_result_parts(call_id, name, content_parts),
        })
    }

    /// Return a snapshot of the tool-result payload as `{ data, llmOverride }`.
    ///
    /// Works for both structured tool results (where the payload lives in the
    /// `toolResult` sibling field) and plain string results (where the payload
    /// lives in `content`). Returns `null` for messages that are not tool
    /// results.
    #[napi(getter, js_name = "toolResultView")]
    pub fn tool_result_view(&self) -> Option<crate::types::tool_output::ToolOutput> {
        self.inner.tool_result_view().map(|(data, llm_override)| {
            crate::types::tool_output::ToolOutput {
                data,
                llm_override: llm_override.map(crate::types::tool_output::LlmPayload::from_rust),
            }
        })
    }

    /// The role of the message author.
    #[napi(getter)]
    pub fn role(&self) -> String {
        match self.inner.role {
            Role::System => "system".to_owned(),
            Role::User => "user".to_owned(),
            Role::Assistant => "assistant".to_owned(),
            Role::Tool => "tool".to_owned(),
        }
    }

    /// The text content of the message, if any.
    #[napi(getter)]
    pub fn content(&self) -> Option<String> {
        self.inner.content.text_content()
    }

    /// The structured tool-result payload, if this message is a tool result
    /// produced by a tool returning a non-string value or carrying an
    /// `llmOverride`. Plain-string tool results live in `content` instead and
    /// this returns `null`.
    #[napi(getter, js_name = "toolResult")]
    pub fn tool_result(&self) -> Option<crate::types::tool_output::ToolOutput> {
        self.inner
            .tool_result
            .as_ref()
            .map(crate::types::tool_output::ToolOutput::from_rust)
    }

    /// The tool call ID this message responds to, for `Role::Tool` messages.
    #[napi(getter, js_name = "toolCallId")]
    pub fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    /// The tool/function name that produced this result, for `Role::Tool` messages.
    #[napi(getter)]
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a role string into a [`Role`], returning a napi error on invalid input.
pub(crate) fn parse_role(role_str: &str) -> Result<Role> {
    match role_str {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Invalid role: \"{role_str}\". Must be one of: system, user, assistant, tool"),
        )),
    }
}

/// Convert a `Vec<JsContentPart>` into `Vec<ContentPart>`.
#[allow(clippy::too_many_lines)]
pub(crate) fn convert_js_parts(parts: Vec<JsContentPart>) -> Result<Vec<ContentPart>> {
    parts
        .into_iter()
        .map(|part| match part.part_type.as_str() {
            "text" => Ok(ContentPart::Text {
                text: part.text.unwrap_or_default(),
            }),
            "image" => {
                let img = part.image.ok_or_else(|| {
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        "Content part with partType \"image\" must include an `image` field",
                    )
                })?;
                let source = match img.source.source_type.as_str() {
                    "url" => ImageSource::Url {
                        url: img.source.url.unwrap_or_default(),
                    },
                    "base64" => ImageSource::Base64 {
                        data: img.source.data.unwrap_or_default(),
                    },
                    other => {
                        return Err(napi::Error::new(
                            napi::Status::InvalidArg,
                            format!(
                                "Invalid image source type: \"{other}\". Must be \"url\" or \"base64\""
                            ),
                        ))
                    }
                };
                Ok(ContentPart::Image(ImageContent {
                    source,
                    media_type: img.media_type,
                }))
            }
            "audio" => {
                let audio = part.audio.ok_or_else(|| {
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        "Content part with partType \"audio\" must include an `audio` field",
                    )
                })?;
                let source = match audio.source.source_type.as_str() {
                    "url" => MediaSource::Url {
                        url: audio.source.url.unwrap_or_default(),
                    },
                    "base64" => MediaSource::Base64 {
                        data: audio.source.data.unwrap_or_default(),
                    },
                    other => {
                        return Err(napi::Error::new(
                            napi::Status::InvalidArg,
                            format!(
                                "Invalid audio source type: \"{other}\". Must be \"url\" or \"base64\""
                            ),
                        ));
                    }
                };
                #[allow(clippy::cast_possible_truncation)]
                let duration = audio.duration_seconds.map(|d| d as f32);
                Ok(ContentPart::Audio(AudioContent {
                    source,
                    media_type: audio.media_type,
                    duration_seconds: duration,
                }))
            }
            "video" => {
                let video = part.video.ok_or_else(|| {
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        "Content part with partType \"video\" must include a `video` field",
                    )
                })?;
                let source = match video.source.source_type.as_str() {
                    "url" => MediaSource::Url {
                        url: video.source.url.unwrap_or_default(),
                    },
                    "base64" => MediaSource::Base64 {
                        data: video.source.data.unwrap_or_default(),
                    },
                    other => {
                        return Err(napi::Error::new(
                            napi::Status::InvalidArg,
                            format!(
                                "Invalid video source type: \"{other}\". Must be \"url\" or \"base64\""
                            ),
                        ));
                    }
                };
                #[allow(clippy::cast_possible_truncation)]
                let duration = video.duration_seconds.map(|d| d as f32);
                Ok(ContentPart::Video(VideoContent {
                    source,
                    media_type: video.media_type,
                    duration_seconds: duration,
                }))
            }
            other => Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!("Invalid content part type: \"{other}\". Must be \"text\", \"image\", \"audio\", or \"video\""),
            )),
        })
        .collect()
}
