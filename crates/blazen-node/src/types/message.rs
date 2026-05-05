//! Chat message types and content part conversion helpers.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::content::{ContentHandle, ContentKind};
use blazen_llm::types::{
    AudioContent, ChatMessage, ContentPart, ImageContent, ImageSource, MediaSource, MessageContent,
    ProviderId, Role, VideoContent,
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

/// How an image / audio / video / file is provided.
///
/// `sourceType` is one of:
/// - `"url"` — set `url` to a public URL.
/// - `"base64"` — set `data` to the base64-encoded payload.
/// - `"file"` — set `url` to a local filesystem path (local backends only).
/// - `"providerFile"` — set `provider` and `id` to reference an
///   already-uploaded file in a provider's file API (`OpenAI` Files,
///   Anthropic Files, Gemini Files, fal.ai storage).
/// - `"handle"` — set `handleId` (and optionally `handleKind`) to reference
///   content registered with a `ContentStore`. The store resolves the
///   handle at request-build time.
#[napi(object)]
pub struct JsImageSource {
    #[napi(js_name = "sourceType")]
    pub source_type: String,
    pub url: Option<String>,
    pub data: Option<String>,
    /// Provider name for `sourceType: "providerFile"` (e.g. `"openai"`,
    /// `"anthropic"`, `"gemini"`, `"fal"`).
    pub provider: Option<String>,
    /// Provider-issued file id for `sourceType: "providerFile"`.
    pub id: Option<String>,
    /// Content handle id for `sourceType: "handle"`.
    #[napi(js_name = "handleId")]
    pub handle_id: Option<String>,
    /// Content handle kind for `sourceType: "handle"` (e.g. `"image"`,
    /// `"audio"`, `"three_d_model"`). See `ContentKind` for the full set.
    #[napi(js_name = "handleKind")]
    pub handle_kind: Option<String>,
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

/// Parse a JS [`JsImageSource`] into a Rust [`ImageSource`].
///
/// Accepts `"url"` / `"base64"` / `"file"` / `"providerFile"` / `"handle"`
/// for the `sourceType` field. Returns a napi error on unknown source
/// types or missing required fields for the chosen type.
pub(crate) fn js_source_to_rust(source: &JsImageSource, kind_label: &str) -> Result<ImageSource> {
    match source.source_type.as_str() {
        "url" => Ok(ImageSource::Url {
            url: source.url.clone().unwrap_or_default(),
        }),
        "base64" => Ok(ImageSource::Base64 {
            data: source.data.clone().unwrap_or_default(),
        }),
        "file" => Ok(ImageSource::File {
            path: std::path::PathBuf::from(source.url.clone().unwrap_or_default()),
        }),
        "providerFile" => {
            let provider_str = source.provider.as_deref().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "{kind_label} source with sourceType \"providerFile\" must include `provider`"
                    ),
                )
            })?;
            let provider = parse_provider_id(provider_str)?;
            let id = source.id.clone().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "{kind_label} source with sourceType \"providerFile\" must include `id`"
                    ),
                )
            })?;
            Ok(ImageSource::ProviderFile { provider, id })
        }
        "handle" => {
            let id = source.handle_id.clone().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "{kind_label} source with sourceType \"handle\" must include `handleId`"
                    ),
                )
            })?;
            let kind = match source.handle_kind.as_deref() {
                Some(k) => parse_content_kind(k)?,
                None => ContentKind::Other,
            };
            Ok(ImageSource::Handle {
                handle: ContentHandle::new(id, kind),
            })
        }
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
                "Invalid {kind_label} source type: \"{other}\". Must be one of: \
                 url, base64, file, providerFile, handle"
            ),
        )),
    }
}

fn parse_provider_id(s: &str) -> Result<ProviderId> {
    match s.to_ascii_lowercase().as_str() {
        "openai" => Ok(ProviderId::OpenAi),
        "openai_compat" | "openaicompat" => Ok(ProviderId::OpenAiCompat),
        "azure" => Ok(ProviderId::Azure),
        "anthropic" => Ok(ProviderId::Anthropic),
        "gemini" => Ok(ProviderId::Gemini),
        "responses" => Ok(ProviderId::Responses),
        "fal" => Ok(ProviderId::Fal),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Invalid provider: \"{other}\""),
        )),
    }
}

fn parse_content_kind(s: &str) -> Result<ContentKind> {
    match s.to_ascii_lowercase().as_str() {
        "image" => Ok(ContentKind::Image),
        "audio" => Ok(ContentKind::Audio),
        "video" => Ok(ContentKind::Video),
        "document" => Ok(ContentKind::Document),
        "three_d_model" | "threedmodel" | "three-d-model" => Ok(ContentKind::ThreeDModel),
        "cad" => Ok(ContentKind::Cad),
        "archive" => Ok(ContentKind::Archive),
        "font" => Ok(ContentKind::Font),
        "code" => Ok(ContentKind::Code),
        "data" => Ok(ContentKind::Data),
        "other" => Ok(ContentKind::Other),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Invalid content kind: \"{other}\""),
        )),
    }
}

/// Render a Rust [`ImageSource`] back to a JS [`JsImageSource`].
#[must_use]
pub(crate) fn rust_source_to_js(source: &MediaSource) -> JsImageSource {
    match source {
        ImageSource::Url { url } => JsImageSource {
            source_type: "url".into(),
            url: Some(url.clone()),
            data: None,
            provider: None,
            id: None,
            handle_id: None,
            handle_kind: None,
        },
        ImageSource::Base64 { data } => JsImageSource {
            source_type: "base64".into(),
            url: None,
            data: Some(data.clone()),
            provider: None,
            id: None,
            handle_id: None,
            handle_kind: None,
        },
        ImageSource::File { path } => JsImageSource {
            source_type: "file".into(),
            url: Some(path.to_string_lossy().into_owned()),
            data: None,
            provider: None,
            id: None,
            handle_id: None,
            handle_kind: None,
        },
        ImageSource::ProviderFile { provider, id } => JsImageSource {
            source_type: "providerFile".into(),
            url: None,
            data: None,
            provider: Some(format!("{provider:?}").to_lowercase()),
            id: Some(id.clone()),
            handle_id: None,
            handle_kind: None,
        },
        ImageSource::Handle { handle } => JsImageSource {
            source_type: "handle".into(),
            url: None,
            data: None,
            provider: None,
            id: None,
            handle_id: Some(handle.id.clone()),
            handle_kind: Some(handle.kind.as_str().to_owned()),
        },
        // `ImageSource` is `#[non_exhaustive]`. Future variants degrade to
        // `unknown` rather than panicking; the JS side can detect this and
        // fall back appropriately.
        _ => JsImageSource {
            source_type: "unknown".into(),
            url: None,
            data: None,
            provider: None,
            id: None,
            handle_id: None,
            handle_kind: None,
        },
    }
}

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
                let source = js_source_to_rust(&img.source, "image")?;
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
                let source = js_source_to_rust(&audio.source, "audio")?;
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
                let source = js_source_to_rust(&video.source, "video")?;
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
