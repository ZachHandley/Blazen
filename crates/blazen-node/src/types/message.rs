//! Chat message types and content part conversion helpers.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::types::{
    ChatMessage, ContentPart, ImageContent, ImageSource, MessageContent, Role,
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

/// A single part in a multi-part message.
#[napi(object)]
pub struct JsContentPart {
    #[napi(js_name = "partType")]
    pub part_type: String,
    pub text: Option<String>,
    pub image: Option<JsImageContent>,
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

    /// Create a user message from an explicit list of content parts.
    #[napi(factory, js_name = "userParts")]
    pub fn user_parts(parts: Vec<JsContentPart>) -> Result<Self> {
        let content_parts = convert_js_parts(parts)?;
        Ok(Self {
            inner: ChatMessage::user_parts(content_parts),
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
            other => Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!("Invalid content part type: \"{other}\". Must be \"text\" or \"image\""),
            )),
        })
        .collect()
}
