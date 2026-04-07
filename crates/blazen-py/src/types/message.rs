//! Python wrappers for chat message types.

use pyo3::prelude::*;

use blazen_llm::{
    AudioContent, ChatMessage, ContentPart, ImageContent, ImageSource, MessageContent, Role,
    VideoContent,
};

use crate::error::BlazenPyError;

// ---------------------------------------------------------------------------
// PyRole
// ---------------------------------------------------------------------------

/// Role constants for chat messages.
///
/// Example:
///     >>> ChatMessage(role=Role.USER, content="Hello!")
///     >>> ChatMessage(role=Role.SYSTEM, content="You are helpful.")
#[pyclass(name = "Role", frozen)]
pub struct PyRole;

#[pymethods]
impl PyRole {
    #[classattr]
    const SYSTEM: &'static str = "system";
    #[classattr]
    const USER: &'static str = "user";
    #[classattr]
    const ASSISTANT: &'static str = "assistant";
    #[classattr]
    const TOOL: &'static str = "tool";
}

// ---------------------------------------------------------------------------
// PyContentPart
// ---------------------------------------------------------------------------

#[pyclass(name = "ContentPart", from_py_object)]
#[derive(Clone)]
pub struct PyContentPart {
    pub(crate) inner: ContentPart,
}

#[pymethods]
impl PyContentPart {
    /// Create a text content part.
    #[staticmethod]
    #[pyo3(signature = (*, text))]
    fn text(text: &str) -> Self {
        Self {
            inner: ContentPart::Text {
                text: text.to_owned(),
            },
        }
    }

    /// Create an image content part from a URL.
    #[staticmethod]
    #[pyo3(signature = (*, url, media_type=None))]
    fn image_url(url: &str, media_type: Option<&str>) -> Self {
        Self {
            inner: ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: url.to_owned(),
                },
                media_type: media_type.map(String::from),
            }),
        }
    }

    /// Create an image content part from base64 data.
    #[staticmethod]
    #[pyo3(signature = (*, data, media_type))]
    fn image_base64(data: &str, media_type: &str) -> Self {
        Self {
            inner: ContentPart::Image(ImageContent {
                source: ImageSource::Base64 {
                    data: data.to_owned(),
                },
                media_type: Some(media_type.to_owned()),
            }),
        }
    }

    /// Create an audio content part from a URL.
    #[staticmethod]
    #[pyo3(signature = (*, url))]
    fn audio_url(url: &str) -> Self {
        Self {
            inner: ContentPart::Audio(AudioContent::from_url(url)),
        }
    }

    /// Create an audio content part from base64-encoded data.
    #[staticmethod]
    #[pyo3(signature = (*, data, media_type))]
    fn audio_base64(data: &str, media_type: &str) -> Self {
        Self {
            inner: ContentPart::Audio(AudioContent::from_base64(data, media_type)),
        }
    }

    /// Create a video content part from a URL.
    #[staticmethod]
    #[pyo3(signature = (*, url))]
    fn video_url(url: &str) -> Self {
        Self {
            inner: ContentPart::Video(VideoContent::from_url(url)),
        }
    }

    /// Create a video content part from base64-encoded data.
    #[staticmethod]
    #[pyo3(signature = (*, data, media_type))]
    fn video_base64(data: &str, media_type: &str) -> Self {
        Self {
            inner: ContentPart::Video(VideoContent::from_base64(data, media_type)),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ContentPart::Text { text } => format!("ContentPart.text(text='{text}')"),
            ContentPart::Image(_) => "ContentPart(image)".to_owned(),
            ContentPart::File(_) => "ContentPart(file)".to_owned(),
            ContentPart::Audio(_) => "ContentPart(audio)".to_owned(),
            ContentPart::Video(_) => "ContentPart(video)".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
///
/// Example:
///     >>> msg = ChatMessage(content="Hello, world!")
///     >>> msg = ChatMessage(role="system", content="You are a helpful assistant.")
///     >>> msg = ChatMessage.user("What is 2+2?")
#[pyclass(name = "ChatMessage", from_py_object)]
#[derive(Clone)]
pub struct PyChatMessage {
    pub(crate) inner: ChatMessage,
}

#[pymethods]
impl PyChatMessage {
    /// Create a new chat message.
    ///
    /// Args:
    ///     role: One of "system", "user", "assistant", "tool" (default: "user").
    ///     content: The message text.
    ///     parts: Optional list of ContentPart objects for multimodal messages.
    #[new]
    #[pyo3(signature = (role="user", content=None, parts=None))]
    fn new(
        role: &str,
        content: Option<&str>,
        parts: Option<Vec<PyRef<'_, PyContentPart>>>,
    ) -> PyResult<Self> {
        let role = match role {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(BlazenPyError::InvalidArgument(format!(
                    "unknown role: '{other}' (expected system, user, assistant, or tool)"
                ))
                .into());
            }
        };

        let message_content = if let Some(parts) = parts {
            MessageContent::Parts(parts.iter().map(|p| p.inner.clone()).collect())
        } else {
            MessageContent::Text(content.unwrap_or("").to_owned())
        };

        Ok(Self {
            inner: ChatMessage {
                role,
                content: message_content,
                tool_call_id: None,
                name: None,
                tool_calls: Vec::new(),
            },
        })
    }

    /// Create a system message.
    #[staticmethod]
    fn system(content: &str) -> Self {
        Self {
            inner: ChatMessage::system(content),
        }
    }

    /// Create a user message.
    #[staticmethod]
    fn user(content: &str) -> Self {
        Self {
            inner: ChatMessage::user(content),
        }
    }

    /// Create an assistant message.
    #[staticmethod]
    fn assistant(content: &str) -> Self {
        Self {
            inner: ChatMessage::assistant(content),
        }
    }

    /// Create a tool result message.
    #[staticmethod]
    fn tool(content: &str) -> Self {
        Self {
            inner: ChatMessage::tool(content),
        }
    }

    /// Create a user message with text and an image URL.
    #[staticmethod]
    #[pyo3(signature = (*, text, url, media_type=None))]
    fn user_image_url(text: &str, url: &str, media_type: Option<&str>) -> Self {
        Self {
            inner: ChatMessage::user_image_url(text, url, media_type),
        }
    }

    /// Create a user message with text and a base64-encoded image.
    #[staticmethod]
    #[pyo3(signature = (*, text, data, media_type))]
    fn user_image_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: ChatMessage::user_image_base64(text, data, media_type),
        }
    }

    /// Create a user message with text and an audio URL.
    #[staticmethod]
    #[pyo3(signature = (*, text, url))]
    fn user_audio(text: &str, url: &str) -> Self {
        Self {
            inner: ChatMessage::user_audio(text, url),
        }
    }

    /// Create a user message with text and base64-encoded audio.
    #[staticmethod]
    #[pyo3(signature = (*, text, data, media_type))]
    fn user_audio_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: ChatMessage::user_audio_base64(text, data, media_type),
        }
    }

    /// Create a user message with text and a video URL.
    #[staticmethod]
    #[pyo3(signature = (*, text, url))]
    fn user_video(text: &str, url: &str) -> Self {
        Self {
            inner: ChatMessage::user_video(text, url),
        }
    }

    /// Create a user message with text and base64-encoded video.
    #[staticmethod]
    #[pyo3(signature = (*, text, data, media_type))]
    fn user_video_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: ChatMessage::user_video_base64(text, data, media_type),
        }
    }

    /// Create a user message from a list of ContentPart objects.
    #[staticmethod]
    #[pyo3(signature = (*, parts))]
    fn user_parts(parts: Vec<PyRef<'_, PyContentPart>>) -> Self {
        let rust_parts: Vec<ContentPart> = parts.iter().map(|p| p.inner.clone()).collect();
        Self {
            inner: ChatMessage::user_parts(rust_parts),
        }
    }

    /// Get the role as a string.
    #[getter]
    fn role(&self) -> &str {
        match self.inner.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    /// Get the message content as a string.
    #[getter]
    fn content(&self) -> Option<&str> {
        self.inner.content.as_text()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatMessage(role='{}', content='{}')",
            self.role(),
            self.content().unwrap_or("")
        )
    }
}
