//! `wasm-bindgen` wrappers for [`blazen_llm::ChatMessage`].
//!
//! Provides a JavaScript-friendly API with static factory methods for
//! creating messages of each role.

use wasm_bindgen::prelude::*;

use blazen_llm::types::{ChatMessage as InnerChatMessage, MessageContent, Role};

// ---------------------------------------------------------------------------
// WasmChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
///
/// Use the static factory methods to create messages:
///
/// ```js
/// const msg = ChatMessage.user("Hello!");
/// const sys = ChatMessage.system("You are helpful.");
/// const asst = ChatMessage.assistant("Sure thing.");
/// const tool = ChatMessage.tool("result text");
/// ```
#[wasm_bindgen(js_name = "ChatMessage")]
pub struct WasmChatMessage {
    pub(crate) inner: InnerChatMessage,
}

#[wasm_bindgen(js_class = "ChatMessage")]
impl WasmChatMessage {
    // -----------------------------------------------------------------------
    // Factory methods
    // -----------------------------------------------------------------------

    /// Create a system message.
    #[wasm_bindgen]
    pub fn system(content: &str) -> Self {
        Self {
            inner: InnerChatMessage::system(content),
        }
    }

    /// Create a user message.
    #[wasm_bindgen]
    pub fn user(content: &str) -> Self {
        Self {
            inner: InnerChatMessage::user(content),
        }
    }

    /// Create an assistant message.
    #[wasm_bindgen]
    pub fn assistant(content: &str) -> Self {
        Self {
            inner: InnerChatMessage::assistant(content),
        }
    }

    /// Create a tool result message.
    #[wasm_bindgen]
    pub fn tool(content: &str) -> Self {
        Self {
            inner: InnerChatMessage::tool(content),
        }
    }

    /// Create a tool result message with an associated tool call ID and function name.
    #[wasm_bindgen(js_name = "toolResult")]
    pub fn tool_result(call_id: &str, name: &str, content: &str) -> Self {
        Self {
            inner: InnerChatMessage::tool_result(call_id, name, content),
        }
    }

    // TODO: expose userParts (multi-modal mixed content) once JsContentPart equivalents are
    // defined for the WASM SDK. Today only the dedicated single-modality factories are available.

    /// Create a user message with text plus an image referenced by URL.
    #[wasm_bindgen(js_name = "userImageUrl")]
    pub fn user_image_url(text: &str, url: &str, media_type: Option<String>) -> Self {
        Self {
            inner: InnerChatMessage::user_image_url(text, url, media_type.as_deref()),
        }
    }

    /// Create a user message with text plus a base64-encoded image.
    #[wasm_bindgen(js_name = "userImageBase64")]
    pub fn user_image_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: InnerChatMessage::user_image_base64(text, data, media_type),
        }
    }

    /// Create a user message with text plus an audio clip referenced by URL.
    #[wasm_bindgen(js_name = "userAudio")]
    pub fn user_audio(text: &str, url: &str) -> Self {
        Self {
            inner: InnerChatMessage::user_audio(text, url),
        }
    }

    /// Create a user message with text plus a base64-encoded audio clip.
    #[wasm_bindgen(js_name = "userAudioBase64")]
    pub fn user_audio_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: InnerChatMessage::user_audio_base64(text, data, media_type),
        }
    }

    /// Create a user message with text plus a video clip referenced by URL.
    #[wasm_bindgen(js_name = "userVideo")]
    pub fn user_video(text: &str, url: &str) -> Self {
        Self {
            inner: InnerChatMessage::user_video(text, url),
        }
    }

    /// Create a user message with text plus a base64-encoded video clip.
    #[wasm_bindgen(js_name = "userVideoBase64")]
    pub fn user_video_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: InnerChatMessage::user_video_base64(text, data, media_type),
        }
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// The role of this message (`"system"`, `"user"`, `"assistant"`, or `"tool"`).
    #[wasm_bindgen(getter)]
    pub fn role(&self) -> String {
        match self.inner.role {
            Role::System => "system".to_owned(),
            Role::User => "user".to_owned(),
            Role::Assistant => "assistant".to_owned(),
            Role::Tool => "tool".to_owned(),
        }
    }

    /// The text content of this message, or `undefined` if it is not plain text.
    #[wasm_bindgen(getter)]
    pub fn content(&self) -> Option<String> {
        match &self.inner.content {
            MessageContent::Text(s) => Some(s.clone()),
            other => other.text_content(),
        }
    }

    /// The tool call ID, if this is a tool result message.
    #[wasm_bindgen(getter, js_name = "toolCallId")]
    pub fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    /// The function name, if this is a tool result message.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Serialize this message to a JSON object.
    #[wasm_bindgen(js_name = "toJSON")]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deserialize a ChatMessage from a JSON object.
    #[wasm_bindgen(js_name = "fromJSON")]
    pub fn from_json(value: JsValue) -> Result<WasmChatMessage, JsValue> {
        let inner: InnerChatMessage = serde_wasm_bindgen::from_value(value)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Convert a `JsValue` (expected to be a JS array of ChatMessage or JSON objects)
/// into a `Vec<InnerChatMessage>`.
pub(crate) fn js_messages_to_vec(
    messages: &JsValue,
) -> Result<Vec<InnerChatMessage>, JsValue> {
    let array = js_sys::Array::from(messages);
    let mut result = Vec::with_capacity(array.length() as usize);

    for i in 0..array.length() {
        let item = array.get(i);

        // Check if the item is a WasmChatMessage instance (has our internal structure).
        // Try to parse it as a JSON object first (covers plain JS objects).
        match serde_wasm_bindgen::from_value::<InnerChatMessage>(item.clone()) {
            Ok(msg) => result.push(msg),
            Err(e) => {
                return Err(JsValue::from_str(&format!(
                    "Failed to parse message at index {i}: {e}"
                )));
            }
        }
    }

    Ok(result)
}
