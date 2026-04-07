//! `OpenAI` Responses API body conversion.
//!
//! The Responses API (at `/v1/responses`) uses a different request shape from
//! chat completions: a single `input` array of role-tagged content blocks
//! whose content parts are tagged `input_text`, `input_image`, `input_audio`,
//! `input_file`, etc., plus separate `function_call` / `function_call_output`
//! blocks for tool flow.
//!
//! This module converts Blazen's `ChatMessage` types into the Responses input
//! shape. It is consumed by `FalProvider::build_openai_responses_body`.

use crate::types::{ChatMessage, ContentPart, ImageSource, MessageContent, Role};

/// Convert a slice of [`ChatMessage`] values into an `OpenAI` Responses API
/// `input` array.
#[must_use]
pub(crate) fn messages_to_responses_input(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg.role {
            Role::System | Role::User | Role::Assistant => {
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => unreachable!(),
                };
                let content = content_to_responses_input_parts(&msg.content, &msg.role);
                out.push(serde_json::json!({
                    "role": role,
                    "content": content,
                }));
                // Assistant tool calls become separate function_call blocks.
                for call in &msg.tool_calls {
                    out.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": &call.id,
                        "name": &call.name,
                        "arguments": call.arguments.to_string(),
                    }));
                }
            }
            Role::Tool => {
                // Tool result -> function_call_output block.
                let output = msg.content.text_content().unwrap_or_default();
                let call_id = msg.tool_call_id.clone().unwrap_or_default();
                out.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
        }
    }
    out
}

/// Convert a [`MessageContent`] into Responses API content-part array entries
/// for the given `role`. Assistant text uses `output_text`; user/system text
/// uses `input_text`.
fn content_to_responses_input_parts(
    content: &MessageContent,
    role: &Role,
) -> Vec<serde_json::Value> {
    let text_kind = if matches!(role, Role::Assistant) {
        "output_text"
    } else {
        "input_text"
    };
    match content {
        MessageContent::Text(s) => {
            vec![serde_json::json!({ "type": text_kind, "text": s })]
        }
        MessageContent::Image(img) => {
            vec![image_to_input_image(&img.source, img.media_type.as_deref())]
        }
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| content_part_to_responses(p, text_kind))
            .collect(),
    }
}

fn content_part_to_responses(part: &ContentPart, text_kind: &str) -> Option<serde_json::Value> {
    match part {
        ContentPart::Text { text } => Some(serde_json::json!({ "type": text_kind, "text": text })),
        ContentPart::Image(img) => {
            Some(image_to_input_image(&img.source, img.media_type.as_deref()))
        }
        ContentPart::Audio(audio) => match &audio.source {
            ImageSource::Base64 { data } => {
                let format = audio
                    .media_type
                    .as_deref()
                    .and_then(|m| m.strip_prefix("audio/"))
                    .unwrap_or("mp3");
                Some(serde_json::json!({
                    "type": "input_audio",
                    "input_audio": { "data": data, "format": format }
                }))
            }
            ImageSource::Url { .. } => {
                tracing::warn!(
                    "fal Responses API: audio URL inputs are not supported; \
                     pass base64 data via AudioContent::from_base64 instead. \
                     Audio content dropped."
                );
                None
            }
        },
        ContentPart::Video(_) => {
            tracing::warn!(
                "fal Responses API: video chat input is not supported; video content dropped."
            );
            None
        }
        ContentPart::File(file) => {
            let url = match &file.source {
                ImageSource::Url { url } => url.clone(),
                ImageSource::Base64 { data } => {
                    format!("data:{};base64,{data}", file.media_type)
                }
            };
            let mut block = serde_json::json!({
                "type": "input_file",
                "file_url": url,
            });
            if let Some(name) = &file.filename {
                block["filename"] = name.clone().into();
            }
            Some(block)
        }
    }
}

fn image_to_input_image(source: &ImageSource, media_type: Option<&str>) -> serde_json::Value {
    let url = match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { data } => {
            let mt = media_type.unwrap_or("image/png");
            format!("data:{mt};base64,{data}")
        }
    };
    serde_json::json!({
        "type": "input_image",
        "image_url": url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_to_responses_input_basic() {
        let messages = vec![
            ChatMessage::system("be helpful"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there"),
        ];
        let input = messages_to_responses_input(&messages);
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[2]["content"][0]["type"], "output_text");
    }

    #[test]
    fn test_messages_to_responses_input_with_image() {
        let messages = vec![ChatMessage::user_image_url(
            "describe",
            "https://i.com/a.png",
            Some("image/png"),
        )];
        let input = messages_to_responses_input(&messages);
        let parts = input[0]["content"].as_array().unwrap();
        assert!(parts.iter().any(|p| p["type"] == "input_image"));
    }

    #[test]
    fn test_messages_to_responses_input_with_function_call_history() {
        use crate::types::ToolCall;
        let messages = vec![
            ChatMessage::user("calc 2+2"),
            ChatMessage::assistant_with_tool_calls(
                None,
                vec![ToolCall {
                    id: "call_1".into(),
                    name: "calculator".into(),
                    arguments: serde_json::json!({"expr": "2+2"}),
                }],
            ),
            ChatMessage::tool_result("call_1", "calculator", "4"),
        ];
        let input = messages_to_responses_input(&messages);
        // user msg, assistant msg + function_call block, function_call_output block
        assert!(input.len() >= 4);
        let function_call_block = input
            .iter()
            .find(|b| b["type"] == "function_call")
            .expect("function_call block");
        assert_eq!(function_call_block["call_id"], "call_1");
        let output_block = input
            .iter()
            .find(|b| b["type"] == "function_call_output")
            .expect("function_call_output block");
        assert_eq!(output_block["call_id"], "call_1");
        assert_eq!(output_block["output"], "4");
    }

    #[test]
    fn test_messages_to_responses_input_drops_video_with_warn() {
        let messages = vec![ChatMessage::user_video("describe", "https://v.com/c.mp4")];
        let input = messages_to_responses_input(&messages);
        // The user message exists but the video part was dropped, leaving only text.
        let parts = input[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "input_text");
    }
}
