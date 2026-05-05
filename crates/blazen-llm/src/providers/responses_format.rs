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
                // Tool result -> function_call_output block. The Responses API
                // requires `output` to be a string. Multimodal parts (images,
                // audio, files) are emitted as a separate, immediately-
                // following `role: "user"` input item so the model can see
                // them in connection with the tool output.
                let split = super::openai_format::split_tool_result_parts(
                    msg,
                    crate::types::ProviderId::Responses,
                );
                let call_id = msg.tool_call_id.clone().unwrap_or_default();
                out.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": split.text,
                }));

                if !split.follow_up_parts.is_empty() {
                    let blocks: Vec<serde_json::Value> = split
                        .follow_up_parts
                        .iter()
                        .filter_map(|p| content_part_to_responses(p, "input_text"))
                        .filter(|v| !v.is_null())
                        .collect();
                    if !blocks.is_empty() {
                        out.push(serde_json::json!({
                            "role": "user",
                            "content": blocks,
                        }));
                    }
                }
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
        ContentPart::Audio(audio) => audio_part_to_responses(audio),
        ContentPart::Video(_) => {
            tracing::warn!(
                "fal Responses API: video chat input is not supported; video content dropped."
            );
            None
        }
        ContentPart::File(file) => file_part_to_responses(file),
    }
}

fn audio_part_to_responses(audio: &crate::types::AudioContent) -> Option<serde_json::Value> {
    match &audio.source {
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
        ImageSource::File { .. } => {
            tracing::warn!(
                "fal Responses API: local file source is not supported — use a URL or base64 \
                 source instead; audio content dropped."
            );
            None
        }
        ImageSource::ProviderFile { provider, id } => {
            crate::content::render::warn_provider_file_mismatch(
                crate::types::ProviderId::Responses,
                *provider,
                id,
                crate::content::render::MediaKindLabel::Audio,
            );
            None
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Responses,
                handle,
                crate::content::render::MediaKindLabel::Audio,
            );
            None
        }
    }
}

fn file_part_to_responses(file: &crate::types::FileContent) -> Option<serde_json::Value> {
    let url = match &file.source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { data } => {
            format!("data:{};base64,{data}", file.media_type)
        }
        ImageSource::File { .. } => {
            tracing::warn!(
                "fal Responses API: local file source is not supported — use a URL or \
                 base64 source instead; file content dropped."
            );
            return None;
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(
                provider,
                crate::types::ProviderId::OpenAi | crate::types::ProviderId::Responses
            ) {
                let mut block = serde_json::json!({
                    "type": "input_file",
                    "file_id": id,
                });
                if let Some(name) = &file.filename {
                    block["filename"] = name.clone().into();
                }
                return Some(block);
            }
            crate::content::render::warn_provider_file_mismatch(
                crate::types::ProviderId::Responses,
                *provider,
                id,
                crate::content::render::MediaKindLabel::File,
            );
            return None;
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Responses,
                handle,
                crate::content::render::MediaKindLabel::File,
            );
            return None;
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

fn image_to_input_image(source: &ImageSource, media_type: Option<&str>) -> serde_json::Value {
    let url = match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { data } => {
            let mt = media_type.unwrap_or("image/png");
            format!("data:{mt};base64,{data}")
        }
        ImageSource::File { .. } => {
            tracing::warn!(
                "fal Responses API: local file source is not supported — use a URL or base64 \
                 source instead; image content dropped."
            );
            return serde_json::Value::Null;
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(
                provider,
                crate::types::ProviderId::OpenAi | crate::types::ProviderId::Responses
            ) {
                return serde_json::json!({
                    "type": "input_image",
                    "file_id": id,
                });
            }
            crate::content::render::warn_provider_file_mismatch(
                crate::types::ProviderId::Responses,
                *provider,
                id,
                crate::content::render::MediaKindLabel::Image,
            );
            return serde_json::Value::Null;
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Responses,
                handle,
                crate::content::render::MediaKindLabel::Image,
            );
            return serde_json::Value::Null;
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
            ChatMessage::tool_result("call_1", "calculator", serde_json::json!("4")),
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
    fn tool_result_with_image_appends_input_image_item() {
        use crate::types::{ContentPart, LlmPayload, ToolOutput};
        let messages = vec![ChatMessage::tool_result(
            "call_1",
            "render",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![
                        ContentPart::text("rendered"),
                        ContentPart::image_base64("AAAA", "image/png"),
                    ],
                },
            ),
        )];
        let input = messages_to_responses_input(&messages);
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["output"], "rendered");
        assert_eq!(input[1]["role"], "user");
        let blocks = input[1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "input_image");
    }

    #[test]
    fn tool_result_text_only_yields_single_function_call_output() {
        let messages = vec![ChatMessage::tool_result(
            "call_1",
            "search",
            serde_json::json!("found 3 results"),
        )];
        let input = messages_to_responses_input(&messages);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["output"], "found 3 results");
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
