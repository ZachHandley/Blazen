//! Shared helpers for OpenAI-compatible wire formats.
//!
//! These functions convert Blazen's [`MessageContent`], [`ContentPart`], and
//! [`ImageContent`] types to the `serde_json::Value` shape expected by
//! `OpenAI`-compatible chat completion APIs (`OpenAI`, Azure, `OpenRouter`, Groq,
//! Together AI, etc.).
//!
//! Also contains shared HTTP utilities used by multiple providers, such as
//! [`parse_retry_after`] for extracting `Retry-After` header values.

use crate::types::{ContentPart, ImageContent, ImageSource, MessageContent};

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Extract a retry-after duration from HTTP response headers.
///
/// The `Retry-After` header can be either a number of seconds (e.g. `30`) or
/// an HTTP date. This function handles the numeric form and returns the value
/// in **milliseconds**. HTTP-date values are currently ignored (returns `None`).
///
/// Accepts a slice of `(name, value)` pairs as provided by
/// [`crate::http::HttpResponse::headers`].
pub(crate) fn parse_retry_after(headers: &[(String, String)]) -> Option<u64> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .and_then(|secs| {
            if secs.is_finite() && secs >= 0.0 {
                // Convert seconds to milliseconds. Values are small in
                // practice (typically < 3600), so precision loss is not
                // a concern.
                let ms = secs * 1000.0;
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                Some(ms.min(u64::MAX as f64) as u64)
            } else {
                None
            }
        })
}

// ---------------------------------------------------------------------------
// Multimodal content helpers
// ---------------------------------------------------------------------------

/// Convert an [`ImageContent`] to the `OpenAI` `image_url` content-part format.
pub(crate) fn image_content_to_openai(img: &ImageContent) -> serde_json::Value {
    let url = match &img.source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { data } => {
            let media_type = img.media_type.as_deref().unwrap_or("image/png");
            format!("data:{media_type};base64,{data}")
        }
        ImageSource::File { .. } => {
            tracing::warn!(
                "openai-compat: local file source is not supported — use a URL or base64 source \
                 instead; image content dropped."
            );
            return serde_json::Value::Null;
        }
        ImageSource::ProviderFile { provider, id } => {
            crate::content::render::warn_provider_file_mismatch(
                crate::types::ProviderId::OpenAi,
                *provider,
                id,
                crate::content::render::MediaKindLabel::Image,
            );
            return serde_json::Value::Null;
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::OpenAi,
                handle,
                crate::content::render::MediaKindLabel::Image,
            );
            return serde_json::Value::Null;
        }
    };
    serde_json::json!({
        "type": "image_url",
        "image_url": { "url": url }
    })
}

/// Convert a single [`ContentPart`] to an `OpenAI` content-array element.
pub(crate) fn content_part_to_openai(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => {
            serde_json::json!({ "type": "text", "text": text })
        }
        ContentPart::Image(img) => image_content_to_openai(img),
        ContentPart::File(file) => {
            // Files are sent as image_url with a data URI (best-effort for
            // OpenAI-compatible endpoints).
            let url = match &file.source {
                ImageSource::Url { url } => url.clone(),
                ImageSource::Base64 { data } => {
                    format!("data:{};base64,{data}", file.media_type)
                }
                ImageSource::File { .. } => {
                    tracing::warn!(
                        "openai-compat: local file source is not supported — use a URL or base64 \
                         source instead; file content dropped."
                    );
                    return serde_json::Value::Null;
                }
                ImageSource::ProviderFile { provider, id } => {
                    crate::content::render::warn_provider_file_mismatch(
                        crate::types::ProviderId::OpenAi,
                        *provider,
                        id,
                        crate::content::render::MediaKindLabel::File,
                    );
                    return serde_json::Value::Null;
                }
                ImageSource::Handle { handle } => {
                    crate::content::render::warn_handle_unresolved(
                        crate::types::ProviderId::OpenAi,
                        handle,
                        crate::content::render::MediaKindLabel::File,
                    );
                    return serde_json::Value::Null;
                }
            };
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": url }
            })
        }
        ContentPart::Audio(audio) => {
            // OpenAI's `gpt-4o-audio-preview` accepts `input_audio` blocks
            // with base64 data and a short format hint. URL-sourced audio is
            // not natively supported -- drop with a warning.
            match &audio.source {
                ImageSource::Base64 { data } => {
                    let format = audio
                        .media_type
                        .as_deref()
                        .and_then(|m| m.strip_prefix("audio/"))
                        .unwrap_or("mp3");
                    serde_json::json!({
                        "type": "input_audio",
                        "input_audio": { "data": data, "format": format }
                    })
                }
                ImageSource::Url { .. } => {
                    tracing::warn!(
                        "openai-compat: audio URL inputs are not supported; \
                         pass base64 data via AudioContent::from_base64 instead. \
                         Audio content dropped."
                    );
                    serde_json::Value::Null
                }
                ImageSource::File { .. } => {
                    tracing::warn!(
                        "openai-compat: local file source is not supported — use a URL or base64 \
                         source instead; audio content dropped."
                    );
                    serde_json::Value::Null
                }
                ImageSource::ProviderFile { provider, id } => {
                    crate::content::render::warn_provider_file_mismatch(
                        crate::types::ProviderId::OpenAi,
                        *provider,
                        id,
                        crate::content::render::MediaKindLabel::Audio,
                    );
                    serde_json::Value::Null
                }
                ImageSource::Handle { handle } => {
                    crate::content::render::warn_handle_unresolved(
                        crate::types::ProviderId::OpenAi,
                        handle,
                        crate::content::render::MediaKindLabel::Audio,
                    );
                    serde_json::Value::Null
                }
            }
        }
        ContentPart::Video(_) => {
            tracing::warn!(
                "openai-compat: video chat input is not supported by this provider; \
                 video content dropped."
            );
            serde_json::Value::Null
        }
    }
}

/// Convert [`MessageContent`] to a `serde_json::Value` suitable for the
/// `OpenAI` `content` field.
///
/// - `Text` -> a plain JSON string (backward-compatible).
/// - `Image` / `Parts` -> a JSON array of content parts.
pub(crate) fn content_to_openai_value(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text(t) => serde_json::Value::String(t.clone()),
        MessageContent::Image(img) => {
            serde_json::json!([image_content_to_openai(img)])
        }
        MessageContent::Parts(parts) => {
            // Drop `Null` entries that the part converter emits for unsupported
            // content (e.g. video parts on a text-only model). Keeps the wire
            // body clean.
            let arr: Vec<serde_json::Value> = parts
                .iter()
                .map(content_part_to_openai)
                .filter(|v| !v.is_null())
                .collect();
            serde_json::json!(arr)
        }
    }
}

/// Result of splitting a tool-result message for OpenAI-shaped serialization.
///
/// The `text` portion goes into the tool message's `content` field. When
/// `follow_up_parts` is non-empty, callers should emit an immediately-
/// following `role: "user"` message containing those parts as multimodal
/// content blocks. `OpenAI` Chat Completions and OpenAI-compatible APIs do
/// not allow non-string content in `role: "tool"` messages, but accept
/// multimodal user messages, so the split-and-follow-up pattern preserves
/// the tool result's media.
pub(crate) struct ToolResultSplit {
    /// Joined text portion of the tool result.
    pub text: String,
    /// Non-text content parts that need to be emitted as a follow-up
    /// multimodal user message. Empty if the tool result is text-only.
    pub follow_up_parts: Vec<crate::types::ContentPart>,
}

/// Split a tool-result message's payload into a text portion and a list
/// of non-text parts.
///
/// Considers both sources of multimodal tool-result content:
/// - `tool_result.llm_override = LlmPayload::Parts { parts }` (the
///   structured-override path used by `ToolOutput::with_override`).
/// - `content = MessageContent::Parts(parts)` (the path used by
///   [`crate::types::ChatMessage::tool_result_parts`]).
///
/// `LlmPayload::Text` / `Json` / `ProviderRaw` overrides resolve to text-
/// only with an empty `follow_up_parts`. `ProviderRaw` is honored only
/// when `self_provider` matches the payload's `provider`.
///
/// Returns an empty `ToolResultSplit` when the message is not a tool-
/// result message.
pub(crate) fn split_tool_result_parts(
    msg: &crate::types::ChatMessage,
    self_provider: crate::types::ProviderId,
) -> ToolResultSplit {
    use crate::types::{LlmPayload, MessageContent, Role};

    if msg.role != Role::Tool {
        return ToolResultSplit {
            text: String::new(),
            follow_up_parts: Vec::new(),
        };
    }

    // Path 1: structured `tool_result` field with optional override.
    if let Some(out) = &msg.tool_result {
        if let Some(payload) = &out.llm_override {
            return match payload {
                LlmPayload::Text { text } => ToolResultSplit {
                    text: text.clone(),
                    follow_up_parts: Vec::new(),
                },
                LlmPayload::Json { value } => ToolResultSplit {
                    text: stringify_value(value),
                    follow_up_parts: Vec::new(),
                },
                LlmPayload::Parts { parts } => split_content_parts(parts),
                LlmPayload::ProviderRaw { provider, value } if *provider == self_provider => {
                    ToolResultSplit {
                        text: stringify_value(value),
                        follow_up_parts: Vec::new(),
                    }
                }
                LlmPayload::ProviderRaw { .. } => ToolResultSplit {
                    text: stringify_value(&out.data),
                    follow_up_parts: Vec::new(),
                },
            };
        }
        return ToolResultSplit {
            text: stringify_value(&out.data),
            follow_up_parts: Vec::new(),
        };
    }

    // Path 2: `tool_result` is None; payload lives in `content`.
    match &msg.content {
        MessageContent::Parts(parts) => split_content_parts(parts),
        MessageContent::Text(s) => ToolResultSplit {
            text: s.clone(),
            follow_up_parts: Vec::new(),
        },
        MessageContent::Image(img) => ToolResultSplit {
            text: String::new(),
            follow_up_parts: vec![crate::types::ContentPart::Image(img.clone())],
        },
    }
}

fn split_content_parts(parts: &[crate::types::ContentPart]) -> ToolResultSplit {
    let mut text_chunks: Vec<String> = Vec::new();
    let mut non_text: Vec<crate::types::ContentPart> = Vec::new();
    for part in parts {
        match part {
            crate::types::ContentPart::Text { text } => text_chunks.push(text.clone()),
            other => non_text.push(other.clone()),
        }
    }
    ToolResultSplit {
        text: text_chunks.join("\n"),
        follow_up_parts: non_text,
    }
}

fn stringify_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn parse_retry_after_integer_seconds() {
        let h = headers(&[("Retry-After", "30")]);
        assert_eq!(parse_retry_after(&h), Some(30_000));
    }

    #[test]
    fn parse_retry_after_fractional_seconds() {
        let h = headers(&[("retry-after", "1.5")]);
        assert_eq!(parse_retry_after(&h), Some(1_500));
    }

    #[test]
    fn parse_retry_after_missing() {
        let h: Vec<(String, String)> = Vec::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn parse_retry_after_http_date_returns_none() {
        let h = headers(&[("Retry-After", "Wed, 21 Oct 2026 07:28:00 GMT")]);
        // HTTP-date form is not yet supported, should return None.
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn parse_retry_after_zero() {
        let h = headers(&[("Retry-After", "0")]);
        assert_eq!(parse_retry_after(&h), Some(0));
    }
}

#[cfg(test)]
mod tool_result_string_tests {
    use super::split_tool_result_parts;
    use crate::types::{ChatMessage, ContentPart, LlmPayload, ProviderId, ToolOutput};

    #[test]
    fn structured_data_stringifies_at_boundary() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!({"k":"v"}));
        assert_eq!(
            split_tool_result_parts(&msg, ProviderId::OpenAi).text,
            "{\"k\":\"v\"}"
        );
    }

    #[test]
    fn string_data_passes_through() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!("hello"));
        assert_eq!(
            split_tool_result_parts(&msg, ProviderId::OpenAi).text,
            "hello"
        );
    }

    #[test]
    fn text_override_wins() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({"items":[1,2,3]}),
                LlmPayload::Text {
                    text: "summary".into(),
                },
            ),
        );
        assert_eq!(
            split_tool_result_parts(&msg, ProviderId::OpenAi).text,
            "summary"
        );
    }

    #[test]
    fn provider_raw_only_applies_to_target() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({"k":"v"}),
                LlmPayload::ProviderRaw {
                    provider: ProviderId::Anthropic,
                    value: serde_json::json!("anthropic-only"),
                },
            ),
        );
        assert_eq!(
            split_tool_result_parts(&msg, ProviderId::OpenAi).text,
            "{\"k\":\"v\"}"
        );
        assert_eq!(
            split_tool_result_parts(&msg, ProviderId::Anthropic).text,
            "anthropic-only"
        );
    }

    #[test]
    fn parts_override_concatenates_text() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![ContentPart::text("line1"), ContentPart::text("line2")],
                },
            ),
        );
        assert_eq!(
            split_tool_result_parts(&msg, ProviderId::OpenAi).text,
            "line1\nline2"
        );
    }
}

#[cfg(test)]
mod split_tool_result_tests {
    use super::split_tool_result_parts;
    use crate::types::{ChatMessage, ContentPart, LlmPayload, ProviderId, ToolOutput};

    #[test]
    fn text_only_parts_have_empty_follow_up() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![ContentPart::text("hello"), ContentPart::text("world")],
                },
            ),
        );
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert_eq!(split.text, "hello\nworld");
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn parts_override_with_image_yields_follow_up() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "render",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![
                        ContentPart::text("rendered:"),
                        ContentPart::image_base64("AAAA", "image/png"),
                    ],
                },
            ),
        );
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert_eq!(split.text, "rendered:");
        assert_eq!(split.follow_up_parts.len(), 1);
        assert!(matches!(split.follow_up_parts[0], ContentPart::Image(_)));
    }

    #[test]
    fn tool_result_parts_constructor_path() {
        // `ChatMessage::tool_result_parts` stores parts in `content` with
        // `tool_result: None`. The split must still find the non-text parts.
        let msg = ChatMessage::tool_result_parts(
            "call_1",
            "render",
            vec![
                ContentPart::text("see attached:"),
                ContentPart::image_url("https://example.com/x.png", None),
            ],
        );
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert_eq!(split.text, "see attached:");
        assert_eq!(split.follow_up_parts.len(), 1);
    }

    #[test]
    fn audio_and_video_parts_appear_in_follow_up() {
        let msg = ChatMessage::tool_result_parts(
            "call_1",
            "media",
            vec![
                ContentPart::audio_base64("AAA", "audio/mp3"),
                ContentPart::video_url("https://example.com/v.mp4"),
            ],
        );
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert!(split.text.is_empty());
        assert_eq!(split.follow_up_parts.len(), 2);
    }

    #[test]
    fn text_payload_has_no_follow_up() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Text { text: "ok".into() },
            ),
        );
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert_eq!(split.text, "ok");
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn provider_raw_match_uses_value_as_text() {
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({"k":"v"}),
                LlmPayload::ProviderRaw {
                    provider: ProviderId::OpenAi,
                    value: serde_json::json!("openai-only"),
                },
            ),
        );
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert_eq!(split.text, "openai-only");
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn non_tool_message_returns_empty() {
        let msg = ChatMessage::user("hello");
        let split = split_tool_result_parts(&msg, ProviderId::OpenAi);
        assert!(split.text.is_empty());
        assert!(split.follow_up_parts.is_empty());
    }
}
