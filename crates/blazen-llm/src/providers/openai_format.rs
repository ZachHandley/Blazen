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

/// Render a tool-result message as a string for OpenAI-family wire formats.
///
/// Honors `LlmPayload::Text`, `Json`, `Parts`, and `ProviderRaw` overrides
/// when present. `ProviderRaw` only applies if `self_provider` matches the
/// payload's `provider`; otherwise it falls back to the default conversion
/// of `data`. Plain-string `data` passes through unchanged; structured
/// `data` is JSON-stringified once at this boundary.
///
/// Returns an empty string when the message is not a tool-result message
/// (i.e. `tool_result_view` returns `None`).
pub(crate) fn tool_result_to_openai_string(
    msg: &crate::types::ChatMessage,
    self_provider: crate::types::ProviderId,
) -> String {
    let Some((data, override_payload)) = msg.tool_result_view() else {
        return String::new();
    };

    if let Some(payload) = override_payload {
        return match payload {
            crate::types::LlmPayload::Text { text } => text.clone(),
            crate::types::LlmPayload::Json { value } => stringify_value(value),
            crate::types::LlmPayload::Parts { parts } => parts
                .iter()
                .filter_map(|p| match p {
                    crate::types::ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            crate::types::LlmPayload::ProviderRaw { provider, value }
                if *provider == self_provider =>
            {
                stringify_value(value)
            }
            crate::types::LlmPayload::ProviderRaw { .. } => stringify_value(&data),
        };
    }

    stringify_value(&data)
}

#[allow(dead_code)] // Used by `tool_result_to_openai_string`; Wave 5 activates the call sites.
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
    use super::tool_result_to_openai_string;
    use crate::types::{ChatMessage, ContentPart, LlmPayload, ProviderId, ToolOutput};

    #[test]
    fn structured_data_stringifies_at_boundary() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!({"k":"v"}));
        assert_eq!(
            tool_result_to_openai_string(&msg, ProviderId::OpenAi),
            "{\"k\":\"v\"}"
        );
    }

    #[test]
    fn string_data_passes_through() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!("hello"));
        assert_eq!(
            tool_result_to_openai_string(&msg, ProviderId::OpenAi),
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
            tool_result_to_openai_string(&msg, ProviderId::OpenAi),
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
        // OpenAI is not the target — falls back to default conversion of data.
        assert_eq!(
            tool_result_to_openai_string(&msg, ProviderId::OpenAi),
            "{\"k\":\"v\"}"
        );
        // Anthropic IS the target — value passes through (stringified).
        assert_eq!(
            tool_result_to_openai_string(&msg, ProviderId::Anthropic),
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
            tool_result_to_openai_string(&msg, ProviderId::OpenAi),
            "line1\nline2"
        );
    }
}
