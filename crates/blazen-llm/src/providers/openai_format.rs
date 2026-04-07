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
