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
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
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
            let arr: Vec<serde_json::Value> = parts.iter().map(content_part_to_openai).collect();
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
    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

    #[test]
    fn parse_retry_after_integer_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("30"));
        assert_eq!(parse_retry_after(&headers), Some(30_000));
    }

    #[test]
    fn parse_retry_after_fractional_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("1.5"));
        assert_eq!(parse_retry_after(&headers), Some(1_500));
    }

    #[test]
    fn parse_retry_after_missing() {
        let headers = HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn parse_retry_after_http_date_returns_none() {
        let mut headers = HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        // HTTP-date form is not yet supported, should return None.
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn parse_retry_after_zero() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("0"));
        assert_eq!(parse_retry_after(&headers), Some(0));
    }
}
