//! LLM provider implementations.
//!
//! All providers are always available. The only opt-in features are:
//! - `reqwest` — enables the native HTTP client (for non-WASM targets)
//! - `tiktoken` — enables exact BPE token counting
//!
//! ## Native providers (custom API formats)
//!
//! - [`openai`] — `OpenAI` Chat Completions API
//! - [`anthropic`] — Anthropic Messages API
//! - [`gemini`] — Google Gemini API
//! - [`azure`] — Azure `OpenAI` Service
//! - [`fal`] — fal.ai compute platform (LLM + media generation)
//!
//! ## OpenAI-compatible providers
//!
//! - [`openai_compat`] — Generic OpenAI-compatible base
//! - [`groq`] — Groq (ultra-fast LPU inference)
//! - [`openrouter`] — `OpenRouter` (400+ models)
//! - [`together`] — Together AI
//! - [`mistral`] — Mistral AI
//! - [`deepseek`] — `DeepSeek`
//! - [`fireworks`] — Fireworks AI
//! - [`perplexity`] — Perplexity
//! - [`xai`] — xAI (Grok)
//! - [`cohere`] — Cohere
//! - [`bedrock`] — AWS Bedrock (via Mantle)

// Shared SSE parser used by OpenAI-compatible and Azure providers.
pub(crate) mod sse;

// Shared multimodal content helpers and HTTP utilities.
pub(crate) mod openai_format;

// OpenAI Responses API body conversion (used by fal "openai-responses" route).
pub(crate) mod responses_format;

// Native providers
pub mod anthropic;
pub mod azure;
pub mod custom;
pub mod fal;
pub mod gemini;
pub mod openai;
mod openai_audio;
pub mod openai_compat;

// OpenAI-compatible dedicated providers
pub mod bedrock;
pub mod cohere;
pub mod deepseek;
pub mod fireworks;
pub mod groq;
pub mod mistral;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod xai;

// ---------------------------------------------------------------------------
// Shared provider-HTTP error helpers
// ---------------------------------------------------------------------------
//
// These helpers are consumed by provider call-sites as they migrate off
// ad-hoc `BlazenError::provider(...)` / `BlazenError::request(...)` onto
// structured `BlazenError::ProviderHttp`. The `#[allow(dead_code)]` below
// covers the window where the helpers exist but no call-site uses them yet;
// remove the attribute once the first provider has been migrated.

use crate::error::{BlazenError, PROVIDER_ERROR_BODY_CAP};
use crate::http::HttpResponse;
use serde::Deserialize;

/// Shape of a typical provider JSON error envelope.
///
/// Providers use one of `detail` / `error` / `message` at the top level:
/// - `fal.ai`, some `OpenAI` endpoints → `{"detail": "..."}`
/// - `DeepSeek`, `OpenRouter` → `{"error": "..."}` (string)
/// - `OpenAI`, Azure, Anthropic → `{"error": {"message": "..."}}`
/// - Gemini, some Cohere paths → `{"message": "..."}`
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ProviderErrorBody {
    #[serde(default)]
    detail: Option<String>,
    #[serde(default)]
    error: Option<serde_json::Value>,
    #[serde(default)]
    message: Option<String>,
}

/// Extract the most useful human-readable "detail" string from a
/// provider error body, or `None` if the body is not JSON or doesn't
/// match a known shape.
#[allow(dead_code)]
fn extract_detail(body: &str) -> Option<String> {
    let parsed: ProviderErrorBody = serde_json::from_str(body).ok()?;
    if let Some(d) = parsed.detail {
        return Some(d);
    }
    if let Some(m) = parsed.message {
        return Some(m);
    }
    match parsed.error {
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Object(obj)) => obj
            .get("message")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

/// Truncate a body to `PROVIDER_ERROR_BODY_CAP` bytes (char-boundary-safe),
/// appending `" ... [truncated N bytes]"` when truncation happens.
#[allow(dead_code)]
fn cap_body(body: &str) -> String {
    if body.len() <= PROVIDER_ERROR_BODY_CAP {
        return body.to_owned();
    }
    let mut end = PROVIDER_ERROR_BODY_CAP;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    let truncated_bytes = body.len() - end;
    format!("{} ... [truncated {truncated_bytes} bytes]", &body[..end])
}

/// Pick the provider request-id header from a response header list.
/// Checks (case-insensitively) `x-fal-request-id`, `x-request-id`,
/// then `request-id`. Returns the first hit.
#[allow(dead_code)]
fn pick_request_id(headers: &[(String, String)]) -> Option<String> {
    const CANDIDATES: &[&str] = &["x-fal-request-id", "x-request-id", "request-id"];
    for name in CANDIDATES {
        if let Some(v) = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
        {
            return Some(v.to_owned());
        }
    }
    None
}

/// Format the tail of a `BlazenError::ProviderHttp` Display string.
/// Used from the `#[error(...)]` attribute on the variant.
#[doc(hidden)]
#[must_use]
pub fn format_provider_http_tail(
    detail: Option<&str>,
    raw_body: &str,
    request_id: Option<&str>,
) -> String {
    let head: std::borrow::Cow<'_, str> = match detail {
        Some(d) => std::borrow::Cow::Borrowed(d),
        None => std::borrow::Cow::Owned(raw_body.chars().take(200).collect::<String>()),
    };
    match request_id {
        Some(id) => format!("{head} (request-id={id})"),
        None => head.into_owned(),
    }
}

/// Build a structured `BlazenError::ProviderHttp` from an `HttpResponse`.
///
/// Reads the body as lossy UTF-8, caps it at `PROVIDER_ERROR_BODY_CAP`,
/// extracts `detail` if the body is JSON with a known key, looks up
/// the request-id header, and parses `Retry-After`.
#[allow(dead_code)]
#[must_use]
pub(crate) fn provider_http_error(
    provider: impl Into<std::borrow::Cow<'static, str>>,
    endpoint: &str,
    response: &HttpResponse,
) -> BlazenError {
    let raw = response.text();
    let detail = extract_detail(&raw);
    let request_id = pick_request_id(&response.headers);
    let raw_body = cap_body(&raw);
    let retry_after_ms = openai_format::parse_retry_after(&response.headers);
    BlazenError::provider_http(
        provider,
        endpoint,
        response.status,
        request_id,
        detail,
        raw_body,
        retry_after_ms,
    )
}

/// Streaming sibling of [`provider_http_error`]. Use when you have
/// `(status, headers, body)` component parts without an `HttpResponse`
/// (as returned by `HttpClient::send_streaming`).
///
/// Pass `body = ""` when you have not read any of the stream body.
#[allow(dead_code)]
#[must_use]
pub(crate) fn provider_http_error_parts(
    provider: impl Into<std::borrow::Cow<'static, str>>,
    endpoint: &str,
    status: u16,
    headers: &[(String, String)],
    body: &str,
) -> BlazenError {
    let detail = extract_detail(body);
    let request_id = pick_request_id(headers);
    let raw_body = cap_body(body);
    let retry_after_ms = openai_format::parse_retry_after(headers);
    BlazenError::provider_http(
        provider,
        endpoint,
        status,
        request_id,
        detail,
        raw_body,
        retry_after_ms,
    )
}

// ---------------------------------------------------------------------------
// Shared `from_options` macro
// ---------------------------------------------------------------------------

/// Generate a `from_options(ProviderOptions) -> Result<Self, BlazenError>`
/// constructor for a "simple" provider that has `Self::new(api_key)`,
/// `with_model(m)`, and (optionally) `with_base_url(url)` builders.
///
/// The API key is resolved from `opts.api_key` first, then from the
/// provider's well-known environment variable (see [`crate::keys`]).
///
/// Use the `, no_base_url` variant for providers that don't expose
/// `with_base_url` (the OpenAI-compatible wrappers).
macro_rules! impl_simple_from_options {
    ($provider:ty, $name:expr) => {
        #[cfg(any(
            all(target_arch = "wasm32", not(target_os = "wasi")),
            feature = "reqwest",
            target_os = "wasi"
        ))]
        impl $provider {
            /// Construct from typed [`ProviderOptions`](crate::types::provider_options::ProviderOptions).
            ///
            /// # Errors
            ///
            /// Returns [`BlazenError::Auth`] if no API key is provided and
            /// the corresponding environment variable is not set.
            pub fn from_options(
                opts: $crate::types::provider_options::ProviderOptions,
            ) -> Result<Self, $crate::BlazenError> {
                let api_key = $crate::keys::resolve_api_key($name, opts.api_key)?;
                let mut p = Self::new_with_client(api_key, $crate::default_http_client());
                if let Some(m) = opts.model {
                    p = p.with_model(m);
                }
                if let Some(url) = opts.base_url {
                    p = p.with_base_url(url);
                }
                Ok(p)
            }
        }
    };
    ($provider:ty, $name:expr, no_base_url) => {
        #[cfg(any(
            all(target_arch = "wasm32", not(target_os = "wasi")),
            feature = "reqwest",
            target_os = "wasi"
        ))]
        impl $provider {
            /// Construct from typed [`ProviderOptions`](crate::types::provider_options::ProviderOptions).
            ///
            /// `base_url` is ignored — this provider's endpoint is fixed.
            ///
            /// # Errors
            ///
            /// Returns [`BlazenError::Auth`] if no API key is provided and
            /// the corresponding environment variable is not set.
            pub fn from_options(
                opts: $crate::types::provider_options::ProviderOptions,
            ) -> Result<Self, $crate::BlazenError> {
                let api_key = $crate::keys::resolve_api_key($name, opts.api_key)?;
                let mut p = Self::new_with_client(api_key, $crate::default_http_client());
                if let Some(m) = opts.model {
                    p = p.with_model(m);
                }
                Ok(p)
            }
        }
    };
}

pub(crate) use impl_simple_from_options;

#[cfg(test)]
mod tests {
    use super::{
        cap_body, extract_detail, format_provider_http_tail, pick_request_id, provider_http_error,
        provider_http_error_parts,
    };
    use crate::error::{BlazenError, PROVIDER_ERROR_BODY_CAP};
    use crate::http::HttpResponse;

    fn resp(status: u16, headers: Vec<(&str, &str)>, body: &[u8]) -> HttpResponse {
        HttpResponse {
            status,
            headers: headers
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
            body: body.to_vec(),
        }
    }

    #[test]
    fn extract_detail_from_detail_key() {
        assert_eq!(
            extract_detail(r#"{"detail":"rate limited"}"#),
            Some("rate limited".to_owned())
        );
    }

    #[test]
    fn extract_detail_from_error_string() {
        assert_eq!(
            extract_detail(r#"{"error":"bad thing"}"#),
            Some("bad thing".to_owned())
        );
    }

    #[test]
    fn extract_detail_from_nested_error_message() {
        assert_eq!(
            extract_detail(r#"{"error":{"message":"nested"}}"#),
            Some("nested".to_owned())
        );
    }

    #[test]
    fn extract_detail_from_top_level_message() {
        assert_eq!(extract_detail(r#"{"message":"y"}"#), Some("y".to_owned()));
    }

    #[test]
    fn extract_detail_returns_none_for_non_json() {
        assert_eq!(extract_detail("<html>Bad Gateway</html>"), None);
    }

    #[test]
    fn cap_body_preserves_short_body() {
        assert_eq!(cap_body("short"), "short");
    }

    #[test]
    fn cap_body_truncates_oversized() {
        let big = "A".repeat(10 * 1024);
        let capped = cap_body(&big);
        assert!(capped.starts_with(&"A".repeat(PROVIDER_ERROR_BODY_CAP)));
        assert!(capped.contains("[truncated 6144 bytes]"));
    }

    #[test]
    fn pick_request_id_fal_header() {
        let headers: Vec<(String, String)> =
            vec![("x-fal-request-id".to_owned(), "abc-123".to_owned())];
        assert_eq!(pick_request_id(&headers), Some("abc-123".to_owned()));
    }

    #[test]
    fn pick_request_id_generic_fallback() {
        let headers: Vec<(String, String)> = vec![("x-request-id".to_owned(), "xyz".to_owned())];
        assert_eq!(pick_request_id(&headers), Some("xyz".to_owned()));
    }

    #[test]
    fn provider_http_error_parses_detail() {
        let r = resp(503, vec![], br#"{"detail":"rate limited"}"#);
        let e = provider_http_error("fal", "https://fal.run/x", &r);
        let BlazenError::ProviderHttp(d) = e else {
            panic!("wrong variant: {e:?}");
        };
        assert_eq!(d.detail.as_deref(), Some("rate limited"));
        assert_eq!(d.status, 503);
        assert_eq!(d.provider.as_ref(), "fal");
    }

    #[test]
    fn provider_http_error_no_json_body() {
        let r = resp(502, vec![], b"<html>Bad Gateway</html>");
        let BlazenError::ProviderHttp(d) = provider_http_error("p", "u", &r) else {
            panic!("wrong variant")
        };
        assert!(d.detail.is_none());
        assert_eq!(d.raw_body, "<html>Bad Gateway</html>");
    }

    #[test]
    fn provider_http_error_display_contains_fields() {
        let r = resp(
            503,
            vec![("x-fal-request-id", "rid")],
            br#"{"detail":"overloaded"}"#,
        );
        let e = provider_http_error("fal", "https://fal.run/foo", &r);
        let s = e.to_string();
        assert!(s.contains("fal"), "missing provider: {s}");
        assert!(s.contains("503"), "missing status: {s}");
        assert!(s.contains("https://fal.run/foo"), "missing endpoint: {s}");
        assert!(s.contains("overloaded"), "missing detail: {s}");
        assert!(s.contains("rid"), "missing request-id: {s}");
    }

    #[test]
    fn provider_http_is_retryable_semantics() {
        let r500 = provider_http_error("p", "u", &resp(500, vec![], b"{}"));
        assert!(r500.is_retryable());
        let r429 = provider_http_error("p", "u", &resp(429, vec![], b"{}"));
        assert!(r429.is_retryable());
        let r400 = provider_http_error("p", "u", &resp(400, vec![], b"{}"));
        assert!(!r400.is_retryable());
        let r503 = provider_http_error("p", "u", &resp(503, vec![], b"{}"));
        assert!(r503.is_retryable());
    }

    #[test]
    fn provider_http_error_parts_empty_body() {
        let headers: Vec<(String, String)> =
            vec![("x-fal-request-id".to_owned(), "rid".to_owned())];
        let e = provider_http_error_parts("fal", "https://x", 502, &headers, "");
        let BlazenError::ProviderHttp(d) = e else {
            panic!("wrong variant")
        };
        assert_eq!(d.status, 502);
        assert_eq!(d.request_id.as_deref(), Some("rid"));
        assert!(d.detail.is_none());
        assert!(d.raw_body.is_empty());
    }

    #[test]
    fn provider_http_error_parses_retry_after() {
        let r = resp(429, vec![("retry-after", "5")], b"{}");
        let BlazenError::ProviderHttp(d) = provider_http_error("p", "u", &r) else {
            panic!("wrong variant")
        };
        assert_eq!(d.retry_after_ms, Some(5_000));
    }

    #[test]
    fn format_provider_http_tail_prefers_detail() {
        let s = format_provider_http_tail(Some("boom"), "raw", Some("rid"));
        assert_eq!(s, "boom (request-id=rid)");
    }

    #[test]
    fn format_provider_http_tail_falls_back_to_raw_body_truncated() {
        let raw = "A".repeat(300);
        let s = format_provider_http_tail(None, &raw, None);
        assert_eq!(s.len(), 200); // truncated at 200 chars
    }
}
