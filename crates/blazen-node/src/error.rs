//! Error conversion utilities for napi-rs.
//!
//! Converts internal `Blazen` errors into [`napi::Error`] for the Node.js side.
//!
//! ## Provider error sentinel protocol
//!
//! napi-rs 3's [`napi::Error`] cannot carry arbitrary typed fields. For
//! [`BlazenError::Provider`] and [`BlazenError::ProviderHttp`] we embed a
//! JSON payload in the error message, prefixed with
//! [`PROVIDER_ERROR_SENTINEL`]. A hand-written JS wrapper
//! (`crates/blazen-node/errors.js`) detects the sentinel and re-throws a
//! typed `ProviderError` with `.provider`, `.status`, `.endpoint`,
//! `.requestId`, `.detail`, `.retryAfterMs` attributes.
//!
//! Raw message format:
//!
//! ```text
//! __BLAZEN_PROVIDER_ERROR__ {"provider":"fal","status":503,...}
//! [ProviderError] fal HTTP 503 at https://fal.run/x: service unavailable (request-id=abc)
//! ```
//!
//! Consumers who don't use the wrapper still get a readable message at
//! the end (minus the sentinel line).

use napi::Status;
use serde::Serialize;

/// Sentinel prefix on provider-error messages. The JS wrapper at
/// `crates/blazen-node/errors.js` pattern-matches on this. Keep in sync.
pub const PROVIDER_ERROR_SENTINEL: &str = "__BLAZEN_PROVIDER_ERROR__";

/// Structured payload embedded in a provider-error message's JSON line.
/// Field names use camelCase to match the receiving JS convention.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProviderErrorPayload<'a> {
    provider: &'a str,
    status: Option<u16>,
    endpoint: Option<&'a str>,
    request_id: Option<&'a str>,
    detail: Option<&'a str>,
    retry_after_ms: Option<u64>,
    // raw_body intentionally omitted — 4 KiB of JSON in an error message
    // is noisy. JS consumers who need it can re-inspect the Rust error.
}

/// Convert any `Display`-able error into a [`napi::Error`].
pub fn to_napi_error(err: impl std::fmt::Display) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`WorkflowError`](blazen_core::WorkflowError) into a [`napi::Error`].
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn workflow_error_to_napi(err: blazen_core::WorkflowError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`BlazenError`](blazen_llm::BlazenError) into a [`napi::Error`].
///
/// For `Provider` / `ProviderHttp` variants, embeds a JSON payload with
/// structured fields (see module docs). For other variants, prefixes the
/// message with the error class name for readable JS logs.
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn blazen_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    use blazen_llm::BlazenError;

    // Fast path — provider errors get a structured sentinel payload.
    if let BlazenError::ProviderHttp(d) = &err {
        let payload = ProviderErrorPayload {
            provider: d.provider.as_ref(),
            status: Some(d.status),
            endpoint: Some(d.endpoint.as_str()),
            request_id: d.request_id.as_deref(),
            detail: d.detail.as_deref(),
            retry_after_ms: d.retry_after_ms,
        };
        let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
        let readable = err.to_string();
        return napi::Error::new(
            Status::GenericFailure,
            format!("{PROVIDER_ERROR_SENTINEL} {json}\n[ProviderError] {readable}"),
        );
    }

    if let BlazenError::Provider {
        provider,
        status_code,
        ..
    } = &err
    {
        let payload = ProviderErrorPayload {
            provider: provider.as_str(),
            status: *status_code,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        };
        let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
        let readable = err.to_string();
        return napi::Error::new(
            Status::GenericFailure,
            format!("{PROVIDER_ERROR_SENTINEL} {json}\n[ProviderError] {readable}"),
        );
    }

    let prefix = match &err {
        BlazenError::Auth { .. } => "AuthError",
        BlazenError::RateLimit { .. } => "RateLimitError",
        BlazenError::Timeout { .. } => "TimeoutError",
        BlazenError::Validation { .. } => "ValidationError",
        BlazenError::ContentPolicy { .. } => "ContentPolicyError",
        BlazenError::Unsupported { .. } => "UnsupportedError",
        _ => "BlazenError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Backwards-compatible alias for [`blazen_error_to_napi`].
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llm_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    blazen_error_to_napi(err)
}
