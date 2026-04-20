//! Unified error types for all Blazen LLM and compute operations.

use std::time::Duration;

/// Maximum bytes retained on `BlazenError::ProviderHttp.raw_body`.
/// Anything larger is truncated; a " ... [truncated N bytes]" marker
/// is appended. Prevents a 20 MB HTML error page from turning
/// `BlazenError` into a memory-bloat vector.
pub const PROVIDER_ERROR_BODY_CAP: usize = 4 * 1024;

/// The unified error type for all Blazen LLM and compute operations.
#[derive(Debug, thiserror::Error)]
pub enum BlazenError {
    // ---- Shared (applies to completions, compute, media, tools) ----
    #[error("authentication failed: {message}")]
    Auth { message: String },

    #[error("rate limited{}", retry_after_ms.map(|ms| format!(": retry after {ms}ms")).unwrap_or_default())]
    RateLimit { retry_after_ms: Option<u64> },

    #[error("timed out after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    #[error("{provider} error: {message}")]
    Provider {
        provider: String,
        message: String,
        status_code: Option<u16>,
    },

    /// Upstream HTTP provider returned a non-success response.
    ///
    /// Populated for !2xx responses from real HTTP calls (fal, `OpenRouter`,
    /// `OpenAI`, Anthropic, Gemini, Azure, groq, etc). Do NOT use for:
    /// - Auth failures (use `Auth`)
    /// - Rate-limit where the provider sent a clean `Retry-After` and no body (use `RateLimit`)
    /// - Network/transport failures (keep `Request`)
    /// - Subclass/custom-provider dispatch (keep `Provider`)
    ///
    /// The payload is boxed to keep `BlazenError` under `clippy::result_large_err`'s
    /// 128-byte threshold; access via the `Box<ProviderHttpDetails>` tuple field.
    #[error("{} HTTP {} at {}: {}",
        _0.provider, _0.status, _0.endpoint,
        crate::providers::format_provider_http_tail(
            _0.detail.as_deref(), &_0.raw_body, _0.request_id.as_deref()
        )
    )]
    ProviderHttp(Box<ProviderHttpDetails>),

    #[error("invalid input: {message}")]
    Validation {
        field: Option<String>,
        message: String,
    },

    #[error("content policy violation: {message}")]
    ContentPolicy { message: String },

    #[error("unsupported: {message}")]
    Unsupported { message: String },

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("request failed: {message}")]
    Request {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // ---- LLM completion-specific ----
    #[error("completion error: {0}")]
    Completion(CompletionErrorKind),

    // ---- Compute job-specific ----
    #[error("compute error: {0}")]
    Compute(ComputeErrorKind),

    // ---- Media-specific ----
    #[error("media error: {0}")]
    Media(MediaErrorKind),

    // ---- Tool-specific ----
    #[error("tool error: {message}")]
    Tool {
        name: Option<String>,
        message: String,
    },
}

/// Structured payload for [`BlazenError::ProviderHttp`].
///
/// Carries enough HTTP context that a caller can decide retry/fail/report
/// without string-matching the error message. Boxed from the enum variant
/// to keep `BlazenError` under `clippy::result_large_err`'s 128-byte
/// threshold (~138 bytes inline -> 16 bytes boxed).
#[derive(Debug)]
pub struct ProviderHttpDetails {
    /// Provider identifier (e.g. `"fal"`, `"openrouter"`). `Cow` so that
    /// compile-time literals AND runtime `String`s (openai-compat wrappers)
    /// fit without leaking.
    pub provider: std::borrow::Cow<'static, str>,
    /// Full URL of the request that failed. Callers should keep PII out of
    /// URLs — we do not sanitize.
    pub endpoint: String,
    /// HTTP status code.
    pub status: u16,
    /// Provider-supplied request/trace id, looked up as
    /// `x-fal-request-id`, `x-request-id`, `request-id` in that order.
    pub request_id: Option<String>,
    /// Human-readable detail extracted from a JSON error body, if the
    /// body was JSON and matched one of the known shapes.
    pub detail: Option<String>,
    /// Raw response body, capped at `PROVIDER_ERROR_BODY_CAP` bytes.
    /// Longer bodies end with `" ... [truncated N bytes]"`.
    pub raw_body: String,
    /// Parsed from the `Retry-After` response header, when present.
    /// Populated on any status (not just 429) so callers can honor a
    /// `503 + Retry-After`.
    pub retry_after_ms: Option<u64>,
}

/// LLM completion-specific error variants.
#[derive(Debug, thiserror::Error)]
pub enum CompletionErrorKind {
    #[error("model returned no content")]
    NoContent,
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("stream error: {0}")]
    Stream(String),
}

/// Compute job-specific error variants.
#[derive(Debug, thiserror::Error)]
pub enum ComputeErrorKind {
    #[error("job failed: {message}")]
    JobFailed {
        message: String,
        error_type: Option<String>,
        retryable: bool,
    },
    #[error("job cancelled")]
    Cancelled,
    #[error("quota exceeded: {message}")]
    QuotaExceeded { message: String },
}

/// Media-specific error variants.
#[derive(Debug, thiserror::Error)]
pub enum MediaErrorKind {
    #[error("invalid media: {message}")]
    Invalid {
        media_type: Option<String>,
        message: String,
    },
    #[error("media too large: {size_bytes} bytes (max {max_bytes})")]
    TooLarge { size_bytes: u64, max_bytes: u64 },
}

impl BlazenError {
    /// Whether this error is likely transient and the request could be retried.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimit { .. } | Self::Timeout { .. } | Self::Request { .. } => true,
            Self::Provider { status_code, .. } => status_code.is_none_or(|code| code >= 500),
            Self::ProviderHttp(d) => d.status >= 500 || d.status == 429,
            Self::Compute(ComputeErrorKind::JobFailed { retryable, .. }) => *retryable,
            _ => false,
        }
    }

    // Convenience constructors

    pub fn auth(message: impl Into<String>) -> Self {
        Self::Auth {
            message: message.into(),
        }
    }

    #[must_use]
    pub fn timeout(elapsed_ms: u64) -> Self {
        Self::Timeout { elapsed_ms }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn timeout_from_duration(elapsed: Duration) -> Self {
        let ms = elapsed.as_millis();
        Self::Timeout {
            elapsed_ms: if ms > u128::from(u64::MAX) {
                u64::MAX
            } else {
                ms as u64
            },
        }
    }

    pub fn request(message: impl Into<String>) -> Self {
        Self::Request {
            message: message.into(),
            source: None,
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            status_code: None,
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn provider_http(
        provider: impl Into<std::borrow::Cow<'static, str>>,
        endpoint: impl Into<String>,
        status: u16,
        request_id: Option<String>,
        detail: Option<String>,
        raw_body: impl Into<String>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self::ProviderHttp(Box::new(ProviderHttpDetails {
            provider: provider.into(),
            endpoint: endpoint.into(),
            status,
            request_id,
            detail,
            raw_body: raw_body.into(),
            retry_after_ms,
        }))
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            field: None,
            message: message.into(),
        }
    }

    pub fn tool_error(message: impl Into<String>) -> Self {
        Self::Tool {
            name: None,
            message: message.into(),
        }
    }

    #[must_use]
    pub fn no_content() -> Self {
        Self::Completion(CompletionErrorKind::NoContent)
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::Completion(CompletionErrorKind::ModelNotFound(model.into()))
    }

    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::Completion(CompletionErrorKind::InvalidResponse(message.into()))
    }

    pub fn stream_error(message: impl Into<String>) -> Self {
        Self::Completion(CompletionErrorKind::Stream(message.into()))
    }

    pub fn job_failed(message: impl Into<String>) -> Self {
        Self::Compute(ComputeErrorKind::JobFailed {
            message: message.into(),
            error_type: None,
            retryable: false,
        })
    }

    #[must_use]
    pub fn cancelled() -> Self {
        Self::Compute(ComputeErrorKind::Cancelled)
    }
}

impl From<serde_json::Error> for BlazenError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

/// Backwards-compatible alias.
#[deprecated(note = "use BlazenError instead")]
pub type LlmError = BlazenError;

/// Backwards-compatible alias.
#[deprecated(note = "use BlazenError instead")]
pub type ComputeError = BlazenError;

/// Result type alias for Blazen operations.
pub type Result<T, E = BlazenError> = std::result::Result<T, E>;
