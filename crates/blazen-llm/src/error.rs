//! Unified error types for all Blazen LLM and compute operations.

use std::time::Duration;

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
