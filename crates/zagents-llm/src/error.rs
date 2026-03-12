//! Error types for the `ZAgents` LLM integration layer.

use std::time::Duration;

use thiserror::Error;

/// Errors produced by LLM provider operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The HTTP request to the provider API failed.
    #[error("API request failed: {0}")]
    RequestFailed(String),

    /// The provider returned a rate-limit response.
    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited {
        /// How long to wait before retrying, if the provider specified it.
        retry_after: Option<Duration>,
    },

    /// The provided API key was rejected by the provider.
    #[error("authentication failed")]
    AuthFailed,

    /// The requested model does not exist or is not accessible.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The provider returned a response with no content.
    #[error("no content in response")]
    NoContent,

    /// The provider response could not be parsed into the expected format.
    #[error("invalid response format: {0}")]
    InvalidResponse(String),

    /// Structured output JSON could not be deserialized into the target type.
    #[error("structured output parse failed: {0}")]
    ParseFailed(#[from] serde_json::Error),

    /// An HTTP transport error occurred.
    ///
    /// Uses `String` rather than `reqwest::Error` to avoid feature-gating the
    /// error enum itself.
    #[error("HTTP error: {0}")]
    Http(String),

    /// An error occurred while processing a streaming response.
    #[error("stream error: {0}")]
    Stream(String),
}

/// Convenience alias for `Result<T, LlmError>`.
pub type Result<T, E = LlmError> = std::result::Result<T, E>;
