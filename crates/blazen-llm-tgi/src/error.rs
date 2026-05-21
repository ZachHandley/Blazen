//! Error type for the TGI proxy backend.
//!
//! Mirrors the shape of [`blazen_llm_vllm::VllmError`] /
//! [`blazen_llm_ollama::OllamaError`] so the upstream
//! `backends/tgi.rs` bridge can map cleanly into
//! [`blazen_llm::BlazenError::Provider`].

use std::fmt;

/// Failures emitted by [`crate::TgiClient`] and [`crate::TgiProvider`].
#[derive(Debug)]
pub enum TgiError {
    /// A required option was missing, empty, or contradictory.
    InvalidOptions(String),
    /// Building / configuring the underlying `reqwest` client failed.
    Init(String),
    /// The HTTP request itself failed (connection reset, DNS, timeout).
    /// This is the transport-level failure; non-2xx responses are
    /// surfaced as [`Self::Http`] / [`Self::Validation`] / [`Self::NotFound`]
    /// / [`Self::Overloaded`] instead.
    Request(String),
    /// TGI returned a non-success HTTP status that doesn't have a more
    /// specific variant. `status` is the code, `body` is the response
    /// body (capped during decoding upstream so memory-bloat is bounded).
    Http { status: u16, body: String },
    /// TGI returned HTTP 404 (model / endpoint not found). Split out from
    /// [`Self::Http`] so the bridge can map it to
    /// `BlazenError::ModelNotFound` rather than a generic provider error.
    NotFound(String),
    /// TGI returned HTTP 422 — request shape was rejected by the
    /// server-side validator (e.g. token count > `max_input_tokens`,
    /// invalid `adapter_id`, schema mismatch). Carries the upstream error
    /// payload so callers can show the upstream message verbatim.
    Validation(String),
    /// TGI returned HTTP 429 — queue is full (`--max-concurrent-requests`
    /// exceeded). The caller can retry with backoff.
    Overloaded(String),
    /// TGI accepted the request but the response body could not be
    /// parsed (unexpected schema, malformed JSON, SSE frame missing
    /// trailing newline, etc.).
    Decode(String),
    /// The provider attempted an unsupported adapter-management
    /// operation. TGI requires `--lora-adapters <id1> <id2> ...` at
    /// startup; runtime mount / unmount is not part of the HTTP surface,
    /// so [`crate::TgiProvider::load_adapter`] only accepts the
    /// already-loaded transport variants.
    Unsupported(String),
}

impl fmt::Display for TgiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "tgi invalid options: {msg}"),
            Self::Init(msg) => write!(f, "tgi init failed: {msg}"),
            Self::Request(msg) => write!(f, "tgi request failed: {msg}"),
            Self::Http { status, body } => write!(f, "tgi http {status}: {body}"),
            Self::NotFound(msg) => write!(f, "tgi not found: {msg}"),
            Self::Validation(msg) => write!(f, "tgi validation error (422): {msg}"),
            Self::Overloaded(msg) => write!(f, "tgi overloaded (429): {msg}"),
            Self::Decode(msg) => write!(f, "tgi response decode failed: {msg}"),
            Self::Unsupported(msg) => write!(f, "tgi unsupported: {msg}"),
        }
    }
}

impl std::error::Error for TgiError {}

impl From<reqwest::Error> for TgiError {
    fn from(e: reqwest::Error) -> Self {
        Self::Request(e.to_string())
    }
}

impl From<serde_json::Error> for TgiError {
    fn from(e: serde_json::Error) -> Self {
        Self::Decode(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_options() {
        let e = TgiError::InvalidOptions("endpoint must not be empty".into());
        assert!(e.to_string().contains("invalid options"));
    }

    #[test]
    fn display_http_includes_status() {
        let e = TgiError::Http {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = e.to_string();
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn display_validation_carries_payload() {
        let e = TgiError::Validation("inputs tokens + max_new_tokens > max_total_tokens".into());
        let s = e.to_string();
        assert!(s.contains("422"));
        assert!(s.contains("max_total_tokens"));
    }

    #[test]
    fn display_overloaded_marks_429() {
        let e = TgiError::Overloaded("Model is overloaded".into());
        assert!(e.to_string().contains("429"));
    }

    #[test]
    fn display_unsupported() {
        let e = TgiError::Unsupported("HttpPush transport".into());
        assert!(e.to_string().contains("unsupported"));
    }
}
