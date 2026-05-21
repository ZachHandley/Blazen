//! Error type for the `llama.cpp` HTTP-server proxy backend.
//!
//! Mirrors the shape of [`blazen_llm_vllm::VllmError`] and
//! [`blazen_llm_ollama::OllamaError`] so the upstream
//! `backends/llamacpp_server.rs` bridge can map cleanly into
//! [`blazen_llm::BlazenError::Provider`] / `BlazenError::Unsupported`.

use std::fmt;

/// Failures emitted by [`crate::LlamacppServerClient`] and
/// [`crate::LlamacppServerProvider`].
#[derive(Debug)]
pub enum LlamacppServerError {
    /// A required option was missing, empty, or contradictory.
    InvalidOptions(String),
    /// Building / configuring the underlying `reqwest` client failed.
    Init(String),
    /// The HTTP request itself failed (connection reset, DNS, timeout).
    /// This is the transport-level failure; non-2xx responses are
    /// surfaced as [`Self::Http`] instead.
    Request(String),
    /// `llama-server` returned a non-success HTTP status. `status` is the
    /// code, `body` is the response body (capped during decoding upstream
    /// so memory-bloat is bounded).
    Http { status: u16, body: String },
    /// `llama-server` responded with HTTP 404 for a model / adapter lookup.
    /// Split out from [`Self::Http`] so the bridge can map it to
    /// `BlazenError::ModelNotFound` rather than a generic provider error.
    NotFound(String),
    /// `llama-server` accepted the request but the response body could
    /// not be parsed (unexpected schema, malformed JSON, SSE frame
    /// missing the `data:` prefix, etc.).
    Decode(String),
    /// An adapter operation (`GET /lora-adapters`, `POST /lora-adapters`)
    /// failed at the upstream.
    AdapterFailed(String),
    /// The provider attempted an unsupported transport mode (e.g.
    /// [`blazen_llm::AdapterTransport::HttpPush`] — `llama-server` has no
    /// binary-upload endpoint; adapters must be preloaded at startup via
    /// `--lora <path>` on the `llama-server` CLI).
    Unsupported(String),
}

impl fmt::Display for LlamacppServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "llamacpp-server invalid options: {msg}"),
            Self::Init(msg) => write!(f, "llamacpp-server init failed: {msg}"),
            Self::Request(msg) => write!(f, "llamacpp-server request failed: {msg}"),
            Self::Http { status, body } => write!(f, "llamacpp-server http {status}: {body}"),
            Self::NotFound(msg) => write!(f, "llamacpp-server not found: {msg}"),
            Self::Decode(msg) => write!(f, "llamacpp-server response decode failed: {msg}"),
            Self::AdapterFailed(msg) => {
                write!(f, "llamacpp-server adapter operation failed: {msg}")
            }
            Self::Unsupported(msg) => write!(f, "llamacpp-server unsupported: {msg}"),
        }
    }
}

impl std::error::Error for LlamacppServerError {}

impl From<reqwest::Error> for LlamacppServerError {
    fn from(e: reqwest::Error) -> Self {
        Self::Request(e.to_string())
    }
}

impl From<serde_json::Error> for LlamacppServerError {
    fn from(e: serde_json::Error) -> Self {
        Self::Decode(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_options() {
        let e = LlamacppServerError::InvalidOptions("endpoint must not be empty".into());
        assert!(e.to_string().contains("invalid options"));
        assert!(e.to_string().contains("endpoint"));
    }

    #[test]
    fn display_http_includes_status() {
        let e = LlamacppServerError::Http {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = e.to_string();
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn display_not_found() {
        let e = LlamacppServerError::NotFound("model 'foo' was not found".into());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn display_adapter_failed() {
        let e = LlamacppServerError::AdapterFailed("adapter 0 is not preloaded".into());
        assert!(e.to_string().contains("adapter"));
    }

    #[test]
    fn display_unsupported() {
        let e = LlamacppServerError::Unsupported("HttpPush rejected".into());
        assert!(e.to_string().contains("unsupported"));
    }
}
