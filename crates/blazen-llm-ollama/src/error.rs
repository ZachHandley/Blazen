//! Error type for the Ollama proxy backend.
//!
//! Mirrors the shape of [`blazen_llm_vllm::VllmError`] so the upstream
//! `backends/ollama.rs` bridge can map cleanly into
//! [`blazen_llm::BlazenError::Provider`].

use std::fmt;

/// Failures emitted by [`crate::OllamaClient`] and [`crate::OllamaProvider`].
#[derive(Debug)]
pub enum OllamaError {
    /// A required option was missing, empty, or contradictory.
    InvalidOptions(String),
    /// Building / configuring the underlying `reqwest` client failed.
    Init(String),
    /// The HTTP request itself failed (connection reset, DNS, timeout).
    /// This is the transport-level failure; non-2xx responses are
    /// surfaced as [`Self::Http`] instead.
    Request(String),
    /// Ollama returned a non-success HTTP status. `status` is the code,
    /// `body` is the response body (capped during decoding upstream so
    /// memory-bloat is bounded).
    Http { status: u16, body: String },
    /// Ollama responded with HTTP 404 for a model lookup
    /// (`/api/show`, `/api/delete`, generate against a missing model).
    /// Split out from [`Self::Http`] so the bridge can map it to
    /// `BlazenError::ModelNotFound` rather than a generic provider error.
    NotFound(String),
    /// Ollama accepted the request but the response body could not be
    /// parsed (unexpected schema, malformed JSON, NDJSON frame missing
    /// trailing newline, etc.).
    Decode(String),
    /// An adapter operation (`/api/create` with an `ADAPTER` Modelfile,
    /// `/api/pull` for an HF-Hub-backed adapter) failed at the upstream.
    AdapterFailed(String),
    /// The provider attempted an unsupported transport mode (e.g.
    /// [`blazen_llm::AdapterTransport::HttpPush`] — Ollama has no
    /// first-class binary-upload endpoint for adapters; the caller must
    /// stage the adapter on a path the server can read or push it to
    /// Hugging Face Hub first).
    Unsupported(String),
}

impl fmt::Display for OllamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "ollama invalid options: {msg}"),
            Self::Init(msg) => write!(f, "ollama init failed: {msg}"),
            Self::Request(msg) => write!(f, "ollama request failed: {msg}"),
            Self::Http { status, body } => write!(f, "ollama http {status}: {body}"),
            Self::NotFound(msg) => write!(f, "ollama not found: {msg}"),
            Self::Decode(msg) => write!(f, "ollama response decode failed: {msg}"),
            Self::AdapterFailed(msg) => write!(f, "ollama adapter operation failed: {msg}"),
            Self::Unsupported(msg) => write!(f, "ollama unsupported: {msg}"),
        }
    }
}

impl std::error::Error for OllamaError {}

impl From<reqwest::Error> for OllamaError {
    fn from(e: reqwest::Error) -> Self {
        Self::Request(e.to_string())
    }
}

impl From<serde_json::Error> for OllamaError {
    fn from(e: serde_json::Error) -> Self {
        Self::Decode(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_options() {
        let e = OllamaError::InvalidOptions("endpoint must not be empty".into());
        assert!(e.to_string().contains("invalid options"));
        assert!(e.to_string().contains("endpoint"));
    }

    #[test]
    fn display_http_includes_status() {
        let e = OllamaError::Http {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = e.to_string();
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn display_not_found() {
        let e = OllamaError::NotFound("model 'foo' was not found".into());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn display_adapter_failed() {
        let e = OllamaError::AdapterFailed("ADAPTER directive rejected".into());
        assert!(e.to_string().contains("adapter"));
    }
}
