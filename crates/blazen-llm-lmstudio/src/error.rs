//! Error type for the LM Studio proxy backend.
//!
//! Mirrors the shape of [`blazen_llm_vllm::VllmError`] and
//! [`blazen_llm_ollama::OllamaError`] so the upstream
//! `backends/lmstudio.rs` bridge can map cleanly into
//! [`blazen_llm::BlazenError::Provider`].

use std::fmt;

/// Failures emitted by [`crate::LmStudioClient`] and
/// [`crate::LmStudioProvider`].
#[derive(Debug)]
pub enum LmStudioError {
    /// A required option was missing, empty, or contradictory.
    InvalidOptions(String),
    /// Building / configuring the underlying `reqwest` client failed.
    Init(String),
    /// The HTTP request itself failed (connection reset, DNS, timeout).
    /// This is the transport-level failure; non-2xx responses are
    /// surfaced as [`Self::Http`] instead.
    Request(String),
    /// LM Studio returned a non-success HTTP status. `status` is the
    /// code, `body` is the response body (capped during decoding
    /// upstream so memory-bloat is bounded).
    Http { status: u16, body: String },
    /// LM Studio responded with HTTP 404 for a model lookup
    /// (`/api/v0/models/load`, `/api/v0/models/unload`, or a chat call
    /// against a model that isn't installed). Split out from
    /// [`Self::Http`] so the bridge can map it to
    /// `BlazenError::ModelNotFound` rather than a generic provider error.
    NotFound(String),
    /// LM Studio responded with HTTP 409 / 422 indicating that no model
    /// is currently loaded for the requested operation (e.g. chat
    /// completion before any model has been loaded). Split out so the
    /// bridge can produce a clear actionable message.
    NoModelLoaded(String),
    /// LM Studio accepted the request but the response body could not
    /// be parsed (unexpected schema, malformed JSON, etc.).
    Decode(String),
    /// A model lifecycle operation (`/api/v0/models/load`,
    /// `/api/v0/models/unload`) failed at the upstream.
    LoadFailed(String),
    /// The caller requested an operation LM Studio does not implement.
    /// The canonical case is [`crate::LmStudioProvider::load_adapter`]
    /// — LM Studio requires `LoRA` adapters to be pre-merged into the
    /// GGUF base model; there is no runtime mount/unmount endpoint.
    Unsupported(String),
}

impl fmt::Display for LmStudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "lmstudio invalid options: {msg}"),
            Self::Init(msg) => write!(f, "lmstudio init failed: {msg}"),
            Self::Request(msg) => write!(f, "lmstudio request failed: {msg}"),
            Self::Http { status, body } => write!(f, "lmstudio http {status}: {body}"),
            Self::NotFound(msg) => write!(f, "lmstudio not found: {msg}"),
            Self::NoModelLoaded(msg) => write!(f, "lmstudio no model loaded: {msg}"),
            Self::Decode(msg) => write!(f, "lmstudio response decode failed: {msg}"),
            Self::LoadFailed(msg) => write!(f, "lmstudio model load operation failed: {msg}"),
            Self::Unsupported(msg) => write!(f, "lmstudio unsupported: {msg}"),
        }
    }
}

impl std::error::Error for LmStudioError {}

impl From<reqwest::Error> for LmStudioError {
    fn from(e: reqwest::Error) -> Self {
        Self::Request(e.to_string())
    }
}

impl From<serde_json::Error> for LmStudioError {
    fn from(e: serde_json::Error) -> Self {
        Self::Decode(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_options() {
        let e = LmStudioError::InvalidOptions("endpoint must not be empty".into());
        assert!(e.to_string().contains("invalid options"));
        assert!(e.to_string().contains("endpoint"));
    }

    #[test]
    fn display_http_includes_status() {
        let e = LmStudioError::Http {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = e.to_string();
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn display_not_found() {
        let e = LmStudioError::NotFound("model 'foo' was not found".into());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn display_no_model_loaded() {
        let e = LmStudioError::NoModelLoaded("no model loaded".into());
        assert!(e.to_string().contains("no model loaded"));
    }

    #[test]
    fn display_load_failed() {
        let e = LmStudioError::LoadFailed("disk full".into());
        assert!(e.to_string().contains("load"));
    }

    #[test]
    fn display_unsupported_mentions_unsupported() {
        let e = LmStudioError::Unsupported("LoRA adapters not supported".into());
        assert!(e.to_string().contains("unsupported"));
    }
}
