//! Error type for the vLLM proxy backend.
//!
//! Mirrors the shape of [`blazen_llm_mistralrs::MistralRsError`] so the
//! upstream `backends/vllm.rs` bridge can map cleanly into
//! [`blazen_llm::BlazenError::Provider`].

use std::fmt;

/// Failures emitted by [`crate::VllmClient`] and [`crate::VllmProvider`].
#[derive(Debug)]
pub enum VllmError {
    /// A required option was missing, empty, or contradictory.
    InvalidOptions(String),
    /// Building / configuring the underlying `reqwest` client failed.
    Init(String),
    /// The HTTP request itself failed (connection reset, DNS, timeout).
    /// This is the transport-level failure; non-2xx responses are
    /// surfaced as [`Self::Http`] instead.
    Request(String),
    /// vLLM returned a non-success HTTP status. `status` is the code,
    /// `body` is the response body (capped during decoding upstream so
    /// memory-bloat is bounded).
    Http { status: u16, body: String },
    /// vLLM accepted the request but the response body could not be
    /// parsed (unexpected schema, malformed JSON, etc.).
    Decode(String),
    /// A `LoRA` adapter operation failed at the upstream
    /// (`/v1/load_lora_adapter` returned non-2xx, or
    /// `/v1/unload_lora_adapter` reported the adapter was not mounted).
    AdapterFailed(String),
    /// The provider attempted an unsupported transport mode
    /// (e.g. [`blazen_llm::AdapterTransport::HttpPush`] —
    /// vLLM has no first-class push API; the caller must place the
    /// adapter on a path the server can read).
    UnsupportedTransport(String),
}

impl fmt::Display for VllmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "vllm invalid options: {msg}"),
            Self::Init(msg) => write!(f, "vllm init failed: {msg}"),
            Self::Request(msg) => write!(f, "vllm request failed: {msg}"),
            Self::Http { status, body } => {
                write!(f, "vllm http {status}: {body}")
            }
            Self::Decode(msg) => write!(f, "vllm response decode failed: {msg}"),
            Self::AdapterFailed(msg) => write!(f, "vllm adapter operation failed: {msg}"),
            Self::UnsupportedTransport(msg) => {
                write!(f, "vllm adapter transport unsupported: {msg}")
            }
        }
    }
}

impl std::error::Error for VllmError {}

impl From<reqwest::Error> for VllmError {
    fn from(e: reqwest::Error) -> Self {
        Self::Request(e.to_string())
    }
}

impl From<serde_json::Error> for VllmError {
    fn from(e: serde_json::Error) -> Self {
        Self::Decode(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_options() {
        let e = VllmError::InvalidOptions("endpoint must not be empty".into());
        assert!(e.to_string().contains("invalid options"));
        assert!(e.to_string().contains("endpoint"));
    }

    #[test]
    fn display_http_includes_status() {
        let e = VllmError::Http {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = e.to_string();
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn display_adapter_failed() {
        let e = VllmError::AdapterFailed("VLLM_ALLOW_RUNTIME_LORA_UPDATING not set".into());
        assert!(e.to_string().contains("adapter"));
    }
}
