//! Configuration options for the `llama.cpp` HTTP-server proxy backend.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How a [`crate::LlamacppServerProvider`] addresses adapters on the
/// upstream `llama-server`.
///
/// `llama-server` cannot load adapter binaries over HTTP at runtime —
/// adapters must be preloaded on the CLI via `--lora <path>` (or
/// `--lora-scaled <path> <scale>`) at startup. At runtime the client
/// can only TOGGLE the active set / scales via `POST /lora-adapters`
/// against the integer adapter indices reported by `GET /lora-adapters`.
///
/// This enum mirrors `blazen_llm::AdapterTransport` so the proxy crate
/// stays a leaf — the `backends/llamacpp_server.rs` bridge converts
/// between the two shapes.
#[derive(Debug, Clone)]
pub enum LlamacppServerAdapterTransport {
    /// Adapter file is reachable on the `llama-server` host's filesystem
    /// at the given path AND was preloaded at startup. The path is
    /// preserved client-side for traceability; the wire toggle is by
    /// integer index, not path.
    LocalFs(PathBuf),

    /// Adapter file is identified by HF Hub repo + revision and was
    /// preloaded at startup. As above the wire toggle is by integer
    /// index — the spec is preserved client-side for traceability.
    HfHub {
        repo: String,
        revision: Option<String>,
    },

    /// Adapter weights live in memory on the Blazen host and would need
    /// to be uploaded to the `llama-server` process. `llama-server` has
    /// no first-class upload endpoint — adapters MUST be preloaded at
    /// CLI startup time. This variant is therefore **rejected** by
    /// [`crate::LlamacppServerProvider::load_adapter`] with
    /// [`crate::LlamacppServerError::Unsupported`]. Pre-stage the
    /// adapter and launch `llama-server` with `--lora <path>`, then use
    /// [`Self::LocalFs`].
    HttpPush(Vec<u8>),
}

impl Default for LlamacppServerAdapterTransport {
    /// Defaults to `LocalFs("")` — sentinel meaning "the caller will
    /// pass the path explicitly on each `load_adapter` call".
    fn default() -> Self {
        Self::LocalFs(PathBuf::new())
    }
}

/// Options for constructing a [`crate::LlamacppServerProvider`].
///
/// `endpoint` and `model` are required. `model` is the model identifier
/// the `llama-server` reports via `GET /v1/models` (defaults to the
/// basename of the `--model` path argument unless overridden via
/// `--alias`). Per-request model selection happens via the `model`
/// field at call time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamacppServerOptions {
    /// Base URL of the `llama-server`, with no trailing slash
    /// (e.g. `"http://localhost:8080"` — `llama-server`'s default port).
    pub endpoint: String,

    /// Optional bearer token. `llama-server` enforces auth only when
    /// launched with `--api-key <KEY>`. Sent as
    /// `Authorization: Bearer <api_key>` on every request when present.
    pub api_key: Option<String>,

    /// The model identifier `llama-server` reports for the loaded
    /// weights (e.g. `"llama-3.2"`, `"qwen2.5-7b-instruct"`). Used as
    /// the default for the `model` field in OpenAI-shaped requests.
    pub model: String,

    /// Transport mode for adapter directories. Not serialised via JSON
    /// because [`LlamacppServerAdapterTransport::HttpPush`] carries raw
    /// bytes.
    #[serde(skip)]
    pub adapter_transport: LlamacppServerAdapterTransport,

    /// Per-request timeout for `/v1/chat/completions`,
    /// `/v1/completions`, `/v1/embeddings`, and `/completion`.
    /// Defaults to 120 s — long enough for a cold-cache first token
    /// but short enough to fail loudly when the server is hung.
    #[serde(default = "default_request_timeout", with = "duration_secs_serde")]
    pub request_timeout: Duration,

    /// Per-request timeout for adapter management endpoints
    /// (`GET /lora-adapters`, `POST /lora-adapters`). Defaults to 60 s
    /// — switching the active adapter set is cheap (no weight reload)
    /// but pathological retries can still stack up under load.
    #[serde(default = "default_adapter_timeout", with = "duration_secs_serde")]
    pub adapter_timeout: Duration,
}

const fn default_request_timeout() -> Duration {
    Duration::from_mins(2)
}

const fn default_adapter_timeout() -> Duration {
    Duration::from_mins(1)
}

mod duration_secs_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}

impl LlamacppServerOptions {
    /// Create options with only the required fields set.
    #[must_use]
    pub fn required(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            adapter_transport: LlamacppServerAdapterTransport::default(),
            request_timeout: default_request_timeout(),
            adapter_timeout: default_adapter_timeout(),
        }
    }

    /// Returns `Err(LlamacppServerError::InvalidOptions)` if any
    /// required field is missing.
    pub(crate) fn validate(&self) -> Result<(), crate::LlamacppServerError> {
        if self.endpoint.is_empty() {
            return Err(crate::LlamacppServerError::InvalidOptions(
                "endpoint must not be empty".into(),
            ));
        }
        if self.model.is_empty() {
            return Err(crate::LlamacppServerError::InvalidOptions(
                "model must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Strip a trailing slash from the endpoint (defensive — the user
    /// may or may not include one).
    pub(crate) fn endpoint_trimmed(&self) -> &str {
        self.endpoint.trim_end_matches('/')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_sets_only_required_fields() {
        let o = LlamacppServerOptions::required("http://localhost:8080", "llama-3.2");
        assert_eq!(o.endpoint, "http://localhost:8080");
        assert_eq!(o.model, "llama-3.2");
        assert!(o.api_key.is_none());
        assert_eq!(o.request_timeout, Duration::from_mins(2));
        assert_eq!(o.adapter_timeout, Duration::from_mins(1));
    }

    #[test]
    fn endpoint_trimmed_strips_trailing_slash() {
        let o = LlamacppServerOptions::required("http://host/", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host");
    }

    #[test]
    fn endpoint_trimmed_leaves_path_intact() {
        let o = LlamacppServerOptions::required("http://host/api", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host/api");
    }

    #[test]
    fn validate_empty_endpoint_errors() {
        let o = LlamacppServerOptions::required("", "m");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_empty_model_errors() {
        let o = LlamacppServerOptions::required("http://h", "");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_ok_for_required() {
        let o = LlamacppServerOptions::required("http://h:8080", "llama");
        assert!(o.validate().is_ok());
    }

    #[test]
    fn serde_roundtrip_omits_transport() {
        let o = LlamacppServerOptions {
            api_key: Some("sk-123".into()),
            ..LlamacppServerOptions::required("http://h", "llama-3.2")
        };
        let j = serde_json::to_string(&o).unwrap();
        // adapter_transport is #[serde(skip)] so it should not appear.
        assert!(!j.contains("adapter_transport"));
        let parsed: LlamacppServerOptions = serde_json::from_str(&j).unwrap();
        assert_eq!(parsed.endpoint, o.endpoint);
        assert_eq!(parsed.api_key, o.api_key);
        assert_eq!(parsed.request_timeout, o.request_timeout);
    }

    #[test]
    fn serde_default_timeouts_when_missing() {
        let j = r#"{"endpoint":"http://h","model":"llama-3.2"}"#;
        let parsed: LlamacppServerOptions = serde_json::from_str(j).unwrap();
        assert_eq!(parsed.request_timeout, Duration::from_mins(2));
        assert_eq!(parsed.adapter_timeout, Duration::from_mins(1));
    }
}
