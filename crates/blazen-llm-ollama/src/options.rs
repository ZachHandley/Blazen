//! Configuration options for the Ollama proxy backend.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How a [`crate::OllamaProvider`] hands adapters off to the upstream
/// Ollama server.
///
/// Ollama mounts adapters via the Modelfile `ADAPTER` directive on a
/// derived model (created via `POST /api/create`). The directive expects
/// either a path the Ollama server can read or an `hf://...` reference
/// (the server will pull from Hugging Face). This enum mirrors
/// `blazen_llm::AdapterTransport` so the proxy crate stays a leaf — the
/// `backends/ollama.rs` bridge converts between the two shapes.
#[derive(Debug, Clone)]
pub enum OllamaAdapterTransport {
    /// Adapter directory or `.gguf`/safetensors file is reachable on the
    /// Ollama server's filesystem at the given path. Default for
    /// single-host / shared-PVC setups.
    LocalFs(PathBuf),

    /// Adapter has been pushed to Hugging Face Hub; Ollama pulls it
    /// itself via `hf://<repo>` when the Modelfile is created.
    HfHub {
        repo: String,
        revision: Option<String>,
    },

    /// Adapter weights live in memory on the Blazen host and would need
    /// to be uploaded to the Ollama server. Ollama has no first-class
    /// upload endpoint for adapter binaries (the `ADAPTER` directive
    /// reads server-local paths or `hf://` refs only), so this variant
    /// is **reserved** and rejected by
    /// [`crate::OllamaProvider::load_adapter`] with
    /// [`crate::OllamaError::Unsupported`]. Stage the adapter on a path
    /// the server can read and use [`Self::LocalFs`] instead.
    HttpPush(Vec<u8>),
}

impl Default for OllamaAdapterTransport {
    /// Defaults to `LocalFs("")` — sentinel meaning "the caller will
    /// pass the path explicitly on each `load_adapter` call".
    fn default() -> Self {
        Self::LocalFs(PathBuf::new())
    }
}

/// Options for constructing a [`crate::OllamaProvider`].
///
/// `endpoint` and `model` are required. `model` is the *base* model the
/// Ollama server has installed (e.g. `"llama3.2"`); per-request model
/// selection happens via the `model` field at call time. Derived models
/// (created by mounting an adapter via [`Self::adapter_transport`]) live
/// alongside the base in Ollama's local store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    /// Base URL of the Ollama server, with no trailing slash
    /// (e.g. `"http://localhost:11434"` — Ollama's default port).
    pub endpoint: String,

    /// Optional bearer token. Ollama does not enforce auth by default,
    /// but downstream reverse proxies (NGINX, Cloudflare Access, ...)
    /// often do. Sent as `Authorization: Bearer <api_key>` on every
    /// request when present.
    pub api_key: Option<String>,

    /// The base model identifier installed on the Ollama server
    /// (e.g. `"llama3.2"`, `"qwen2.5:7b"`).
    pub model: String,

    /// Transport mode for adapter directories. Not serialised via JSON
    /// because [`OllamaAdapterTransport::HttpPush`] carries raw bytes.
    #[serde(skip)]
    pub adapter_transport: OllamaAdapterTransport,

    /// Per-request timeout for `/api/generate` and `/api/chat`. Defaults
    /// to 120 s — long enough for a cold-cache first token but short
    /// enough to fail loudly when the server is hung.
    #[serde(default = "default_request_timeout", with = "duration_secs_serde")]
    pub request_timeout: Duration,

    /// Per-request timeout for `/api/create`, `/api/pull`, and
    /// `/api/delete`. Defaults to 600 s because pulling a large model
    /// from cold cache can be slow (multi-GB downloads happen on first
    /// reference).
    #[serde(default = "default_adapter_timeout", with = "duration_secs_serde")]
    pub adapter_timeout: Duration,
}

const fn default_request_timeout() -> Duration {
    Duration::from_mins(2)
}

const fn default_adapter_timeout() -> Duration {
    Duration::from_mins(10)
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

impl OllamaOptions {
    /// Create options with only the required fields set.
    #[must_use]
    pub fn required(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            adapter_transport: OllamaAdapterTransport::default(),
            request_timeout: default_request_timeout(),
            adapter_timeout: default_adapter_timeout(),
        }
    }

    /// Returns `Err(OllamaError::InvalidOptions)` if any required field
    /// is missing.
    pub(crate) fn validate(&self) -> Result<(), crate::OllamaError> {
        if self.endpoint.is_empty() {
            return Err(crate::OllamaError::InvalidOptions(
                "endpoint must not be empty".into(),
            ));
        }
        if self.model.is_empty() {
            return Err(crate::OllamaError::InvalidOptions(
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
        let o = OllamaOptions::required("http://localhost:11434", "llama3.2");
        assert_eq!(o.endpoint, "http://localhost:11434");
        assert_eq!(o.model, "llama3.2");
        assert!(o.api_key.is_none());
        assert_eq!(o.request_timeout, Duration::from_mins(2));
    }

    #[test]
    fn endpoint_trimmed_strips_trailing_slash() {
        let o = OllamaOptions::required("http://host/", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host");
    }

    #[test]
    fn endpoint_trimmed_leaves_path_intact() {
        let o = OllamaOptions::required("http://host/api", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host/api");
    }

    #[test]
    fn validate_empty_endpoint_errors() {
        let o = OllamaOptions::required("", "m");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_empty_model_errors() {
        let o = OllamaOptions::required("http://h", "");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_ok_for_required() {
        let o = OllamaOptions::required("http://h:11434", "llama3.2");
        assert!(o.validate().is_ok());
    }

    #[test]
    fn serde_roundtrip_omits_transport() {
        let o = OllamaOptions {
            api_key: Some("sk-123".into()),
            ..OllamaOptions::required("http://h", "llama3.2")
        };
        let j = serde_json::to_string(&o).unwrap();
        // adapter_transport is #[serde(skip)] so it should not appear.
        assert!(!j.contains("adapter_transport"));
        let parsed: OllamaOptions = serde_json::from_str(&j).unwrap();
        assert_eq!(parsed.endpoint, o.endpoint);
        assert_eq!(parsed.api_key, o.api_key);
        assert_eq!(parsed.request_timeout, o.request_timeout);
    }

    #[test]
    fn serde_default_timeouts_when_missing() {
        let j = r#"{"endpoint":"http://h","model":"llama3.2"}"#;
        let parsed: OllamaOptions = serde_json::from_str(j).unwrap();
        assert_eq!(parsed.request_timeout, Duration::from_mins(2));
        assert_eq!(parsed.adapter_timeout, Duration::from_mins(10));
    }
}
