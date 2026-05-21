//! Configuration options for the TGI proxy backend.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How a [`crate::TgiProvider`] hands adapter information off to the
/// upstream TGI server.
///
/// TGI does **not** mount adapters at runtime. The server must be
/// launched with `--lora-adapters <id1> <id2> ...` (or the equivalent
/// `LORA_ADAPTERS` env var) and each request selects from that
/// preloaded set via the `adapter_id` field. This enum exists for
/// parity with `blazen_llm::AdapterTransport`; the
/// `backends/tgi.rs` bridge translates between the two and the
/// provider rejects transports that imply runtime mounting.
#[derive(Debug, Clone)]
pub enum TgiAdapterTransport {
    /// The adapter is already loaded on the TGI server (it was passed
    /// to `--lora-adapters` at startup). The path is informational only
    /// — surfaced in [`crate::ActiveAdapter::source_dir`] for parity
    /// with other backends. Default.
    LocalFs(PathBuf),

    /// The adapter is on Hugging Face Hub and was preloaded via
    /// `--lora-adapters <repo>` at server start. Repo is informational
    /// only; the server already resolved it. TGI accepts the bare
    /// `<org/repo>` id (no `hf://` prefix needed).
    HfHub {
        repo: String,
        revision: Option<String>,
    },

    /// Adapter weights live in memory on the Blazen host. TGI has no
    /// binary-upload endpoint, so this variant is **reserved** and
    /// rejected by [`crate::TgiProvider::load_adapter`] with
    /// [`crate::TgiError::Unsupported`]. Stage the adapter on a path
    /// the server can read (or push to HF Hub) and restart the server
    /// with `--lora-adapters <id>` to make it available.
    HttpPush(Vec<u8>),
}

impl Default for TgiAdapterTransport {
    /// Defaults to `LocalFs("")` — sentinel meaning "the caller will
    /// pass the path explicitly on each `load_adapter` call".
    fn default() -> Self {
        Self::LocalFs(PathBuf::new())
    }
}

/// Options for constructing a [`crate::TgiProvider`].
///
/// `endpoint` and `model` are required. `model` is the *base* model id
/// the TGI server was launched with (e.g. `"meta-llama/Llama-3.2-3B"`).
/// Per-request adapter selection happens by setting the `adapter_id`
/// field on each call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TgiOptions {
    /// Base URL of the TGI server, with no trailing slash
    /// (e.g. `"http://localhost:8080"` — TGI's default port).
    pub endpoint: String,

    /// Optional bearer token. TGI does not enforce auth by default but
    /// downstream reverse proxies often do. Sent as
    /// `Authorization: Bearer <api_key>` on every request when present.
    pub api_key: Option<String>,

    /// The base model identifier the TGI server was launched with.
    /// Used as the `OpenAI` `model` field when the request omits one.
    pub model: String,

    /// Transport mode for adapter references. Not serialised via JSON
    /// because [`TgiAdapterTransport::HttpPush`] carries raw bytes.
    #[serde(skip)]
    pub adapter_transport: TgiAdapterTransport,

    /// Per-request timeout for `/generate`, `/generate_stream`,
    /// `/v1/chat/completions`, `/v1/completions`. Defaults to 120 s —
    /// long enough for a cold-cache first token but short enough to
    /// fail loudly when the server is hung.
    #[serde(default = "default_request_timeout", with = "duration_secs_serde")]
    pub request_timeout: Duration,

    /// Per-request timeout for housekeeping (`/info`, `/v1/models`,
    /// `/metrics`). Defaults to 30 s.
    #[serde(default = "default_meta_timeout", with = "duration_secs_serde")]
    pub meta_timeout: Duration,
}

const fn default_request_timeout() -> Duration {
    Duration::from_mins(2)
}

const fn default_meta_timeout() -> Duration {
    Duration::from_secs(30)
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

impl TgiOptions {
    /// Create options with only the required fields set.
    #[must_use]
    pub fn required(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            adapter_transport: TgiAdapterTransport::default(),
            request_timeout: default_request_timeout(),
            meta_timeout: default_meta_timeout(),
        }
    }

    /// Returns `Err(TgiError::InvalidOptions)` if any required field
    /// is missing.
    pub(crate) fn validate(&self) -> Result<(), crate::TgiError> {
        if self.endpoint.is_empty() {
            return Err(crate::TgiError::InvalidOptions(
                "endpoint must not be empty".into(),
            ));
        }
        if self.model.is_empty() {
            return Err(crate::TgiError::InvalidOptions(
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
        let o = TgiOptions::required("http://localhost:8080", "meta-llama/Llama-3.2-3B");
        assert_eq!(o.endpoint, "http://localhost:8080");
        assert_eq!(o.model, "meta-llama/Llama-3.2-3B");
        assert!(o.api_key.is_none());
        assert_eq!(o.request_timeout, Duration::from_mins(2));
        assert_eq!(o.meta_timeout, Duration::from_secs(30));
    }

    #[test]
    fn endpoint_trimmed_strips_trailing_slash() {
        let o = TgiOptions::required("http://host/", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host");
    }

    #[test]
    fn endpoint_trimmed_leaves_path_intact() {
        let o = TgiOptions::required("http://host/api", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host/api");
    }

    #[test]
    fn validate_empty_endpoint_errors() {
        let o = TgiOptions::required("", "m");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_empty_model_errors() {
        let o = TgiOptions::required("http://h", "");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_ok_for_required() {
        let o = TgiOptions::required("http://h:8080", "m");
        assert!(o.validate().is_ok());
    }

    #[test]
    fn serde_roundtrip_omits_transport() {
        let o = TgiOptions {
            api_key: Some("sk-123".into()),
            ..TgiOptions::required("http://h", "llama")
        };
        let j = serde_json::to_string(&o).unwrap();
        assert!(!j.contains("adapter_transport"));
        let parsed: TgiOptions = serde_json::from_str(&j).unwrap();
        assert_eq!(parsed.endpoint, o.endpoint);
        assert_eq!(parsed.api_key, o.api_key);
        assert_eq!(parsed.request_timeout, o.request_timeout);
    }

    #[test]
    fn serde_default_timeouts_when_missing() {
        let j = r#"{"endpoint":"http://h","model":"llama"}"#;
        let parsed: TgiOptions = serde_json::from_str(j).unwrap();
        assert_eq!(parsed.request_timeout, Duration::from_mins(2));
        assert_eq!(parsed.meta_timeout, Duration::from_secs(30));
    }
}
