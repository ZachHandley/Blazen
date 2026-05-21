//! Configuration options for the LM Studio proxy backend.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How a [`crate::LmStudioProvider`] would *theoretically* hand adapters
/// off to the upstream LM Studio server.
///
/// LM Studio does not implement a runtime LoRA-adapter mount API — the
/// adapter must be baked into the GGUF file (a merged checkpoint such
/// as those produced by Blazen's `merge_lora_into_base` helper, then
/// converted to GGUF) and loaded as a full model via
/// `/api/v0/models/load`. This enum is therefore **reserved** for parity
/// with the other proxy backends and every variant is rejected by
/// [`crate::LmStudioProvider::load_adapter`] with
/// [`crate::LmStudioError::Unsupported`].
///
/// Mirrors `blazen_llm::AdapterTransport` so the proxy crate stays a
/// leaf — the `backends/lmstudio.rs` bridge converts between the two
/// shapes when it intercepts the call and returns the unsupported error.
#[derive(Debug, Clone)]
pub enum LmStudioAdapterTransport {
    /// Reserved. Adapter directory or `.gguf`/safetensors file is on
    /// the LM Studio server's filesystem at the given path. Rejected.
    LocalFs(PathBuf),

    /// Reserved. Adapter has been pushed to Hugging Face Hub.
    /// Rejected.
    HfHub {
        repo: String,
        revision: Option<String>,
    },

    /// Reserved. Adapter weights live in memory on the Blazen host.
    /// Rejected.
    HttpPush(Vec<u8>),
}

impl Default for LmStudioAdapterTransport {
    /// Defaults to `LocalFs("")` — sentinel meaning "the caller will
    /// pass the path explicitly on each `load_adapter` call".
    fn default() -> Self {
        Self::LocalFs(PathBuf::new())
    }
}

/// Options for constructing a [`crate::LmStudioProvider`].
///
/// `endpoint` and `model` are required. `model` is the model identifier
/// LM Studio uses (e.g. `"qwen2.5-7b-instruct-q4_k_m"`); per-request
/// model selection happens via the `OpenAI` `model` field at call time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioOptions {
    /// Base URL of the LM Studio server, with no trailing slash
    /// (e.g. `"http://localhost:1234"` — LM Studio's default port).
    pub endpoint: String,

    /// Optional bearer token. LM Studio does not enforce auth by
    /// default, but downstream reverse proxies (NGINX, Cloudflare
    /// Access, ...) often do. Sent as `Authorization: Bearer <api_key>`
    /// on every request when present.
    pub api_key: Option<String>,

    /// The default model identifier to use when a request omits one.
    pub model: String,

    /// Reserved adapter transport (LM Studio doesn't support runtime
    /// `LoRA` mounting — see [`LmStudioAdapterTransport`]). Not
    /// serialised via JSON because the `HttpPush` variant carries raw
    /// bytes.
    #[serde(skip)]
    pub adapter_transport: LmStudioAdapterTransport,

    /// Per-request timeout for `/v1/chat/completions`,
    /// `/v1/completions`, and `/v1/embeddings`. Defaults to 120 s —
    /// long enough for a cold-cache first token but short enough to
    /// fail loudly when the server is hung.
    #[serde(default = "default_request_timeout", with = "duration_secs_serde")]
    pub request_timeout: Duration,

    /// Per-request timeout for the model lifecycle endpoints
    /// (`/api/v0/models/load`, `/api/v0/models/unload`). Defaults to
    /// 600 s — loading a 70B GGUF from cold disk can be slow.
    #[serde(default = "default_load_timeout", with = "duration_secs_serde")]
    pub load_timeout: Duration,
}

const fn default_request_timeout() -> Duration {
    Duration::from_mins(2)
}

const fn default_load_timeout() -> Duration {
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

impl LmStudioOptions {
    /// Create options with only the required fields set.
    #[must_use]
    pub fn required(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            adapter_transport: LmStudioAdapterTransport::default(),
            request_timeout: default_request_timeout(),
            load_timeout: default_load_timeout(),
        }
    }

    /// Returns `Err(LmStudioError::InvalidOptions)` if any required
    /// field is missing.
    pub(crate) fn validate(&self) -> Result<(), crate::LmStudioError> {
        if self.endpoint.is_empty() {
            return Err(crate::LmStudioError::InvalidOptions(
                "endpoint must not be empty".into(),
            ));
        }
        if self.model.is_empty() {
            return Err(crate::LmStudioError::InvalidOptions(
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
        let o = LmStudioOptions::required("http://localhost:1234", "qwen2.5-7b-instruct");
        assert_eq!(o.endpoint, "http://localhost:1234");
        assert_eq!(o.model, "qwen2.5-7b-instruct");
        assert!(o.api_key.is_none());
        assert_eq!(o.request_timeout, Duration::from_mins(2));
        assert_eq!(o.load_timeout, Duration::from_mins(10));
    }

    #[test]
    fn endpoint_trimmed_strips_trailing_slash() {
        let o = LmStudioOptions::required("http://host/", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host");
    }

    #[test]
    fn endpoint_trimmed_leaves_path_intact() {
        let o = LmStudioOptions::required("http://host/api", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host/api");
    }

    #[test]
    fn validate_empty_endpoint_errors() {
        let o = LmStudioOptions::required("", "m");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_empty_model_errors() {
        let o = LmStudioOptions::required("http://h", "");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_ok_for_required() {
        let o = LmStudioOptions::required("http://h:1234", "qwen2.5-7b-instruct");
        assert!(o.validate().is_ok());
    }

    #[test]
    fn serde_roundtrip_omits_transport() {
        let o = LmStudioOptions {
            api_key: Some("sk-123".into()),
            ..LmStudioOptions::required("http://h", "qwen2.5-7b-instruct")
        };
        let j = serde_json::to_string(&o).unwrap();
        assert!(!j.contains("adapter_transport"));
        let parsed: LmStudioOptions = serde_json::from_str(&j).unwrap();
        assert_eq!(parsed.endpoint, o.endpoint);
        assert_eq!(parsed.api_key, o.api_key);
        assert_eq!(parsed.request_timeout, o.request_timeout);
    }

    #[test]
    fn serde_default_timeouts_when_missing() {
        let j = r#"{"endpoint":"http://h","model":"qwen2.5-7b-instruct"}"#;
        let parsed: LmStudioOptions = serde_json::from_str(j).unwrap();
        assert_eq!(parsed.request_timeout, Duration::from_mins(2));
        assert_eq!(parsed.load_timeout, Duration::from_mins(10));
    }
}
