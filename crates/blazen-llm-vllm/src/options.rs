//! Configuration options for the vLLM proxy backend.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How a [`crate::VllmProvider`] hands adapters off to the upstream vLLM
/// server. This is a local copy of `blazen_llm::AdapterTransport` so the
/// proxy crate stays a leaf with no dependency on `blazen-llm` — the
/// `backends/vllm.rs` bridge converts between the two shapes.
#[derive(Debug, Clone)]
pub enum VllmAdapterTransport {
    /// Adapter directory is reachable on the vLLM server's filesystem
    /// at the given path. Default for single-host / shared-PVC setups.
    LocalFs(PathBuf),

    /// Adapter has been pushed to Hugging Face Hub; vLLM (with the
    /// `lora_huggingface_resolver` plugin enabled) pulls it itself
    /// when referenced by the `lora_name` field.
    HfHub {
        repo: String,
        revision: Option<String>,
    },

    /// Adapter weights live in memory on the Blazen host and would need
    /// to be uploaded to the vLLM server. vLLM has no first-class upload
    /// endpoint as of v0.10 (the runtime API reads from `lora_path` on
    /// the server's filesystem only), so this variant is **reserved**
    /// and currently rejected by [`crate::VllmProvider::load_adapter`]
    /// with [`crate::VllmError::UnsupportedTransport`]. Stand up a
    /// sidecar uploader and use [`Self::LocalFs`] instead.
    HttpPush(Vec<u8>),
}

impl Default for VllmAdapterTransport {
    /// Defaults to `LocalFs("")` — sentinel meaning "the caller will
    /// pass the path explicitly on each `load_adapter` call".
    fn default() -> Self {
        Self::LocalFs(PathBuf::new())
    }
}

/// Options for constructing a [`crate::VllmProvider`].
///
/// `endpoint` and `model` are required. `model` is the *base* model the
/// vLLM server was launched with (e.g.
/// `meta-llama/Llama-3.2-3B-Instruct`); per-request adapter selection
/// happens via the `OpenAI` `model` field at call time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmOptions {
    /// Base URL of the vLLM server, with no trailing slash
    /// (e.g. `"http://localhost:8000"`).
    pub endpoint: String,

    /// Optional bearer token. Required when vLLM was started with
    /// `--api-key <KEY>`. Sent as `Authorization: Bearer <api_key>` on
    /// every request.
    pub api_key: Option<String>,

    /// The base model identifier the vLLM server was launched with.
    pub model: String,

    /// Transport mode for adapter directories. Not serialised via JSON
    /// because [`VllmAdapterTransport::HttpPush`] carries raw bytes.
    #[serde(skip)]
    pub adapter_transport: VllmAdapterTransport,

    /// Whether the upstream vLLM server was launched with
    /// `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`. When `false`,
    /// [`crate::VllmProvider::load_adapter`] rejects the call with
    /// [`crate::VllmError::AdapterFailed`] up-front instead of waiting
    /// for the server to 4xx.
    #[serde(default = "default_runtime_lora")]
    pub runtime_lora_updating: bool,

    /// Per-request timeout for `/v1/chat/completions`. Defaults to
    /// 120 s — long enough for a cold-cache first token but short
    /// enough to fail loudly when the server is hung.
    #[serde(default = "default_request_timeout", with = "duration_secs_serde")]
    pub request_timeout: Duration,

    /// Per-request timeout for adapter management endpoints
    /// (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`). Defaults
    /// to 300 s because loading a large adapter for the first time
    /// from cold storage can be slow.
    #[serde(default = "default_adapter_timeout", with = "duration_secs_serde")]
    pub adapter_timeout: Duration,
}

const fn default_runtime_lora() -> bool {
    true
}

const fn default_request_timeout() -> Duration {
    Duration::from_mins(2)
}

const fn default_adapter_timeout() -> Duration {
    Duration::from_mins(5)
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

impl VllmOptions {
    /// Create options with only the required fields set.
    #[must_use]
    pub fn required(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            adapter_transport: VllmAdapterTransport::default(),
            runtime_lora_updating: default_runtime_lora(),
            request_timeout: default_request_timeout(),
            adapter_timeout: default_adapter_timeout(),
        }
    }

    /// Returns `Err(VllmError::InvalidOptions)` if any required field is
    /// missing.
    pub(crate) fn validate(&self) -> Result<(), crate::VllmError> {
        if self.endpoint.is_empty() {
            return Err(crate::VllmError::InvalidOptions(
                "endpoint must not be empty".into(),
            ));
        }
        if self.model.is_empty() {
            return Err(crate::VllmError::InvalidOptions(
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
        let o = VllmOptions::required("http://localhost:8000", "llama");
        assert_eq!(o.endpoint, "http://localhost:8000");
        assert_eq!(o.model, "llama");
        assert!(o.api_key.is_none());
        assert!(o.runtime_lora_updating);
        assert_eq!(o.request_timeout, Duration::from_mins(2));
    }

    #[test]
    fn endpoint_trimmed_strips_trailing_slash() {
        let o = VllmOptions::required("http://host/", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host");
    }

    #[test]
    fn endpoint_trimmed_leaves_path_intact() {
        let o = VllmOptions::required("http://host/api", "m");
        assert_eq!(o.endpoint_trimmed(), "http://host/api");
    }

    #[test]
    fn validate_empty_endpoint_errors() {
        let o = VllmOptions::required("", "m");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_empty_model_errors() {
        let o = VllmOptions::required("http://h", "");
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_ok_for_required() {
        let o = VllmOptions::required("http://h:8000", "llama");
        assert!(o.validate().is_ok());
    }

    #[test]
    fn serde_roundtrip_omits_transport() {
        let o = VllmOptions {
            api_key: Some("sk-123".into()),
            ..VllmOptions::required("http://h", "llama")
        };
        let j = serde_json::to_string(&o).unwrap();
        // adapter_transport is #[serde(skip)] so it should not appear.
        assert!(!j.contains("adapter_transport"));
        let parsed: VllmOptions = serde_json::from_str(&j).unwrap();
        assert_eq!(parsed.endpoint, o.endpoint);
        assert_eq!(parsed.api_key, o.api_key);
        assert_eq!(parsed.request_timeout, o.request_timeout);
    }

    #[test]
    fn serde_default_timeouts_when_missing() {
        let j = r#"{"endpoint":"http://h","model":"llama"}"#;
        let parsed: VllmOptions = serde_json::from_str(j).unwrap();
        assert_eq!(parsed.request_timeout, Duration::from_mins(2));
        assert_eq!(parsed.adapter_timeout, Duration::from_mins(5));
        assert!(parsed.runtime_lora_updating);
    }
}
