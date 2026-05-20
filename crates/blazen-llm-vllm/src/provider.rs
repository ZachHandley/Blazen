//! [`VllmProvider`] — the public type that proxies inference and adapter
//! management to a vLLM server.
//!
//! Inherent methods (`complete`, `stream`, `load_adapter`,
//! `unload_adapter`, `list_adapters`) match the surface that the
//! `blazen-llm` bridge in `backends/vllm.rs` calls — keeping the trait
//! impls in the upstream crate (where the trait lives) and the wire
//! logic here.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;

use crate::VllmError;
use crate::client::{VllmClient, VllmModelEntry};
use crate::options::{VllmAdapterTransport, VllmOptions};

/// One `LoRA` adapter currently mounted on the upstream vLLM server,
/// tracked client-side so the bridge can answer `list_adapters` without
/// re-querying `/v1/models` for every call.
#[derive(Debug, Clone)]
pub struct MountedAdapter {
    /// vLLM's `lora_name` for the adapter — also the value clients put
    /// in the `model` field of `/v1/chat/completions` to route a request
    /// through this adapter.
    pub adapter_id: String,
    /// Path / hub-spec / sentinel describing where the adapter came from.
    /// Surfaced as [`blazen_llm::AdapterStatus::source_dir`] via the bridge.
    pub source_dir: std::path::PathBuf,
    /// Scale at mount time. vLLM does not accept a per-mount scale —
    /// the field is preserved for parity with other backends; the
    /// in-memory value is always `1.0`.
    pub scale: f32,
}

/// vLLM proxy provider.
///
/// Stateless on the wire — every call goes to the upstream server.
/// State held locally:
/// - the [`VllmOptions`] for headers, timeouts, transport mode,
/// - an `Arc<VllmClient>` (cheap clones share the connection pool),
/// - a `RwLock<Vec<MountedAdapter>>` mirroring the upstream adapter set
///   so `list_adapters` can return without a round-trip.
#[derive(Debug, Clone)]
pub struct VllmProvider {
    options: VllmOptions,
    client: Arc<VllmClient>,
    mounted_adapters: Arc<RwLock<Vec<MountedAdapter>>>,
}

impl VllmProvider {
    /// Build a provider from options. Validates fields and constructs
    /// the underlying HTTP client immediately so misconfiguration fails
    /// at startup, not on the first inference call.
    ///
    /// # Errors
    /// [`VllmError::InvalidOptions`] when a required field is missing,
    /// or [`VllmError::Init`] when the `reqwest::Client` cannot be built.
    pub fn from_options(options: VllmOptions) -> Result<Self, VllmError> {
        let client = VllmClient::new(&options)?;
        Ok(Self {
            options,
            client: Arc::new(client),
            mounted_adapters: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Borrow the stored options.
    #[must_use]
    pub fn options(&self) -> &VllmOptions {
        &self.options
    }

    /// Borrow the underlying HTTP client (escape hatch for raw calls).
    #[must_use]
    pub fn client(&self) -> Arc<VllmClient> {
        Arc::clone(&self.client)
    }

    /// Base model id the provider was constructed with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.options.model
    }

    // -----------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------

    /// Send a chat-completion request to vLLM. The body must already be
    /// shaped for vLLM's OpenAI-compat surface — the bridge in
    /// `blazen-llm/src/backends/vllm.rs` performs the typed translation.
    ///
    /// # Errors
    /// Propagates [`VllmError`] from the underlying client.
    pub async fn complete(&self, body: Value) -> Result<Value, VllmError> {
        self.client.chat_completions(&body).await
    }

    /// Streaming variant; returns the raw `reqwest::Response` so the
    /// bridge can drive an SSE parser.
    ///
    /// # Errors
    /// As [`Self::complete`].
    pub async fn stream(&self, body: Value) -> Result<reqwest::Response, VllmError> {
        self.client.chat_completions_stream(&body).await
    }

    // -----------------------------------------------------------------
    // Adapter management
    // -----------------------------------------------------------------

    /// Mount a `LoRA` adapter on the upstream vLLM server.
    ///
    /// The `path_or_dir` argument is the path the caller would pass
    /// via [`blazen_llm::LocalModel::load_adapter`]. The actual string
    /// sent to vLLM is resolved from
    /// [`VllmOptions::adapter_transport`]:
    ///
    /// - [`VllmAdapterTransport::LocalFs`] with a populated path — sent
    ///   as `lora_path`.
    /// - [`VllmAdapterTransport::LocalFs`] with an empty path — falls
    ///   through to `path_or_dir`.
    /// - [`VllmAdapterTransport::HfHub`] — sent as `lora_path:
    ///   "<repo>@<revision>"` (the resolver plugin parses this shape).
    /// - [`VllmAdapterTransport::HttpPush`] — rejected with
    ///   [`VllmError::UnsupportedTransport`] (vLLM has no upload API).
    ///
    /// # Errors
    /// - [`VllmError::AdapterFailed`] when the runtime-update env var
    ///   isn't set, the adapter is already mounted, or vLLM returns
    ///   non-2xx.
    /// - [`VllmError::UnsupportedTransport`] for `HttpPush`.
    pub async fn load_adapter(
        &self,
        adapter_id: impl Into<String>,
        path_or_dir: &std::path::Path,
    ) -> Result<MountedAdapter, VllmError> {
        let adapter_id = adapter_id.into();

        if !self.options.runtime_lora_updating {
            return Err(VllmError::AdapterFailed(
                "vLLM runtime LoRA updating is disabled; \
                 launch the server with VLLM_ALLOW_RUNTIME_LORA_UPDATING=true"
                    .into(),
            ));
        }

        {
            let guard = self.mounted_adapters.read().await;
            if guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(VllmError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is already mounted"
                )));
            }
        }

        let lora_path = match &self.options.adapter_transport {
            VllmAdapterTransport::LocalFs(p) if p.as_os_str().is_empty() => {
                path_or_dir.display().to_string()
            }
            VllmAdapterTransport::LocalFs(p) => p.display().to_string(),
            VllmAdapterTransport::HfHub { repo, revision } => match revision {
                Some(rev) => format!("{repo}@{rev}"),
                None => repo.clone(),
            },
            VllmAdapterTransport::HttpPush(_) => {
                return Err(VllmError::UnsupportedTransport(
                    "VllmAdapterTransport::HttpPush — vLLM has no upload endpoint. \
                     Stage the adapter on a path the server can read and use LocalFs."
                        .into(),
                ));
            }
        };

        self.client
            .load_lora_adapter(&adapter_id, &lora_path)
            .await?;

        let mounted = MountedAdapter {
            adapter_id: adapter_id.clone(),
            source_dir: path_or_dir.to_path_buf(),
            scale: 1.0,
        };

        self.mounted_adapters.write().await.push(mounted.clone());
        Ok(mounted)
    }

    /// Unmount a previously mounted adapter.
    ///
    /// # Errors
    /// [`VllmError::AdapterFailed`] when the adapter isn't tracked
    /// locally, or vLLM returns non-2xx.
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), VllmError> {
        {
            let guard = self.mounted_adapters.read().await;
            if !guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(VllmError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is not mounted"
                )));
            }
        }

        self.client.unload_lora_adapter(adapter_id).await?;

        let mut guard = self.mounted_adapters.write().await;
        guard.retain(|a| a.adapter_id != adapter_id);
        Ok(())
    }

    /// Locally-cached snapshot of mounted adapters. Use
    /// [`Self::refresh_adapters_from_server`] to reconcile with the
    /// upstream state (e.g. after another client mounted an adapter).
    pub async fn list_adapters(&self) -> Vec<MountedAdapter> {
        self.mounted_adapters.read().await.clone()
    }

    /// GET `/v1/models` and update the local mounted-adapter cache.
    /// Returns the full upstream listing so callers can introspect.
    ///
    /// # Errors
    /// Propagates [`VllmError`] from the underlying client.
    pub async fn refresh_adapters_from_server(&self) -> Result<Vec<VllmModelEntry>, VllmError> {
        let listing = self.client.list_models().await?;
        let base = self.options.model.as_str();

        let derived: Vec<MountedAdapter> = listing
            .iter()
            .filter(|row| row.parent.as_deref() == Some(base) && row.id != base)
            .map(|row| MountedAdapter {
                adapter_id: row.id.clone(),
                source_dir: std::path::PathBuf::new(),
                scale: 1.0,
            })
            .collect();

        *self.mounted_adapters.write().await = derived;
        Ok(listing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> VllmProvider {
        VllmProvider::from_options(VllmOptions::required("http://localhost:8000", "llama"))
            .expect("build provider")
    }

    #[test]
    fn from_options_rejects_empty_endpoint() {
        let opts = VllmOptions::required("", "llama");
        assert!(VllmProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_empty_model() {
        let opts = VllmOptions::required("http://h", "");
        assert!(VllmProvider::from_options(opts).is_err());
    }

    #[test]
    fn provider_exposes_model_id() {
        let p = make_provider();
        assert_eq!(p.model_id(), "llama");
    }

    #[tokio::test]
    async fn load_adapter_rejected_when_runtime_lora_disabled() {
        let opts = VllmOptions {
            runtime_lora_updating: false,
            ..VllmOptions::required("http://localhost:8000", "llama")
        };
        let p = VllmProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("a1", std::path::Path::new("/srv/loras/a1"))
            .await
            .expect_err("should reject when runtime updates are off");
        assert!(matches!(err, VllmError::AdapterFailed(_)));
    }

    #[tokio::test]
    async fn load_adapter_rejects_http_push_transport() {
        let opts = VllmOptions {
            adapter_transport: VllmAdapterTransport::HttpPush(vec![1, 2, 3]),
            ..VllmOptions::required("http://localhost:8000", "llama")
        };
        let p = VllmProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("a1", std::path::Path::new("/srv/loras/a1"))
            .await
            .expect_err("HttpPush should not be supported");
        assert!(matches!(err, VllmError::UnsupportedTransport(_)));
    }

    #[tokio::test]
    async fn unload_adapter_unknown_id_errors() {
        let p = make_provider();
        let err = p
            .unload_adapter("never-mounted")
            .await
            .expect_err("unknown adapter should error");
        assert!(matches!(err, VllmError::AdapterFailed(_)));
    }
}
