//! [`LlamacppServerProvider`] — the public type that proxies inference
//! and adapter management to a `llama-server` process.
//!
//! Inherent methods (`chat_completions`, `completions`, `embeddings`,
//! `completion`, `load_adapter`, `unload_adapter`, `list_adapters`, ...)
//! match the surface the `blazen-llm` bridge in
//! `backends/llamacpp_server.rs` calls — keeping the trait impls in the
//! upstream crate (where the trait lives) and the wire logic here.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;

use crate::LlamacppServerError;
use crate::client::{
    LlamacppServerClient, LlamacppServerHealth, LlamacppServerLoraAdapter,
    LlamacppServerLoraToggle, LlamacppServerModelEntry, LlamacppServerSlot,
};
use crate::options::{LlamacppServerAdapterTransport, LlamacppServerOptions};

/// One `LoRA` adapter currently activated on the upstream `llama-server`,
/// tracked client-side so the bridge can answer `list_adapters` without
/// re-querying `GET /lora-adapters` on every call.
///
/// `llama-server` adapters are addressed by integer id at the wire
/// level; the human-readable `adapter_id` here is the per-mount label
/// the bridge / caller picks (and we resolve it to an id by matching
/// the source path against the rows from `GET /lora-adapters`).
#[derive(Debug, Clone)]
pub struct MountedAdapter {
    /// Caller-facing label for the adapter. Used by `unload_adapter` to
    /// look the mount back up in the local cache.
    pub adapter_id: String,
    /// Integer id reported by `GET /lora-adapters` — what
    /// `POST /lora-adapters` actually toggles.
    pub server_index: u32,
    /// Filesystem path the adapter was loaded from (server-side path).
    /// Surfaced as [`blazen_llm::AdapterStatus::source_dir`] via the bridge.
    pub source_dir: std::path::PathBuf,
    /// Scale currently set on the server (`1.0` for a freshly mounted
    /// adapter, `0.0` after `unload_adapter`).
    pub scale: f32,
}

/// `llama-server` proxy provider.
///
/// Stateless on the wire — every call goes to the upstream server.
/// State held locally:
/// - the [`LlamacppServerOptions`] for headers, timeouts, transport
///   mode,
/// - an `Arc<LlamacppServerClient>` (cheap clones share the connection
///   pool),
/// - a `RwLock<Vec<MountedAdapter>>` mirroring the upstream
///   active-adapter set so `list_adapters` can return without a
///   round-trip.
#[derive(Debug, Clone)]
pub struct LlamacppServerProvider {
    options: LlamacppServerOptions,
    client: Arc<LlamacppServerClient>,
    mounted_adapters: Arc<RwLock<Vec<MountedAdapter>>>,
}

impl LlamacppServerProvider {
    /// Build a provider from options. Validates fields and constructs
    /// the underlying HTTP client immediately so misconfiguration fails
    /// at startup, not on the first inference call.
    ///
    /// # Errors
    /// [`LlamacppServerError::InvalidOptions`] when a required field is
    /// missing, or [`LlamacppServerError::Init`] when the
    /// `reqwest::Client` cannot be built.
    pub fn from_options(options: LlamacppServerOptions) -> Result<Self, LlamacppServerError> {
        let client = LlamacppServerClient::new(&options)?;
        Ok(Self {
            options,
            client: Arc::new(client),
            mounted_adapters: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Borrow the stored options.
    #[must_use]
    pub fn options(&self) -> &LlamacppServerOptions {
        &self.options
    }

    /// Borrow the underlying HTTP client (escape hatch for raw calls).
    #[must_use]
    pub fn client(&self) -> Arc<LlamacppServerClient> {
        Arc::clone(&self.client)
    }

    /// Default model id the provider was constructed with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.options.model
    }

    // -----------------------------------------------------------------
    // Inference — OpenAI-compatible surface
    // -----------------------------------------------------------------

    /// Send a `/v1/chat/completions` request to `llama-server`.
    ///
    /// # Errors
    /// Propagates [`LlamacppServerError`] from the underlying client.
    pub async fn chat_completions(&self, body: Value) -> Result<Value, LlamacppServerError> {
        self.client.chat_completions(&body).await
    }

    /// Streaming variant of [`Self::chat_completions`]; returns the raw
    /// `reqwest::Response` so the bridge can drive an SSE parser.
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn chat_completions_stream(
        &self,
        body: Value,
    ) -> Result<reqwest::Response, LlamacppServerError> {
        self.client.chat_completions_stream(&body).await
    }

    /// Send a `/v1/completions` request.
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn completions(&self, body: Value) -> Result<Value, LlamacppServerError> {
        self.client.completions(&body).await
    }

    /// Send a `/v1/embeddings` request.
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn embeddings(&self, body: Value) -> Result<Value, LlamacppServerError> {
        self.client.embeddings(&body).await
    }

    // -----------------------------------------------------------------
    // Inference — llama.cpp-native surface
    // -----------------------------------------------------------------

    /// Send a `/completion` request (llama.cpp's native shape, NOT
    /// OpenAI-compat).
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn completion(&self, body: Value) -> Result<Value, LlamacppServerError> {
        self.client.completion(&body).await
    }

    // -----------------------------------------------------------------
    // Server introspection
    // -----------------------------------------------------------------

    /// GET `/v1/models`.
    ///
    /// # Errors
    /// Propagates [`LlamacppServerError`] from the underlying client.
    pub async fn models(&self) -> Result<Vec<LlamacppServerModelEntry>, LlamacppServerError> {
        self.client.list_models().await
    }

    /// GET `/health`.
    ///
    /// # Errors
    /// Propagates [`LlamacppServerError`] from the underlying client.
    pub async fn health(&self) -> Result<LlamacppServerHealth, LlamacppServerError> {
        self.client.health().await
    }

    /// GET `/slots`.
    ///
    /// # Errors
    /// Propagates [`LlamacppServerError`] from the underlying client.
    pub async fn slots(&self) -> Result<Vec<LlamacppServerSlot>, LlamacppServerError> {
        self.client.slots().await
    }

    // -----------------------------------------------------------------
    // Adapter management
    // -----------------------------------------------------------------

    /// Activate a preloaded `LoRA` adapter on the upstream `llama-server`.
    ///
    /// Strategy (this is the headline distinction from
    /// [`blazen_llm_vllm`] and [`blazen_llm_ollama`]):
    ///
    /// 1. `llama-server` cannot accept adapter binaries over HTTP —
    ///    adapters MUST be preloaded at startup via `--lora <path>` /
    ///    `--lora-scaled <path> <scale>`.
    /// 2. At runtime the only knob is `POST /lora-adapters` which
    ///    toggles which preloaded adapters are active and at what
    ///    scale. The wire protocol identifies adapters by their
    ///    integer index in CLI declaration order.
    /// 3. This method calls `GET /lora-adapters` to discover the
    ///    preloaded index, resolves `path_or_dir` (or the path encoded
    ///    in [`LlamacppServerOptions::adapter_transport`]) to one of
    ///    the rows by suffix-matching the server-reported `path`, then
    ///    POSTs the merged active-set (existing active adapters PLUS
    ///    this one at `scale: 1.0`) to `/lora-adapters`.
    ///
    /// Transport handling:
    /// - [`LlamacppServerAdapterTransport::LocalFs`] (populated) — use
    ///   the configured path as the match key (overrides
    ///   `path_or_dir`).
    /// - [`LlamacppServerAdapterTransport::LocalFs`] (empty) — fall
    ///   through to `path_or_dir`.
    /// - [`LlamacppServerAdapterTransport::HfHub`] — rejected with
    ///   [`LlamacppServerError::Unsupported`]; `llama-server` cannot
    ///   pull from HF Hub. Pre-cache the adapter and use `LocalFs`.
    /// - [`LlamacppServerAdapterTransport::HttpPush`] — rejected with
    ///   [`LlamacppServerError::Unsupported`]; `llama-server` cannot
    ///   accept binaries over HTTP at all.
    ///
    /// # Errors
    /// - [`LlamacppServerError::AdapterFailed`] when the adapter is
    ///   already mounted client-side, no preloaded row matches the
    ///   resolved path, or `POST /lora-adapters` returns non-2xx.
    /// - [`LlamacppServerError::Unsupported`] for `HfHub` /
    ///   `HttpPush`.
    pub async fn load_adapter(
        &self,
        adapter_id: impl Into<String>,
        path_or_dir: &std::path::Path,
    ) -> Result<MountedAdapter, LlamacppServerError> {
        let adapter_id = adapter_id.into();

        {
            let guard = self.mounted_adapters.read().await;
            if guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(LlamacppServerError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is already mounted"
                )));
            }
        }

        let resolved_path = match &self.options.adapter_transport {
            LlamacppServerAdapterTransport::LocalFs(p) if p.as_os_str().is_empty() => {
                path_or_dir.to_path_buf()
            }
            LlamacppServerAdapterTransport::LocalFs(p) => p.clone(),
            LlamacppServerAdapterTransport::HfHub { .. } => {
                return Err(LlamacppServerError::Unsupported(
                    "LlamacppServerAdapterTransport::HfHub — llama-server has no \
                     HF Hub resolver. Pre-cache the adapter on the server's filesystem, \
                     launch llama-server with --lora <path>, and use LocalFs."
                        .into(),
                ));
            }
            LlamacppServerAdapterTransport::HttpPush(_) => {
                return Err(LlamacppServerError::Unsupported(
                    "LlamacppServerAdapterTransport::HttpPush — llama-server has no \
                     binary-upload endpoint. Adapters must be preloaded at startup \
                     via the --lora CLI flag; use LocalFs."
                        .into(),
                ));
            }
        };

        let preloaded = self.client.list_lora_adapters().await?;
        let match_key = resolved_path.to_string_lossy().to_string();
        let row = preloaded
            .iter()
            .find(|row| row.path == match_key || row.path.ends_with(&match_key))
            .ok_or_else(|| {
                LlamacppServerError::AdapterFailed(format!(
                    "adapter path '{match_key}' is not preloaded on llama-server \
                     (run llama-server with --lora '{match_key}' at startup); \
                     preloaded paths: {:?}",
                    preloaded.iter().map(|r| &r.path).collect::<Vec<_>>()
                ))
            })?;
        let server_index = row.id;

        // Build the new active set: every adapter currently tracked
        // locally as mounted (scale > 0.0) PLUS the new one at 1.0.
        let mut toggles: Vec<LlamacppServerLoraToggle> = {
            let guard = self.mounted_adapters.read().await;
            guard
                .iter()
                .map(|a| LlamacppServerLoraToggle {
                    id: a.server_index,
                    scale: a.scale,
                })
                .collect()
        };
        toggles.push(LlamacppServerLoraToggle {
            id: server_index,
            scale: 1.0,
        });

        self.client.set_lora_adapters(&toggles).await?;

        let mounted = MountedAdapter {
            adapter_id: adapter_id.clone(),
            server_index,
            source_dir: resolved_path,
            scale: 1.0,
        };

        self.mounted_adapters.write().await.push(mounted.clone());
        Ok(mounted)
    }

    /// Deactivate a previously activated adapter by sending a new
    /// `POST /lora-adapters` request that omits it (`llama-server`
    /// interprets an absent id as `scale: 0.0`).
    ///
    /// # Errors
    /// [`LlamacppServerError::AdapterFailed`] when the adapter isn't
    /// tracked locally; otherwise propagates from `POST /lora-adapters`.
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), LlamacppServerError> {
        {
            let guard = self.mounted_adapters.read().await;
            if !guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(LlamacppServerError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is not mounted"
                )));
            }
        }

        // Build the new active set: every adapter currently tracked
        // EXCEPT the one being unloaded.
        let toggles: Vec<LlamacppServerLoraToggle> = {
            let guard = self.mounted_adapters.read().await;
            guard
                .iter()
                .filter(|a| a.adapter_id != adapter_id)
                .map(|a| LlamacppServerLoraToggle {
                    id: a.server_index,
                    scale: a.scale,
                })
                .collect()
        };

        self.client.set_lora_adapters(&toggles).await?;

        let mut guard = self.mounted_adapters.write().await;
        guard.retain(|a| a.adapter_id != adapter_id);
        Ok(())
    }

    /// Locally-cached snapshot of mounted adapters. Use
    /// [`Self::refresh_adapters_from_server`] to reconcile with the
    /// upstream state.
    pub async fn list_adapters(&self) -> Vec<MountedAdapter> {
        self.mounted_adapters.read().await.clone()
    }

    /// GET `/lora-adapters` and update the local mounted-adapter cache.
    /// Returns the full upstream listing so callers can introspect.
    ///
    /// Heuristic: every row with `scale > 0.0` is treated as active and
    /// mirrored into the local cache. `adapter_id` defaults to
    /// `"lora-{id}"` because the upstream wire format has no
    /// human-readable label.
    ///
    /// # Errors
    /// Propagates [`LlamacppServerError`] from the underlying client.
    pub async fn refresh_adapters_from_server(
        &self,
    ) -> Result<Vec<LlamacppServerLoraAdapter>, LlamacppServerError> {
        let listing = self.client.list_lora_adapters().await?;

        let derived: Vec<MountedAdapter> = listing
            .iter()
            .filter(|row| row.scale > 0.0)
            .map(|row| MountedAdapter {
                adapter_id: format!("lora-{}", row.id),
                server_index: row.id,
                source_dir: std::path::PathBuf::from(&row.path),
                scale: row.scale,
            })
            .collect();

        *self.mounted_adapters.write().await = derived;
        Ok(listing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> LlamacppServerProvider {
        LlamacppServerProvider::from_options(LlamacppServerOptions::required(
            "http://localhost:8080",
            "llama-3.2",
        ))
        .expect("build provider")
    }

    #[test]
    fn from_options_rejects_empty_endpoint() {
        let opts = LlamacppServerOptions::required("", "llama-3.2");
        assert!(LlamacppServerProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_empty_model() {
        let opts = LlamacppServerOptions::required("http://h", "");
        assert!(LlamacppServerProvider::from_options(opts).is_err());
    }

    #[test]
    fn provider_exposes_model_id() {
        let p = make_provider();
        assert_eq!(p.model_id(), "llama-3.2");
    }

    #[tokio::test]
    async fn load_adapter_rejects_http_push_transport() {
        let opts = LlamacppServerOptions {
            adapter_transport: LlamacppServerAdapterTransport::HttpPush(vec![1, 2, 3]),
            ..LlamacppServerOptions::required("http://localhost:8080", "llama-3.2")
        };
        let p = LlamacppServerProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("a1", std::path::Path::new("/srv/loras/a1.gguf"))
            .await
            .expect_err("HttpPush should not be supported");
        assert!(matches!(err, LlamacppServerError::Unsupported(_)));
    }

    #[tokio::test]
    async fn load_adapter_rejects_hf_hub_transport() {
        let opts = LlamacppServerOptions {
            adapter_transport: LlamacppServerAdapterTransport::HfHub {
                repo: "tloen/alpaca-lora-7b".into(),
                revision: None,
            },
            ..LlamacppServerOptions::required("http://localhost:8080", "llama-3.2")
        };
        let p = LlamacppServerProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("a1", std::path::Path::new("/srv/loras/a1.gguf"))
            .await
            .expect_err("HfHub should not be supported");
        assert!(matches!(err, LlamacppServerError::Unsupported(_)));
    }

    #[tokio::test]
    async fn unload_adapter_unknown_id_errors() {
        let p = make_provider();
        let err = p
            .unload_adapter("never-mounted")
            .await
            .expect_err("unknown adapter should error");
        assert!(matches!(err, LlamacppServerError::AdapterFailed(_)));
    }
}
