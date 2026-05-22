//! [`OllamaProvider`] — the public type that proxies inference and
//! adapter management to an Ollama server.
//!
//! Inherent methods (`generate`, `chat`, `embed`, `load_adapter`,
//! `unload_adapter`, `list_adapters`, ...) match the surface the
//! `blazen-llm` bridge in `backends/ollama.rs` calls — keeping the trait
//! impls in the upstream crate (where the trait lives) and the wire
//! logic here.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;

use crate::OllamaError;
use crate::client::{OllamaClient, OllamaModelEntry, OllamaPullProgress};
use crate::options::{OllamaAdapterTransport, OllamaOptions};

/// One adapter currently mounted on the upstream Ollama server, tracked
/// client-side so the bridge can answer `list_adapters` without
/// re-querying `/api/tags` for every call.
///
/// Ollama models adapters as full derived models (created via
/// `/api/create` + Modelfile), so the `adapter_id` here is the derived
/// model name clients pass in the request `model` field.
#[derive(Debug, Clone)]
pub struct MountedAdapter {
    /// Derived model name on the upstream — clients put this in the
    /// `model` field of `/api/generate` or `/api/chat` to route through
    /// the adapter-augmented model.
    pub adapter_id: String,
    /// Path / hub-spec / sentinel describing where the adapter came from.
    /// Surfaced as [`blazen_llm::AdapterStatus::source_dir`] via the bridge.
    pub source_dir: std::path::PathBuf,
    /// Scale at mount time. The Modelfile `ADAPTER` directive does not
    /// accept a scale — the field is preserved for parity with other
    /// backends; the in-memory value is always `1.0`.
    pub scale: f32,
}

/// Ollama proxy provider.
///
/// Stateless on the wire — every call goes to the upstream server.
/// State held locally:
/// - the [`OllamaOptions`] for headers, timeouts, transport mode,
/// - an `Arc<OllamaClient>` (cheap clones share the connection pool),
/// - a `RwLock<Vec<MountedAdapter>>` mirroring the upstream derived-model
///   set so `list_adapters` can return without a round-trip.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    options: OllamaOptions,
    client: Arc<OllamaClient>,
    mounted_adapters: Arc<RwLock<Vec<MountedAdapter>>>,
}

impl OllamaProvider {
    /// Build a provider from options. Validates fields and constructs
    /// the underlying HTTP client immediately so misconfiguration fails
    /// at startup, not on the first inference call.
    ///
    /// # Errors
    /// [`OllamaError::InvalidOptions`] when a required field is missing,
    /// or [`OllamaError::Init`] when the `reqwest::Client` cannot be built.
    pub fn from_options(options: OllamaOptions) -> Result<Self, OllamaError> {
        let client = OllamaClient::new(&options)?;
        Ok(Self {
            options,
            client: Arc::new(client),
            mounted_adapters: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Borrow the stored options.
    #[must_use]
    pub fn options(&self) -> &OllamaOptions {
        &self.options
    }

    /// Borrow the underlying HTTP client (escape hatch for raw calls).
    #[must_use]
    pub fn client(&self) -> Arc<OllamaClient> {
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

    /// Send a `/api/generate` request to Ollama. The body must already
    /// be shaped for Ollama's native surface; the bridge in
    /// `blazen-llm/src/backends/ollama.rs` performs the typed translation
    /// from `ModelRequest`.
    ///
    /// # Errors
    /// Propagates [`OllamaError`] from the underlying client.
    pub async fn generate(&self, body: Value) -> Result<Value, OllamaError> {
        self.client.generate(&body).await
    }

    /// Streaming variant of [`Self::generate`]; returns the raw
    /// `reqwest::Response` so the bridge can drive an NDJSON parser.
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn generate_stream(&self, body: Value) -> Result<reqwest::Response, OllamaError> {
        self.client.generate_stream(&body).await
    }

    /// Send a `/api/chat` request.
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn chat(&self, body: Value) -> Result<Value, OllamaError> {
        self.client.chat(&body).await
    }

    /// Streaming variant of [`Self::chat`].
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn chat_stream(&self, body: Value) -> Result<reqwest::Response, OllamaError> {
        self.client.chat_stream(&body).await
    }

    /// Send a `/api/embeddings` request.
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn embed(&self, body: Value) -> Result<Value, OllamaError> {
        self.client.embeddings(&body).await
    }

    // -----------------------------------------------------------------
    // Model management
    // -----------------------------------------------------------------

    /// GET `/api/tags`. Returns every installed model row.
    ///
    /// # Errors
    /// Propagates [`OllamaError`] from the underlying client.
    pub async fn tags(&self) -> Result<Vec<OllamaModelEntry>, OllamaError> {
        self.client.tags().await
    }

    /// POST `/api/pull` for the given model, draining the NDJSON
    /// progress stream and invoking `on_progress` for each frame.
    ///
    /// # Errors
    /// Propagates [`OllamaError`] from the underlying client.
    pub async fn pull(
        &self,
        model: &str,
        on_progress: impl FnMut(&OllamaPullProgress),
    ) -> Result<(), OllamaError> {
        self.client.pull(model, on_progress).await
    }

    /// DELETE `/api/delete` for the given model.
    ///
    /// # Errors
    /// Propagates [`OllamaError`] from the underlying client.
    pub async fn delete(&self, model: &str) -> Result<(), OllamaError> {
        self.client.delete(model).await
    }

    // -----------------------------------------------------------------
    // Adapter management
    // -----------------------------------------------------------------

    /// Mount a `LoRA` adapter on the upstream Ollama server.
    ///
    /// Strategy: build a Modelfile of the form
    /// ```text
    /// FROM <base>
    /// ADAPTER <adapter-ref>
    /// ```
    /// then POST it to `/api/create` with the supplied `adapter_id` as
    /// the derived model name. `<adapter-ref>` is resolved from
    /// [`OllamaOptions::adapter_transport`]:
    ///
    /// - [`OllamaAdapterTransport::LocalFs`] with a populated path —
    ///   used verbatim.
    /// - [`OllamaAdapterTransport::LocalFs`] with an empty path —
    ///   falls through to `path_or_dir`.
    /// - [`OllamaAdapterTransport::HfHub`] — the helper first issues a
    ///   `/api/pull` for the repo so the bytes are cached locally, then
    ///   the Modelfile references `hf://<repo>[:<revision>]`.
    /// - [`OllamaAdapterTransport::HttpPush`] — rejected with
    ///   [`OllamaError::Unsupported`] (Ollama has no binary-upload API).
    ///
    /// # Errors
    /// - [`OllamaError::AdapterFailed`] when the adapter is already
    ///   mounted, `/api/pull` fails for the HF case, or `/api/create`
    ///   returns non-2xx / an error frame.
    /// - [`OllamaError::Unsupported`] for `HttpPush`.
    pub async fn load_adapter(
        &self,
        adapter_id: impl Into<String>,
        path_or_dir: &std::path::Path,
    ) -> Result<MountedAdapter, OllamaError> {
        let adapter_id = adapter_id.into();

        {
            let guard = self.mounted_adapters.read().await;
            if guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(OllamaError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is already mounted"
                )));
            }
        }

        let adapter_ref = match &self.options.adapter_transport {
            OllamaAdapterTransport::LocalFs(p) if p.as_os_str().is_empty() => {
                path_or_dir.display().to_string()
            }
            OllamaAdapterTransport::LocalFs(p) => p.display().to_string(),
            OllamaAdapterTransport::HfHub { repo, revision } => {
                // Pre-pull so the Modelfile ADAPTER reference resolves
                // against locally-cached bytes. Errors propagate.
                let pull_ref = match revision {
                    Some(rev) => format!("hf://{repo}:{rev}"),
                    None => format!("hf://{repo}"),
                };
                self.client.pull(&pull_ref, |_| {}).await?;
                pull_ref
            }
            OllamaAdapterTransport::HttpPush(_) => {
                return Err(OllamaError::Unsupported(
                    "OllamaAdapterTransport::HttpPush — Ollama has no binary-upload endpoint. \
                     Stage the adapter on a path the server can read and use LocalFs, or push \
                     it to Hugging Face Hub and use HfHub."
                        .into(),
                ));
            }
        };

        let modelfile = format!("FROM {}\nADAPTER {}\n", self.options.model, adapter_ref);

        self.client.create(&adapter_id, &modelfile).await?;

        let mounted = MountedAdapter {
            adapter_id: adapter_id.clone(),
            source_dir: path_or_dir.to_path_buf(),
            scale: 1.0,
        };

        self.mounted_adapters.write().await.push(mounted.clone());
        Ok(mounted)
    }

    /// Unmount a previously mounted adapter by deleting the derived
    /// model from the upstream server.
    ///
    /// # Errors
    /// [`OllamaError::AdapterFailed`] when the adapter isn't tracked
    /// locally; otherwise propagates from `/api/delete`.
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), OllamaError> {
        {
            let guard = self.mounted_adapters.read().await;
            if !guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(OllamaError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is not mounted"
                )));
            }
        }

        self.client.delete(adapter_id).await?;

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

    /// GET `/api/tags` and update the local mounted-adapter cache.
    /// Returns the full upstream listing so callers can introspect.
    ///
    /// Heuristic: any model whose name is not the base model is treated
    /// as a derived adapter row. Ollama tags are flat — there is no
    /// `parent` field — so this is the best the proxy can do without a
    /// `/api/show` round trip per row.
    ///
    /// # Errors
    /// Propagates [`OllamaError`] from the underlying client.
    pub async fn refresh_adapters_from_server(&self) -> Result<Vec<OllamaModelEntry>, OllamaError> {
        let listing = self.client.tags().await?;
        let base = self.options.model.as_str();

        let derived: Vec<MountedAdapter> = listing
            .iter()
            .filter(|row| row.name != base)
            .map(|row| MountedAdapter {
                adapter_id: row.name.clone(),
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

    fn make_provider() -> OllamaProvider {
        OllamaProvider::from_options(OllamaOptions::required(
            "http://localhost:11434",
            "llama3.2",
        ))
        .expect("build provider")
    }

    #[test]
    fn from_options_rejects_empty_endpoint() {
        let opts = OllamaOptions::required("", "llama3.2");
        assert!(OllamaProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_empty_model() {
        let opts = OllamaOptions::required("http://h", "");
        assert!(OllamaProvider::from_options(opts).is_err());
    }

    #[test]
    fn provider_exposes_model_id() {
        let p = make_provider();
        assert_eq!(p.model_id(), "llama3.2");
    }

    #[tokio::test]
    async fn load_adapter_rejects_http_push_transport() {
        let opts = OllamaOptions {
            adapter_transport: OllamaAdapterTransport::HttpPush(vec![1, 2, 3]),
            ..OllamaOptions::required("http://localhost:11434", "llama3.2")
        };
        let p = OllamaProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("a1", std::path::Path::new("/srv/loras/a1"))
            .await
            .expect_err("HttpPush should not be supported");
        assert!(matches!(err, OllamaError::Unsupported(_)));
    }

    #[tokio::test]
    async fn unload_adapter_unknown_id_errors() {
        let p = make_provider();
        let err = p
            .unload_adapter("never-mounted")
            .await
            .expect_err("unknown adapter should error");
        assert!(matches!(err, OllamaError::AdapterFailed(_)));
    }
}
