//! [`TgiProvider`] — the public type that proxies inference and
//! adapter selection to a `HuggingFace` TGI server.
//!
//! Inherent methods (`generate`, `chat`, `complete`, `info`,
//! `list_models`, `load_adapter`, `unload_adapter`, `list_adapters`)
//! match the surface the `blazen-llm` bridge in `backends/tgi.rs`
//! calls — keeping the trait impls in the upstream crate (where the
//! trait lives) and the wire logic here.
//!
//! # Adapter strategy
//!
//! TGI loads adapters at server startup via
//! `--lora-adapters <id1> <id2> ...`. The HTTP surface has no
//! runtime mount / unmount endpoints — adapters are selected
//! per-request by setting the `adapter_id` field. The provider
//! therefore models adapter management as a *registration* of an
//! already-loaded id, not a mount operation:
//!
//! - [`TgiProvider::load_adapter`] verifies via `GET /v1/models` that
//!   the adapter id is preloaded on the server, then stores it in a
//!   `Mutex<Option<String>>` as the **active** adapter for subsequent
//!   requests (the bridge can also override per-request).
//! - Transports that imply runtime mounting
//!   ([`TgiAdapterTransport::HttpPush`]) are rejected up-front with
//!   [`TgiError::Unsupported`].
//! - [`TgiProvider::unload_adapter`] clears the active adapter
//!   (the upstream weights stay loaded until the server is restarted).

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{Mutex, RwLock};

use crate::TgiError;
use crate::client::{TgiClient, TgiInfo, TgiModelEntry};
use crate::options::{TgiAdapterTransport, TgiOptions};

/// One adapter the proxy considers active on the upstream TGI server.
/// Unlike vLLM / Ollama where the proxy actually mounts the weights,
/// TGI "registration" just means "the bridge will set `adapter_id` to
/// this string on outgoing requests".
#[derive(Debug, Clone)]
pub struct ActiveAdapter {
    /// Adapter id — must match one of the ids passed to
    /// `--lora-adapters` at server startup. The bridge sets the
    /// request `adapter_id` field to this string.
    pub adapter_id: String,
    /// Path / hub-spec / sentinel describing where the adapter came
    /// from. Informational only — TGI does not consume it. Surfaced
    /// as [`blazen_llm::AdapterStatus::source_dir`] via the bridge.
    pub source_dir: std::path::PathBuf,
    /// Scale at registration time. TGI does not accept a per-request
    /// scale — preserved for parity with other backends; always `1.0`.
    pub scale: f32,
}

/// TGI proxy provider.
///
/// Stateless on the wire — every call goes to the upstream server.
/// State held locally:
/// - the [`TgiOptions`] for headers, timeouts, transport mode,
/// - an `Arc<TgiClient>` (cheap clones share the connection pool),
/// - a `Mutex<Option<String>>` holding the active adapter id (the
///   default per-request `adapter_id` value the bridge attaches when
///   the caller does not override per request),
/// - a `RwLock<Vec<ActiveAdapter>>` mirroring the set of adapter ids
///   the proxy has registered, so `list_adapters` can answer without
///   a round-trip.
#[derive(Debug, Clone)]
pub struct TgiProvider {
    options: TgiOptions,
    client: Arc<TgiClient>,
    active_adapter: Arc<Mutex<Option<String>>>,
    registered_adapters: Arc<RwLock<Vec<ActiveAdapter>>>,
}

impl TgiProvider {
    /// Build a provider from options. Validates fields and constructs
    /// the underlying HTTP client immediately so misconfiguration fails
    /// at startup, not on the first inference call.
    ///
    /// # Errors
    /// [`TgiError::InvalidOptions`] when a required field is missing,
    /// or [`TgiError::Init`] when the `reqwest::Client` cannot be built.
    pub fn from_options(options: TgiOptions) -> Result<Self, TgiError> {
        let client = TgiClient::new(&options)?;
        Ok(Self {
            options,
            client: Arc::new(client),
            active_adapter: Arc::new(Mutex::new(None)),
            registered_adapters: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Borrow the stored options.
    #[must_use]
    pub fn options(&self) -> &TgiOptions {
        &self.options
    }

    /// Borrow the underlying HTTP client (escape hatch for raw calls).
    #[must_use]
    pub fn client(&self) -> Arc<TgiClient> {
        Arc::clone(&self.client)
    }

    /// Base model id the provider was constructed with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.options.model
    }

    /// The currently-active adapter id, if any has been registered via
    /// [`Self::load_adapter`]. The bridge attaches this to outgoing
    /// requests as `adapter_id` unless the caller overrides per request.
    pub async fn active_adapter_id(&self) -> Option<String> {
        self.active_adapter.lock().await.clone()
    }

    // -----------------------------------------------------------------
    // Inference — native TGI shape
    // -----------------------------------------------------------------

    /// POST `/generate` (native TGI shape, single response).
    ///
    /// The bridge in `blazen-llm/src/backends/tgi.rs` performs the typed
    /// translation from `CompletionRequest`; this is a thin pass-through
    /// for `serde_json::Value` bodies that already match TGI's schema.
    /// If an adapter is active and the body does not already carry
    /// `adapter_id`, the active id is attached.
    ///
    /// # Errors
    /// Propagates [`TgiError`] from the underlying client.
    pub async fn generate(&self, body: Value) -> Result<Value, TgiError> {
        let body = self.attach_active_adapter(body).await;
        self.client.generate(&body).await
    }

    /// Streaming variant of [`Self::generate`]; returns the raw
    /// `reqwest::Response` so the bridge can drive the SSE parser.
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn generate_stream(&self, body: Value) -> Result<reqwest::Response, TgiError> {
        let body = self.attach_active_adapter(body).await;
        self.client.generate_stream(&body).await
    }

    // -----------------------------------------------------------------
    // Inference — OpenAI-compatible shape
    // -----------------------------------------------------------------

    /// POST `/v1/chat/completions` (non-streaming).
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn chat(&self, body: Value) -> Result<Value, TgiError> {
        let body = self.attach_active_adapter(body).await;
        self.client.chat_completions(&body).await
    }

    /// Streaming variant of [`Self::chat`].
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn chat_stream(&self, body: Value) -> Result<reqwest::Response, TgiError> {
        let body = self.attach_active_adapter(body).await;
        self.client.chat_completions_stream(&body).await
    }

    /// POST `/v1/completions` (OpenAI-compatible legacy completion,
    /// non-streaming).
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn complete(&self, body: Value) -> Result<Value, TgiError> {
        let body = self.attach_active_adapter(body).await;
        self.client.completions(&body).await
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// GET `/info`.
    ///
    /// # Errors
    /// Propagates [`TgiError`] from the underlying client.
    pub async fn info(&self) -> Result<TgiInfo, TgiError> {
        self.client.info().await
    }

    /// GET `/v1/models`.
    ///
    /// # Errors
    /// Propagates [`TgiError`] from the underlying client.
    pub async fn list_models(&self) -> Result<Vec<TgiModelEntry>, TgiError> {
        self.client.list_models().await
    }

    /// GET `/metrics` (Prometheus text body).
    ///
    /// # Errors
    /// Propagates [`TgiError`] from the underlying client.
    pub async fn metrics(&self) -> Result<String, TgiError> {
        self.client.metrics().await
    }

    // -----------------------------------------------------------------
    // Adapter management — TGI has no runtime mount/unmount endpoint,
    // so these methods register a preloaded adapter id rather than
    // pushing weights upstream.
    // -----------------------------------------------------------------

    /// Register a preloaded adapter id as the **active** adapter for
    /// subsequent requests.
    ///
    /// Strategy: TGI requires `--lora-adapters <id1> <id2> ...` at
    /// server startup; the proxy verifies the id is in `GET /v1/models`
    /// then stores it as the active id (attached to outgoing requests
    /// as `adapter_id`).
    ///
    /// Transports that imply runtime mounting are rejected:
    ///
    /// - [`TgiAdapterTransport::LocalFs`] — accepted (path informational).
    /// - [`TgiAdapterTransport::HfHub`]   — accepted (repo informational).
    /// - [`TgiAdapterTransport::HttpPush`] — rejected with
    ///   [`TgiError::Unsupported`] (TGI has no binary-upload API).
    ///
    /// # Errors
    /// - [`TgiError::Unsupported`] for `HttpPush`.
    /// - [`TgiError::NotFound`] when the id is not in `/v1/models`
    ///   (the server was started without it).
    /// - Any [`TgiError`] from the underlying `/v1/models` call.
    pub async fn load_adapter(
        &self,
        adapter_id: impl Into<String>,
        path_or_dir: &std::path::Path,
    ) -> Result<ActiveAdapter, TgiError> {
        let adapter_id = adapter_id.into();

        // Reject transports that imply runtime mounting up-front,
        // before doing any wire calls.
        if matches!(
            self.options.adapter_transport,
            TgiAdapterTransport::HttpPush(_)
        ) {
            return Err(TgiError::Unsupported(
                "TgiAdapterTransport::HttpPush — TGI has no binary-upload endpoint. \
                 Restart the server with `--lora-adapters <id>` (after staging the \
                 adapter on a path the server can read, or pushing it to HF Hub) \
                 and use LocalFs / HfHub instead."
                    .into(),
            ));
        }

        // Verify the id is in the preloaded set. Adapter rows in TGI
        // ≥ 2.0 are reported with `object == "lora"`; older versions
        // omit the object field, so accept either form.
        let listing = self.client.list_models().await?;
        let known = listing.iter().any(|row| row.id == adapter_id);
        if !known {
            return Err(TgiError::NotFound(format!(
                "adapter '{adapter_id}' is not loaded on the TGI server (start it with \
                 `--lora-adapters {adapter_id}` to make it available)"
            )));
        }

        let active = ActiveAdapter {
            adapter_id: adapter_id.clone(),
            source_dir: path_or_dir.to_path_buf(),
            scale: 1.0,
        };

        {
            let mut guard = self.registered_adapters.write().await;
            // Idempotent — re-registering the same id overrides the
            // source_dir / scale but does not duplicate the row.
            guard.retain(|a| a.adapter_id != adapter_id);
            guard.push(active.clone());
        }
        *self.active_adapter.lock().await = Some(adapter_id);
        Ok(active)
    }

    /// Drop a previously-registered adapter from the local cache and
    /// clear the active id if it pointed at this adapter. Note that
    /// the upstream TGI server keeps the weights loaded until restart;
    /// this is purely a client-side "stop selecting it" operation.
    ///
    /// # Errors
    /// [`TgiError::Unsupported`] when the adapter id was never
    /// registered (so this is not idempotent — matches the contract of
    /// the other proxy backends).
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), TgiError> {
        {
            let guard = self.registered_adapters.read().await;
            if !guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(TgiError::Unsupported(format!(
                    "adapter '{adapter_id}' is not registered on this provider \
                     (TGI keeps weights loaded until server restart — this only \
                     clears the client-side active-id cache)"
                )));
            }
        }
        self.registered_adapters
            .write()
            .await
            .retain(|a| a.adapter_id != adapter_id);
        let mut guard = self.active_adapter.lock().await;
        if guard.as_deref() == Some(adapter_id) {
            *guard = None;
        }
        Ok(())
    }

    /// Locally-cached snapshot of registered adapters.
    pub async fn list_adapters(&self) -> Vec<ActiveAdapter> {
        self.registered_adapters.read().await.clone()
    }

    /// GET `/v1/models` and rebuild the local registered-adapter cache
    /// from the upstream listing. Any row whose id differs from the
    /// base model is treated as a preloaded adapter.
    ///
    /// # Errors
    /// Propagates [`TgiError`] from the underlying client.
    pub async fn refresh_adapters_from_server(&self) -> Result<Vec<TgiModelEntry>, TgiError> {
        let listing = self.client.list_models().await?;
        let base = self.options.model.as_str();
        let derived: Vec<ActiveAdapter> = listing
            .iter()
            .filter(|row| row.id != base)
            .map(|row| ActiveAdapter {
                adapter_id: row.id.clone(),
                source_dir: std::path::PathBuf::new(),
                scale: 1.0,
            })
            .collect();
        *self.registered_adapters.write().await = derived;
        Ok(listing)
    }

    // -----------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------

    /// If an `adapter_id` is registered as active and the body does not
    /// already carry one, attach it.
    async fn attach_active_adapter(&self, mut body: Value) -> Value {
        if body.get("adapter_id").is_some() {
            return body;
        }
        if let Some(ref id) = *self.active_adapter.lock().await {
            body["adapter_id"] = Value::String(id.clone());
        }
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> TgiProvider {
        TgiProvider::from_options(TgiOptions::required(
            "http://localhost:8080",
            "meta-llama/Llama-3.2-3B",
        ))
        .expect("build provider")
    }

    #[test]
    fn from_options_rejects_empty_endpoint() {
        let opts = TgiOptions::required("", "llama");
        assert!(TgiProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_empty_model() {
        let opts = TgiOptions::required("http://h", "");
        assert!(TgiProvider::from_options(opts).is_err());
    }

    #[test]
    fn provider_exposes_model_id() {
        let p = make_provider();
        assert_eq!(p.model_id(), "meta-llama/Llama-3.2-3B");
    }

    #[tokio::test]
    async fn load_adapter_rejects_http_push_transport() {
        let opts = TgiOptions {
            adapter_transport: TgiAdapterTransport::HttpPush(vec![1, 2, 3]),
            ..TgiOptions::required("http://localhost:8080", "llama")
        };
        let p = TgiProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("a1", std::path::Path::new("/srv/loras/a1"))
            .await
            .expect_err("HttpPush should not be supported");
        assert!(matches!(err, TgiError::Unsupported(_)));
    }

    #[tokio::test]
    async fn unload_adapter_unknown_id_errors() {
        let p = make_provider();
        let err = p
            .unload_adapter("never-registered")
            .await
            .expect_err("unknown adapter should error");
        assert!(matches!(err, TgiError::Unsupported(_)));
    }

    #[tokio::test]
    async fn active_adapter_starts_unset() {
        let p = make_provider();
        assert!(p.active_adapter_id().await.is_none());
    }

    #[tokio::test]
    async fn attach_active_adapter_skips_when_body_has_one() {
        let p = make_provider();
        *p.active_adapter.lock().await = Some("active-id".into());
        let body = serde_json::json!({"inputs": "x", "adapter_id": "request-id"});
        let attached = p.attach_active_adapter(body).await;
        assert_eq!(attached["adapter_id"], "request-id");
    }

    #[tokio::test]
    async fn attach_active_adapter_no_op_when_none_active() {
        let p = make_provider();
        let body = serde_json::json!({"inputs": "x"});
        let attached = p.attach_active_adapter(body).await;
        assert!(attached.get("adapter_id").is_none());
    }
}
