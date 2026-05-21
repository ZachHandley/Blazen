//! [`LmStudioProvider`] — the public type that proxies inference and
//! model lifecycle to an LM Studio server.
//!
//! Inherent methods (`complete`, `stream`, `completions`, `embed`,
//! `load_model`, `unload_model`, `load_adapter`, ...) match the surface
//! the `blazen-llm` bridge in `backends/lmstudio.rs` calls — keeping
//! the trait impls in the upstream crate (where the trait lives) and
//! the wire logic here.

use std::sync::Arc;

use serde_json::Value;

use crate::LmStudioError;
use crate::client::{
    LmStudioClient, LmStudioModelEntry, LmStudioNativeModelEntry, LmStudioNativeModelState,
};
use crate::options::{LmStudioAdapterTransport, LmStudioOptions};

/// LM Studio proxy provider.
///
/// Stateless on the wire — every call goes to the upstream server.
/// State held locally:
/// - the [`LmStudioOptions`] for headers, timeouts, transport mode,
/// - an `Arc<LmStudioClient>` (cheap clones share the connection pool).
///
/// Unlike `OllamaProvider` and `VllmProvider`, this proxy holds no
/// adapter cache — LM Studio doesn't support runtime `LoRA` mounting, so
/// there is no per-adapter state to mirror locally.
#[derive(Debug, Clone)]
pub struct LmStudioProvider {
    options: LmStudioOptions,
    client: Arc<LmStudioClient>,
}

impl LmStudioProvider {
    /// Build a provider from options. Validates fields and constructs
    /// the underlying HTTP client immediately so misconfiguration fails
    /// at startup, not on the first inference call.
    ///
    /// # Errors
    /// [`LmStudioError::InvalidOptions`] when a required field is
    /// missing, or [`LmStudioError::Init`] when the `reqwest::Client`
    /// cannot be built.
    pub fn from_options(options: LmStudioOptions) -> Result<Self, LmStudioError> {
        let client = LmStudioClient::new(&options)?;
        Ok(Self {
            options,
            client: Arc::new(client),
        })
    }

    /// Borrow the stored options.
    #[must_use]
    pub fn options(&self) -> &LmStudioOptions {
        &self.options
    }

    /// Borrow the underlying HTTP client (escape hatch for raw calls).
    #[must_use]
    pub fn client(&self) -> Arc<LmStudioClient> {
        Arc::clone(&self.client)
    }

    /// Default model id the provider was constructed with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.options.model
    }

    // -----------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------

    /// Send a chat-completion request to LM Studio. The body must
    /// already be shaped for LM Studio's OpenAI-compat surface — the
    /// bridge in `blazen-llm/src/backends/lmstudio.rs` performs the
    /// typed translation from `CompletionRequest`.
    ///
    /// # Errors
    /// Propagates [`LmStudioError`] from the underlying client.
    pub async fn complete(&self, body: Value) -> Result<Value, LmStudioError> {
        self.client.chat_completions(&body).await
    }

    /// Streaming variant of [`Self::complete`]; returns the raw
    /// `reqwest::Response` so the bridge can drive an SSE parser.
    ///
    /// # Errors
    /// As [`Self::complete`].
    pub async fn stream(&self, body: Value) -> Result<reqwest::Response, LmStudioError> {
        self.client.chat_completions_stream(&body).await
    }

    /// Legacy `/v1/completions` (single-prompt) variant.
    ///
    /// # Errors
    /// As [`Self::complete`].
    pub async fn completions(&self, body: Value) -> Result<Value, LmStudioError> {
        self.client.completions(&body).await
    }

    /// Send a `/v1/embeddings` request.
    ///
    /// # Errors
    /// As [`Self::complete`].
    pub async fn embed(&self, body: Value) -> Result<Value, LmStudioError> {
        self.client.embeddings(&body).await
    }

    // -----------------------------------------------------------------
    // Model management
    // -----------------------------------------------------------------

    /// GET `/v1/models` (OAI-shaped listing).
    ///
    /// # Errors
    /// Propagates [`LmStudioError`] from the underlying client.
    pub async fn list_models(&self) -> Result<Vec<LmStudioModelEntry>, LmStudioError> {
        self.client.list_models().await
    }

    /// GET `/api/v0/models` (LM-Studio-native listing with load state).
    ///
    /// # Errors
    /// Propagates [`LmStudioError`] from the underlying client.
    pub async fn native_models(&self) -> Result<Vec<LmStudioNativeModelEntry>, LmStudioError> {
        self.client.native_models().await
    }

    /// POST `/api/v0/models/load` for the given model id.
    ///
    /// # Errors
    /// Propagates [`LmStudioError`] from the underlying client.
    pub async fn load_model(&self, model: &str) -> Result<(), LmStudioError> {
        self.client.load_model(model).await
    }

    /// POST `/api/v0/models/unload` for the given model id.
    ///
    /// # Errors
    /// Propagates [`LmStudioError`] from the underlying client.
    pub async fn unload_model(&self, model: &str) -> Result<(), LmStudioError> {
        self.client.unload_model(model).await
    }

    /// Returns `true` when the upstream native listing reports the
    /// provider's configured model as `loaded`.
    ///
    /// # Errors
    /// Propagates [`LmStudioError`] from the underlying client.
    pub async fn is_model_loaded(&self) -> Result<bool, LmStudioError> {
        let rows = self.native_models().await?;
        Ok(rows
            .into_iter()
            .any(|m| m.id == self.options.model && m.state == LmStudioNativeModelState::Loaded))
    }

    // -----------------------------------------------------------------
    // Adapter management
    // -----------------------------------------------------------------

    /// LM Studio does not support runtime LoRA-adapter mounting.
    /// Always returns [`LmStudioError::Unsupported`] with guidance on
    /// merging the adapter into a GGUF base.
    ///
    /// # Errors
    /// Always [`LmStudioError::Unsupported`].
    //
    // `async` is intentional: this method shares the inherent surface
    // with [`OllamaProvider::load_adapter`] / [`VllmProvider::load_adapter`],
    // both of which await network calls. Keeping the signature async
    // means the upstream `backends/lmstudio.rs` bridge can call
    // `self.load_adapter(...).await` without per-backend special-casing,
    // and lets a future LM Studio release that adds runtime LoRA
    // mounting flip the body without a breaking signature change.
    #[allow(clippy::unused_async)]
    pub async fn load_adapter(
        &self,
        adapter_id: impl Into<String>,
        path_or_dir: &std::path::Path,
    ) -> Result<(), LmStudioError> {
        let adapter_id = adapter_id.into();
        // Inspect the transport so the error message can describe what
        // the caller asked for, but reject every variant — LM Studio
        // has no runtime mount API.
        let transport_hint = match &self.options.adapter_transport {
            LmStudioAdapterTransport::LocalFs(p) if p.as_os_str().is_empty() => {
                format!("LocalFs({})", path_or_dir.display())
            }
            LmStudioAdapterTransport::LocalFs(p) => format!("LocalFs({})", p.display()),
            LmStudioAdapterTransport::HfHub { repo, revision } => match revision {
                Some(rev) => format!("HfHub({repo}@{rev})"),
                None => format!("HfHub({repo})"),
            },
            LmStudioAdapterTransport::HttpPush(bytes) => {
                format!("HttpPush({} bytes)", bytes.len())
            }
        };
        Err(LmStudioError::Unsupported(format!(
            "load_adapter(adapter_id='{adapter_id}', transport={transport_hint}): LM Studio does \
             not support runtime LoRA adapter mounting. Adapters must be baked into the GGUF \
             model file: (1) merge the LoRA into the base safetensors with Blazen's \
             `merge_lora_into_base` (PR-AM), (2) convert the merged checkpoint to GGUF via \
             llama.cpp's `convert_hf_to_gguf.py` (or download a pre-merged \
             `mradermacher/*-i1-GGUF`-style checkpoint), then (3) load it as a full model with \
             `LmStudioProvider::load_model`."
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_provider() -> LmStudioProvider {
        LmStudioProvider::from_options(LmStudioOptions::required(
            "http://localhost:1234",
            "qwen2.5-7b-instruct",
        ))
        .expect("build provider")
    }

    #[test]
    fn from_options_rejects_empty_endpoint() {
        let opts = LmStudioOptions::required("", "qwen2.5-7b-instruct");
        assert!(LmStudioProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_empty_model() {
        let opts = LmStudioOptions::required("http://h", "");
        assert!(LmStudioProvider::from_options(opts).is_err());
    }

    #[test]
    fn provider_exposes_model_id() {
        let p = make_provider();
        assert_eq!(p.model_id(), "qwen2.5-7b-instruct");
    }

    #[tokio::test]
    async fn load_adapter_always_returns_unsupported_for_localfs() {
        let p = make_provider();
        let err = p
            .load_adapter("sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
            .await
            .expect_err("LM Studio does not support runtime LoRA");
        assert!(matches!(err, LmStudioError::Unsupported(_)));
        let msg = err.to_string();
        assert!(msg.contains("merge_lora_into_base") || msg.contains("merged"));
        assert!(msg.contains("GGUF"));
    }

    #[tokio::test]
    async fn load_adapter_returns_unsupported_for_hf_hub() {
        let opts = LmStudioOptions {
            adapter_transport: LmStudioAdapterTransport::HfHub {
                repo: "tloen/alpaca-lora-7b".into(),
                revision: Some("v1.0".into()),
            },
            ..LmStudioOptions::required("http://localhost:1234", "qwen2.5-7b-instruct")
        };
        let p = LmStudioProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("alpaca", &PathBuf::from("/unused"))
            .await
            .expect_err("HfHub should not be supported");
        assert!(matches!(err, LmStudioError::Unsupported(_)));
        let msg = err.to_string();
        assert!(msg.contains("tloen/alpaca-lora-7b"));
        assert!(msg.contains("v1.0"));
    }

    #[tokio::test]
    async fn load_adapter_returns_unsupported_for_http_push() {
        let opts = LmStudioOptions {
            adapter_transport: LmStudioAdapterTransport::HttpPush(vec![0u8; 1024]),
            ..LmStudioOptions::required("http://localhost:1234", "qwen2.5-7b-instruct")
        };
        let p = LmStudioProvider::from_options(opts).unwrap();
        let err = p
            .load_adapter("inline", &PathBuf::from("/unused"))
            .await
            .expect_err("HttpPush should not be supported");
        assert!(matches!(err, LmStudioError::Unsupported(_)));
        assert!(err.to_string().contains("1024"));
    }
}
