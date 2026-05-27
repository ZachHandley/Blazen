//! Concrete per-engine embedding provider classes.
//!
//! Each `<Engine>Provider` newtype wraps the matching upstream
//! embedding backend (fastembed, tract, candle, `OpenAI`, fal) and
//! implements the polymorphic
//! [`crate::providers::root::BaseProvider`] root plus the
//! [`crate::providers::capabilities::EmbeddingProvider`] capability
//! sub-trait.
//!
//! Construction patterns mirror the existing `new_*_embedding_model`
//! factories used in `crates/blazen-uniffi/src/compute.rs` so the
//! binding crates (napi-rs / `PyO3` / `UniFFI` / cabi / Ruby / WASM)
//! can wrap these structs directly without re-deriving the
//! HF-download / config plumbing.
//!
//! Cloud providers (`OpenAI`, fal) delegate to their existing
//! `EmbeddingModel` impls in [`crate::providers::openai`] /
//! [`crate::providers::fal`] so retry / HTTP-client / queue-poll plumbing
//! stays in one place.

#![allow(unused_imports)]

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::BlazenError;
use crate::providers::capabilities::EmbeddingProvider;
use crate::providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// FastembedProvider — local fastembed (ORT) text embeddings
// ---------------------------------------------------------------------------

/// Concrete [`EmbeddingProvider`] backed by
/// [`blazen_embed_fastembed::FastEmbedModel`] (ONNX Runtime).
///
/// Only available on ORT-supported targets
/// (`x86_64-{linux,windows,macos}`, `aarch64-apple-darwin`). On other
/// targets use [`TractEmbedProvider`] instead — same model catalog,
/// pure-Rust runtime.
#[cfg(feature = "embed-fastembed")]
pub struct FastembedProvider {
    inner: Arc<blazen_embed_fastembed::FastEmbedModel>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "embed-fastembed")]
impl FastembedProvider {
    /// Construct a fastembed provider from
    /// [`blazen_embed_fastembed::FastEmbedOptions`].
    ///
    /// Defaults (when `opts` is `FastEmbedOptions::default()`) resolve
    /// to `BAAI/bge-small-en-v1.5` on CPU with fastembed's built-in
    /// cache. Model selection uses the fastembed enum-name string
    /// (e.g. `"BGESmallENV15"`) — see the catalog in
    /// `blazen_embed_fastembed`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] with provider `"fastembed"` if
    /// the underlying `FastEmbedModel::from_options` call fails (unknown
    /// model name, HF download failure, ORT init error).
    pub fn new(opts: blazen_embed_fastembed::FastEmbedOptions) -> Result<Self, BlazenError> {
        let model = blazen_embed_fastembed::FastEmbedModel::from_options(opts)
            .map_err(|e| BlazenError::provider("fastembed", e.to_string()))?;
        let version = model.model_id().to_owned();
        let metadata =
            ProviderMetadata::new("fastembed", CapabilityKind::Embedding).with_version(version);
        Ok(Self {
            inner: Arc::new(model),
            metadata,
        })
    }

    /// Construct from an already-loaded `FastEmbedModel` handle.
    ///
    /// Useful when the caller wants to share one weights load across
    /// multiple consumer surfaces (e.g. capability-erased
    /// `Arc<dyn BaseProvider>` and a direct typed handle).
    #[must_use]
    pub fn from_model(model: Arc<blazen_embed_fastembed::FastEmbedModel>) -> Self {
        let version = model.model_id().to_owned();
        let metadata =
            ProviderMetadata::new("fastembed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: model,
            metadata,
        }
    }

    /// Borrow the inner backend — escape hatch for advanced callers
    /// that need to drive the fastembed model directly.
    #[must_use]
    pub fn model(&self) -> &Arc<blazen_embed_fastembed::FastEmbedModel> {
        &self.inner
    }
}

#[cfg(feature = "embed-fastembed")]
impl std::fmt::Debug for FastembedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastembedProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "embed-fastembed")]
impl BaseProvider for FastembedProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "embed-fastembed")]
#[async_trait]
impl EmbeddingProvider for FastembedProvider {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        let response = self
            .inner
            .embed(&texts)
            .await
            .map_err(|e| BlazenError::provider("fastembed", e.to_string()))?;
        Ok(response.embeddings)
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}

// ---------------------------------------------------------------------------
// TractEmbedProvider — local tract (pure-Rust ONNX) text embeddings
// ---------------------------------------------------------------------------

/// Concrete [`EmbeddingProvider`] backed by
/// [`blazen_embed_tract::TractEmbedModel`] (pure-Rust ONNX inference).
///
/// Drop-in replacement for [`FastembedProvider`] on targets that lack
/// ORT prebuilt binaries (musl Linux, aarch64-linux, wasm). Loads the
/// same fastembed model catalog via `tract_onnx`.
#[cfg(feature = "embed-tract")]
pub struct TractEmbedProvider {
    inner: Arc<blazen_embed_tract::TractEmbedModel>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "embed-tract")]
impl TractEmbedProvider {
    /// Construct a tract embed provider from
    /// [`blazen_embed_tract::TractOptions`].
    ///
    /// Defaults to `BGESmallENV15`. Model names are the same
    /// case-insensitive variant strings used by fastembed (e.g.
    /// `"BGESmallENV15"`, `"AllMiniLML6V2"`).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] with provider `"tract-embed"`
    /// if `TractEmbedModel::from_options` fails (unknown model name,
    /// HF download failure, ONNX parse error).
    pub fn new(opts: blazen_embed_tract::TractOptions) -> Result<Self, BlazenError> {
        let model = blazen_embed_tract::TractEmbedModel::from_options(opts)
            .map_err(|e| BlazenError::provider("tract-embed", e.to_string()))?;
        let version = model.model_id().to_owned();
        let metadata =
            ProviderMetadata::new("tract-embed", CapabilityKind::Embedding).with_version(version);
        Ok(Self {
            inner: Arc::new(model),
            metadata,
        })
    }

    /// Construct from an already-loaded `TractEmbedModel` handle.
    #[must_use]
    pub fn from_model(model: Arc<blazen_embed_tract::TractEmbedModel>) -> Self {
        let version = model.model_id().to_owned();
        let metadata =
            ProviderMetadata::new("tract-embed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: model,
            metadata,
        }
    }

    /// Borrow the inner backend.
    #[must_use]
    pub fn model(&self) -> &Arc<blazen_embed_tract::TractEmbedModel> {
        &self.inner
    }
}

#[cfg(feature = "embed-tract")]
impl std::fmt::Debug for TractEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TractEmbedProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "embed-tract")]
impl BaseProvider for TractEmbedProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "embed-tract")]
#[async_trait]
impl EmbeddingProvider for TractEmbedProvider {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        let response = self
            .inner
            .embed(&texts)
            .await
            .map_err(|e| BlazenError::provider("tract-embed", e.to_string()))?;
        Ok(response.embeddings)
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}

// ---------------------------------------------------------------------------
// CandleEmbedProvider — local candle BERT-family text embeddings
// ---------------------------------------------------------------------------

/// Concrete [`EmbeddingProvider`] backed by
/// [`blazen_embed_candle::CandleEmbedModel`].
///
/// Loads BERT-family sentence-transformer weights from `HuggingFace`
/// Hub and runs the forward pass via candle. Supports CPU, CUDA, and
/// Metal devices (the `engine` feature on `blazen-embed-candle` must
/// also be enabled for actual inference; without it the stub returns
/// `EngineNotAvailable` at embed time).
///
/// Note: [`CandleEmbedModel::dimensions`] returns `0` when the engine
/// feature is off (no model has been loaded). This is preserved
/// verbatim — see the inner type for the rationale.
#[cfg(feature = "candle-embed")]
pub struct CandleEmbedProvider {
    inner: Arc<blazen_embed_candle::CandleEmbedModel>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "candle-embed")]
impl CandleEmbedProvider {
    /// Construct a candle embed provider from
    /// [`blazen_embed_candle::CandleEmbedOptions`].
    ///
    /// Defaults to `sentence-transformers/all-MiniLM-L6-v2` on CPU.
    /// The HF download + weights load is eager — this is an async fn
    /// because `CandleEmbedModel::from_options` is async.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] with provider `"candle-embed"`
    /// if option validation, HF download, or model construction fails.
    pub async fn new(opts: blazen_embed_candle::CandleEmbedOptions) -> Result<Self, BlazenError> {
        let model = blazen_embed_candle::CandleEmbedModel::from_options(opts)
            .await
            .map_err(|e| BlazenError::provider("candle-embed", e.to_string()))?;
        let version = model.model_id().to_owned();
        let metadata =
            ProviderMetadata::new("candle-embed", CapabilityKind::Embedding).with_version(version);
        Ok(Self {
            inner: Arc::new(model),
            metadata,
        })
    }

    /// Construct from an already-loaded `CandleEmbedModel` handle.
    #[must_use]
    pub fn from_model(model: Arc<blazen_embed_candle::CandleEmbedModel>) -> Self {
        let version = model.model_id().to_owned();
        let metadata =
            ProviderMetadata::new("candle-embed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: model,
            metadata,
        }
    }

    /// Borrow the inner backend.
    #[must_use]
    pub fn model(&self) -> &Arc<blazen_embed_candle::CandleEmbedModel> {
        &self.inner
    }
}

#[cfg(feature = "candle-embed")]
impl std::fmt::Debug for CandleEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbedProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "candle-embed")]
impl BaseProvider for CandleEmbedProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "candle-embed")]
#[async_trait]
impl EmbeddingProvider for CandleEmbedProvider {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        let response = self
            .inner
            .embed(&texts)
            .await
            .map_err(|e| BlazenError::provider("candle-embed", e.to_string()))?;
        Ok(response.embeddings)
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}

// ---------------------------------------------------------------------------
// OpenAiEmbeddingProvider — OpenAI /v1/embeddings cloud client
// ---------------------------------------------------------------------------

/// Concrete [`EmbeddingProvider`] backed by
/// [`crate::providers::openai::OpenAiEmbeddingModel`].
///
/// Targets `https://api.openai.com/v1/embeddings` by default. Defaults
/// to `text-embedding-3-small` (1536 dims); use the builder methods on
/// the inner `OpenAiEmbeddingModel` for `with_model` / `with_base_url`
/// (e.g. for `OpenAI`-compatible proxies).
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
pub struct OpenAiEmbeddingProvider {
    inner: Arc<crate::providers::openai::OpenAiEmbeddingModel>,
    metadata: ProviderMetadata,
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
impl OpenAiEmbeddingProvider {
    /// Construct an `OpenAI` embedding provider with the default HTTP
    /// client and the default model
    /// (`text-embedding-3-small`, 1536 dims).
    ///
    /// For non-default models / base URLs, build an
    /// [`crate::providers::openai::OpenAiEmbeddingModel`] directly and
    /// pass it to [`Self::from_model`].
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        let model = crate::providers::openai::OpenAiEmbeddingModel::new(api_key);
        let version = <crate::providers::openai::OpenAiEmbeddingModel as crate::traits::EmbeddingModel>::model_id(&model).to_owned();
        let metadata =
            ProviderMetadata::new("openai-embed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: Arc::new(model),
            metadata,
        }
    }

    /// Construct from a fully configured
    /// [`crate::providers::openai::OpenAiEmbeddingModel`] (use this when
    /// you need to override the model id / base URL / retry config /
    /// HTTP client).
    #[must_use]
    pub fn from_model(model: Arc<crate::providers::openai::OpenAiEmbeddingModel>) -> Self {
        let version = <crate::providers::openai::OpenAiEmbeddingModel as crate::traits::EmbeddingModel>::model_id(&model).to_owned();
        let metadata =
            ProviderMetadata::new("openai-embed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: model,
            metadata,
        }
    }

    /// Borrow the inner model.
    #[must_use]
    pub fn model(&self) -> &Arc<crate::providers::openai::OpenAiEmbeddingModel> {
        &self.inner
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
impl std::fmt::Debug for OpenAiEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbeddingProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
impl BaseProvider for OpenAiEmbeddingProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        let response =
            <crate::providers::openai::OpenAiEmbeddingModel as crate::traits::EmbeddingModel>::embed(
                &self.inner,
                &texts,
            )
            .await?;
        Ok(response.embeddings)
    }

    fn dimensions(&self) -> usize {
        <crate::providers::openai::OpenAiEmbeddingModel as crate::traits::EmbeddingModel>::dimensions(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// FalEmbeddingProvider — fal.ai cloud client (OpenAI-compat through queue)
// ---------------------------------------------------------------------------

/// Concrete [`EmbeddingProvider`] backed by
/// [`crate::providers::fal::FalEmbeddingModel`].
///
/// Targets fal's queue API at
/// `https://queue.fal.run/openrouter/router/openai/v1/embeddings`
/// (submit + poll + fetch). Defaults to `openai/text-embedding-3-small`
/// (1536 dims).
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
pub struct FalEmbeddingProvider {
    inner: Arc<crate::providers::fal::FalEmbeddingModel>,
    metadata: ProviderMetadata,
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
impl FalEmbeddingProvider {
    /// Construct a fal embedding provider with the default HTTP client
    /// and the default model (`openai/text-embedding-3-small`, 1536
    /// dims).
    ///
    /// For non-default models / dimensions, build a
    /// [`crate::providers::fal::FalEmbeddingModel`] directly and pass it
    /// to [`Self::from_model`].
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        let model = crate::providers::fal::FalEmbeddingModel::new(api_key);
        let version =
            <crate::providers::fal::FalEmbeddingModel as crate::traits::EmbeddingModel>::model_id(
                &model,
            )
            .to_owned();
        let metadata =
            ProviderMetadata::new("fal-embed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: Arc::new(model),
            metadata,
        }
    }

    /// Construct from a fully configured
    /// [`crate::providers::fal::FalEmbeddingModel`].
    #[must_use]
    pub fn from_model(model: Arc<crate::providers::fal::FalEmbeddingModel>) -> Self {
        let version =
            <crate::providers::fal::FalEmbeddingModel as crate::traits::EmbeddingModel>::model_id(
                &model,
            )
            .to_owned();
        let metadata =
            ProviderMetadata::new("fal-embed", CapabilityKind::Embedding).with_version(version);
        Self {
            inner: model,
            metadata,
        }
    }

    /// Borrow the inner model.
    #[must_use]
    pub fn model(&self) -> &Arc<crate::providers::fal::FalEmbeddingModel> {
        &self.inner
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
impl std::fmt::Debug for FalEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalEmbeddingProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
impl BaseProvider for FalEmbeddingProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
#[async_trait]
impl EmbeddingProvider for FalEmbeddingProvider {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        let response =
            <crate::providers::fal::FalEmbeddingModel as crate::traits::EmbeddingModel>::embed(
                &self.inner,
                &texts,
            )
            .await?;
        Ok(response.embeddings)
    }

    fn dimensions(&self) -> usize {
        <crate::providers::fal::FalEmbeddingModel as crate::traits::EmbeddingModel>::dimensions(
            &self.inner,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[cfg(feature = "embed-fastembed")]
    #[test]
    fn fastembed_provider_metadata_is_stamped() {
        // We can't actually load a model without network — just verify
        // that the metadata stamp logic compiles + the type implements
        // the right traits. Negative: load with an obviously invalid
        // model name and assert we get a Provider error with id
        // "fastembed".
        use blazen_embed_fastembed::FastEmbedOptions;
        let opts = FastEmbedOptions {
            model_name: Some("definitely-not-a-real-model-xyz".to_owned()),
            ..FastEmbedOptions::default()
        };
        let res = super::FastembedProvider::new(opts);
        assert!(res.is_err(), "expected unknown-model failure");
        if let Err(crate::error::BlazenError::Provider { provider, .. }) = res {
            assert_eq!(provider, "fastembed");
        } else {
            panic!("expected BlazenError::Provider");
        }
    }

    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[test]
    fn openai_provider_metadata_is_stamped() {
        use crate::providers::root::{BaseProvider, CapabilityKind};
        let p = super::OpenAiEmbeddingProvider::new("sk-test");
        assert_eq!(p.provider_id(), "openai-embed");
        assert_eq!(p.capability(), CapabilityKind::Embedding);
        assert_eq!(
            p.metadata().version.as_deref(),
            Some("text-embedding-3-small")
        );
    }

    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[test]
    fn fal_provider_metadata_is_stamped() {
        use crate::providers::root::{BaseProvider, CapabilityKind};
        let p = super::FalEmbeddingProvider::new("fal-test");
        assert_eq!(p.provider_id(), "fal-embed");
        assert_eq!(p.capability(), CapabilityKind::Embedding);
        assert_eq!(
            p.metadata().version.as_deref(),
            Some("openai/text-embedding-3-small")
        );
    }
}
