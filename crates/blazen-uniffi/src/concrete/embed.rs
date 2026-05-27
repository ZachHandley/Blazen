//! Per-engine embedding `#[uniffi::Object]` providers.
//!
//! Each `<Engine>Provider` here is a thin UniFFI-exported wrapper around
//! the matching canonical concrete provider in
//! [`blazen_llm::providers::concrete::embed`]. Foreign bindgen (Go /
//! Swift / Kotlin / Ruby) emits a real class per engine —
//! `FastembedProvider`, `TractEmbedProvider`, `CandleEmbedProvider`,
//! `OpenAiEmbeddingProvider`, `FalEmbeddingProvider` — rather than
//! overloading the central `EmbeddingModel` opaque factory in
//! [`crate::compute`].
//!
//! Embedding routes through
//! [`blazen_llm::providers::capabilities::EmbeddingProvider`] (the
//! upstream capability trait — distinct from the same-named
//! [`crate::concrete::bases::EmbeddingProvider`] trait re-declared on
//! this crate's binding surface).
//!
//! ## Constructor argument shape
//!
//! The upstream concretes accept rich `*Options` structs
//! (`FastEmbedOptions`, `TractOptions`, `CandleEmbedOptions`) carrying
//! `PathBuf` / `Option<usize>` / `Option<bool>` fields that don't map
//! cleanly to a flat uniffi `Record`. Each wrapper here flattens the
//! two most useful fields (`model_id` + `cache_dir`) into the
//! constructor signature and builds the options struct internally; the
//! remaining defaults (batch size, download progress, device, revision)
//! match the underlying crate's `*Options::default()`.

#![allow(unused_imports)]

use std::sync::Arc;

use crate::errors::BlazenError;

// ---------------------------------------------------------------------------
// FastembedProvider — local fastembed (ORT) text embeddings
// ---------------------------------------------------------------------------

/// Concrete embedding provider wrapping
/// [`blazen_llm::providers::concrete::embed::FastembedProvider`].
///
/// Only available on ORT-supported targets
/// (`x86_64-{linux,windows,macos}`, `aarch64-apple-darwin`). On other
/// targets use [`TractEmbedProvider`] instead — same model catalog,
/// pure-Rust runtime.
#[cfg(feature = "fastembed")]
#[derive(uniffi::Object)]
pub struct FastembedProvider {
    inner: Arc<blazen_llm::providers::concrete::embed::FastembedProvider>,
}

#[cfg(feature = "fastembed")]
#[uniffi::export(async_runtime = "tokio")]
impl FastembedProvider {
    /// Construct a fastembed provider.
    ///
    /// `model_id` selects a fastembed model variant by debug-spelling
    /// name (e.g. `"BGESmallENV15"`, `"AllMiniLML6V2"`); `None` resolves
    /// to fastembed's default (`BGESmallENV15`). `cache_dir` overrides
    /// the model-weights cache directory; `None` uses fastembed's
    /// built-in cache (driven by `FASTEMBED_CACHE_DIR` / `HF_HOME`).
    #[uniffi::constructor]
    pub fn new(
        model_id: Option<String>,
        cache_dir: Option<String>,
    ) -> Result<Arc<Self>, BlazenError> {
        let opts = blazen_embed_fastembed::FastEmbedOptions {
            model_name: model_id,
            cache_dir: cache_dir.map(std::path::PathBuf::from),
            ..blazen_embed_fastembed::FastEmbedOptions::default()
        };
        let inner = blazen_llm::providers::concrete::embed::FastembedProvider::new(opts)?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Compute embedding vectors for each input string.
    pub async fn embed(self: Arc<Self>, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    /// Return the dimensionality of vectors produced by [`embed`](Self::embed).
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

#[cfg(feature = "fastembed")]
#[uniffi::export]
impl FastembedProvider {
    /// Synchronous variant of [`embed`](Self::embed).
    pub fn embed_blocking(
        self: Arc<Self>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.embed(texts).await })
    }
}

// ---------------------------------------------------------------------------
// TractEmbedProvider — local tract (pure-Rust ONNX) text embeddings
// ---------------------------------------------------------------------------

/// Concrete embedding provider wrapping
/// [`blazen_llm::providers::concrete::embed::TractEmbedProvider`].
///
/// Drop-in replacement for [`FastembedProvider`] on targets that lack
/// ORT prebuilt binaries (musl Linux, aarch64-linux, wasm). Loads the
/// same fastembed model catalog via `tract_onnx`.
#[cfg(feature = "tract")]
#[derive(uniffi::Object)]
pub struct TractEmbedProvider {
    inner: Arc<blazen_llm::providers::concrete::embed::TractEmbedProvider>,
}

#[cfg(feature = "tract")]
#[uniffi::export(async_runtime = "tokio")]
impl TractEmbedProvider {
    /// Construct a tract embed provider.
    ///
    /// `model_id` selects a model variant by debug-spelling name
    /// (case-insensitive, same catalog as [`FastembedProvider`]); `None`
    /// resolves to `BGESmallENV15`. `cache_dir` overrides the model
    /// cache directory; `None` falls back to `blazen-model-cache`'s
    /// default.
    #[uniffi::constructor]
    pub fn new(
        model_id: Option<String>,
        cache_dir: Option<String>,
    ) -> Result<Arc<Self>, BlazenError> {
        let opts = blazen_embed_tract::TractOptions {
            model_name: model_id,
            cache_dir: cache_dir.map(std::path::PathBuf::from),
            ..blazen_embed_tract::TractOptions::default()
        };
        let inner = blazen_llm::providers::concrete::embed::TractEmbedProvider::new(opts)?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Compute embedding vectors for each input string.
    pub async fn embed(self: Arc<Self>, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    /// Return the dimensionality of vectors produced by [`embed`](Self::embed).
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

#[cfg(feature = "tract")]
#[uniffi::export]
impl TractEmbedProvider {
    /// Synchronous variant of [`embed`](Self::embed).
    pub fn embed_blocking(
        self: Arc<Self>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.embed(texts).await })
    }
}

// ---------------------------------------------------------------------------
// CandleEmbedProvider — local candle BERT-family text embeddings
// ---------------------------------------------------------------------------

/// Concrete embedding provider wrapping
/// [`blazen_llm::providers::concrete::embed::CandleEmbedProvider`].
///
/// Loads BERT-family sentence-transformer weights from `HuggingFace`
/// Hub and runs the forward pass via candle. Supports CPU, CUDA, and
/// Metal devices (the `engine` feature on `blazen-embed-candle` must
/// also be enabled for actual inference).
#[cfg(feature = "candle-embed")]
#[derive(uniffi::Object)]
pub struct CandleEmbedProvider {
    inner: Arc<blazen_llm::providers::concrete::embed::CandleEmbedProvider>,
}

#[cfg(feature = "candle-embed")]
#[uniffi::export(async_runtime = "tokio")]
impl CandleEmbedProvider {
    /// Construct a candle embed provider.
    ///
    /// `model_id` is the HuggingFace repo id
    /// (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`); `None`
    /// defaults to `sentence-transformers/all-MiniLM-L6-v2`. `cache_dir`
    /// overrides the model cache directory; `None` falls back to
    /// `blazen-model-cache`'s default.
    ///
    /// The underlying `CandleEmbedProvider::new` is async (HF download
    /// + weights load). UniFFI constructors are sync, so we drive it
    /// through the shared tokio [`crate::runtime`] — matching the
    /// `TripoSrProvider` pattern in [`crate::concrete::three_d`].
    #[uniffi::constructor]
    pub fn new(
        model_id: Option<String>,
        cache_dir: Option<String>,
    ) -> Result<Arc<Self>, BlazenError> {
        let opts = blazen_llm::CandleEmbedOptions {
            model_id,
            cache_dir: cache_dir.map(std::path::PathBuf::from),
            ..blazen_llm::CandleEmbedOptions::default()
        };
        let inner = crate::runtime::runtime()
            .block_on(blazen_llm::providers::concrete::embed::CandleEmbedProvider::new(opts))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Compute embedding vectors for each input string.
    pub async fn embed(self: Arc<Self>, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    /// Return the dimensionality of vectors produced by [`embed`](Self::embed).
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

#[cfg(feature = "candle-embed")]
#[uniffi::export]
impl CandleEmbedProvider {
    /// Synchronous variant of [`embed`](Self::embed).
    pub fn embed_blocking(
        self: Arc<Self>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.embed(texts).await })
    }
}

// ---------------------------------------------------------------------------
// OpenAiEmbeddingProvider — OpenAI /v1/embeddings cloud client
// ---------------------------------------------------------------------------

/// Concrete embedding provider wrapping
/// [`blazen_llm::providers::concrete::embed::OpenAiEmbeddingProvider`].
///
/// Targets `https://api.openai.com/v1/embeddings`. Defaults to
/// `text-embedding-3-small` (1536 dims). `model` overrides the default
/// model id — note that the `OpenAiEmbeddingModel` builder couples
/// model and dimensions, so this wrapper only exposes the model name
/// and leaves dimensions at the OpenAI-published default for the
/// chosen model (1536 for `-small`, 3072 for `-large`).
#[derive(uniffi::Object)]
pub struct OpenAiEmbeddingProvider {
    inner: Arc<blazen_llm::providers::concrete::embed::OpenAiEmbeddingProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl OpenAiEmbeddingProvider {
    /// Construct an `OpenAI` embedding provider.
    ///
    /// `model` overrides the default `text-embedding-3-small` model id
    /// when `Some`. For full control over dimensions / base URL / retry
    /// config, build a
    /// [`blazen_llm::providers::openai::OpenAiEmbeddingModel`] directly
    /// and pass it via [`OpenAiEmbeddingProvider::from_model`] (not
    /// exposed across the foreign binding surface — Rust-side only).
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner_model = blazen_llm::providers::openai::OpenAiEmbeddingModel::new(api_key);
        if let Some(m) = model {
            // OpenAI's published dim defaults: 1536 for `-small`, 3072 for `-large`,
            // 1536 for the legacy `ada-002`. Default to 1536 if we can't tell.
            let dims = if m.contains("3-large") { 3072 } else { 1536 };
            inner_model = inner_model.with_model(m, dims);
        }
        let inner = blazen_llm::providers::concrete::embed::OpenAiEmbeddingProvider::from_model(
            Arc::new(inner_model),
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Compute embedding vectors for each input string.
    pub async fn embed(self: Arc<Self>, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    /// Return the dimensionality of vectors produced by [`embed`](Self::embed).
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

#[uniffi::export]
impl OpenAiEmbeddingProvider {
    /// Synchronous variant of [`embed`](Self::embed).
    pub fn embed_blocking(
        self: Arc<Self>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.embed(texts).await })
    }
}

// ---------------------------------------------------------------------------
// FalEmbeddingProvider — fal.ai cloud embeddings
// ---------------------------------------------------------------------------

/// Concrete embedding provider wrapping
/// [`blazen_llm::providers::concrete::embed::FalEmbeddingProvider`].
///
/// Targets fal's queue API at
/// `https://queue.fal.run/openrouter/router/openai/v1/embeddings`.
/// Defaults to `openai/text-embedding-3-small` (1536 dims).
#[derive(uniffi::Object)]
pub struct FalEmbeddingProvider {
    inner: Arc<blazen_llm::providers::concrete::embed::FalEmbeddingProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalEmbeddingProvider {
    /// Construct a fal embedding provider.
    ///
    /// `model` overrides the default `openai/text-embedding-3-small`
    /// model id when `Some`.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner_model = blazen_llm::providers::fal::FalEmbeddingModel::new(api_key);
        if let Some(m) = model {
            inner_model = inner_model.with_model(m);
        }
        let inner = blazen_llm::providers::concrete::embed::FalEmbeddingProvider::from_model(
            Arc::new(inner_model),
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Compute embedding vectors for each input string.
    pub async fn embed(self: Arc<Self>, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    /// Return the dimensionality of vectors produced by [`embed`](Self::embed).
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

#[uniffi::export]
impl FalEmbeddingProvider {
    /// Synchronous variant of [`embed`](Self::embed).
    pub fn embed_blocking(
        self: Arc<Self>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.embed(texts).await })
    }
}

// ---------------------------------------------------------------------------
// Polymorphic capability-base trait impls
// ---------------------------------------------------------------------------
//
// Each engine implements both [`crate::concrete::bases::BaseProvider`] and
// [`crate::concrete::bases::EmbeddingProvider`] so foreign (Kotlin/Swift/Go)
// consumers can hold a polymorphic `EmbeddingProvider` reference and
// Rust-side code can collect engines into capability-erased
// `Arc<dyn BaseProvider>` containers. The inherent `embed` methods on
// each engine (which use `self: Arc<Self>`) continue to take precedence
// at the call site `engine.embed(...)`; the trait methods (which use
// `&self`) are reachable via UFCS / `dyn EmbeddingProvider` dispatch.

// FastembedProvider --------------------------------------------------------

#[cfg(feature = "fastembed")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FastembedProvider {
    fn provider_id(&self) -> String {
        "fastembed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }
}

#[cfg(feature = "fastembed")]
#[async_trait::async_trait]
impl crate::concrete::bases::EmbeddingProvider for FastembedProvider {
    fn provider_id(&self) -> String {
        "fastembed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

// TractEmbedProvider -------------------------------------------------------

#[cfg(feature = "tract")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for TractEmbedProvider {
    fn provider_id(&self) -> String {
        "tract-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }
}

#[cfg(feature = "tract")]
#[async_trait::async_trait]
impl crate::concrete::bases::EmbeddingProvider for TractEmbedProvider {
    fn provider_id(&self) -> String {
        "tract-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

// CandleEmbedProvider ------------------------------------------------------

#[cfg(feature = "candle-embed")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for CandleEmbedProvider {
    fn provider_id(&self) -> String {
        "candle-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }
}

#[cfg(feature = "candle-embed")]
#[async_trait::async_trait]
impl crate::concrete::bases::EmbeddingProvider for CandleEmbedProvider {
    fn provider_id(&self) -> String {
        "candle-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

// OpenAiEmbeddingProvider --------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for OpenAiEmbeddingProvider {
    fn provider_id(&self) -> String {
        "openai-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::EmbeddingProvider for OpenAiEmbeddingProvider {
    fn provider_id(&self) -> String {
        "openai-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}

// FalEmbeddingProvider -----------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FalEmbeddingProvider {
    fn provider_id(&self) -> String {
        "fal-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::EmbeddingProvider for FalEmbeddingProvider {
    fn provider_id(&self) -> String {
        "fal-embed".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Embedding
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        Ok(self.inner.embed(texts).await?)
    }

    fn dimensions(&self) -> u32 {
        use blazen_llm::providers::capabilities::EmbeddingProvider as _;
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}
