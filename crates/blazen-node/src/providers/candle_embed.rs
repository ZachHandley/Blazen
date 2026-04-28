//! JavaScript bindings for the local candle embedding provider.
//!
//! Exposes [`JsCandleEmbedProvider`] as a NAPI class with an async factory
//! constructor, async `embed` method, and `LocalModel` lifecycle controls
//! (`load`, `unload`, `isLoaded`).
//!
//! Runs embedding inference entirely on-device using the candle engine.
//! No API key is required.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::traits::{EmbeddingModel, LocalModel};
use blazen_llm::{CandleEmbedModel, CandleEmbedOptions};

use crate::error::{blazen_error_to_napi, llm_error_to_napi};

// ---------------------------------------------------------------------------
// JsCandleEmbedOptions
// ---------------------------------------------------------------------------

/// Options for the local candle embedding backend.
///
/// All fields are optional. Defaults to `sentence-transformers/all-MiniLM-L6-v2`
/// on CPU.
///
/// ```javascript
/// const provider = await CandleEmbedProvider.create({
///   modelId: "BAAI/bge-small-en-v1.5",
///   device: "cuda:0",
/// });
/// ```
#[napi(object)]
pub struct JsCandleEmbedOptions {
    /// `HuggingFace` model repository ID.
    #[napi(js_name = "modelId")]
    pub model_id: Option<String>,
    /// Hardware device specifier (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub device: Option<String>,
    /// `HuggingFace` revision / git ref.
    pub revision: Option<String>,
    /// Path to cache downloaded models.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

impl From<JsCandleEmbedOptions> for CandleEmbedOptions {
    fn from(val: JsCandleEmbedOptions) -> Self {
        Self {
            model_id: val.model_id,
            device: val.device,
            revision: val.revision,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
        }
    }
}

// ---------------------------------------------------------------------------
// JsCandleEmbedProvider NAPI class
// ---------------------------------------------------------------------------

/// A local candle embedding provider.
///
/// ```javascript
/// const provider = await CandleEmbedProvider.create({
///   modelId: "BAAI/bge-small-en-v1.5",
/// });
/// const vectors = await provider.embed(["hello", "world"]);
/// console.log(vectors.length); // 2
/// ```
#[napi(js_name = "CandleEmbedProvider")]
pub struct JsCandleEmbedProvider {
    inner: Arc<CandleEmbedModel>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation
)]
impl JsCandleEmbedProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new candle embedding provider.
    ///
    /// This is async because candle may download model weights from
    /// `HuggingFace` on first use.
    #[napi(factory)]
    pub async fn create(options: Option<JsCandleEmbedOptions>) -> Result<Self> {
        let opts: CandleEmbedOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                CandleEmbedModel::from_options(opts)
                    .await
                    .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        EmbeddingModel::model_id(self.inner.as_ref()).to_owned()
    }

    /// Get the dimensionality of the produced embedding vectors.
    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        EmbeddingModel::dimensions(self.inner.as_ref()) as u32
    }

    // -----------------------------------------------------------------
    // Embed
    // -----------------------------------------------------------------

    /// Embed one or more texts.
    ///
    /// Returns a list of embedding vectors (one per input text). JS
    /// `Number` is `f64`, so vectors are widened from `f32` for transport.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        let response = EmbeddingModel::embed(self.inner.as_ref(), &texts)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(response
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect())
            .collect())
    }

    // -----------------------------------------------------------------
    // LocalModel lifecycle
    // -----------------------------------------------------------------

    /// Explicitly load the model weights into memory / `VRAM`.
    #[napi]
    pub async fn load(&self) -> Result<()> {
        LocalModel::load(self.inner.as_ref())
            .await
            .map_err(blazen_error_to_napi)
    }

    /// Drop the loaded model and free its memory / `VRAM`.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        LocalModel::unload(self.inner.as_ref())
            .await
            .map_err(blazen_error_to_napi)
    }

    /// Whether the model is currently loaded.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> bool {
        LocalModel::is_loaded(self.inner.as_ref()).await
    }
}
