//! JavaScript bindings for the fastembed (ONNX Runtime) embedding provider.
//!
//! Exposes [`JsFastEmbedModel`] as a NAPI class with a factory
//! constructor and an async `embed` method, alongside the
//! [`JsFastEmbedOptions`] input shape and the [`JsFastEmbedResponse`]
//! output shape.
//!
//! Fastembed runs ONNX inference via the prebuilt `onnxruntime` C++
//! library shipped by Microsoft. This is the high-performance path
//! used on every non-musl target. On musl targets (where the prebuilt
//! native library is unavailable), use
//! [`crate::providers::JsTractEmbedModel`] instead -- it wraps the
//! same fastembed model catalog but runs inference in pure Rust via
//! `tract_onnx`.
//!
//! [`crate::providers::JsEmbedProvider`] continues to provide the
//! target-conditional facade that picks fastembed or tract
//! automatically; this provider exists so callers can opt into
//! fastembed explicitly when they need the typed standalone class.

#![cfg(feature = "fastembed")]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_embed_fastembed::{FastEmbedModel, FastEmbedOptions, FastEmbedResponse};

use crate::error::to_napi_error;

// ---------------------------------------------------------------------------
// JsFastEmbedOptions
// ---------------------------------------------------------------------------

/// Options for the fastembed local embedding backend.
///
/// All fields are optional. Defaults produce a working model using
/// `BGESmallENV15` (Hugging Face: `Xenova/bge-small-en-v1.5`) on CPU
/// with fastembed's built-in cache.
///
/// ```javascript
/// const provider = FastEmbedModel.create({
///   modelName: "BGEBaseENV15",
/// });
/// ```
#[napi(object)]
pub struct JsFastEmbedOptions {
    /// Fastembed model variant name (e.g. `"BGESmallENV15"`).
    /// Parsed via `fastembed::EmbeddingModel::from_str`.
    #[napi(js_name = "modelName")]
    pub model_name: Option<String>,
    /// Model cache directory. When absent, fastembed uses its built-in
    /// cache (controlled by `FASTEMBED_CACHE_DIR` / `HF_HOME` env vars).
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
    /// Maximum batch size for embedding. When absent, fastembed uses
    /// its default (256).
    #[napi(js_name = "maxBatchSize")]
    pub max_batch_size: Option<u32>,
    /// Whether to display download progress when fetching models from
    /// Hugging Face. Defaults to `true`.
    #[napi(js_name = "showDownloadProgress")]
    pub show_download_progress: Option<bool>,
}

impl From<JsFastEmbedOptions> for FastEmbedOptions {
    fn from(val: JsFastEmbedOptions) -> Self {
        Self {
            model_name: val.model_name,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
            max_batch_size: val.max_batch_size.map(|v| v as usize),
            show_download_progress: val.show_download_progress,
        }
    }
}

// ---------------------------------------------------------------------------
// JsFastEmbedResponse
// ---------------------------------------------------------------------------

/// Response from a fastembed embedding operation.
///
/// Mirrors [`blazen_embed_fastembed::FastEmbedResponse`]. Vectors are
/// widened from `f32` to `f64` for transport across the JS boundary.
#[napi(object)]
pub struct JsFastEmbedResponse {
    /// The embedding vectors -- one per input text.
    pub embeddings: Vec<Vec<f64>>,
    /// The model identifier that produced these embeddings.
    pub model: String,
}

impl From<FastEmbedResponse> for JsFastEmbedResponse {
    fn from(val: FastEmbedResponse) -> Self {
        Self {
            embeddings: val
                .embeddings
                .into_iter()
                .map(|v| v.into_iter().map(f64::from).collect())
                .collect(),
            model: val.model,
        }
    }
}

// ---------------------------------------------------------------------------
// JsFastEmbedModel
// ---------------------------------------------------------------------------

/// A local fastembed (ONNX Runtime) embedding model.
///
/// Loads the same fastembed model catalog that backs
/// [`crate::providers::JsEmbedProvider`] on non-musl targets, but
/// exposes the typed standalone class for callers that want explicit
/// feature gating or per-instance options.
///
/// ```javascript
/// const model = FastEmbedModel.create({ modelName: "BGESmallENV15" });
/// const response = await model.embed(["hello", "world"]);
/// console.log(response.embeddings.length); // 2
/// ```
#[napi(js_name = "FastEmbedModel")]
pub struct JsFastEmbedModel {
    inner: Arc<FastEmbedModel>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation
)]
impl JsFastEmbedModel {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new fastembed model.
    ///
    /// This is a synchronous factory even though the underlying
    /// constructor may download model weights from Hugging Face on
    /// first use -- fastembed handles the runtime bridging
    /// internally.
    #[napi(factory)]
    pub fn create(options: Option<JsFastEmbedOptions>) -> Result<Self> {
        let opts: FastEmbedOptions = options.map(Into::into).unwrap_or_default();
        let model = FastEmbedModel::from_options(opts).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the Hugging Face model id this instance was loaded from.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Get the dimensionality of the produced embedding vectors.
    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    // -----------------------------------------------------------------
    // Embed
    // -----------------------------------------------------------------

    /// Embed one or more texts.
    ///
    /// Returns a [`JsFastEmbedResponse`] carrying one vector per input
    /// text and the model identifier that produced them.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<JsFastEmbedResponse> {
        let response = self.inner.embed(&texts).await.map_err(to_napi_error)?;
        Ok(response.into())
    }
}
