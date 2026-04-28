//! JavaScript bindings for the tract (pure-Rust ONNX) embedding provider.
//!
//! Exposes [`JsTractEmbedModel`] as a NAPI class with a factory
//! constructor and an async `embed` method, alongside the
//! [`JsTractOptions`] input shape and the [`JsTractResponse`] output
//! shape.
//!
//! Tract runs ONNX inference entirely in pure Rust, with no
//! `onnxruntime` C++ dependency. This is the musl / minimal-binary
//! equivalent of `FastEmbedProvider`. On non-musl targets the
//! `EmbedProvider` facade resolves to fastembed; this provider lets
//! callers opt into tract explicitly when they need a smaller binary
//! or want to avoid linking the ONNX Runtime native library.

#![cfg(feature = "tract")]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_embed_tract::{TractEmbedModel, TractOptions, TractResponse};

use crate::error::to_napi_error;

// ---------------------------------------------------------------------------
// JsTractOptions
// ---------------------------------------------------------------------------

/// Options for the tract local embedding backend.
///
/// All fields are optional. Defaults produce a working model using
/// `BGESmallENV15` (Hugging Face: `Xenova/bge-small-en-v1.5`) on the
/// platform's default model cache directory.
///
/// ```javascript
/// const provider = TractEmbedModel.create({
///   modelName: "BGEBaseENV15",
/// });
/// ```
#[napi(object)]
pub struct JsTractOptions {
    /// Tract model variant name matching the fastembed registry
    /// (e.g. `"BGESmallENV15"`). Case-insensitive.
    #[napi(js_name = "modelName")]
    pub model_name: Option<String>,
    /// Model cache directory. When absent, uses the default from
    /// `blazen_model_cache`.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
    /// Maximum batch size for embedding. When absent, the entire input
    /// vector is processed in a single forward pass.
    #[napi(js_name = "maxBatchSize")]
    pub max_batch_size: Option<u32>,
    /// Whether to display download progress when fetching models from
    /// Hugging Face.
    #[napi(js_name = "showDownloadProgress")]
    pub show_download_progress: Option<bool>,
}

impl From<JsTractOptions> for TractOptions {
    fn from(val: JsTractOptions) -> Self {
        Self {
            model_name: val.model_name,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
            max_batch_size: val.max_batch_size.map(|v| v as usize),
            show_download_progress: val.show_download_progress,
        }
    }
}

// ---------------------------------------------------------------------------
// JsTractResponse
// ---------------------------------------------------------------------------

/// Response from a tract embedding operation.
///
/// Mirrors [`blazen_embed_tract::TractResponse`]. Vectors are widened
/// from `f32` to `f64` for transport across the JS boundary.
#[napi(object)]
pub struct JsTractResponse {
    /// The embedding vectors -- one per input text.
    pub embeddings: Vec<Vec<f64>>,
    /// The model identifier that produced these embeddings.
    pub model: String,
}

impl From<TractResponse> for JsTractResponse {
    fn from(val: TractResponse) -> Self {
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
// JsTractEmbedModel
// ---------------------------------------------------------------------------

/// A local tract (pure-Rust ONNX) embedding model.
///
/// Loads the same fastembed model catalog via tract-onnx instead of
/// the ONNX Runtime native library. This is the musl-friendly
/// equivalent of [`crate::providers::JsFastEmbedModel`].
///
/// ```javascript
/// const model = TractEmbedModel.create({ modelName: "BGESmallENV15" });
/// const response = await model.embed(["hello", "world"]);
/// console.log(response.embeddings.length); // 2
/// ```
#[napi(js_name = "TractEmbedModel")]
pub struct JsTractEmbedModel {
    inner: Arc<TractEmbedModel>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation
)]
impl JsTractEmbedModel {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new tract embedding model.
    ///
    /// This is a synchronous factory even though the underlying
    /// constructor may download model weights from Hugging Face on
    /// first use -- tract's loader handles the runtime bridging
    /// internally.
    #[napi(factory)]
    pub fn create(options: Option<JsTractOptions>) -> Result<Self> {
        let opts: TractOptions = options.map(Into::into).unwrap_or_default();
        let model = TractEmbedModel::from_options(opts).map_err(to_napi_error)?;
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
    /// Returns a [`JsTractResponse`] carrying one L2-normalized vector
    /// per input text and the model identifier that produced them.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<JsTractResponse> {
        let response = self.inner.embed(&texts).await.map_err(to_napi_error)?;
        Ok(response.into())
    }
}
