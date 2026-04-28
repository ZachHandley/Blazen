//! JavaScript bindings for the local embedding provider.
//!
//! Exposes [`JsEmbedProvider`] as a NAPI class with a factory constructor
//! and an async `embed` method.
//!
//! `EmbedModel` is a target-conditional alias re-exported from
//! [`blazen_llm`]:
//!
//! - On non-musl targets, this resolves to `FastEmbedModel` (ONNX Runtime).
//! - On musl targets, this resolves to `TractEmbedModel` (pure-Rust ONNX
//!   inference via tract).
//!
//! The Node binding only sees the unified `EmbedModel` / `EmbedOptions`
//! surface -- backend selection is invisible.

#![cfg(feature = "embed")]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::{EmbedModel, EmbedOptions};

use crate::error::to_napi_error;
use crate::types::embedding::JsEmbedOptions;

// ---------------------------------------------------------------------------
// JsEmbedProvider NAPI class
// ---------------------------------------------------------------------------

/// A local embedding provider.
///
/// Reuses [`JsEmbedOptions`] from [`crate::types::embedding`] so the same
/// option type drives both this standalone provider and the
/// [`crate::types::embedding::JsEmbeddingModel::embed_local`] factory.
///
/// ```javascript
/// const provider = EmbedProvider.create();
/// const vectors = await provider.embed(["hello", "world"]);
/// console.log(vectors.length); // 2
/// ```
#[napi(js_name = "EmbedProvider")]
pub struct JsEmbedProvider {
    inner: Arc<EmbedModel>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation
)]
impl JsEmbedProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new local embedding provider.
    ///
    /// Defaults to `BAAI/bge-small-en-v1.5` (384 dimensions) on the
    /// platform's default ONNX backend.
    #[napi(factory)]
    pub fn create(options: Option<JsEmbedOptions>) -> Result<Self> {
        let opts: EmbedOptions = options.map(Into::into).unwrap_or_default();
        let model = EmbedModel::from_options(opts).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model identifier (e.g. `"Xenova/bge-small-en-v1.5"`).
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
    /// Returns a list of embedding vectors (one per input text). JS
    /// `Number` is `f64`, so vectors are widened from `f32` for transport.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        let response = self.inner.embed(&texts).await.map_err(to_napi_error)?;
        Ok(response
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect())
            .collect())
    }
}
