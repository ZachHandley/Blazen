//! `wasm-bindgen` wrapper for [`blazen_embed_tract::TractEmbedModel`].
//!
//! `tract` is the only embedding backend that compiles to `wasm32`, so it is
//! the natural fit for in-browser local embeddings without relying on a JS
//! library like `@huggingface/transformers`.
//!
//! Two surfaces are exposed:
//!
//! * [`WasmTractEmbedModel`] — a standalone class published as `TractEmbedModel`
//!   in JS, with `modelId` / `dimensions` getters and an async `embed()`.
//! * [`TractEmbeddingAdapter`] — internal `EmbeddingModel` trait impl used by
//!   the `EmbeddingModel.tract()` factory on [`WasmEmbeddingModel`] so the
//!   tract backend slots into the same pipeline as the OpenAI / Cohere /
//!   transformers.js variants.
//!
//! ```js
//! import init, { TractEmbedModel, TractOptions } from '@blazen/sdk';
//! await init();
//!
//! const opts = new TractOptions();
//! opts.modelName = 'BGESmallENV15';
//! const model = await TractEmbedModel.create(opts);
//! const vectors = await model.embed(['Hello world']);
//! ```

use std::pin::Pin;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_embed_tract::{TractEmbedModel, TractOptions};
use blazen_llm::types::EmbeddingResponse;
use blazen_llm::BlazenError;

use crate::embedding::WasmEmbeddingModel;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as js_embedding.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-`Send` future.
///
/// SAFETY: WASM is single-threaded so a vacuous `Send` impl is sound.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: we are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// WasmTractOptions
// ---------------------------------------------------------------------------

/// Construction options for [`WasmTractEmbedModel`].
///
/// Mirrors the relevant subset of [`blazen_embed_tract::TractOptions`] that
/// makes sense to expose to JavaScript. The `cache_dir` field is intentionally
/// omitted because there is no useful filesystem path for browser callers; the
/// `show_download_progress` flag is omitted for the same reason.
#[wasm_bindgen(js_name = "TractOptions")]
#[derive(Default)]
pub struct WasmTractOptions {
    model_name: Option<String>,
    max_batch_size: Option<u32>,
}

#[wasm_bindgen(js_class = "TractOptions")]
impl WasmTractOptions {
    /// Create default options. Resolves to the fastembed default model
    /// (`BGESmallENV15`) when `modelName` is left unset.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Model name matching a variant of fastembed's `EmbeddingModel` enum
    /// (e.g. `"BGESmallENV15"`). Case-insensitive. `null` resolves to the
    /// fastembed default.
    #[wasm_bindgen(getter, js_name = "modelName")]
    #[must_use]
    pub fn model_name(&self) -> Option<String> {
        self.model_name.clone()
    }

    /// Setter for [`Self::model_name`].
    #[wasm_bindgen(setter, js_name = "modelName")]
    pub fn set_model_name(&mut self, value: Option<String>) {
        self.model_name = value;
    }

    /// Optional maximum batch size override for inference. `null` lets the
    /// backend run the whole input vector in one pass.
    #[wasm_bindgen(getter, js_name = "maxBatchSize")]
    #[must_use]
    pub fn max_batch_size(&self) -> Option<u32> {
        self.max_batch_size
    }

    /// Setter for [`Self::max_batch_size`].
    #[wasm_bindgen(setter, js_name = "maxBatchSize")]
    pub fn set_max_batch_size(&mut self, value: Option<u32>) {
        self.max_batch_size = value;
    }
}

impl WasmTractOptions {
    /// Convert into the upstream [`TractOptions`] shape.
    fn into_tract_options(self) -> TractOptions {
        TractOptions {
            model_name: self.model_name,
            cache_dir: None,
            max_batch_size: self.max_batch_size.map(|n| n as usize),
            show_download_progress: None,
        }
    }
}

// ---------------------------------------------------------------------------
// TractEmbeddingAdapter — EmbeddingModel trait impl for the tract backend
// ---------------------------------------------------------------------------

/// Internal adapter that implements [`blazen_llm::traits::EmbeddingModel`] on
/// top of a [`TractEmbedModel`]. This is what backs both [`WasmTractEmbedModel`]
/// and the `EmbeddingModel.tract()` factory on [`WasmEmbeddingModel`].
pub(crate) struct TractEmbeddingAdapter {
    model: Arc<TractEmbedModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for TractEmbeddingAdapter {}
unsafe impl Sync for TractEmbeddingAdapter {}

impl TractEmbeddingAdapter {
    pub(crate) fn new(model: Arc<TractEmbedModel>) -> Self {
        Self { model }
    }

    async fn embed_impl(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let response = self
            .model
            .embed(texts)
            .await
            .map_err(|e| BlazenError::provider("tract", e.to_string()))?;
        Ok(EmbeddingResponse {
            embeddings: response.embeddings,
            model: response.model,
            usage: None,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }
}

#[async_trait::async_trait]
impl blazen_llm::traits::EmbeddingModel for TractEmbeddingAdapter {
    fn model_id(&self) -> &str {
        self.model.model_id()
    }

    fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        // SAFETY: WASM is single-threaded, Send is vacuously satisfied.
        SendFuture(self.embed_impl(texts)).await
    }
}

// ---------------------------------------------------------------------------
// WasmTractEmbedModel — standalone JS class
// ---------------------------------------------------------------------------

/// A local embedding model backed by `tract-onnx` (pure-Rust ONNX inference).
///
/// `tract` is the only embedding backend that compiles to `wasm32`, so this is
/// the natural choice for fully-local in-browser embeddings without depending
/// on a JavaScript library.
///
/// ```js
/// const opts = new TractOptions();
/// opts.modelName = 'BGESmallENV15';
/// const model = await TractEmbedModel.create(opts);
/// const vectors = await model.embed(['Hello world']);
/// ```
#[wasm_bindgen(js_name = "TractEmbedModel")]
pub struct WasmTractEmbedModel {
    inner: Arc<TractEmbedModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmTractEmbedModel {}
unsafe impl Sync for WasmTractEmbedModel {}

#[wasm_bindgen(js_class = "TractEmbedModel")]
impl WasmTractEmbedModel {
    /// Create a new tract-backed embedding model.
    ///
    /// Loading downloads the ONNX weights and tokenizer for the requested
    /// model on first call; subsequent calls reuse the local cache.
    ///
    /// Returns a `Promise<TractEmbedModel>`.
    ///
    /// # Errors
    ///
    /// The returned promise rejects when the model name is unknown or when
    /// downloading / parsing the ONNX graph fails.
    #[wasm_bindgen(js_name = "create")]
    pub fn create(options: Option<WasmTractOptions>) -> js_sys::Promise {
        let opts = options.unwrap_or_default().into_tract_options();
        future_to_promise(async move {
            // `TractEmbedModel::from_options` is sync but performs blocking
            // downloads internally; on wasm there is no `block_in_place` so
            // the upstream impl must run inside a current-thread runtime that
            // it sets up itself. We just call it directly.
            let model = TractEmbedModel::from_options(opts)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let wrapper = WasmTractEmbedModel {
                inner: Arc::new(model),
            };
            Ok(JsValue::from(wrapper))
        })
    }

    /// The Hugging Face model id this instance was loaded from
    /// (e.g. `"Xenova/bge-small-en-v1.5"`).
    #[wasm_bindgen(getter, js_name = "modelId")]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Output embedding dimensionality.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.dimensions() as u32
        }
    }

    /// Embed one or more texts, returning a nested array of float vectors.
    ///
    /// Returns a `Promise<Float32Array[]>`.
    #[wasm_bindgen]
    pub fn embed(&self, texts: Vec<String>) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let response = model
                .embed(&texts)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let outer = js_sys::Array::new_with_length(response.embeddings.len() as u32);
            for (i, embedding) in response.embeddings.iter().enumerate() {
                let inner = js_sys::Float32Array::new_with_length(embedding.len() as u32);
                inner.copy_from(embedding);
                outer.set(i as u32, inner.into());
            }

            Ok(outer.into())
        })
    }
}

// ---------------------------------------------------------------------------
// EmbeddingModel.tract() factory
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_class = "EmbeddingModel")]
impl WasmEmbeddingModel {
    /// Local embedding via `tract-onnx` (pure-Rust ONNX inference).
    ///
    /// This is the only embedding backend that runs entirely inside the WASM
    /// module — no JS libraries required. Weights and tokenizer are downloaded
    /// on first use and cached locally.
    ///
    /// Returns a `Promise<EmbeddingModel>`.
    ///
    /// ```js
    /// const opts = new TractOptions();
    /// opts.modelName = 'BGESmallENV15';
    /// const embedder = await EmbeddingModel.tract(opts);
    /// const vecs = await embedder.embed(['Hello world']);
    /// ```
    #[wasm_bindgen(js_name = "tract")]
    pub fn tract(options: Option<WasmTractOptions>) -> js_sys::Promise {
        let opts = options.unwrap_or_default().into_tract_options();
        future_to_promise(async move {
            let model = TractEmbedModel::from_options(opts)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let adapter = TractEmbeddingAdapter::new(Arc::new(model));
            let wrapper = WasmEmbeddingModel::from_inner(Arc::new(adapter));
            Ok(JsValue::from(wrapper))
        })
    }
}
