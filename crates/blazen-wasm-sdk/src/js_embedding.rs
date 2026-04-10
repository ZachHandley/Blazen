//! JS callback-based embedding model for local/in-browser inference.
//!
//! [`JsEmbeddingHandler`] implements [`EmbeddingModel`] by delegating to a
//! user-supplied JavaScript function. This lets browser users wrap libraries
//! like `transformers.js` or ONNX Runtime Web and use them from Blazen's
//! Rust WASM code.
//!
//! ```js
//! import { EmbeddingModel } from '@blazen/sdk';
//! import { pipeline } from '@xenova/transformers';
//!
//! const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
//! const embedder = EmbeddingModel.fromJsHandler('all-MiniLM-L6-v2', 384, async (texts) => {
//!   const results = await pipe(texts, { pooling: 'mean', normalize: true });
//!   return Array.from({ length: texts.length }, (_, i) => results[i].data);
//! });
//!
//! const vecs = await embedder.embed(['Hello world']);
//! ```

use std::pin::Pin;

use js_sys::{Array, Float32Array, Function, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use blazen_llm::types::EmbeddingResponse;
use blazen_llm::BlazenError;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as agent.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// JsEmbeddingHandler
// ---------------------------------------------------------------------------

/// An [`EmbeddingModel`] backed by a user-supplied JavaScript function.
///
/// The JS handler receives a `string[]` and must return
/// `Promise<Float32Array[]> | Float32Array[]`.
pub struct JsEmbeddingHandler {
    model_id: String,
    dimensions: usize,
    handler: Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsEmbeddingHandler {}
unsafe impl Sync for JsEmbeddingHandler {}

impl JsEmbeddingHandler {
    /// Create a new handler wrapping the given JS callback.
    #[must_use]
    pub fn new(model_id: String, dimensions: usize, handler: Function) -> Self {
        Self {
            model_id,
            dimensions,
            handler,
        }
    }

    /// The actual async embed implementation (non-Send).
    #[allow(clippy::cast_possible_truncation)]
    async fn embed_impl(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        // Build a JS Array of strings from `texts`.
        let js_array = Array::new_with_length(texts.len() as u32);
        for (i, text) in texts.iter().enumerate() {
            js_array.set(i as u32, JsValue::from_str(text));
        }

        // Call the JS handler.
        let result = self
            .handler
            .call1(&JsValue::NULL, &js_array)
            .map_err(|e| BlazenError::provider("js_embedding", format!("{e:?}")))?;

        // If the result is a Promise, await it.
        let result = if result.has_type::<Promise>() {
            let promise: Promise = result.unchecked_into();
            JsFuture::from(promise)
                .await
                .map_err(|e| BlazenError::provider("js_embedding", format!("{e:?}")))?
        } else {
            result
        };

        // Parse result as Array of Float32Array, convert to Vec<Vec<f32>>.
        let outer_array: Array = result.dyn_into().map_err(|_| {
            BlazenError::provider(
                "js_embedding",
                "handler must return an array of Float32Array",
            )
        })?;

        let len = outer_array.length();
        let mut embeddings = Vec::with_capacity(len as usize);

        for i in 0..len {
            let item = outer_array.get(i);
            let typed: Float32Array = item.dyn_into().map_err(|_| {
                BlazenError::provider(
                    "js_embedding",
                    format!("element at index {i} is not a Float32Array"),
                )
            })?;
            embeddings.push(typed.to_vec());
        }

        Ok(EmbeddingResponse {
            embeddings,
            model: self.model_id.clone(),
            usage: None,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }
}

#[async_trait::async_trait]
impl blazen_llm::traits::EmbeddingModel for JsEmbeddingHandler {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        // SAFETY: WASM is single-threaded, Send is vacuously satisfied.
        SendFuture(self.embed_impl(texts)).await
    }
}
