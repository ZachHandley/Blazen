//! JS-callback ABCs that fill the audit's `ImageModel` / `EmbedModel` slots
//! for the WASM SDK.
//!
//! - [`WasmImageModel`] mirrors the audit's `ImageModel` slot (an alias for
//!   [`blazen_llm::compute::ImageGeneration`]). It's exposed as a plain
//!   `#[wasm_bindgen]` class that delegates `generateImage` and
//!   `upscaleImage` to JS callbacks. Native compute providers
//!   (e.g. `fal.ai`) are still preferred when an HTTP backend is acceptable;
//!   this ABC exists so callers can plug in fully-local image pipelines
//!   (`webgpu` diffusion, on-device upscalers, etc.) and still satisfy
//!   the binding-parity audit.
//! - [`WasmEmbedModel`] is the JS-callback ABC variant of
//!   [`crate::embedding::WasmEmbeddingModel`]. The existing
//!   [`crate::embedding::WasmEmbeddingModel::from_js_handler`] already
//!   accepts a JS callback, but it's exposed as `EmbeddingModel` (matching
//!   the runtime trait `blazen_llm::traits::EmbeddingModel`). The audit
//!   separately tracks the `EmbedModel` name (re-exported from
//!   `blazen-embed` as a struct alias for FastEmbed/Tract); this thin
//!   wrapper exposes the same JS-callback surface under the `EmbedModel`
//!   name so neither the runtime trait nor the model-cache symbol shows
//!   up as an unbound gap.
//!
//! Both classes hold their JS callbacks behind the standard "WASM is
//! single-threaded so `Send`/`Sync` are vacuous" pattern used elsewhere in
//! this crate.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Call a `JsValue` that is expected to be a function with one argument and
/// await the returned value if it is a `Promise`.
pub(crate) async fn call1_await(
    func: &js_sys::Function,
    this: &JsValue,
    arg: JsValue,
) -> Result<JsValue, JsValue> {
    let result = func
        .call1(this, &arg)
        .map_err(|e| JsValue::from_str(&format!("callback threw: {e:?}")))?;
    if result.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = result.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("callback rejected: {e:?}")))
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_MODEL_ABCS: &str = r#"
/**
 * Implementation object for `ImageModel`. Each method may return a value
 * directly or a `Promise` that resolves to it.
 *
 * - `generateImage(request)` -- generate an image from a prompt; returns an
 *   `ImageResult`-shaped object.
 * - `upscaleImage(request)` -- upscale an existing image; returns an
 *   `ImageResult`-shaped object.
 */
export interface ImageModelImpl {
    generateImage(request: any): Promise<any> | any;
    upscaleImage(request: any): Promise<any> | any;
}

/**
 * Implementation object for `EmbedModel`. Each method may return a value
 * directly or a `Promise` that resolves to it.
 *
 * - `embed(texts)` -- embed an array of strings; returns
 *   `Float32Array[]` or `number[][]`.
 * - `dimensions` -- the dimensionality of the produced vectors.
 * - `modelId` -- identifier string for this embedding model.
 */
export interface EmbedModelImpl {
    embed(texts: string[]): Promise<Float32Array[] | number[][]> | Float32Array[] | number[][];
    dimensions: number;
    modelId: string;
}
"#;

// ---------------------------------------------------------------------------
// WasmImageModel — JS-callback ABC for `ImageGeneration`
// ---------------------------------------------------------------------------

/// JS-callback ABC for [`blazen_llm::compute::ImageGeneration`] (the
/// `ImageModel` alias).
///
/// Wraps a JS object exposing `generateImage` and `upscaleImage` methods so
/// users can plug a local image pipeline (browser-side diffusion, WebGPU
/// upscalers, etc.) into Blazen's compute surface without writing Rust.
#[wasm_bindgen(js_name = "ImageModel")]
pub struct WasmImageModel {
    impl_obj: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmImageModel {}
unsafe impl Sync for WasmImageModel {}

#[wasm_bindgen(js_class = "ImageModel")]
impl WasmImageModel {
    /// Wrap a JS object implementing the `ImageModelImpl` interface.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `impl_obj` is missing one of the
    /// required methods (`generateImage`, `upscaleImage`).
    #[wasm_bindgen(constructor)]
    pub fn new(impl_obj: JsValue) -> Result<WasmImageModel, JsValue> {
        for method in &["generateImage", "upscaleImage"] {
            let val = js_sys::Reflect::get(&impl_obj, &JsValue::from_str(method))
                .map_err(|_| JsValue::from_str(&format!("ImageModel impl missing '{method}'")))?;
            if !val.is_function() {
                return Err(JsValue::from_str(&format!(
                    "ImageModel.{method} must be a function"
                )));
            }
        }
        Ok(Self { impl_obj })
    }

    /// Generate an image from a request object.
    ///
    /// `request` is forwarded as-is to the JS `generateImage` callback.
    /// Returns a `Promise<any>` resolving to the value the callback returned.
    #[wasm_bindgen(js_name = "generateImage")]
    pub fn generate_image(&self, request: JsValue) -> js_sys::Promise {
        let func: js_sys::Function =
            js_sys::Reflect::get(&self.impl_obj, &JsValue::from_str("generateImage"))
                .unwrap_or(JsValue::UNDEFINED)
                .unchecked_into();
        let this = self.impl_obj.clone();
        future_to_promise(async move { call1_await(&func, &this, request).await })
    }

    /// Upscale an image from a request object.
    ///
    /// `request` is forwarded as-is to the JS `upscaleImage` callback.
    /// Returns a `Promise<any>` resolving to the value the callback returned.
    #[wasm_bindgen(js_name = "upscaleImage")]
    pub fn upscale_image(&self, request: JsValue) -> js_sys::Promise {
        let func: js_sys::Function =
            js_sys::Reflect::get(&self.impl_obj, &JsValue::from_str("upscaleImage"))
                .unwrap_or(JsValue::UNDEFINED)
                .unchecked_into();
        let this = self.impl_obj.clone();
        future_to_promise(async move { call1_await(&func, &this, request).await })
    }
}

// ---------------------------------------------------------------------------
// WasmEmbedModel — JS-callback ABC for `EmbedModel`
// ---------------------------------------------------------------------------

/// JS-callback ABC matching the audit's `EmbedModel` slot.
///
/// Distinct from [`crate::embedding::WasmEmbeddingModel`] (which surfaces
/// the runtime [`blazen_llm::traits::EmbeddingModel`] trait): this thin
/// wrapper exists so callers can satisfy the `EmbedModel` name from
/// `blazen-embed` (a native FastEmbed/Tract alias) with a fully JS-side
/// implementation. The interface is identical to the existing
/// `EmbeddingModel.fromJsHandler` factory but is exposed as a class so
/// frameworks can pass it around as a typed value.
#[wasm_bindgen(js_name = "EmbedModel")]
pub struct WasmEmbedModel {
    impl_obj: JsValue,
    dimensions: u32,
    model_id: String,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmEmbedModel {}
unsafe impl Sync for WasmEmbedModel {}

#[wasm_bindgen(js_class = "EmbedModel")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
impl WasmEmbedModel {
    /// Wrap a JS object implementing the `EmbedModelImpl` interface.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `impl_obj` is missing the `embed`
    /// function, the `modelId` string, or the `dimensions` number.
    #[wasm_bindgen(constructor)]
    pub fn new(impl_obj: JsValue) -> Result<WasmEmbedModel, JsValue> {
        let embed_fn = js_sys::Reflect::get(&impl_obj, &JsValue::from_str("embed"))
            .map_err(|_| JsValue::from_str("EmbedModel impl missing 'embed'"))?;
        if !embed_fn.is_function() {
            return Err(JsValue::from_str("EmbedModel.embed must be a function"));
        }
        let model_id = js_sys::Reflect::get(&impl_obj, &JsValue::from_str("modelId"))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| JsValue::from_str("EmbedModel.modelId must be a string"))?;
        let dimensions = js_sys::Reflect::get(&impl_obj, &JsValue::from_str("dimensions"))
            .ok()
            .and_then(|v| v.as_f64())
            .ok_or_else(|| JsValue::from_str("EmbedModel.dimensions must be a number"))?
            as u32;
        Ok(Self {
            impl_obj,
            dimensions,
            model_id,
        })
    }

    /// The model identifier string supplied at construction.
    #[wasm_bindgen(getter, js_name = "modelId")]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// The dimensionality of the produced vectors.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Embed an array of strings via the wrapped JS callback. Returns a
    /// `Promise<number[][]>` matching the
    /// [`crate::embedding::WasmEmbeddingModel::embed`] return shape.
    pub fn embed(&self, texts: Vec<String>) -> js_sys::Promise {
        let func: js_sys::Function =
            js_sys::Reflect::get(&self.impl_obj, &JsValue::from_str("embed"))
                .unwrap_or(JsValue::UNDEFINED)
                .unchecked_into();
        let this = self.impl_obj.clone();
        future_to_promise(async move {
            let arr = js_sys::Array::new_with_length(texts.len() as u32);
            for (i, t) in texts.iter().enumerate() {
                arr.set(i as u32, JsValue::from_str(t));
            }
            let result = call1_await(&func, &this, arr.into()).await?;
            let outer: &js_sys::Array = result
                .dyn_ref::<js_sys::Array>()
                .ok_or_else(|| JsValue::from_str("EmbedModel.embed must return an array"))?;
            let out = js_sys::Array::new_with_length(outer.length());
            for i in 0..outer.length() {
                let entry = outer.get(i);
                if entry.has_type::<js_sys::Float32Array>() {
                    out.set(i, entry);
                } else if let Some(inner) = entry.dyn_ref::<js_sys::Array>() {
                    let f32a = js_sys::Float32Array::new_with_length(inner.length());
                    for j in 0..inner.length() {
                        let v = inner.get(j).as_f64().unwrap_or(0.0) as f32;
                        f32a.set_index(j, v);
                    }
                    out.set(i, f32a.into());
                } else {
                    return Err(JsValue::from_str(
                        "EmbedModel.embed entries must be Float32Array or number[]",
                    ));
                }
            }
            Ok(out.into())
        })
    }
}
