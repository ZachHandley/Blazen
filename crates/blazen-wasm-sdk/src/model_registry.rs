//! JS-callback ABC for [`blazen_llm::traits::ModelRegistry`].
//!
//! [`WasmModelRegistry`] mirrors the audit's `ModelRegistry` slot and brings
//! the WASM SDK to parity with the Python (`PyModelRegistry`) and Node
//! (`JsModelRegistry`) bindings of the same trait. It wraps a JS object
//! exposing `listModels()` and `getModel(modelId)` so browser callers can
//! plug a custom registry into Blazen's model-info lookup surface (catalog
//! lookups, capability discovery, dynamic model menus, etc.) without
//! writing Rust.
//!
//! Like the other JS-callback ABCs in [`crate::model_abcs`], the wrapped
//! callbacks are held behind the standard "WASM is single-threaded so
//! `Send`/`Sync` are vacuous" pattern.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::model_abcs::call1_await;

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_MODEL_REGISTRY: &str = r#"
/**
 * Implementation object for `ModelRegistry`. Each method may return a value
 * directly or a `Promise` that resolves to it.
 *
 * - `listModels()` -- list all models advertised by this registry.
 * - `getModel(modelId)` -- look up a single model by its identifier;
 *   return `null` or `undefined` if not found.
 */
export interface ModelRegistryImpl {
    listModels(): Promise<ModelInfo[]> | ModelInfo[];
    getModel(modelId: string): Promise<ModelInfo | null> | ModelInfo | null;
}
"#;

// ---------------------------------------------------------------------------
// WasmModelRegistry — JS-callback ABC for `ModelRegistry`
// ---------------------------------------------------------------------------

/// JS-callback ABC for [`blazen_llm::traits::ModelRegistry`].
///
/// Wraps a JS object exposing `listModels` and `getModel` methods so callers
/// can plug a custom model catalog (a static manifest, a remote control
/// plane, an in-browser registry, etc.) into Blazen's model-info lookup
/// surface. This is the WASM SDK counterpart of `PyModelRegistry` /
/// `JsModelRegistry`.
#[wasm_bindgen(js_name = "ModelRegistry")]
pub struct WasmModelRegistry {
    impl_obj: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmModelRegistry {}
unsafe impl Sync for WasmModelRegistry {}

#[wasm_bindgen(js_class = "ModelRegistry")]
impl WasmModelRegistry {
    /// Wrap a JS object implementing the `ModelRegistryImpl` interface.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `impl_obj` is missing one of the
    /// required methods (`listModels`, `getModel`).
    #[wasm_bindgen(constructor)]
    pub fn new(impl_obj: JsValue) -> Result<WasmModelRegistry, JsValue> {
        for method in &["listModels", "getModel"] {
            let val =
                js_sys::Reflect::get(&impl_obj, &JsValue::from_str(method)).map_err(|_| {
                    JsValue::from_str(&format!("ModelRegistry impl missing '{method}'"))
                })?;
            if !val.is_function() {
                return Err(JsValue::from_str(&format!(
                    "ModelRegistry.{method} must be a function"
                )));
            }
        }
        Ok(Self { impl_obj })
    }

    /// List every model advertised by this registry.
    ///
    /// Returns a `Promise<ModelInfo[]>` resolving to whatever the JS
    /// `listModels` callback returned.
    #[wasm_bindgen(js_name = "listModels")]
    pub fn list_models(&self) -> js_sys::Promise {
        let func: js_sys::Function =
            js_sys::Reflect::get(&self.impl_obj, &JsValue::from_str("listModels"))
                .unwrap_or(JsValue::UNDEFINED)
                .unchecked_into();
        let this = self.impl_obj.clone();
        future_to_promise(async move { call1_await(&func, &this, JsValue::UNDEFINED).await })
    }

    /// Look up a single model by its identifier.
    ///
    /// `modelId` is forwarded as a string argument to the JS `getModel`
    /// callback. Returns a `Promise<ModelInfo | null>` resolving to whatever
    /// the callback returned (`null`/`undefined` when the model is not
    /// known).
    #[wasm_bindgen(js_name = "getModel")]
    pub fn get_model(&self, model_id: String) -> js_sys::Promise {
        let func: js_sys::Function =
            js_sys::Reflect::get(&self.impl_obj, &JsValue::from_str("getModel"))
                .unwrap_or(JsValue::UNDEFINED)
                .unchecked_into();
        let this = self.impl_obj.clone();
        future_to_promise(
            async move { call1_await(&func, &this, JsValue::from_str(&model_id)).await },
        )
    }
}
