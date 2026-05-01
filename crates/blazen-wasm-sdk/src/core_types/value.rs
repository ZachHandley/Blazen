//! `wasm-bindgen` wrappers for [`blazen_core::BytesWrapper`] and
//! [`blazen_core::StateValue`].
//!
//! Both types are normally used through the `Context` namespace getters
//! (which transparently round-trip JS values through `serde-wasm-bindgen`),
//! but bindings need to construct, inspect, and serialise them directly when
//! interoperating with [`WorkflowSnapshot`] payloads, capability providers,
//! or low-level adapters that traffic in [`StateValue`] enums.
//!
//! These bindings keep the Rust types intact (no field-by-field projection)
//! and expose a small surface area covering construction, variant inspection,
//! and JSON marshalling. Marshalling uses
//! [`serde_wasm_bindgen::Serializer::serialize_maps_as_objects`] so the JS
//! side sees plain objects instead of `Map` instances.

use blazen_core::{BytesWrapper, StateValue};
use serde::Serialize;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Marshalling helpers
// ---------------------------------------------------------------------------

/// Convert a `Serialize` value into a `JsValue` shaped as a plain JS object.
///
/// Mirrors the `marshal_to_js` helper used by `workflow.rs` and `handler.rs`
/// so JSON stringification of values produced by this module round-trips
/// cleanly with the rest of the SDK.
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

// ---------------------------------------------------------------------------
// WasmBytesWrapper
// ---------------------------------------------------------------------------

/// JS-facing handle for a [`blazen_core::BytesWrapper`].
///
/// Keeps the inner `Vec<u8>` so that callers who need a stable handle can
/// pass it around instead of cloning the whole buffer through `Uint8Array`s.
#[wasm_bindgen(js_name = "BytesWrapper")]
pub struct WasmBytesWrapper {
    inner: BytesWrapper,
}

impl WasmBytesWrapper {
    /// Wrap an existing [`BytesWrapper`]. Used by sibling modules that
    /// receive a `BytesWrapper` from the engine (e.g. when extracting a
    /// raw payload out of a `StateValue`).
    #[must_use]
    pub(crate) fn from_inner(inner: BytesWrapper) -> Self {
        Self { inner }
    }

    /// Borrow the underlying [`BytesWrapper`].
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &BytesWrapper {
        &self.inner
    }

    /// Consume `self` and return the underlying [`BytesWrapper`].
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> BytesWrapper {
        self.inner
    }
}

#[wasm_bindgen(js_class = "BytesWrapper")]
impl WasmBytesWrapper {
    /// Construct a new `BytesWrapper` from a `Uint8Array`.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(data: js_sys::Uint8Array) -> Self {
        Self {
            inner: BytesWrapper(data.to_vec()),
        }
    }

    /// Number of bytes in the buffer.
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn length(&self) -> u32 {
        self.inner.0.len() as u32
    }

    /// Return a fresh `Uint8Array` copy of the underlying bytes.
    #[wasm_bindgen(js_name = "toUint8Array")]
    #[must_use]
    pub fn to_uint8_array(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(self.inner.0.as_slice())
    }
}

// ---------------------------------------------------------------------------
// WasmStateValue
// ---------------------------------------------------------------------------

/// JS-facing handle for a [`blazen_core::StateValue`].
///
/// `StateValue` is an enum with three variants (`Json`, `Bytes`, `Native`).
/// JS construction goes through the static factories
/// [`WasmStateValue::json`], [`WasmStateValue::bytes`],
/// [`WasmStateValue::native`]; inspection uses the variant predicates
/// (`isJson` / `isBytes` / `isNative`) and unwrap helpers
/// (`asJson` / `asBytes` / `asNative`).
#[wasm_bindgen(js_name = "StateValue")]
pub struct WasmStateValue {
    inner: StateValue,
}

impl WasmStateValue {
    /// Wrap an existing [`StateValue`]. Used when surfacing engine-produced
    /// state entries to JS.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn from_inner(inner: StateValue) -> Self {
        Self { inner }
    }

    /// Borrow the underlying [`StateValue`].
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &StateValue {
        &self.inner
    }

    /// Consume `self` and return the underlying [`StateValue`].
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> StateValue {
        self.inner
    }
}

#[wasm_bindgen(js_class = "StateValue")]
impl WasmStateValue {
    /// Construct a `StateValue::Json` from a JS value.
    ///
    /// The value must be JSON-serialisable (via `serde-wasm-bindgen`).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `value` is not JSON-serialisable.
    #[wasm_bindgen(js_name = "json")]
    pub fn json(value: JsValue) -> Result<WasmStateValue, JsValue> {
        let json: serde_json::Value = serde_wasm_bindgen::from_value(value)
            .map_err(|e| JsValue::from_str(&format!("StateValue.json: not JSON: {e}")))?;
        Ok(Self {
            inner: StateValue::Json(json),
        })
    }

    /// Construct a `StateValue::Bytes` from a `Uint8Array`.
    #[wasm_bindgen(js_name = "bytes")]
    #[must_use]
    pub fn bytes(data: js_sys::Uint8Array) -> WasmStateValue {
        Self {
            inner: StateValue::Bytes(BytesWrapper(data.to_vec())),
        }
    }

    /// Construct a `StateValue::Native` from a `Uint8Array`.
    ///
    /// Use this when the binding owns a platform-specific serialised payload
    /// that should be opaque to the engine.
    #[wasm_bindgen(js_name = "native")]
    #[must_use]
    pub fn native(data: js_sys::Uint8Array) -> WasmStateValue {
        Self {
            inner: StateValue::native(data.to_vec()),
        }
    }

    /// Returns `true` if this value contains structured JSON data.
    #[wasm_bindgen(js_name = "isJson")]
    #[must_use]
    pub fn is_json(&self) -> bool {
        self.inner.is_json()
    }

    /// Returns `true` if this value contains raw bytes.
    #[wasm_bindgen(js_name = "isBytes")]
    #[must_use]
    pub fn is_bytes(&self) -> bool {
        self.inner.is_bytes()
    }

    /// Returns `true` if this value contains a platform-serialised opaque
    /// blob.
    #[wasm_bindgen(js_name = "isNative")]
    #[must_use]
    pub fn is_native(&self) -> bool {
        self.inner.is_native()
    }

    /// Return the inner JSON value, or `null` if this is not a JSON variant.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JSON value cannot be marshalled
    /// into a `JsValue` (this should not happen for any value the engine
    /// produced).
    #[wasm_bindgen(js_name = "asJson")]
    pub fn as_json(&self) -> Result<JsValue, JsValue> {
        self.inner
            .as_json()
            .map_or(Ok(JsValue::NULL), marshal_to_js)
    }

    /// Return the inner bytes as a `Uint8Array`, or `null` if this is not a
    /// `Bytes` variant.
    #[wasm_bindgen(js_name = "asBytes")]
    #[must_use]
    pub fn as_bytes(&self) -> JsValue {
        self.inner
            .as_bytes()
            .map_or(JsValue::NULL, |b| js_sys::Uint8Array::from(b).into())
    }

    /// Return the inner native bytes as a `Uint8Array`, or `null` if this
    /// is not a `Native` variant.
    #[wasm_bindgen(js_name = "asNative")]
    #[must_use]
    pub fn as_native(&self) -> JsValue {
        self.inner
            .as_native()
            .map_or(JsValue::NULL, |b| js_sys::Uint8Array::from(b).into())
    }

    /// Marshal this value into a JS-friendly representation:
    ///
    /// - `Json` → the underlying JSON shape.
    /// - `Bytes` → a `Uint8Array`.
    /// - `Native` → a `Uint8Array`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JSON variant cannot be marshalled.
    #[wasm_bindgen(js_name = "toJsValue")]
    pub fn to_js_value(&self) -> Result<JsValue, JsValue> {
        match &self.inner {
            StateValue::Json(v) => marshal_to_js(v),
            StateValue::Bytes(b) | StateValue::Native(b) => {
                Ok(js_sys::Uint8Array::from(b.0.as_slice()).into())
            }
        }
    }
}
