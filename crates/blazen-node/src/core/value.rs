//! Typed wrappers for [`BytesWrapper`](blazen_core::BytesWrapper) and
//! [`StateValue`](blazen_core::StateValue).
//!
//! `StateValue` is a tagged enum on the Rust side. napi-rs cannot expose
//! a Rust enum-with-data directly as a class, so the JS surface uses a
//! discriminated object: a [`JsStateValueKind`] string-enum lives in the
//! `kind` field and the variant payload lives in `json` / `bytes` /
//! `native`. Factory methods construct each variant cleanly.

use blazen_core::{BytesWrapper, StateValue};
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// BytesWrapper
// ---------------------------------------------------------------------------

/// Thin newtype around a byte vector. Mirrors
/// [`blazen_core::BytesWrapper`] and exists so that JS callers can hand
/// the engine a typed binary blob without going through `Buffer` / JSON
/// round-trips.
#[napi(js_name = "BytesWrapper")]
pub struct JsBytesWrapper {
    pub(crate) inner: BytesWrapper,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsBytesWrapper {
    /// Create a new wrapper from a `Buffer` / `Uint8Array`.
    #[allow(clippy::needless_pass_by_value)]
    #[napi(constructor)]
    pub fn new(bytes: Buffer) -> Self {
        Self {
            inner: BytesWrapper(bytes.to_vec()),
        }
    }

    /// Returns the wrapped bytes as a `Buffer`.
    #[napi(js_name = "toBuffer")]
    pub fn to_buffer(&self) -> Buffer {
        Buffer::from(self.inner.0.clone())
    }

    /// Length of the wrapped byte vector.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        u32::try_from(self.inner.0.len()).unwrap_or(u32::MAX)
    }
}

impl JsBytesWrapper {
    #[allow(dead_code)]
    pub(crate) fn from_native(inner: BytesWrapper) -> Self {
        Self { inner }
    }

    #[allow(dead_code)]
    pub(crate) fn into_native(self) -> BytesWrapper {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// StateValue
// ---------------------------------------------------------------------------

/// Variant tag for [`JsStateValue`].
#[napi(string_enum, js_name = "StateValueKind")]
#[derive(Clone, Copy)]
pub enum JsStateValueKind {
    /// Structured JSON data.
    Json,
    /// Raw binary blob.
    Bytes,
    /// Platform-serialized opaque bytes.
    Native,
}

/// Discriminated wrapper around [`blazen_core::StateValue`].
///
/// JS code constructs an instance via one of the factory methods
/// (`json`, `bytes`, `native`) and inspects the variant via the
/// [`JsStateValue::kind`] getter. Use [`JsStateValue::asJson`] /
/// [`JsStateValue::asBytes`] / [`JsStateValue::asNative`] to read the
/// payload of the active variant.
#[napi(js_name = "StateValue")]
pub struct JsStateValue {
    pub(crate) inner: StateValue,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsStateValue {
    /// Construct a JSON-backed value.
    #[napi(factory)]
    pub fn json(value: serde_json::Value) -> Self {
        Self {
            inner: StateValue::Json(value),
        }
    }

    /// Construct a raw-bytes value from a `Buffer`.
    #[allow(clippy::needless_pass_by_value)]
    #[napi(factory)]
    pub fn bytes(buf: Buffer) -> Self {
        Self {
            inner: StateValue::Bytes(BytesWrapper(buf.to_vec())),
        }
    }

    /// Construct a platform-native opaque-bytes value.
    #[allow(clippy::needless_pass_by_value)]
    #[napi(factory)]
    pub fn native(buf: Buffer) -> Self {
        Self {
            inner: StateValue::Native(BytesWrapper(buf.to_vec())),
        }
    }

    /// Active variant tag.
    #[napi(getter)]
    pub fn kind(&self) -> JsStateValueKind {
        match self.inner {
            StateValue::Json(_) => JsStateValueKind::Json,
            StateValue::Bytes(_) => JsStateValueKind::Bytes,
            StateValue::Native(_) => JsStateValueKind::Native,
        }
    }

    /// Returns the JSON payload, or `null` if this is not a JSON variant.
    #[napi(js_name = "asJson")]
    pub fn as_json(&self) -> Option<serde_json::Value> {
        match &self.inner {
            StateValue::Json(v) => Some(v.clone()),
            StateValue::Bytes(_) | StateValue::Native(_) => None,
        }
    }

    /// Returns the raw bytes payload as a `Buffer`, or `null` if this is
    /// not a `Bytes` variant.
    #[napi(js_name = "asBytes")]
    pub fn as_bytes(&self) -> Option<Buffer> {
        match &self.inner {
            StateValue::Bytes(b) => Some(Buffer::from(b.0.clone())),
            StateValue::Json(_) | StateValue::Native(_) => None,
        }
    }

    /// Returns the platform-native bytes payload as a `Buffer`, or `null`
    /// if this is not a `Native` variant.
    #[napi(js_name = "asNative")]
    pub fn as_native(&self) -> Option<Buffer> {
        match &self.inner {
            StateValue::Native(b) => Some(Buffer::from(b.0.clone())),
            StateValue::Json(_) | StateValue::Bytes(_) => None,
        }
    }

    /// Convenience: `true` when [`Self::kind`] is [`JsStateValueKind::Json`].
    #[napi(js_name = "isJson")]
    pub fn is_json(&self) -> bool {
        matches!(self.inner, StateValue::Json(_))
    }

    /// Convenience: `true` when [`Self::kind`] is [`JsStateValueKind::Bytes`].
    #[napi(js_name = "isBytes")]
    pub fn is_bytes(&self) -> bool {
        matches!(self.inner, StateValue::Bytes(_))
    }

    /// Convenience: `true` when [`Self::kind`] is [`JsStateValueKind::Native`].
    #[napi(js_name = "isNative")]
    pub fn is_native(&self) -> bool {
        matches!(self.inner, StateValue::Native(_))
    }
}

impl JsStateValue {
    #[allow(dead_code)]
    pub(crate) fn from_native(inner: StateValue) -> Self {
        Self { inner }
    }

    #[allow(dead_code)]
    pub(crate) fn into_native(self) -> StateValue {
        self.inner
    }
}
