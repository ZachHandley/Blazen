//! Value types for the workflow context state map.
//!
//! [`StateValue`] supports JSON-compatible structured data, raw binary blobs,
//! and platform-serialized opaque objects, allowing workflows to store anything
//! from simple numbers to complex language-specific objects (e.g. Python Pydantic
//! models via pickle).

use serde::{Deserialize, Serialize};

/// Wrapper around `Vec<u8>` that uses `serde_bytes` for efficient
/// binary serialization (especially with `MessagePack`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BytesWrapper(#[serde(with = "serde_bytes")] pub Vec<u8>);

/// A value that can be stored in the workflow context state map.
///
/// Supports JSON-compatible structured data, raw binary blobs, and
/// platform-serialized opaque objects. The enum tag tells the deserializer
/// which variant it is, and `serde_bytes` correctly handles the inner bytes
/// for efficient binary serialization formats like `MessagePack`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StateValue {
    /// Structured data (numbers, strings, booleans, arrays, objects, null).
    Json(serde_json::Value),
    /// Raw binary data (files, images, audio, etc.).
    Bytes(BytesWrapper),
    /// Platform-serialized opaque object (Python pickle, etc.).
    ///
    /// Bindings use their native serializer to roundtrip arbitrary objects
    /// that are not JSON-serializable. The bytes are opaque to the Rust core
    /// and should only be deserialized by the same binding that created them.
    Native(BytesWrapper),
}

impl StateValue {
    /// Returns `true` if this value contains raw bytes.
    #[must_use]
    pub fn is_bytes(&self) -> bool {
        matches!(self, Self::Bytes(_))
    }

    /// Returns `true` if this value contains structured JSON data.
    #[must_use]
    pub fn is_json(&self) -> bool {
        matches!(self, Self::Json(_))
    }

    /// Returns `true` if this value contains a platform-serialized opaque object.
    #[must_use]
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native(_))
    }

    /// Returns a reference to the inner JSON value, or `None` if this is
    /// not a JSON variant.
    #[must_use]
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json(v) => Some(v),
            Self::Bytes(_) | Self::Native(_) => None,
        }
    }

    /// Returns a reference to the inner byte slice, or `None` if this is
    /// not a bytes variant.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(b) => Some(&b.0),
            Self::Json(_) | Self::Native(_) => None,
        }
    }

    /// Returns a reference to the inner native-serialized byte slice, or
    /// `None` if this is not a native variant.
    #[must_use]
    pub fn as_native(&self) -> Option<&[u8]> {
        match self {
            Self::Native(b) => Some(&b.0),
            Self::Json(_) | Self::Bytes(_) => None,
        }
    }

    /// Consumes `self` and returns the inner JSON value, or `None` if this
    /// is not a JSON variant.
    #[must_use]
    pub fn into_json(self) -> Option<serde_json::Value> {
        match self {
            Self::Json(v) => Some(v),
            Self::Bytes(_) | Self::Native(_) => None,
        }
    }

    /// Consumes `self` and returns the inner byte vector, or `None` if this
    /// is not a bytes variant.
    #[must_use]
    pub fn into_bytes(self) -> Option<Vec<u8>> {
        match self {
            Self::Bytes(b) => Some(b.0),
            Self::Json(_) | Self::Native(_) => None,
        }
    }

    /// Consumes `self` and returns the inner native-serialized byte vector,
    /// or `None` if this is not a native variant.
    #[must_use]
    pub fn into_native(self) -> Option<Vec<u8>> {
        match self {
            Self::Native(b) => Some(b.0),
            Self::Json(_) | Self::Bytes(_) => None,
        }
    }

    /// Create a `Native` variant from raw bytes.
    ///
    /// Use this instead of `From<Vec<u8>>` (which creates `Bytes`) when the
    /// data is a platform-serialized object that needs to be deserialized back
    /// by the binding layer.
    #[must_use]
    pub fn native(data: Vec<u8>) -> Self {
        Self::Native(BytesWrapper(data))
    }
}

impl From<serde_json::Value> for StateValue {
    fn from(v: serde_json::Value) -> Self {
        Self::Json(v)
    }
}

impl From<Vec<u8>> for StateValue {
    fn from(v: Vec<u8>) -> Self {
        Self::Bytes(BytesWrapper(v))
    }
}

impl From<&[u8]> for StateValue {
    fn from(v: &[u8]) -> Self {
        Self::Bytes(BytesWrapper(v.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_variant_accessors() {
        let val = StateValue::Json(serde_json::json!(42));
        assert!(val.is_json());
        assert!(!val.is_bytes());
        assert!(!val.is_native());
        assert_eq!(val.as_json(), Some(&serde_json::json!(42)));
        assert_eq!(val.as_bytes(), None);
        assert_eq!(val.as_native(), None);
    }

    #[test]
    fn bytes_variant_accessors() {
        let val = StateValue::Bytes(BytesWrapper(vec![1, 2, 3]));
        assert!(val.is_bytes());
        assert!(!val.is_json());
        assert!(!val.is_native());
        assert_eq!(val.as_bytes(), Some([1, 2, 3].as_slice()));
        assert_eq!(val.as_json(), None);
        assert_eq!(val.as_native(), None);
    }

    #[test]
    fn native_variant_accessors() {
        let val = StateValue::Native(BytesWrapper(vec![0x80, 0x04, 0x95]));
        assert!(val.is_native());
        assert!(!val.is_json());
        assert!(!val.is_bytes());
        assert_eq!(val.as_native(), Some([0x80, 0x04, 0x95].as_slice()));
        assert_eq!(val.as_json(), None);
        assert_eq!(val.as_bytes(), None);
    }

    #[test]
    fn into_json_consumes() {
        let val = StateValue::Json(serde_json::json!("hello"));
        assert_eq!(val.into_json(), Some(serde_json::json!("hello")));

        let val = StateValue::Bytes(BytesWrapper(vec![1]));
        assert_eq!(val.into_json(), None);

        let val = StateValue::Native(BytesWrapper(vec![1]));
        assert_eq!(val.into_json(), None);
    }

    #[test]
    fn into_bytes_consumes() {
        let val = StateValue::Bytes(BytesWrapper(vec![4, 5, 6]));
        assert_eq!(val.into_bytes(), Some(vec![4, 5, 6]));

        let val = StateValue::Json(serde_json::json!(null));
        assert_eq!(val.into_bytes(), None);

        let val = StateValue::Native(BytesWrapper(vec![1]));
        assert_eq!(val.into_bytes(), None);
    }

    #[test]
    fn into_native_consumes() {
        let val = StateValue::Native(BytesWrapper(vec![7, 8, 9]));
        assert_eq!(val.into_native(), Some(vec![7, 8, 9]));

        let val = StateValue::Json(serde_json::json!(null));
        assert_eq!(val.into_native(), None);

        let val = StateValue::Bytes(BytesWrapper(vec![1]));
        assert_eq!(val.into_native(), None);
    }

    #[test]
    fn native_constructor() {
        let val = StateValue::native(vec![0xDE, 0xAD]);
        assert!(val.is_native());
        assert_eq!(val.as_native(), Some([0xDE, 0xAD].as_slice()));
    }

    #[test]
    fn from_json_value() {
        let sv: StateValue = serde_json::json!({"key": "value"}).into();
        assert!(sv.is_json());
    }

    #[test]
    fn from_vec_u8() {
        let sv: StateValue = vec![10, 20, 30].into();
        assert!(sv.is_bytes());
        assert_eq!(sv.as_bytes(), Some([10, 20, 30].as_slice()));
    }

    #[test]
    fn from_slice() {
        let data: &[u8] = &[7, 8, 9];
        let sv: StateValue = data.into();
        assert!(sv.is_bytes());
        assert_eq!(sv.as_bytes(), Some([7, 8, 9].as_slice()));
    }

    #[test]
    fn json_serde_roundtrip() {
        let val = StateValue::Json(serde_json::json!({"nested": [1, 2, 3]}));
        let serialized = serde_json::to_string(&val).unwrap();
        let deserialized: StateValue = serde_json::from_str(&serialized).unwrap();
        assert_eq!(val, deserialized);
    }

    #[test]
    fn bytes_serde_roundtrip() {
        let val = StateValue::Bytes(BytesWrapper(vec![0xFF, 0xFE, 0xFD]));
        let serialized = serde_json::to_string(&val).unwrap();
        let deserialized: StateValue = serde_json::from_str(&serialized).unwrap();
        assert_eq!(val, deserialized);
    }

    #[test]
    fn native_serde_json_roundtrip() {
        let val = StateValue::Native(BytesWrapper(vec![0x80, 0x04, 0x95, 0x10]));
        let serialized = serde_json::to_string(&val).unwrap();
        let deserialized: StateValue = serde_json::from_str(&serialized).unwrap();
        assert_eq!(val, deserialized);
    }

    #[test]
    fn msgpack_json_roundtrip() {
        let val = StateValue::Json(serde_json::json!({"key": 42}));
        let bytes = rmp_serde::to_vec(&val).unwrap();
        let restored: StateValue = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn msgpack_bytes_roundtrip() {
        let val = StateValue::Bytes(BytesWrapper(vec![0xDE, 0xAD, 0xBE, 0xEF]));
        let bytes = rmp_serde::to_vec(&val).unwrap();
        let restored: StateValue = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn msgpack_native_roundtrip() {
        let val = StateValue::Native(BytesWrapper(vec![0x80, 0x04, 0x95, 0x10]));
        let bytes = rmp_serde::to_vec(&val).unwrap();
        let restored: StateValue = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(val, restored);
    }
}
