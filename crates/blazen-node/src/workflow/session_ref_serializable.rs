//! Node-side implementation of the
//! [`SessionRefSerializable`](blazen_core::session_ref::SessionRefSerializable)
//! protocol.
//!
//! ## Why this looks different from the Python version
//!
//! The Python binding (see
//! `crates/blazen-py/src/session_ref_serializable.rs`) wraps a live
//! `Py<PyAny>` and reaches back into Python at snapshot time to call
//! `__blazen_serialize__`, then calls a `__blazen_deserialize__`
//! classmethod on resume. That works because pyo3 lets Rust hold a
//! `Py<PyAny>` across thread boundaries safely.
//!
//! On the Node side, napi-rs's `Reference<T>` is **not** `Send` — its
//! `Drop` must execute on the v8 main thread, so we cannot keep a live
//! `napi::JsObject` inside the cross-thread session-ref registry. The
//! current Node step bridge therefore routes step-handler arguments and
//! return values through `serde_json::Value`, which means we cannot
//! transparently detect a `serialize()` method on a JS object the way
//! Python detects `__blazen_serialize__`.
//!
//! Instead, this module exposes a **bytes-in / bytes-out** adapter:
//!
//! 1. JS callers serialize the value themselves (e.g. `Buffer.from(...)`).
//! 2. They call `ctx.insertSessionRefSerializable(typeName, bytes)` to
//!    store the bytes alongside a stable type-tag string in the
//!    [`SessionRefRegistry`](blazen_core::session_ref::SessionRefRegistry).
//! 3. The snapshot walker captures the bytes through
//!    [`SessionRefSerializable::blazen_serialize`] just like the Python
//!    path.
//! 4. On resume, [`node_deserializer_trampoline`] reconstructs a fresh
//!    [`NodeSessionRefSerializable`] holding the same bytes; JS callers
//!    use `ctx.getSessionRefSerializable(key)` to read the bytes back
//!    and deserialize them in user code.
//!
//! This trade-off keeps the implementation completely synchronous on
//! the Rust side and avoids the complexity of a `ThreadsafeFunction`
//! trampoline that would have to call back into JS during `resume`.
//! Once the napi bridge gains end-to-end `napi::JsUnknown` support
//! (tracked separately as a Phase 13 refactor) we can lift this out and
//! match the Python identity-preserving path.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use blazen_core::SessionRefDeserializerFn;
use blazen_core::session_ref::{SessionRefError, SessionRefSerializable};

// ---------------------------------------------------------------------------
// Type-tag interning pool
// ---------------------------------------------------------------------------

/// Process-global intern pool that turns dynamic JS-supplied type names
/// into the `&'static str` values required by
/// [`SessionRefSerializable::blazen_type_tag`].
///
/// Strings are leaked on first insertion and reused thereafter, so the
/// total memory cost is bounded by the number of distinct serializable
/// type names used by the host process — not by the number of live
/// instances.
static TYPE_TAG_POOL: LazyLock<Mutex<HashMap<String, &'static str>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Intern a string and return a `&'static str` reference. Subsequent
/// calls with the same string return the same pointer so downstream
/// `==` comparisons on the leaked reference are meaningful.
///
/// Uses [`Box::leak`] on a fresh `String` the first time a value is
/// seen. The leaked memory lives for the remainder of the process; we
/// accept this because the caller is the Node bindings, where the set
/// of distinct type names is bounded by the application code.
pub(crate) fn intern_type_tag(s: &str) -> &'static str {
    let mut pool = TYPE_TAG_POOL.lock().expect("type tag intern pool poisoned");
    if let Some(&existing) = pool.get(s) {
        return existing;
    }
    let owned: String = s.to_owned();
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    pool.insert(leaked.to_owned(), leaked);
    leaked
}

// ---------------------------------------------------------------------------
// NodeSessionRefSerializable
// ---------------------------------------------------------------------------

/// Adapter that wraps a serialized-bytes payload plus the type-tag
/// string the JS caller supplied at insertion time, satisfying the
/// [`SessionRefSerializable`] trait so the snapshot walker can capture
/// it under the
/// [`SessionPausePolicy::PickleOrSerialize`](blazen_core::session_ref::SessionPausePolicy)
/// policy.
///
/// The serialized representation produced by [`Self::blazen_serialize`]
/// is the user-supplied bytes prefixed with `[4-byte BE
/// tag_len][tag_bytes]`, mirroring
/// [`blazen_py::session_ref_serializable::PySerializableSessionRef`].
/// The prefix lets [`node_deserializer_trampoline`] reconstruct the
/// original tag without needing a side-channel — the resume path
/// receives only the bytes from the snapshot walker.
pub struct NodeSessionRefSerializable {
    /// Interned type tag, kept as `&'static str` to satisfy
    /// [`SessionRefSerializable::blazen_type_tag`].
    type_tag: &'static str,
    /// Raw bytes the JS caller passed to
    /// `ctx.insertSessionRefSerializable(typeName, bytes)`.
    user_bytes: Vec<u8>,
}

impl NodeSessionRefSerializable {
    /// Build a new adapter from a JS-supplied type name and the bytes
    /// the user already serialized.
    ///
    /// The type name is interned so successive insertions of the same
    /// type produce the same `&'static str` pointer.
    #[must_use]
    pub fn new(type_name: &str, serialized: Vec<u8>) -> Self {
        let type_tag = intern_type_tag(type_name);
        Self {
            type_tag,
            user_bytes: serialized,
        }
    }

    /// Return the type tag without going through the trait method.
    /// Used by [`crate::workflow::context::JsContext::get_session_ref_serializable`]
    /// to read back the metadata stored alongside the bytes.
    #[must_use]
    pub fn type_tag(&self) -> &'static str {
        self.type_tag
    }

    /// Return a slice over the user-supplied bytes (the original
    /// payload, **not** the prefixed wire format).
    #[must_use]
    pub fn user_bytes(&self) -> &[u8] {
        &self.user_bytes
    }
}

impl SessionRefSerializable for NodeSessionRefSerializable {
    fn blazen_serialize(&self) -> Result<Vec<u8>, SessionRefError> {
        // Self-describing wire format so [`node_deserializer_trampoline`]
        // can recover the type tag from the bytes alone. The core also
        // exposes the type tag separately in the snapshot record, but
        // the deserializer callback only receives the raw bytes — so we
        // duplicate the tag inside the payload for symmetry with the
        // Python adapter.
        let tag_bytes = self.type_tag.as_bytes();
        let tag_len: u32 =
            u32::try_from(tag_bytes.len()).map_err(|_| SessionRefError::SerializationFailed {
                type_tag: self.type_tag.to_owned(),
                source: Box::<dyn std::error::Error + Send + Sync>::from(
                    "type tag longer than u32::MAX bytes",
                ),
            })?;
        let mut out = Vec::with_capacity(4 + tag_bytes.len() + self.user_bytes.len());
        out.extend_from_slice(&tag_len.to_be_bytes());
        out.extend_from_slice(tag_bytes);
        out.extend_from_slice(&self.user_bytes);
        Ok(out)
    }

    fn blazen_type_tag(&self) -> &'static str {
        self.type_tag
    }
}

// ---------------------------------------------------------------------------
// Deserializer trampoline
// ---------------------------------------------------------------------------

/// Parse the self-describing prefix produced by
/// [`NodeSessionRefSerializable::blazen_serialize`] and return
/// `(type_tag, user_bytes)`.
fn split_prefix(bytes: &[u8]) -> Result<(&str, &[u8]), SessionRefError> {
    if bytes.len() < 4 {
        return Err(SessionRefError::SerializationFailed {
            type_tag: "<unknown>".to_owned(),
            source: Box::<dyn std::error::Error + Send + Sync>::from(
                "payload too short to contain type tag prefix",
            ),
        });
    }
    let mut tag_len_bytes = [0_u8; 4];
    tag_len_bytes.copy_from_slice(&bytes[..4]);
    let tag_len = u32::from_be_bytes(tag_len_bytes) as usize;
    if bytes.len() < 4 + tag_len {
        return Err(SessionRefError::SerializationFailed {
            type_tag: "<unknown>".to_owned(),
            source: Box::<dyn std::error::Error + Send + Sync>::from(
                "payload shorter than declared type tag length",
            ),
        });
    }
    let tag = std::str::from_utf8(&bytes[4..4 + tag_len]).map_err(|e| {
        SessionRefError::SerializationFailed {
            type_tag: "<unknown>".to_owned(),
            source: Box::new(e),
        }
    })?;
    Ok((tag, &bytes[4 + tag_len..]))
}

/// Single fn pointer registered for every type tag referenced by a
/// snapshot's `__blazen_serialized_session_refs` sidecar.
///
/// Unlike the Python equivalent (which imports a class and invokes a
/// classmethod) this Rust-only trampoline simply re-wraps the captured
/// bytes in a fresh [`NodeSessionRefSerializable`]. The intent is that
/// JS application code retrieves the rehydrated entry afterwards via
/// `ctx.getSessionRefSerializable(key)`, which returns
/// `{ typeName, bytes }`, and reconstructs the original JS object from
/// those raw bytes itself.
///
/// The payload layout is produced by
/// [`NodeSessionRefSerializable::blazen_serialize`]:
///
/// ```text
/// [4-byte BE tag_len][tag bytes...][user bytes...]
/// ```
pub(crate) fn node_deserializer_trampoline(
    bytes: &[u8],
) -> Result<Arc<dyn SessionRefSerializable>, SessionRefError> {
    let (type_tag, user_bytes) = split_prefix(bytes)?;
    let interned = intern_type_tag(type_tag);
    Ok(Arc::new(NodeSessionRefSerializable {
        type_tag: interned,
        user_bytes: user_bytes.to_vec(),
    }))
}

/// Static reference to [`node_deserializer_trampoline`] coerced to the
/// core [`SessionRefDeserializerFn`] alias. Re-exported for use in
/// [`crate::workflow::workflow::JsWorkflow::resume_with_serializable_refs`].
pub(crate) const DESERIALIZER_FN: SessionRefDeserializerFn = node_deserializer_trampoline;

/// Compile-time assertion that [`NodeSessionRefSerializable`] satisfies
/// the bounds required by `Arc<dyn SessionRefSerializable>` and by the
/// `Arc<dyn Any + Send + Sync>` main registry map.
const _ASSERT_BOUNDS: () = {
    fn assert_send_sync_any<T: Any + Send + Sync>() {}
    fn _bounds() {
        assert_send_sync_any::<NodeSessionRefSerializable>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_same_pointer_for_same_string() {
        let a = intern_type_tag("foo::Bar");
        let b = intern_type_tag("foo::Bar");
        assert!(std::ptr::eq(a.as_ptr(), b.as_ptr()));
    }

    #[test]
    fn intern_returns_distinct_pointers_for_different_strings() {
        let a = intern_type_tag("first::Type");
        let b = intern_type_tag("second::Type");
        assert!(!std::ptr::eq(a.as_ptr(), b.as_ptr()));
    }

    #[test]
    fn serialize_round_trip_through_trampoline() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let original = NodeSessionRefSerializable::new("my::Type", payload.clone());

        // Serialize via the trait method to get the wire format.
        let wire = original.blazen_serialize().expect("serialize ok");

        // Deserialize via the trampoline; should recover the same tag
        // and bytes.
        let recovered = node_deserializer_trampoline(&wire).expect("deserialize ok");
        assert_eq!(recovered.blazen_type_tag(), "my::Type");

        // Round-trip the bytes through the recovered adapter to confirm
        // the user payload is preserved exactly (i.e. the prefix is
        // re-applied identically on a second serialize).
        let wire2 = recovered.blazen_serialize().expect("serialize ok");
        assert_eq!(wire, wire2);
    }

    #[test]
    fn split_prefix_rejects_short_payload() {
        let result = split_prefix(&[0x00, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn split_prefix_rejects_truncated_tag() {
        // Declares 100 bytes of tag but supplies only 4.
        let bad = [0x00, 0x00, 0x00, 0x64, b'a', b'b', b'c', b'd'];
        let result = split_prefix(&bad);
        assert!(result.is_err());
    }
}
