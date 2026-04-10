//! Python-side implementation of the
//! [`SessionRefSerializable`](blazen_core::session_ref::SessionRefSerializable)
//! protocol.
//!
//! Python classes that define `__blazen_serialize__(self) -> bytes` and
//! (optionally) a classmethod `__blazen_deserialize__(cls, data: bytes)
//! -> instance` become serializable session refs: when the workflow is
//! configured with
//! [`SessionPausePolicy::PickleOrSerialize`](blazen_core::session_ref::SessionPausePolicy)
//! and is snapshotted, the engine captures the binary payload returned
//! by `__blazen_serialize__` into snapshot metadata and the matching
//! classmethod is invoked on resume to reconstruct the value.
//!
//! This module provides:
//!
//! - [`PySerializableSessionRef`], the Rust adapter that wraps a Python
//!   object and caches the serialized bytes + interned type tag so the
//!   core trait can work synchronously.
//! - [`intern_type_tag`], a process-global string interning helper so
//!   we can produce the `&'static str` that the core trait requires.
//! - [`py_deserializer_trampoline`], the single function pointer we
//!   register in the
//!   [`SessionRefDeserializerFn`](blazen_core::SessionRefDeserializerFn)
//!   map for every known type tag. Because the core passes only the
//!   serialized bytes to the deserializer, we prepend a length-prefixed
//!   copy of the type tag to the payload inside
//!   [`PySerializableSessionRef::blazen_serialize`] so the trampoline
//!   can recover the class name and dispatch to the right Python
//!   classmethod on the resume side.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use blazen_core::SessionRefDeserializerFn;
use blazen_core::session_ref::{SessionRefError, SessionRefSerializable};

/// Dunder name Python classes implement to opt into serializable
/// session refs on the `py_to_json` write side.
pub(crate) const SERIALIZE_DUNDER: &str = "__blazen_serialize__";

/// Dunder name Python classes must expose as a classmethod to
/// reconstruct a value from its captured bytes on the resume side.
pub(crate) const DESERIALIZE_DUNDER: &str = "__blazen_deserialize__";

// ---------------------------------------------------------------------------
// Type-tag interning pool
// ---------------------------------------------------------------------------

/// Global intern pool that turns dynamic Python class qualnames into
/// the `&'static str` values required by
/// [`SessionRefSerializable::blazen_type_tag`]. Strings are leaked on
/// first insertion and reused thereafter, so the total memory cost is
/// bounded by the number of distinct serializable classes in the
/// process (not by the number of live instances).
static TYPE_TAG_POOL: LazyLock<Mutex<HashMap<String, &'static str>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Intern a string and return a `&'static str` reference. Subsequent
/// calls with the same string return the same pointer so downstream
/// `==` comparisons on the leaked reference are meaningful.
///
/// Uses [`Box::leak`] on a fresh `String` the first time a value is
/// seen. The leaked memory lives for the remainder of the process; we
/// accept this because the caller is the Python bindings, where Python
/// classes themselves live for the process lifetime anyway.
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
// PySerializableSessionRef
// ---------------------------------------------------------------------------

/// Adapter that bridges a Python object exposing
/// `__blazen_serialize__` to the Rust [`SessionRefSerializable`]
/// trait.
///
/// The serialized bytes and type tag are captured at construction time
/// (inside an `Python::attach`-guarded call) so the trait methods can
/// return them synchronously without re-entering Python.
pub struct PySerializableSessionRef {
    /// The originating Python object. Held so `json_to_py` can return
    /// the exact same instance for identity preservation within a
    /// single workflow run, and so resumed values can be handed back
    /// to downstream steps.
    pub(crate) obj: Py<PyAny>,
    /// Interned qualified name (`module.Qualname`) used as the stable
    /// deserializer key.
    type_tag: &'static str,
    /// Raw user-supplied bytes returned by `__blazen_serialize__`.
    /// The self-describing prefix added by
    /// [`Self::blazen_serialize`] is NOT stored here — it's produced
    /// on demand.
    user_bytes: Vec<u8>,
}

impl PySerializableSessionRef {
    /// Construct an adapter by invoking `__blazen_serialize__` on the
    /// given Python object.
    ///
    /// Returns `Err` if the dunder is missing, the call raises, or the
    /// return value is not `bytes`/`bytearray`.
    pub(crate) fn try_new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Pull the qualified name (module + qualname) for the class so
        // the resume side can import and dispatch to it.
        let cls = obj.get_type();
        let module: String = cls
            .getattr("__module__")
            .and_then(|m| m.extract())
            .unwrap_or_else(|_| "<unknown>".to_owned());
        let qualname: String = cls
            .getattr("__qualname__")
            .and_then(|q| q.extract())
            .unwrap_or_else(|_| "<unknown>".to_owned());
        let tag = format!("{module}.{qualname}");
        let type_tag = intern_type_tag(&tag);

        // Call `__blazen_serialize__()` and demand a bytes-like return.
        let result = obj.call_method0(SERIALIZE_DUNDER)?;
        let user_bytes: Vec<u8> = result.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{SERIALIZE_DUNDER} on {type_tag} must return bytes, got {}",
                result
                    .get_type()
                    .name()
                    .map_or_else(|_| "<unknown>".to_owned(), |n| n.to_string())
            ))
        })?;

        Ok(Self {
            obj: obj.clone().unbind(),
            type_tag,
            user_bytes,
        })
    }
}

impl SessionRefSerializable for PySerializableSessionRef {
    fn blazen_serialize(&self) -> Result<Vec<u8>, SessionRefError> {
        // Self-describing payload so [`py_deserializer_trampoline`] can
        // recover the class name without any side-channel state. The
        // core does expose the type_tag separately in the snapshot
        // record (via `blazen_type_tag`), but the deserializer callback
        // only receives bytes — so we duplicate the tag here.
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
/// [`PySerializableSessionRef::blazen_serialize`] and return
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

/// Single fn pointer used for every type tag registered with
/// [`blazen_core::Workflow::resume_with_deserializers`] from the
/// Python binding.
///
/// The payload layout is produced by
/// [`PySerializableSessionRef::blazen_serialize`]:
///
/// ```text
/// [4-byte BE tag_len][tag bytes...][user bytes...]
/// ```
///
/// The trampoline splits the prefix, imports the class by
/// `module.qualname`, calls the `__blazen_deserialize__` classmethod
/// with the user bytes, and wraps the resulting Python instance in a
/// fresh [`PySerializableSessionRef`] so the rehydrated entry still
/// participates in the serializable sidecar if the resumed workflow
/// pauses again.
pub(crate) fn py_deserializer_trampoline(
    bytes: &[u8],
) -> Result<Arc<dyn SessionRefSerializable>, SessionRefError> {
    let (type_tag, user_bytes) = split_prefix(bytes)?;
    let owned_tag = type_tag.to_owned();

    // Reacquire the GIL so we can import the class and call the
    // classmethod. The trampoline runs inside a tokio task (spawned
    // by `pyo3-async-runtimes`), so we may or may not already hold
    // the GIL — `Python::attach` handles both cases.
    Python::attach(
        |py| -> Result<Arc<dyn SessionRefSerializable>, SessionRefError> {
            let (module_name, class_path) = split_module_and_class(&owned_tag);

            let module =
                py.import(module_name)
                    .map_err(|e| SessionRefError::SerializationFailed {
                        type_tag: owned_tag.clone(),
                        source: Box::new(PyErrWrap(e.to_string())),
                    })?;

            // Traverse nested qualnames like "Outer.Inner" by walking
            // getattrs from the module root.
            let mut cls: Bound<'_, PyAny> = module.into_any();
            for part in class_path.split('.') {
                cls = cls
                    .getattr(part)
                    .map_err(|e| SessionRefError::SerializationFailed {
                        type_tag: owned_tag.clone(),
                        source: Box::new(PyErrWrap(format!(
                            "failed to resolve `{part}` while looking up {owned_tag}: {e}"
                        ))),
                    })?;
            }

            // Call `cls.__blazen_deserialize__(bytes)`.
            let py_bytes = PyBytes::new(py, user_bytes);
            let instance = cls
                .call_method1(DESERIALIZE_DUNDER, (py_bytes,))
                .map_err(|e| SessionRefError::SerializationFailed {
                    type_tag: owned_tag.clone(),
                    source: Box::new(PyErrWrap(format!(
                        "{DESERIALIZE_DUNDER} raised while reconstructing {owned_tag}: {e}"
                    ))),
                })?;

            // Re-wrap in a `PySerializableSessionRef` so the rehydrated
            // entry is also serializable if the resumed workflow pauses
            // again. We cache the *original* user bytes here rather than
            // re-invoking `__blazen_serialize__` — the instance we just
            // built is semantically equal to the snapshot, so reusing the
            // captured payload is correct and cheaper.
            let interned = intern_type_tag(&owned_tag);
            Ok(Arc::new(PySerializableSessionRef {
                obj: instance.unbind(),
                type_tag: interned,
                user_bytes: user_bytes.to_vec(),
            }))
        },
    )
}

/// Split `"some.module.ClassName"` into `("some.module", "ClassName")`.
/// Handles nested qualnames like `"pkg.mod.Outer.Inner"` by assuming the
/// module path stops at the first segment that begins with an uppercase
/// letter; if ambiguous we fall back to treating everything before the
/// last `.` as the module and everything after as the class.
fn split_module_and_class(tag: &str) -> (&str, &str) {
    // Primary strategy: last-dot split — matches `module.qualname`
    // produced by `__module__` + `__qualname__` for simple cases. For
    // nested qualnames the caller traverses the `class_path` via
    // `str::split('.')` so we don't need to disambiguate here.
    match tag.rfind('.') {
        Some(idx) => (&tag[..idx], &tag[idx + 1..]),
        None => ("", tag),
    }
}

/// Small error wrapper so we can stuff a Python-side error string into
/// [`SessionRefError::SerializationFailed`]'s `source` field without
/// pulling in another error-chain crate.
#[derive(Debug)]
struct PyErrWrap(String);

impl std::fmt::Display for PyErrWrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PyErrWrap {}

// ---------------------------------------------------------------------------
// Helpers for the Python-side `resume_with_session_refs` API
// ---------------------------------------------------------------------------

/// Static reference to [`py_deserializer_trampoline`] coerced to the
/// core [`SessionRefDeserializerFn`] alias. Re-exported for use in
/// `workflow::workflow::PyWorkflow::resume_with_session_refs`.
pub(crate) const DESERIALIZER_FN: SessionRefDeserializerFn = py_deserializer_trampoline;

/// Assert at compile time that [`PySerializableSessionRef`] satisfies
/// the bounds required by `Arc<dyn SessionRefSerializable>` and by the
/// `Arc<dyn Any + Send + Sync>` main registry map.
const _ASSERT_BOUNDS: () = {
    fn assert_send_sync_any<T: Any + Send + Sync>() {}
    fn _bounds() {
        assert_send_sync_any::<PySerializableSessionRef>();
    }
};
