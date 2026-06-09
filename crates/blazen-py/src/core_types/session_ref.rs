//! Python wrappers for the per-context session-ref registry types from
//! [`blazen_core::session_ref`].
//!
//! These bind the *core-crate* variant of `RemoteRefDescriptor` (used by
//! the in-process [`SessionRefRegistry`]). The
//! [`crate::peer`] module exposes the wire-protocol variant under the
//! distinct name `PeerRemoteRefDescriptor`.

use std::sync::Arc;

use blazen_macros::py_async;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_core::session_ref::{RefLifetime, RegistryKey, RemoteRefDescriptor, SessionRefRegistry};

// ---------------------------------------------------------------------------
// PyRefLifetime
// ---------------------------------------------------------------------------

/// Lifetime policy for a single session-ref entry.
///
/// Mirrors [`RefLifetime`]:
///
/// - `UntilContextDrop` -- purged when the owning `Context` is dropped
///   (default, preserves pre-Phase-11.2 behaviour).
/// - `UntilExplicitDrop` -- never purged automatically; caller must
///   invoke `SessionRefRegistry.remove(key)`.
/// - `UntilSnapshot` -- purged the next time the snapshot walker runs.
/// - `UntilParentFinish` -- purged only when the registry-owning
///   `Context` (typically the parent in a parent/child workflow) drops.
#[gen_stub_pyclass_enum]
#[pyclass(name = "RefLifetime", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRefLifetime {
    UntilContextDrop,
    UntilExplicitDrop,
    UntilSnapshot,
    UntilParentFinish,
}

impl From<PyRefLifetime> for RefLifetime {
    fn from(p: PyRefLifetime) -> Self {
        match p {
            PyRefLifetime::UntilContextDrop => RefLifetime::UntilContextDrop,
            PyRefLifetime::UntilExplicitDrop => RefLifetime::UntilExplicitDrop,
            PyRefLifetime::UntilSnapshot => RefLifetime::UntilSnapshot,
            PyRefLifetime::UntilParentFinish => RefLifetime::UntilParentFinish,
        }
    }
}

impl From<RefLifetime> for PyRefLifetime {
    fn from(c: RefLifetime) -> Self {
        match c {
            RefLifetime::UntilContextDrop => PyRefLifetime::UntilContextDrop,
            RefLifetime::UntilExplicitDrop => PyRefLifetime::UntilExplicitDrop,
            RefLifetime::UntilSnapshot => PyRefLifetime::UntilSnapshot,
            RefLifetime::UntilParentFinish => PyRefLifetime::UntilParentFinish,
        }
    }
}

// ---------------------------------------------------------------------------
// PyRegistryKey
// ---------------------------------------------------------------------------

/// Strongly-typed UUID key identifying an entry in a
/// [`PySessionRefRegistry`].
///
/// Construct one from a UUID string with `RegistryKey(uuid_str)` or mint
/// a fresh random one with `RegistryKey.new()`.
#[gen_stub_pyclass]
#[pyclass(name = "RegistryKey", frozen, eq, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyRegistryKey {
    pub(crate) inner: RegistryKey,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegistryKey {
    /// Construct a [`PyRegistryKey`] by parsing a UUID string.
    #[new]
    fn new_py(uuid_str: &str) -> PyResult<Self> {
        let inner = RegistryKey::parse(uuid_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid UUID: {e}")))?;
        Ok(Self { inner })
    }

    /// Mint a fresh random key.
    #[staticmethod]
    fn new_random() -> Self {
        Self {
            inner: RegistryKey::new(),
        }
    }

    /// Return the wrapped UUID as a string.
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.0.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("RegistryKey({})", self.inner)
    }
}

impl From<RegistryKey> for PyRegistryKey {
    fn from(inner: RegistryKey) -> Self {
        Self { inner }
    }
}

impl From<PyRegistryKey> for RegistryKey {
    fn from(py: PyRegistryKey) -> Self {
        py.inner
    }
}

// ---------------------------------------------------------------------------
// PyRemoteRefDescriptor (core variant)
// ---------------------------------------------------------------------------

/// In-process `RemoteRefDescriptor` (the core-crate variant stored in the
/// `SessionRefRegistry::remote_refs` sidecar).
///
/// This is *distinct* from the wire-protocol descriptor exposed by the
/// peer module as `PeerRemoteRefDescriptor` -- that one carries an
/// envelope version and is what the gRPC layer encodes. The class is
/// named `RemoteRefDescriptor` on the Python side because it matches
/// the type name in `blazen_core::session_ref`.
#[gen_stub_pyclass]
#[pyclass(name = "RemoteRefDescriptor", from_py_object)]
#[derive(Clone)]
pub struct PyRemoteRefDescriptor {
    pub(crate) inner: RemoteRefDescriptor,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRemoteRefDescriptor {
    /// Construct a new descriptor.
    ///
    /// Args:
    ///     origin_node_id: Stable identifier of the peer node that owns
    ///         the underlying value.
    ///     type_tag: Type tag mirroring
    ///         `SessionRefSerializable.blazen_type_tag`.
    ///     created_at_epoch_ms: Wall-clock creation time on the origin
    ///         node, in milliseconds since the Unix epoch.
    #[new]
    fn new(origin_node_id: String, type_tag: String, created_at_epoch_ms: u64) -> Self {
        Self {
            inner: RemoteRefDescriptor {
                origin_node_id,
                type_tag,
                created_at_epoch_ms,
            },
        }
    }

    #[getter]
    fn origin_node_id(&self) -> &str {
        &self.inner.origin_node_id
    }

    #[getter]
    fn type_tag(&self) -> &str {
        &self.inner.type_tag
    }

    #[getter]
    fn created_at_epoch_ms(&self) -> u64 {
        self.inner.created_at_epoch_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "RemoteRefDescriptor(origin_node_id={:?}, type_tag={:?}, created_at_epoch_ms={})",
            self.inner.origin_node_id, self.inner.type_tag, self.inner.created_at_epoch_ms,
        )
    }
}

impl From<RemoteRefDescriptor> for PyRemoteRefDescriptor {
    fn from(inner: RemoteRefDescriptor) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PySessionRefRegistry
// ---------------------------------------------------------------------------

/// Python-visible handle to a [`SessionRefRegistry`].
///
/// The methods that mutate or query the registry are synchronous on the
/// Python side -- they bridge through [`block_on_context`] to drive the
/// underlying async `RwLock` to completion. This matches the existing
/// internal `_SessionRegistryHandle` shim and keeps the API ergonomic for
/// step bodies that already run inside a Tokio worker (where
/// `block_in_place` is the right primitive).
#[gen_stub_pyclass]
#[pyclass(name = "SessionRefRegistry", frozen, from_py_object)]
#[derive(Clone)]
pub struct PySessionRefRegistry {
    pub(crate) inner: Arc<SessionRefRegistry>,
}

#[py_async]
#[gen_stub_pymethods]
#[pymethods]
impl PySessionRefRegistry {
    /// Construct an empty registry.
    #[new]
    fn new_py() -> Self {
        Self {
            inner: Arc::new(SessionRefRegistry::new()),
        }
    }

    /// Number of currently live entries (main map).
    async fn len(&self) -> PyResult<usize> {
        let r = Arc::clone(&self.inner);
        Ok(r.len().await)
    }

    /// Whether the registry has any live entries.
    async fn is_empty(&self) -> PyResult<bool> {
        let r = Arc::clone(&self.inner);
        Ok(r.is_empty().await)
    }

    /// Snapshot every live key in the registry.
    async fn keys(&self) -> PyResult<Vec<PyRegistryKey>> {
        let r = Arc::clone(&self.inner);
        let raw: Vec<RegistryKey> = r.keys().await;
        Ok(raw.into_iter().map(PyRegistryKey::from).collect())
    }

    /// Look up the [`RefLifetime`] policy for a key.
    async fn lifetime_of(&self, key: PyRegistryKey) -> PyResult<Option<PyRefLifetime>> {
        let r = Arc::clone(&self.inner);
        Ok(r.lifetime_of(key.inner).await.map(PyRefLifetime::from))
    }

    /// Look up an in-process remote-ref descriptor by key, if the key is
    /// stored in the remote-ref sidecar instead of the main map.
    async fn get_remote(&self, key: PyRegistryKey) -> PyResult<Option<PyRemoteRefDescriptor>> {
        let r = Arc::clone(&self.inner);
        Ok(r.get_remote(key.inner)
            .await
            .map(PyRemoteRefDescriptor::from))
    }

    /// Whether `key` resolves to a remote-ref entry rather than a local
    /// live value.
    async fn is_remote(&self, key: PyRegistryKey) -> PyResult<bool> {
        let r = Arc::clone(&self.inner);
        Ok(r.is_remote(key.inner).await)
    }

    /// Snapshot every `(key, descriptor)` pair currently in the
    /// remote-ref sidecar.
    async fn remote_entries(&self) -> PyResult<Vec<(PyRegistryKey, PyRemoteRefDescriptor)>> {
        let r = Arc::clone(&self.inner);
        let entries = r.remote_entries().await;
        Ok(entries
            .into_iter()
            .map(|(k, v)| (PyRegistryKey::from(k), PyRemoteRefDescriptor::from(v)))
            .collect())
    }

    /// Remove a single entry. Returns `True` if the key was present.
    ///
    /// Also clears any matching serializable, lifetime, and remote-ref
    /// sidecar entries.
    async fn remove(&self, key: PyRegistryKey) -> PyResult<bool> {
        let r = Arc::clone(&self.inner);
        Ok(r.remove(key.inner).await.is_some())
    }

    /// Drain all entries (main map and every sidecar). Returns the
    /// number of entries removed from the main map.
    async fn drain(&self) -> PyResult<usize> {
        let r = Arc::clone(&self.inner);
        Ok(r.drain().await)
    }

    /// Purge all refs whose [`RefLifetime`] policy says they should be
    /// dropped when the owning `Context` is dropped. Returns the number
    /// removed.
    ///
    /// Args:
    ///     owns_registry: Mirror of `Context::owns_registry` -- `True`
    ///         means the calling `Context` owns this registry.
    async fn clear_on_context_drop(&self, owns_registry: bool) -> PyResult<usize> {
        let r = Arc::clone(&self.inner);
        Ok(r.clear_on_context_drop(owns_registry).await)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let r = Arc::clone(&self.inner);
        let n = crate::convert::block_on_context(py, async move { r.len().await });
        format!("SessionRefRegistry(len={n})")
    }
}

impl PySessionRefRegistry {
    /// Internal helper for callers that already hold an
    /// `Arc<SessionRefRegistry>` (e.g. a `Context` that wants to expose
    /// its registry to Python).
    #[must_use]
    pub fn from_arc(inner: Arc<SessionRefRegistry>) -> Self {
        Self { inner }
    }
}
