//! Typed Node bindings for the session-ref registry primitives in
//! [`blazen_core::session_ref`].
//!
//! The Rust API stores live values keyed by [`RegistryKey`] (a UUID
//! newtype) and tracks per-ref [`RefLifetime`] policy plus an optional
//! remote-ref descriptor sidecar. JS code typically only needs to
//! inspect / round-trip the *metadata* — the live `Arc<dyn Any>` payload
//! has no JS analogue and remains owned by the Rust side. The wrappers
//! here therefore expose:
//!
//! - [`JsRegistryKey`]: opaque UUID handle.
//! - [`JsRefLifetime`]: per-ref lifetime policy (string enum).
//! - [`JsRemoteRefDescriptor`]: descriptor for remote-owned refs (the
//!   core-crate variant — distinct from `PeerRemoteRefDescriptor` in
//!   `crate::peer::types` which mirrors the wire-protocol form).
//! - [`JsSessionRefRegistry`]: registry handle that lets JS callers
//!   list keys, query lifetimes, look up remote descriptors, and remove
//!   entries. Insert paths that need a typed `Arc<dyn Any>` are not
//!   exposed here — those happen via the workflow context.

use std::sync::Arc;

use blazen_core::session_ref::{RefLifetime, RegistryKey, RemoteRefDescriptor, SessionRefRegistry};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use uuid::Uuid;

use crate::error::to_napi_error;

// ---------------------------------------------------------------------------
// RefLifetime
// ---------------------------------------------------------------------------

/// Per-ref lifetime policy. Mirrors
/// [`blazen_core::session_ref::RefLifetime`].
#[napi(string_enum, js_name = "RefLifetime")]
#[derive(Clone, Copy)]
pub enum JsRefLifetime {
    /// Purged when the owning context drops the registry.
    UntilContextDrop,
    /// Never auto-purged. Caller must remove explicitly.
    UntilExplicitDrop,
    /// Purged the next time the snapshot walker runs.
    UntilSnapshot,
    /// Purged when the parent context that owns the registry finishes.
    UntilParentFinish,
}

impl From<JsRefLifetime> for RefLifetime {
    fn from(p: JsRefLifetime) -> Self {
        match p {
            JsRefLifetime::UntilContextDrop => Self::UntilContextDrop,
            JsRefLifetime::UntilExplicitDrop => Self::UntilExplicitDrop,
            JsRefLifetime::UntilSnapshot => Self::UntilSnapshot,
            JsRefLifetime::UntilParentFinish => Self::UntilParentFinish,
        }
    }
}

impl From<RefLifetime> for JsRefLifetime {
    fn from(p: RefLifetime) -> Self {
        match p {
            RefLifetime::UntilContextDrop => Self::UntilContextDrop,
            RefLifetime::UntilExplicitDrop => Self::UntilExplicitDrop,
            RefLifetime::UntilSnapshot => Self::UntilSnapshot,
            RefLifetime::UntilParentFinish => Self::UntilParentFinish,
        }
    }
}

// ---------------------------------------------------------------------------
// RegistryKey
// ---------------------------------------------------------------------------

/// Opaque UUID handle for a session-ref entry. Mirrors
/// [`blazen_core::session_ref::RegistryKey`].
#[napi(js_name = "RegistryKey")]
pub struct JsRegistryKey {
    pub(crate) inner: RegistryKey,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::new_without_default
)]
impl JsRegistryKey {
    /// Mint a fresh random key.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RegistryKey::new(),
        }
    }

    /// Parse a key from a UUID string.
    #[allow(clippy::needless_pass_by_value)]
    #[napi(factory)]
    pub fn parse(text: String) -> Result<Self> {
        let inner = RegistryKey::parse(&text).map_err(to_napi_error)?;
        Ok(Self { inner })
    }

    /// String form of the underlying UUID.
    #[napi(js_name = "toString")]
    pub fn render_uuid(&self) -> String {
        self.inner.0.to_string()
    }

    /// Alias for [`Self::render_uuid`] exposed as a getter so JS can do
    /// `key.uuid` in addition to `key.toString()`.
    #[napi(getter)]
    pub fn uuid(&self) -> String {
        self.inner.0.to_string()
    }
}

impl JsRegistryKey {
    pub(crate) fn from_native(inner: RegistryKey) -> Self {
        Self { inner }
    }

    #[allow(dead_code)]
    pub(crate) fn into_native(self) -> RegistryKey {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// RemoteRefDescriptor (core variant)
// ---------------------------------------------------------------------------

/// Descriptor for a session ref whose value lives on a remote peer.
/// Mirrors [`blazen_core::session_ref::RemoteRefDescriptor`].
///
/// This is the **core**-crate variant. The wire-protocol variant
/// (mirroring `blazen_peer::protocol::RemoteRefDescriptor`) is exposed
/// in `crate::peer::types` as `PeerRemoteRefDescriptor`; the two are
/// structurally identical but live on different sides of the
/// transport boundary.
#[napi(object, js_name = "RemoteRefDescriptor")]
pub struct JsRemoteRefDescriptor {
    /// Stable identifier of the peer node that owns the underlying value.
    #[napi(js_name = "originNodeId")]
    pub origin_node_id: String,
    /// Stable type tag matching `SessionRefSerializable::blazen_type_tag`.
    #[napi(js_name = "typeTag")]
    pub type_tag: String,
    /// Wall-clock creation time on the origin node, in milliseconds
    /// since the Unix epoch.
    #[napi(js_name = "createdAtEpochMs")]
    pub created_at_epoch_ms: i64,
}

impl JsRemoteRefDescriptor {
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn from_native(value: RemoteRefDescriptor) -> Self {
        Self {
            origin_node_id: value.origin_node_id,
            type_tag: value.type_tag,
            created_at_epoch_ms: value.created_at_epoch_ms as i64,
        }
    }

    #[allow(clippy::cast_sign_loss)]
    pub(crate) fn into_native(self) -> RemoteRefDescriptor {
        RemoteRefDescriptor {
            origin_node_id: self.origin_node_id,
            type_tag: self.type_tag,
            created_at_epoch_ms: self.created_at_epoch_ms.max(0) as u64,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionRefRegistry
// ---------------------------------------------------------------------------

/// Per-context registry of live session references.
///
/// Mirrors [`blazen_core::session_ref::SessionRefRegistry`]. JS callers
/// can construct an empty registry (rarely useful — workflow contexts
/// own one already) or wrap one obtained from another binding through
/// [`JsSessionRefRegistry::from_arc`].
///
/// The methods exposed here cover the read-only / metadata side of the
/// registry: enumerating keys, querying lifetimes, listing remote-ref
/// descriptors, removing entries, and draining. Insert paths that need
/// to register a live `Arc<dyn Any>` are not exposed because there is
/// no JS analogue for an arbitrary `Send + Sync + 'static` Rust value
/// — those go through the workflow context.
#[napi(js_name = "SessionRefRegistry")]
pub struct JsSessionRefRegistry {
    pub(crate) inner: Arc<SessionRefRegistry>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::new_without_default,
    clippy::needless_pass_by_value
)]
impl JsSessionRefRegistry {
    /// Construct an empty registry.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SessionRefRegistry::new()),
        }
    }

    /// Number of currently live entries in the main map.
    #[napi]
    pub async fn len(&self) -> u32 {
        u32::try_from(self.inner.len().await).unwrap_or(u32::MAX)
    }

    /// Whether the registry has any live entries.
    #[napi(js_name = "isEmpty")]
    pub async fn is_empty(&self) -> bool {
        self.inner.is_empty().await
    }

    /// Snapshot every key currently in the registry.
    #[napi]
    pub async fn keys(&self) -> Vec<JsRegistryKey> {
        self.inner
            .keys()
            .await
            .into_iter()
            .map(JsRegistryKey::from_native)
            .collect()
    }

    /// Look up the [`JsRefLifetime`] policy registered for `key`. Returns
    /// `null` if no entry exists under `key`.
    #[napi(js_name = "lifetimeOf")]
    pub async fn lifetime_of(&self, key: &JsRegistryKey) -> Option<JsRefLifetime> {
        self.inner
            .lifetime_of(key.inner)
            .await
            .map(JsRefLifetime::from)
    }

    /// Look up the remote-ref descriptor for `key`. Returns `null` when
    /// `key` is not a remote ref (it may still be a local ref — call
    /// [`Self::has`] to check).
    #[napi(js_name = "getRemote")]
    pub async fn get_remote(&self, key: &JsRegistryKey) -> Option<JsRemoteRefDescriptor> {
        self.inner
            .get_remote(key.inner)
            .await
            .map(JsRemoteRefDescriptor::from_native)
    }

    /// `true` if `key` is registered as a remote ref.
    #[napi(js_name = "isRemote")]
    pub async fn is_remote(&self, key: &JsRegistryKey) -> bool {
        self.inner.is_remote(key.inner).await
    }

    /// Insert (or overwrite) the remote-ref descriptor for `key`.
    #[napi(js_name = "insertRemote")]
    pub async fn insert_remote(
        &self,
        key: &JsRegistryKey,
        descriptor: JsRemoteRefDescriptor,
    ) -> Result<()> {
        self.inner
            .insert_remote(key.inner, descriptor.into_native())
            .await
            .map_err(to_napi_error)
    }

    /// All remote-ref descriptors currently tracked, as a `[uuid, descriptor]`
    /// tuple list. The UUID is rendered as its canonical string form so
    /// it round-trips cleanly through napi.
    #[napi(js_name = "remoteEntries")]
    pub async fn remote_entries(&self) -> Vec<RemoteEntry> {
        self.inner
            .remote_entries()
            .await
            .into_iter()
            .map(|(k, v)| RemoteEntry {
                uuid: k.0.to_string(),
                descriptor: JsRemoteRefDescriptor::from_native(v),
            })
            .collect()
    }

    /// Remove the entry under `key` (and any associated lifetime,
    /// serializable, or remote-ref sidecar). Returns `true` if a value
    /// was removed.
    #[napi]
    pub async fn remove(&self, key: &JsRegistryKey) -> bool {
        self.inner.remove(key.inner).await.is_some()
    }

    /// Drain the registry, returning the number of entries that were
    /// removed.
    #[napi]
    pub async fn drain(&self) -> u32 {
        u32::try_from(self.inner.drain().await).unwrap_or(u32::MAX)
    }

    /// Returns `true` when `key` is currently registered (either as a
    /// local entry or as a remote-ref placeholder).
    #[napi]
    pub async fn has(&self, key: &JsRegistryKey) -> bool {
        let local = self.inner.get_any(key.inner).await.is_some();
        if local {
            return true;
        }
        self.inner.is_remote(key.inner).await
    }

    /// Mint a duplicate key that resolves to the same underlying value
    /// as `srcKey`. The new key inherits the source's lifetime policy.
    #[napi(js_name = "cloneRef")]
    pub async fn clone_ref(&self, src_key: &JsRegistryKey) -> Result<JsRegistryKey> {
        let new_key = self
            .inner
            .clone_ref(src_key.inner)
            .await
            .map_err(to_napi_error)?;
        Ok(JsRegistryKey::from_native(new_key))
    }

    /// Move `srcKey` from this registry into `dstRegistry`. The key is
    /// preserved across the transfer so existing `__blazen_session_ref__`
    /// markers continue to resolve.
    #[napi(js_name = "transferRef")]
    pub async fn transfer_ref(
        &self,
        src_key: &JsRegistryKey,
        dst_registry: &JsSessionRefRegistry,
    ) -> Result<JsRegistryKey> {
        let key = self
            .inner
            .transfer_ref(src_key.inner, &dst_registry.inner)
            .await
            .map_err(to_napi_error)?;
        Ok(JsRegistryKey::from_native(key))
    }

    /// Parse a UUID string and return `true` if the registry currently
    /// holds an entry under that key. Convenience helper that avoids
    /// constructing a [`JsRegistryKey`] just for the lookup.
    #[napi(js_name = "hasUuid")]
    pub async fn has_uuid(&self, uuid: String) -> Result<bool> {
        let key = RegistryKey(Uuid::parse_str(&uuid).map_err(to_napi_error)?);
        let local = self.inner.get_any(key).await.is_some();
        if local {
            return Ok(true);
        }
        Ok(self.inner.is_remote(key).await)
    }
}

impl JsSessionRefRegistry {
    /// Wrap an existing `Arc<SessionRefRegistry>` (e.g. obtained from a
    /// running workflow context) so JS callers can inspect it.
    #[allow(dead_code)]
    pub(crate) fn from_arc(inner: Arc<SessionRefRegistry>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying `Arc` for code that needs to install the
    /// registry as the ambient session-ref registry.
    #[allow(dead_code)]
    pub(crate) fn arc(&self) -> Arc<SessionRefRegistry> {
        Arc::clone(&self.inner)
    }
}

/// Tuple-shaped `(uuid, descriptor)` entry returned by
/// [`JsSessionRefRegistry::remote_entries`].
#[napi(object, js_name = "RemoteRefEntry")]
pub struct RemoteEntry {
    /// UUID of the registry entry rendered as a canonical string.
    pub uuid: String,
    /// Descriptor of the remote ref.
    pub descriptor: JsRemoteRefDescriptor,
}
