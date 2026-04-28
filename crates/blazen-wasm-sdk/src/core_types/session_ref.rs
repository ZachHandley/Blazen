//! `wasm-bindgen` wrappers for the session-ref machinery exposed by
//! [`blazen_core::session_ref`].
//!
//! The WASM SDK does not currently support inserting arbitrary
//! `Send + Sync + 'static` Rust values from JavaScript — the registry's
//! [`SessionRefRegistry::insert`] generic API requires a concrete type that
//! is only meaningful inside Rust. What JS callers DO need is the ability
//! to:
//!
//! - inspect a registry passed to them by the engine (e.g. for diagnostics
//!   or capability provider plumbing),
//! - construct [`RegistryKey`]s from existing UUID strings carried inside
//!   a snapshot or a remote-peer response,
//! - read [`RemoteRefDescriptor`]s out of a registry to surface them to JS
//!   for telemetry / UI display, and
//! - serialise / inspect [`RefLifetime`] policy values.
//!
//! Mutation that requires a concrete Rust type (the `insert*` family) is
//! intentionally NOT exposed here. JS-side session refs travel through the
//! `Context.session` namespace as JSON-serialised values, which is the
//! shape the engine itself stores them in for `wasm32` builds.

use blazen_core::{RefLifetime, RegistryKey, RemoteRefDescriptor, SessionRefRegistry};
use serde::Serialize;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

use super::block_on_local;

// ---------------------------------------------------------------------------
// Marshalling helpers
// ---------------------------------------------------------------------------

/// Convert a `Serialize` value into a `JsValue` shaped as a plain JS object.
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

// ---------------------------------------------------------------------------
// WasmRegistryKey
// ---------------------------------------------------------------------------

/// JS-facing handle for a [`blazen_core::RegistryKey`].
///
/// `RegistryKey` is a thin newtype around a [`uuid::Uuid`]; the JS surface
/// exposes string parsing, formatting, and a fresh-key constructor so JS
/// callers can hold a stable key reference between calls without serialising
/// through the wire JSON every time.
#[wasm_bindgen(js_name = "RegistryKey")]
#[derive(Clone, Copy)]
pub struct WasmRegistryKey {
    inner: RegistryKey,
}

impl WasmRegistryKey {
    /// Wrap an existing [`RegistryKey`].
    #[must_use]
    pub(crate) fn from_inner(inner: RegistryKey) -> Self {
        Self { inner }
    }

    /// Borrow the underlying [`RegistryKey`].
    #[must_use]
    pub(crate) fn inner(&self) -> RegistryKey {
        self.inner
    }
}

#[wasm_bindgen(js_class = "RegistryKey")]
impl WasmRegistryKey {
    /// Mint a fresh, never-before-seen [`RegistryKey`]. Mirrors
    /// [`RegistryKey::new`].
    #[wasm_bindgen(js_name = "fresh")]
    #[must_use]
    pub fn fresh() -> WasmRegistryKey {
        Self {
            inner: RegistryKey::new(),
        }
    }

    /// Parse a [`RegistryKey`] from its UUID string representation.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `s` is not a valid UUID.
    #[wasm_bindgen(js_name = "parse")]
    pub fn parse(s: &str) -> Result<WasmRegistryKey, JsValue> {
        RegistryKey::parse(s)
            .map(|k| Self { inner: k })
            .map_err(|e| JsValue::from_str(&format!("RegistryKey.parse: {e}")))
    }

    /// Return the canonical UUID string for this key.
    #[wasm_bindgen(js_name = "toString")]
    #[must_use]
    pub fn to_js_string(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// WasmRemoteRefDescriptor
// ---------------------------------------------------------------------------

/// JS-facing handle for a [`blazen_core::RemoteRefDescriptor`].
///
/// The descriptor identifies a session ref whose underlying value lives on
/// a remote peer. JS callers receive these from
/// [`WasmSessionRefRegistry::remoteEntries`] and can inspect the
/// peer's `origin_node_id`, the value's `type_tag`, and the creation
/// timestamp without touching the underlying transport.
#[wasm_bindgen(js_name = "RemoteRefDescriptor")]
pub struct WasmRemoteRefDescriptor {
    inner: RemoteRefDescriptor,
}

impl WasmRemoteRefDescriptor {
    /// Wrap an existing [`RemoteRefDescriptor`].
    #[must_use]
    pub(crate) fn from_inner(inner: RemoteRefDescriptor) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = "RemoteRefDescriptor")]
impl WasmRemoteRefDescriptor {
    /// Construct a fresh descriptor.
    ///
    /// `created_at_epoch_ms` is a wall-clock epoch in milliseconds; bindings
    /// typically obtain it via `Date.now()` on the JS side.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(
        origin_node_id: String,
        type_tag: String,
        created_at_epoch_ms: u64,
    ) -> WasmRemoteRefDescriptor {
        Self {
            inner: RemoteRefDescriptor {
                origin_node_id,
                type_tag,
                created_at_epoch_ms,
            },
        }
    }

    /// Stable identifier of the peer node that owns the underlying value.
    #[wasm_bindgen(getter, js_name = "originNodeId")]
    #[must_use]
    pub fn origin_node_id(&self) -> String {
        self.inner.origin_node_id.clone()
    }

    /// Type tag matching the peer's
    /// [`SessionRefSerializable::blazen_type_tag`].
    #[wasm_bindgen(getter, js_name = "typeTag")]
    #[must_use]
    pub fn type_tag(&self) -> String {
        self.inner.type_tag.clone()
    }

    /// Wall-clock epoch in milliseconds at which the ref was created on the
    /// peer.
    #[wasm_bindgen(getter, js_name = "createdAtEpochMs")]
    #[must_use]
    pub fn created_at_epoch_ms(&self) -> u64 {
        self.inner.created_at_epoch_ms
    }

    /// Marshal the descriptor into a plain JS object.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(js_name = "toJsObject")]
    pub fn to_js_object(&self) -> Result<JsValue, JsValue> {
        marshal_to_js(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// WasmRefLifetime
// ---------------------------------------------------------------------------

/// JS-facing enum mirroring [`blazen_core::RefLifetime`].
///
/// Surfaced as a string-valued enum so the JS side reads e.g.
/// `"until_context_drop"` rather than an opaque numeric discriminant. This
/// matches the `serde(rename_all = "snake_case")` representation used by
/// [`RefLifetime`] when round-tripping through JSON.
#[wasm_bindgen(js_name = "RefLifetime")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WasmRefLifetime {
    /// Purged when the owning [`blazen_core::Context`] is dropped. Default.
    UntilContextDrop = 0,
    /// Never purged automatically.
    UntilExplicitDrop = 1,
    /// Purged the next time a snapshot is taken.
    UntilSnapshot = 2,
    /// Purged only when the parent (registry-owning) context finishes.
    UntilParentFinish = 3,
}

impl From<RefLifetime> for WasmRefLifetime {
    fn from(value: RefLifetime) -> Self {
        match value {
            RefLifetime::UntilContextDrop => Self::UntilContextDrop,
            RefLifetime::UntilExplicitDrop => Self::UntilExplicitDrop,
            RefLifetime::UntilSnapshot => Self::UntilSnapshot,
            RefLifetime::UntilParentFinish => Self::UntilParentFinish,
        }
    }
}

impl From<WasmRefLifetime> for RefLifetime {
    fn from(value: WasmRefLifetime) -> Self {
        match value {
            WasmRefLifetime::UntilContextDrop => Self::UntilContextDrop,
            WasmRefLifetime::UntilExplicitDrop => Self::UntilExplicitDrop,
            WasmRefLifetime::UntilSnapshot => Self::UntilSnapshot,
            WasmRefLifetime::UntilParentFinish => Self::UntilParentFinish,
        }
    }
}

// ---------------------------------------------------------------------------
// WasmSessionRefRegistry
// ---------------------------------------------------------------------------

/// JS-facing handle for a [`blazen_core::SessionRefRegistry`].
///
/// The registry is `Arc`-backed, so cloning this wrapper bumps refcounts
/// without copying any data. JS callers receive registries from the engine
/// (typically via a capability provider or a context accessor); they can
/// inspect the live key set, look up [`RefLifetime`] policies, and read
/// [`RemoteRefDescriptor`]s, but they cannot insert arbitrary Rust values
/// (see the module-level note for why).
#[wasm_bindgen(js_name = "SessionRefRegistry")]
pub struct WasmSessionRefRegistry {
    inner: Arc<SessionRefRegistry>,
}

impl WasmSessionRefRegistry {
    /// Wrap an existing `Arc<SessionRefRegistry>`.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn from_arc(inner: Arc<SessionRefRegistry>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying `Arc<SessionRefRegistry>`.
    #[allow(dead_code)]
    pub(crate) fn arc(&self) -> &Arc<SessionRefRegistry> {
        &self.inner
    }
}

#[wasm_bindgen(js_class = "SessionRefRegistry")]
impl WasmSessionRefRegistry {
    /// Construct a fresh, empty registry.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> WasmSessionRefRegistry {
        Self {
            inner: Arc::new(SessionRefRegistry::new()),
        }
    }

    /// Number of live entries currently in the registry.
    #[wasm_bindgen]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> u32 {
        let inner = Arc::clone(&self.inner);
        block_on_local(async move { inner.len().await }) as u32
    }

    /// Whether the registry has any live entries.
    #[wasm_bindgen(js_name = "isEmpty")]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let inner = Arc::clone(&self.inner);
        block_on_local(async move { inner.is_empty().await })
    }

    /// Return every live key as an array of [`WasmRegistryKey`].
    #[wasm_bindgen]
    #[must_use]
    pub fn keys(&self) -> js_sys::Array {
        let inner = Arc::clone(&self.inner);
        let raw = block_on_local(async move { inner.keys().await });
        let out = js_sys::Array::new();
        for k in raw {
            out.push(&JsValue::from(WasmRegistryKey::from_inner(k)));
        }
        out
    }

    /// Look up the [`RefLifetime`] policy registered for `key`. Returns
    /// `undefined` if no entry exists under `key`.
    #[wasm_bindgen(js_name = "lifetimeOf")]
    #[must_use]
    pub fn lifetime_of(&self, key: &WasmRegistryKey) -> Option<WasmRefLifetime> {
        let inner = Arc::clone(&self.inner);
        let registry_key = key.inner();
        block_on_local(async move { inner.lifetime_of(registry_key).await })
            .map(WasmRefLifetime::from)
    }

    /// Whether `key` resolves to a value living on a remote peer.
    #[wasm_bindgen(js_name = "isRemote")]
    #[must_use]
    pub fn is_remote(&self, key: &WasmRegistryKey) -> bool {
        let inner = Arc::clone(&self.inner);
        let registry_key = key.inner();
        block_on_local(async move { inner.is_remote(registry_key).await })
    }

    /// Look up the [`RemoteRefDescriptor`] for `key`, if any.
    #[wasm_bindgen(js_name = "getRemote")]
    #[must_use]
    pub fn get_remote(&self, key: &WasmRegistryKey) -> Option<WasmRemoteRefDescriptor> {
        let inner = Arc::clone(&self.inner);
        let registry_key = key.inner();
        block_on_local(async move { inner.get_remote(registry_key).await })
            .map(WasmRemoteRefDescriptor::from_inner)
    }

    /// Iterate every remote-ref descriptor currently tracked, returning a
    /// JS array of `[key, descriptor]` tuples.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(js_name = "remoteEntries")]
    pub fn remote_entries(&self) -> Result<js_sys::Array, JsValue> {
        let inner = Arc::clone(&self.inner);
        let entries = block_on_local(async move { inner.remote_entries().await });
        let out = js_sys::Array::new();
        for (k, v) in entries {
            let pair = js_sys::Array::new();
            pair.push(&JsValue::from(WasmRegistryKey::from_inner(k)));
            pair.push(&JsValue::from(WasmRemoteRefDescriptor::from_inner(v)));
            out.push(&pair);
        }
        Ok(out)
    }

    /// Remove the entry under `key`. Returns `true` if an entry was removed.
    #[wasm_bindgen]
    pub fn remove(&self, key: &WasmRegistryKey) -> bool {
        let inner = Arc::clone(&self.inner);
        let registry_key = key.inner();
        block_on_local(async move { inner.remove(registry_key).await }).is_some()
    }

    /// Drain every entry from the registry, returning the number removed.
    #[wasm_bindgen]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn drain(&self) -> u32 {
        let inner = Arc::clone(&self.inner);
        block_on_local(async move { inner.drain().await }) as u32
    }
}

impl Default for WasmSessionRefRegistry {
    fn default() -> Self {
        Self::new()
    }
}
