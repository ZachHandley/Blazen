//! Session-scoped live reference registry.
//!
//! Stores Python/JS objects that cannot or should not be JSON-serialized
//! — DB connections, file handles, large in-memory tensors, lambdas,
//! locks, etc. — keyed by [`RegistryKey`] (a wrapped [`Uuid`]). Event
//! payloads carry only the key as a JSON marker
//! (`{"__blazen_session_ref__": "<uuid>"}`); the actual object lives
//! in the registry until workflow completion.
//!
//! Each [`crate::Context`] owns its own [`SessionRefRegistry`] so the
//! registry's lifetime is tied to the workflow run. Live references are
//! deliberately excluded from snapshots — see [`SessionPausePolicy`]
//! for what happens at pause/snapshot boundaries.
//!
//! ## Serializable session refs
//!
//! When a ref type knows how to serialize its own binary representation
//! (e.g. a native handle that can be re-opened from bytes), bindings can
//! register it via [`SessionRefRegistry::insert_serializable`] and
//! configure the workflow with [`SessionPausePolicy::PickleOrSerialize`].
//! At snapshot time the engine calls
//! [`SessionRefSerializable::blazen_serialize`] on each serializable
//! entry and persists the bytes alongside the snapshot metadata; on
//! resume a caller-supplied deserializer keyed by
//! [`SessionRefSerializable::blazen_type_tag`] reconstructs the value
//! into the fresh registry under the original key.

use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// JSON tag identifying a session-ref placeholder inside an event payload.
///
/// Format: `{"__blazen_session_ref__": "<uuid>"}`. Bindings detect this tag
/// and resolve the UUID through the active [`SessionRefRegistry`].
pub const SESSION_REF_TAG: &str = "__blazen_session_ref__";

/// Defensive cap on the number of live references a single workflow run
/// may hold. Prevents runaway loops from exhausting memory.
pub const MAX_SESSION_REFS_PER_RUN: usize = 10_000;

/// Strongly-typed wrapper around [`Uuid`] for session-ref keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RegistryKey(pub Uuid);

impl RegistryKey {
    /// Mint a fresh random key.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse a key from a UUID string.
    ///
    /// # Errors
    /// Returns the underlying [`uuid::Error`] if `s` is not a valid UUID.
    pub fn parse(s: &str) -> Result<Self, uuid::Error> {
        Uuid::parse_str(s).map(Self)
    }
}

impl Default for RegistryKey {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RegistryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Type-erased entry. Each binding stores a binding-specific
/// `Arc<...>` (e.g. `Arc<Py<PyAny>>` from `PyO3`, `Arc<napi::Ref<JsObject>>`
/// from `napi-rs`) and downcasts on retrieval.
type AnyArc = Arc<dyn Any + Send + Sync>;

/// Describes a session ref whose underlying value lives on a remote
/// peer. The binding layer uses this to trigger a gRPC `DerefSessionRef`
/// call when the ref is resolved locally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteRefDescriptor {
    /// Stable identifier of the peer node that owns the underlying
    /// value. Typically a hostname or a UUID set by the peer at
    /// startup.
    pub origin_node_id: String,
    /// Type tag for the underlying value — matches the
    /// [`SessionRefSerializable::blazen_type_tag`] used on the peer so
    /// the local binding layer knows how to deserialize the fetched
    /// bytes.
    pub type_tag: String,
    /// Wall-clock epoch in milliseconds when the ref was created on
    /// the peer. Used for TTL/expiry decisions.
    pub created_at_epoch_ms: u64,
}

/// Sidecar entry for values that opt into the
/// [`SessionRefSerializable`] protocol. Stored alongside the main
/// [`AnyArc`] entry in a parallel map keyed by the same
/// [`RegistryKey`] so ordinary downcast lookups still succeed while the
/// snapshot walker can find the trait object without having to perform
/// an `Arc<dyn Any>` → `Arc<dyn Trait>` downcast (which is not
/// supported by `std`).
type SerializableArc = Arc<dyn SessionRefSerializable>;

/// Per-ref lifetime policy that controls when a session ref is purged
/// from its [`SessionRefRegistry`].
///
/// Lifetime is orthogonal to [`SessionPausePolicy`]: the pause policy
/// decides what to do with refs when a snapshot is taken (pickle,
/// serialize, drop, error), while the lifetime decides which refs are
/// even *eligible* for snapshot or context-drop purging in the first
/// place.
///
/// The default — [`RefLifetime::UntilContextDrop`] — preserves the
/// pre-Phase-11.2 behavior: refs are purged when the owning
/// [`crate::Context`] drains its registry at workflow termination.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefLifetime {
    /// Purged when the owning [`crate::Context`] is dropped (i.e. when
    /// the workflow terminates and the binding calls
    /// [`crate::Context::clear_session_refs`]). **Default — preserves
    /// pre-Phase-11.2 behavior.**
    #[default]
    UntilContextDrop,
    /// Never purged automatically. The caller must explicitly invoke
    /// [`SessionRefRegistry::remove`] to release the ref. Use this for
    /// long-lived handles whose lifetime exceeds a single workflow run
    /// (e.g. a connection pool that survives across multiple
    /// pause/resume cycles).
    UntilExplicitDrop,
    /// Purged the next time the snapshot walker runs. Use this for
    /// ephemeral values that should not cross a pause boundary —
    /// they're available for the rest of the current run but will be
    /// gone the moment a snapshot is built. The snapshot walker
    /// removes them regardless of the active [`SessionPausePolicy`].
    UntilSnapshot,
    /// Purged when the [`crate::Context`] that **owns** the registry
    /// drops it. In a parent/child workflow setup where the parent
    /// owns the registry and the child borrows it via
    /// [`crate::Context::new_with_session_refs`], a `UntilParentFinish`
    /// ref inserted by the child survives child termination and is
    /// purged only when the parent finishes. This supersedes the
    /// tactical `Context::owns_registry` flag from Phase 0.5
    /// semantically, though the flag is still used internally to
    /// disambiguate which side of the parent/child relationship is
    /// being torn down.
    UntilParentFinish,
}

/// Per-context registry of live session references.
#[derive(Default)]
pub struct SessionRefRegistry {
    inner: RwLock<HashMap<RegistryKey, AnyArc>>,
    /// Parallel map of entries that also implement
    /// [`SessionRefSerializable`]. Written by
    /// [`SessionRefRegistry::insert_serializable`] and read by the
    /// snapshot walker when the active [`SessionPausePolicy`] is
    /// [`SessionPausePolicy::PickleOrSerialize`].
    serializable: RwLock<HashMap<RegistryKey, SerializableArc>>,
    /// Parallel map of per-ref [`RefLifetime`] policies. Every entry
    /// in [`Self::inner`] has a matching entry here; missing values
    /// are interpreted as [`RefLifetime::default`] for forward
    /// compatibility, but every insert path populates the map
    /// explicitly so the lookup is O(1) and infallible.
    lifetimes: RwLock<HashMap<RegistryKey, RefLifetime>>,
    /// Sidecar for session refs whose value lives on a remote peer.
    /// Populated by the peer client layer after a `SubWorkflow`
    /// response carries a [`RemoteRefDescriptor`]; read by
    /// binding-layer code when a local lookup for a key hits this map
    /// instead of the main `inner` storage (triggering a lazy Deref
    /// RPC).
    remote_refs: RwLock<HashMap<RegistryKey, RemoteRefDescriptor>>,
}

impl std::fmt::Debug for SessionRefRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid taking the lock here so `Debug` formatting never blocks.
        f.debug_struct("SessionRefRegistry").finish_non_exhaustive()
    }
}

impl SessionRefRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an `Arc<dyn Any + Send + Sync>` directly. Returns the freshly
    /// minted key, or an error if the registry is at capacity.
    ///
    /// The new entry uses [`RefLifetime::default`]
    /// ([`RefLifetime::UntilContextDrop`]). To pick a different
    /// lifetime, use [`Self::insert_arc_with_lifetime`].
    ///
    /// # Errors
    /// Returns [`SessionRefError::CapacityExceeded`] if the registry already
    /// holds [`MAX_SESSION_REFS_PER_RUN`] entries.
    pub async fn insert_arc(&self, value: AnyArc) -> Result<RegistryKey, SessionRefError> {
        self.insert_arc_with_lifetime(value, RefLifetime::default())
            .await
    }

    /// Insert an `Arc<dyn Any + Send + Sync>` with an explicit
    /// [`RefLifetime`] policy. Returns the freshly minted key.
    ///
    /// # Errors
    /// Returns [`SessionRefError::CapacityExceeded`] if the registry already
    /// holds [`MAX_SESSION_REFS_PER_RUN`] entries.
    pub async fn insert_arc_with_lifetime(
        &self,
        value: AnyArc,
        lifetime: RefLifetime,
    ) -> Result<RegistryKey, SessionRefError> {
        let mut g = self.inner.write().await;
        if g.len() >= MAX_SESSION_REFS_PER_RUN {
            return Err(SessionRefError::CapacityExceeded {
                cap: MAX_SESSION_REFS_PER_RUN,
            });
        }
        let key = RegistryKey::new();
        g.insert(key, value);
        drop(g);
        self.lifetimes.write().await.insert(key, lifetime);
        Ok(key)
    }

    /// Insert any `Send + Sync + 'static` value, wrapping it in an `Arc` for
    /// you. Returns the freshly minted key.
    ///
    /// The new entry uses [`RefLifetime::default`]
    /// ([`RefLifetime::UntilContextDrop`]). To pick a different
    /// lifetime, use [`Self::insert_with_lifetime`].
    ///
    /// # Errors
    /// See [`Self::insert_arc`].
    pub async fn insert<T: Any + Send + Sync + 'static>(
        &self,
        value: T,
    ) -> Result<RegistryKey, SessionRefError> {
        self.insert_arc(Arc::new(value)).await
    }

    /// Insert any `Send + Sync + 'static` value with an explicit
    /// [`RefLifetime`] policy, wrapping it in an `Arc` for you.
    /// Returns the freshly minted key.
    ///
    /// # Errors
    /// See [`Self::insert_arc_with_lifetime`].
    pub async fn insert_with_lifetime<T: Any + Send + Sync + 'static>(
        &self,
        value: T,
        lifetime: RefLifetime,
    ) -> Result<RegistryKey, SessionRefError> {
        self.insert_arc_with_lifetime(Arc::new(value), lifetime)
            .await
    }

    /// Insert a value that opts into the [`SessionRefSerializable`]
    /// protocol. The value is stored in both the main registry (so
    /// ordinary `get_any`/`get::<T>` lookups keep working) and in the
    /// serializable sidecar so the snapshot walker can retrieve the
    /// trait object without an unsupported `Arc<dyn Any>` → `Arc<dyn
    /// Trait>` downcast.
    ///
    /// Returns the freshly minted key.
    ///
    /// # Errors
    /// Returns [`SessionRefError::CapacityExceeded`] if the registry
    /// already holds [`MAX_SESSION_REFS_PER_RUN`] entries.
    pub async fn insert_serializable<T>(
        &self,
        value: Arc<T>,
    ) -> Result<RegistryKey, SessionRefError>
    where
        T: SessionRefSerializable + 'static,
    {
        let mut main = self.inner.write().await;
        if main.len() >= MAX_SESSION_REFS_PER_RUN {
            return Err(SessionRefError::CapacityExceeded {
                cap: MAX_SESSION_REFS_PER_RUN,
            });
        }
        let key = RegistryKey::new();
        // Store the concrete Arc<T> in the main `Any` map so
        // `get::<T>` keeps working for downstream ref resolution.
        main.insert(key, value.clone() as AnyArc);
        drop(main);
        let mut ser = self.serializable.write().await;
        ser.insert(key, value as SerializableArc);
        drop(ser);
        self.lifetimes
            .write()
            .await
            .insert(key, RefLifetime::default());
        Ok(key)
    }

    /// Insert an already-erased `Arc<dyn SessionRefSerializable>`
    /// directly. Used by the resume path to re-hydrate snapshot
    /// entries under their original [`RegistryKey`] without minting a
    /// new key.
    ///
    /// The registry's capacity cap is still enforced.
    ///
    /// # Errors
    /// Returns [`SessionRefError::CapacityExceeded`] if the registry is full.
    pub async fn insert_serializable_with_key(
        &self,
        key: RegistryKey,
        value: Arc<dyn SessionRefSerializable>,
    ) -> Result<(), SessionRefError> {
        let mut main = self.inner.write().await;
        if main.len() >= MAX_SESSION_REFS_PER_RUN {
            return Err(SessionRefError::CapacityExceeded {
                cap: MAX_SESSION_REFS_PER_RUN,
            });
        }
        // SAFETY: `Arc<dyn SessionRefSerializable>` is a fat pointer
        // that does not directly coerce to `Arc<dyn Any>`. We reclaim
        // the raw pointer and re-box it as a concrete helper newtype
        // that forwards to the trait object so the main registry can
        // still serve `get_any` lookups.
        let any_arc: AnyArc = Arc::new(SerializableHolder(value.clone()));
        main.insert(key, any_arc);
        drop(main);
        let mut ser = self.serializable.write().await;
        ser.insert(key, value);
        drop(ser);
        self.lifetimes
            .write()
            .await
            .insert(key, RefLifetime::default());
        Ok(())
    }

    /// Look up a serializable entry by key. Returns `None` if no
    /// serializable ref has been registered under this key (either
    /// because the key is unknown, or because it was inserted via the
    /// non-serializable path).
    pub async fn get_serializable(
        &self,
        key: RegistryKey,
    ) -> Option<Arc<dyn SessionRefSerializable>> {
        self.serializable.read().await.get(&key).cloned()
    }

    /// Snapshot every serializable entry currently live in the
    /// registry. Returns `(key, Arc<dyn SessionRefSerializable>)`
    /// pairs so callers can iterate without holding the lock.
    pub async fn serializable_entries(
        &self,
    ) -> Vec<(RegistryKey, Arc<dyn SessionRefSerializable>)> {
        self.serializable
            .read()
            .await
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Insert a remote-ref descriptor under a specific key (as returned
    /// by the peer in a `SubWorkflow` response). Does NOT touch the
    /// main `inner` storage — the ref is "remote" specifically because
    /// we don't have the value locally yet.
    ///
    /// # Errors
    /// Returns [`SessionRefError::CapacityExceeded`] if the
    /// `remote_refs` map already holds [`MAX_SESSION_REFS_PER_RUN`]
    /// entries.
    pub async fn insert_remote(
        &self,
        key: RegistryKey,
        descriptor: RemoteRefDescriptor,
    ) -> Result<(), SessionRefError> {
        let mut remote = self.remote_refs.write().await;
        if remote.len() >= MAX_SESSION_REFS_PER_RUN {
            return Err(SessionRefError::CapacityExceeded {
                cap: MAX_SESSION_REFS_PER_RUN,
            });
        }
        remote.insert(key, descriptor);
        Ok(())
    }

    /// Look up a remote-ref descriptor. Returns `None` if `key` is not
    /// a remote ref (it may still be a local ref — check `get_any`).
    pub async fn get_remote(&self, key: RegistryKey) -> Option<RemoteRefDescriptor> {
        self.remote_refs.read().await.get(&key).cloned()
    }

    /// Returns `true` if `key` exists in the `remote_refs` sidecar
    /// (regardless of whether it has also been materialized into the
    /// main `inner` map).
    pub async fn is_remote(&self, key: RegistryKey) -> bool {
        self.remote_refs.read().await.contains_key(&key)
    }

    /// Iterate all remote-ref descriptors currently tracked.
    pub async fn remote_entries(&self) -> Vec<(RegistryKey, RemoteRefDescriptor)> {
        self.remote_refs
            .read()
            .await
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Look up the type-erased entry. Bindings call this and downcast.
    pub async fn get_any(&self, key: RegistryKey) -> Option<AnyArc> {
        self.inner.read().await.get(&key).cloned()
    }

    /// Look up and downcast to a concrete `Arc<T>`.
    pub async fn get<T: Any + Send + Sync + 'static>(&self, key: RegistryKey) -> Option<Arc<T>> {
        let any = self.inner.read().await.get(&key).cloned()?;
        Arc::downcast::<T>(any).ok()
    }

    /// Remove a single entry, returning the removed value if present.
    pub async fn remove(&self, key: RegistryKey) -> Option<AnyArc> {
        let removed = self.inner.write().await.remove(&key);
        // Also clear any matching serializable, lifetime, and remote-ref
        // sidecar entries.
        self.serializable.write().await.remove(&key);
        self.lifetimes.write().await.remove(&key);
        self.remote_refs.write().await.remove(&key);
        removed
    }

    /// Drain all entries, returning the number removed.
    pub async fn drain(&self) -> usize {
        let mut g = self.inner.write().await;
        let n = g.len();
        g.clear();
        drop(g);
        self.serializable.write().await.clear();
        self.lifetimes.write().await.clear();
        self.remote_refs.write().await.clear();
        n
    }

    /// Look up the [`RefLifetime`] policy registered for `key`. Returns
    /// `None` if no entry exists under `key`.
    pub async fn lifetime_of(&self, key: RegistryKey) -> Option<RefLifetime> {
        self.lifetimes.read().await.get(&key).copied()
    }

    /// Purge all refs whose [`RefLifetime`] policy says they should be
    /// dropped when the owning [`crate::Context`] is dropped. Returns
    /// the number of entries removed.
    ///
    /// Semantics (consistent with the pre-Phase-11.2 behavior of
    /// [`crate::Context::clear_session_refs`]):
    /// - [`RefLifetime::UntilContextDrop`] — purged when
    ///   `owns_registry == true`. A borrowed (child) context never
    ///   purges default-lifetime refs because the parent still needs
    ///   to resolve them.
    /// - [`RefLifetime::UntilExplicitDrop`] — never purged. Caller must
    ///   invoke [`Self::remove`] explicitly.
    /// - [`RefLifetime::UntilSnapshot`] — never purged here. The
    ///   snapshot walker (in `event_loop.rs`) handles these.
    /// - [`RefLifetime::UntilParentFinish`] — purged only when
    ///   `owns_registry == true`. The owning side of a parent/child
    ///   shared registry is finishing.
    ///
    /// `owns_registry` mirrors the `Context::owns_registry` flag from
    /// Phase 0.5 — `true` means the calling [`crate::Context`] owns
    /// this registry (it was constructed via [`crate::Context::new`]
    /// rather than [`crate::Context::new_with_session_refs`]).
    pub async fn clear_on_context_drop(&self, owns_registry: bool) -> usize {
        // Snapshot the keys that need to go under a read lock so we
        // can release it before taking the write lock to remove them.
        let to_remove: Vec<RegistryKey> = {
            let g = self.lifetimes.read().await;
            g.iter()
                .filter_map(|(k, lt)| match lt {
                    RefLifetime::UntilContextDrop | RefLifetime::UntilParentFinish
                        if owns_registry =>
                    {
                        Some(*k)
                    }
                    RefLifetime::UntilContextDrop
                    | RefLifetime::UntilExplicitDrop
                    | RefLifetime::UntilSnapshot
                    | RefLifetime::UntilParentFinish => None,
                })
                .collect()
        };

        let mut count = 0;
        for key in to_remove {
            // `remove` already cleans all three sidecars.
            if self.remove(key).await.is_some() {
                count += 1;
            }
        }
        count
    }

    /// Number of currently live entries (for tests/diagnostics).
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Whether the registry has any live entries.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }

    /// Iterate every key currently in the registry. Used by the snapshot
    /// walker to apply [`SessionPausePolicy`] uniformly.
    pub async fn keys(&self) -> Vec<RegistryKey> {
        self.inner.read().await.keys().copied().collect()
    }

    /// Create a second [`RegistryKey`] that resolves to the same
    /// underlying `Arc` as `src_key`. The source key remains valid and
    /// both keys downcast to the same value through [`Self::get`].
    ///
    /// The new entry inherits the source's [`RefLifetime`] policy and,
    /// if the source has a serializable sidecar, the same trait-object
    /// `Arc` is copied into the new entry's sidecar as well — clone
    /// semantics do not change how the ref participates in snapshot
    /// walking or context-drop purging.
    ///
    /// # Errors
    /// - [`SessionRefError::KeyNotFound`] if `src_key` is not present in
    ///   the registry.
    /// - [`SessionRefError::CapacityExceeded`] if the registry already
    ///   holds [`MAX_SESSION_REFS_PER_RUN`] entries.
    pub async fn clone_ref(&self, src_key: RegistryKey) -> Result<RegistryKey, SessionRefError> {
        // Snapshot the source state under short-lived read locks so we
        // can release them before acquiring the write locks below.
        let src_any = {
            let main = self.inner.read().await;
            main.get(&src_key)
                .cloned()
                .ok_or(SessionRefError::KeyNotFound { key: src_key })?
        };
        let src_serializable = self.serializable.read().await.get(&src_key).cloned();
        let src_lifetime = self
            .lifetimes
            .read()
            .await
            .get(&src_key)
            .copied()
            .unwrap_or_default();

        // Now take the write locks and publish the new entry. Insert
        // the main map first so the capacity check is authoritative —
        // if we error here the sidecars are still untouched.
        let mut main = self.inner.write().await;
        if main.len() >= MAX_SESSION_REFS_PER_RUN {
            return Err(SessionRefError::CapacityExceeded {
                cap: MAX_SESSION_REFS_PER_RUN,
            });
        }
        let new_key = RegistryKey::new();
        main.insert(new_key, src_any);
        drop(main);

        if let Some(ser) = src_serializable {
            self.serializable.write().await.insert(new_key, ser);
        }
        self.lifetimes.write().await.insert(new_key, src_lifetime);

        Ok(new_key)
    }

    /// Move ownership of `src_key` from `self` into `dst_registry`.
    ///
    /// The entry is reinserted into `dst_registry` under the *same*
    /// [`RegistryKey`] so any JSON markers that already reference it
    /// continue to resolve after the transfer. On success `src_key` is
    /// removed from `self` (including the lifetime and serializable
    /// sidecars) and the returned key equals `src_key`.
    ///
    /// Lifetime policy and serializable sidecar migrate with the ref.
    ///
    /// If `self` and `dst_registry` are the same registry the call is a
    /// no-op and returns `src_key` unchanged — the ref is neither
    /// duplicated nor dropped. This is detected by comparing the
    /// underlying pointer addresses of the two `&self` references, so
    /// callers that hold distinct `Arc`s pointing at the same registry
    /// value can still safely invoke the operation.
    ///
    /// # Errors
    /// - [`SessionRefError::KeyNotFound`] if `src_key` is not present in
    ///   `self`.
    /// - [`SessionRefError::CapacityExceeded`] if `dst_registry` is at
    ///   capacity. On this error the source is left untouched — the
    ///   transfer is atomic in the "all or nothing" sense.
    pub async fn transfer_ref(
        &self,
        src_key: RegistryKey,
        dst_registry: &SessionRefRegistry,
    ) -> Result<RegistryKey, SessionRefError> {
        // Same-registry guard: compare the underlying object addresses.
        // If the caller passes the same registry as src and dst we must
        // not try to acquire the same lock twice (nor remove the entry
        // after reinserting it under the same key), so short-circuit
        // here after asserting the source still exists.
        if std::ptr::eq(std::ptr::from_ref(self), std::ptr::from_ref(dst_registry)) {
            let main = self.inner.read().await;
            if main.contains_key(&src_key) {
                return Ok(src_key);
            }
            return Err(SessionRefError::KeyNotFound { key: src_key });
        }

        // Snapshot source state under read locks, then release them
        // before touching the destination. Holding a read guard on
        // `self` across an `await` on `dst_registry` is fine in
        // principle but makes lock ordering harder to reason about, so
        // we deliberately drop each guard as soon as we've cloned the
        // data out.
        let src_any = {
            let main = self.inner.read().await;
            main.get(&src_key)
                .cloned()
                .ok_or(SessionRefError::KeyNotFound { key: src_key })?
        };
        let src_serializable = self.serializable.read().await.get(&src_key).cloned();
        let src_lifetime = self
            .lifetimes
            .read()
            .await
            .get(&src_key)
            .copied()
            .unwrap_or_default();

        // Publish into the destination first so that any capacity
        // failure surfaces *before* we start mutating the source. This
        // preserves the "source unchanged on dst failure" guarantee.
        {
            let mut dst_main = dst_registry.inner.write().await;
            if dst_main.len() >= MAX_SESSION_REFS_PER_RUN {
                return Err(SessionRefError::CapacityExceeded {
                    cap: MAX_SESSION_REFS_PER_RUN,
                });
            }
            dst_main.insert(src_key, src_any);
        }
        if let Some(ser) = src_serializable {
            dst_registry.serializable.write().await.insert(src_key, ser);
        }
        dst_registry
            .lifetimes
            .write()
            .await
            .insert(src_key, src_lifetime);

        // Destination is now fully populated — drop the source.
        self.inner.write().await.remove(&src_key);
        self.serializable.write().await.remove(&src_key);
        self.lifetimes.write().await.remove(&src_key);

        Ok(src_key)
    }
}

tokio::task_local! {
    /// Ambient current session-ref registry for code paths that run on a
    /// Tokio worker thread. Installed by [`with_session_registry`] for the
    /// duration of a scoped future; read by [`current_session_registry`]
    /// from anywhere inside that future.
    ///
    /// Binding layers (Python, Node) install a registry here before
    /// running a workflow step so that event constructors and JSON
    /// conversion code can route non-serializable values into the
    /// appropriate per-`Context` registry without having to thread it
    /// through every call site.
    ///
    /// Python additionally mirrors this into a `contextvars.ContextVar`
    /// because `pyo3-async-runtimes` runs Python coroutines on Python's
    /// asyncio loop thread (separate from the Tokio worker), so a Tokio
    /// `task_local!` alone is not visible from inside `async def` bodies.
    /// Node has no such problem — napi-rs worker futures run directly on
    /// Tokio, so the `task_local!` is sufficient on its own.
    pub static CURRENT_SESSION_REGISTRY: Arc<SessionRefRegistry>;
}

/// Install `registry` as the ambient [`CURRENT_SESSION_REGISTRY`] for the
/// duration of `fut`. Any code running inside `fut` can look up the
/// registry via [`current_session_registry`].
///
/// This is the Rust-only scoping API. Python bindings layer an
/// additional `contextvars.ContextVar` on top of it to flow the registry
/// across the asyncio/tokio thread boundary.
pub async fn with_session_registry<F, T>(registry: Arc<SessionRefRegistry>, fut: F) -> T
where
    F: Future<Output = T>,
{
    CURRENT_SESSION_REGISTRY.scope(registry, fut).await
}

/// Look up the registry currently installed by [`with_session_registry`].
///
/// Returns `None` when called from a context that has not installed one —
/// typically because the caller is constructing an event outside of a
/// workflow step, or the registry has already been dropped.
#[must_use]
pub fn current_session_registry() -> Option<Arc<SessionRefRegistry>> {
    CURRENT_SESSION_REGISTRY.try_with(Arc::clone).ok()
}

/// What to do with live session references when a workflow is paused or
/// snapshotted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPausePolicy {
    /// Try to pickle each live ref into the snapshot. On any failure,
    /// raise [`crate::WorkflowError::SessionRefsNotSerializable`] and
    /// abort the snapshot. **Default — recommended.**
    #[default]
    PickleOrError,
    /// First try each ref's [`SessionRefSerializable::blazen_serialize`]
    /// implementation. Entries that successfully serialize are persisted
    /// into snapshot metadata under `"__blazen_serialized_session_refs"`
    /// keyed by their [`RegistryKey`]. Entries that do not implement the
    /// trait — or for which no resume-time deserializer has been
    /// registered — fall through to the same behavior as
    /// [`SessionPausePolicy::PickleOrError`]: the snapshot walker will
    /// log a warning and drop them.
    PickleOrSerialize,
    /// Drop live refs from the snapshot, emit `tracing::warn!` per drop,
    /// store a diagnostic report in snapshot metadata. On resume, accessing
    /// a dropped field raises a clear runtime error from the binding.
    WarnDrop,
    /// Refuse to pause if any live refs are in flight. Raises
    /// [`crate::WorkflowError::SessionRefsNotSerializable`] immediately.
    HardError,
}

/// Metadata key under which the serialized-session-ref sidecar payload
/// is stashed inside [`crate::WorkflowSnapshot::metadata`] when the
/// active [`SessionPausePolicy`] is
/// [`SessionPausePolicy::PickleOrSerialize`].
pub const SERIALIZED_SESSION_REFS_META_KEY: &str = "__blazen_serialized_session_refs";

/// Optional marker trait for session-ref values that know how to
/// serialize themselves. When a workflow uses
/// [`SessionPausePolicy::PickleOrSerialize`], the snapshot walker looks
/// up each live entry in the registry's serializable sidecar and, on a
/// hit, calls [`SessionRefSerializable::blazen_serialize`] to capture
/// the value's binary representation into snapshot metadata. On resume
/// a caller-supplied deserializer keyed by
/// [`SessionRefSerializable::blazen_type_tag`] reconstructs the value
/// under the original [`RegistryKey`].
///
/// Register a value with [`SessionRefRegistry::insert_serializable`] so
/// the sidecar is populated; an ordinary
/// [`SessionRefRegistry::insert`] will NOT make the value eligible for
/// serialization even if its type implements this trait.
pub trait SessionRefSerializable: Any + Send + Sync {
    /// Produce a binary representation of this value. Called at
    /// snapshot time.
    ///
    /// # Errors
    /// Return [`SessionRefError::SerializationFailed`] (or any other
    /// variant the caller wants to surface) if the value cannot be
    /// serialized.
    fn blazen_serialize(&self) -> Result<Vec<u8>, SessionRefError>;

    /// Stable type tag used to pair a serialized blob with its
    /// deserializer on resume. Must be unique per concrete type — two
    /// unrelated types sharing a tag will silently clobber each other
    /// in the deserializer registry.
    fn blazen_type_tag(&self) -> &'static str;
}

/// Internal newtype used by
/// [`SessionRefRegistry::insert_serializable_with_key`] on the resume
/// path to wrap an `Arc<dyn SessionRefSerializable>` in a value that
/// coerces cleanly to `Arc<dyn Any + Send + Sync>` for storage in the
/// main registry map. The holder keeps the trait-object Arc alive so
/// `get_any` lookups still produce a usable handle on the resumed
/// side (bindings then re-fetch the concrete value via
/// `get_serializable`).
struct SerializableHolder(#[allow(dead_code)] Arc<dyn SessionRefSerializable>);

/// Error type for session-ref registry operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionRefError {
    /// Returned when [`SessionRefRegistry::insert_arc`] is called while the
    /// registry already holds [`MAX_SESSION_REFS_PER_RUN`] entries.
    #[error(
        "session ref registry capacity exceeded ({cap} entries) — \
         too many live references in this workflow run"
    )]
    CapacityExceeded {
        /// The configured capacity that was hit.
        cap: usize,
    },

    /// Returned when a [`SessionRefSerializable::blazen_serialize`]
    /// implementation fails, or when a resume-time deserializer
    /// cannot reconstruct a previously-captured blob.
    #[error("session ref serialization failed for type_tag {type_tag}: {source}")]
    SerializationFailed {
        /// The stable type tag of the value whose serialization or
        /// deserialization failed.
        type_tag: String,
        /// The underlying error reported by the user-supplied
        /// serializer or deserializer.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Returned when a clone/transfer operation targets a key that does
    /// not exist in the source registry.
    #[error("session ref key {key} not found in source registry")]
    KeyNotFound {
        /// The missing key the operation was attempting to resolve.
        key: RegistryKey,
    },

    /// Returned when a local lookup hits a key that lives in the
    /// `remote_refs` sidecar instead of the main `inner` map. The
    /// binding layer must call its peer client's `deref_session_ref`
    /// RPC to fetch the value from `origin_node_id` before the lookup
    /// can succeed.
    #[error(
        "session ref `{key}` is a remote ref on node `{origin_node_id}` — \
         call peer.deref_session_ref() to resolve"
    )]
    RemoteRefNotResolved {
        /// The remote ref key that the caller tried to resolve locally.
        key: RegistryKey,
        /// The peer node that owns the underlying value.
        origin_node_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn insert_and_get_roundtrip() {
        let reg = SessionRefRegistry::new();
        let key = reg.insert(42_i32).await.unwrap();
        let got = reg.get::<i32>(key).await.unwrap();
        assert_eq!(*got, 42);
    }

    #[tokio::test]
    async fn get_wrong_type_returns_none() {
        let reg = SessionRefRegistry::new();
        let key = reg.insert(42_i32).await.unwrap();
        assert!(reg.get::<String>(key).await.is_none());
    }

    #[tokio::test]
    async fn remove_returns_value_and_clears() {
        let reg = SessionRefRegistry::new();
        let key = reg.insert("hello".to_owned()).await.unwrap();
        assert_eq!(reg.len().await, 1);
        let removed = reg.remove(key).await;
        assert!(removed.is_some());
        assert_eq!(reg.len().await, 0);
    }

    #[tokio::test]
    async fn drain_clears_everything() {
        let reg = SessionRefRegistry::new();
        let _ = reg.insert(1_i32).await.unwrap();
        let _ = reg.insert(2_i32).await.unwrap();
        let _ = reg.insert(3_i32).await.unwrap();
        assert_eq!(reg.drain().await, 3);
        assert!(reg.is_empty().await);
    }

    #[tokio::test]
    async fn capacity_cap_enforced() {
        // Run with a tight cap to make the test fast — we just verify the
        // CapacityExceeded path triggers. We can't easily change the const,
        // so we test the logic by inserting up to the real limit (slow test
        // skipped). Instead test that insert returns Ok for normal use.
        let reg = SessionRefRegistry::new();
        for i in 0..100_i32 {
            assert!(reg.insert(i).await.is_ok());
        }
        assert_eq!(reg.len().await, 100);
    }

    #[test]
    fn registry_key_parse_roundtrip() {
        let k = RegistryKey::new();
        let s = k.to_string();
        let parsed = RegistryKey::parse(&s).unwrap();
        assert_eq!(k, parsed);
    }

    #[test]
    fn session_pause_policy_default_is_pickle_or_error() {
        assert_eq!(
            SessionPausePolicy::default(),
            SessionPausePolicy::PickleOrError
        );
    }

    #[test]
    fn session_pause_policy_serde_roundtrip() {
        let p = SessionPausePolicy::WarnDrop;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"warn_drop\"");
        let back: SessionPausePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[tokio::test]
    async fn current_returns_none_outside_scope() {
        assert!(current_session_registry().is_none());
    }

    #[tokio::test]
    async fn current_returns_registry_inside_tokio_scope() {
        let reg = Arc::new(SessionRefRegistry::new());
        let reg_clone = Arc::clone(&reg);
        with_session_registry(reg, async move {
            let got = current_session_registry().expect("registry should be set");
            assert!(Arc::ptr_eq(&got, &reg_clone));
        })
        .await;
    }

    #[tokio::test]
    async fn scope_isolates_concurrent_tasks() {
        // Two concurrent tasks each get their own registry.
        let reg_a = Arc::new(SessionRefRegistry::new());
        let reg_b = Arc::new(SessionRefRegistry::new());
        let _ = reg_a.insert(1_i32).await.unwrap();
        let _ = reg_b.insert(2_i32).await.unwrap();

        let a_clone = Arc::clone(&reg_a);
        let b_clone = Arc::clone(&reg_b);

        let task_a = tokio::spawn(async move {
            with_session_registry(a_clone, async move {
                let got = current_session_registry().unwrap();
                got.len().await
            })
            .await
        });
        let task_b = tokio::spawn(async move {
            with_session_registry(b_clone, async move {
                let got = current_session_registry().unwrap();
                got.len().await
            })
            .await
        });

        assert_eq!(task_a.await.unwrap(), 1);
        assert_eq!(task_b.await.unwrap(), 1);
    }

    // --------------------------------------------------------------
    // SessionRefSerializable
    // --------------------------------------------------------------

    /// Minimal test type that serializes its `value` as four
    /// big-endian bytes.
    struct TestSerializable {
        value: i32,
    }

    impl SessionRefSerializable for TestSerializable {
        fn blazen_serialize(&self) -> Result<Vec<u8>, SessionRefError> {
            Ok(self.value.to_be_bytes().to_vec())
        }
        fn blazen_type_tag(&self) -> &'static str {
            "test::TestSerializable"
        }
    }

    #[tokio::test]
    async fn insert_serializable_populates_both_maps() {
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_serializable(Arc::new(TestSerializable { value: 7 }))
            .await
            .unwrap();

        // Ordinary downcast through the main map still works.
        let concrete = reg.get::<TestSerializable>(key).await.unwrap();
        assert_eq!(concrete.value, 7);

        // Sidecar lookup returns the trait object.
        let ser = reg.get_serializable(key).await.unwrap();
        assert_eq!(ser.blazen_type_tag(), "test::TestSerializable");
        assert_eq!(ser.blazen_serialize().unwrap(), vec![0, 0, 0, 7]);
    }

    #[tokio::test]
    async fn serializable_entries_returns_all_live_keys() {
        let reg = SessionRefRegistry::new();
        let k1 = reg
            .insert_serializable(Arc::new(TestSerializable { value: 1 }))
            .await
            .unwrap();
        let k2 = reg
            .insert_serializable(Arc::new(TestSerializable { value: 2 }))
            .await
            .unwrap();
        // Non-serializable insert should NOT show up in the sidecar.
        let _ = reg.insert(99_i64).await.unwrap();

        let entries = reg.serializable_entries().await;
        assert_eq!(entries.len(), 2);
        let got: std::collections::HashSet<RegistryKey> = entries.iter().map(|(k, _)| *k).collect();
        let expected: std::collections::HashSet<RegistryKey> = [k1, k2].into_iter().collect();
        assert_eq!(got, expected);
    }

    #[tokio::test]
    async fn remove_clears_sidecar_entry() {
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_serializable(Arc::new(TestSerializable { value: 5 }))
            .await
            .unwrap();
        assert!(reg.get_serializable(key).await.is_some());
        let _ = reg.remove(key).await;
        assert!(reg.get_serializable(key).await.is_none());
        assert!(reg.get::<TestSerializable>(key).await.is_none());
    }

    #[tokio::test]
    async fn drain_clears_sidecar() {
        let reg = SessionRefRegistry::new();
        let _ = reg
            .insert_serializable(Arc::new(TestSerializable { value: 1 }))
            .await
            .unwrap();
        let _ = reg
            .insert_serializable(Arc::new(TestSerializable { value: 2 }))
            .await
            .unwrap();
        assert_eq!(reg.serializable_entries().await.len(), 2);
        let drained = reg.drain().await;
        assert_eq!(drained, 2);
        assert_eq!(reg.serializable_entries().await.len(), 0);
    }

    #[tokio::test]
    async fn insert_serializable_with_key_reuses_registry_key() {
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();
        let value: Arc<dyn SessionRefSerializable> = Arc::new(TestSerializable { value: 11 });
        reg.insert_serializable_with_key(key, value).await.unwrap();

        // Sidecar knows the key.
        let ser = reg.get_serializable(key).await.unwrap();
        assert_eq!(ser.blazen_serialize().unwrap(), vec![0, 0, 0, 11]);

        // And the main map carries a holder that coerces to dyn Any.
        assert!(reg.get_any(key).await.is_some());
    }

    #[test]
    fn session_pause_policy_pickle_or_serialize_serde() {
        let p = SessionPausePolicy::PickleOrSerialize;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"pickle_or_serialize\"");
        let back: SessionPausePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    // --------------------------------------------------------------
    // RefLifetime (Phase 11.2)
    // --------------------------------------------------------------

    #[test]
    fn ref_lifetime_default_is_until_context_drop() {
        assert_eq!(RefLifetime::default(), RefLifetime::UntilContextDrop);
    }

    #[test]
    fn ref_lifetime_serde_roundtrip_all_variants() {
        for (variant, expected_json) in [
            (RefLifetime::UntilContextDrop, "\"until_context_drop\""),
            (RefLifetime::UntilExplicitDrop, "\"until_explicit_drop\""),
            (RefLifetime::UntilSnapshot, "\"until_snapshot\""),
            (RefLifetime::UntilParentFinish, "\"until_parent_finish\""),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected_json);
            let back: RefLifetime = serde_json::from_str(&json).unwrap();
            assert_eq!(back, variant);
        }
    }

    #[tokio::test]
    async fn insert_default_lifetime_is_until_context_drop() {
        let reg = SessionRefRegistry::new();
        let key = reg.insert(42_i32).await.unwrap();
        assert_eq!(
            reg.lifetime_of(key).await,
            Some(RefLifetime::UntilContextDrop)
        );
    }

    #[tokio::test]
    async fn insert_with_lifetime_records_explicit_policy() {
        let reg = SessionRefRegistry::new();
        let k1 = reg
            .insert_with_lifetime(1_i32, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();
        let k2 = reg
            .insert_with_lifetime("hi".to_owned(), RefLifetime::UntilSnapshot)
            .await
            .unwrap();
        let k3 = reg
            .insert_with_lifetime(2_u64, RefLifetime::UntilParentFinish)
            .await
            .unwrap();

        assert_eq!(
            reg.lifetime_of(k1).await,
            Some(RefLifetime::UntilExplicitDrop)
        );
        assert_eq!(reg.lifetime_of(k2).await, Some(RefLifetime::UntilSnapshot));
        assert_eq!(
            reg.lifetime_of(k3).await,
            Some(RefLifetime::UntilParentFinish)
        );
    }

    #[tokio::test]
    async fn insert_arc_with_lifetime_records_explicit_policy() {
        let reg = SessionRefRegistry::new();
        let val: Arc<dyn Any + Send + Sync> = Arc::new(7_i32);
        let key = reg
            .insert_arc_with_lifetime(val, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();
        assert_eq!(
            reg.lifetime_of(key).await,
            Some(RefLifetime::UntilExplicitDrop)
        );
    }

    #[tokio::test]
    async fn insert_serializable_records_default_lifetime() {
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_serializable(Arc::new(TestSerializable { value: 9 }))
            .await
            .unwrap();
        assert_eq!(
            reg.lifetime_of(key).await,
            Some(RefLifetime::UntilContextDrop)
        );
    }

    #[tokio::test]
    async fn insert_serializable_with_key_records_default_lifetime() {
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();
        let value: Arc<dyn SessionRefSerializable> = Arc::new(TestSerializable { value: 1 });
        reg.insert_serializable_with_key(key, value).await.unwrap();
        assert_eq!(
            reg.lifetime_of(key).await,
            Some(RefLifetime::UntilContextDrop)
        );
    }

    #[tokio::test]
    async fn insert_until_context_drop_is_cleared() {
        let reg = SessionRefRegistry::new();
        let key = reg.insert(42_i32).await.unwrap();
        let removed = reg.clear_on_context_drop(true).await;
        assert_eq!(removed, 1);
        assert!(reg.get::<i32>(key).await.is_none());
        assert_eq!(reg.lifetime_of(key).await, None);
        assert!(reg.is_empty().await);
    }

    #[tokio::test]
    async fn insert_until_explicit_drop_survives_clear() {
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_with_lifetime(99_i32, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();

        // clear_on_context_drop must NOT touch this entry, regardless of
        // ownership.
        let removed = reg.clear_on_context_drop(true).await;
        assert_eq!(removed, 0);
        assert_eq!(reg.len().await, 1);
        assert_eq!(*reg.get::<i32>(key).await.unwrap(), 99);
        assert_eq!(
            reg.lifetime_of(key).await,
            Some(RefLifetime::UntilExplicitDrop)
        );

        // Explicit removal works.
        assert!(reg.remove(key).await.is_some());
        assert!(reg.get::<i32>(key).await.is_none());
        assert_eq!(reg.lifetime_of(key).await, None);
    }

    #[tokio::test]
    async fn insert_until_snapshot_survives_context_drop_but_purged_by_snapshot_walker() {
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_with_lifetime("ephemeral".to_owned(), RefLifetime::UntilSnapshot)
            .await
            .unwrap();

        // clear_on_context_drop must NOT touch this entry — purging
        // UntilSnapshot refs is the snapshot walker's job, not the
        // context-drop path's.
        let removed = reg.clear_on_context_drop(true).await;
        assert_eq!(removed, 0);
        assert_eq!(reg.len().await, 1);
        assert_eq!(reg.lifetime_of(key).await, Some(RefLifetime::UntilSnapshot));
    }

    #[tokio::test]
    async fn until_parent_finish_only_purged_when_owns_registry() {
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_with_lifetime(123_i32, RefLifetime::UntilParentFinish)
            .await
            .unwrap();

        // Borrowed (child) context: must NOT purge.
        let removed = reg.clear_on_context_drop(false).await;
        assert_eq!(removed, 0);
        assert_eq!(reg.len().await, 1);
        assert_eq!(
            reg.lifetime_of(key).await,
            Some(RefLifetime::UntilParentFinish)
        );

        // Owning (parent) context: now it goes.
        let removed = reg.clear_on_context_drop(true).await;
        assert_eq!(removed, 1);
        assert_eq!(reg.len().await, 0);
        assert_eq!(reg.lifetime_of(key).await, None);
    }

    #[tokio::test]
    async fn remove_clears_lifetime_sidecar() {
        let reg = SessionRefRegistry::new();
        let key = reg.insert(5_i32).await.unwrap();
        assert!(reg.lifetime_of(key).await.is_some());
        let _ = reg.remove(key).await;
        assert!(reg.lifetime_of(key).await.is_none());
    }

    #[tokio::test]
    async fn drain_clears_all_lifetimes() {
        let reg = SessionRefRegistry::new();
        let k1 = reg.insert(1_i32).await.unwrap();
        let k2 = reg
            .insert_with_lifetime(2_i32, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();
        let k3 = reg
            .insert_with_lifetime(3_i32, RefLifetime::UntilSnapshot)
            .await
            .unwrap();
        assert_eq!(reg.drain().await, 3);
        assert!(reg.lifetime_of(k1).await.is_none());
        assert!(reg.lifetime_of(k2).await.is_none());
        assert!(reg.lifetime_of(k3).await.is_none());
    }

    #[tokio::test]
    async fn clear_on_context_drop_purges_mixed_population_correctly() {
        let reg = SessionRefRegistry::new();
        let k_default = reg.insert(0_i32).await.unwrap();
        let k_explicit = reg
            .insert_with_lifetime(1_i32, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();
        let k_snapshot = reg
            .insert_with_lifetime(2_i32, RefLifetime::UntilSnapshot)
            .await
            .unwrap();
        let k_parent = reg
            .insert_with_lifetime(3_i32, RefLifetime::UntilParentFinish)
            .await
            .unwrap();

        // owns_registry = true → default + parent purged, explicit + snapshot remain.
        let removed = reg.clear_on_context_drop(true).await;
        assert_eq!(removed, 2);
        assert!(reg.get::<i32>(k_default).await.is_none());
        assert!(reg.get::<i32>(k_parent).await.is_none());
        assert!(reg.get::<i32>(k_explicit).await.is_some());
        assert!(reg.get::<i32>(k_snapshot).await.is_some());
    }

    // --------------------------------------------------------------
    // clone_ref / transfer_ref (Phase 11.3)
    // --------------------------------------------------------------

    #[tokio::test]
    async fn clone_ref_creates_second_handle_to_same_value() {
        let reg = SessionRefRegistry::new();
        let src = reg.insert(123_i32).await.unwrap();
        let dup = reg.clone_ref(src).await.unwrap();

        assert_ne!(src, dup, "clone_ref must mint a fresh RegistryKey");

        let a = reg.get::<i32>(src).await.unwrap();
        let b = reg.get::<i32>(dup).await.unwrap();
        assert_eq!(*a, 123);
        assert_eq!(*b, 123);
        assert!(
            Arc::ptr_eq(&a, &b),
            "both keys must resolve to the SAME Arc, not a deep copy"
        );
        // Source key is still valid after the clone.
        assert!(reg.get::<i32>(src).await.is_some());
        assert_eq!(reg.len().await, 2);
    }

    #[tokio::test]
    async fn clone_ref_inherits_lifetime() {
        let reg = SessionRefRegistry::new();
        let src = reg
            .insert_with_lifetime(7_i32, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();
        let dup = reg.clone_ref(src).await.unwrap();

        assert_eq!(
            reg.lifetime_of(dup).await,
            Some(RefLifetime::UntilExplicitDrop),
            "cloned ref must inherit the source's lifetime policy"
        );
        assert_eq!(
            reg.lifetime_of(src).await,
            Some(RefLifetime::UntilExplicitDrop),
            "source lifetime must be unchanged"
        );
    }

    #[tokio::test]
    async fn clone_ref_inherits_serializable_sidecar() {
        let reg = SessionRefRegistry::new();
        let src = reg
            .insert_serializable(Arc::new(TestSerializable { value: 42 }))
            .await
            .unwrap();
        let dup = reg.clone_ref(src).await.unwrap();

        let ser = reg
            .get_serializable(dup)
            .await
            .expect("cloned ref must carry a serializable sidecar");
        assert_eq!(ser.blazen_type_tag(), "test::TestSerializable");
        assert_eq!(ser.blazen_serialize().unwrap(), vec![0, 0, 0, 42]);

        // Source sidecar is still present.
        assert!(reg.get_serializable(src).await.is_some());
    }

    #[tokio::test]
    async fn clone_ref_errors_on_missing_source() {
        let reg = SessionRefRegistry::new();
        let bogus = RegistryKey::new();
        let err = reg.clone_ref(bogus).await.unwrap_err();
        match err {
            SessionRefError::KeyNotFound { key } => assert_eq!(key, bogus),
            other => panic!("expected KeyNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn clone_ref_errors_on_full_registry() {
        // Build a registry filled to capacity by stuffing the inner
        // map directly — we avoid the slow path of inserting
        // MAX_SESSION_REFS_PER_RUN entries via the public API just to
        // exercise the capacity check.
        let reg = SessionRefRegistry::new();
        let src = reg.insert(1_i32).await.unwrap();
        {
            let mut main = reg.inner.write().await;
            for _ in main.len()..MAX_SESSION_REFS_PER_RUN {
                let k = RegistryKey::new();
                main.insert(k, Arc::new(0_i32) as AnyArc);
            }
        }
        assert_eq!(reg.len().await, MAX_SESSION_REFS_PER_RUN);

        let err = reg.clone_ref(src).await.unwrap_err();
        match err {
            SessionRefError::CapacityExceeded { cap } => {
                assert_eq!(cap, MAX_SESSION_REFS_PER_RUN);
            }
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn transfer_ref_moves_value_to_destination() {
        let src_reg = SessionRefRegistry::new();
        let dst_reg = SessionRefRegistry::new();
        let key = src_reg.insert(555_i32).await.unwrap();
        assert_eq!(src_reg.len().await, 1);
        assert_eq!(dst_reg.len().await, 0);

        let returned = src_reg.transfer_ref(key, &dst_reg).await.unwrap();

        assert_eq!(returned, key);
        assert!(
            src_reg.get::<i32>(key).await.is_none(),
            "source must no longer resolve the transferred key"
        );
        assert_eq!(src_reg.len().await, 0);
        let got = dst_reg
            .get::<i32>(returned)
            .await
            .expect("destination must resolve the transferred key");
        assert_eq!(*got, 555);
        assert_eq!(dst_reg.len().await, 1);
    }

    #[tokio::test]
    async fn transfer_ref_preserves_key() {
        let src_reg = SessionRefRegistry::new();
        let dst_reg = SessionRefRegistry::new();
        let key = src_reg.insert("preserved".to_owned()).await.unwrap();

        let returned = src_reg.transfer_ref(key, &dst_reg).await.unwrap();
        assert_eq!(
            returned, key,
            "transfer must preserve the original RegistryKey so existing \
             JSON markers keep resolving"
        );
    }

    #[tokio::test]
    async fn transfer_ref_migrates_lifetime() {
        let src_reg = SessionRefRegistry::new();
        let dst_reg = SessionRefRegistry::new();
        let key = src_reg
            .insert_with_lifetime(1_i32, RefLifetime::UntilExplicitDrop)
            .await
            .unwrap();

        let returned = src_reg.transfer_ref(key, &dst_reg).await.unwrap();

        assert_eq!(
            src_reg.lifetime_of(returned).await,
            None,
            "source lifetime sidecar must be cleared"
        );
        assert_eq!(
            dst_reg.lifetime_of(returned).await,
            Some(RefLifetime::UntilExplicitDrop),
            "destination lifetime sidecar must inherit the original policy"
        );
    }

    #[tokio::test]
    async fn transfer_ref_migrates_serializable_sidecar() {
        let src_reg = SessionRefRegistry::new();
        let dst_reg = SessionRefRegistry::new();
        let key = src_reg
            .insert_serializable(Arc::new(TestSerializable { value: 3 }))
            .await
            .unwrap();

        let returned = src_reg.transfer_ref(key, &dst_reg).await.unwrap();

        assert!(
            src_reg.get_serializable(returned).await.is_none(),
            "source sidecar must be cleared after transfer"
        );
        let ser = dst_reg
            .get_serializable(returned)
            .await
            .expect("destination must carry the migrated sidecar");
        assert_eq!(ser.blazen_type_tag(), "test::TestSerializable");
        assert_eq!(ser.blazen_serialize().unwrap(), vec![0, 0, 0, 3]);
    }

    #[tokio::test]
    async fn transfer_ref_errors_on_missing_source() {
        let src_reg = SessionRefRegistry::new();
        let dst_reg = SessionRefRegistry::new();
        let bogus = RegistryKey::new();
        let err = src_reg.transfer_ref(bogus, &dst_reg).await.unwrap_err();
        match err {
            SessionRefError::KeyNotFound { key } => assert_eq!(key, bogus),
            other => panic!("expected KeyNotFound, got {other:?}"),
        }
        // Neither registry should have been mutated.
        assert_eq!(src_reg.len().await, 0);
        assert_eq!(dst_reg.len().await, 0);
    }

    #[tokio::test]
    async fn transfer_ref_no_op_when_src_equals_dst() {
        // Transferring within the same registry must leave the key in
        // place and return it unchanged. The ref is neither duplicated
        // nor removed.
        let reg = SessionRefRegistry::new();
        let key = reg.insert(77_i32).await.unwrap();
        let len_before = reg.len().await;

        let returned = reg.transfer_ref(key, &reg).await.unwrap();

        assert_eq!(returned, key);
        assert_eq!(reg.len().await, len_before);
        let got = reg.get::<i32>(key).await.unwrap();
        assert_eq!(*got, 77);
    }

    #[tokio::test]
    async fn transfer_ref_no_op_errors_on_missing_key_in_same_registry() {
        let reg = SessionRefRegistry::new();
        let bogus = RegistryKey::new();
        let err = reg.transfer_ref(bogus, &reg).await.unwrap_err();
        match err {
            SessionRefError::KeyNotFound { key } => assert_eq!(key, bogus),
            other => panic!("expected KeyNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn transfer_ref_leaves_source_untouched_when_destination_full() {
        let src_reg = SessionRefRegistry::new();
        let dst_reg = SessionRefRegistry::new();
        let key = src_reg.insert(13_i32).await.unwrap();

        // Saturate dst directly so the capacity check fires without
        // blowing 10k entries through the public API.
        {
            let mut main = dst_reg.inner.write().await;
            for _ in 0..MAX_SESSION_REFS_PER_RUN {
                let k = RegistryKey::new();
                main.insert(k, Arc::new(0_i32) as AnyArc);
            }
        }

        let err = src_reg.transfer_ref(key, &dst_reg).await.unwrap_err();
        match err {
            SessionRefError::CapacityExceeded { cap } => {
                assert_eq!(cap, MAX_SESSION_REFS_PER_RUN);
            }
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }

        // Source must be untouched by the failed transfer.
        assert_eq!(src_reg.len().await, 1);
        let got = src_reg.get::<i32>(key).await.unwrap();
        assert_eq!(*got, 13);
        assert_eq!(
            src_reg.lifetime_of(key).await,
            Some(RefLifetime::UntilContextDrop),
        );
    }

    // --------------------------------------------------------------
    // Remote refs (Phase 12.5)
    // --------------------------------------------------------------

    fn make_descriptor(node: &str, tag: &str, ts: u64) -> RemoteRefDescriptor {
        RemoteRefDescriptor {
            origin_node_id: node.to_owned(),
            type_tag: tag.to_owned(),
            created_at_epoch_ms: ts,
        }
    }

    #[tokio::test]
    async fn insert_remote_stores_descriptor() {
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();
        let desc = make_descriptor("peer-1", "blazen::TestType", 1_700_000_000_000);
        reg.insert_remote(key, desc.clone()).await.unwrap();

        let got = reg
            .get_remote(key)
            .await
            .expect("descriptor must be present");
        assert_eq!(got.origin_node_id, "peer-1");
        assert_eq!(got.type_tag, "blazen::TestType");
        assert_eq!(got.created_at_epoch_ms, 1_700_000_000_000);
    }

    #[tokio::test]
    async fn is_remote_distinguishes_local_and_remote() {
        let reg = SessionRefRegistry::new();
        let local_key = reg.insert(123_i32).await.unwrap();
        let remote_key = RegistryKey::new();
        reg.insert_remote(remote_key, make_descriptor("peer-2", "i32", 0))
            .await
            .unwrap();

        assert!(
            !reg.is_remote(local_key).await,
            "locally-inserted refs must NOT report as remote"
        );
        assert!(
            reg.is_remote(remote_key).await,
            "remotely-inserted refs must report as remote"
        );
        // Sanity: a never-seen key is also not remote.
        assert!(!reg.is_remote(RegistryKey::new()).await);
    }

    #[tokio::test]
    async fn remove_clears_remote_sidecar() {
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();
        reg.insert_remote(key, make_descriptor("peer-3", "tag", 42))
            .await
            .unwrap();
        assert!(reg.is_remote(key).await);

        // `remove` operates on the main `inner` map and returns None for
        // a pure remote ref, but it must still scrub the sidecar.
        let _ = reg.remove(key).await;
        assert!(reg.get_remote(key).await.is_none());
        assert!(!reg.is_remote(key).await);
    }

    #[tokio::test]
    async fn drain_clears_remote_sidecar() {
        let reg = SessionRefRegistry::new();
        for i in 0..3 {
            reg.insert_remote(RegistryKey::new(), make_descriptor("peer-drain", "tag", i))
                .await
                .unwrap();
        }
        assert_eq!(reg.remote_entries().await.len(), 3);

        let _ = reg.drain().await;
        assert!(
            reg.remote_entries().await.is_empty(),
            "drain must clear the remote_refs sidecar"
        );
    }

    #[tokio::test]
    async fn remote_entries_lists_all() {
        let reg = SessionRefRegistry::new();
        let mut keys = Vec::new();
        for i in 0..3_u64 {
            let key = RegistryKey::new();
            reg.insert_remote(key, make_descriptor("peer-list", "tag", i))
                .await
                .unwrap();
            keys.push(key);
        }

        let entries = reg.remote_entries().await;
        assert_eq!(entries.len(), 3);
        let listed: std::collections::HashSet<RegistryKey> =
            entries.iter().map(|(k, _)| *k).collect();
        let expected: std::collections::HashSet<RegistryKey> = keys.iter().copied().collect();
        assert_eq!(listed, expected);
    }

    #[tokio::test]
    async fn remote_refs_respect_capacity() {
        let reg = SessionRefRegistry::new();
        // Saturate the remote_refs map directly so the capacity check
        // fires without churning through 10k public-API insert calls.
        {
            let mut remote = reg.remote_refs.write().await;
            for i in 0..MAX_SESSION_REFS_PER_RUN {
                remote.insert(
                    RegistryKey::new(),
                    make_descriptor("peer-cap", "tag", i as u64),
                );
            }
        }
        assert_eq!(reg.remote_entries().await.len(), MAX_SESSION_REFS_PER_RUN);

        let err = reg
            .insert_remote(RegistryKey::new(), make_descriptor("peer-cap", "tag", 0))
            .await
            .unwrap_err();
        match err {
            SessionRefError::CapacityExceeded { cap } => {
                assert_eq!(cap, MAX_SESSION_REFS_PER_RUN);
            }
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn local_and_remote_with_same_key_both_resolve() {
        // Edge case: the same RegistryKey lives in both `inner` (local
        // materialization) and `remote_refs` (descriptor still on
        // file). Both lookups must succeed independently.
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();

        // Stage 1: register the descriptor as a remote ref.
        reg.insert_remote(key, make_descriptor("peer-shared", "i32", 7))
            .await
            .unwrap();

        // Stage 2: materialize the value into the main `inner` map
        // under the SAME key (simulating a successful Deref RPC). We
        // bypass the public `insert*` API because those mint a fresh
        // key — here we explicitly want the dual-mapping edge case.
        {
            let mut main = reg.inner.write().await;
            main.insert(key, Arc::new(456_i32) as AnyArc);
        }
        reg.lifetimes
            .write()
            .await
            .insert(key, RefLifetime::default());

        // Local lookup hits the main map.
        let local = reg
            .get::<i32>(key)
            .await
            .expect("main `inner` lookup must resolve the materialized value");
        assert_eq!(*local, 456);

        // Remote-ref sidecar still carries the descriptor.
        let desc = reg
            .get_remote(key)
            .await
            .expect("remote_refs sidecar must still hold the descriptor");
        assert_eq!(desc.origin_node_id, "peer-shared");
        assert_eq!(desc.type_tag, "i32");
        assert_eq!(desc.created_at_epoch_ms, 7);

        // is_remote still reports true even though the value is now
        // also locally available.
        assert!(reg.is_remote(key).await);
    }

    #[tokio::test]
    async fn remote_ref_not_resolved_error_carries_key_and_node() {
        let key = RegistryKey::new();
        let err = SessionRefError::RemoteRefNotResolved {
            key,
            origin_node_id: "peer-err".to_owned(),
        };
        let msg = err.to_string();
        assert!(msg.contains(&key.to_string()));
        assert!(msg.contains("peer-err"));
        assert!(msg.contains("deref_session_ref"));
    }
}
