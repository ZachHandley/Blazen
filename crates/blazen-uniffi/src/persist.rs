//! Checkpoint-store surface for the UniFFI bindings.
//!
//! Exposes [`blazen_persist::CheckpointStore`] backends to foreign callers
//! (Go / Swift / Kotlin / Ruby) via a single opaque [`CheckpointStore`]
//! handle. Two factories are provided:
//!
//! - [`new_redb_checkpoint_store`] — embedded, pure-Rust ACID file store.
//! - [`new_valkey_checkpoint_store`] — Redis/ValKey server-backed store.
//!
//! Both factories return `Arc<CheckpointStore>` and are sync from the
//! foreign side; the valkey constructor blocks on the shared Tokio runtime
//! to establish the initial connection so callers don't have to engage
//! their host-language async machinery just to build a store.
//!
//! ## Wire-format records
//!
//! [`WorkflowCheckpoint`] and [`PersistedEvent`] are
//! [`uniffi::Record`]s — value types crossed across the FFI by value. The
//! mapping is documented per-field on each type; in short:
//!
//! - `run_id` becomes a UUID string (`"550e8400-e29b-41d4-a716-446655440000"`).
//! - `timestamp` becomes Unix-epoch milliseconds (`u64`).
//! - `state` / `metadata` (Rust `HashMap<String, serde_json::Value>`) become
//!   JSON-encoded strings (`state_json` / `metadata_json`) so foreign
//!   callers can parse them with their language's stdlib JSON decoder.
//!
//! ## Naming deviation from upstream
//!
//! The task spec for this module asked for a `workflow_id` /
//! `step_name` / `state_json` / `created_at_ms` shape, but the upstream
//! [`blazen_persist::WorkflowCheckpoint`] type has no `step_name` field
//! and uses `run_id` + `workflow_name` + `timestamp` + `state`. Per the
//! task's explicit "use the real names" instruction this module mirrors
//! the upstream fields directly: `workflow_name`, `run_id` (as a UUID
//! string), `timestamp_ms`, `state_json`, `pending_events`,
//! `metadata_json`.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::TimeZone;
use uuid::Uuid;

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

use blazen_persist::CheckpointStore as CoreCheckpointStore;
use blazen_persist::valkey::ValkeyCheckpointStore as CoreValkeyCheckpointStore;
use blazen_persist::{
    PersistError, RedbCheckpointStore as CoreRedbCheckpointStore,
    SerializedEvent as CoreSerializedEvent, WorkflowCheckpoint as CoreWorkflowCheckpoint,
};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a [`blazen_persist::PersistError`] into the canonical
/// [`BlazenError::Persist`] variant.
///
/// `errors.rs` doesn't carry a blanket `From<PersistError>` impl (it would
/// pull `blazen-persist` into the error module's compile graph for every
/// build), so this module's call sites use this helper inline via
/// `.map_err(persist_err)`.
fn persist_err(err: PersistError) -> BlazenError {
    BlazenError::Persist {
        message: err.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// A serialized representation of a queued event captured in a checkpoint.
///
/// Mirrors [`blazen_persist::SerializedEvent`]. The `data_json` field is
/// the JSON-encoded payload of the original event (the upstream type
/// stores a `serde_json::Value`, which is not a UniFFI-supported wire
/// type — JSON strings cross cleanly instead).
#[derive(Debug, Clone, uniffi::Record)]
pub struct PersistedEvent {
    /// The event type identifier (e.g. `"blazen::StartEvent"`).
    pub event_type: String,
    /// The event payload, JSON-encoded. Decode with the host language's
    /// standard JSON library on the foreign side.
    pub data_json: String,
}

impl TryFrom<PersistedEvent> for CoreSerializedEvent {
    type Error = BlazenError;

    fn try_from(value: PersistedEvent) -> Result<Self, Self::Error> {
        let data: serde_json::Value = serde_json::from_str(&value.data_json)?;
        Ok(Self {
            event_type: value.event_type,
            data,
        })
    }
}

impl TryFrom<CoreSerializedEvent> for PersistedEvent {
    type Error = BlazenError;

    fn try_from(value: CoreSerializedEvent) -> Result<Self, Self::Error> {
        let data_json = serde_json::to_string(&value.data)?;
        Ok(Self {
            event_type: value.event_type,
            data_json,
        })
    }
}

/// A snapshot of a workflow's state at a point in time.
///
/// Foreign-language wrappers typically marshal these to/from native types
/// (Go structs, Swift `Codable`, Kotlin `@Serializable`, Ruby hashes) just
/// outside this module's boundary.
///
/// See the module docs for the upstream field-name mapping.
#[derive(Debug, Clone, uniffi::Record)]
pub struct WorkflowCheckpoint {
    /// The name of the workflow that produced this checkpoint.
    pub workflow_name: String,
    /// Unique identifier for this workflow run, formatted as a UUID
    /// string (`"550e8400-e29b-41d4-a716-446655440000"`).
    pub run_id: String,
    /// When the checkpoint was created, as Unix-epoch milliseconds.
    pub timestamp_ms: u64,
    /// Serialized context state, as a JSON object encoded into a string
    /// (`"{\"counter\":42}"`). Decode with the host language's JSON library.
    pub state_json: String,
    /// Events in the queue at checkpoint time.
    pub pending_events: Vec<PersistedEvent>,
    /// Arbitrary metadata attached to this checkpoint, as a JSON object
    /// encoded into a string. Decode with the host language's JSON library.
    pub metadata_json: String,
}

impl TryFrom<WorkflowCheckpoint> for CoreWorkflowCheckpoint {
    type Error = BlazenError;

    fn try_from(value: WorkflowCheckpoint) -> Result<Self, Self::Error> {
        let run_id = if value.run_id.is_empty() {
            Uuid::new_v4()
        } else {
            Uuid::parse_str(&value.run_id).map_err(|e| BlazenError::Validation {
                message: format!("invalid run_id UUID: {e}"),
            })?
        };

        let timestamp = chrono::Utc
            .timestamp_millis_opt(i64::try_from(value.timestamp_ms).map_err(|_| {
                BlazenError::Validation {
                    message: "timestamp_ms exceeds i64 range".into(),
                }
            })?)
            .single()
            .ok_or_else(|| BlazenError::Validation {
                message: format!("invalid timestamp_ms value: {}", value.timestamp_ms),
            })?;

        let state: HashMap<String, serde_json::Value> = if value.state_json.is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&value.state_json).map_err(|e| BlazenError::Validation {
                message: format!("state_json must be a JSON object: {e}"),
            })?
        };

        let metadata: HashMap<String, serde_json::Value> = if value.metadata_json.is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&value.metadata_json).map_err(|e| BlazenError::Validation {
                message: format!("metadata_json must be a JSON object: {e}"),
            })?
        };

        let pending_events = value
            .pending_events
            .into_iter()
            .map(CoreSerializedEvent::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            workflow_name: value.workflow_name,
            run_id,
            timestamp,
            state,
            pending_events,
            metadata,
        })
    }
}

impl TryFrom<CoreWorkflowCheckpoint> for WorkflowCheckpoint {
    type Error = BlazenError;

    fn try_from(value: CoreWorkflowCheckpoint) -> Result<Self, Self::Error> {
        let timestamp_ms = u64::try_from(value.timestamp.timestamp_millis()).map_err(|_| {
            BlazenError::Internal {
                message: "checkpoint timestamp is before the Unix epoch".into(),
            }
        })?;
        let state_json = serde_json::to_string(&value.state)?;
        let metadata_json = serde_json::to_string(&value.metadata)?;
        let pending_events = value
            .pending_events
            .into_iter()
            .map(PersistedEvent::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            workflow_name: value.workflow_name,
            run_id: value.run_id.to_string(),
            timestamp_ms,
            state_json,
            pending_events,
            metadata_json,
        })
    }
}

// ---------------------------------------------------------------------------
// CheckpointStore opaque handle
// ---------------------------------------------------------------------------

/// A workflow-checkpoint store handle.
///
/// Wraps any [`blazen_persist::CheckpointStore`] implementation behind a
/// uniform FFI surface. Construct via:
///
/// - [`new_redb_checkpoint_store`] for an embedded file-backed store.
/// - [`new_valkey_checkpoint_store`] for a Redis/ValKey server-backed store.
///
/// Each method has both an async variant (recommended on Swift / Kotlin /
/// modern Ruby fibers) and a `_blocking` variant (handy for Go `main`
/// functions and quick scripts).
#[derive(uniffi::Object)]
pub struct CheckpointStore {
    inner: Arc<dyn CoreCheckpointStore>,
}

impl CheckpointStore {
    /// Wrap a concrete `blazen_persist` store in the FFI handle.
    ///
    /// Reserved for in-crate factories; not exposed across the FFI.
    pub(crate) fn from_arc(inner: Arc<dyn CoreCheckpointStore>) -> Arc<Self> {
        Arc::new(Self { inner })
    }

    fn parse_run_id(run_id: &str) -> BlazenResult<Uuid> {
        Uuid::parse_str(run_id).map_err(|e| BlazenError::Validation {
            message: format!("invalid run_id UUID: {e}"),
        })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl CheckpointStore {
    /// Persist a checkpoint, overwriting any existing entry with the same
    /// `run_id`. Async on Swift / Kotlin; blocking-with-suspension on Go.
    pub async fn save(self: Arc<Self>, checkpoint: WorkflowCheckpoint) -> BlazenResult<()> {
        let core_cp: CoreWorkflowCheckpoint = checkpoint.try_into()?;
        self.inner.save(&core_cp).await.map_err(persist_err)
    }

    /// Load a checkpoint by its run id (UUID string). Returns `None` when no
    /// checkpoint exists for the given id.
    pub async fn load(self: Arc<Self>, run_id: String) -> BlazenResult<Option<WorkflowCheckpoint>> {
        let id = Self::parse_run_id(&run_id)?;
        let loaded = self.inner.load(&id).await.map_err(persist_err)?;
        loaded.map(WorkflowCheckpoint::try_from).transpose()
    }

    /// Delete the checkpoint for the given run id (UUID string). Succeeds
    /// even when no checkpoint exists for the id (the underlying backends
    /// treat delete-of-missing as a no-op).
    pub async fn delete(self: Arc<Self>, run_id: String) -> BlazenResult<()> {
        let id = Self::parse_run_id(&run_id)?;
        self.inner.delete(&id).await.map_err(persist_err)
    }

    /// List all stored checkpoints, ordered by timestamp descending (most
    /// recent first).
    pub async fn list(self: Arc<Self>) -> BlazenResult<Vec<WorkflowCheckpoint>> {
        let items = self.inner.list().await.map_err(persist_err)?;
        items
            .into_iter()
            .map(WorkflowCheckpoint::try_from)
            .collect()
    }

    /// List all stored run ids (as UUID strings), ordered by timestamp
    /// descending (most recent first).
    ///
    /// Cheaper than [`list`](Self::list) when callers only need to
    /// enumerate ids — but note that the underlying backend still loads
    /// each checkpoint to read its timestamp for ordering, so this is a
    /// convenience wrapper rather than a true index scan.
    pub async fn list_run_ids(self: Arc<Self>) -> BlazenResult<Vec<String>> {
        let items = self.inner.list().await.map_err(persist_err)?;
        Ok(items.into_iter().map(|c| c.run_id.to_string()).collect())
    }
}

#[uniffi::export]
impl CheckpointStore {
    /// Synchronous variant of [`save`](Self::save).
    pub fn save_blocking(self: Arc<Self>, checkpoint: WorkflowCheckpoint) -> BlazenResult<()> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.save(checkpoint).await })
    }

    /// Synchronous variant of [`load`](Self::load).
    pub fn load_blocking(
        self: Arc<Self>,
        run_id: String,
    ) -> BlazenResult<Option<WorkflowCheckpoint>> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.load(run_id).await })
    }

    /// Synchronous variant of [`delete`](Self::delete).
    pub fn delete_blocking(self: Arc<Self>, run_id: String) -> BlazenResult<()> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.delete(run_id).await })
    }

    /// Synchronous variant of [`list`](Self::list).
    pub fn list_blocking(self: Arc<Self>) -> BlazenResult<Vec<WorkflowCheckpoint>> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.list().await })
    }

    /// Synchronous variant of [`list_run_ids`](Self::list_run_ids).
    pub fn list_run_ids_blocking(self: Arc<Self>) -> BlazenResult<Vec<String>> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.list_run_ids().await })
    }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/// Build an embedded redb-backed checkpoint store rooted at `path`.
///
/// The database file is created if it does not exist. Re-opening an
/// existing file is safe and preserves prior checkpoints. The returned
/// handle is cheap to clone and safe to share across threads / tasks.
#[uniffi::export]
pub fn new_redb_checkpoint_store(path: String) -> BlazenResult<Arc<CheckpointStore>> {
    let store = CoreRedbCheckpointStore::new(&path).map_err(persist_err)?;
    let inner: Arc<dyn CoreCheckpointStore> = Arc::new(store);
    Ok(CheckpointStore::from_arc(inner))
}

/// Build a Redis/ValKey-backed checkpoint store connected to `url`.
///
/// `url` is in the form `redis://host:port/db` (or `rediss://` for TLS).
/// When `ttl_seconds` is provided every saved checkpoint will auto-expire
/// after that many seconds — useful for transient workflows where old
/// checkpoints should not accumulate indefinitely.
///
/// The initial connection is established eagerly on the shared Tokio
/// runtime; subsequent reconnections are handled automatically by the
/// underlying connection manager.
///
/// # Naming deviation from spec
///
/// The task spec named the second argument `namespace`, but
/// [`blazen_persist::valkey::ValkeyCheckpointStore`] has no namespace
/// concept — instead it supports an optional per-key TTL via
/// [`with_ttl`](blazen_persist::valkey::ValkeyCheckpointStore::with_ttl).
/// This factory exposes that real option.
#[uniffi::export]
pub fn new_valkey_checkpoint_store(
    url: String,
    ttl_seconds: Option<u64>,
) -> BlazenResult<Arc<CheckpointStore>> {
    let store = runtime().block_on(async move {
        match ttl_seconds {
            Some(ttl) => CoreValkeyCheckpointStore::with_ttl(&url, ttl).await,
            None => CoreValkeyCheckpointStore::new(&url).await,
        }
    });
    let store = store.map_err(persist_err)?;
    let inner: Arc<dyn CoreCheckpointStore> = Arc::new(store);
    Ok(CheckpointStore::from_arc(inner))
}
