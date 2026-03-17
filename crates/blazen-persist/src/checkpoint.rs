//! Checkpoint storage for workflow state persistence.
//!
//! [`WorkflowCheckpoint`] captures a snapshot of a workflow's state at a point
//! in time. The [`CheckpointStore`] trait abstracts the storage backend, with
//! [`RedbCheckpointStore`] providing an embedded key-value implementation
//! backed by [`redb`] and [`ValkeyCheckpointStore`](crate::valkey::ValkeyCheckpointStore)
//! providing a Redis/ValKey-backed implementation.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::PersistError;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A snapshot of workflow state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowCheckpoint {
    /// The name of the workflow that produced this checkpoint.
    pub workflow_name: String,
    /// Unique identifier for this workflow run.
    pub run_id: Uuid,
    /// When the checkpoint was created.
    pub timestamp: DateTime<Utc>,
    /// Serialized context state (key -> JSON value).
    pub state: HashMap<String, serde_json::Value>,
    /// Events in the queue at checkpoint time.
    pub pending_events: Vec<SerializedEvent>,
    /// Arbitrary metadata attached to this checkpoint.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A serialized representation of an event for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEvent {
    /// The event type identifier (e.g. `"blazen::StartEvent"`).
    pub event_type: String,
    /// The event data as a JSON value.
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// CheckpointStore trait
// ---------------------------------------------------------------------------

/// Trait for checkpoint storage backends.
///
/// Implementors provide durable (or in-memory) storage for workflow
/// checkpoints, enabling pause/resume and crash recovery.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Persist a checkpoint. If a checkpoint with the same `run_id` already
    /// exists it is overwritten.
    async fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<(), PersistError>;

    /// Load a checkpoint by its run ID. Returns `None` if no checkpoint
    /// exists for the given run ID.
    async fn load(&self, run_id: &Uuid) -> Result<Option<WorkflowCheckpoint>, PersistError>;

    /// List all stored checkpoints, ordered by timestamp descending
    /// (most recent first).
    async fn list(&self) -> Result<Vec<WorkflowCheckpoint>, PersistError>;

    /// Delete the checkpoint for the given run ID.
    async fn delete(&self, run_id: &Uuid) -> Result<(), PersistError>;
}

// ---------------------------------------------------------------------------
// RedbCheckpointStore
// ---------------------------------------------------------------------------

#[cfg(feature = "redb")]
mod redb_backend {
    use std::path::Path;

    use async_trait::async_trait;
    use redb::ReadableDatabase;
    use redb::ReadableTable;
    use uuid::Uuid;

    use super::{CheckpointStore, WorkflowCheckpoint};
    use crate::error::PersistError;

    /// Table definition: `run_id` bytes (16) -> serialized checkpoint bytes.
    const CHECKPOINTS: redb::TableDefinition<&[u8], &[u8]> =
        redb::TableDefinition::new("checkpoints");

    /// Redb-backed checkpoint store.
    ///
    /// Uses a single table where the key is the 16-byte UUID of the run and the
    /// value is the MessagePack-serialized [`WorkflowCheckpoint`]. Legacy
    /// JSON-encoded entries are transparently decoded on read for backward
    /// compatibility.
    pub struct RedbCheckpointStore {
        db: redb::Database,
    }

    impl RedbCheckpointStore {
        /// Create a new store backed by a file at `path`.
        ///
        /// The file is created if it does not exist.
        ///
        /// # Errors
        ///
        /// Returns [`PersistError::Database`] if the database cannot be opened.
        pub fn new(path: impl AsRef<Path>) -> Result<Self, PersistError> {
            let db = redb::Database::create(path)?;
            // Ensure the table exists by running an initial write transaction.
            let write_txn = db.begin_write()?;
            {
                // Opening the table in a write transaction creates it if absent.
                let _table = write_txn.open_table(CHECKPOINTS)?;
            }
            write_txn.commit()?;
            Ok(Self { db })
        }

        /// Create an in-memory store (useful for tests).
        ///
        /// # Errors
        ///
        /// Returns [`PersistError::Database`] if the in-memory backend cannot be
        /// initialised.
        pub fn in_memory() -> Result<Self, PersistError> {
            let backend = redb::backends::InMemoryBackend::new();
            let db = redb::Database::builder()
                .create_with_backend(backend)
                .map_err(|e| PersistError::Database(e.to_string()))?;
            let write_txn = db.begin_write()?;
            {
                let _table = write_txn.open_table(CHECKPOINTS)?;
            }
            write_txn.commit()?;
            Ok(Self { db })
        }
    }

    impl std::fmt::Debug for RedbCheckpointStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RedbCheckpointStore")
                .finish_non_exhaustive()
        }
    }

    #[async_trait]
    impl CheckpointStore for RedbCheckpointStore {
        async fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<(), PersistError> {
            let key = checkpoint.run_id.as_bytes().to_vec();
            let value = rmp_serde::to_vec_named(checkpoint)?;
            let write_txn = self.db.begin_write()?;
            {
                let mut table = write_txn.open_table(CHECKPOINTS)?;
                table.insert(key.as_slice(), value.as_slice())?;
            }
            write_txn.commit()?;
            Ok(())
        }

        async fn load(&self, run_id: &Uuid) -> Result<Option<WorkflowCheckpoint>, PersistError> {
            let key = run_id.as_bytes().to_vec();
            let read_txn = self.db.begin_read()?;
            let table = read_txn.open_table(CHECKPOINTS)?;
            match table.get(key.as_slice())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    // Try MessagePack first (new format), fall back to JSON (legacy).
                    let checkpoint: WorkflowCheckpoint = rmp_serde::from_slice(bytes)
                        .or_else(|_| serde_json::from_slice(bytes).map_err(PersistError::from))?;
                    Ok(Some(checkpoint))
                }
                None => Ok(None),
            }
        }

        async fn list(&self) -> Result<Vec<WorkflowCheckpoint>, PersistError> {
            let read_txn = self.db.begin_read()?;
            let table = read_txn.open_table(CHECKPOINTS)?;
            let mut checkpoints = Vec::new();
            let iter = table.iter()?;
            for entry in iter {
                let (_key_guard, value_guard) = entry?;
                let bytes: &[u8] = value_guard.value();
                // Try MessagePack first (new format), fall back to JSON (legacy).
                let checkpoint: WorkflowCheckpoint = rmp_serde::from_slice(bytes)
                    .or_else(|_| serde_json::from_slice(bytes).map_err(PersistError::from))?;
                checkpoints.push(checkpoint);
            }
            // Sort by timestamp descending (most recent first).
            checkpoints.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            Ok(checkpoints)
        }

        async fn delete(&self, run_id: &Uuid) -> Result<(), PersistError> {
            let key = run_id.as_bytes().to_vec();
            let write_txn = self.db.begin_write()?;
            {
                let mut table = write_txn.open_table(CHECKPOINTS)?;
                table.remove(key.as_slice())?;
            }
            write_txn.commit()?;
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use chrono::Utc;

        use super::*;
        use crate::checkpoint::SerializedEvent;

        fn sample_checkpoint(name: &str) -> WorkflowCheckpoint {
            WorkflowCheckpoint {
                workflow_name: name.to_owned(),
                run_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                state: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("counter".to_owned(), serde_json::json!(42));
                    m
                },
                pending_events: vec![SerializedEvent {
                    event_type: "blazen::StartEvent".to_owned(),
                    data: serde_json::json!({"input": "hello"}),
                }],
                metadata: std::collections::HashMap::new(),
            }
        }

        #[tokio::test]
        async fn save_and_load() {
            let store = RedbCheckpointStore::in_memory().unwrap();
            let cp = sample_checkpoint("test_workflow");
            let run_id = cp.run_id;

            store.save(&cp).await.unwrap();
            let loaded = store.load(&run_id).await.unwrap().unwrap();
            assert_eq!(loaded.workflow_name, "test_workflow");
            assert_eq!(loaded.run_id, run_id);
            assert_eq!(loaded.state["counter"], serde_json::json!(42));
        }

        #[tokio::test]
        async fn load_missing_returns_none() {
            let store = RedbCheckpointStore::in_memory().unwrap();
            let result = store.load(&Uuid::new_v4()).await.unwrap();
            assert!(result.is_none());
        }

        #[tokio::test]
        async fn list_returns_all_sorted() {
            let store = RedbCheckpointStore::in_memory().unwrap();

            let mut cp1 = sample_checkpoint("wf_a");
            cp1.timestamp = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc);

            let mut cp2 = sample_checkpoint("wf_b");
            cp2.timestamp = chrono::DateTime::parse_from_rfc3339("2025-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc);

            store.save(&cp1).await.unwrap();
            store.save(&cp2).await.unwrap();

            let list = store.list().await.unwrap();
            assert_eq!(list.len(), 2);
            // Most recent first.
            assert_eq!(list[0].workflow_name, "wf_b");
            assert_eq!(list[1].workflow_name, "wf_a");
        }

        #[tokio::test]
        async fn delete_removes_checkpoint() {
            let store = RedbCheckpointStore::in_memory().unwrap();
            let cp = sample_checkpoint("delete_me");
            let run_id = cp.run_id;

            store.save(&cp).await.unwrap();
            assert!(store.load(&run_id).await.unwrap().is_some());

            store.delete(&run_id).await.unwrap();
            assert!(store.load(&run_id).await.unwrap().is_none());
        }

        #[tokio::test]
        async fn save_overwrites_existing() {
            let store = RedbCheckpointStore::in_memory().unwrap();
            let mut cp = sample_checkpoint("overwrite");
            let run_id = cp.run_id;

            store.save(&cp).await.unwrap();

            cp.state.insert("counter".to_owned(), serde_json::json!(99));
            store.save(&cp).await.unwrap();

            let loaded = store.load(&run_id).await.unwrap().unwrap();
            assert_eq!(loaded.state["counter"], serde_json::json!(99));
        }
    }
}

#[cfg(feature = "redb")]
pub use redb_backend::RedbCheckpointStore;
