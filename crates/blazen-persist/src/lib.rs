//! # `Blazen` Persistence
//!
//! Provides checkpoint storage for workflow state, enabling pause/resume and
//! crash recovery.
//!
//! ## Backends
//!
//! Two storage backends are available behind feature flags:
//!
//! | Feature | Backend | Description |
//! |---------|---------|-------------|
//! | `redb` (default) | [`RedbCheckpointStore`] | Embedded, pure-Rust ACID key-value store |
//! | `valkey` | [`valkey::ValkeyCheckpointStore`] | Redis/ValKey server-backed store |
//!
//! ## Quick start (redb)
//!
//! ```rust,no_run
//! use blazen_persist::{CheckpointStore, RedbCheckpointStore, WorkflowCheckpoint};
//!
//! # async fn example() -> Result<(), blazen_persist::PersistError> {
//! let store = RedbCheckpointStore::new("workflow.db")?;
//!
//! // Save a checkpoint
//! # let checkpoint = WorkflowCheckpoint {
//! #     workflow_name: "demo".into(),
//! #     run_id: uuid::Uuid::new_v4(),
//! #     timestamp: chrono::Utc::now(),
//! #     state: Default::default(),
//! #     pending_events: vec![],
//! #     metadata: Default::default(),
//! # };
//! store.save(&checkpoint).await?;
//!
//! // Load it back
//! let loaded = store.load(&checkpoint.run_id).await?;
//! # Ok(())
//! # }
//! ```

pub mod checkpoint;
pub mod error;

#[cfg(feature = "valkey")]
pub mod valkey;

// Always-available exports (trait + data types + error).
pub use checkpoint::{CheckpointStore, SerializedEvent, WorkflowCheckpoint};
pub use error::PersistError;

// Backend-specific re-exports.
#[cfg(feature = "redb")]
pub use checkpoint::RedbCheckpointStore;
