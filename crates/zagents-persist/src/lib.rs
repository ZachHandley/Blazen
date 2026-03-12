//! # `ZAgents` Persistence
//!
//! Provides embedded checkpoint storage for workflow state, enabling
//! pause/resume and crash recovery. The default backend uses [`redb`]
//! -- a pure-Rust, embedded, ACID key-value store.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use zagents_persist::{CheckpointStore, RedbCheckpointStore, WorkflowCheckpoint};
//!
//! # async fn example() -> Result<(), zagents_persist::PersistError> {
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

pub use checkpoint::{
    CheckpointStore, RedbCheckpointStore, SerializedEvent, WorkflowCheckpoint,
};
pub use error::PersistError;
