//! Python bindings for the `blazen-persist` crate.
//!
//! `blazen-py` always pulls `blazen-persist` with both the `redb` and
//! `valkey` features enabled (see `Cargo.toml`), so the concrete store
//! wrappers are unconditionally available here.

pub mod checkpoint;
pub mod error;
pub mod redb_store;
pub mod store;
pub mod valkey_store;

pub use checkpoint::{PyPersistedEvent, PyWorkflowCheckpoint};
pub use error::{PersistException, persist_err, register as register_exceptions};
pub use redb_store::PyRedbCheckpointStore;
pub use store::{PyCheckpointStore, PyHostCheckpointStore};
pub use valkey_store::PyValkeyCheckpointStore;
