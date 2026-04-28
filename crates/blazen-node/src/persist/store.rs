//! Subclassable base class for checkpoint storage backends.

use napi::bindgen_prelude::Result;
use napi_derive::napi;

use super::checkpoint::JsWorkflowCheckpoint;

/// Base class for custom checkpoint storage backends.
///
/// Extend and override all methods to implement a custom backend
/// (e.g. `PostgreSQL`, `SQLite`, `S3`).
///
/// ```javascript
/// class PostgresStore extends CheckpointStore {
///     async save(checkpoint) { /* ... */ }
///     async load(runId) { /* ... */ }
///     async list() { /* ... */ }
///     async delete(runId) { /* ... */ }
/// }
/// ```
#[napi(js_name = "CheckpointStore")]
pub struct JsCheckpointStore {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsCheckpointStore {
    /// Create a new checkpoint store base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Persist a checkpoint. If a checkpoint with the same run ID already
    /// exists it is overwritten. Subclasses **must** override this method.
    #[napi]
    pub async fn save(&self, _checkpoint: &JsWorkflowCheckpoint) -> Result<()> {
        Err(napi::Error::from_reason("subclass must override save()"))
    }

    /// Load a checkpoint by its run ID. Returns `null` if no checkpoint
    /// exists for the given run ID. Subclasses **must** override this method.
    #[napi]
    pub async fn load(&self, _run_id: String) -> Result<Option<JsWorkflowCheckpoint>> {
        Err(napi::Error::from_reason("subclass must override load()"))
    }

    /// List all stored checkpoints, ordered by timestamp descending
    /// (most recent first). Subclasses **must** override this method.
    #[napi]
    pub async fn list(&self) -> Result<Vec<JsWorkflowCheckpoint>> {
        Err(napi::Error::from_reason("subclass must override list()"))
    }

    /// Delete the checkpoint for the given run ID. Subclasses **must**
    /// override this method.
    #[napi]
    pub async fn delete(&self, _run_id: String) -> Result<()> {
        Err(napi::Error::from_reason("subclass must override delete()"))
    }
}
