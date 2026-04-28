//! Redb-backed checkpoint store binding.

use std::sync::Arc;

use napi::bindgen_prelude::Result;
use napi_derive::napi;
use uuid::Uuid;

use blazen_persist::{CheckpointStore, RedbCheckpointStore};

use super::checkpoint::JsWorkflowCheckpoint;
use crate::error::{persist_error_to_napi, to_napi_error};

/// An embedded, file-backed checkpoint store powered by `redb`.
///
/// ```javascript
/// const store = await RedbCheckpointStore.create("./workflow.db");
/// await store.save(checkpoint);
/// ```
#[napi(js_name = "RedbCheckpointStore")]
pub struct JsRedbCheckpointStore {
    inner: Arc<RedbCheckpointStore>,
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsRedbCheckpointStore {
    /// Create a redb-backed checkpoint store at the given file path.
    ///
    /// The file is created if it does not exist.
    #[napi(factory)]
    pub fn create(path: String) -> Result<Self> {
        let store = RedbCheckpointStore::new(&path).map_err(persist_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Create an in-memory redb-backed checkpoint store (useful for tests).
    #[napi(factory, js_name = "inMemory")]
    pub fn in_memory() -> Result<Self> {
        let store = RedbCheckpointStore::in_memory().map_err(persist_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Persist a checkpoint. If a checkpoint with the same run ID already
    /// exists it is overwritten.
    #[napi]
    pub async fn save(&self, checkpoint: &JsWorkflowCheckpoint) -> Result<()> {
        self.inner
            .save(checkpoint.inner_ref())
            .await
            .map_err(persist_error_to_napi)
    }

    /// Load a checkpoint by its run ID. Returns `null` if not found.
    #[napi]
    pub async fn load(&self, run_id: String) -> Result<Option<JsWorkflowCheckpoint>> {
        let id = Uuid::parse_str(&run_id).map_err(to_napi_error)?;
        let cp = self.inner.load(&id).await.map_err(persist_error_to_napi)?;
        Ok(cp.map(JsWorkflowCheckpoint::from_inner))
    }

    /// List all stored checkpoints, ordered by timestamp descending
    /// (most recent first).
    #[napi]
    pub async fn list(&self) -> Result<Vec<JsWorkflowCheckpoint>> {
        let list = self.inner.list().await.map_err(persist_error_to_napi)?;
        Ok(list
            .into_iter()
            .map(JsWorkflowCheckpoint::from_inner)
            .collect())
    }

    /// Delete the checkpoint for the given run ID.
    #[napi]
    pub async fn delete(&self, run_id: String) -> Result<()> {
        let id = Uuid::parse_str(&run_id).map_err(to_napi_error)?;
        self.inner.delete(&id).await.map_err(persist_error_to_napi)
    }
}
