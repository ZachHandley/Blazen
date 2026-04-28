//! ValKey/Redis-backed checkpoint store binding.

use std::sync::Arc;

use napi::bindgen_prelude::Result;
use napi_derive::napi;
use uuid::Uuid;

use blazen_persist::CheckpointStore;
use blazen_persist::valkey::ValkeyCheckpointStore;

use super::checkpoint::JsWorkflowCheckpoint;
use crate::error::{persist_error_to_napi, to_napi_error};

/// A ValKey/Redis-backed checkpoint store.
///
/// ```javascript
/// const store = await ValkeyCheckpointStore.create("redis://127.0.0.1/");
/// await store.save(checkpoint);
/// ```
#[napi(js_name = "ValkeyCheckpointStore")]
pub struct JsValkeyCheckpointStore {
    inner: Arc<ValkeyCheckpointStore>,
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsValkeyCheckpointStore {
    /// Create a checkpoint store connected to the given Redis/ValKey URL.
    ///
    /// The URL should be in standard Redis format, e.g.
    /// `redis://localhost:6379` (or `rediss://` for TLS).
    #[napi(factory)]
    pub async fn create(url: String) -> Result<Self> {
        let store = ValkeyCheckpointStore::new(&url)
            .await
            .map_err(persist_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Create a checkpoint store with an automatic key expiration (TTL in
    /// seconds). Keys are stored with `SETEX` and expire after `ttlSeconds`.
    #[napi(factory, js_name = "withTtl")]
    pub async fn with_ttl(url: String, ttl_seconds: u32) -> Result<Self> {
        let store = ValkeyCheckpointStore::with_ttl(&url, u64::from(ttl_seconds))
            .await
            .map_err(persist_error_to_napi)?;
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
