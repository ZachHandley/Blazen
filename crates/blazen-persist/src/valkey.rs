//! ValKey/Redis-backed checkpoint storage.
//!
//! [`ValkeyCheckpointStore`] implements the [`CheckpointStore`](crate::CheckpointStore)
//! trait using a Redis/ValKey server as the persistence backend. It uses the
//! `redis` crate's [`ConnectionManager`](redis::aio::ConnectionManager) for
//! automatic reconnection and multiplexed async I/O.
//!
//! Checkpoints are stored as MessagePack-encoded bytes under keys of the form
//! `blazen:checkpoint:{run_id}`. Legacy JSON-encoded entries are transparently
//! decoded on read for backward compatibility.

use async_trait::async_trait;
use redis::AsyncCommands;
use uuid::Uuid;

use crate::checkpoint::{CheckpointStore, WorkflowCheckpoint};
use crate::error::PersistError;

/// Key prefix used for all checkpoint entries in Redis/ValKey.
const KEY_PREFIX: &str = "blazen:checkpoint:";

/// A [`CheckpointStore`] backed by a Redis/ValKey server.
///
/// Uses [`ConnectionManager`](redis::aio::ConnectionManager) under the hood,
/// which provides automatic reconnection and is cheaply cloneable -- making it
/// safe to share across tasks.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example() -> Result<(), blazen_persist::PersistError> {
/// use blazen_persist::valkey::ValkeyCheckpointStore;
///
/// // Connect to a local ValKey/Redis instance.
/// let store = ValkeyCheckpointStore::new("redis://127.0.0.1/").await?;
///
/// // Or with a TTL so checkpoints auto-expire after 24 hours.
/// let store = ValkeyCheckpointStore::with_ttl("redis://127.0.0.1/", 86400).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ValkeyCheckpointStore {
    conn: redis::aio::ConnectionManager,
    /// Optional TTL in seconds. When set, keys are stored with SETEX.
    ttl_seconds: Option<u64>,
}

impl ValkeyCheckpointStore {
    /// Create a new store connected to the given Redis/ValKey URL.
    ///
    /// The URL should be in the form `redis://host:port/db` (or
    /// `rediss://` for TLS). A connection is established immediately;
    /// subsequent reconnections are handled automatically.
    ///
    /// # Errors
    ///
    /// Returns [`PersistError::Redis`] if the initial connection fails.
    pub async fn new(url: &str) -> Result<Self, PersistError> {
        let client = redis::Client::open(url)
            .map_err(|e| PersistError::Redis(format!("failed to parse URL: {e}")))?;
        let conn = redis::aio::ConnectionManager::new(client).await?;
        Ok(Self {
            conn,
            ttl_seconds: None,
        })
    }

    /// Create a new store with an automatic key expiration (TTL).
    ///
    /// Checkpoints saved through this store will expire after
    /// `ttl_seconds` seconds. This is useful for transient workflows
    /// where old checkpoints should not accumulate indefinitely.
    ///
    /// # Errors
    ///
    /// Returns [`PersistError::Redis`] if the initial connection fails.
    pub async fn with_ttl(url: &str, ttl_seconds: u64) -> Result<Self, PersistError> {
        let client = redis::Client::open(url)
            .map_err(|e| PersistError::Redis(format!("failed to parse URL: {e}")))?;
        let conn = redis::aio::ConnectionManager::new(client).await?;
        Ok(Self {
            conn,
            ttl_seconds: Some(ttl_seconds),
        })
    }

    /// Build the Redis key for a given run ID.
    fn key(run_id: &Uuid) -> String {
        format!("{KEY_PREFIX}{run_id}")
    }
}

impl std::fmt::Debug for ValkeyCheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValkeyCheckpointStore")
            .field("ttl_seconds", &self.ttl_seconds)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl CheckpointStore for ValkeyCheckpointStore {
    async fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<(), PersistError> {
        let key = Self::key(&checkpoint.run_id);
        let value: Vec<u8> = rmp_serde::to_vec_named(checkpoint)?;
        let mut conn = self.conn.clone();

        if let Some(ttl) = self.ttl_seconds {
            let () = conn.set_ex(&key, &value, ttl).await?;
        } else {
            let () = conn.set(&key, &value).await?;
        }

        Ok(())
    }

    async fn load(&self, run_id: &Uuid) -> Result<Option<WorkflowCheckpoint>, PersistError> {
        let key = Self::key(run_id);
        let mut conn = self.conn.clone();

        let value: Option<Vec<u8>> = conn.get(&key).await?;

        match value {
            Some(bytes) => {
                // Try MessagePack first (new format), fall back to JSON (legacy).
                let checkpoint: WorkflowCheckpoint = rmp_serde::from_slice(&bytes)
                    .or_else(|_| serde_json::from_slice(&bytes).map_err(PersistError::from))?;
                Ok(Some(checkpoint))
            }
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<WorkflowCheckpoint>, PersistError> {
        let mut conn = self.conn.clone();
        let pattern = format!("{KEY_PREFIX}*");

        // Use SCAN for production safety (does not block the server like KEYS).
        let mut cursor: u64 = 0;
        let mut all_keys: Vec<String> = Vec::new();

        loop {
            let result: (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut conn)
                .await?;

            let (next_cursor, keys) = result;
            all_keys.extend(keys);
            cursor = next_cursor;

            if cursor == 0 {
                break;
            }
        }

        // GET each key and deserialize.
        let mut checkpoints = Vec::with_capacity(all_keys.len());
        for key in &all_keys {
            let value: Option<Vec<u8>> = conn.get(key).await?;
            if let Some(bytes) = value {
                // Try MessagePack first (new format), fall back to JSON (legacy).
                match rmp_serde::from_slice::<WorkflowCheckpoint>(&bytes)
                    .or_else(|_| serde_json::from_slice(&bytes))
                {
                    Ok(cp) => checkpoints.push(cp),
                    Err(e) => {
                        // Log and skip malformed entries rather than failing
                        // the entire list operation.
                        tracing::warn!(
                            key = %key,
                            error = %e,
                            "skipping malformed checkpoint entry"
                        );
                    }
                }
            }
        }

        // Sort by timestamp descending (most recent first).
        checkpoints.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(checkpoints)
    }

    async fn delete(&self, run_id: &Uuid) -> Result<(), PersistError> {
        let key = Self::key(run_id);
        let mut conn = self.conn.clone();
        let () = conn.del(&key).await?;
        Ok(())
    }
}
