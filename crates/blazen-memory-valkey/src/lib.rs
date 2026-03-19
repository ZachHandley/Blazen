//! # blazen-memory-valkey
//!
//! A Valkey/Redis backend for [`blazen-memory`].
//!
//! This crate provides [`ValkeyBackend`], an implementation of
//! [`MemoryBackend`](blazen_memory::MemoryBackend) that persists entries in
//! Valkey (or Redis-compatible) using the following key layout:
//!
//! | Key pattern                  | Type   | Contents                        |
//! |------------------------------|--------|---------------------------------|
//! | `{prefix}entry:{id}`         | STRING | JSON-serialized `StoredEntry`   |
//! | `{prefix}bands:{band_value}` | SET    | Entry IDs sharing this LSH band |
//! | `{prefix}ids`                | SET    | All entry IDs                   |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use blazen_memory_valkey::ValkeyBackend;
//! use blazen_memory::{Memory, MemoryStore, MemoryEntry};
//!
//! # async fn example() -> blazen_memory::error::Result<()> {
//! let backend = ValkeyBackend::new("redis://localhost:6379")
//!     .expect("failed to create Valkey client");
//!
//! let memory = Memory::local(backend);
//! memory.add(vec![MemoryEntry::new("hello world")]).await?;
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use redis::AsyncCommands;
use tracing::instrument;

use blazen_memory::error::{MemoryError, Result};
use blazen_memory::store::MemoryBackend;
use blazen_memory::types::StoredEntry;

/// Default key prefix used for namespacing in Valkey/Redis.
const DEFAULT_PREFIX: &str = "blazen:memory:";

/// A [`MemoryBackend`] implementation backed by Valkey (or any Redis-compatible server).
///
/// Uses [`redis::aio::MultiplexedConnection`] for async I/O and pipelines
/// multi-key operations for efficiency.
pub struct ValkeyBackend {
    client: redis::Client,
    prefix: String,
}

impl ValkeyBackend {
    /// Create a new `ValkeyBackend` connected to the given Redis/Valkey URL.
    ///
    /// The URL should be in the standard Redis format, e.g. `redis://localhost:6379`
    /// or `rediss://...` for TLS.
    ///
    /// # Errors
    ///
    /// Returns a [`redis::RedisError`] if the URL is invalid or the client cannot
    /// be constructed.
    pub fn new(url: &str) -> std::result::Result<Self, redis::RedisError> {
        let client = redis::Client::open(url)?;
        Ok(Self {
            client,
            prefix: DEFAULT_PREFIX.to_owned(),
        })
    }

    /// Override the default key prefix (`blazen:memory:`).
    ///
    /// This is useful when running multiple logical stores against the same
    /// Valkey instance.
    #[must_use]
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_owned();
        self
    }

    /// Obtain a multiplexed async connection from the underlying client.
    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey connection error: {e}")))
    }

    // -- key helpers ----------------------------------------------------------

    fn entry_key(&self, id: &str) -> String {
        format!("{}entry:{}", self.prefix, id)
    }

    fn band_key(&self, band: &str) -> String {
        format!("{}bands:{}", self.prefix, band)
    }

    fn ids_key(&self) -> String {
        format!("{}ids", self.prefix)
    }
}

#[async_trait]
impl MemoryBackend for ValkeyBackend {
    #[instrument(skip(self, entry), fields(id = %entry.id))]
    async fn put(&self, entry: StoredEntry) -> Result<()> {
        let mut conn = self.conn().await?;

        let json = serde_json::to_string(&entry)?;
        let entry_key = self.entry_key(&entry.id);
        let ids_key = self.ids_key();

        // Pipeline: SET entry, SADD to ids set, SADD to each band set.
        let mut pipe = redis::pipe();
        pipe.atomic();
        pipe.set(&entry_key, &json).ignore();
        pipe.sadd(&ids_key, &entry.id).ignore();
        for band in &entry.bands {
            pipe.sadd(self.band_key(band), &entry.id).ignore();
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey put error: {e}")))?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn get(&self, id: &str) -> Result<Option<StoredEntry>> {
        let mut conn = self.conn().await?;
        let entry_key = self.entry_key(id);

        let raw: Option<String> = conn
            .get(&entry_key)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey get error: {e}")))?;

        match raw {
            Some(json) => {
                let entry: StoredEntry = serde_json::from_str(&json)?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    #[instrument(skip(self))]
    async fn delete(&self, id: &str) -> Result<bool> {
        let mut conn = self.conn().await?;

        // Fetch the entry first so we know which band sets to clean up.
        let entry_key = self.entry_key(id);
        let raw: Option<String> = conn
            .get(&entry_key)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey delete/get error: {e}")))?;

        let entry = match raw {
            Some(json) => serde_json::from_str::<StoredEntry>(&json)?,
            None => return Ok(false),
        };

        // Pipeline: DEL entry key, SREM from ids set, SREM from each band set.
        let mut pipe = redis::pipe();
        pipe.atomic();
        pipe.del(&entry_key).ignore();
        pipe.srem(self.ids_key(), id).ignore();
        for band in &entry.bands {
            pipe.srem(self.band_key(band), id).ignore();
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey delete error: {e}")))?;

        Ok(true)
    }

    #[instrument(skip(self))]
    async fn list(&self) -> Result<Vec<StoredEntry>> {
        let mut conn = self.conn().await?;

        let ids: Vec<String> = conn
            .smembers(self.ids_key())
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey list/smembers error: {e}")))?;

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // MGET all entry keys at once.
        let keys: Vec<String> = ids.iter().map(|id| self.entry_key(id)).collect();
        let values: Vec<Option<String>> = conn
            .mget(&keys)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey list/mget error: {e}")))?;

        let mut entries = Vec::with_capacity(values.len());
        for raw in values.into_iter().flatten() {
            let entry: StoredEntry = serde_json::from_str(&raw)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    #[instrument(skip(self))]
    async fn len(&self) -> Result<usize> {
        let mut conn = self.conn().await?;

        let count: usize = conn
            .scard(self.ids_key())
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey len error: {e}")))?;

        Ok(count)
    }

    #[instrument(skip(self, bands))]
    async fn search_by_bands(&self, bands: &[String], limit: usize) -> Result<Vec<StoredEntry>> {
        if bands.is_empty() {
            return Ok(Vec::new());
        }

        let mut conn = self.conn().await?;

        let band_keys: Vec<String> = bands.iter().map(|b| self.band_key(b)).collect();

        // SUNION across all matching band sets to get candidate IDs.
        let candidate_ids: Vec<String> = redis::cmd("SUNION")
            .arg(&band_keys)
            .query_async(&mut conn)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey search/sunion error: {e}")))?;

        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Cap candidates to limit before fetching full entries.
        let capped: &[String] = if candidate_ids.len() > limit {
            &candidate_ids[..limit]
        } else {
            &candidate_ids
        };

        let keys: Vec<String> = capped.iter().map(|id| self.entry_key(id)).collect();
        let values: Vec<Option<String>> = conn
            .mget(&keys)
            .await
            .map_err(|e| MemoryError::Backend(format!("Valkey search/mget error: {e}")))?;

        let mut entries = Vec::with_capacity(values.len());
        for raw in values.into_iter().flatten() {
            let entry: StoredEntry = serde_json::from_str(&raw)?;
            entries.push(entry);
        }

        Ok(entries)
    }
}

// ---------------------------------------------------------------------------
// Tests -- gated behind #[ignore] since they require a running Valkey/Redis
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn valkey_url() -> String {
        std::env::var("VALKEY_URL").unwrap_or_else(|_| "redis://localhost:6379".into())
    }

    /// Generate a unique prefix per test run to avoid collisions.
    fn test_prefix() -> String {
        let id = uuid::Uuid::new_v4();
        format!("blazen:test:{id}:")
    }

    fn make_entry(id: &str, text: &str, bands: Vec<String>) -> StoredEntry {
        StoredEntry {
            id: id.to_owned(),
            text: text.to_owned(),
            elid: None,
            simhash_hex: None,
            text_simhash: 0,
            bands,
            metadata: serde_json::Value::Null,
        }
    }

    async fn backend() -> ValkeyBackend {
        ValkeyBackend::new(&valkey_url())
            .expect("failed to create ValkeyBackend")
            .with_prefix(&test_prefix())
    }

    /// Clean up all keys under the given prefix.
    async fn cleanup(backend: &ValkeyBackend) {
        // Best-effort cleanup: delete all known IDs and their associated keys.
        if let Ok(entries) = backend.list().await {
            for entry in &entries {
                let _ = backend.delete(&entry.id).await;
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_put_get() {
        let b = backend().await;
        let entry = make_entry("e1", "hello world", vec!["b0".into(), "b1".into()]);

        b.put(entry).await.unwrap();

        let got = b.get("e1").await.unwrap();
        assert!(got.is_some());
        let got = got.unwrap();
        assert_eq!(got.id, "e1");
        assert_eq!(got.text, "hello world");
        assert_eq!(got.bands, vec!["b0", "b1"]);

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_get_missing() {
        let b = backend().await;
        let got = b.get("nonexistent").await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_delete() {
        let b = backend().await;
        b.put(make_entry("d1", "delete me", vec!["x".into()]))
            .await
            .unwrap();

        assert!(b.delete("d1").await.unwrap());
        assert!(!b.delete("d1").await.unwrap());
        assert!(b.get("d1").await.unwrap().is_none());

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_put_overwrites() {
        let b = backend().await;
        b.put(make_entry("o1", "first", vec!["a".into()]))
            .await
            .unwrap();
        b.put(make_entry("o1", "second", vec!["a".into()]))
            .await
            .unwrap();

        assert_eq!(b.len().await.unwrap(), 1);
        let got = b.get("o1").await.unwrap().unwrap();
        assert_eq!(got.text, "second");

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_list() {
        let b = backend().await;
        b.put(make_entry("l1", "alpha", vec![])).await.unwrap();
        b.put(make_entry("l2", "beta", vec![])).await.unwrap();

        let all = b.list().await.unwrap();
        assert_eq!(all.len(), 2);

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_len() {
        let b = backend().await;
        assert_eq!(b.len().await.unwrap(), 0);

        b.put(make_entry("n1", "one", vec![])).await.unwrap();
        assert_eq!(b.len().await.unwrap(), 1);

        b.put(make_entry("n2", "two", vec![])).await.unwrap();
        assert_eq!(b.len().await.unwrap(), 2);

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_search_by_bands() {
        let b = backend().await;

        b.put(make_entry("s1", "alpha", vec!["aaa".into(), "bbb".into()]))
            .await
            .unwrap();
        b.put(make_entry("s2", "beta", vec!["ccc".into(), "ddd".into()]))
            .await
            .unwrap();
        b.put(make_entry("s3", "gamma", vec!["eee".into(), "fff".into()]))
            .await
            .unwrap();

        // Search for bands that match s1 and s2 but not s3.
        let results = b
            .search_by_bands(&["aaa".into(), "ddd".into()], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        let ids: Vec<&str> = results.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"s1"));
        assert!(ids.contains(&"s2"));

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_search_by_bands_no_match() {
        let b = backend().await;
        b.put(make_entry("x1", "solo", vec!["aaa".into()]))
            .await
            .unwrap();

        let results = b.search_by_bands(&["zzz".into()], 10).await.unwrap();
        assert!(results.is_empty());

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_search_by_bands_respects_limit() {
        let b = backend().await;

        for i in 0..10 {
            b.put(make_entry(
                &format!("lim{i}"),
                &format!("entry {i}"),
                vec!["shared".into()],
            ))
            .await
            .unwrap();
        }

        let results = b.search_by_bands(&["shared".into()], 3).await.unwrap();
        assert!(results.len() <= 3);

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_delete_cleans_bands() {
        let b = backend().await;
        b.put(make_entry("bc1", "band cleanup", vec!["band_x".into()]))
            .await
            .unwrap();

        b.delete("bc1").await.unwrap();

        // The band set should no longer return this id.
        let results = b.search_by_bands(&["band_x".into()], 10).await.unwrap();
        assert!(results.is_empty());

        cleanup(&b).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_valkey_custom_prefix() {
        let b = ValkeyBackend::new(&valkey_url())
            .unwrap()
            .with_prefix(&format!("custom:{}:", uuid::Uuid::new_v4()));

        b.put(make_entry("p1", "prefixed", vec![])).await.unwrap();
        assert_eq!(b.len().await.unwrap(), 1);

        cleanup(&b).await;
    }
}
