//! In-memory backend backed by a `HashMap` behind a `RwLock`.
//!
//! Suitable for testing, prototyping, and short-lived processes.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::store::MemoryBackend;
use crate::types::StoredEntry;

/// A simple in-memory backend.
///
/// Data lives only as long as the process; nothing is persisted to disk.
pub struct InMemoryBackend {
    entries: Arc<RwLock<HashMap<String, StoredEntry>>>,
}

impl InMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryBackend for InMemoryBackend {
    async fn put(&self, entry: StoredEntry) -> Result<()> {
        let mut map = self.entries.write().await;
        map.insert(entry.id.clone(), entry);
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<StoredEntry>> {
        let map = self.entries.read().await;
        Ok(map.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let mut map = self.entries.write().await;
        Ok(map.remove(id).is_some())
    }

    async fn list(&self) -> Result<Vec<StoredEntry>> {
        let map = self.entries.read().await;
        Ok(map.values().cloned().collect())
    }

    async fn len(&self) -> Result<usize> {
        let map = self.entries.read().await;
        Ok(map.len())
    }

    async fn search_by_bands(&self, bands: &[String], limit: usize) -> Result<Vec<StoredEntry>> {
        let map = self.entries.read().await;
        let mut results = Vec::new();

        for entry in map.values() {
            if results.len() >= limit {
                break;
            }
            // Check if any of the query bands match any of the entry's bands.
            let matches = entry
                .bands
                .iter()
                .zip(bands.iter())
                .any(|(entry_band, query_band)| entry_band == query_band);
            if matches {
                results.push(entry.clone());
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, text: &str) -> StoredEntry {
        StoredEntry {
            id: id.to_string(),
            text: text.to_string(),
            elid: None,
            simhash_hex: None,
            text_simhash: elid::simhash(text),
            bands: vec!["band0".into(), "band1".into()],
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn test_put_and_get() {
        let backend = InMemoryBackend::new();

        backend.put(make_entry("e1", "hello")).await.unwrap();

        let got = backend.get("e1").await.unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().text, "hello");
    }

    #[tokio::test]
    async fn test_get_missing() {
        let backend = InMemoryBackend::new();
        let got = backend.get("nope").await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let backend = InMemoryBackend::new();
        backend.put(make_entry("e1", "hello")).await.unwrap();

        assert!(backend.delete("e1").await.unwrap());
        assert!(!backend.delete("e1").await.unwrap());
        assert!(backend.get("e1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list() {
        let backend = InMemoryBackend::new();
        backend.put(make_entry("a", "alpha")).await.unwrap();
        backend.put(make_entry("b", "beta")).await.unwrap();

        let all = backend.list().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_len() {
        let backend = InMemoryBackend::new();
        assert_eq!(backend.len().await.unwrap(), 0);

        backend.put(make_entry("a", "alpha")).await.unwrap();
        assert_eq!(backend.len().await.unwrap(), 1);

        backend.put(make_entry("b", "beta")).await.unwrap();
        assert_eq!(backend.len().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_put_overwrites() {
        let backend = InMemoryBackend::new();
        backend.put(make_entry("a", "first")).await.unwrap();
        backend.put(make_entry("a", "second")).await.unwrap();

        assert_eq!(backend.len().await.unwrap(), 1);
        let got = backend.get("a").await.unwrap().unwrap();
        assert_eq!(got.text, "second");
    }

    #[tokio::test]
    async fn test_search_by_bands_matching() {
        let backend = InMemoryBackend::new();

        let mut entry = make_entry("a", "alpha");
        entry.bands = vec!["aaa".into(), "bbb".into()];
        backend.put(entry).await.unwrap();

        let mut entry = make_entry("b", "beta");
        entry.bands = vec!["ccc".into(), "ddd".into()];
        backend.put(entry).await.unwrap();

        // Query bands: position 0 = "aaa" matches entry "a", position 1 = "ddd" matches entry "b".
        let results = backend
            .search_by_bands(&["aaa".into(), "ddd".into()], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_by_bands_no_match() {
        let backend = InMemoryBackend::new();

        let mut entry = make_entry("a", "alpha");
        entry.bands = vec!["aaa".into(), "bbb".into()];
        backend.put(entry).await.unwrap();

        let results = backend
            .search_by_bands(&["zzz".into(), "yyy".into()], 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_by_bands_respects_limit() {
        let backend = InMemoryBackend::new();

        for i in 0..10 {
            let mut entry = make_entry(&format!("e{i}"), &format!("entry {i}"));
            entry.bands = vec!["shared".into()];
            backend.put(entry).await.unwrap();
        }

        let results = backend
            .search_by_bands(&["shared".into()], 3)
            .await
            .unwrap();
        assert!(results.len() <= 3);
    }
}
