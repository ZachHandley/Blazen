//! JSONL file-backed memory backend.
//!
//! Loads all entries into RAM on construction, then appends new entries to the
//! file. Deletions are handled by rewriting the file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::debug;

use crate::error::{MemoryError, Result};
use crate::store::MemoryBackend;
use crate::types::StoredEntry;

/// A JSONL file-backed memory backend.
///
/// Each line in the file is a JSON-serialized [`StoredEntry`]. On startup
/// the file is loaded into RAM. Writes append to the file; deletions trigger
/// a full rewrite (amortised cost is acceptable for small-to-medium stores).
pub struct JsonlBackend {
    path: PathBuf,
    entries: Arc<RwLock<HashMap<String, StoredEntry>>>,
}

impl JsonlBackend {
    /// Create a new JSONL backend at the given path.
    ///
    /// If the file exists, its contents are loaded. Otherwise an empty store
    /// is created (the file is written on the first `put`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let entries = if path.exists() {
            Self::load_from_file(&path).await?
        } else {
            HashMap::new()
        };

        debug!(
            path = %path.display(),
            count = entries.len(),
            "JSONL backend initialized"
        );

        Ok(Self {
            path,
            entries: Arc::new(RwLock::new(entries)),
        })
    }

    /// Load entries from a JSONL file. Invalid lines are skipped with a warning.
    async fn load_from_file(path: &Path) -> Result<HashMap<String, StoredEntry>> {
        let content = tokio::fs::read_to_string(path).await?;
        let mut map = HashMap::new();

        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<StoredEntry>(line) {
                Ok(entry) => {
                    map.insert(entry.id.clone(), entry);
                }
                Err(e) => {
                    tracing::warn!(
                        line = line_no + 1,
                        error = %e,
                        "skipping invalid JSONL line"
                    );
                }
            }
        }

        Ok(map)
    }

    /// Append a single entry as one JSONL line.
    async fn append_entry(&self, entry: &StoredEntry) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let mut line = serde_json::to_string(entry)?;
        line.push('\n');

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;

        file.write_all(line.as_bytes()).await?;
        file.flush().await?;

        Ok(())
    }

    /// Rewrite the entire file from the in-memory map.
    async fn rewrite(&self) -> Result<()> {
        let map = self.entries.read().await;
        let mut content = String::new();

        for entry in map.values() {
            let line = serde_json::to_string(entry)
                .map_err(|e| MemoryError::Serialization(e.to_string()))?;
            content.push_str(&line);
            content.push('\n');
        }

        tokio::fs::write(&self.path, content.as_bytes()).await?;

        Ok(())
    }
}

#[async_trait]
impl MemoryBackend for JsonlBackend {
    async fn put(&self, entry: StoredEntry) -> Result<()> {
        // If updating an existing entry, rewrite the file; otherwise just append.
        let is_update = {
            let map = self.entries.read().await;
            map.contains_key(&entry.id)
        };

        {
            let mut map = self.entries.write().await;
            map.insert(entry.id.clone(), entry.clone());
        }

        if is_update {
            self.rewrite().await?;
        } else {
            self.append_entry(&entry).await?;
        }

        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<StoredEntry>> {
        let map = self.entries.read().await;
        Ok(map.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let existed = {
            let mut map = self.entries.write().await;
            map.remove(id).is_some()
        };

        if existed {
            self.rewrite().await?;
        }

        Ok(existed)
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
    use std::io::Write as _;

    fn make_entry(id: &str, text: &str) -> StoredEntry {
        StoredEntry {
            id: id.to_string(),
            text: text.to_string(),
            elid: None,
            simhash_hex: None,
            text_simhash: elid::simhash(text),
            bands: vec!["b0".into(), "b1".into()],
            metadata: serde_json::json!({"key": "value"}),
        }
    }

    #[tokio::test]
    async fn test_jsonl_persistence() {
        // Create a temp file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        // Write entries.
        {
            let backend = JsonlBackend::new(&path).await.unwrap();
            backend.put(make_entry("a", "alpha")).await.unwrap();
            backend.put(make_entry("b", "beta")).await.unwrap();
            assert_eq!(backend.len().await.unwrap(), 2);
        }

        // Reload from file.
        {
            let backend = JsonlBackend::new(&path).await.unwrap();
            assert_eq!(backend.len().await.unwrap(), 2);

            let a = backend.get("a").await.unwrap().unwrap();
            assert_eq!(a.text, "alpha");

            let b = backend.get("b").await.unwrap().unwrap();
            assert_eq!(b.text, "beta");
        }
    }

    #[tokio::test]
    async fn test_jsonl_delete_rewrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let backend = JsonlBackend::new(&path).await.unwrap();
        backend.put(make_entry("a", "alpha")).await.unwrap();
        backend.put(make_entry("b", "beta")).await.unwrap();

        backend.delete("a").await.unwrap();

        // Reload to verify deletion was persisted.
        let backend2 = JsonlBackend::new(&path).await.unwrap();
        assert_eq!(backend2.len().await.unwrap(), 1);
        assert!(backend2.get("a").await.unwrap().is_none());
        assert!(backend2.get("b").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_jsonl_update_rewrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let backend = JsonlBackend::new(&path).await.unwrap();
        backend.put(make_entry("a", "first")).await.unwrap();
        backend.put(make_entry("a", "second")).await.unwrap();

        // Reload.
        let backend2 = JsonlBackend::new(&path).await.unwrap();
        assert_eq!(backend2.len().await.unwrap(), 1);
        let a = backend2.get("a").await.unwrap().unwrap();
        assert_eq!(a.text, "second");
    }

    #[tokio::test]
    async fn test_jsonl_new_file_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brand_new.jsonl");

        assert!(!path.exists());

        let backend = JsonlBackend::new(&path).await.unwrap();
        assert_eq!(backend.len().await.unwrap(), 0);

        backend.put(make_entry("a", "alpha")).await.unwrap();
        assert!(path.exists());
    }

    #[tokio::test]
    async fn test_jsonl_skips_invalid_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.jsonl");

        // Manually write a file with a valid line, an invalid line, and another valid line.
        {
            let entry_a = make_entry("a", "alpha");
            let entry_b = make_entry("b", "beta");
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "{}", serde_json::to_string(&entry_a).unwrap()).unwrap();
            writeln!(f, "THIS IS NOT VALID JSON").unwrap();
            writeln!(f, "{}", serde_json::to_string(&entry_b).unwrap()).unwrap();
        }

        let backend = JsonlBackend::new(&path).await.unwrap();
        assert_eq!(backend.len().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_jsonl_search_by_bands() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let backend = JsonlBackend::new(&path).await.unwrap();

        let mut e1 = make_entry("a", "alpha");
        e1.bands = vec!["aaa".into(), "bbb".into()];
        backend.put(e1).await.unwrap();

        let mut e2 = make_entry("b", "beta");
        e2.bands = vec!["ccc".into(), "ddd".into()];
        backend.put(e2).await.unwrap();

        // Match position 0 = "aaa"
        let results = backend
            .search_by_bands(&["aaa".into(), "xxx".into()], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }
}
