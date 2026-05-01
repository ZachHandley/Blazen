//! The main `Memory` struct that ties together embedding, ELID encoding, and backends.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, instrument};

use blazen_llm::EmbeddingModel;
use elid::embeddings::{self, Profile};

use crate::error::{MemoryError, Result};
use crate::search::{
    compute_elid_similarity, compute_embedding_simhash_similarity, compute_text_simhash_similarity,
    simhash_from_hex, simhash_to_hex,
};
use crate::store::{MemoryBackend, MemoryStore};
use crate::types::{MemoryEntry, MemoryResult, StoredEntry};

/// Default number of LSH bands (4 bands of 32 bits each from Mini128's 128 bits).
const DEFAULT_NUM_BANDS: u8 = 4;

/// Default seed for `SimHash` (matches ELID's conventional seed).
const DEFAULT_SEED: u64 = 0x454c_4944_5349_4d48; // "ELIDSIMH"

/// A memory store that uses ELID for vector indexing.
///
/// `Memory` can operate in two modes:
///
/// - **Full mode** (`Memory::new`): uses an [`EmbeddingModel`] to generate
///   embeddings, encodes them with ELID, and stores LSH bands for efficient
///   approximate nearest-neighbor search.
///
/// - **Local mode** (`Memory::local`): uses only string-level `SimHash`
///   (`elid::simhash`) for similarity. No embedding model is required, but
///   `search()` will return an error—use `search_local()` instead.
pub struct Memory {
    embedder: Option<Arc<dyn EmbeddingModel>>,
    backend: Arc<dyn MemoryBackend>,
    profile: Profile,
    num_bands: u8,
    seed: u64,
}

impl Memory {
    /// Create a `Memory` with an embedding model for full ELID-based search.
    pub fn new(embedder: Arc<dyn EmbeddingModel>, backend: impl MemoryBackend + 'static) -> Self {
        Self {
            embedder: Some(embedder),
            backend: Arc::new(backend),
            profile: Profile::default(), // Mini128
            num_bands: DEFAULT_NUM_BANDS,
            seed: DEFAULT_SEED,
        }
    }

    /// Create a `Memory` with an embedding model, accepting a pre-wrapped `Arc<dyn MemoryBackend>`.
    ///
    /// This is useful when the backend is already behind an `Arc` (e.g. from FFI bindings).
    pub fn new_arc(embedder: Arc<dyn EmbeddingModel>, backend: Arc<dyn MemoryBackend>) -> Self {
        Self {
            embedder: Some(embedder),
            backend,
            profile: Profile::default(),
            num_bands: DEFAULT_NUM_BANDS,
            seed: DEFAULT_SEED,
        }
    }

    /// Create a `Memory` without an embedding model (local-only mode).
    ///
    /// Only `search_local()` is available; `search()` will return
    /// [`MemoryError::NoEmbedder`].
    pub fn local(backend: impl MemoryBackend + 'static) -> Self {
        Self {
            embedder: None,
            backend: Arc::new(backend),
            profile: Profile::default(),
            num_bands: DEFAULT_NUM_BANDS,
            seed: DEFAULT_SEED,
        }
    }

    /// Create a local-only `Memory`, accepting a pre-wrapped `Arc<dyn MemoryBackend>`.
    ///
    /// This is useful when the backend is already behind an `Arc` (e.g. from FFI bindings).
    pub fn local_arc(backend: Arc<dyn MemoryBackend>) -> Self {
        Self {
            embedder: None,
            backend,
            profile: Profile::default(),
            num_bands: DEFAULT_NUM_BANDS,
            seed: DEFAULT_SEED,
        }
    }

    /// Override the ELID profile (default: `Mini128`).
    #[must_use]
    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    /// Override the number of LSH bands (default: 4).
    ///
    /// Valid values: 1, 2, 4, 8, 16 (must evenly divide 16 bytes).
    #[must_use]
    pub fn with_num_bands(mut self, num_bands: u8) -> Self {
        self.num_bands = num_bands;
        self
    }

    /// Override the `SimHash` seed (default: `0x454c4944_53494d48`).
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Check whether `entry_meta` is a superset of `filter`.
    ///
    /// - For objects: every key in `filter` must exist in `entry_meta` with a
    ///   recursively matching value.
    /// - For arrays: exact equality.
    /// - For scalars (string, number, bool, null): exact equality.
    fn metadata_matches(entry_meta: &serde_json::Value, filter: &serde_json::Value) -> bool {
        match (filter, entry_meta) {
            (serde_json::Value::Object(f), serde_json::Value::Object(e)) => f
                .iter()
                .all(|(k, fv)| e.get(k).is_some_and(|ev| Self::metadata_matches(ev, fv))),
            (f, e) => f == e,
        }
    }

    /// Embed a single text and return the raw f32 vector.
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let embedder = self.embedder.as_ref().ok_or(MemoryError::NoEmbedder)?;
        let response = embedder.embed(&[text.to_string()]).await?;
        response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("embedding model returned no vectors".into()))
    }

    /// Embed many texts in one batch and return the raw vectors.
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embedder = self.embedder.as_ref().ok_or(MemoryError::NoEmbedder)?;
        let response = embedder.embed(texts).await?;
        Ok(response.embeddings)
    }

    /// Encode an embedding into an ELID string + `SimHash` hex + LSH bands.
    fn encode_embedding(&self, embedding: &[f32]) -> Result<(String, String, Vec<String>)> {
        let elid = embeddings::encode(embedding, &self.profile)?;
        let elid_str = elid.as_str().to_string();

        let hash = embeddings::vector_simhash::simhash_128(embedding, self.seed);
        let hash_hex = simhash_to_hex(hash);

        let hash_bytes = embeddings::vector_simhash::simhash_to_bytes(hash);
        let bands = embeddings::vector_simhash::mini128_to_bands(&hash_bytes, self.num_bands);

        Ok((elid_str, hash_hex, bands))
    }

    /// Generate an id for an entry if one is not provided.
    fn ensure_id(id: &str) -> String {
        if id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            id.to_string()
        }
    }
}

#[async_trait]
impl MemoryStore for Memory {
    #[instrument(skip(self, entries), fields(count = entries.len()))]
    async fn add(&self, entries: Vec<MemoryEntry>) -> Result<Vec<String>> {
        // Compute text-level simhashes (always available).
        let text_simhashes: Vec<u64> = entries.iter().map(|e| elid::simhash(&e.text)).collect();

        // If we have an embedder, compute embeddings + ELIDs in batch.
        let embeddings = if self.embedder.is_some() {
            let texts: Vec<String> = entries.iter().map(|e| e.text.clone()).collect();
            Some(self.embed_texts(&texts).await?)
        } else {
            None
        };

        let mut ids = Vec::with_capacity(entries.len());

        for (i, entry) in entries.into_iter().enumerate() {
            let id = Self::ensure_id(&entry.id);

            let (elid_str, simhash_hex, bands) = if let Some(ref embs) = embeddings {
                let (elid_str, hash_hex, bands) = self.encode_embedding(&embs[i])?;
                (Some(elid_str), Some(hash_hex), bands)
            } else {
                (None, None, Vec::new())
            };

            let stored = StoredEntry {
                id: id.clone(),
                text: entry.text,
                elid: elid_str,
                simhash_hex,
                text_simhash: text_simhashes[i],
                bands,
                metadata: entry.metadata,
            };

            self.backend.put(stored).await?;
            ids.push(id);
        }

        debug!(count = ids.len(), "added entries to memory");
        Ok(ids)
    }

    #[instrument(skip(self, metadata_filter))]
    async fn search(
        &self,
        query: &str,
        limit: usize,
        metadata_filter: Option<&serde_json::Value>,
    ) -> Result<Vec<MemoryResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Step 1: Embed query.
        let query_embedding = self.embed_text(query).await?;

        // Step 2: Encode query embedding — get ELID + SimHash + bands.
        let (query_elid, query_hash_hex, query_bands) = self.encode_embedding(&query_embedding)?;
        let query_hash = simhash_from_hex(&query_hash_hex).unwrap_or(0);

        // Step 3: Get candidates via LSH band matching (fetch extra for re-ranking).
        let candidates = self
            .backend
            .search_by_bands(&query_bands, limit.saturating_mul(3))
            .await?;

        // If band search returned nothing, fall back to full scan.
        let candidates = if candidates.is_empty() {
            self.backend.list().await?
        } else {
            candidates
        };

        // Step 4: Score each candidate, applying metadata filter.
        let mut scored: Vec<MemoryResult> = candidates
            .into_iter()
            .filter(|entry| {
                metadata_filter.is_none_or(|f| Self::metadata_matches(&entry.metadata, f))
            })
            .map(|entry| {
                // Prefer ELID-based similarity, fall back to embedding SimHash, then text SimHash.
                let score = if let (Some(entry_elid), true) = (&entry.elid, !query_elid.is_empty())
                {
                    compute_elid_similarity(entry_elid, &query_elid).unwrap_or_else(|_| {
                        // Fallback to SimHash if ELID comparison fails (e.g. profile mismatch).
                        entry
                            .simhash_hex
                            .as_deref()
                            .and_then(simhash_from_hex)
                            .map_or(0.0, |h| compute_embedding_simhash_similarity(h, query_hash))
                    })
                } else if let Some(ref hex) = entry.simhash_hex {
                    simhash_from_hex(hex)
                        .map_or(0.0, |h| compute_embedding_simhash_similarity(h, query_hash))
                } else {
                    // No embedding data at all — use text SimHash.
                    compute_text_simhash_similarity(entry.text_simhash, elid::simhash(query))
                };

                MemoryResult {
                    id: entry.id,
                    text: entry.text,
                    score,
                    metadata: entry.metadata,
                }
            })
            .collect();

        // Step 5: Sort by descending score and take top `limit`.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);

        debug!(results = scored.len(), "search complete");
        Ok(scored)
    }

    #[instrument(skip(self, metadata_filter))]
    async fn search_local(
        &self,
        query: &str,
        limit: usize,
        metadata_filter: Option<&serde_json::Value>,
    ) -> Result<Vec<MemoryResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let query_hash = elid::simhash(query);

        let all = self.backend.list().await?;
        let mut scored: Vec<MemoryResult> = all
            .into_iter()
            .filter(|entry| {
                metadata_filter.is_none_or(|f| Self::metadata_matches(&entry.metadata, f))
            })
            .map(|entry| {
                let score = compute_text_simhash_similarity(entry.text_simhash, query_hash);
                MemoryResult {
                    id: entry.id,
                    text: entry.text,
                    score,
                    metadata: entry.metadata,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);

        debug!(results = scored.len(), "local search complete");
        Ok(scored)
    }

    async fn get(&self, id: &str) -> Result<Option<StoredEntry>> {
        self.backend.get(id).await
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        self.backend.delete(id).await
    }

    async fn len(&self) -> Result<usize> {
        self.backend.len().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::inmemory::InMemoryBackend;
    use blazen_llm::{BlazenError, EmbeddingResponse};

    // -----------------------------------------------------------------------
    // Mock embedding model
    // -----------------------------------------------------------------------

    /// A deterministic mock embedding model for testing.
    ///
    /// Produces embeddings by hashing the input text into a 128-dimensional
    /// vector of f32 values in [-1, 1].
    struct MockEmbedder;

    impl MockEmbedder {
        fn deterministic_embedding(text: &str) -> Vec<f32> {
            // Use a simple hash-based approach for deterministic embeddings.
            // Each dimension is derived from the text bytes.
            let bytes = text.as_bytes();
            let mut embedding = vec![0.0f32; 128];
            for (i, slot) in embedding.iter_mut().enumerate() {
                let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
                for &b in bytes {
                    hash ^= u64::from(b);
                    hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
                }
                hash ^= i as u64;
                hash = hash.wrapping_mul(0x0100_0000_01b3);
                // Map to [-1, 1]
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    clippy::cast_possible_wrap
                )]
                {
                    *slot = ((hash as i64) as f64 / i64::MAX as f64) as f32;
                }
            }
            embedding
        }
    }

    #[async_trait]
    impl EmbeddingModel for MockEmbedder {
        fn model_id(&self) -> &'static str {
            "mock-embedder"
        }

        fn dimensions(&self) -> usize {
            128
        }

        async fn embed(
            &self,
            texts: &[String],
        ) -> std::result::Result<EmbeddingResponse, BlazenError> {
            let embeddings = texts
                .iter()
                .map(|t| Self::deterministic_embedding(t))
                .collect();
            Ok(EmbeddingResponse {
                embeddings,
                model: "mock-embedder".to_string(),
                usage: None,
                cost: None,
                timing: None,
                metadata: serde_json::Value::Null,
            })
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_add_and_search() {
        let memory = Memory::new(Arc::new(MockEmbedder), InMemoryBackend::new());

        let ids = memory
            .add(vec![
                MemoryEntry::new("The cat sat on the mat"),
                MemoryEntry::new("The dog played in the park"),
                MemoryEntry::new("Quantum computing is the future"),
            ])
            .await
            .unwrap();

        assert_eq!(ids.len(), 3);

        let results = memory.search("cats sitting", 2, None).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // All scores should be in [0, 1].
        for r in &results {
            assert!(
                (0.0..=1.0).contains(&r.score),
                "score out of range: {}",
                r.score
            );
        }
    }

    #[tokio::test]
    async fn test_add_and_search_local() {
        let memory = Memory::local(InMemoryBackend::new());

        let ids = memory
            .add(vec![
                MemoryEntry::new("The cat sat on the mat"),
                MemoryEntry::new("The dog played in the park"),
                MemoryEntry::new("Quantum computing is the future"),
            ])
            .await
            .unwrap();

        assert_eq!(ids.len(), 3);

        // search() should fail in local mode.
        let err = memory.search("cats", 2, None).await;
        assert!(err.is_err());

        // search_local() should work.
        let results = memory.search_local("cat mat", 2, None).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        for r in &results {
            assert!(
                (0.0..=1.0).contains(&r.score),
                "score out of range: {}",
                r.score
            );
        }
    }

    #[tokio::test]
    async fn test_get_and_delete() {
        let memory = Memory::local(InMemoryBackend::new());

        let ids = memory
            .add(vec![MemoryEntry::new("Hello world")])
            .await
            .unwrap();
        let id = &ids[0];

        // Get
        let entry = memory.get(id).await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().text, "Hello world");

        // Delete
        let deleted = memory.delete(id).await.unwrap();
        assert!(deleted);

        // Get again
        let entry = memory.get(id).await.unwrap();
        assert!(entry.is_none());
    }

    #[tokio::test]
    async fn test_len() {
        let memory = Memory::local(InMemoryBackend::new());

        assert_eq!(memory.len().await.unwrap(), 0);

        memory
            .add(vec![MemoryEntry::new("one"), MemoryEntry::new("two")])
            .await
            .unwrap();

        assert_eq!(memory.len().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_custom_id() {
        let memory = Memory::local(InMemoryBackend::new());

        let ids = memory
            .add(vec![MemoryEntry::new("test").with_id("my-custom-id")])
            .await
            .unwrap();

        assert_eq!(ids[0], "my-custom-id");

        let entry = memory.get("my-custom-id").await.unwrap();
        assert!(entry.is_some());
    }

    #[tokio::test]
    async fn test_search_limit_zero() {
        let memory = Memory::local(InMemoryBackend::new());

        memory.add(vec![MemoryEntry::new("test")]).await.unwrap();

        let results = memory.search_local("test", 0, None).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_metadata_filter_local() {
        let memory = Memory::local(InMemoryBackend::new());

        memory
            .add(vec![
                MemoryEntry::new("Paris is the capital of France")
                    .with_metadata(serde_json::json!({"category": "geography", "lang": "en"})),
                MemoryEntry::new("Berlin is the capital of Germany")
                    .with_metadata(serde_json::json!({"category": "geography", "lang": "de"})),
                MemoryEntry::new("Rust is a systems programming language")
                    .with_metadata(serde_json::json!({"category": "tech", "lang": "en"})),
            ])
            .await
            .unwrap();

        // No filter — all 3 results.
        let all = memory.search_local("capital", 10, None).await.unwrap();
        assert_eq!(all.len(), 3);

        // Filter by category=geography — 2 results.
        let geo_filter = serde_json::json!({"category": "geography"});
        let geo = memory
            .search_local("capital", 10, Some(&geo_filter))
            .await
            .unwrap();
        assert_eq!(geo.len(), 2);
        for r in &geo {
            assert!(
                r.text.contains("capital"),
                "expected geography entries, got: {}",
                r.text
            );
        }

        // Filter by lang=en — 2 results.
        let en_filter = serde_json::json!({"lang": "en"});
        let en = memory
            .search_local("capital", 10, Some(&en_filter))
            .await
            .unwrap();
        assert_eq!(en.len(), 2);

        // Filter by category=geography AND lang=de — 1 result.
        let de_geo_filter = serde_json::json!({"category": "geography", "lang": "de"});
        let de_geo = memory
            .search_local("capital", 10, Some(&de_geo_filter))
            .await
            .unwrap();
        assert_eq!(de_geo.len(), 1);
        assert!(de_geo[0].text.contains("Berlin"));

        // Filter with no matches.
        let none_filter = serde_json::json!({"category": "sports"});
        let none = memory
            .search_local("capital", 10, Some(&none_filter))
            .await
            .unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn test_metadata_filter_nested() {
        let memory = Memory::local(InMemoryBackend::new());

        memory
            .add(vec![
                MemoryEntry::new("entry one")
                    .with_metadata(serde_json::json!({"source": {"type": "web", "url": "a.com"}})),
                MemoryEntry::new("entry two")
                    .with_metadata(serde_json::json!({"source": {"type": "file", "path": "/tmp"}})),
            ])
            .await
            .unwrap();

        let filter = serde_json::json!({"source": {"type": "web"}});
        let results = memory
            .search_local("entry", 10, Some(&filter))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("one"));
    }

    #[tokio::test]
    async fn test_metadata_filter_with_embedder() {
        let memory = Memory::new(Arc::new(MockEmbedder), InMemoryBackend::new());

        memory
            .add(vec![
                MemoryEntry::new("The cat sat on the mat")
                    .with_metadata(serde_json::json!({"animal": "cat"})),
                MemoryEntry::new("The dog played in the park")
                    .with_metadata(serde_json::json!({"animal": "dog"})),
            ])
            .await
            .unwrap();

        let filter = serde_json::json!({"animal": "cat"});
        let results = memory
            .search("cat sitting", 5, Some(&filter))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("cat"));
    }

    #[test]
    fn test_metadata_matches_basic() {
        let entry = serde_json::json!({"a": 1, "b": "hello", "c": true});

        // Subset match.
        assert!(Memory::metadata_matches(
            &entry,
            &serde_json::json!({"a": 1})
        ));
        assert!(Memory::metadata_matches(
            &entry,
            &serde_json::json!({"a": 1, "b": "hello"})
        ));

        // Full match.
        assert!(Memory::metadata_matches(
            &entry,
            &serde_json::json!({"a": 1, "b": "hello", "c": true})
        ));

        // Mismatch.
        assert!(!Memory::metadata_matches(
            &entry,
            &serde_json::json!({"a": 2})
        ));
        assert!(!Memory::metadata_matches(
            &entry,
            &serde_json::json!({"missing": true})
        ));

        // Null entry metadata matches only null filter.
        assert!(Memory::metadata_matches(
            &serde_json::Value::Null,
            &serde_json::Value::Null
        ));
        assert!(!Memory::metadata_matches(
            &serde_json::Value::Null,
            &serde_json::json!({"a": 1})
        ));
    }
}
