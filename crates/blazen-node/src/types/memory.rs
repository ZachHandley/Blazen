//! Memory store and backend bindings for the Node.js SDK.
//!
//! Wraps `blazen-memory` types and backends for use from JavaScript/TypeScript.

use std::sync::Arc;

use async_trait::async_trait;
use napi::bindgen_prelude::Result;
use napi_derive::napi;

use blazen_memory::Memory;
use blazen_memory::backends::inmemory::InMemoryBackend;
use blazen_memory::backends::jsonl::JsonlBackend;
use blazen_memory::store::{MemoryBackend, MemoryStore};
use blazen_memory::types::StoredEntry;
use blazen_memory_valkey::ValkeyBackend;

use super::embedding::JsEmbeddingModel;
use crate::error::to_napi_error;

// ---------------------------------------------------------------------------
// Backend wrapper enum -- dispatches MemoryBackend through the three variants
// ---------------------------------------------------------------------------

/// Internal enum that implements [`MemoryBackend`] by dispatching to one of
/// the concrete backend types. Required because napi constructors cannot accept
/// trait objects.
enum BackendKind {
    InMemory(Arc<InMemoryBackend>),
    Jsonl(Arc<JsonlBackend>),
    Valkey(Arc<ValkeyBackend>),
}

#[async_trait]
impl MemoryBackend for BackendKind {
    async fn put(&self, entry: StoredEntry) -> blazen_memory::error::Result<()> {
        match self {
            Self::InMemory(b) => b.put(entry).await,
            Self::Jsonl(b) => b.put(entry).await,
            Self::Valkey(b) => b.put(entry).await,
        }
    }

    async fn get(&self, id: &str) -> blazen_memory::error::Result<Option<StoredEntry>> {
        match self {
            Self::InMemory(b) => b.get(id).await,
            Self::Jsonl(b) => b.get(id).await,
            Self::Valkey(b) => b.get(id).await,
        }
    }

    async fn delete(&self, id: &str) -> blazen_memory::error::Result<bool> {
        match self {
            Self::InMemory(b) => b.delete(id).await,
            Self::Jsonl(b) => b.delete(id).await,
            Self::Valkey(b) => b.delete(id).await,
        }
    }

    async fn list(&self) -> blazen_memory::error::Result<Vec<StoredEntry>> {
        match self {
            Self::InMemory(b) => b.list().await,
            Self::Jsonl(b) => b.list().await,
            Self::Valkey(b) => b.list().await,
        }
    }

    async fn len(&self) -> blazen_memory::error::Result<usize> {
        match self {
            Self::InMemory(b) => b.len().await,
            Self::Jsonl(b) => b.len().await,
            Self::Valkey(b) => b.len().await,
        }
    }

    async fn search_by_bands(
        &self,
        bands: &[String],
        limit: usize,
    ) -> blazen_memory::error::Result<Vec<StoredEntry>> {
        match self {
            Self::InMemory(b) => b.search_by_bands(bands, limit).await,
            Self::Jsonl(b) => b.search_by_bands(bands, limit).await,
            Self::Valkey(b) => b.search_by_bands(bands, limit).await,
        }
    }
}

// ---------------------------------------------------------------------------
// JsInMemoryBackend
// ---------------------------------------------------------------------------

/// An in-memory backend for the memory store.
///
/// Data lives only as long as the process; nothing is persisted.
///
/// ```javascript
/// const backend = new InMemoryBackend();
/// const memory = new Memory(embedder, backend);
/// ```
#[napi(js_name = "InMemoryBackend")]
pub struct JsInMemoryBackend {
    inner: Arc<InMemoryBackend>,
}

#[napi]
#[allow(clippy::new_without_default, clippy::must_use_candidate)]
impl JsInMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryBackend::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// JsJsonlBackend
// ---------------------------------------------------------------------------

/// A JSONL file-backed backend for the memory store.
///
/// Loads entries from the file on creation, appends new entries, and rewrites
/// on updates/deletes.
///
/// ```javascript
/// const backend = await JsonlBackend.create("./memory.jsonl");
/// const memory = new Memory(embedder, backend);
/// ```
#[napi(js_name = "JsonlBackend")]
pub struct JsJsonlBackend {
    inner: Arc<JsonlBackend>,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsJsonlBackend {
    /// Create a JSONL backend at the given file path.
    ///
    /// If the file exists, its contents are loaded. Otherwise an empty store
    /// is created and the file is written on the first insert.
    #[napi(factory)]
    pub async fn create(path: String) -> Result<Self> {
        let backend = JsonlBackend::new(&path).await.map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }
}

// ---------------------------------------------------------------------------
// JsValkeyBackend
// ---------------------------------------------------------------------------

/// A Valkey/Redis-backed backend for the memory store.
///
/// ```javascript
/// const backend = ValkeyBackend.create("redis://localhost:6379");
/// const memory = new Memory(embedder, backend);
/// ```
#[napi(js_name = "ValkeyBackend")]
pub struct JsValkeyBackend {
    inner: Arc<ValkeyBackend>,
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsValkeyBackend {
    /// Create a Valkey backend connected to the given Redis/Valkey URL.
    ///
    /// The URL should be in standard Redis format, e.g. `redis://localhost:6379`.
    #[napi(factory)]
    pub fn create(url: String) -> Result<Self> {
        let backend = ValkeyBackend::new(&url).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }
}

// ---------------------------------------------------------------------------
// Result / Entry objects
// ---------------------------------------------------------------------------

/// A search result from the memory store.
#[napi(object)]
pub struct JsMemoryResult {
    /// The entry id.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// Similarity score in 0.0..=1.0 (higher = more similar).
    pub score: f64,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

/// A stored entry retrieved from the memory store.
#[napi(object)]
pub struct JsMemoryEntry {
    /// The entry id.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// JsMemory
// ---------------------------------------------------------------------------

/// A memory store that uses ELID for vector indexing and similarity search.
///
/// Supports two modes:
/// - **Full mode** (`new Memory(embedder, backend)`): embedding-based search
/// - **Local mode** (`Memory.local(backend)`): text `SimHash` only, no embedder
///
/// ```javascript
/// import { Memory, EmbeddingModel, InMemoryBackend } from 'blazen';
///
/// const embedder = EmbeddingModel.openai(key);
/// const memory = new Memory(embedder, new InMemoryBackend());
///
/// await memory.add("doc1", "Paris is the capital of France");
/// const results = await memory.search("What is France's capital?", 5);
/// console.log(results[0].text);
/// ```
#[napi(js_name = "Memory")]
pub struct JsMemory {
    inner: Memory,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsMemory {
    /// Create a memory store with an embedding model for full ELID-based search.
    ///
    /// @param embedder - The embedding model to use for vectorisation.
    /// @param backend  - An `InMemoryBackend` instance.
    #[napi(constructor)]
    pub fn new(embedder: &JsEmbeddingModel, backend: &JsInMemoryBackend) -> Self {
        let inner = Memory::new(
            embedder.inner_arc(),
            BackendKind::InMemory(Arc::clone(&backend.inner)),
        );
        Self { inner }
    }

    /// Create a memory store with an embedding model and a `JsonlBackend`.
    #[napi(factory)]
    pub fn with_jsonl(embedder: &JsEmbeddingModel, backend: &JsJsonlBackend) -> Self {
        let inner = Memory::new(
            embedder.inner_arc(),
            BackendKind::Jsonl(Arc::clone(&backend.inner)),
        );
        Self { inner }
    }

    /// Create a memory store with an embedding model and a `ValkeyBackend`.
    #[napi(factory)]
    pub fn with_valkey(embedder: &JsEmbeddingModel, backend: &JsValkeyBackend) -> Self {
        let inner = Memory::new(
            embedder.inner_arc(),
            BackendKind::Valkey(Arc::clone(&backend.inner)),
        );
        Self { inner }
    }

    /// Create a memory store in local-only mode (no embedding model) with an `InMemoryBackend`.
    ///
    /// Only `searchLocal()` is available; `search()` will throw.
    #[napi(factory)]
    pub fn local(backend: &JsInMemoryBackend) -> Self {
        let inner = Memory::local(BackendKind::InMemory(Arc::clone(&backend.inner)));
        Self { inner }
    }

    /// Create a memory store in local-only mode with a `JsonlBackend`.
    #[napi(factory)]
    pub fn local_jsonl(backend: &JsJsonlBackend) -> Self {
        let inner = Memory::local(BackendKind::Jsonl(Arc::clone(&backend.inner)));
        Self { inner }
    }

    /// Create a memory store in local-only mode with a `ValkeyBackend`.
    #[napi(factory)]
    pub fn local_valkey(backend: &JsValkeyBackend) -> Self {
        let inner = Memory::local(BackendKind::Valkey(Arc::clone(&backend.inner)));
        Self { inner }
    }

    /// Add a text entry to the memory store.
    ///
    /// @param id       - A unique identifier for this entry.
    /// @param text     - The text content to store.
    /// @param metadata - Optional arbitrary metadata (will be stored as JSON).
    /// @returns The id of the stored entry.
    #[napi]
    pub async fn add(
        &self,
        id: String,
        text: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let entry = blazen_memory::MemoryEntry {
            id,
            text,
            metadata: metadata.unwrap_or(serde_json::Value::Null),
        };

        let ids = self.inner.add(vec![entry]).await.map_err(to_napi_error)?;

        Ok(ids.into_iter().next().unwrap_or_default())
    }

    /// Add multiple text entries to the memory store in a single batch.
    ///
    /// @param entries - An array of `{ id, text, metadata? }` objects.
    /// @returns The ids of the stored entries.
    #[napi]
    pub async fn add_many(&self, entries: Vec<JsAddEntry>) -> Result<Vec<String>> {
        let entries: Vec<blazen_memory::MemoryEntry> = entries
            .into_iter()
            .map(|e| blazen_memory::MemoryEntry {
                id: e.id,
                text: e.text,
                metadata: e.metadata.unwrap_or(serde_json::Value::Null),
            })
            .collect();

        self.inner.add(entries).await.map_err(to_napi_error)
    }

    /// Search using a query string with the configured embedding model.
    ///
    /// Returns up to `limit` results sorted by descending similarity.
    /// Requires an embedding model (throws if created with `Memory.local()`).
    ///
    /// @param query - The search query text.
    /// @param limit - Maximum number of results (default: 10).
    #[napi]
    pub async fn search(&self, query: String, limit: Option<u32>) -> Result<Vec<JsMemoryResult>> {
        let limit = limit.unwrap_or(10) as usize;
        let results = self
            .inner
            .search(&query, limit)
            .await
            .map_err(to_napi_error)?;

        Ok(results
            .into_iter()
            .map(|r| JsMemoryResult {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Search using only text-level `SimHash` (no embedding model required).
    ///
    /// This is a cheaper, lower-quality search that works in local-only mode.
    ///
    /// @param query - The search query text.
    /// @param limit - Maximum number of results (default: 10).
    #[napi]
    pub async fn search_local(
        &self,
        query: String,
        limit: Option<u32>,
    ) -> Result<Vec<JsMemoryResult>> {
        let limit = limit.unwrap_or(10) as usize;
        let results = self
            .inner
            .search_local(&query, limit)
            .await
            .map_err(to_napi_error)?;

        Ok(results
            .into_iter()
            .map(|r| JsMemoryResult {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Retrieve a single entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns The entry, or `null` if not found.
    #[napi]
    pub async fn get(&self, id: String) -> Result<Option<JsMemoryEntry>> {
        let entry = self.inner.get(&id).await.map_err(to_napi_error)?;

        Ok(entry.map(|e| JsMemoryEntry {
            id: e.id,
            text: e.text,
            metadata: e.metadata,
        }))
    }

    /// Delete an entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns `true` if the entry existed and was deleted.
    #[napi]
    pub async fn delete(&self, id: String) -> Result<bool> {
        self.inner.delete(&id).await.map_err(to_napi_error)
    }

    /// Return the number of entries in the store.
    #[napi]
    pub async fn count(&self) -> Result<u32> {
        let len = self.inner.len().await.map_err(to_napi_error)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(len as u32)
    }
}

// ---------------------------------------------------------------------------
// Input type for add_many
// ---------------------------------------------------------------------------

/// An entry to add to the memory store (used by `addMany`).
#[napi(object)]
pub struct JsAddEntry {
    /// Unique identifier. If empty, one will be generated.
    pub id: String,
    /// The text content to store.
    pub text: String,
    /// Optional arbitrary metadata.
    pub metadata: Option<serde_json::Value>,
}
