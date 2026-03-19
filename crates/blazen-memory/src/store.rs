//! Trait definitions for `MemoryStore` (high-level) and `MemoryBackend` (storage).

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{MemoryEntry, MemoryResult, StoredEntry};

// ---------------------------------------------------------------------------
// MemoryStore — the user-facing interface
// ---------------------------------------------------------------------------

/// High-level memory store that handles embedding, ELID encoding, and search.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Add one or more entries to the store.
    ///
    /// If an entry's `id` field is empty, a UUID will be generated.
    async fn add(&self, entries: Vec<MemoryEntry>) -> Result<Vec<String>>;

    /// Search using a query string with the configured embedding model.
    ///
    /// Returns up to `limit` results sorted by descending similarity.
    /// Requires an embedding model to be configured (returns `MemoryError::NoEmbedder` otherwise).
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryResult>>;

    /// Search using only text-level `SimHash` (no embedding model required).
    ///
    /// This is a cheaper, lower-quality search that works in local-only mode.
    async fn search_local(&self, query: &str, limit: usize) -> Result<Vec<MemoryResult>>;

    /// Retrieve a single entry by id.
    async fn get(&self, id: &str) -> Result<Option<StoredEntry>>;

    /// Delete an entry by id.
    async fn delete(&self, id: &str) -> Result<bool>;

    /// Return the number of entries.
    async fn len(&self) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// MemoryBackend — the storage layer
// ---------------------------------------------------------------------------

/// Low-level storage backend used by [`Memory`](crate::memory::Memory).
///
/// Backends are responsible for persistence and band-based candidate retrieval.
/// They do NOT perform embedding or ELID encoding; that is the job of
/// [`Memory`](crate::memory::Memory).
#[async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Insert or update a stored entry.
    async fn put(&self, entry: StoredEntry) -> Result<()>;

    /// Retrieve a stored entry by its id.
    async fn get(&self, id: &str) -> Result<Option<StoredEntry>>;

    /// Delete a stored entry by id. Returns true if it existed.
    async fn delete(&self, id: &str) -> Result<bool>;

    /// Return all stored entries (used for local search and small stores).
    async fn list(&self) -> Result<Vec<StoredEntry>>;

    /// Return the number of stored entries.
    async fn len(&self) -> Result<usize>;

    /// Return candidate entries that share at least one LSH band with the query bands.
    ///
    /// `limit` is a soft cap — backends may return more if convenient.
    async fn search_by_bands(&self, bands: &[String], limit: usize) -> Result<Vec<StoredEntry>>;
}
