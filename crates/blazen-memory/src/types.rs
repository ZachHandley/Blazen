//! Core data types for the memory store.

use serde::{Deserialize, Serialize};

/// An entry as persisted in a backend, including ELID and LSH data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// ELID-encoded embedding string (None when running in local-only mode).
    pub elid: Option<String>,
    /// 128-bit `SimHash` of the embedding (for ELID-based search).
    /// Stored as a hex string for serialization portability.
    pub simhash_hex: Option<String>,
    /// String-level `SimHash` of the raw text (always available, for local search).
    pub text_simhash: u64,
    /// LSH band strings derived from the embedding `SimHash`.
    pub bands: Vec<String>,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

/// A search result returned to the caller.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    /// The entry id.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// Similarity score in 0.0..=1.0, where higher means more similar.
    pub score: f64,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

/// A lightweight input struct for adding entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier. If empty, one will be generated.
    pub id: String,
    /// The text content to store.
    pub text: String,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

impl MemoryEntry {
    /// Create a new entry with just text and default metadata.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            text: text.into(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Set the id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Attach metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}
