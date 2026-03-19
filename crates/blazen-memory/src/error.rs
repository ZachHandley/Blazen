//! Error types for blazen-memory operations.

/// Errors that can occur during memory store operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// No embedder was configured but an operation requiring one was called.
    #[error("no embedding model configured — use Memory::new() instead of Memory::local()")]
    NoEmbedder,

    /// ELID encoding or decoding failed.
    #[error("ELID error: {0}")]
    Elid(#[from] elid::embeddings::ElidError),

    /// The upstream embedding model returned an error.
    #[error("embedding model error: {0}")]
    Embedding(String),

    /// An entry was not found by the given id.
    #[error("entry not found: {0}")]
    NotFound(String),

    /// Serialization / deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// I/O error (file backend).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A backend-specific error.
    #[error("backend error: {0}")]
    Backend(String),
}

impl From<serde_json::Error> for MemoryError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

impl From<blazen_llm::BlazenError> for MemoryError {
    fn from(e: blazen_llm::BlazenError) -> Self {
        Self::Embedding(e.to_string())
    }
}

/// Result alias for memory operations.
pub type Result<T, E = MemoryError> = std::result::Result<T, E>;
