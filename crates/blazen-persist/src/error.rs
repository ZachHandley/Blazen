//! Error types for the persistence layer.

use thiserror::Error;

/// Errors produced by the checkpoint storage backends.
#[derive(Debug, Error)]
pub enum PersistError {
    /// A storage-level error from the underlying backend.
    #[error("storage error: {0}")]
    Storage(String),

    /// JSON serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The requested checkpoint was not found.
    #[error("checkpoint not found: {0}")]
    NotFound(uuid::Uuid),

    /// An error from the redb database engine.
    #[error("database error: {0}")]
    Database(String),

    /// An error from the Redis/ValKey backend.
    #[error("redis error: {0}")]
    Redis(String),
}

// ---------------------------------------------------------------------------
// redb error conversions (only available when the `redb` feature is enabled)
// ---------------------------------------------------------------------------

// redb 3.x has multiple error types (DatabaseError, StorageError, TableError,
// TransactionError, CommitError) that all convert into the top-level
// `redb::Error`. We provide `From` for the top-level enum as well as the
// individual error types that our code directly encounters.

#[cfg(feature = "redb")]
impl From<redb::Error> for PersistError {
    fn from(err: redb::Error) -> Self {
        Self::Database(err.to_string())
    }
}

#[cfg(feature = "redb")]
impl From<redb::DatabaseError> for PersistError {
    fn from(err: redb::DatabaseError) -> Self {
        Self::Database(err.to_string())
    }
}

#[cfg(feature = "redb")]
impl From<redb::StorageError> for PersistError {
    fn from(err: redb::StorageError) -> Self {
        Self::Database(err.to_string())
    }
}

#[cfg(feature = "redb")]
impl From<redb::TableError> for PersistError {
    fn from(err: redb::TableError) -> Self {
        Self::Database(err.to_string())
    }
}

#[cfg(feature = "redb")]
impl From<redb::TransactionError> for PersistError {
    fn from(err: redb::TransactionError) -> Self {
        Self::Database(err.to_string())
    }
}

#[cfg(feature = "redb")]
impl From<redb::CommitError> for PersistError {
    fn from(err: redb::CommitError) -> Self {
        Self::Database(err.to_string())
    }
}

// ---------------------------------------------------------------------------
// Redis/ValKey error conversion (only available when the `valkey` feature is
// enabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "valkey")]
impl From<redis::RedisError> for PersistError {
    fn from(err: redis::RedisError) -> Self {
        Self::Redis(err.to_string())
    }
}
