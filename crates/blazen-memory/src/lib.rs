//! # blazen-memory
//!
//! Memory and vector store for Blazen with ELID integration.
//!
//! This crate provides a memory store that can operate in two modes:
//!
//! - **Full mode**: Uses an [`EmbeddingModel`](blazen_llm::EmbeddingModel) to
//!   generate embeddings, encodes them with [ELID](elid) into compact sortable
//!   identifiers, and uses LSH (Locality-Sensitive Hashing) bands for efficient
//!   approximate nearest-neighbor retrieval.
//!
//! - **Local mode**: Uses string-level `SimHash` for lightweight similarity search
//!   without requiring an embedding model.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use blazen_memory::{Memory, MemoryStore, MemoryEntry, InMemoryBackend};
//! use std::sync::Arc;
//!
//! # async fn example(embedder: Arc<dyn blazen_llm::EmbeddingModel>) -> blazen_memory::error::Result<()> {
//! let memory = Memory::new(embedder, InMemoryBackend::new());
//!
//! memory.add(vec![
//!     MemoryEntry::new("The cat sat on the mat"),
//!     MemoryEntry::new("The dog played in the park"),
//! ]).await?;
//!
//! let results = memory.search("animals sitting", 5, None).await?;
//! for r in results {
//!     println!("{:.3} — {}", r.score, r.text);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Local-only mode
//!
//! ```rust,no_run
//! use blazen_memory::{Memory, MemoryStore, MemoryEntry, InMemoryBackend};
//!
//! # async fn example() -> blazen_memory::error::Result<()> {
//! let memory = Memory::local(InMemoryBackend::new());
//!
//! memory.add(vec![MemoryEntry::new("hello world")]).await?;
//!
//! // Use search_local instead of search — no embedding model needed.
//! let results = memory.search_local("hello", 5, None).await?;
//! # Ok(())
//! # }
//! ```

pub mod backends;
pub mod error;
pub mod memory;
// `retry` is available on both native and wasm32. On wasm32 the
// `RetryMemoryBackend::sleep_ms` helper routes through `gloo-timers`
// (see `crates/blazen-memory/src/retry.rs`) instead of
// `tokio::time::sleep`, which has no time driver on
// `wasm32-unknown-unknown`.
pub mod retry;
pub mod search;
pub mod store;
pub mod types;

// Re-exports for ergonomic imports.
pub use backends::InMemoryBackend;
#[cfg(all(feature = "jsonl", not(target_arch = "wasm32")))]
pub use backends::JsonlBackend;
pub use error::MemoryError;
pub use memory::Memory;
pub use retry::RetryMemoryBackend;
pub use store::{MemoryBackend, MemoryStore};
pub use types::{MemoryEntry, MemoryResult, StoredEntry};
