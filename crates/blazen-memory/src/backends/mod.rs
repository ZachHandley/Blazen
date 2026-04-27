//! Backend implementations for the memory store.

pub mod inmemory;
#[cfg(all(feature = "jsonl", not(target_arch = "wasm32")))]
pub mod jsonl;

pub use inmemory::InMemoryBackend;
#[cfg(all(feature = "jsonl", not(target_arch = "wasm32")))]
pub use jsonl::JsonlBackend;
