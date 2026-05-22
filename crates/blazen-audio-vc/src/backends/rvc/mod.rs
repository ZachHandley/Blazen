//! Retrieval-based Voice Conversion (RVC) backend.
//!
//! Submodules are wired in as empty placeholders for Wave D.2, which
//! lands the F0 extractor, content (`HuBERT`) encoder, kNN feature
//! retrieval index, generator network, end-to-end decoding pipeline,
//! and weights loader.

pub mod content;
pub mod f0;
pub mod generator;
pub mod pipeline;
pub mod retrieval;
pub mod weights;

pub use pipeline::RvcBackend;
