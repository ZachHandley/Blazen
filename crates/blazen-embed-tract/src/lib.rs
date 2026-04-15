//! Pure-Rust ONNX inference backend for Blazen embeddings via `tract-onnx`.
//!
//! Mirrors the public API of `blazen-embed-fastembed` so the two backends are
//! swappable via `cfg` gating in `blazen-llm`. Exists because `fastembed`/`ort`
//! require Microsoft's prebuilt ONNX Runtime binaries which are not published
//! for several target triples (notably `*-unknown-linux-musl` and `wasm32-*`).

pub mod options;
pub mod provider;

pub use options::TractOptions;
pub use provider::{TractEmbedModel, TractError, TractResponse};
