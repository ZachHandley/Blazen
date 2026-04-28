//! Pure-Rust ONNX inference backend for Blazen embeddings via `tract-onnx`.
//!
//! Mirrors the public API of `blazen-embed-fastembed` so the two backends are
//! swappable via `cfg` gating in `blazen-llm`. Exists because `fastembed`/`ort`
//! require Microsoft's prebuilt ONNX Runtime binaries which are not published
//! for several target triples (notably `*-unknown-linux-musl` and `wasm32-*`).
//!
//! ## wasm32 support
//!
//! On `wasm32-*` targets the [`provider`] module (which owns the actual
//! [`TractEmbedModel`] implementation) is compiled out: tract's ONNX runtime
//! is fine on wasm in principle, but the `from_options` constructor relies on
//! `tokio` runtime primitives and on `blazen-model-cache`'s `HuggingFace` Hub
//! downloader, neither of which compiles to wasm32. The [`options`] module
//! (registry + [`TractOptions`]) is still available so wasm consumers can use
//! the same model catalog if they download weights through a different path
//! (e.g. JavaScript `fetch`).

pub mod options;

#[cfg(not(target_arch = "wasm32"))]
pub mod provider;

pub use options::TractOptions;

#[cfg(not(target_arch = "wasm32"))]
pub use provider::{TractEmbedModel, TractError, TractResponse};
