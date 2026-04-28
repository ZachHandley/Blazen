//! Pure-Rust ONNX inference backend for Blazen embeddings via `tract-onnx`.
//!
//! Mirrors the public API of `blazen-embed-fastembed` so the two backends are
//! swappable via `cfg` gating in `blazen-llm`. Exists because `fastembed`/`ort`
//! require Microsoft's prebuilt ONNX Runtime binaries which are not published
//! for several target triples (notably `*-unknown-linux-musl` and `wasm32-*`).
//!
//! ## wasm32 support
//!
//! On `wasm32-*` targets the native [`provider`] module is compiled out
//! because its `from_options` constructor relies on `tokio` runtime primitives
//! and on `blazen-model-cache`'s `HuggingFace` Hub downloader, neither of
//! which compiles to wasm32. In its place, [`wasm_provider`] exposes
//! [`WasmTractEmbedModel`] which downloads ONNX weights and the tokenizer via
//! `web_sys::fetch` and runs inference through the same `tract-onnx` pipeline.

pub mod options;

#[cfg(not(target_arch = "wasm32"))]
pub mod provider;

#[cfg(target_arch = "wasm32")]
pub mod wasm_provider;

pub use options::TractOptions;

#[cfg(not(target_arch = "wasm32"))]
pub use provider::{TractEmbedModel, TractError, TractResponse};

#[cfg(target_arch = "wasm32")]
pub use wasm_provider::{WasmTractEmbedModel, WasmTractError, WasmTractResponse};
