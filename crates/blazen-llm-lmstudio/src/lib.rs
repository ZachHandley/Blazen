//! LM Studio HTTP-proxy backend for Blazen.
//!
//! [`LmStudioProvider`] forwards inference, embedding, and model
//! lifecycle operations to a running [LM Studio](https://lmstudio.ai)
//! local server over its dual HTTP surface. LM Studio exposes both an
//! OpenAI-compatible namespace (`/v1/*`) and an LM-Studio-native
//! namespace (`/api/v0/*`) for model management. The endpoints
//! exercised here, as of LM Studio 0.3:
//!
//! - `POST /v1/chat/completions`   — OAI-shaped chat completion (SSE stream).
//! - `POST /v1/completions`        — OAI-shaped legacy completion.
//! - `POST /v1/embeddings`         — OAI-shaped vector embeddings.
//! - `GET  /v1/models`             — OAI-shaped model listing.
//! - `GET  /api/v0/models`         — native listing with load status.
//! - `POST /api/v0/models/load`    — load a model into the server.
//! - `POST /api/v0/models/unload`  — unload a previously-loaded model.
//!
//! Unlike vLLM and Ollama, LM Studio does **not** expose a runtime
//! LoRA-adapter mount/unmount API — adapters must be baked into the
//! GGUF model file (`mradermacher/*-i1-GGUF`-style merged checkpoints)
//! and then loaded via `/api/v0/models/load`. See
//! [`LmStudioProvider::load_adapter`] for the strategy.
//!
//! This crate is intentionally a leaf — it does **not** depend on
//! `blazen-llm`. The trait-impl bridge lives in
//! `crates/blazen-llm/src/backends/lmstudio.rs` behind the `lmstudio`
//! Cargo feature on `blazen-llm`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                  |
//! |----------|----------------------------------------------|
//! | `engine` | Reserved (currently a no-op for parity with  |
//! |          | other local-backend crates).                 |
//!
//! There is no runtime to optionally link — LM Studio lives in another
//! process.

mod client;
mod error;
mod options;
mod provider;

pub use client::{
    LmStudioClient, LmStudioModelEntry, LmStudioNativeModelEntry, LmStudioNativeModelState,
};
pub use error::LmStudioError;
pub use options::{LmStudioAdapterTransport, LmStudioOptions};
pub use provider::LmStudioProvider;
