//! Ollama HTTP-proxy backend for Blazen.
//!
//! [`OllamaProvider`] forwards inference, embedding, and adapter
//! operations to a running [Ollama](https://ollama.com) server over its
//! native HTTP API. The endpoints exercised here, as of Ollama 0.3:
//!
//! - `POST /api/generate`       — completion (NDJSON stream or single).
//! - `POST /api/chat`           — chat completion (NDJSON stream or single).
//! - `POST /api/embeddings`     — embed one or more inputs.
//! - `GET  /api/tags`           — list installed models.
//! - `POST /api/show`           — model metadata (template, parameters, ...).
//! - `POST /api/pull`           — pull a model from the registry
//!   (NDJSON progress stream).
//! - `POST /api/create`         — create a derived model from a
//!   Modelfile. Used here to mount a `LoRA` adapter via the `ADAPTER`
//!   directive.
//! - `DELETE /api/delete`       — remove an installed model.
//!
//! Unlike vLLM, Ollama does not expose a runtime adapter mount/unmount
//! pair. Adapters become full derived models via `/api/create` + a
//! Modelfile, which is the contract the upstream supports. See
//! [`OllamaProvider::load_adapter`] for the strategy.
//!
//! This crate is intentionally a leaf — it does **not** depend on
//! `blazen-llm`. The trait-impl bridge lives in
//! `crates/blazen-llm/src/backends/ollama.rs` behind the `ollama` Cargo
//! feature on `blazen-llm`.

mod client;
mod error;
mod options;
mod provider;

pub use client::{
    OllamaClient, OllamaModelEntry, OllamaPullProgress, OllamaShowResponse, OllamaStreamPart,
};
pub use error::OllamaError;
pub use options::{OllamaAdapterTransport, OllamaOptions};
pub use provider::{MountedAdapter, OllamaProvider};
