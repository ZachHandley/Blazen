//! `llama.cpp` HTTP-server proxy backend for Blazen.
//!
//! [`LlamacppServerProvider`] forwards inference, embedding, and
//! adapter operations to a running `llama-server` binary (from
//! `ggerganov/llama.cpp`) over its HTTP API. The endpoints exercised
//! here, as of `llama.cpp` build b4334+:
//!
//! - `POST /v1/chat/completions` — OpenAI-compat chat (JSON or SSE).
//! - `POST /v1/completions`      — OpenAI-compat text completion.
//! - `POST /v1/embeddings`       — OpenAI-compat embeddings.
//! - `GET  /v1/models`           — model listing.
//! - `POST /completion`          — llama.cpp-native completion shape.
//! - `GET  /health`              — readiness probe.
//! - `GET  /slots`               — per-slot introspection.
//! - `GET  /lora-adapters`       — list preloaded `LoRA` adapters.
//! - `POST /lora-adapters`       — toggle the active adapter set /
//!   scales.
//!
//! **This crate is the network proxy** — distinct from
//! `blazen-llm-llamacpp`, which links the `llama-cpp-sys-2` C++ runtime
//! in-process. Use the proxy when:
//!
//! - you want to share one `llama-server` instance across multiple
//!   Blazen workers or processes,
//! - you need to dodge the `llama-cpp-sys-2` build (no C++ toolchain on
//!   the Blazen host, no GPU drivers in the Blazen container),
//! - you want `llama-server`'s slot-based scheduling for concurrent
//!   requests.
//!
//! ## Adapter strategy
//!
//! `llama-server` cannot accept adapter binaries over HTTP. Adapters
//! must be preloaded at server startup via the `--lora <path>` /
//! `--lora-scaled <path> <scale>` CLI flags. At runtime,
//! `POST /lora-adapters` only TOGGLES which preloaded adapters are
//! active and at what scale (no weight reload, sub-millisecond switch).
//!
//! [`LlamacppServerProvider::load_adapter`] therefore:
//!
//! 1. Calls `GET /lora-adapters` to discover the preloaded set.
//! 2. Matches the supplied path against the server-reported `path`
//!    field.
//! 3. POSTs the union of currently-active adapters PLUS the new one at
//!    `scale: 1.0` to `/lora-adapters`.
//!
//! [`LlamacppServerAdapterTransport::HttpPush`] and
//! [`LlamacppServerAdapterTransport::HfHub`] are both rejected with
//! [`LlamacppServerError::Unsupported`] — `llama-server` has neither a
//! binary-upload endpoint nor a Hugging Face Hub resolver.
//!
//! This crate is intentionally a leaf — it does **not** depend on
//! `blazen-llm`. The trait-impl bridge lives in
//! `crates/blazen-llm/src/backends/llamacpp_server.rs` behind the
//! `llamacpp-server` Cargo feature on `blazen-llm`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                  |
//! |----------|----------------------------------------------|
//! | `engine` | Reserved (currently a no-op for parity with  |
//! |          | other local-backend crates).                 |
//!
//! There is no runtime to optionally link — `llama-server` lives in
//! another process.

mod client;
mod error;
mod options;
mod provider;

pub use client::{
    LlamacppServerClient, LlamacppServerHealth, LlamacppServerLoraAdapter,
    LlamacppServerLoraToggle, LlamacppServerModelEntry, LlamacppServerSlot,
};
pub use error::LlamacppServerError;
pub use options::{LlamacppServerAdapterTransport, LlamacppServerOptions};
pub use provider::{LlamacppServerProvider, MountedAdapter};
