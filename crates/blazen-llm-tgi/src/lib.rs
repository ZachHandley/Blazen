//! `HuggingFace` Text Generation Inference (TGI) HTTP-proxy backend for
//! Blazen.
//!
//! [`TgiProvider`] forwards inference and adapter-selection operations to
//! a running [TGI](https://huggingface.co/docs/text-generation-inference)
//! server. TGI exposes two parallel surfaces on the same port:
//!
//! - **Native TGI shape** (always available):
//!   - `POST /generate`        — single-shot completion
//!   - `POST /generate_stream` — SSE-framed token stream
//!   - `GET  /info`            — model id + runtime configuration
//!   - `GET  /metrics`         — Prometheus-format metrics (optional)
//! - **OpenAI-compatible shape** (TGI ≥ 1.4):
//!   - `POST /v1/chat/completions` — OAI chat (single + SSE)
//!   - `POST /v1/completions`      — OAI legacy completion
//!   - `GET  /v1/models`           — base model + loaded adapters
//!
//! Both shapes accept an `adapter_id` field for per-request adapter
//! selection. TGI does **not** mount adapters at runtime — the server
//! must be started with `--lora-adapters <id1> <id2> ...` and the proxy
//! picks one of the preloaded ids per request. See
//! [`TgiProvider::load_adapter`] for the strategy.
//!
//! This crate is intentionally a leaf — it does **not** depend on
//! `blazen-llm`. The trait-impl bridge lives in
//! `crates/blazen-llm/src/backends/tgi.rs` behind the `tgi` Cargo
//! feature on `blazen-llm`.

mod client;
mod error;
mod options;
mod provider;

pub use client::{TgiClient, TgiInfo, TgiModelEntry};
pub use error::TgiError;
pub use options::{TgiAdapterTransport, TgiOptions};
pub use provider::{ActiveAdapter, TgiProvider};
