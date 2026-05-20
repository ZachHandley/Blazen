//! vLLM HTTP-proxy backend for Blazen.
//!
//! [`VllmProvider`] forwards inference and LoRA-adapter operations to a
//! running [vLLM](https://github.com/vllm-project/vllm) server over its
//! OpenAI-compatible HTTP surface. As of vLLM 0.10 the relevant
//! endpoints are:
//!
//! - `POST /v1/chat/completions`     — OAI-shaped chat completion.
//! - `POST /v1/load_lora_adapter`    — runtime `LoRA` mount (gated by the
//!   `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` server env var).
//! - `POST /v1/unload_lora_adapter`  — runtime `LoRA` unmount.
//! - `GET  /v1/models`               — base model + mounted adapter rows.
//!
//! This crate is intentionally a leaf — it does **not** depend on
//! `blazen-llm`. The trait-impl bridge lives in
//! `crates/blazen-llm/src/backends/vllm.rs` behind the `vllm` Cargo
//! feature on `blazen-llm`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                  |
//! |----------|----------------------------------------------|
//! | `engine` | Reserved (currently a no-op for parity with  |
//! |          | other local-backend crates).                 |
//!
//! There is no runtime to optionally link — vLLM lives in another
//! process. The flag exists so binding crates can opt into the proxy
//! the same way they opt into mistral.rs / llama.cpp / candle today.

mod client;
mod error;
mod options;
mod provider;

pub use client::{VllmClient, VllmModelEntry};
pub use error::VllmError;
pub use options::{VllmAdapterTransport, VllmOptions};
pub use provider::{MountedAdapter, VllmProvider};
