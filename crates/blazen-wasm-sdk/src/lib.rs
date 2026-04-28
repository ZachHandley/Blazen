#![cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
//! # Blazen WASM SDK
//!
//! A client-side SDK that compiles to WebAssembly, exposing the Blazen
//! workflow engine, LLM completion, agent loop, and pipeline to
//! TypeScript / JavaScript in the browser and Node.js.
//!
//! HTTP calls are routed through the browser `fetch()` API via
//! [`blazen_llm::FetchHttpClient`] which implements the
//! [`blazen_llm::http::HttpClient`] trait.
//!
//! ## Quick start (TypeScript)
//!
//! ```typescript
//! import init, { CompletionModel, ChatMessage } from '@blazen/sdk';
//!
//! await init(); // load WASM module
//!
//! const model = CompletionModel.openrouter();
//! const response = await model.complete([ChatMessage.user('Hello!')]);
//! console.log(response.content);
//! ```

pub mod agent;
pub mod agent_types;
pub mod batch;
pub mod capability_providers;
pub mod chat_message;
pub mod completion_model;
pub mod completion_types;
pub mod compute_provider;
pub mod context;
pub mod core_types;
pub mod decorators;
// `embed_tract` is intentionally omitted on wasm32: the upstream
// `blazen-embed-tract` crate's `provider` module (which owns
// `TractEmbedModel`) is gated native-only because its `from_options`
// constructor relies on `tokio` `rt`/`fs` features and on the HuggingFace Hub
// downloader (`hf-hub`), neither of which compiles to wasm32. A future
// iteration can re-introduce a wasm-friendly tract binding once weights are
// fed in via `fetch()` instead of `hf-hub`.
pub mod embedding;
pub mod events;
pub mod handler;
pub mod http_client;
pub mod js_completion;
pub mod js_embedding;
pub mod manager;
pub mod memory;
pub mod memory_types;
pub mod middleware;
pub mod model_abcs;
pub mod pipeline;
pub mod pricing;
pub mod providers;
pub mod session_pause_policy;
pub mod telemetry;
pub mod tokens;
pub mod types;
pub mod workflow;
pub mod workflow_events;

use wasm_bindgen::prelude::*;

/// Initialise the WASM module.
///
/// Sets the panic hook so that Rust panics produce readable console
/// error messages instead of `unreachable` traps.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
