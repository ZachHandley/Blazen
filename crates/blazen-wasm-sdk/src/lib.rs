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
//! const model = CompletionModel.openrouter('or-key');
//! const response = await model.complete([ChatMessage.user('Hello!')]);
//! console.log(response.content);
//! ```

pub mod agent;
pub mod chat_message;
pub mod completion_model;
pub mod embedding;
pub mod tokens;
pub mod types;
pub mod workflow;

use wasm_bindgen::prelude::*;

/// Initialise the WASM module.
///
/// Sets the panic hook so that Rust panics produce readable console
/// error messages instead of `unreachable` traps.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
