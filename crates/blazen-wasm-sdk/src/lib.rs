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
//! import init, { CompletionModel, ChatMessage } from '@blazen-dev/wasm';
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
pub mod content;
pub mod context;
pub mod core_types;
pub mod decorators;
pub mod embed_tract;
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
pub mod model_registry;
pub mod pipeline;
pub mod pricing;
pub mod providers;
pub mod session_pause_policy;
pub mod subworkflow_step;
pub mod telemetry;
pub mod tokens;
pub mod types;
pub mod workflow;
pub mod workflow_events;

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// TypeScript type alias declarations
// ---------------------------------------------------------------------------
//
// `MediaSource` is a Rust type alias (`pub type MediaSource = ImageSource`) in
// `blazen-llm`. Tsify only emits type definitions for `#[derive(Tsify)]`
// structs/enums, so the alias never appears in the generated `.d.ts` even
// though several emitted shapes (`ImageContent`, `AudioContent`,
// `VideoContent`, `FileContent`) reference `MediaSource` as a field type.
// Without this declaration, `import { MediaSource } from 'blazen_wasm_sdk'`
// fails to type-check on the consumer side.
#[wasm_bindgen(typescript_custom_section)]
const TS_MEDIA_SOURCE_ALIAS: &str = r#"
/**
 * Source of a media payload (image, audio, video, or file). Re-exported as an
 * alias of `ImageSource`; the same `Url`/`Base64` shape is used across all
 * modalities.
 */
export type MediaSource = ImageSource;
"#;

/// Initialise the WASM module.
///
/// Sets the panic hook so that Rust panics produce readable console
/// error messages instead of `unreachable` traps.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
