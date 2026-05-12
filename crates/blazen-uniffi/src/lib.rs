//! `blazen-uniffi` — multi-language bindings for Blazen via Mozilla UniFFI.
//!
//! This crate is the source of truth for Blazen's Go, Swift, Kotlin, and Ruby
//! bindings. A single set of `#[uniffi::export]` annotations on the public
//! surface here drives four foreign-language bindgens:
//!
//! - Go     via NordSecurity/uniffi-bindgen-go      → `bindings/go/`
//! - Swift  via mozilla/uniffi-rs's swift bindgen   → `bindings/swift/`
//! - Kotlin via mozilla/uniffi-rs's kotlin bindgen  → `bindings/kotlin/`
//! - Ruby   via mozilla/uniffi-rs's ruby bindgen    → `bindings/ruby/`
//!
//! Python is intentionally NOT covered here — Blazen ships `blazen-py` via
//! PyO3 which is more mature and idiomatic than UniFFI's Python output.
//!
//! Async Rust functions are exposed as language-native async (Swift
//! `async`/`await`, Kotlin `suspend fun`) or as blocking calls that compose
//! naturally with the host runtime (Go goroutines, Ruby fibers). Tokio runs
//! invisibly underneath via `uniffi`'s `tokio` feature.

#![allow(unsafe_code)] // UniFFI scaffolding contains generated `extern "C"` thunks.
#![allow(
    clippy::all, // UniFFI-generated scaffolding doesn't pass workspace lints.
    clippy::pedantic,
    dead_code
)]

uniffi::include_scaffolding!("blazen");

pub mod errors;
pub mod llm;
pub mod pipeline;
pub mod runtime;
pub mod workflow;

pub use errors::{BlazenError, BlazenResult};
pub use llm::{
    ChatMessage, CompletionModel, CompletionRequest, CompletionResponse, EmbeddingModel,
    EmbeddingResponse, Media, TokenUsage, Tool, ToolCall,
};
pub use pipeline::{Pipeline, PipelineBuilder};
pub use runtime::init;
pub use workflow::{Event, StepHandler, StepOutput, Workflow, WorkflowBuilder, WorkflowResult};

/// Returns the `blazen-uniffi` crate version baked in at compile time.
///
/// Exposed in the UDL as `string version()` so every foreign binding has a
/// stable way to query the underlying native lib version (useful for
/// diagnosing version-skew issues between a Go/Swift/Kotlin/Ruby module
/// and its embedded native lib).
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
