//! `Blazen` Node.js bindings.
//!
//! Exposes the `Blazen` framework to Node.js / TypeScript via napi-rs.
//!
//! # Modules
//!
//! - [`types`] -- Shared type definitions (messages, completions, tools, media).
//! - [`batch`] -- Batch completion with bounded concurrency.
//! - [`compute`] -- Compute request, result, and job types.
//! - [`providers`] -- LLM completion model wrappers and provider factories.
//! - [`error`] -- Error conversion utilities.
//! - [`agent`] -- Agentic tool execution loop bindings.
//! - [`peer`] -- Distributed peer gRPC server and client bindings.
//! - [`workflow`] -- Workflow builder, runner, context, handler, and events.

pub mod agent;
pub mod batch;
pub mod compute;
pub mod core;
pub mod error;
pub mod generated;
pub mod manager;
pub mod model_cache;
pub mod peer;
pub mod persist;
pub mod pipeline;
pub mod providers;
pub mod telemetry;
pub mod types;
pub mod workflow;

use napi_derive::napi;

/// Initialize the Rust `tracing` subscriber once when the native `.node`
/// module is first loaded into Node.js.
///
/// Without this, every `tracing::debug!` / `info!` / `warn!` call made by
/// the underlying Rust crates is silently dropped because no subscriber is
/// listening — setting `RUST_LOG` has no effect. `try_init` is a no-op if a
/// subscriber is already installed (e.g. by a host embedder), so this is
/// safe to call unconditionally. The filter honors `RUST_LOG` and defaults
/// to `warn` when unset. Output goes to stderr so `node:test` passes it
/// through without mixing into captured stdout.
#[napi_derive::module_init]
fn init() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();
}

/// Returns the version of the blazen library.
#[napi]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
