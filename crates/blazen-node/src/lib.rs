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
//! - [`peer`] -- Distributed peer gRPC server and client bindings (native targets only).
//! - [`peer_http`] -- Wasi-compatible HTTP/JSON peer client (and native fallback). Available on every target.
//! - [`workflow`] -- Workflow builder, runner, context, handler, and events.

pub mod agent;
pub mod batch;
pub mod compute;
pub mod content;
pub mod core;
pub mod error;
pub mod generated;
pub mod manager;
#[cfg(not(target_os = "wasi"))]
pub mod model_cache;
pub mod peer;
pub mod peer_http;
#[cfg(not(target_os = "wasi"))]
pub mod persist;
pub mod pipeline;
pub mod providers;
pub mod telemetry;
pub mod types;
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub mod wasi_async;
pub mod workflow;

use napi::bindgen_prelude::{Env, Object};
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

/// Module-level export hook invoked by napi-rs after the native module's
/// `exports` object has been populated.
///
/// Used on the wasi target to install the JS-microtask-based async
/// dispatcher (see [`wasi_async`]) into `blazen_core::runtime` so every
/// `runtime::spawn` polls through the JS event loop instead of tokio's
/// `std::thread::spawn` driver — which workerd's single-isolate WASI
/// runtime forbids. On native targets this hook is a no-op so tokio's
/// default driver remains in charge.
#[napi(module_exports)]
#[allow(
    dead_code,
    clippy::needless_pass_by_value,
    clippy::unnecessary_wraps,
    clippy::missing_const_for_fn
)]
fn module_exports(_exports: Object, env: Env) -> napi::Result<()> {
    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    wasi_async::install(&env)?;
    let _ = env;
    Ok(())
}
