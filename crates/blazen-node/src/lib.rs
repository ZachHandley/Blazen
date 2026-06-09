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
//! - [`controlplane`] -- Control-plane orchestrator client and worker bindings (native targets only).
//! - [`workflow`] -- Workflow builder, runner, context, handler, and events.

pub mod agent;
pub mod batch;
pub mod compute;
pub mod content;
#[cfg(not(target_os = "wasi"))]
pub mod controlplane;
pub mod core;
pub mod error;
pub mod error_classes;
pub mod generated;
pub mod manager;
#[cfg(not(target_os = "wasi"))]
pub mod model_cache;
#[cfg(feature = "audio-music")]
pub mod music;
pub mod peer;
pub mod peer_http;
#[cfg(not(target_os = "wasi"))]
pub mod persist;
pub mod pipeline;
pub mod providers;
pub mod telemetry;
#[cfg(feature = "threed-compat-proxy")]
pub mod threed;
pub mod types;
#[cfg(feature = "audio-vc")]
pub mod vc;
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub mod wasi_async;
pub mod workflow;

use napi::bindgen_prelude::{Env, Object};
use napi_derive::napi;

/// Initialize Blazen's shared global tracing subscriber when the native
/// `.node` module is first loaded into Node.js.
///
/// This installs a registry-backed subscriber that already carries an
/// empty reload-handle slot, so later `init_otlp` / `init_langfuse`
/// calls can swap their exporter Layers in without a second
/// `set_global_default` (which would panic).
///
/// Idempotent and panic-free: if a host embedder has already installed
/// a subscriber, the installer returns Err and exporter calls fall back
/// to their own host-friendly `try_init` paths.
#[napi_derive::module_init]
fn init() {
    let _ = blazen_telemetry::install_global_subscriber();

    // Install the native-event serializer hook so DynamicEvents lazily
    // materialize to JSON via the registered fallback. First registration
    // wins; subsequent calls are no-ops.
    crate::workflow::native_passthrough::install_native_serializer();
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
fn module_exports(exports: Object, env: Env) -> napi::Result<()> {
    // Register every JS `Error` subclass Blazen exposes (BlazenError +
    // ProviderError + per-backend subclasses + Peer/Persist/Cache/Prompt/
    // Memory error families) AND bind each class onto `exports` so JS
    // callers can do `require('blazen').ProviderError` and use `instanceof`
    // against the same class object the runtime throws as. Rust code emits
    // typed throws via
    // `napi::Error::with_class("ProviderError", reason).with_field(...)`.
    error_classes::register_all_classes(&env, &exports)?;

    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    wasi_async::install(&env)?;
    Ok(())
}
