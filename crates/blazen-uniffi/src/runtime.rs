//! Tokio runtime glue for the UniFFI bindings.
//!
//! UniFFI's `tokio` feature handles async-fn execution transparently via a
//! shared multi-thread runtime created on first use. This module exposes a
//! small surface for foreign-language consumers to:
//!
//! - Eagerly warm up the runtime (avoid pause-on-first-call latency).
//! - Initialise tracing exactly once (so log lines come out the same way
//!   regardless of which binding triggered the first call).
//! - Read the crate version (handy for diagnosing version-skew between a
//!   Go/Swift/Kotlin/Ruby module and the embedded native lib).
//!
//! Cancellation is intentionally *not* exposed here as a global concept —
//! UniFFI cancellation lives on individual async calls via the foreign
//! runtime's native cancellation primitive (`context.Context` cancel,
//! `Task.cancel()`, `CoroutineScope.cancel()`, Ruby fiber abort).

use std::sync::OnceLock;
use tokio::runtime::{Builder, Runtime};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();
static TRACING_INIT: OnceLock<()> = OnceLock::new();

/// Returns the shared Tokio runtime, building it on first call.
///
/// Used internally by hand-rolled blocking shims (where the UniFFI-managed
/// runtime isn't reachable). Foreign callers don't see this directly.
pub(crate) fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .enable_all()
            .thread_name("blazen-uniffi")
            .build()
            .expect("failed to build Tokio runtime for blazen-uniffi")
    })
}

/// Eagerly initialise the Tokio runtime and tracing subscriber.
///
/// Safe to call multiple times — both initialisations are idempotent.
/// Foreign callers typically invoke this once at app startup
/// (`blazen.Init()` in Go, `Blazen.initialize()` in Swift, etc.) so the
/// first real async call doesn't pay runtime-build latency.
#[uniffi::export]
pub fn init() {
    let _ = runtime();
    TRACING_INIT.get_or_init(|| {
        // Best-effort. If a subscriber is already set globally (e.g. by an
        // embedding application), silently leave it in place rather than
        // panicking — UniFFI bindings can be loaded into hosts that already
        // configured tracing.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            )
            .try_init();
    });
}
