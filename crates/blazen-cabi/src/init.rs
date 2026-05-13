//! Process-level init / shutdown for the cabi surface.
//!
//! `blazen_init` warms up the cabi tokio runtime and delegates to
//! `blazen_uniffi::init` so the UniFFI-managed runtime and the tracing
//! subscriber are also primed. `blazen_shutdown` flushes telemetry
//! exporters via `blazen_uniffi::shutdown_telemetry`.
//!
//! Both calls are idempotent — FFI hosts typically invoke `blazen_init`
//! once at process start and `blazen_shutdown` once at process exit, but
//! repeat calls are safe.

use crate::runtime::runtime;

/// Initialises the blazen-cabi runtime: builds the cabi tokio runtime if it
/// hasn't been built yet, and delegates to `blazen_uniffi::init` to warm the
/// UniFFI-managed runtime + install a default tracing subscriber when none
/// is already set globally.
///
/// Idempotent — safe to call multiple times from any thread. Returns `0`
/// on success. No failure modes today, but the return slot is reserved so
/// future fallible initialisation can surface a non-zero status without an
/// ABI break.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_init() -> i32 {
    // Force-build the cabi tokio runtime so the first real async call doesn't
    // pay the build latency.
    let _ = runtime();
    // Mirror the same warm-up on the blazen-uniffi side (its runtime is
    // distinct from ours — see `crates/blazen-cabi/src/runtime.rs`).
    blazen_uniffi::init();
    0
}

/// Flushes telemetry exporters and shuts down background tasks owned by the
/// blazen telemetry stack. Idempotent — safe to call multiple times and from
/// any thread. Returns `0` on success.
///
/// This does NOT tear down the tokio runtime — runtimes are leaked at
/// process exit on purpose (matching the rest of the workspace). FFI hosts
/// only need to call this to ensure traces / metrics / langfuse spans get
/// drained before the process dies.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_shutdown() -> i32 {
    blazen_uniffi::shutdown_telemetry();
    0
}
