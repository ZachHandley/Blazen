//! Tokio runtime glue for the hand-rolled C ABI.
//!
//! Mirrors `blazen-uniffi::runtime` but stays deliberately separate: the C ABI
//! is its own surface and shouldn't share a runtime with the `UniFFI` bindings
//! (we want worker-thread names to reflect which entry point spawned the
//! task, and the two crates may have differing future lifecycle semantics).
//!
//! Subsequent phases use this runtime in two ways:
//!
//! - **Sync wrappers** (`*_blocking` extern fns) call
//!   `runtime().block_on(async { ... })` to bridge async-internal/sync-FFI.
//! - **Future-returning wrappers** call `runtime().spawn(...)` and hand back
//!   a `BlazenFuture*` handle whose pipe read-fd unblocks once the task
//!   completes, suitable for `Fiber.scheduler`-aware Ruby consumers.
//!
//! Cancellation is intentionally not exposed here as a global concept —
//! per-call cancellation flows through the future handle in later phases.

// Foundation utility: callers land in Phase R3+ when sync/async wrappers are
// implemented. Phase R1 lays the runtime down ahead of those wrappers.
#![allow(dead_code)]

use std::sync::OnceLock;

use tokio::runtime::{Builder, Runtime};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Returns the shared Tokio runtime, building it on first call.
///
/// The runtime is multi-threaded with all features enabled (IO, time, signal)
/// and uses the thread-name prefix `"blazen-cabi"` so worker threads are
/// distinguishable from `blazen-uniffi`'s pool in process diagnostics.
///
/// Worker thread count is left at Tokio's default (= number of CPU cores),
/// matching `blazen-uniffi::runtime`.
pub(crate) fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .enable_all()
            .thread_name("blazen-cabi")
            .build()
            .expect("failed to build Tokio runtime for blazen-cabi")
    })
}
