//! Process-wide tracing subscriber with a reload-handle slot for exporter
//! layers (OTLP, Langfuse, ...).
//!
//! ## Why this exists
//!
//! `tracing_subscriber::Subscriber` can only be installed as the *global
//! default* once per process — a second `set_global_default` panics. The
//! Python / Node / `UniFFI` bindings install a small `fmt` subscriber at
//! module-load time so users see warn-level logs out of the box. If
//! [`crate::init_otlp`] then tried to install its own subscriber later
//! (the original implementation called `tracing_subscriber::...init()`,
//! the unwrap-or-panic variant), the second install would panic on a
//! tokio worker and trip an uncatchable SIGABRT in Python.
//!
//! The fix is the standard *reload-layer* pattern: the bindings install
//! ONE subscriber at startup that already carries an empty
//! [`reload::Layer`] slot, and [`crate::init_otlp`] / `init_langfuse`
//! swap their exporter `Layer` into that slot via the cached
//! [`reload::Handle`]. No second `set_global_default`, no panic, fully
//! idempotent.
//!
//! If the host application owns its own subscriber (no
//! [`install_global_subscriber`] call from Blazen), the OTLP / langfuse
//! init functions fall back to `try_init` and propagate `Err` — they
//! never panic.

use std::sync::OnceLock;

use tracing_subscriber::{
    EnvFilter, Layer, fmt, layer::SubscriberExt, registry::Registry, reload,
    util::SubscriberInitExt,
};

use crate::error::TelemetryError;

/// The type-erased exporter slot driven by the reload handle.
///
/// `None` is the no-op state (events pass through to the rest of the
/// subscriber and to `Registry`). `Some(layer)` activates the exporter.
type LayerSlot = Option<Box<dyn Layer<Registry> + Send + Sync + 'static>>;

/// Process-wide reload handle, set exactly once by
/// [`install_global_subscriber`]. Subsequent installer calls observe the
/// existing handle and become no-ops.
static RELOAD_HANDLE: OnceLock<reload::Handle<LayerSlot, Registry>> = OnceLock::new();

/// Install the shared global `tracing` subscriber for Blazen bindings.
///
/// The composed subscriber is:
///
/// ```text
/// Registry
///   + reload::Layer<Option<Box<dyn Layer<Registry>>>>   <- exporter slot
///   + EnvFilter (RUST_LOG, default = "warn")
///   + fmt::Layer (writes to stderr)
/// ```
///
/// Calling this function more than once is safe: the second call is a
/// no-op and returns `Ok(())`. If the host application has already
/// installed its own global subscriber, this returns
/// [`TelemetryError::SubscriberAlreadyInstalled`] — the reload handle is
/// NOT stashed in that case, which causes [`swap_exporter_layer`] to
/// return [`TelemetryError::NoReloadHandle`] and the exporter falls back
/// to its host-friendly `try_init` path.
///
/// # Errors
///
/// Returns [`TelemetryError::SubscriberAlreadyInstalled`] when a foreign
/// global subscriber is already in place.
pub fn install_global_subscriber() -> Result<(), TelemetryError> {
    if RELOAD_HANDLE.get().is_some() {
        return Ok(());
    }

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    let fmt_layer = fmt::layer().with_writer(std::io::stderr);

    let initial: LayerSlot = None;
    let (reload_layer, handle) = reload::Layer::new(initial);

    let subscriber = tracing_subscriber::registry()
        .with(reload_layer)
        .with(env_filter)
        .with(fmt_layer);

    match subscriber.try_init() {
        Ok(()) => {
            // The handle is only useful if the matching subscriber is live,
            // so we only stash it after a successful install.
            let _ = RELOAD_HANDLE.set(handle);
            Ok(())
        }
        Err(_) => Err(TelemetryError::SubscriberAlreadyInstalled),
    }
}

/// Swap a new exporter `Layer` into the reload slot.
///
/// Called by [`crate::init_otlp`] (via `install_provider`) and by the
/// langfuse `init_langfuse_global` path. The slot accepts any
/// `Layer<Registry> + Send + Sync + 'static`; pass `layer.boxed()` to
/// satisfy the type.
///
/// # Errors
///
/// Returns [`TelemetryError::NoReloadHandle`] when
/// [`install_global_subscriber`] has not run (the host owns the
/// subscriber). The caller should fall back to `try_init` on its own
/// subscriber stack so the exporter still gets installed in non-Blazen
/// hosts.
pub fn swap_exporter_layer(
    layer: Box<dyn Layer<Registry> + Send + Sync + 'static>,
) -> Result<(), TelemetryError> {
    let handle = RELOAD_HANDLE.get().ok_or(TelemetryError::NoReloadHandle)?;

    handle
        .modify(|slot| {
            *slot = Some(layer);
        })
        .map_err(|e| TelemetryError::ReloadHandle(format!("{e}")))
}

/// Returns `true` if [`install_global_subscriber`] has previously
/// completed in this process. Useful for tests + assertions.
#[must_use]
pub fn has_reload_handle() -> bool {
    RELOAD_HANDLE.get().is_some()
}
