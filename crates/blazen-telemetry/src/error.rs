//! Error types for the `blazen-telemetry` crate.

use std::fmt;

/// Errors that can occur while initializing or operating telemetry exporters.
#[derive(Debug)]
pub enum TelemetryError {
    /// The shared global `tracing` subscriber could not be installed because a
    /// foreign subscriber (e.g. host application) already claimed the global
    /// dispatcher.
    SubscriberAlreadyInstalled,
    /// A reload-handle swap was attempted before
    /// [`crate::subscriber::install_global_subscriber`] had stashed a handle —
    /// i.e. the host owns the subscriber and our reload slot does not exist.
    NoReloadHandle,
    /// The reload-handle `modify` call returned an error (handle dropped).
    ReloadHandle(String),
    /// A Langfuse exporter failed to initialize or operate.
    #[cfg(feature = "langfuse")]
    Langfuse(String),
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SubscriberAlreadyInstalled => {
                f.write_str("a global tracing subscriber is already installed")
            }
            Self::NoReloadHandle => f.write_str(
                "blazen-telemetry's reload handle is not installed; call install_global_subscriber first",
            ),
            Self::ReloadHandle(msg) => write!(f, "reload handle error: {msg}"),
            #[cfg(feature = "langfuse")]
            Self::Langfuse(msg) => write!(f, "langfuse error: {msg}"),
        }
    }
}

impl std::error::Error for TelemetryError {}
