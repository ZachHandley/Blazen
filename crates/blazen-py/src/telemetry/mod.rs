//! Python bindings for the `blazen-telemetry` crate.

pub mod history;
pub mod tracing_model;

#[cfg(feature = "otlp")]
pub mod otlp;

#[cfg(feature = "prometheus")]
pub mod prometheus;

pub use history::{PyHistoryEvent, PyHistoryEventKind, PyPauseReason, PyWorkflowHistory};
pub use tracing_model::wrap_with_tracing;

#[cfg(feature = "otlp")]
pub use otlp::{PyOtlpConfig, init_otlp};

#[cfg(feature = "prometheus")]
pub use prometheus::init_prometheus;
