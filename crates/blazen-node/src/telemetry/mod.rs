//! Node bindings for the `blazen-telemetry` crate.

pub mod history;

#[cfg(feature = "otlp")]
pub mod otlp;

#[cfg(feature = "prometheus")]
pub mod prometheus;

pub mod tracing_model;

pub use history::{JsHistoryEvent, JsHistoryEventKind, JsPauseReason, JsWorkflowHistory};

#[cfg(feature = "otlp")]
pub use otlp::{JsOtlpConfig, init_otlp};

#[cfg(feature = "prometheus")]
pub use prometheus::init_prometheus;
