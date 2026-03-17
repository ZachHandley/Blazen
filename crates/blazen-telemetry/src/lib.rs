//! Observability and telemetry for Blazen workflows.
//!
//! Feature flags:
//! - `spans` (default): `TracingCompletionModel` wrapper
//! - `history`: Append-only workflow event history
//! - `otlp`: OpenTelemetry OTLP export
//! - `prometheus`: Prometheus metrics endpoint
//! - `langfuse`: Langfuse trace export

#[cfg(feature = "spans")]
pub mod spans;

#[cfg(feature = "history")]
pub mod history;

#[cfg(any(feature = "otlp", feature = "prometheus", feature = "langfuse"))]
pub mod exporters;

#[cfg(feature = "prometheus")]
pub mod metrics;

// Re-exports
#[cfg(feature = "spans")]
pub use spans::TracingCompletionModel;

#[cfg(feature = "history")]
pub use history::{HistoryEvent, HistoryEventKind, PauseReason, WorkflowHistory};

#[cfg(feature = "otlp")]
pub use exporters::otlp::{OtlpConfig, init_otlp};

#[cfg(feature = "langfuse")]
pub use exporters::langfuse::{LangfuseConfig, LangfuseLayer, init_langfuse};

#[cfg(feature = "prometheus")]
pub use exporters::prometheus::init_prometheus;

#[cfg(feature = "prometheus")]
pub use metrics::MetricsLayer;
