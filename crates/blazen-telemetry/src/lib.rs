//! Observability and telemetry for Blazen workflows.
//!
//! Feature flags:
//! - `spans` (default): `TracingModel` wrapper
//! - `history`: Append-only workflow event history
//! - `otlp`: OpenTelemetry OTLP export over gRPC (tonic)
//! - `otlp-http`: OpenTelemetry OTLP export over HTTP (binary protobuf)
//! - `prometheus`: Prometheus metrics endpoint
//! - `langfuse`: Langfuse trace export

#[cfg(feature = "spans")]
pub mod spans;

#[cfg(feature = "history")]
pub mod history;

#[cfg(any(
    feature = "otlp",
    feature = "otlp-http",
    feature = "prometheus",
    feature = "langfuse"
))]
pub mod exporters;

#[cfg(feature = "prometheus")]
pub mod metrics;

pub mod error;

pub mod subscriber;

// Re-exports
#[cfg(feature = "spans")]
pub use spans::{TracingConfig, TracingModel};

pub use error::TelemetryError;

pub use subscriber::{install_global_subscriber, swap_exporter_layer};

#[cfg(feature = "history")]
pub use history::{HistoryEvent, HistoryEventKind, PauseReason, WorkflowHistory};

#[cfg(any(feature = "otlp", feature = "otlp-http"))]
pub use exporters::otlp::{OtlpConfig, OtlpProtocol, init_otlp};

#[cfg(feature = "otlp-http")]
pub use exporters::otlp::init_otlp_http;

#[cfg(feature = "langfuse")]
pub use exporters::langfuse::{LangfuseConfig, LangfuseLayer, init_langfuse, init_langfuse_global};

#[cfg(feature = "prometheus")]
pub use exporters::prometheus::init_prometheus;

#[cfg(feature = "prometheus")]
pub use metrics::MetricsLayer;
