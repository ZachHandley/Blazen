//! Telemetry bindings for the WASM SDK.
//!
//! Bound features:
//! - `history`: Append-only workflow event history.
//! - `otlp-http`: OpenTelemetry OTLP export over HTTP/protobuf.
//! - `langfuse`: Langfuse trace exporter via reqwest's wasm32 fetch backend.
//!
//! Out of scope on wasm32:
//! - `spans` (TracingCompletionModel) -- pulls in additional tracing
//!   subscriber machinery not currently bound here.
//! - `otlp` (gRPC variant) -- tonic + h2 sockets do not compile to wasm32;
//!   the `otlp-http` variant above is the wasm-friendly equivalent.
//! - `prometheus` -- the metrics-exporter-prometheus pull collector binds a
//!   TCP listener which wasm32 cannot open.

pub mod history;
pub mod langfuse;
pub mod otlp;
