//! Telemetry bindings for the WASM SDK.
//!
//! Bound features:
//! - `history`: Append-only workflow event history.
//! - `otlp-http`: OpenTelemetry OTLP export over HTTP/protobuf.
//! - `langfuse`: Langfuse trace exporter via reqwest's wasm32 fetch backend.
//! - `spans` (TracingConfig): the privacy-capture config for tracing-wrapped
//!   models. The `TracingModel` wrapper itself is surfaced on the native
//!   bindings via `wrap_with_tracing`; on wasm32 the config object feeds the
//!   OTLP / Langfuse export lanes above.
//!
//! Out of scope on wasm32:
//! - `otlp` (gRPC variant) -- tonic + h2 sockets do not compile to wasm32;
//!   the `otlp-http` variant above is the wasm-friendly equivalent.
//! - `prometheus` -- the metrics-exporter-prometheus pull collector binds a
//!   TCP listener which wasm32 cannot open.

pub mod history;
pub mod langfuse;
pub mod otlp;
pub mod tracing;
