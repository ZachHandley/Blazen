//! Telemetry bindings for the WASM SDK.
//!
//! Bound features:
//! - `history`: Append-only workflow event history.
//! - `otlp-http`: OpenTelemetry OTLP export over HTTP/protobuf.
//!
//! Other features (`spans` (TracingCompletionModel), `otlp` (gRPC),
//! `prometheus`, `langfuse`) require native dependencies that do not
//! compile to `wasm32` and are therefore out of scope for the WASM SDK.

pub mod history;
pub mod otlp;
