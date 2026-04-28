//! Telemetry bindings for the WASM SDK.
//!
//! Only the `history` feature of `blazen-telemetry` is bound here.
//! Other features (`spans`, `otlp`, `prometheus`, `langfuse`) require
//! native dependencies that do not compile to `wasm32` and are therefore
//! out of scope for the WASM SDK.

pub mod history;
