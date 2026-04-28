//! Telemetry exporters for shipping traces and metrics to external systems.

#[cfg(any(feature = "otlp", feature = "otlp-http"))]
pub mod otlp;

// wasm32-only fetch-backed `opentelemetry_http::HttpClient`. The reqwest impls
// shipped by `opentelemetry-http` are `!Send` on wasm32 (Rc<RefCell<JsFuture>>)
// which violates the trait's `Send + Sync` bound; this module supplies a
// drop-in replacement.
#[cfg(all(feature = "otlp-http", target_arch = "wasm32"))]
pub mod wasm_otlp_client;

#[cfg(feature = "langfuse")]
pub mod langfuse;

#[cfg(feature = "prometheus")]
pub mod prometheus;
