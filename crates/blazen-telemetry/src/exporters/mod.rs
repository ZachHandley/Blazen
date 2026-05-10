//! Telemetry exporters for shipping traces and metrics to external systems.

#[cfg(any(feature = "otlp", feature = "otlp-http"))]
pub mod otlp;

// Browser-only fetch-backed `opentelemetry_http::HttpClient`. The reqwest impls
// shipped by `opentelemetry-http` are `!Send` on wasm32 (Rc<RefCell<JsFuture>>)
// which violates the trait's `Send + Sync` bound; this module supplies a
// drop-in replacement using `web_sys::fetch`. Excluded on wasi (no DOM).
#[cfg(all(feature = "otlp-http", target_arch = "wasm32", not(target_os = "wasi")))]
pub mod wasm_otlp_client;

// Wasi OTLP HTTP transport: implements `opentelemetry_http::HttpClient` over
// `Arc<dyn blazen_llm::http::HttpClient>` so Cloudflare Workers / Deno can
// export traces without `reqwest` (no socket access on wasi) or `web_sys`
// (no DOM bindings on wasi).
#[cfg(all(feature = "otlp-http", target_arch = "wasm32", target_os = "wasi"))]
pub mod wasi_otlp_client;

#[cfg(feature = "langfuse")]
pub mod langfuse;

// Wasi Langfuse exporter: routes ingestion batches through
// `Arc<dyn blazen_llm::http::HttpClient>` instead of `reqwest`.
#[cfg(all(feature = "langfuse", target_arch = "wasm32", target_os = "wasi"))]
pub mod wasi_langfuse_client;

#[cfg(feature = "prometheus")]
pub mod prometheus;
