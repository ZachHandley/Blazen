//! `wasm-bindgen` wrapper for [`blazen_telemetry::TracingConfig`].
//!
//! Exposes the runtime configuration that controls whether a tracing-wrapped
//! model captures raw prompt + completion message text as span attributes.
//! The wrapper itself ([`blazen_telemetry::TracingModel`]) is surfaced on the
//! native bindings via the `wrap_with_tracing` free function; on `wasm32`
//! callers configure capture through this config object, which feeds the
//! OTLP / Langfuse export lanes already bound in this crate.
//!
//! ```js
//! import { TracingConfig } from '@blazen-dev/wasm';
//!
//! const cfg = new TracingConfig();
//! cfg.withMessageCapture(true); // opt into raw message capture
//! ```

use blazen_telemetry::TracingConfig;
use wasm_bindgen::prelude::*;

/// Runtime configuration for tracing-instrumented models.
///
/// Wraps [`blazen_telemetry::TracingConfig`]. Defaults are privacy-safe:
/// token counts, model id, provider, and finish reason are always recorded,
/// but raw prompt + completion message text is NOT recorded unless
/// [`with_message_capture`](WasmTracingConfig::with_message_capture) is opted
/// into.
#[wasm_bindgen(js_name = "TracingConfig")]
#[derive(Debug, Clone, Copy, Default)]
pub struct WasmTracingConfig {
    inner: TracingConfig,
}

#[wasm_bindgen(js_class = "TracingConfig")]
impl WasmTracingConfig {
    /// Create a new tracing config with privacy-safe defaults (no raw
    /// message capture).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: TracingConfig::default(),
        }
    }

    /// Enable or disable raw prompt + completion message capture as span
    /// attributes (`llm.input_messages` / `llm.output_messages`).
    ///
    /// When enabled, the tracing wrapper serializes the incoming request
    /// messages and the outgoing completion content to JSON and attaches them
    /// to the span — what Phoenix's eval-grade surfaces need. Privacy-sensitive
    /// deployments should leave this off (the default).
    #[wasm_bindgen(js_name = "withMessageCapture")]
    pub fn with_message_capture(&mut self, capture: bool) {
        self.inner = self.inner.with_message_capture(capture);
    }

    /// Whether raw messages are captured.
    #[wasm_bindgen(getter, js_name = "captureMessages")]
    #[must_use]
    pub fn capture_messages(&self) -> bool {
        self.inner.capture_messages()
    }
}
