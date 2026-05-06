//! `wasm-bindgen` wrappers for [`blazen_telemetry::LangfuseConfig`] and
//! [`blazen_telemetry::init_langfuse`].
//!
//! Exposes the Langfuse trace exporter to JS/TS callers. The HTTP transport
//! routes through `reqwest` (which on wasm32 uses the browser `fetch()` API
//! under the hood) and a `wasm_bindgen_futures::spawn_local`-backed
//! background dispatcher batches events before flushing. Per-event flushing
//! is available by passing `withBatchSize(1)`.
//!
//! ```js
//! import { LangfuseConfig, initLangfuse } from '@blazen-dev/sdk';
//!
//! const cfg = new LangfuseConfig('pk-lf-...', 'sk-lf-...')
//!   .withHost('https://cloud.langfuse.com')
//!   .withBatchSize(20);
//!
//! initLangfuse(cfg);  // installs a global tracing-subscriber stack
//! ```

use blazen_telemetry::LangfuseConfig;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use wasm_bindgen::prelude::*;

/// Configuration for the Langfuse trace exporter.
///
/// Wraps [`blazen_telemetry::LangfuseConfig`] for `wasm-bindgen` interop.
/// Exposes a builder-style mutator surface so JS callers can chain
/// configuration changes after construction.
#[wasm_bindgen(js_name = "LangfuseConfig")]
pub struct WasmLangfuseConfig {
    inner: LangfuseConfig,
}

#[wasm_bindgen(js_class = "LangfuseConfig")]
impl WasmLangfuseConfig {
    /// Construct a new Langfuse exporter configuration.
    ///
    /// `publicKey` and `secretKey` are the project credentials issued by
    /// the Langfuse dashboard; they are sent as the HTTP Basic-auth
    /// username and password on every ingestion request.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(public_key: String, secret_key: String) -> Self {
        Self {
            inner: LangfuseConfig::new(public_key, secret_key),
        }
    }

    /// The Langfuse public API key.
    #[wasm_bindgen(getter, js_name = "publicKey")]
    #[must_use]
    pub fn public_key(&self) -> String {
        self.inner.public_key.clone()
    }

    /// The Langfuse secret API key.
    #[wasm_bindgen(getter, js_name = "secretKey")]
    #[must_use]
    pub fn secret_key(&self) -> String {
        self.inner.secret_key.clone()
    }

    /// The configured Langfuse host URL, if overridden.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn host(&self) -> Option<String> {
        self.inner.host.clone()
    }

    /// Maximum batched events before an automatic flush.
    #[wasm_bindgen(getter, js_name = "batchSize")]
    #[must_use]
    pub fn batch_size(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.batch_size as u32
        }
    }

    /// Background flush interval in milliseconds.
    ///
    /// On wasm32 the dispatcher does not poll on a timer — it flushes
    /// strictly on `batch_size` and on channel close — so this value
    /// is recorded for parity with the native bindings but does not
    /// affect runtime behaviour. Set `batch_size` to 1 for per-event
    /// flushing in the browser.
    #[wasm_bindgen(getter, js_name = "flushIntervalMs")]
    #[must_use]
    pub fn flush_interval_ms(&self) -> u64 {
        self.inner.flush_interval_ms
    }

    /// Override the Langfuse host URL (defaults to
    /// `https://cloud.langfuse.com`).
    #[wasm_bindgen(js_name = "withHost")]
    #[must_use]
    pub fn with_host(mut self, host: String) -> Self {
        self.inner.host = Some(host);
        self
    }

    /// Override the maximum batch size before an automatic flush.
    #[wasm_bindgen(js_name = "withBatchSize")]
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.inner.batch_size = batch_size as usize;
        self
    }

    /// Override the background flush interval (in milliseconds). Native
    /// only; ignored on wasm32 (see `flushIntervalMs` getter doc).
    #[wasm_bindgen(js_name = "withFlushIntervalMs")]
    #[must_use]
    pub fn with_flush_interval_ms(mut self, flush_interval_ms: u64) -> Self {
        self.inner.flush_interval_ms = flush_interval_ms;
        self
    }
}

/// Initialise the global Langfuse trace exporter.
///
/// Builds a [`blazen_telemetry::LangfuseLayer`] from the configuration and
/// installs it on a fresh `tracing_subscriber::Registry` as the global
/// subscriber. Must be called once at startup; subsequent calls will fail
/// because `tracing_subscriber::Registry::init` can only be invoked once.
///
/// # Errors
///
/// Returns a JS error if the underlying HTTP client cannot be constructed
/// or if a global subscriber has already been installed.
#[wasm_bindgen(js_name = "initLangfuse")]
pub fn init_langfuse(config: &WasmLangfuseConfig) -> Result<(), JsValue> {
    let layer = blazen_telemetry::init_langfuse(config.inner.clone())
        .map_err(|e| JsValue::from_str(&format!("[BlazenError] init_langfuse: {e}")))?;
    tracing_subscriber::registry()
        .with(layer)
        .try_init()
        .map_err(|e| JsValue::from_str(&format!("[BlazenError] init_langfuse: {e}")))
}
