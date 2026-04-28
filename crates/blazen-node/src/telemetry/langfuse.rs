//! Node binding for the Langfuse trace exporter.
//!
//! Wraps [`blazen_telemetry::init_langfuse`], which builds a
//! [`blazen_telemetry::LangfuseLayer`] that is composed into a
//! `tracing_subscriber::Registry` and installed as the global subscriber.
//!
//! ```javascript
//! const { LangfuseConfig, initLangfuse } = require("blazen");
//! const cfg = new LangfuseConfig("pk-lf-...", "sk-lf-...", undefined, 50, 2500);
//! initLangfuse(cfg);
//! ```
//!
//! Calling `initLangfuse` more than once in a single process will not panic;
//! the second call's subscriber install is a no-op because a global
//! subscriber is already in place. The Langfuse layer from the first call
//! remains active.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use blazen_telemetry::{LangfuseConfig, init_langfuse as rust_init_langfuse};

/// Configuration for the Langfuse exporter.
///
/// Wraps [`blazen_telemetry::LangfuseConfig`]. Construct with the public and
/// secret API keys; optionally override host, batch size, and flush interval.
#[napi(js_name = "LangfuseConfig")]
pub struct JsLangfuseConfig {
    inner: LangfuseConfig,
}

#[napi]
impl JsLangfuseConfig {
    /// Create a new Langfuse configuration.
    ///
    /// `public_key` / `secret_key` are required. `host` defaults to
    /// `https://cloud.langfuse.com`. `batch_size` defaults to 100 events.
    /// `flush_interval_ms` defaults to 5000 ms.
    #[napi(constructor)]
    #[must_use]
    pub fn new(
        public_key: String,
        secret_key: String,
        host: Option<String>,
        batch_size: Option<u32>,
        flush_interval_ms: Option<u32>,
    ) -> Self {
        let mut cfg = LangfuseConfig::new(public_key, secret_key);
        if let Some(h) = host {
            cfg = cfg.with_host(h);
        }
        if let Some(b) = batch_size {
            cfg = cfg.with_batch_size(b as usize);
        }
        if let Some(i) = flush_interval_ms {
            cfg = cfg.with_flush_interval_ms(u64::from(i));
        }
        Self { inner: cfg }
    }

    /// The Langfuse public API key.
    #[napi(getter, js_name = "publicKey")]
    #[must_use]
    pub fn public_key(&self) -> &str {
        &self.inner.public_key
    }

    /// The Langfuse secret API key.
    #[napi(getter, js_name = "secretKey")]
    #[must_use]
    pub fn secret_key(&self) -> &str {
        &self.inner.secret_key
    }

    /// The configured Langfuse host URL, or `null` when defaulted.
    #[napi(getter)]
    #[must_use]
    pub fn host(&self) -> Option<&str> {
        self.inner.host.as_deref()
    }

    /// Maximum number of events buffered before an automatic flush.
    /// Clamped to `u32::MAX` if the underlying `usize` exceeds it.
    #[napi(getter, js_name = "batchSize")]
    #[must_use]
    pub fn batch_size(&self) -> u32 {
        u32::try_from(self.inner.batch_size).unwrap_or(u32::MAX)
    }

    /// Background flush interval in milliseconds.
    /// Clamped to `u32::MAX` if the underlying `u64` exceeds it.
    #[napi(getter, js_name = "flushIntervalMs")]
    #[must_use]
    pub fn flush_interval_ms(&self) -> u32 {
        u32::try_from(self.inner.flush_interval_ms).unwrap_or(u32::MAX)
    }
}

/// Initialize the Langfuse exporter and install it as a layer on the global
/// `tracing` subscriber.
///
/// A background tokio task is spawned to periodically flush buffered span
/// envelopes to the Langfuse ingestion API; this requires an active tokio
/// runtime (always true under napi-rs).
///
/// Calling this more than once in a single process is safe: the second
/// install is a no-op because a global subscriber is already registered.
#[napi(js_name = "initLangfuse")]
#[allow(clippy::missing_errors_doc)]
pub fn init_langfuse(config: &JsLangfuseConfig) -> Result<()> {
    let layer = rust_init_langfuse(config.inner.clone()).map_err(|e| {
        napi::Error::new(napi::Status::GenericFailure, format!("[BlazenError] {e}"))
    })?;
    // `try_init` returns Err if a global subscriber is already installed
    // (e.g. by `module_init` in lib.rs). That is a benign no-op here -- we
    // still want the call to succeed so JS callers can configure Langfuse
    // unconditionally.
    let _ = tracing_subscriber::registry().with(layer).try_init();
    Ok(())
}
