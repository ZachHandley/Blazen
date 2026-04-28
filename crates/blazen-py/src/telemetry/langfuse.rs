//! Python binding for `blazen_telemetry::LangfuseConfig` and `init_langfuse`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use blazen_telemetry::{LangfuseConfig, init_langfuse as rust_init_langfuse};

use crate::error::BlazenPyError;

// ---------------------------------------------------------------------------
// PyLangfuseConfig
// ---------------------------------------------------------------------------

/// Configuration for the Langfuse LLM observability exporter.
///
/// Example:
///     >>> from blazen import LangfuseConfig, init_langfuse
///     >>> cfg = LangfuseConfig(
///     ...     public_key="pk-lf-...",
///     ...     secret_key="sk-lf-...",
///     ...     host="https://cloud.langfuse.com",
///     ...     batch_size=100,
///     ...     flush_interval_ms=5000,
///     ... )
///     >>> init_langfuse(cfg)
#[gen_stub_pyclass]
#[pyclass(name = "LangfuseConfig", from_py_object)]
#[derive(Clone)]
pub struct PyLangfuseConfig {
    pub(crate) inner: LangfuseConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLangfuseConfig {
    /// Create a new Langfuse configuration.
    ///
    /// Args:
    ///     public_key: Langfuse public API key (Basic-auth username).
    ///     secret_key: Langfuse secret API key (Basic-auth password).
    ///     host: Optional Langfuse host URL. Defaults to ``https://cloud.langfuse.com``.
    ///     batch_size: Maximum number of events buffered before an automatic flush.
    ///     flush_interval_ms: Background flush interval in milliseconds.
    #[new]
    #[pyo3(signature = (
        public_key,
        secret_key,
        host = None,
        batch_size = 100,
        flush_interval_ms = 5000,
    ))]
    fn new(
        public_key: String,
        secret_key: String,
        host: Option<String>,
        batch_size: usize,
        flush_interval_ms: u64,
    ) -> Self {
        let mut cfg = LangfuseConfig::new(public_key, secret_key);
        if let Some(h) = host {
            cfg = cfg.with_host(h);
        }
        cfg = cfg
            .with_batch_size(batch_size)
            .with_flush_interval_ms(flush_interval_ms);
        Self { inner: cfg }
    }

    #[getter]
    fn public_key(&self) -> String {
        self.inner.public_key.clone()
    }
    #[setter]
    fn set_public_key(&mut self, v: String) {
        self.inner.public_key = v;
    }

    #[getter]
    fn secret_key(&self) -> String {
        self.inner.secret_key.clone()
    }
    #[setter]
    fn set_secret_key(&mut self, v: String) {
        self.inner.secret_key = v;
    }

    #[getter]
    fn host(&self) -> Option<String> {
        self.inner.host.clone()
    }
    #[setter]
    fn set_host(&mut self, v: Option<String>) {
        self.inner.host = v;
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size
    }
    #[setter]
    fn set_batch_size(&mut self, v: usize) {
        self.inner.batch_size = v;
    }

    #[getter]
    fn flush_interval_ms(&self) -> u64 {
        self.inner.flush_interval_ms
    }
    #[setter]
    fn set_flush_interval_ms(&mut self, v: u64) {
        self.inner.flush_interval_ms = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "LangfuseConfig(public_key={}, host={:?}, batch_size={}, flush_interval_ms={})",
            self.inner.public_key,
            self.inner.host,
            self.inner.batch_size,
            self.inner.flush_interval_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// init_langfuse
// ---------------------------------------------------------------------------

/// Initialize the Langfuse exporter and install it as the global tracing
/// subscriber layer.
///
/// Spawns a background tokio task that periodically flushes buffered LLM call
/// traces, token usage, and latency data to the Langfuse ingestion API.
///
/// Call this once at process startup, before any traced work. If a global
/// tracing subscriber is already installed, this is a soft failure: the
/// underlying `LangfuseLayer` is constructed (so its background dispatcher is
/// running) and the function returns ``Ok(())`` without overwriting the
/// existing subscriber. Composing Langfuse with an existing subscriber from
/// Python is not supported; install Langfuse before any other exporter.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn init_langfuse(config: &PyLangfuseConfig) -> PyResult<()> {
    let layer = rust_init_langfuse(config.inner.clone())
        .map_err(|e| BlazenPyError::Workflow(format!("init_langfuse failed: {e}")))?;

    // `try_init` returns Err if a global subscriber is already installed; we
    // treat that as a soft failure so binding callers can opportunistically
    // enable Langfuse without crashing when another exporter (OTLP, fmt) won
    // the race. The `LangfuseLayer` and its background dispatcher remain
    // alive because they were constructed above.
    let _ = tracing_subscriber::registry().with(layer).try_init();

    Ok(())
}
