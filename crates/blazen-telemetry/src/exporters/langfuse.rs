//! Langfuse exporter for LLM observability.
//!
//! Exports LLM call traces, token usage, and latency data to Langfuse
//! for prompt engineering analytics and cost tracking.
//!
//! # Span mapping
//!
//! | Blazen span name        | Langfuse concept | Ingestion event type     |
//! |-------------------------|------------------|--------------------------|
//! | `workflow.run`          | **Trace**        | `trace-create`           |
//! | `workflow.step`         | **Span**         | `span-create`            |
//! | `llm.complete`          | **Generation**   | `generation-create`      |
//! | `pipeline.run`          | **Trace**        | `trace-create`           |
//! | `pipeline.stage`        | **Span**         | `span-create`            |

use std::collections::HashMap;
#[cfg(all(not(target_arch = "wasm32"), not(target_os = "wasi")))]
use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
#[cfg(all(not(target_arch = "wasm32"), not(target_os = "wasi")))]
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use uuid::Uuid;

#[cfg(not(target_os = "wasi"))]
use crate::error::TelemetryError;

#[cfg(not(target_os = "wasi"))]
const DEFAULT_HOST: &str = "https://cloud.langfuse.com";
const DEFAULT_BATCH_SIZE: usize = 100;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 5000;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Langfuse exporter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangfuseConfig {
    /// Langfuse public API key (used as Basic-auth username).
    pub public_key: String,
    /// Langfuse secret API key (used as Basic-auth password).
    pub secret_key: String,
    /// Langfuse host URL. `None` resolves to `https://cloud.langfuse.com`.
    #[serde(default)]
    pub host: Option<String>,
    /// Maximum number of events buffered before an automatic flush.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Background flush interval in milliseconds.
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
}

fn default_batch_size() -> usize {
    DEFAULT_BATCH_SIZE
}

fn default_flush_interval_ms() -> u64 {
    DEFAULT_FLUSH_INTERVAL_MS
}

impl LangfuseConfig {
    /// Create a new config with the required public and secret keys, defaulting
    /// host to `https://cloud.langfuse.com`, batch size to 100, and flush
    /// interval to 5000 ms.
    #[must_use]
    pub fn new(public_key: impl Into<String>, secret_key: impl Into<String>) -> Self {
        Self {
            public_key: public_key.into(),
            secret_key: secret_key.into(),
            host: None,
            batch_size: DEFAULT_BATCH_SIZE,
            flush_interval_ms: DEFAULT_FLUSH_INTERVAL_MS,
        }
    }

    /// Override the Langfuse host URL.
    #[must_use]
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Override the maximum batch size before an automatic flush.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Override the background flush interval (in milliseconds).
    #[must_use]
    pub fn with_flush_interval_ms(mut self, flush_interval_ms: u64) -> Self {
        self.flush_interval_ms = flush_interval_ms;
        self
    }
}

impl Default for LangfuseConfig {
    fn default() -> Self {
        Self {
            public_key: String::new(),
            secret_key: String::new(),
            host: None,
            batch_size: DEFAULT_BATCH_SIZE,
            flush_interval_ms: DEFAULT_FLUSH_INTERVAL_MS,
        }
    }
}

// ---------------------------------------------------------------------------
// Span data collected via the visitor
// ---------------------------------------------------------------------------

/// Data we stash in span extensions so we can read it back at close time.
#[derive(Debug, Clone, Default)]
struct SpanData {
    name: String,
    fields: HashMap<String, serde_json::Value>,
    start_time: Option<chrono::DateTime<Utc>>,
    /// Langfuse trace ID propagated from the root span.
    trace_id: Option<String>,
    /// The Langfuse ID assigned to this span's parent (for nesting).
    parent_langfuse_id: Option<String>,
}

/// Visitor that records span fields into a `HashMap`.
struct FieldVisitor<'a> {
    fields: &'a mut HashMap<String, serde_json::Value>,
}

impl Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields.insert(
            field.name().to_owned(),
            serde_json::Value::String(format!("{value:?}")),
        );
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields.insert(
            field.name().to_owned(),
            serde_json::Value::String(value.to_owned()),
        );
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_owned(), serde_json::json!(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_owned(), serde_json::json!(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields
            .insert(field.name().to_owned(), serde_json::json!(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_owned(), serde_json::json!(value));
    }
}

// ---------------------------------------------------------------------------
// Envelope channel
// ---------------------------------------------------------------------------

/// One ingestion event (already shaped per Langfuse v1.2 wire format) heading
/// to the dispatcher task.
///
/// `pub(super)` so the wasi sibling dispatcher
/// (`super::wasi_langfuse_client`) can build a typed channel against the
/// same envelope shape.
pub(super) type Envelope = serde_json::Value;

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "wasi")))]
fn spawn_dispatcher(
    client: reqwest::Client,
    host: String,
    public_key: String,
    secret_key: String,
    batch_size: usize,
    flush_interval_ms: u64,
    mut rx: mpsc::UnboundedReceiver<Envelope>,
) -> Result<(), TelemetryError> {
    // `tokio::spawn` requires an active runtime; surface that as a typed error
    // instead of panicking inside the binding boundaries.
    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| TelemetryError::Langfuse(format!("no tokio runtime available: {e}")))?;

    let interval = std::time::Duration::from_millis(flush_interval_ms.max(1));

    handle.spawn(async move {
        let buffer: Arc<Mutex<Vec<Envelope>>> =
            Arc::new(Mutex::new(Vec::with_capacity(batch_size)));
        let mut tick = tokio::time::interval(interval);
        // Skip the immediate first tick.
        tick.tick().await;

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    if let Some(env) = msg {
                        let mut buf = buffer.lock().await;
                        buf.push(env);
                        if buf.len() >= batch_size {
                            let drained = std::mem::take(&mut *buf);
                            drop(buf);
                            send_batch(&client, &host, &public_key, &secret_key, drained).await;
                        }
                    } else {
                        // Channel closed: drain and exit.
                        let mut buf = buffer.lock().await;
                        let drained = std::mem::take(&mut *buf);
                        drop(buf);
                        send_batch(&client, &host, &public_key, &secret_key, drained).await;
                        break;
                    }
                }
                _ = tick.tick() => {
                    let mut buf = buffer.lock().await;
                    if !buf.is_empty() {
                        let drained = std::mem::take(&mut *buf);
                        drop(buf);
                        send_batch(&client, &host, &public_key, &secret_key, drained).await;
                    }
                }
            }
        }
    });

    Ok(())
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
fn spawn_dispatcher(
    client: reqwest::Client,
    host: String,
    public_key: String,
    secret_key: String,
    batch_size: usize,
    _flush_interval_ms: u64,
    mut rx: mpsc::UnboundedReceiver<Envelope>,
) -> Result<(), TelemetryError> {
    // wasm32 has no `tokio::time::interval` (no time driver) and no
    // multi-threaded runtime, so the dispatcher cannot do periodic flushing.
    // Instead it batches strictly by `batch_size` and on channel close.
    // Per-event flushing remains an option for callers who want tight latency:
    // construct the config with `with_batch_size(1)`.
    //
    // `spawn_local` accepts `!Send` futures, which lets us hold the
    // wasm32 `reqwest::Client` (whose response futures are `!Send`) across
    // `.await` points. The mpsc channel itself is `Send + Sync`, so the
    // `LangfuseLayer` (holding the sender) remains `Send + Sync` for the
    // `tracing_subscriber::Layer` trait bound.
    wasm_bindgen_futures::spawn_local(async move {
        let mut buffer: Vec<Envelope> = Vec::with_capacity(batch_size);
        while let Some(env) = rx.recv().await {
            buffer.push(env);
            if buffer.len() >= batch_size {
                let drained = std::mem::take(&mut buffer);
                send_batch(&client, &host, &public_key, &secret_key, drained).await;
            }
        }
        // Channel closed: flush whatever's left.
        if !buffer.is_empty() {
            let drained = std::mem::take(&mut buffer);
            send_batch(&client, &host, &public_key, &secret_key, drained).await;
        }
    });
    Ok(())
}

#[cfg(not(target_os = "wasi"))]
async fn send_batch(
    client: &reqwest::Client,
    host: &str,
    public_key: &str,
    secret_key: &str,
    events: Vec<Envelope>,
) {
    if events.is_empty() {
        return;
    }

    let url = format!("{host}/api/public/ingestion");
    let payload = serde_json::json!({ "batch": events });

    let result = client
        .post(&url)
        .basic_auth(public_key, Some(secret_key))
        .json(&payload)
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(count = events.len(), "langfuse: flushed batch");
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(
                %status,
                body = %body,
                "langfuse: ingestion request returned non-success status"
            );
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                "langfuse: failed to send ingestion request"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The Layer
// ---------------------------------------------------------------------------

/// A `tracing_subscriber::Layer` that exports Blazen workflow, step, and LLM
/// spans to Langfuse as traces, spans, and generations respectively.
///
/// Constructed via [`init_langfuse`].
#[derive(Debug)]
pub struct LangfuseLayer {
    pub(super) tx: mpsc::UnboundedSender<Envelope>,
    pub(super) batch_size: usize,
}

impl LangfuseLayer {
    /// Construct from a sender + batch size. Visible to sibling modules so the
    /// wasi dispatcher (`super::wasi_langfuse_client`) can wire its own
    /// channel into the same layer type.
    #[cfg(target_os = "wasi")]
    pub(super) fn from_parts(tx: mpsc::UnboundedSender<Envelope>, batch_size: usize) -> Self {
        Self { tx, batch_size }
    }
}

impl<S> Layer<S> for LangfuseLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let span_name = attrs.metadata().name().to_owned();

        let mut data = SpanData {
            name: span_name.clone(),
            start_time: Some(Utc::now()),
            ..Default::default()
        };

        let mut visitor = FieldVisitor {
            fields: &mut data.fields,
        };
        attrs.record(&mut visitor);

        // Inherit trace context from the parent (if any).
        if let Some(parent) = span.parent() {
            let extensions = parent.extensions();
            if let Some(parent_data) = extensions.get::<SpanData>() {
                if is_root_span_name(&parent_data.name) {
                    data.trace_id = parent_data
                        .fields
                        .get("run_id")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                        .or_else(|| {
                            parent_data
                                .fields
                                .get("_langfuse_id")
                                .and_then(|v| v.as_str())
                                .map(ToOwned::to_owned)
                        });
                    data.parent_langfuse_id.clone_from(&data.trace_id);
                } else {
                    data.trace_id.clone_from(&parent_data.trace_id);
                    data.parent_langfuse_id = parent_data
                        .fields
                        .get("_langfuse_id")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned);
                }
            }
        }

        // Assign this span's Langfuse ID.
        let langfuse_id = if is_root_span_name(&span_name) {
            data.fields
                .get("run_id")
                .and_then(|v| v.as_str())
                .map_or_else(|| Uuid::new_v4().to_string(), ToOwned::to_owned)
        } else {
            Uuid::new_v4().to_string()
        };
        data.fields.insert(
            "_langfuse_id".to_owned(),
            serde_json::Value::String(langfuse_id),
        );

        // Root spans are their own trace.
        if is_root_span_name(&data.name) && data.trace_id.is_none() {
            data.trace_id = data
                .fields
                .get("_langfuse_id")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned);
        }

        let mut extensions = span.extensions_mut();
        extensions.insert(data);
    }

    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut extensions = span.extensions_mut();
        if let Some(data) = extensions.get_mut::<SpanData>() {
            let mut visitor = FieldVisitor {
                fields: &mut data.fields,
            };
            values.record(&mut visitor);
        }
    }

    // TODO: accumulate `on_event` log lines into trailing span metadata.

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };

        let data = {
            let extensions = span.extensions();
            match extensions.get::<SpanData>() {
                Some(d) => d.clone(),
                None => return,
            }
        };

        let now = Utc::now();
        let start_time = data
            .start_time
            .unwrap_or(now)
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let end_time = now.to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        let event = match data.name.as_str() {
            "workflow.run" | "pipeline.run" => Some(Self::build_trace_create(&data, &start_time)),
            "workflow.step"
            | "pipeline.stage"
            | "pipeline.stage.sequential"
            | "pipeline.stage.parallel" => {
                Some(Self::build_span_create(&data, &start_time, &end_time))
            }
            "llm.complete" | "llm.stream" => {
                Some(Self::build_generation_create(&data, &start_time, &end_time))
            }
            _ => None,
        };

        if let Some(evt) = event {
            // The receiver lives on a background task; a send error means the
            // dispatcher already shut down and we silently drop the envelope.
            let _ = self.tx.send(evt);
        }

        // batch_size is enforced by the dispatcher; touch the field so the
        // compiler keeps it (and so callers can introspect).
        let _ = self.batch_size;
    }
}

fn is_root_span_name(name: &str) -> bool {
    matches!(name, "workflow.run" | "pipeline.run")
}

impl LangfuseLayer {
    fn build_trace_create(data: &SpanData, start_time: &str) -> serde_json::Value {
        let trace_id = data
            .fields
            .get("_langfuse_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let name = data
            .fields
            .get("workflow_name")
            .or_else(|| data.fields.get("pipeline_name"))
            .and_then(|v| v.as_str())
            .unwrap_or(&data.name);

        let metadata = collect_metadata(&data.fields, &[]);

        serde_json::json!({
            "id": Uuid::new_v4().to_string(),
            "type": "trace-create",
            "timestamp": start_time,
            "body": {
                "id": trace_id,
                "name": name,
                "timestamp": start_time,
                "metadata": metadata,
            }
        })
    }

    fn build_span_create(data: &SpanData, start_time: &str, end_time: &str) -> serde_json::Value {
        let langfuse_id = data
            .fields
            .get("_langfuse_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let trace_id = data.trace_id.as_deref().unwrap_or("unknown");

        let name = data
            .fields
            .get("step_name")
            .or_else(|| data.fields.get("stage_name"))
            .and_then(|v| v.as_str())
            .unwrap_or(&data.name);

        let level = data
            .fields
            .get("otel.status_code")
            .and_then(|v| v.as_str())
            .map_or("DEFAULT", |s| {
                if s.eq_ignore_ascii_case("error") {
                    "ERROR"
                } else {
                    "DEFAULT"
                }
            });

        let metadata = collect_metadata(&data.fields, &[]);

        let mut body = serde_json::json!({
            "id": langfuse_id,
            "traceId": trace_id,
            "name": name,
            "startTime": start_time,
            "endTime": end_time,
            "level": level,
            "metadata": metadata,
        });

        if let Some(parent_id) = &data.parent_langfuse_id {
            body["parentObservationId"] = serde_json::json!(parent_id);
        }

        serde_json::json!({
            "id": Uuid::new_v4().to_string(),
            "type": "span-create",
            "timestamp": start_time,
            "body": body,
        })
    }

    fn build_generation_create(
        data: &SpanData,
        start_time: &str,
        end_time: &str,
    ) -> serde_json::Value {
        let langfuse_id = data
            .fields
            .get("_langfuse_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let trace_id = data.trace_id.as_deref().unwrap_or("unknown");

        let model = data
            .fields
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let name = data
            .fields
            .get("provider")
            .and_then(|v| v.as_str())
            .map_or_else(|| model.to_owned(), |p| format!("{p}/{model}"));

        let mut usage = serde_json::Map::new();
        if let Some(v) = data.fields.get("prompt_tokens") {
            usage.insert("input".to_owned(), v.clone());
        }
        if let Some(v) = data.fields.get("completion_tokens") {
            usage.insert("output".to_owned(), v.clone());
        }
        if let Some(v) = data.fields.get("total_tokens") {
            usage.insert("total".to_owned(), v.clone());
        }

        let metadata = collect_metadata(
            &data.fields,
            &[
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "model",
                "provider",
            ],
        );

        let mut body = serde_json::json!({
            "id": langfuse_id,
            "traceId": trace_id,
            "name": name,
            "startTime": start_time,
            "endTime": end_time,
            "model": model,
            "metadata": metadata,
        });

        if !usage.is_empty() {
            body["usage"] = serde_json::Value::Object(usage);
        }

        if let Some(parent_id) = &data.parent_langfuse_id {
            body["parentObservationId"] = serde_json::json!(parent_id);
        }

        serde_json::json!({
            "id": Uuid::new_v4().to_string(),
            "type": "generation-create",
            "timestamp": start_time,
            "body": body,
        })
    }
}

fn collect_metadata(
    fields: &HashMap<String, serde_json::Value>,
    excluded: &[&str],
) -> serde_json::Map<String, serde_json::Value> {
    let mut metadata = serde_json::Map::new();
    for (k, v) in fields {
        if k.starts_with('_') || excluded.contains(&k.as_str()) {
            continue;
        }
        metadata.insert(k.clone(), v.clone());
    }
    metadata
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

/// Initialize the Langfuse exporter layer.
///
/// Returns a [`LangfuseLayer`] that should be composed into a
/// `tracing_subscriber::Registry` via `.with()`. A background tokio task is
/// spawned (on non-wasm32 targets) to periodically flush buffered events to
/// the Langfuse ingestion API.
///
/// # Errors
///
/// Returns [`TelemetryError::Langfuse`] when no tokio runtime is available
/// to host the background dispatcher (native targets only), or when the
/// underlying HTTP client cannot be constructed.
///
/// # Example
///
/// ```rust,no_run
/// use blazen_telemetry::{LangfuseConfig, init_langfuse};
/// use tracing_subscriber::prelude::*;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let config = LangfuseConfig::new("pk-lf-...", "sk-lf-...")
///     .with_batch_size(50)
///     .with_flush_interval_ms(2_500);
///
/// let layer = init_langfuse(config)?;
/// tracing_subscriber::registry().with(layer).init();
/// # Ok(())
/// # }
/// ```
#[cfg(not(target_os = "wasi"))]
pub fn init_langfuse(config: LangfuseConfig) -> Result<LangfuseLayer, TelemetryError> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| TelemetryError::Langfuse(format!("failed to build HTTP client: {e}")))?;

    let host = config.host.unwrap_or_else(|| DEFAULT_HOST.to_owned());
    let LangfuseConfig {
        public_key,
        secret_key,
        batch_size,
        flush_interval_ms,
        ..
    } = config;

    let (tx, rx) = mpsc::unbounded_channel::<Envelope>();

    spawn_dispatcher(
        client,
        host,
        public_key,
        secret_key,
        batch_size,
        flush_interval_ms,
        rx,
    )?;

    Ok(LangfuseLayer { tx, batch_size })
}

/// Re-export of the wasi-specific `init_langfuse`. On wasi we can't use
/// `reqwest`, so the implementation lives in
/// [`super::wasi_langfuse_client`] and routes batches through
/// `Arc<dyn blazen_llm::http::HttpClient>`.
#[cfg(target_os = "wasi")]
pub use super::wasi_langfuse_client::init_langfuse;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_os = "wasi")))]
mod tests {
    use super::{
        DEFAULT_BATCH_SIZE, DEFAULT_FLUSH_INTERVAL_MS, DEFAULT_HOST, LangfuseConfig, init_langfuse,
        is_root_span_name,
    };

    #[test]
    fn default_config_uses_constants() {
        let cfg = LangfuseConfig::default();
        assert!(cfg.host.is_none());
        assert_eq!(cfg.batch_size, DEFAULT_BATCH_SIZE);
        assert_eq!(cfg.flush_interval_ms, DEFAULT_FLUSH_INTERVAL_MS);
        let resolved = cfg.host.unwrap_or_else(|| DEFAULT_HOST.to_owned());
        assert_eq!(resolved, DEFAULT_HOST);
    }

    #[test]
    fn builder_chains() {
        let cfg = LangfuseConfig::new("pk", "sk")
            .with_host("https://eu.langfuse.com")
            .with_batch_size(25)
            .with_flush_interval_ms(1_000);
        assert_eq!(cfg.public_key, "pk");
        assert_eq!(cfg.secret_key, "sk");
        assert_eq!(cfg.host.as_deref(), Some("https://eu.langfuse.com"));
        assert_eq!(cfg.batch_size, 25);
        assert_eq!(cfg.flush_interval_ms, 1_000);
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = LangfuseConfig::new("pk-lf-test", "sk-lf-test")
            .with_host("https://custom.langfuse.com")
            .with_batch_size(20)
            .with_flush_interval_ms(7_500);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let de: LangfuseConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(de.public_key, "pk-lf-test");
        assert_eq!(de.secret_key, "sk-lf-test");
        assert_eq!(de.host.as_deref(), Some("https://custom.langfuse.com"));
        assert_eq!(de.batch_size, 20);
        assert_eq!(de.flush_interval_ms, 7_500);
    }

    #[test]
    fn root_span_names() {
        assert!(is_root_span_name("workflow.run"));
        assert!(is_root_span_name("pipeline.run"));
        assert!(!is_root_span_name("workflow.step"));
        assert!(!is_root_span_name("llm.complete"));
    }

    #[tokio::test]
    async fn init_succeeds_in_runtime() {
        let cfg = LangfuseConfig::new("pk", "sk").with_flush_interval_ms(60_000);
        let layer = init_langfuse(cfg).expect("init succeeds inside a tokio runtime");
        // Drop the layer; channel closes and dispatcher exits cleanly.
        drop(layer);
    }

    #[test]
    fn init_fails_without_runtime() {
        let cfg = LangfuseConfig::new("pk", "sk");
        let err = init_langfuse(cfg).expect_err("must fail outside a tokio runtime");
        assert!(format!("{err}").contains("langfuse"));
    }
}
