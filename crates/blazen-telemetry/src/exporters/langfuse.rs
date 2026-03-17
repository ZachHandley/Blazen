//! Langfuse exporter for LLM observability.
//!
//! Exports LLM call traces, token usage, and latency data to Langfuse
//! for prompt engineering analytics and cost tracking.
//!
//! # Span mapping
//!
//! | Blazen span name  | Langfuse concept | Ingestion event type     |
//! |--------------------|------------------|--------------------------|
//! | `workflow.run`     | **Trace**        | `trace-create`           |
//! | `workflow.step`    | **Span**         | `span-create`            |
//! | `llm.complete`     | **Generation**   | `generation-create`      |
//! | `pipeline.run`     | **Trace**        | `trace-create`           |
//! | `pipeline.stage`   | **Span**         | `span-create`            |

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use uuid::Uuid;

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
    /// Langfuse host URL (defaults to `https://cloud.langfuse.com`).
    #[serde(default = "default_host")]
    pub host: String,
    /// Maximum number of events buffered before an automatic flush.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// How often the background task flushes buffered events.
    #[serde(default = "default_flush_interval")]
    pub flush_interval: Duration,
}

fn default_host() -> String {
    "https://cloud.langfuse.com".into()
}
fn default_batch_size() -> usize {
    10
}
fn default_flush_interval() -> Duration {
    Duration::from_secs(5)
}

impl Default for LangfuseConfig {
    fn default() -> Self {
        Self {
            public_key: String::new(),
            secret_key: String::new(),
            host: default_host(),
            batch_size: default_batch_size(),
            flush_interval: default_flush_interval(),
        }
    }
}

// ---------------------------------------------------------------------------
// Span data collected via the visitor
// ---------------------------------------------------------------------------

/// Data we stash in span extensions so we can read it back at close time.
#[derive(Debug, Clone, Default)]
struct SpanData {
    /// The tracing span name (e.g. `workflow.run`, `llm.complete`).
    name: String,
    /// Arbitrary key-value fields recorded on the span.
    fields: HashMap<String, serde_json::Value>,
    /// When the span was opened (wall-clock).
    start_time: Option<chrono::DateTime<Utc>>,
    /// Langfuse trace ID propagated from a parent `workflow.run` span.
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
// Shared buffer
// ---------------------------------------------------------------------------

/// Thread-safe event buffer shared between the layer and the flush task.
#[derive(Debug, Clone)]
struct EventBuffer {
    inner: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl EventBuffer {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Push an event into the buffer and return the current buffer length.
    fn push(&self, event: serde_json::Value) -> usize {
        let mut buf = self.inner.lock().expect("langfuse buffer poisoned");
        buf.push(event);
        buf.len()
    }

    /// Drain all buffered events and return them.
    fn drain(&self) -> Vec<serde_json::Value> {
        let mut buf = self.inner.lock().expect("langfuse buffer poisoned");
        std::mem::take(&mut *buf)
    }
}

// ---------------------------------------------------------------------------
// Flush logic
// ---------------------------------------------------------------------------

/// Send a batch of events to Langfuse.
async fn flush_to_langfuse(
    client: &reqwest::Client,
    host: &str,
    public_key: &str,
    secret_key: &str,
    events: Vec<serde_json::Value>,
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
            tracing::debug!(
                count = events.len(),
                "langfuse: flushed {} events",
                events.len()
            );
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

/// Spawn a background tokio task that periodically flushes the buffer.
fn spawn_flush_task(
    client: reqwest::Client,
    config: LangfuseConfig,
    buffer: EventBuffer,
) -> tokio::sync::mpsc::Sender<()> {
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(config.flush_interval);
        // First tick completes immediately; skip it.
        interval.tick().await;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let events = buffer.drain();
                    flush_to_langfuse(
                        &client,
                        &config.host,
                        &config.public_key,
                        &config.secret_key,
                        events,
                    )
                    .await;
                }
                _ = shutdown_rx.recv() => {
                    // Final flush on shutdown.
                    let events = buffer.drain();
                    flush_to_langfuse(
                        &client,
                        &config.host,
                        &config.public_key,
                        &config.secret_key,
                        events,
                    )
                    .await;
                    break;
                }
            }
        }
    });

    shutdown_tx
}

// ---------------------------------------------------------------------------
// The Layer
// ---------------------------------------------------------------------------

/// A `tracing_subscriber::Layer` that exports Blazen workflow, step, and LLM
/// spans to Langfuse as traces, spans, and generations respectively.
///
/// Constructed via [`init_langfuse`].
pub struct LangfuseLayer {
    client: reqwest::Client,
    config: LangfuseConfig,
    buffer: EventBuffer,
    /// Sends a shutdown signal to the periodic flush task.
    _shutdown_tx: tokio::sync::mpsc::Sender<()>,
}

impl Drop for LangfuseLayer {
    fn drop(&mut self) {
        // The `_shutdown_tx` is dropped here, which closes the channel and
        // causes the background task to perform its final flush and exit.
    }
}

impl<S> Layer<S> for LangfuseLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span not found in on_new_span");
        let span_name = attrs.metadata().name().to_owned();

        let mut data = SpanData {
            name: span_name.clone(),
            start_time: Some(Utc::now()),
            ..Default::default()
        };

        // Record all fields from the span attributes.
        let mut visitor = FieldVisitor {
            fields: &mut data.fields,
        };
        attrs.record(&mut visitor);

        // Propagate trace_id from parent spans.
        if let Some(parent) = span.parent() {
            let extensions = parent.extensions();
            if let Some(parent_data) = extensions.get::<SpanData>() {
                // If the parent is a workflow.run / pipeline.run, use its run_id
                // as the Langfuse trace ID. Otherwise inherit the parent's trace_id.
                if parent_data.name == "workflow.run" || parent_data.name == "pipeline.run" {
                    data.trace_id = parent_data
                        .fields
                        .get("run_id")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                        .or_else(|| Some(Uuid::new_v4().to_string()));
                    // The parent's Langfuse ID is the trace_id itself for root traces.
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

        // Assign a Langfuse ID for this span so children can reference it.
        let langfuse_id = match span_name.as_str() {
            "workflow.run" | "pipeline.run" => {
                // Use run_id / pipeline_name as the trace id if available.
                data.fields
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .map_or_else(|| Uuid::new_v4().to_string(), ToOwned::to_owned)
            }
            _ => Uuid::new_v4().to_string(),
        };
        data.fields.insert(
            "_langfuse_id".to_owned(),
            serde_json::Value::String(langfuse_id),
        );

        // For workflow.run, set trace_id to itself (it IS the trace).
        if (data.name == "workflow.run" || data.name == "pipeline.run") && data.trace_id.is_none() {
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
        if let Some(span) = ctx.span(id) {
            let mut extensions = span.extensions_mut();
            if let Some(data) = extensions.get_mut::<SpanData>() {
                let mut visitor = FieldVisitor {
                    fields: &mut data.fields,
                };
                values.record(&mut visitor);
            }
        }
    }

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
            let len = self.buffer.push(evt);
            if len >= self.config.batch_size {
                // Eagerly flush when we hit the batch size threshold.
                let client = self.client.clone();
                let host = self.config.host.clone();
                let pk = self.config.public_key.clone();
                let sk = self.config.secret_key.clone();
                let buffer = self.buffer.clone();
                tokio::spawn(async move {
                    let events = buffer.drain();
                    flush_to_langfuse(&client, &host, &pk, &sk, events).await;
                });
            }
        }
    }
}

impl LangfuseLayer {
    /// Build a `trace-create` event for `workflow.run` / `pipeline.run` spans.
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

        let mut metadata = serde_json::Map::new();
        for (k, v) in &data.fields {
            if !k.starts_with('_') {
                metadata.insert(k.clone(), v.clone());
            }
        }

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

    /// Build a `span-create` event for `workflow.step` / `pipeline.stage` spans.
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

        let status = data
            .fields
            .get("otel.status_code")
            .and_then(|v| v.as_str())
            .unwrap_or("UNSET");

        let mut metadata = serde_json::Map::new();
        for (k, v) in &data.fields {
            if !k.starts_with('_') {
                metadata.insert(k.clone(), v.clone());
            }
        }

        let mut body = serde_json::json!({
            "id": langfuse_id,
            "traceId": trace_id,
            "name": name,
            "startTime": start_time,
            "endTime": end_time,
            "statusMessage": status,
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

    /// Build a `generation-create` event for `llm.complete` / `llm.stream` spans.
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

        // Build usage details from token fields.
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

        let mut metadata = serde_json::Map::new();
        for (k, v) in &data.fields {
            if !k.starts_with('_')
                && k != "prompt_tokens"
                && k != "completion_tokens"
                && k != "total_tokens"
                && k != "model"
                && k != "provider"
            {
                metadata.insert(k.clone(), v.clone());
            }
        }

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
            body["usageDetails"] = serde_json::Value::Object(usage);
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

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

/// Initialize the Langfuse exporter layer.
///
/// Returns a [`LangfuseLayer`] that should be composed into a
/// `tracing_subscriber::Registry` via `.with()`.
///
/// A background tokio task is spawned to periodically flush buffered events
/// to the Langfuse ingestion API. The task is shut down when the layer is
/// dropped (performing a final flush).
///
/// # Example
///
/// ```rust,no_run
/// use blazen_telemetry::exporters::langfuse::{init_langfuse, LangfuseConfig};
/// use tracing_subscriber::prelude::*;
///
/// let config = LangfuseConfig {
///     public_key: "pk-lf-...".into(),
///     secret_key: "sk-lf-...".into(),
///     ..Default::default()
/// };
///
/// tracing_subscriber::registry()
///     .with(init_langfuse(config))
///     .init();
/// ```
#[must_use]
pub fn init_langfuse(config: LangfuseConfig) -> LangfuseLayer {
    let client = reqwest::Client::new();
    let buffer = EventBuffer::new();

    let shutdown_tx = spawn_flush_task(client.clone(), config.clone(), buffer.clone());

    LangfuseLayer {
        client,
        config,
        buffer,
        _shutdown_tx: shutdown_tx,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LangfuseConfig::default();
        assert_eq!(config.host, "https://cloud.langfuse.com");
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.flush_interval, Duration::from_secs(5));
        assert!(config.public_key.is_empty());
        assert!(config.secret_key.is_empty());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = LangfuseConfig {
            public_key: "pk-lf-test".into(),
            secret_key: "sk-lf-test".into(),
            host: "https://custom.langfuse.com".into(),
            batch_size: 20,
            flush_interval: Duration::from_secs(10),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LangfuseConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.public_key, "pk-lf-test");
        assert_eq!(deserialized.secret_key, "sk-lf-test");
        assert_eq!(deserialized.host, "https://custom.langfuse.com");
        assert_eq!(deserialized.batch_size, 20);
    }

    #[test]
    fn test_event_buffer_push_and_drain() {
        let buffer = EventBuffer::new();
        assert_eq!(buffer.push(serde_json::json!({"test": 1})), 1);
        assert_eq!(buffer.push(serde_json::json!({"test": 2})), 2);

        let drained = buffer.drain();
        assert_eq!(drained.len(), 2);

        // Buffer should be empty after drain.
        let drained_again = buffer.drain();
        assert!(drained_again.is_empty());
    }

    #[test]
    fn test_field_visitor_records_types() {
        let mut fields = HashMap::new();
        let visitor = FieldVisitor {
            fields: &mut fields,
        };

        // We can't easily construct Field instances outside of tracing internals,
        // so we verify the visitor implements the right trait methods by checking
        // the type implements Visit.
        fn assert_visit<T: Visit>(_v: &T) {}
        assert_visit(&visitor);

        // Just verify the HashMap starts empty.
        assert!(visitor.fields.is_empty());
    }
}
