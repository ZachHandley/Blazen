//! Telemetry surface for the UniFFI bindings.
//!
//! Wraps the feature-gated exporters in [`blazen_telemetry`] (Langfuse, OTLP,
//! Prometheus) so foreign callers can stand them up the same way Python and
//! Node bindings do. Each initializer is feature-gated independently — bindings
//! built without `langfuse` / `otlp` / `prometheus` will simply not include the
//! corresponding symbol in their generated foreign API.
//!
//! ## Workflow history
//!
//! Upstream [`blazen_telemetry::WorkflowHistory`] is not a global registry —
//! it is collected per-handler via `WorkflowHandler::collect_history` after a
//! run finishes. The current UniFFI [`crate::workflow::Workflow`] surface
//! consumes the handler internally inside `run()`, so the handler-level
//! history accessor is not directly reachable from foreign code yet. As a
//! pragmatic bridge, this module exposes:
//!
//! - [`WorkflowHistoryEntry`]: a flat, FFI-friendly record matching one slot
//!   of an upstream [`blazen_telemetry::HistoryEvent`].
//! - [`parse_workflow_history`]: decode a JSON-serialised
//!   [`blazen_telemetry::WorkflowHistory`] (the format produced by
//!   `serde_json::to_string(&history)`) into a flat
//!   `Vec<WorkflowHistoryEntry>`. This mirrors the Python binding's
//!   `WorkflowHistory.from_json` shape and lets foreign code consume history
//!   that was either produced inside Rust (and serialised out) or produced
//!   by a future handler-exposing API revision.
//!
//! When the workflow surface grows a `Workflow::run_with_history` (or similar)
//! that returns both result and serialised history, foreign callers can feed
//! the JSON straight into [`parse_workflow_history`] without any further
//! changes to this module.

use crate::errors::{BlazenError, BlazenResult};

// ---------------------------------------------------------------------------
// Langfuse
// ---------------------------------------------------------------------------

/// Initialize the Langfuse LLM-observability exporter and install it as the
/// global `tracing` subscriber layer.
///
/// Spawns a background tokio task that periodically flushes buffered LLM
/// call traces, token usage, and latency data to the Langfuse ingestion API.
/// Call once at process startup, before any traced work.
///
/// Arguments:
///   - `public_key`: Langfuse public API key (HTTP Basic-auth username).
///   - `secret_key`: Langfuse secret API key (HTTP Basic-auth password).
///   - `host`: optional Langfuse host URL; defaults to
///     `https://cloud.langfuse.com` when `None`.
///
/// Batch size and flush interval use upstream defaults (100 events / 5000 ms).
/// If a finer-grained config knob is needed, expose it here later — upstream's
/// `LangfuseConfig` supports both via `with_batch_size` / `with_flush_interval_ms`.
///
/// If a global `tracing` subscriber is already installed, the underlying
/// `LangfuseLayer` is still constructed (so its background dispatcher runs)
/// and this returns `Ok(())` without overwriting the existing subscriber.
///
/// # Errors
///
/// Returns [`BlazenError::Internal`] if the underlying HTTP client or
/// dispatcher cannot be built.
#[cfg(feature = "langfuse")]
#[uniffi::export]
pub fn init_langfuse(
    public_key: String,
    secret_key: String,
    host: Option<String>,
) -> BlazenResult<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let mut config = blazen_telemetry::LangfuseConfig::new(public_key, secret_key);
    if let Some(h) = host {
        config = config.with_host(h);
    }

    let layer = blazen_telemetry::init_langfuse(config).map_err(|e| BlazenError::Internal {
        message: format!("init_langfuse failed: {e}"),
    })?;

    let _ = tracing_subscriber::registry().with(layer).try_init();
    Ok(())
}

// ---------------------------------------------------------------------------
// OTLP
// ---------------------------------------------------------------------------

/// Initialize the OpenTelemetry OTLP (gRPC/tonic) trace exporter and install
/// it as the global `tracing` subscriber stack.
///
/// Arguments:
///   - `endpoint`: OTLP gRPC endpoint URL (e.g. `"http://localhost:4317"`).
///   - `service_name`: service name reported to the backend; defaults to
///     `"blazen"` when `None`.
///
/// Upstream's [`blazen_telemetry::OtlpConfig`] does not currently accept
/// per-request headers — if your backend needs an `Authorization` header
/// (Honeycomb, Datadog, Grafana Cloud, etc.), set it via the
/// `OTEL_EXPORTER_OTLP_HEADERS` environment variable, which the
/// `opentelemetry-otlp` crate reads at exporter-build time.
///
/// # Errors
///
/// Returns [`BlazenError::Internal`] if the OTLP exporter or tracer provider
/// cannot be constructed.
#[cfg(feature = "otlp")]
#[uniffi::export]
pub fn init_otlp(endpoint: String, service_name: Option<String>) -> BlazenResult<()> {
    let config = blazen_telemetry::OtlpConfig {
        endpoint,
        service_name: service_name.unwrap_or_else(|| "blazen".to_string()),
    };

    blazen_telemetry::init_otlp(config).map_err(|e| BlazenError::Internal {
        message: format!("init_otlp failed: {e}"),
    })
}

// ---------------------------------------------------------------------------
// Prometheus
// ---------------------------------------------------------------------------

/// Initialize the Prometheus metrics exporter and start the HTTP listener.
///
/// Installs a global `metrics` recorder backed by Prometheus and starts an
/// HTTP server serving the `/metrics` endpoint.
///
/// `listen_address` accepts a `host:port` string (e.g. `"0.0.0.0:9100"`).
/// Upstream [`blazen_telemetry::init_prometheus`] always binds `0.0.0.0` and
/// only takes a port, so the host portion of `listen_address` is parsed for
/// validation but does **not** override the upstream bind address — the
/// listener always accepts traffic on every interface. Pass a plain port
/// string like `"9100"` to skip the host portion.
///
/// # Errors
///
/// Returns [`BlazenError::Validation`] if `listen_address` is not a
/// well-formed `host:port` (or bare port) string, or
/// [`BlazenError::Internal`] if the HTTP listener cannot be bound or the
/// global metrics recorder cannot be installed (e.g. one is already set).
#[cfg(feature = "prometheus")]
#[uniffi::export]
pub fn init_prometheus(listen_address: String) -> BlazenResult<()> {
    let port = parse_listen_port(&listen_address)?;
    blazen_telemetry::init_prometheus(port).map_err(|e| BlazenError::Internal {
        message: format!("init_prometheus failed: {e}"),
    })
}

/// Parse a `host:port` or bare-port string into a `u16` port number.
#[cfg(feature = "prometheus")]
fn parse_listen_port(addr: &str) -> BlazenResult<u16> {
    if let Ok(socket_addr) = addr.parse::<std::net::SocketAddr>() {
        return Ok(socket_addr.port());
    }
    if let Some((_host, port_str)) = addr.rsplit_once(':') {
        return port_str
            .parse::<u16>()
            .map_err(|e| BlazenError::Validation {
                message: format!("invalid listen_address '{addr}': bad port: {e}"),
            });
    }
    addr.parse::<u16>().map_err(|e| BlazenError::Validation {
        message: format!("invalid listen_address '{addr}': expected 'host:port' or bare port: {e}"),
    })
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

/// Best-effort flush + shutdown of any initialised telemetry exporters.
///
/// Upstream [`blazen_telemetry`] does not currently expose explicit
/// shutdown hooks: the Langfuse dispatcher flushes on its own interval and
/// drops cleanly when the process exits; the OTLP `SdkTracerProvider` is
/// owned globally by `opentelemetry::global` and flushed on drop; the
/// Prometheus listener runs until the process exits. Calling this function
/// is therefore safe but currently has no observable effect — it exists so
/// foreign callers can wire a single "shutdown" hook into their app
/// lifecycle without conditionally branching on features. When upstream
/// grows explicit shutdown APIs, this function will route to them.
///
/// Safe to call even if no exporter was initialised.
#[uniffi::export]
pub fn shutdown_telemetry() {}

// ---------------------------------------------------------------------------
// Workflow history
// ---------------------------------------------------------------------------

/// One flattened slot of a workflow execution history, suitable for FFI.
///
/// Mirrors the wire shape of an upstream
/// [`blazen_telemetry::HistoryEvent`] but collapses the typed
/// [`blazen_telemetry::HistoryEventKind`] enum into three flat fields
/// (`event_type`, `event_data_json`, plus surface-pulled `step_name`,
/// `duration_ms`, `error`) so that Go / Swift / Kotlin / Ruby callers don't
/// need to model an open-ended sum type. The full typed payload is always
/// available in `event_data_json` as the serde JSON representation.
///
/// `workflow_id` is propagated from the enclosing
/// [`blazen_telemetry::WorkflowHistory::run_id`] so each entry is
/// self-identifying.
#[derive(Debug, Clone, uniffi::Record)]
pub struct WorkflowHistoryEntry {
    /// UUID of the workflow run this event belongs to.
    pub workflow_id: String,
    /// Step name when the event is step- or LLM-call-scoped; empty otherwise.
    pub step_name: String,
    /// Variant tag of the upstream `HistoryEventKind` (e.g.
    /// `"WorkflowStarted"`, `"StepCompleted"`, `"LlmCallFailed"`).
    pub event_type: String,
    /// Full serde JSON payload of the upstream `HistoryEventKind` variant —
    /// always includes the variant tag plus every typed field.
    pub event_data_json: String,
    /// Event timestamp as Unix epoch milliseconds.
    pub timestamp_ms: u64,
    /// Step / LLM-call duration in milliseconds, when the variant carries it.
    pub duration_ms: Option<u64>,
    /// Error message, when the variant is a failure variant.
    pub error: Option<String>,
}

/// Decode a JSON-serialised upstream [`blazen_telemetry::WorkflowHistory`]
/// into a flat `Vec<WorkflowHistoryEntry>`.
///
/// The expected input is the exact format produced by
/// `serde_json::to_string(&history)` on a
/// [`blazen_telemetry::WorkflowHistory`] (i.e. an object with `run_id`,
/// `workflow_name`, and `events: [{timestamp, sequence, kind}]`). This is
/// the same shape the Python binding's `WorkflowHistory.from_json` accepts,
/// so foreign callers can round-trip history JSON across bindings.
///
/// Returns an empty vector if the history has no events.
///
/// `blazen-telemetry`'s `history` feature is hard-pinned on in this crate's
/// `Cargo.toml`, so this function is always available regardless of which
/// optional exporter features are enabled.
///
/// # Errors
///
/// Returns [`BlazenError::Validation`] when `history_json` fails to
/// deserialise as a [`blazen_telemetry::WorkflowHistory`].
#[uniffi::export]
pub fn parse_workflow_history(history_json: String) -> BlazenResult<Vec<WorkflowHistoryEntry>> {
    parse_workflow_history_impl(&history_json)
}

fn parse_workflow_history_impl(history_json: &str) -> BlazenResult<Vec<WorkflowHistoryEntry>> {
    let history: blazen_telemetry::WorkflowHistory =
        serde_json::from_str(history_json).map_err(|e| BlazenError::Validation {
            message: format!("invalid workflow-history JSON: {e}"),
        })?;

    let workflow_id = history.run_id.to_string();
    let entries = history
        .events
        .into_iter()
        .map(|event| flatten_event(&workflow_id, event))
        .collect();
    Ok(entries)
}

/// Flatten an upstream [`blazen_telemetry::HistoryEvent`] into the FFI
/// record shape.
fn flatten_event(workflow_id: &str, event: blazen_telemetry::HistoryEvent) -> WorkflowHistoryEntry {
    use blazen_telemetry::HistoryEventKind as K;

    let event_type = variant_tag(&event.kind).to_string();
    let event_data_json = serde_json::to_string(&event.kind).unwrap_or_else(|_| "{}".to_string());
    let timestamp_ms = u64::try_from(event.timestamp.timestamp_millis().max(0)).unwrap_or(0);

    let (step_name, duration_ms, error) = match &event.kind {
        K::StepDispatched { step_name, .. } => (step_name.clone(), None, None),
        K::StepCompleted {
            step_name,
            duration_ms,
            ..
        } => (step_name.clone(), Some(*duration_ms), None),
        K::StepFailed {
            step_name,
            error,
            duration_ms,
        } => (step_name.clone(), Some(*duration_ms), Some(error.clone())),
        K::LlmCallStarted { provider, model } => (format!("{provider}:{model}"), None, None),
        K::LlmCallCompleted {
            provider,
            model,
            duration_ms,
            ..
        } => (format!("{provider}:{model}"), Some(*duration_ms), None),
        K::LlmCallFailed {
            provider,
            model,
            error,
            duration_ms,
        } => (
            format!("{provider}:{model}"),
            Some(*duration_ms),
            Some(error.clone()),
        ),
        K::WorkflowCompleted { duration_ms } => (String::new(), Some(*duration_ms), None),
        K::WorkflowFailed { error, duration_ms } => {
            (String::new(), Some(*duration_ms), Some(error.clone()))
        }
        K::WorkflowTimedOut { elapsed_ms } => (String::new(), Some(*elapsed_ms), None),
        _ => (String::new(), None, None),
    };

    WorkflowHistoryEntry {
        workflow_id: workflow_id.to_string(),
        step_name,
        event_type,
        event_data_json,
        timestamp_ms,
        duration_ms,
        error,
    }
}

/// Return the variant-tag string for a [`blazen_telemetry::HistoryEventKind`].
fn variant_tag(kind: &blazen_telemetry::HistoryEventKind) -> &'static str {
    use blazen_telemetry::HistoryEventKind as K;
    match kind {
        K::WorkflowStarted { .. } => "WorkflowStarted",
        K::EventReceived { .. } => "EventReceived",
        K::StepDispatched { .. } => "StepDispatched",
        K::StepCompleted { .. } => "StepCompleted",
        K::StepFailed { .. } => "StepFailed",
        K::LlmCallStarted { .. } => "LlmCallStarted",
        K::LlmCallCompleted { .. } => "LlmCallCompleted",
        K::LlmCallFailed { .. } => "LlmCallFailed",
        K::WorkflowPaused { .. } => "WorkflowPaused",
        K::WorkflowResumed => "WorkflowResumed",
        K::InputRequested { .. } => "InputRequested",
        K::InputReceived { .. } => "InputReceived",
        K::WorkflowCompleted { .. } => "WorkflowCompleted",
        K::WorkflowFailed { .. } => "WorkflowFailed",
        K::WorkflowTimedOut { .. } => "WorkflowTimedOut",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_telemetry::{HistoryEventKind, PauseReason, WorkflowHistory};
    use uuid::Uuid;

    #[test]
    fn empty_history_round_trips() {
        let history = WorkflowHistory::new(Uuid::nil(), "demo".into());
        let json = serde_json::to_string(&history).unwrap();
        let entries = parse_workflow_history_impl(&json).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn flattens_step_completed() {
        let mut history = WorkflowHistory::new(Uuid::nil(), "demo".into());
        history.push(HistoryEventKind::StepCompleted {
            step_name: "fetch".into(),
            duration_ms: 42,
            output_type: "DoneEvent".into(),
        });
        let json = serde_json::to_string(&history).unwrap();
        let entries = parse_workflow_history_impl(&json).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.event_type, "StepCompleted");
        assert_eq!(e.step_name, "fetch");
        assert_eq!(e.duration_ms, Some(42));
        assert!(e.error.is_none());
        assert!(e.event_data_json.contains("StepCompleted"));
        assert!(e.event_data_json.contains("fetch"));
    }

    #[test]
    fn flattens_llm_call_failed() {
        let mut history = WorkflowHistory::new(Uuid::nil(), "demo".into());
        history.push(HistoryEventKind::LlmCallFailed {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            error: "boom".into(),
            duration_ms: 7,
        });
        let json = serde_json::to_string(&history).unwrap();
        let entries = parse_workflow_history_impl(&json).unwrap();
        let e = &entries[0];
        assert_eq!(e.event_type, "LlmCallFailed");
        assert_eq!(e.step_name, "openai:gpt-4o");
        assert_eq!(e.duration_ms, Some(7));
        assert_eq!(e.error.as_deref(), Some("boom"));
    }

    #[test]
    fn flattens_paused() {
        let mut history = WorkflowHistory::new(Uuid::nil(), "demo".into());
        history.push(HistoryEventKind::WorkflowPaused {
            reason: PauseReason::InputRequired,
            pending_count: 3,
        });
        let json = serde_json::to_string(&history).unwrap();
        let entries = parse_workflow_history_impl(&json).unwrap();
        let e = &entries[0];
        assert_eq!(e.event_type, "WorkflowPaused");
        assert_eq!(e.step_name, "");
        assert!(e.duration_ms.is_none());
        assert!(e.error.is_none());
        assert!(e.event_data_json.contains("InputRequired"));
    }

    #[test]
    fn rejects_invalid_json() {
        let err = parse_workflow_history_impl("not json").unwrap_err();
        match err {
            BlazenError::Validation { message } => {
                assert!(message.contains("invalid workflow-history JSON"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn workflow_id_propagates() {
        let run_id = Uuid::from_u128(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
        let mut history = WorkflowHistory::new(run_id, "demo".into());
        history.push(HistoryEventKind::WorkflowResumed);
        let json = serde_json::to_string(&history).unwrap();
        let entries = parse_workflow_history_impl(&json).unwrap();
        assert_eq!(entries[0].workflow_id, run_id.to_string());
    }

    #[cfg(feature = "prometheus")]
    #[test]
    fn parse_listen_port_accepts_socket_addr() {
        assert_eq!(parse_listen_port("0.0.0.0:9100").unwrap(), 9100);
        assert_eq!(parse_listen_port("127.0.0.1:8080").unwrap(), 8080);
        assert_eq!(parse_listen_port("[::1]:9090").unwrap(), 9090);
    }

    #[cfg(feature = "prometheus")]
    #[test]
    fn parse_listen_port_accepts_bare_port() {
        assert_eq!(parse_listen_port("9100").unwrap(), 9100);
    }

    #[cfg(feature = "prometheus")]
    #[test]
    fn parse_listen_port_accepts_host_port_pair() {
        assert_eq!(parse_listen_port("my-host:9100").unwrap(), 9100);
    }

    #[cfg(feature = "prometheus")]
    #[test]
    fn parse_listen_port_rejects_garbage() {
        assert!(parse_listen_port("not-a-port").is_err());
        assert!(parse_listen_port("host:not-a-port").is_err());
    }
}
