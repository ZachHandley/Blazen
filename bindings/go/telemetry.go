package blazen

import (
	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// This file collects the telemetry-exporter lifecycle wrappers and the
// workflow-history JSON parser exposed by blazen-go.
//
// Each InitXxx below is feature-gated in the underlying native lib —
// bindings built without the matching feature ("langfuse", "otlp",
// "prometheus") return [*UnsupportedError] (BlazenError::Unsupported) at
// call time. Inspect [Version] and the native build configuration to
// confirm availability before relying on these in production.
//
// ShutdownTelemetry is intentionally NOT re-exported here — the package
// already surfaces it as the broader [Shutdown] helper in blazen.go,
// which flushes any initialised exporters before process exit.

// WorkflowHistoryEntry is one entry from a parsed workflow history log.
//
// The TimestampMs is Unix-epoch milliseconds; DurationMs is the
// step/LLM-call duration in milliseconds when the variant carries it,
// zero otherwise (use the zero-value rather than a sentinel — the FFI
// returns a non-nil *uint64 only when the upstream variant supplied a
// duration field, and the wrapper hides that distinction for clarity).
//
// EventType is the variant tag of the upstream HistoryEventKind
// (e.g. "WorkflowStarted", "StepCompleted", "LlmCallFailed").
// EventDataJSON carries the full serde JSON payload of that variant —
// always including the variant tag plus every typed field — so callers
// that need richer fields than the flattened struct exposes can decode
// it with [encoding/json].
type WorkflowHistoryEntry struct {
	// WorkflowID is the UUID of the workflow run this event belongs to.
	WorkflowID string
	// StepName is the step name when the event is step- or LLM-call-scoped;
	// empty otherwise.
	StepName string
	// EventType is the variant tag of the upstream HistoryEventKind.
	EventType string
	// EventDataJSON is the full serde JSON payload of the upstream
	// HistoryEventKind variant.
	EventDataJSON string
	// TimestampMs is the event timestamp as Unix epoch milliseconds.
	TimestampMs uint64
	// DurationMs is the step/LLM-call duration in milliseconds, or 0 when
	// the upstream variant did not report one.
	DurationMs uint64
	// Error carries the failure message for failure-variant events
	// ("StepFailed", "LlmCallFailed", ...). Empty for success variants.
	Error string
}

// workflowHistoryEntryFromFFI converts an uniffi-generated history entry
// into the wrapper struct, flattening the optional *uint64 DurationMs
// to a plain uint64 (zero when the upstream variant did not report a
// duration).
func workflowHistoryEntryFromFFI(e uniffiblazen.WorkflowHistoryEntry) WorkflowHistoryEntry {
	out := WorkflowHistoryEntry{
		WorkflowID:    e.WorkflowId,
		StepName:      e.StepName,
		EventType:     e.EventType,
		EventDataJSON: e.EventDataJson,
		TimestampMs:   e.TimestampMs,
	}
	if e.DurationMs != nil {
		out.DurationMs = *e.DurationMs
	}
	if e.Error != nil {
		out.Error = *e.Error
	}
	return out
}

// ParseWorkflowHistory decodes the JSON history log produced by the
// Blazen runtime into typed entries.
//
// The expected input is the exact format produced by
// serde_json::to_string on the Rust side's blazen_telemetry::WorkflowHistory
// (an object with run_id, workflow_name, and events: [{timestamp,
// sequence, kind}]). Foreign callers can therefore round-trip history
// JSON across bindings. Returns an empty slice when the history has no
// events.
//
// Returns [*ValidationError] when historyJSON fails to deserialise as a
// WorkflowHistory.
func ParseWorkflowHistory(historyJSON string) ([]WorkflowHistoryEntry, error) {
	ensureInit()
	entries, err := uniffiblazen.ParseWorkflowHistory(historyJSON)
	if err != nil {
		return nil, wrapErr(err)
	}
	out := make([]WorkflowHistoryEntry, len(entries))
	for i, e := range entries {
		out[i] = workflowHistoryEntryFromFFI(e)
	}
	return out, nil
}

// InitLangfuse configures the Langfuse trace exporter and installs it as
// the global tracing subscriber layer. Call once at process startup.
//
// publicKey and secretKey come from the Langfuse project settings.
// host is optional — pass "" for the default cloud endpoint.
//
// Returns [*UnsupportedError] if the langfuse feature wasn't compiled
// into the embedded blazen-uniffi (BlazenError::Unsupported), or
// [*InternalError] if Langfuse rejects the credentials at startup or
// the underlying HTTP client / dispatcher cannot be built.
func InitLangfuse(publicKey, secretKey, host string) error {
	ensureInit()
	return wrapErr(uniffiblazen.InitLangfuse(publicKey, secretKey, optString(host)))
}

// InitOTLP configures the OpenTelemetry OTLP (gRPC/tonic) trace exporter
// and installs it as the global tracing subscriber stack.
//
// endpoint is the OTLP gRPC collector URL (e.g. "http://localhost:4317").
// serviceName is what shows up in the trace UI; pass "" to use the
// default ("blazen").
//
// Upstream's OtlpConfig does not currently accept per-request headers —
// if your backend needs an Authorization header (Honeycomb, Datadog,
// Grafana Cloud, etc.), set it via the OTEL_EXPORTER_OTLP_HEADERS
// environment variable, which the opentelemetry-otlp crate reads at
// exporter-build time.
//
// Returns [*UnsupportedError] if the otlp feature wasn't compiled into
// the embedded blazen-uniffi, or [*InternalError] if the OTLP exporter
// or tracer provider cannot be constructed.
func InitOTLP(endpoint, serviceName string) error {
	ensureInit()
	return wrapErr(uniffiblazen.InitOtlp(endpoint, optString(serviceName)))
}

// InitPrometheus starts a Prometheus scrape endpoint at listenAddress
// (e.g. ":9090" or "0.0.0.0:9090") and installs a global metrics
// recorder backed by Prometheus serving the /metrics path.
//
// Upstream blazen_telemetry::init_prometheus always binds 0.0.0.0 and
// only takes a port, so the host portion of listenAddress is parsed for
// validation but does not override the upstream bind address — the
// listener always accepts traffic on every interface. Pass a plain port
// string like "9100" to skip the host portion.
//
// Returns [*UnsupportedError] if the prometheus feature wasn't compiled
// into the embedded blazen-uniffi, [*ValidationError] if listenAddress
// is not a well-formed host:port (or bare port) string, or
// [*InternalError] if the HTTP listener cannot be bound or the global
// metrics recorder cannot be installed (e.g. one is already set).
func InitPrometheus(listenAddress string) error {
	ensureInit()
	return wrapErr(uniffiblazen.InitPrometheus(listenAddress))
}
