package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * A single workflow history entry as exposed by Blazen's telemetry
 * subsystem.
 *
 * History entries are emitted in workflow-event order and capture both
 * `event` (the wire-format event that fired) and `stepName` (the handler
 * that produced it). `latencyMs` is the wall-clock time the step took.
 */
@Serializable
public data class WorkflowHistoryEntry(
    val stepName: String,
    val event: Event,
    val latencyMs: Long,
)

/**
 * Static configuration knobs passed to Blazen's telemetry exporters.
 *
 * Each factory below produces a discriminated config record that the
 * generated UniFFI surface knows how to install once the corresponding
 * feature flag is enabled in the native build.
 */
public object Telemetry {
    /** OTLP (OpenTelemetry Line Protocol) exporter config. */
    @Serializable
    public data class OtlpConfig(
        val endpoint: String,
        val headers: Map<String, String> = emptyMap(),
        val serviceName: String = "blazen",
    )

    /** Prometheus metrics-pull exporter config. */
    @Serializable
    public data class PrometheusConfig(
        val bindAddress: String,
        val namespace: String = "blazen",
    )

    /** Langfuse hosted-LLM-observability exporter config. */
    @Serializable
    public data class LangfuseConfig(
        val publicKey: String,
        val secretKey: String,
        val host: String = "https://cloud.langfuse.com",
    )
}
