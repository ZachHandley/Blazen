import Foundation
import UniFFIBlazen

/// One flattened slot of a workflow execution history.
public typealias WorkflowHistoryEntry = UniFFIBlazen.WorkflowHistoryEntry

/// OTLP wire-level transport (`.grpc` or `.httpProto`).
public typealias OtlpProtocol = UniFFIBlazen.OtlpProtocol

/// Telemetry exporter lifecycle helpers.
///
/// Each `init*` here is feature-gated in the underlying native lib —
/// bindings built without the matching feature (`langfuse`, `otlp`,
/// `prometheus`) will throw `BlazenError.Unsupported` (or fail to link
/// in the symbol at all when the feature is fully absent). Inspect
/// `Blazen.version` and the native build configuration to confirm
/// availability before relying on these in production.
public enum Telemetry {
    /// Initialise the Langfuse LLM-observability exporter and install
    /// it as the global `tracing` subscriber layer. Call once at
    /// process startup.
    public static func initLangfuse(
        publicKey: String,
        secretKey: String,
        host: String? = nil
    ) throws {
        try UniFFIBlazen.initLangfuse(
            publicKey: publicKey,
            secretKey: secretKey,
            host: host
        )
    }

    /// Initialise the OpenTelemetry OTLP trace exporter.
    ///
    /// `endpoint` is the OTLP collector URL (HTTP: `"https://collector/v1/traces"`,
    /// gRPC: `"http://collector:4317"`). `serviceName` defaults to `"blazen"`.
    /// `protocol` defaults to `.httpProto` (HTTP/binary-protobuf); pass `.grpc`
    /// for tonic. `headers` (e.g. `["Authorization": "Bearer ..."]`) are honored
    /// on HTTP and dropped with a warning on gRPC — use HTTP for header-based auth.
    public static func initOtlp(
        endpoint: String,
        serviceName: String? = nil,
        `protocol`: OtlpProtocol = .httpProto,
        headers: [String: String]? = nil
    ) throws {
        try UniFFIBlazen.initOtlp(
            endpoint: endpoint,
            serviceName: serviceName,
            protocol: `protocol`,
            headers: headers
        )
    }

    /// Initialise the Prometheus metrics exporter and start the HTTP
    /// listener. `listenAddress` accepts either a `host:port` string or
    /// a bare port (the host portion is parsed for validation only —
    /// the upstream listener always binds `0.0.0.0`).
    public static func initPrometheus(listenAddress: String) throws {
        try UniFFIBlazen.initPrometheus(listenAddress: listenAddress)
    }

    /// Decode a JSON-serialised `blazen-telemetry::WorkflowHistory` into
    /// a flat `[WorkflowHistoryEntry]`. The expected input is the exact
    /// format produced by `serde_json::to_string(&history)` on the Rust
    /// side, so foreign callers can round-trip history JSON across
    /// bindings.
    public static func parseHistory(_ json: String) throws -> [WorkflowHistoryEntry] {
        try parseWorkflowHistory(historyJson: json)
    }
}
