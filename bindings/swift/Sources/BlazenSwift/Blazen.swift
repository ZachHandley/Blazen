import Foundation
import UniFFIBlazen

/// Top-level entry point for the Blazen Swift binding.
///
/// `Blazen` is a namespace of static helpers covering one-time process
/// lifecycle (`initialize`, `shutdown`) and version introspection.
/// Everything else lives on dedicated types: `Workflow`, `Pipeline`,
/// `CompletionModel`, `EmbeddingModel`, `Agent`, the `Providers` factories,
/// etc.
public enum Blazen {
    /// Run Blazen's one-time process initialisation.
    ///
    /// Safe to call multiple times — subsequent calls are no-ops. Brings up
    /// the shared Tokio runtime that backs every `async` method in this
    /// binding. Recommended at app launch (e.g. in `@main`) so the first
    /// request a user issues doesn't pay the runtime spin-up cost.
    public static func initialize() {
        UniFFIBlazen.`init`()
    }

    /// Best-effort flush + shutdown of any initialised telemetry exporters.
    ///
    /// Always safe to call, even when no exporters were initialised. Call
    /// once near process exit (e.g. inside an `atexit`-style hook) so any
    /// buffered telemetry has a chance to ship before the runtime is torn
    /// down.
    public static func shutdown() {
        UniFFIBlazen.shutdownTelemetry()
    }

    /// The semantic version of the underlying `blazen-uniffi` native lib,
    /// as compiled into the loaded `libblazen_uniffi` artefact.
    ///
    /// Useful for diagnosing version skew between the Swift wrapper and
    /// the native lib shipped alongside it.
    public static var version: String {
        UniFFIBlazen.version()
    }
}
