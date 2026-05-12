import Foundation
import UniFFIBlazen

/// A built workflow ready to run.
///
/// Re-exported from the UniFFI glue under the canonical Blazen namespace.
/// Construct via `WorkflowBuilder(name:)` then `.step(...)...build()`, then
/// invoke `.run(input:)` to execute.
public typealias Workflow = UniFFIBlazen.Workflow

/// Builder for [`Workflow`]. Start with `WorkflowBuilder(name:)`, chain
/// `.step(...)`, optional timeouts, and finish with `.build()`.
public typealias WorkflowBuilder = UniFFIBlazen.WorkflowBuilder

/// Final result of a workflow run.
public typealias WorkflowResult = UniFFIBlazen.WorkflowResult

/// Event crossed across the FFI boundary.
///
/// `eventType` is the free-form class name (e.g. `"StartEvent"`,
/// `"StopEvent"`, `"MyCustomEvent"`); `dataJson` is a JSON-encoded payload.
public typealias Event = UniFFIBlazen.Event

/// What a `StepHandler` returns: zero, one, or many events to publish.
public typealias StepOutput = UniFFIBlazen.StepOutput

/// Foreign-callable step handler protocol. Implement this on any
/// `Sendable` reference type and pass it to `WorkflowBuilder.step(...)` to
/// have your handler invoked whenever a matching event arrives.
public typealias StepHandler = UniFFIBlazen.StepHandler

// MARK: - Idiomatic helpers

public extension Workflow {
    /// Run the workflow with a JSON-encodable input value.
    ///
    /// Convenience over the underlying `run(inputJson:)` that accepts any
    /// `Encodable` — typically a struct or dictionary. The value is JSON-
    /// encoded with `JSONEncoder` and handed to the workflow engine as the
    /// `StartEvent` payload.
    ///
    /// Pass `()` (or call `runEmpty()`) to start a workflow whose
    /// `StartEvent` has no payload.
    func run<Input: Encodable>(_ input: Input) async throws -> WorkflowResult {
        let data = try JSONEncoder().encode(input)
        let json = String(data: data, encoding: .utf8) ?? "null"
        return try await run(inputJson: json)
    }

    /// Run the workflow with a `null` JSON input. Equivalent to
    /// `run(inputJson: "null")`.
    func runEmpty() async throws -> WorkflowResult {
        try await run(inputJson: "null")
    }

    /// Step names in registration order. Re-exposed as a Swift property
    /// so call sites don't pay the `()` boilerplate of the underlying
    /// `stepNames()` method.
    var steps: [String] {
        stepNames()
    }
}

public extension WorkflowBuilder {
    /// Builder-style step timeout. Equivalent to `stepTimeoutMs(millis:)`
    /// but takes a `TimeInterval` (seconds) so call sites read in human
    /// units (`builder.stepTimeout(30)` rather than
    /// `builder.stepTimeoutMs(millis: 30_000)`).
    @discardableResult
    func stepTimeout(_ seconds: TimeInterval) throws -> WorkflowBuilder {
        try stepTimeoutMs(millis: UInt64(max(0, seconds * 1000)))
    }

    /// Builder-style overall workflow timeout. Equivalent to
    /// `timeoutMs(millis:)` but takes a `TimeInterval` (seconds).
    @discardableResult
    func timeout(_ seconds: TimeInterval) throws -> WorkflowBuilder {
        try timeoutMs(millis: UInt64(max(0, seconds * 1000)))
    }
}

// MARK: - Codable on the wire records

extension Event: Codable {
    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case dataJson = "data_json"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            eventType: try c.decode(String.self, forKey: .eventType),
            dataJson: try c.decode(String.self, forKey: .dataJson)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(eventType, forKey: .eventType)
        try c.encode(dataJson, forKey: .dataJson)
    }
}
