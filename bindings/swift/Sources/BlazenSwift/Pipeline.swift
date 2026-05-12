import Foundation
import UniFFIBlazen

/// A validated, runnable pipeline composed of one or more workflows.
public typealias Pipeline = UniFFIBlazen.Pipeline

/// Builder for `Pipeline`. Start with `PipelineBuilder(name:)`, add stages
/// with `.addWorkflow(...)`, `.stage(...)`, or `.parallel(...)`, then call
/// `.build()`.
public typealias PipelineBuilder = UniFFIBlazen.PipelineBuilder

public extension Pipeline {
    /// Run the pipeline with a JSON-encodable input. Convenience over the
    /// raw `run(inputJson:)` API — same semantics as `Workflow.run(_:)`.
    func run<Input: Encodable>(_ input: Input) async throws -> WorkflowResult {
        let data = try JSONEncoder().encode(input)
        let json = String(data: data, encoding: .utf8) ?? "null"
        return try await run(inputJson: json)
    }

    /// Stage names in registration order, exposed as a property so call
    /// sites match the `Workflow.steps` shape.
    var stages: [String] {
        stageNames()
    }
}

public extension PipelineBuilder {
    /// Per-stage timeout in seconds. Equivalent to
    /// `timeoutPerStageMs(millis:)` but in seconds.
    @discardableResult
    func timeoutPerStage(_ seconds: TimeInterval) throws -> PipelineBuilder {
        try timeoutPerStageMs(millis: UInt64(max(0, seconds * 1000)))
    }

    /// Total pipeline timeout in seconds. Equivalent to
    /// `totalTimeoutMs(millis:)` but in seconds.
    @discardableResult
    func totalTimeout(_ seconds: TimeInterval) throws -> PipelineBuilder {
        try totalTimeoutMs(millis: UInt64(max(0, seconds * 1000)))
    }
}
