import Foundation
import BlazenSwift

/// Trivial single-step workflow demo.
///
/// Mirrors the cross-binding "hello world" used in the Go / Kotlin / Ruby
/// examples: a single `greet` step accepts a `StartEvent` carrying a
/// `name` field, returns a `StopEvent` whose JSON payload is
/// `{"result": "Hello, <name>!"}`, and the main entry runs it and prints
/// the terminal payload.
@main
struct HelloWorkflow {
    static func main() async throws {
        Blazen.initialize()
        defer { Blazen.shutdown() }

        let builder = WorkflowBuilder(name: "greeter")
        _ = try builder.step(
            name: "greet",
            accepts: ["blazen::StartEvent"],
            emits: ["blazen::StopEvent"],
            handler: GreetHandler()
        )
        let workflow = try builder.build()

        let result = try await workflow.run(["name": "Zach"])

        print("event:    \(result.event.eventType)")
        print("payload:  \(result.event.dataJson)")
        print("tokens:   in=\(result.totalInputTokens) out=\(result.totalOutputTokens)")
    }
}

/// Step handler that pulls the `name` field out of the incoming
/// `StartEvent` and emits a `StopEvent` carrying a greeting.
private final class GreetHandler: StepHandler, @unchecked Sendable {
    func invoke(event: Event) async throws -> StepOutput {
        let payload = event.dataJson.data(using: .utf8) ?? Data()
        let parsed = try JSONSerialization.jsonObject(with: payload) as? [String: Any] ?? [:]
        let name = parsed["name"] as? String ?? "world"
        let response: [String: String] = ["result": "Hello, \(name)!"]
        let data = try JSONSerialization.data(withJSONObject: response)
        let resultJson = String(data: data, encoding: .utf8) ?? "{}"
        return .single(event: Event(eventType: "StopEvent", dataJson: resultJson))
    }
}
