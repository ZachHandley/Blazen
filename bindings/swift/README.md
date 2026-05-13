# Blazen for Swift

Event-driven AI workflow engine with first-class LLM integration, exposed as
an idiomatic Swift package. The same Rust core that powers `blazen` on
crates.io, PyPI, and npm — compiled into `libblazen_uniffi` and wrapped in a
hand-written Swift API that hides every UniFFI seam.

## Status

v0 binding. SwiftPM only. macOS / iOS / tvOS / watchOS binary artefacts ship
as an XCFramework on GitHub Releases tagged `bindings/swift/v*` once the
release CI lands; until then, Apple builds compile against a host-built
`libblazen_uniffi`. Linux requires Swift 5.10+; macOS requires 13+.

## Install

```swift
.package(url: "https://github.com/zachhandley/Blazen", from: "0.1.0")
```

Then depend on `BlazenSwift` from your target. SwiftPM resolves against tags
of the form `bindings/swift/v*` (e.g. `bindings/swift/v0.1.0`); pin with
`.exact("0.1.0")` for byte-stable builds.

## Hello workflow

A single-step workflow that echoes its input back as a `StopEvent`. Mirrors
`Tests/BlazenSwiftTests/BlazenSwiftTests.swift::testSingleStepWorkflowRoundTrip`.

```swift
import BlazenSwift

@main
struct Hello {
    static func main() async throws {
        Blazen.initialize()
        defer { Blazen.shutdown() }

        let builder = WorkflowBuilder(name: "echo")
        _ = try builder.step(
            name: "greet",
            accepts: ["blazen::StartEvent"],
            emits: ["blazen::StopEvent"],
            handler: EchoHandler()
        )
        let workflow = try builder.build()

        let result = try await workflow.run(["name": "Zach"])
        print(result.event.eventType)   // "StopEvent"
        print(result.event.dataJson)    // {"echo":{"name":"Zach"}}
    }
}

final class EchoHandler: StepHandler, @unchecked Sendable {
    func invoke(event: Event) async throws -> StepOutput {
        let wrapped = "{\"echo\":\(event.dataJson)}"
        return .single(event: Event(eventType: "StopEvent", dataJson: wrapped))
    }
}
```

`Blazen.initialize()` is idempotent and spins up the shared Tokio runtime
that backs every `async` call. Call it once at process start.

## Async story

`StepHandler.invoke(event:)` is `async throws`; handlers run on the Blazen
runtime, awaited via Swift's native structured concurrency. Cancelling the
`Task` that owns `workflow.run(_:)` surfaces inside the handler as a thrown
`CancellationError`, which the binding folds into `BlazenError.Cancelled`.
Builder-side deadlines are available as `builder.stepTimeout(5)` and
`builder.timeout(30)` (both in seconds).

## Error handling

`BlazenError` is a flat enum carrying a single `message: String` on every
variant. UniFFI's `flat_error` representation can't transport structured
payloads, so retry hints, status codes, and request ids are folded into the
message string before crossing the FFI boundary.

```swift
do {
    let result = try await workflow.run(payload)
    handle(result)
} catch let error as BlazenError {
    switch error {
    case .Auth:         print("auth failed: \(error.message)")
    case .RateLimit:    print("rate limited: \(error.message)")
    case .Timeout:      print("timed out: \(error.message)")
    case .Validation:   print("bad input: \(error.message)")
    case .Cancelled:    print("cancelled")
    default:            print("blazen error: \(error.message)")
    }
}
```

Use the package-level `wrap(_:)` helper to fold arbitrary `Error` values
into a `BlazenError` (Swift `CancellationError` becomes `.Cancelled`,
everything else becomes `.Internal`).

## XCFramework / macOS install

Apple platform consumers currently build against a host-compiled
`libblazen_uniffi` in `target/release/` (see `Package.swift`'s
`unsafeFlags(["-L../../target/release"])`). Once the release CI ships,
callers will download a notarised `BlazenSwift.xcframework.zip` from the
GitHub Releases page tagged `bindings/swift/v*` and SwiftPM will resolve it
as a `binaryTarget`. Linux builds keep linking against the workspace's
`libblazen_uniffi.{a,so}`.

## Where to go from here

- Quickstart: <https://blazen.dev/docs/guides/swift/quickstart>
- LLM completions: <https://blazen.dev/docs/guides/swift/llm>
- Streaming: <https://blazen.dev/docs/guides/swift/streaming>
- Agent loop: <https://blazen.dev/docs/guides/swift/agent>

## License

Blazen is licensed under the MPL-2.0. See [LICENSE](../../LICENSE) in the
workspace root.
