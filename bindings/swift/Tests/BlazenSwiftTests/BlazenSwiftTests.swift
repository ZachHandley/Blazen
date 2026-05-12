import XCTest
@testable import BlazenSwift

/// Smoke tests for the Blazen Swift binding. These do not exercise any
/// network-bound functionality — they only confirm that the FFI surface
/// links, the Tokio runtime starts, and the wrapper layer's bookkeeping
/// (workflow builder, step registration, JSON round-trip) holds together.
final class BlazenSwiftTests: XCTestCase {
    /// `Blazen.initialize()` must be idempotent — calling it from many
    /// test methods (or many app start paths) should never panic.
    override func setUp() {
        super.setUp()
        Blazen.initialize()
    }

    /// The version string is baked in at compile time and must always be
    /// non-empty; an empty value indicates a build-time misconfiguration.
    func testVersionIsNonEmpty() {
        XCTAssertFalse(Blazen.version.isEmpty)
    }

    /// `Blazen.initialize()` is documented as safe to call multiple
    /// times. Verify by calling it again here without crashing.
    func testInitializeIsIdempotent() {
        Blazen.initialize()
        Blazen.initialize()
        XCTAssertFalse(Blazen.version.isEmpty)
    }

    /// `Blazen.shutdown()` is a best-effort flush hook and must be
    /// callable even before any telemetry has been initialised.
    func testShutdownIsAlwaysSafe() {
        Blazen.shutdown()
    }

    /// Build a trivial single-step workflow whose handler echoes its
    /// input into a `StopEvent`, run it, and verify the result event is
    /// `StopEvent` carrying the same payload.
    func testSingleStepWorkflowRoundTrip() async throws {
        let handler = EchoHandler()
        let builder = WorkflowBuilder(name: "echo")
        _ = try builder.step(
            name: "greet",
            accepts: ["blazen::StartEvent"],
            emits: ["blazen::StopEvent"],
            handler: handler
        )
        let workflow = try builder.build()

        let payload = ["name": "Zach"]
        let result = try await workflow.run(payload)

        XCTAssertTrue(
            result.event.eventType.contains("StopEvent"),
            "Expected event type to contain StopEvent, got \(result.event.eventType)"
        )
        XCTAssertTrue(result.event.dataJson.contains("Zach"))
        XCTAssertEqual(workflow.steps, ["greet"])
    }

    /// `WorkflowBuilder.stepTimeout(_:)` and `.timeout(_:)` convenience
    /// methods must chain cleanly and not consume the builder before
    /// `.build()`.
    func testWorkflowBuilderTimeoutChaining() throws {
        let builder = WorkflowBuilder(name: "timed")
        _ = try builder
            .step(
                name: "step",
                accepts: ["blazen::StartEvent"],
                emits: ["blazen::StopEvent"],
                handler: EchoHandler()
            )
        _ = try builder.stepTimeout(5)
        _ = try builder.timeout(30)
        let workflow = try builder.build()
        XCTAssertEqual(workflow.steps, ["step"])
    }

    /// The Blazen-side `BlazenError` enum must expose the same `message`
    /// payload regardless of variant.
    func testBlazenErrorMessageAccessor() {
        let err = BlazenError.Validation(message: "bad input")
        XCTAssertEqual(err.message, "bad input")
        XCTAssertEqual(BlazenError.Cancelled(message: "cancelled").message, "cancelled")
    }
}

/// Trivial `StepHandler` that echoes the start-event payload back as a
/// `StopEvent` carrying `{"echo": <originalPayload>}`.
private final class EchoHandler: StepHandler, @unchecked Sendable {
    func invoke(event: Event) async throws -> StepOutput {
        let wrapped = "{\"echo\":\(event.dataJson)}"
        return .single(event: Event(eventType: "StopEvent", dataJson: wrapped))
    }
}
