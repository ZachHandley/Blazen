import XCTest
@testable import BlazenSwift
import UniFFIBlazen

/// Structural tests for the caller-error preservation path on the Swift
/// binding.
///
/// Background
///
/// Blazen 0.5.4 adds a typed `CallerError` variant to the workspace's
/// `BlazenError` enum. The contract: when a foreign-language `ToolHandler`
/// throws a typed Swift error, the Rust UniFFI adapter converts it into a
/// `BlazenError::CallerError { name: Option<String>, message: String,
/// propertiesJson: String }` and propagates it back through `run(userInput:)`
/// so the host caller can pattern-match the case, inspect the original
/// exception's class name, and JSON-decode the structured payload.
///
/// As of this scaffold's authoring, `./scripts/regen-bindings.sh` (which
/// regenerates Swift via the local `uniffi-bindgen`) hasn't yet been run
/// against the 0.5.4 UDL, so the generated `BlazenError.callerError` case
/// doesn't exist in `BlazenFFI.swift`. This test pins the EXPECTED post-regen
/// shape:
///
///   1. A typed Swift error a tool handler can throw.
///   2. A `ToolHandler` that throws that typed error.
///   3. The agent surfaces the failure as `BlazenError.callerError(name:
///      message: propertiesJson:)`, with `name == "MyCallerError"` and a
///      JSON-decodable `propertiesJson` blob carrying the payload.
///
/// The test is skipped right now via `XCTSkipIf(true, ...)` so the suite
/// remains green until the regen surfaces the variant. Once
/// `./scripts/regen-bindings.sh` lands the new case, remove the skip and
/// uncomment the real assertions below.
final class CallerErrorTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Blazen.initialize()
    }

    /// A typed Swift error a tool handler might throw to signal a
    /// domain-specific failure. The `payload` fields are the kind of
    /// structured data that should survive the FFI round-trip via the
    /// CallerError `propertiesJson` blob.
    struct MyCallerError: Error {
        let code: String
        let detail: String
    }

    /// Minimal `ToolHandler` whose `execute` always throws a typed
    /// `MyCallerError`. The Rust adapter is expected to reflect this back
    /// as `BlazenError.callerError(name: "MyCallerError", ...)` (or
    /// whatever stable-name format the regen picks for Swift types).
    private final class RaisingHandler: ToolHandler, @unchecked Sendable {
        func execute(toolName: String, argumentsJson: String) async throws -> String {
            throw MyCallerError(code: "E_DOMAIN", detail: "tool refused")
        }
    }

    /// EXPECTED post-regen behaviour:
    ///
    /// ```swift
    /// do {
    ///     _ = try await agent.run(userInput: "...")
    ///     XCTFail("expected callerError")
    /// } catch BlazenError.callerError(let name, _, let propertiesJson) {
    ///     XCTAssertEqual(name, "MyCallerError")
    ///     let data = propertiesJson.data(using: .utf8)!
    ///     let payload = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    ///     XCTAssertEqual(payload["code"] as? String, "E_DOMAIN")
    ///     XCTAssertEqual(payload["detail"] as? String, "tool refused")
    /// }
    /// ```
    ///
    /// Until the regen lands, the test is skipped so CI stays green.
    func testToolHandlerCallerErrorIsPreserved() async throws {
        try XCTSkipIf(true, "needs CallerError variant regen via ./scripts/regen-bindings.sh")

        // ---------------------------------------------------------------
        // Post-regen body. References that don't yet exist
        // (`BlazenError.callerError`, mock model factory) are kept inside
        // this skipped block as documentation; uncomment + activate once
        // the regen surfaces them.
        //
        // let model = try MockCompletionModel.with(modelId: "mock") // regen helper
        // defer { model.close() }
        //
        // let tool = Tool(
        //     name: "domain_op",
        //     description: "Always throws a typed caller error.",
        //     parametersJson: "{\"type\":\"object\",\"properties\":{}}"
        // )
        // let agent = Agent(
        //     model: model,
        //     systemPrompt: nil,
        //     tools: [tool],
        //     toolHandler: RaisingHandler(),
        //     maxIterations: 2
        // )
        //
        // do {
        //     _ = try await agent.run(userInput: "please call the tool")
        //     XCTFail("expected callerError to propagate")
        // } catch BlazenError.callerError(let name, _, let propertiesJson) {
        //     XCTAssertEqual(name, "MyCallerError")
        //     guard
        //         let data = propertiesJson.data(using: .utf8),
        //         let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        //     else {
        //         XCTFail("propertiesJson is not valid JSON: \(propertiesJson)")
        //         return
        //     }
        //     XCTAssertEqual(payload["code"] as? String, "E_DOMAIN")
        //     XCTAssertEqual(payload["detail"] as? String, "tool refused")
        // } catch {
        //     XCTFail("expected BlazenError.callerError, got \(type(of: error)): \(error)")
        // }

        // Keep `RaisingHandler` and `MyCallerError` referenced so the
        // compiler doesn't warn / drop them while parked behind the skip.
        _ = RaisingHandler()
        _ = MyCallerError(code: "E_DOMAIN", detail: "tool refused")
    }
}
