import XCTest
@testable import BlazenSwift
import UniFFIBlazen

/// Smoke + shape tests for the control-plane UniFFI surface. These do
/// not require a running control-plane server — they exercise the
/// type-construction path through the generated bindings and confirm
/// that bogus endpoint URIs are rejected eagerly by both the client
/// and worker constructors. End-to-end behaviour is covered by the
/// Rust integration suite in blazen-controlplane.
final class ControlPlaneTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Blazen.initialize()
    }

    /// Worker capability struct constructs with the expected memberwise
    /// initializer and preserves field values.
    func testWorkerCapabilityConstructs() {
        let cap = ControlPlaneWorkerCapability(kind: "workflow:hello", version: 1)
        XCTAssertEqual(cap.kind, "workflow:hello")
        XCTAssertEqual(cap.version, 1)
    }

    /// Admission struct constructs in fixed mode and round-trips its
    /// optional fields without coercion.
    func testAdmissionFixedConstructs() {
        let admission = ControlPlaneAdmission(
            mode: .fixed,
            maxInFlight: 4,
            totalMb: nil
        )
        XCTAssertEqual(admission.mode, .fixed)
        XCTAssertEqual(admission.maxInFlight, 4)
        XCTAssertNil(admission.totalMb)
    }

    /// Reactive admission keeps both optional fields nil.
    func testAdmissionReactiveConstructs() {
        let admission = ControlPlaneAdmission(
            mode: .reactive,
            maxInFlight: nil,
            totalMb: nil
        )
        XCTAssertEqual(admission.mode, .reactive)
        XCTAssertNil(admission.maxInFlight)
        XCTAssertNil(admission.totalMb)
    }

    /// Submit request constructs with every field present.
    func testSubmitRequestConstructs() {
        let req = ControlPlaneSubmitRequest(
            workflowName: "summarize",
            inputJson: "{\"text\":\"hello\"}",
            workflowVersion: nil,
            requiredTags: ["region=us-west"],
            idempotencyKey: "dedupe-1",
            deadlineMs: 60_000,
            waitForWorker: true
        )
        XCTAssertEqual(req.workflowName, "summarize")
        XCTAssertTrue(req.waitForWorker)
        XCTAssertEqual(req.requiredTags, ["region=us-west"])
    }

    /// All five run-status variants are present on the generated enum.
    /// Compile-time pin against accidental variant deletions.
    func testRunStatusVariantsPresent() {
        let statuses: [ControlPlaneRunStatus] = [
            .pending, .running, .completed, .failed, .cancelled,
        ]
        XCTAssertEqual(statuses.count, 5)
    }

    /// The blocking client constructor surfaces a transport error when
    /// handed a malformed endpoint URI.
    func testClientConnectBlockingRejectsBadEndpoint() {
        XCTAssertThrowsError(try ControlPlaneClient.connectBlocking(endpoint: "not a uri", bearerToken: nil)) { err in
            // Any thrown error is acceptable here; we just need to be
            // sure the FFI surface didn't return a half-built handle.
            _ = err
        }
    }

    /// The blocking worker constructor validates the endpoint URI
    /// eagerly so callers don't enter a retry loop on misconfiguration.
    func testWorkerNewBlockingRejectsBadEndpoint() {
        XCTAssertThrowsError(
            try ControlPlaneWorker.newBlocking(
                endpoint: "not a uri",
                nodeId: "node-test",
                capabilities: [
                    ControlPlaneWorkerCapability(kind: "workflow:hello", version: 1),
                ],
                bearerToken: nil
            )
        ) { err in
            _ = err
        }
    }
}
