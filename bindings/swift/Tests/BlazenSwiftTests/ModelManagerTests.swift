import XCTest
@testable import BlazenSwift
import UniFFIBlazen

/// Smoke tests for the `ModelManager` wrapper. These never load a real
/// model — they confirm the FFI shape, constructor variants, and that
/// negative-path errors propagate as `BlazenError` instead of crashing.
final class ModelManagerTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Blazen.initialize()
    }

    /// All three constructors must succeed without throwing or trapping.
    /// `init(poolBudgets:)` parses pool labels and is the only one that
    /// can throw — feed it valid `"cpu"` / `"gpu:0"` keys.
    func testConstructorsDoNotThrow() throws {
        _ = ModelManager()
        _ = ModelManager(cpuRamGB: 8.0, gpuVramGB: 4.0)
        _ = try ModelManager(poolBudgets: ["cpu": 8.0, "gpu:0": 4.0])
    }

    /// `init(poolBudgets:)` rejects malformed pool labels with a
    /// `BlazenError.Validation` instead of trapping.
    func testInitPoolBudgetsRejectsBadLabel() {
        XCTAssertThrowsError(try ModelManager(poolBudgets: ["bogus": 1.0])) { err in
            guard case BlazenError.Validation = err else {
                XCTFail("expected BlazenError.Validation, got \(err)")
                return
            }
        }
    }

    /// A freshly constructed manager has no registered models, so the
    /// status snapshot is empty.
    func testStatusOnEmptyManagerIsEmpty() async {
        let manager = ModelManager()
        let statuses = await manager.status()
        XCTAssertTrue(statuses.isEmpty)
    }

    /// The no-arg constructor seeds the default `Cpu` + `Gpu(0)` pools
    /// with `UInt64.max` budgets.
    func testPoolsReturnsDefaultPools() {
        let manager = ModelManager()
        let pools = manager.pools()
        XCTAssertEqual(pools.count, 2)
        let labels = Set(pools.map(\.pool))
        XCTAssertEqual(labels, ["cpu", "gpu:0"])
        for pool in pools {
            XCTAssertEqual(pool.budgetBytes, UInt64.max)
        }
    }

    /// `load` on an unregistered model id must surface a `BlazenError`
    /// (the underlying `ModelManager::load` returns `Err(ModelNotFound)`).
    func testLoadUnknownModelThrows() async {
        let manager = ModelManager()
        do {
            try await manager.load("nonexistent-model-id")
            XCTFail("expected load() to throw")
        } catch is BlazenError {
            // Why: we only assert "is BlazenError" because the precise
            // variant (Validation vs Internal vs Workflow) depends on
            // how the upstream manager labels "model not registered"
            // errors and we don't want this smoke test to break on a
            // variant rename.
        } catch {
            XCTFail("expected BlazenError, got \(type(of: error)): \(error)")
        }
    }

    /// `isLoaded` for an unregistered model id is documented as a plain
    /// boolean query (no error) and must return `false`.
    func testIsLoadedUnknownModelReturnsFalse() async {
        let manager = ModelManager()
        let loaded = await manager.isLoaded("nonexistent-model-id")
        XCTAssertFalse(loaded)
    }

    /// `loadAdapter` on an unregistered model must throw rather than
    /// silently no-op. The fake adapter dir is never actually opened
    /// because the model-id lookup fails first.
    func testLoadAdapterUnknownModelThrows() async {
        let manager = ModelManager()
        do {
            _ = try await manager.loadAdapter(
                modelID: "nonexistent-model-id",
                adapterDir: "/nonexistent/adapter",
                options: AdapterOptions()
            )
            XCTFail("expected loadAdapter() to throw")
        } catch is BlazenError {
            // expected
        } catch {
            XCTFail("expected BlazenError, got \(type(of: error)): \(error)")
        }
    }

    /// `AdapterOptions` has documented default values that produce the
    /// expected wire form when narrowed to `Float`.
    func testAdapterOptionsDefaults() {
        let opts = AdapterOptions()
        XCTAssertEqual(opts.adapterId, "")
        XCTAssertEqual(opts.scale, 1.0)

        let custom = AdapterOptions(adapterId: "my-lora", scale: 0.75)
        XCTAssertEqual(custom.adapterId, "my-lora")
        XCTAssertEqual(custom.scale, 0.75)
    }
}
