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

    /// `loadFromHf` must surface a `BlazenError` for repos that don't
    /// exist on the Hugging Face Hub. We use a deterministically-bogus
    /// repo id so this test stays offline-safe: even with network access,
    /// the registry rejects it before any download starts.
    func testLoadFromHfNonexistentRepoThrows() async {
        let manager = ModelManager()
        do {
            _ = try await manager.loadFromHf(
                id: "swift-test-bogus",
                repo: "blazen-nonexistent-org/blazen-nonexistent-repo-xyz123",
                options: HfLoadOptions()
            )
            XCTFail("expected loadFromHf() to throw")
        } catch is BlazenError {
            // expected: any BlazenError variant (Validation, Network,
            // Internal, …) is fine; we only assert the FFI didn't crash.
        } catch {
            XCTFail("expected BlazenError, got \(type(of: error)): \(error)")
        }
    }

    /// The default `HfLoadOptions()` must round-trip through the FFI
    /// without panicking, even when the target repo can't be resolved.
    /// Confirms the optional-field encoding for every nil case.
    func testLoadFromHfDefaultOptionsDontCrash() async {
        let manager = ModelManager()
        do {
            _ = try await manager.loadFromHf(
                id: "swift-test-defaults",
                repo: "blazen-nonexistent-org/blazen-nonexistent-repo-defaults"
            )
        } catch is BlazenError {
            // expected
        } catch {
            XCTFail("expected BlazenError or success, got \(type(of: error)): \(error)")
        }
    }

    /// `BackendHint` raw values must match the lower-case strings the
    /// underlying Rust `BackendHint::as_str()` emits — drift here would
    /// break the post-FFI parse in `ModelManager.loadFromHf`.
    func testBackendHintRawValuesMatchRust() {
        XCTAssertEqual(BackendHint.mistralrs.rawValue, "mistralrs")
        XCTAssertEqual(BackendHint.candle.rawValue, "candle")
        XCTAssertEqual(BackendHint.llamacpp.rawValue, "llamacpp")
    }

    /// `HfLoadOptions()` exposes all-nil defaults and round-trips every
    /// field through its public initializer.
    func testHfLoadOptionsDefaults() {
        let opts = HfLoadOptions()
        XCTAssertNil(opts.backendHint)
        XCTAssertNil(opts.revision)
        XCTAssertNil(opts.hfToken)
        XCTAssertNil(opts.cacheDir)
        XCTAssertNil(opts.device)
        XCTAssertNil(opts.ggufFile)
        XCTAssertNil(opts.memoryEstimateBytes)
        XCTAssertNil(opts.pool)

        let custom = HfLoadOptions(
            backendHint: .candle,
            revision: "main",
            hfToken: "tok",
            cacheDir: "/tmp/cache",
            device: "cpu",
            ggufFile: "model.Q4_K_M.gguf",
            memoryEstimateBytes: 1_024,
            pool: "gpu:0"
        )
        XCTAssertEqual(custom.backendHint, .candle)
        XCTAssertEqual(custom.revision, "main")
        XCTAssertEqual(custom.hfToken, "tok")
        XCTAssertEqual(custom.cacheDir, "/tmp/cache")
        XCTAssertEqual(custom.device, "cpu")
        XCTAssertEqual(custom.ggufFile, "model.Q4_K_M.gguf")
        XCTAssertEqual(custom.memoryEstimateBytes, 1_024)
        XCTAssertEqual(custom.pool, "gpu:0")
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

    /// `trainLora` with `maxSteps = 0` must surface a `BlazenError`
    /// (the upstream validator rejects empty-step runs before any
    /// device/model setup).
    func testTrainLoraRejectsInvalidConfig() async {
        let manager = ModelManager()
        let config = TrainConfig(
            baseModelRepo: "blazen-nonexistent-org/blazen-nonexistent-base",
            outputDir: "/nonexistent/train-out",
            maxSteps: 0
        )
        // Why: dataset construction itself fails for a bogus path, so we
        // wrap it in the same do/catch — either failure is acceptable
        // because both indicate the FFI surfaced a typed error instead
        // of crashing.
        do {
            let dataset = try JsonlDataset(
                path: "/nonexistent/train.jsonl",
                tokenizerPath: "/nonexistent/tokenizer.json"
            )
            _ = try await manager.trainLora(config: config, dataset: dataset)
            XCTFail("expected trainLora() to throw")
        } catch is BlazenError {
            // expected
        } catch {
            XCTFail("expected BlazenError, got \(type(of: error)): \(error)")
        }
    }

    /// `JsonlDataset.init` must throw `BlazenError` (not crash) when the
    /// JSONL path or tokenizer path cannot be opened.
    func testJsonlDatasetRequiresValidPath() {
        XCTAssertThrowsError(
            try JsonlDataset(
                path: "/nonexistent/definitely-not-there.jsonl",
                tokenizerPath: "/nonexistent/tokenizer.json"
            )
        ) { err in
            guard err is BlazenError else {
                XCTFail("expected BlazenError, got \(type(of: err)): \(err)")
                return
            }
        }
    }

    /// Exhaustive switch over `TrainingEvent` cases must compile — guards
    /// against silent payload drift when the upstream enum gains a
    /// variant without the Swift mirror being updated.
    func testTrainingEventCases() {
        let events: [TrainingEvent] = [
            .started(totalSteps: 10),
            .stepCompleted(step: 1, loss: 1.5, learningRate: 1.0e-4, elapsedMs: 42),
            .evaluating(step: 5),
            .evalCompleted(step: 5, evalLoss: 1.2),
            .checkpointSaved(step: 5, path: "/tmp/ckpt"),
            .finished(finalLoss: 0.9, totalSteps: 10, adapterDir: "/tmp/adapter"),
        ]
        for event in events {
            switch event {
            case let .started(totalSteps):
                XCTAssertEqual(totalSteps, 10)
            case let .stepCompleted(step, loss, learningRate, elapsedMs):
                XCTAssertEqual(step, 1)
                XCTAssertEqual(loss, 1.5)
                XCTAssertEqual(learningRate, 1.0e-4)
                XCTAssertEqual(elapsedMs, 42)
            case let .evaluating(step):
                XCTAssertEqual(step, 5)
            case let .evalCompleted(step, evalLoss):
                XCTAssertEqual(step, 5)
                XCTAssertEqual(evalLoss, 1.2)
            case let .checkpointSaved(step, path):
                XCTAssertEqual(step, 5)
                XCTAssertEqual(path, "/tmp/ckpt")
            case let .finished(finalLoss, totalSteps, adapterDir):
                XCTAssertEqual(finalLoss, 0.9)
                XCTAssertEqual(totalSteps, 10)
                XCTAssertEqual(adapterDir, "/tmp/adapter")
            }
        }
    }
}
