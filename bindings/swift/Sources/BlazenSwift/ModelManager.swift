import Foundation
import UniFFIBlazen

/// Per-model state snapshot returned by `ModelManager.status()`.
public typealias ModelStatus = UniFFIBlazen.ModelStatusRecord

/// Per-pool budget snapshot returned by `ModelManager.pools()`.
public typealias PoolStatus = UniFFIBlazen.PoolStatusRecord

/// Snapshot of a single mounted PEFT/LoRA adapter as reported by
/// `ModelManager.listAdapters(_:)`.
public typealias AdapterStatus = UniFFIBlazen.AdapterStatusRecord

/// Handle returned by `ForeignLocalModel.loadAdapter(adapterDir:options:)`.
/// `mountStrategy` is one of `"attached"`, `"rebuilt"`, `"merged"` —
/// kept as a string discriminator so future strategies don't break the FFI.
public typealias AdapterHandle = UniFFIBlazen.AdapterHandleRecord

/// Foreign-language `LocalModel` callback interface. Conforming reference
/// types can be registered with a `ModelManager` via
/// `registerLocal(id:model:memoryEstimateBytes:)`.
///
/// Re-exported under the `BlazenSwift` namespace so callers never reach into
/// `UniFFIBlazen` directly. Implementations that don't care about a verb
/// should return the documented neutral value (`false` for `isLoaded()`,
/// `nil` for `memoryBytes()`, `"cpu"` for `device()`, an empty list for
/// `listAdapters()`) or throw `BlazenError.Unsupported` from the adapter
/// verbs.
public typealias ForeignLocalModel = UniFFIBlazen.ForeignLocalModel

/// Adapter mount options for `ModelManager.loadAdapter(modelID:adapterDir:options:)`.
///
/// `adapterId` defaults to `""` which lets the backend auto-assign an id;
/// pass a non-empty value to mount under a stable, caller-chosen handle.
/// `scale` is expressed as `Double` for ergonomic parity with the rest of
/// this binding (Python / Node both expose it as `f64`); it is narrowed to
/// `Float` at the FFI boundary because the upstream Rust type is `f32`.
public struct AdapterOptions: Sendable, Equatable, Hashable {
    public var adapterId: String
    public var scale: Double

    public init(adapterId: String = "", scale: Double = 1.0) {
        self.adapterId = adapterId
        self.scale = scale
    }

    fileprivate var record: AdapterOptionsRecord {
        AdapterOptionsRecord(adapterId: adapterId, scale: Float(scale))
    }
}

/// Memory-budget-aware model manager with per-pool LRU eviction.
///
/// Register `ForeignLocalModel` handles with `registerLocal`, then drive
/// loads / unloads / adapter lifecycle. Idiomatic Swift wrapper over
/// `UniFFIBlazen.UniffiModelManager` that drops the `Uniffi` prefix and the
/// `modelId:` argument label in favour of an unlabelled `modelID` parameter
/// matching the rest of the BlazenSwift surface.
public final class ModelManager: @unchecked Sendable {
    // Why: `UniffiModelManager` is `@unchecked Sendable` upstream and
    // already serialises mutations behind a Rust `tokio::sync::Mutex` on the
    // inner `blazen_manager::ModelManager`. The Swift wrapper holds the
    // handle by reference and adds no Swift-side mutable state, so the same
    // unchecked-Sendable contract is sound here.
    private let inner: UniffiModelManager

    public init() {
        self.inner = UniffiModelManager()
    }

    public init(cpuRamGB: Double, gpuVramGB: Double) {
        self.inner = UniffiModelManager.withBudgetsGb(
            cpuRamGb: cpuRamGB,
            gpuVramGb: gpuVramGB
        )
    }

    public init(poolBudgets: [String: Double]) throws {
        self.inner = try UniffiModelManager.withPoolBudgets(perPoolBudgets: poolBudgets)
    }

    public func registerLocal(
        id: String,
        model: ForeignLocalModel,
        memoryEstimateBytes: UInt64
    ) async throws {
        try await inner.registerLocal(
            id: id,
            model: model,
            memoryEstimateBytes: memoryEstimateBytes
        )
    }

    public func load(_ modelID: String) async throws {
        try await inner.load(modelId: modelID)
    }

    public func unload(_ modelID: String) async throws {
        try await inner.unload(modelId: modelID)
    }

    public func isLoaded(_ modelID: String) async -> Bool {
        await inner.isLoaded(modelId: modelID)
    }

    public func ensureLoaded(_ modelID: String) async throws {
        try await inner.ensureLoaded(modelId: modelID)
    }

    public func status() async -> [ModelStatus] {
        await inner.status()
    }

    public func pools() -> [PoolStatus] {
        inner.pools()
    }

    public func usedBytes(_ pool: String) async throws -> UInt64 {
        try await inner.usedBytes(pool: pool)
    }

    public func availableBytes(_ pool: String) async throws -> UInt64 {
        try await inner.availableBytes(pool: pool)
    }

    /// Mount a PEFT-format LoRA adapter and return the adapter id reported
    /// by the backend (matches the value of `options.adapterId` when caller-
    /// provided and non-empty; otherwise an auto-assigned id).
    public func loadAdapter(
        modelID: String,
        adapterDir: String,
        options: AdapterOptions = AdapterOptions()
    ) async throws -> String {
        try await inner.loadAdapter(
            modelId: modelID,
            adapterDir: adapterDir,
            options: options.record
        )
    }

    public func unloadAdapter(modelID: String, adapterID: String) async throws {
        try await inner.unloadAdapter(modelId: modelID, adapterId: adapterID)
    }

    public func listAdapters(_ modelID: String) async throws -> [AdapterStatus] {
        try await inner.listAdapters(modelId: modelID)
    }
}
