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

/// Local-inference backend identifier returned by
/// `ModelManager.loadFromHf(id:repo:options:)` and accepted as a forced
/// override on `HfLoadOptions.backendHint`.
///
/// `rawValue` matches the stable lower-case form emitted by the underlying
/// Rust `BackendHint::as_str()` so callers can round-trip through string
/// configs (env vars, JSON) without bespoke parsing.
public enum BackendHint: String, Sendable, Equatable, Hashable, CaseIterable {
    case mistralrs
    case candle
    case llamacpp

    fileprivate var ffi: BackendHintEnum {
        switch self {
        case .mistralrs: return .mistralrs
        case .candle: return .candle
        case .llamacpp: return .llamacpp
        }
    }

    fileprivate init(ffi: BackendHintEnum) {
        switch ffi {
        case .mistralrs: self = .mistralrs
        case .candle: self = .candle
        case .llamacpp: self = .llamacpp
        }
    }
}

/// Caller-supplied options for `ModelManager.loadFromHf(id:repo:options:)`.
///
/// Mirrors `UniFFIBlazen.HfLoadOptionsRecord` with every field optional and
/// defaulting to `nil`, so `HfLoadOptions()` selects the same behaviour as
/// the Rust `HfLoadOptions::default()`.
public struct HfLoadOptions: Sendable, Equatable, Hashable {
    public let backendHint: BackendHint?
    public let revision: String?
    public let hfToken: String?
    public let cacheDir: String?
    public let device: String?
    public let ggufFile: String?
    public let memoryEstimateBytes: UInt64?
    public let pool: String?

    public init(
        backendHint: BackendHint? = nil,
        revision: String? = nil,
        hfToken: String? = nil,
        cacheDir: String? = nil,
        device: String? = nil,
        ggufFile: String? = nil,
        memoryEstimateBytes: UInt64? = nil,
        pool: String? = nil
    ) {
        self.backendHint = backendHint
        self.revision = revision
        self.hfToken = hfToken
        self.cacheDir = cacheDir
        self.device = device
        self.ggufFile = ggufFile
        self.memoryEstimateBytes = memoryEstimateBytes
        self.pool = pool
    }

    fileprivate var record: HfLoadOptionsRecord {
        HfLoadOptionsRecord(
            backendHint: backendHint?.ffi,
            revision: revision,
            hfToken: hfToken,
            cacheDir: cacheDir,
            device: device,
            ggufFile: ggufFile,
            memoryEstimateBytes: memoryEstimateBytes,
            pool: pool
        )
    }
}

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

    /// Resolve a Hugging Face repo, pick a backend (or honour
    /// `options.backendHint`), register it under `id`, and load it through
    /// the per-pool budget tracker. Returns the backend that was actually
    /// selected.
    public func loadFromHf(
        id: String,
        repo: String,
        options: HfLoadOptions = HfLoadOptions()
    ) async throws -> BackendHint {
        let raw = try await inner.loadFromHf(
            id: id,
            repo: repo,
            options: options.record
        )
        // Why: UniFFI returns the backend as the lower-case stable string
        // (`BackendHint::as_str()`) rather than the enum itself, so we
        // parse here. An unknown value indicates a Rust/Swift drift bug
        // and is surfaced as `BlazenError.Internal` rather than crashing.
        guard let hint = BackendHint(rawValue: raw) else {
            throw BlazenError.Internal(
                message: "unknown BackendHint string returned by load_from_hf: \(raw)"
            )
        }
        return hint
    }
}
