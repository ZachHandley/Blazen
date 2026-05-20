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

    /// Run a LoRA fine-tune against `dataset` using the configured base
    /// model and hyperparameters. `onEvent` (if provided) receives every
    /// `TrainingEvent` published by the trainer on the underlying tokio
    /// worker; throwing from the closure cancels the run and surfaces as
    /// `BlazenError.Cancelled`.
    public func trainLora(
        config: TrainConfig,
        dataset: JsonlDataset,
        onEvent: (@Sendable (TrainingEvent) -> Void)? = nil
    ) async throws -> TrainedAdapter {
        let progress: ForeignTrainingProgress? = onEvent.map { TrainingProgressSink(callback: $0) }
        let record = try await inner.trainLora(
            config: config.record,
            dataset: dataset.inner,
            progress: progress
        )
        return TrainedAdapter(record: record)
    }
}

// MARK: - Training types

/// Learning-rate schedule shape passed to the LoRA trainer.
public enum SchedulerKind: String, Sendable, Equatable, Hashable, CaseIterable {
    case constant
    case linear
    case cosine

    fileprivate var ffi: SchedulerKindEnum {
        switch self {
        case .constant: return .constant
        case .linear: return .linear
        case .cosine: return .cosine
        }
    }
}

/// Mixed-precision mode for the LoRA trainer.
public enum MixedPrecision: String, Sendable, Equatable, Hashable, CaseIterable {
    case none
    case bf16

    fileprivate var ffi: MixedPrecisionEnum {
        switch self {
        case .none: return .none
        case .bf16: return .bf16
        }
    }
}

/// LoRA adapter shape: rank, scaling factor, dropout, and the list of
/// linear-projection modules to wrap.
public struct LoraConfig: Sendable, Equatable, Hashable {
    public let rank: UInt32
    public let alpha: Float
    public let dropout: Float
    public let targetModules: [String]

    public init(
        rank: UInt32 = 16,
        alpha: Float = 32.0,
        dropout: Float = 0.0,
        targetModules: [String] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetModules = targetModules
    }

    fileprivate var record: LoraConfigRecord {
        LoraConfigRecord(
            rank: rank,
            alpha: alpha,
            dropout: dropout,
            targetModules: targetModules
        )
    }
}

/// AdamW-style optimiser hyperparameters. `gradientClip` of `nil`
/// disables global-norm gradient clipping.
public struct OptimConfig: Sendable, Equatable, Hashable {
    public let learningRate: Double
    public let beta1: Double
    public let beta2: Double
    public let epsilon: Double
    public let weightDecay: Double
    public let gradientClip: Float?

    public init(
        learningRate: Double = 2.0e-4,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1.0e-8,
        weightDecay: Double = 0.0,
        gradientClip: Float? = 1.0
    ) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weightDecay = weightDecay
        self.gradientClip = gradientClip
    }

    fileprivate var record: OptimConfigRecord {
        OptimConfigRecord(
            learningRate: learningRate,
            beta1: beta1,
            beta2: beta2,
            epsilon: epsilon,
            weightDecay: weightDecay,
            gradientClip: gradientClip
        )
    }
}

/// Learning-rate-schedule configuration.
public struct SchedulerConfig: Sendable, Equatable, Hashable {
    public let kind: SchedulerKind
    public let warmupSteps: UInt32

    public init(kind: SchedulerKind = .cosine, warmupSteps: UInt32 = 0) {
        self.kind = kind
        self.warmupSteps = warmupSteps
    }

    fileprivate var record: SchedulerConfigRecord {
        SchedulerConfigRecord(kind: kind.ffi, warmupSteps: warmupSteps)
    }
}

/// Top-level training-run descriptor passed to
/// `ModelManager.trainLora(config:dataset:onEvent:)`.
///
/// `evalSteps` / `saveSteps` set to `nil` disable evaluation and
/// intermediate checkpoint emission respectively. `device` of `nil`
/// defers to the trainer's default device selection.
public struct TrainConfig: Sendable, Equatable, Hashable {
    public let baseModelRepo: String
    public let outputDir: String
    public let lora: LoraConfig
    public let optim: OptimConfig
    public let scheduler: SchedulerConfig
    public let maxSteps: UInt32
    public let batchSize: UInt32
    public let gradientAccumulationSteps: UInt32
    public let maxSeqLen: UInt32
    public let evalSteps: UInt32?
    public let saveSteps: UInt32?
    public let seed: UInt64
    public let mixedPrecision: MixedPrecision
    public let device: String?

    public init(
        baseModelRepo: String,
        outputDir: String,
        lora: LoraConfig = LoraConfig(),
        optim: OptimConfig = OptimConfig(),
        scheduler: SchedulerConfig = SchedulerConfig(),
        maxSteps: UInt32 = 1_000,
        batchSize: UInt32 = 1,
        gradientAccumulationSteps: UInt32 = 1,
        maxSeqLen: UInt32 = 2048,
        evalSteps: UInt32? = nil,
        saveSteps: UInt32? = nil,
        seed: UInt64 = 42,
        mixedPrecision: MixedPrecision = .none,
        device: String? = nil
    ) {
        self.baseModelRepo = baseModelRepo
        self.outputDir = outputDir
        self.lora = lora
        self.optim = optim
        self.scheduler = scheduler
        self.maxSteps = maxSteps
        self.batchSize = batchSize
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.maxSeqLen = maxSeqLen
        self.evalSteps = evalSteps
        self.saveSteps = saveSteps
        self.seed = seed
        self.mixedPrecision = mixedPrecision
        self.device = device
    }

    fileprivate var record: TrainConfigRecord {
        TrainConfigRecord(
            baseModelRepo: baseModelRepo,
            outputDir: outputDir,
            lora: lora.record,
            optim: optim.record,
            scheduler: scheduler.record,
            maxSteps: maxSteps,
            batchSize: batchSize,
            gradientAccumulationSteps: gradientAccumulationSteps,
            maxSeqLen: maxSeqLen,
            evalSteps: evalSteps,
            saveSteps: saveSteps,
            seed: seed,
            mixedPrecision: mixedPrecision.ffi,
            device: device
        )
    }
}

/// On-disk descriptor of a completed LoRA adapter, returned by
/// `ModelManager.trainLora(config:dataset:onEvent:)`.
public struct TrainedAdapter: Sendable, Equatable, Hashable {
    public let adapterDir: String
    public let finalLoss: Float
    public let totalSteps: UInt64

    public init(adapterDir: String, finalLoss: Float, totalSteps: UInt64) {
        self.adapterDir = adapterDir
        self.finalLoss = finalLoss
        self.totalSteps = totalSteps
    }

    fileprivate init(record: TrainedAdapterRecord) {
        self.adapterDir = record.adapterDir
        self.finalLoss = record.finalLoss
        self.totalSteps = record.totalSteps
    }
}

/// One observable event published during a training run. The trainer
/// emits exactly one `.started` then a series of `.stepCompleted` /
/// `.evaluating` / `.evalCompleted` / `.checkpointSaved` events, and
/// terminates with a single `.finished`.
public enum TrainingEvent: Sendable, Equatable, Hashable {
    case started(totalSteps: UInt64)
    case stepCompleted(step: UInt64, loss: Float, learningRate: Double, elapsedMs: UInt64)
    case evaluating(step: UInt64)
    case evalCompleted(step: UInt64, evalLoss: Float)
    case checkpointSaved(step: UInt64, path: String)
    case finished(finalLoss: Float, totalSteps: UInt64, adapterDir: String)

    fileprivate init(ffi: TrainingEventEnum) {
        switch ffi {
        case let .started(totalSteps):
            self = .started(totalSteps: totalSteps)
        case let .stepCompleted(step, loss, learningRate, elapsedMs):
            self = .stepCompleted(step: step, loss: loss, learningRate: learningRate, elapsedMs: elapsedMs)
        case let .evaluating(step):
            self = .evaluating(step: step)
        case let .evalCompleted(step, evalLoss):
            self = .evalCompleted(step: step, evalLoss: evalLoss)
        case let .checkpointSaved(step, path):
            self = .checkpointSaved(step: step, path: path)
        case let .finished(finalLoss, totalSteps, adapterDir):
            self = .finished(finalLoss: finalLoss, totalSteps: totalSteps, adapterDir: adapterDir)
        }
    }
}

/// JSONL-backed training dataset. Construct once and hand to one or more
/// `ModelManager.trainLora(...)` calls; the underlying handle is
/// reference-counted on the Rust side so it can be safely re-used.
public final class JsonlDataset: @unchecked Sendable {
    // Why: `UniffiJsonlDataset` is `@unchecked Sendable` upstream and
    // holds an `Arc`-shared dataset; the Swift wrapper adds no mutable
    // state so the same contract carries over.
    fileprivate let inner: UniffiJsonlDataset

    public init(
        path: String,
        tokenizerPath: String,
        chatTemplate: String? = nil,
        maxSeqLen: UInt32 = 2048,
        device: String? = nil,
        padTokenId: UInt32 = 0
    ) throws {
        self.inner = try UniffiJsonlDataset.fromPath(
            path: path,
            tokenizerPath: tokenizerPath,
            chatTemplate: chatTemplate,
            maxSeqLen: maxSeqLen,
            device: device,
            padTokenId: padTokenId
        )
    }

    public func isEmpty() -> Bool {
        inner.isEmpty()
    }

    public func count() -> UInt64 {
        inner.len()
    }
}

// Why: `ForeignTrainingProgress` is a sync UniFFI callback interface;
// the trainer invokes `onEvent` from a tokio worker thread. We wrap the
// caller's closure in a reference type that conforms to the protocol,
// translate the tagged UniFFI enum into the public `TrainingEvent`, and
// rely on the upstream contract that throwing cancels the run.
private final class TrainingProgressSink: ForeignTrainingProgress, @unchecked Sendable {
    private let callback: @Sendable (TrainingEvent) -> Void

    init(callback: @escaping @Sendable (TrainingEvent) -> Void) {
        self.callback = callback
    }

    func onEvent(event: TrainingEventEnum) throws {
        callback(TrainingEvent(ffi: event))
    }
}
