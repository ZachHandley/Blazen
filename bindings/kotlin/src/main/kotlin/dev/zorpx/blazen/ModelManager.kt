package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.AdapterHandleRecord
import dev.zorpx.blazen.uniffi.AdapterOptionsRecord
import dev.zorpx.blazen.uniffi.AdapterStatusRecord
import dev.zorpx.blazen.uniffi.BackendHintEnum
import dev.zorpx.blazen.uniffi.DpoConfigRecord
import dev.zorpx.blazen.uniffi.FullFineTuneConfigRecord
import dev.zorpx.blazen.uniffi.FullFineTuneResultRecord
import dev.zorpx.blazen.uniffi.HfLoadOptionsRecord
import dev.zorpx.blazen.uniffi.KtoConfigRecord
import dev.zorpx.blazen.uniffi.LoraConfigRecord
import dev.zorpx.blazen.uniffi.MixedPrecisionEnum
import dev.zorpx.blazen.uniffi.ModelStatusRecord
import dev.zorpx.blazen.uniffi.OptimConfigRecord
import dev.zorpx.blazen.uniffi.OrpoConfigRecord
import dev.zorpx.blazen.uniffi.PoolStatusRecord
import dev.zorpx.blazen.uniffi.SchedulerConfigRecord
import dev.zorpx.blazen.uniffi.SchedulerKindEnum
import dev.zorpx.blazen.uniffi.SimpoConfigRecord
import dev.zorpx.blazen.uniffi.TrainConfigRecord
import dev.zorpx.blazen.uniffi.TrainCoreConfigRecord
import dev.zorpx.blazen.uniffi.TrainedAdapterRecord
import dev.zorpx.blazen.uniffi.TrainingEventEnum
import dev.zorpx.blazen.uniffi.UniffiJsonlDataset
import dev.zorpx.blazen.uniffi.UniffiModelManager
import dev.zorpx.blazen.uniffi.UniffiPreferenceJsonlDataset
import dev.zorpx.blazen.uniffi.UniffiRatedJsonlDataset
import java.util.UUID
import dev.zorpx.blazen.uniffi.ForeignLocalModel as UniffiForeignLocalModel
import dev.zorpx.blazen.uniffi.ForeignTrainingProgress as UniffiForeignTrainingProgress

/**
 * Memory-budget-aware model manager with per-pool LRU eviction.
 *
 * Mirrors the Python `blazen.ModelManager` surface and the
 * `crates/blazen-uniffi/src/manager.rs` Rust verbs. Foreign-implemented
 * [ForeignLocalModel]s are registered against a manager; loads, unloads,
 * status snapshots, and PEFT LoRA adapter mounts then flow through the
 * shared per-pool budget book-keeping in the Rust runtime.
 *
 * All async methods are exposed as Kotlin `suspend` functions and inherit
 * the caller's `CoroutineScope` and dispatcher — the wrapper does not
 * impose a `Dispatchers.IO` hop because the generated UniFFI glue already
 * dispatches onto the Tokio runtime backing the native crate.
 *
 * Exceptions surface as `dev.zorpx.blazen.uniffi.BlazenException`; this
 * wrapper deliberately does not translate them into the hand-written
 * [BlazenException] sealed class because no existing wrapper in this
 * binding does that translation either and doing it asymmetrically here
 * would be confusing.
 */
public class ModelManager : AutoCloseable {
    private val inner: UniffiModelManager

    /**
     * Construct a manager with no budget enforcement (both `cpu` and
     * `gpu:0` pools seeded with `ULong.MAX_VALUE`).
     */
    public constructor() {
        this.inner = UniffiModelManager()
    }

    /**
     * Construct a manager with one CPU-pool budget and one GPU-pool
     * (`gpu:0`) budget, both expressed in gigabytes.
     */
    public constructor(cpuRamGb: Double, gpuVramGb: Double) {
        this.inner = UniffiModelManager.withBudgetsGb(cpuRamGb, gpuVramGb)
    }

    /**
     * Construct a manager with explicit per-pool budgets.
     *
     * Keys are pool labels (`"cpu"`, `"gpu"`, `"gpu:0"`, `"gpu:1"`, ...);
     * values are budgets in **gigabytes** — matches the Python binding's
     * `pool_budgets` ergonomics so trivial values stay readable.
     */
    public constructor(poolBudgets: Map<String, Double>) {
        this.inner = UniffiModelManager.withPoolBudgets(poolBudgets)
    }

    /**
     * Register a foreign-implemented [ForeignLocalModel] under [id].
     *
     * [memoryEstimateBytes] is the model's estimated footprint and is
     * charged against the pool derived from the foreign model's `device()`
     * when it is loaded.
     */
    public suspend fun registerLocal(
        id: String,
        model: ForeignLocalModel,
        memoryEstimateBytes: ULong,
    ) {
        inner.registerLocal(id, ForeignLocalModelBridge(model), memoryEstimateBytes)
    }

    public suspend fun load(modelId: String) {
        inner.load(modelId)
    }

    /**
     * Probe a Hugging Face repo, pick a local-inference backend, build the
     * provider, and register it under [id]. Returns the chosen
     * [BackendHint]. The model starts unloaded — call [load] or
     * [ensureLoaded] to materialize it.
     *
     * Errors on empty repo id, gated/missing repo, PEFT-adapter-only repo
     * (use [loadAdapter] instead), missing backend feature, or any provider
     * construction failure.
     */
    public suspend fun loadFromHf(
        id: String,
        repo: String,
        options: HfLoadOptions = HfLoadOptions(),
    ): BackendHint = BackendHint.fromWire(inner.loadFromHf(id, repo, options.toRecord()))

    public suspend fun unload(modelId: String) {
        inner.unload(modelId)
    }

    public suspend fun isLoaded(modelId: String): Boolean = inner.isLoaded(modelId)

    public suspend fun ensureLoaded(modelId: String) {
        inner.ensureLoaded(modelId)
    }

    public suspend fun status(): List<ModelStatus> =
        inner.status().map { it.toDomain() }

    /** List configured pools and their budgets in bytes. */
    public fun pools(): List<PoolStatus> =
        inner.pools().map { PoolStatus(pool = it.pool, budgetBytes = it.budgetBytes) }

    public suspend fun usedBytes(pool: String): ULong = inner.usedBytes(pool)

    public suspend fun availableBytes(pool: String): ULong = inner.availableBytes(pool)

    /**
     * Mount a PEFT-format LoRA adapter and return the adapter id reported
     * by the backend.
     */
    public suspend fun loadAdapter(
        modelId: String,
        adapterDir: String,
        options: AdapterOptions = AdapterOptions(),
    ): String = inner.loadAdapter(modelId, adapterDir, options.toRecord())

    public suspend fun unloadAdapter(modelId: String, adapterId: String) {
        inner.unloadAdapter(modelId, adapterId)
    }

    public suspend fun listAdapters(modelId: String): List<AdapterStatus> =
        inner.listAdapters(modelId).map { it.toDomain() }

    /**
     * Train a LoRA adapter end-to-end on the configured base model.
     *
     * Downloads the base model from HuggingFace (cached), runs the AdamW +
     * LoRA training loop driven by [dataset], and writes the resulting
     * PEFT-format adapter to `config.outputDir`. The returned
     * [TrainedAdapter] points at an on-disk adapter directory immediately
     * mountable via [loadAdapter] on a compatible backend.
     *
     * If [onEvent] is provided it is invoked synchronously for each
     * Started / StepCompleted / Evaluating / EvalCompleted /
     * CheckpointSaved / Finished transition. Throwing from the callback
     * cancels the run; the trainer surfaces it as a Blazen exception.
     */
    public suspend fun trainLora(
        config: TrainConfig,
        dataset: JsonlDataset,
        onEvent: ((TrainingEvent) -> Unit)? = null,
    ): TrainedAdapter {
        val progress = onEvent?.let { TrainingProgressBridge(it) }
        return inner.trainLora(config.toRecord(), dataset.inner, progress).toDomain()
    }

    /**
     * Train a LoRA adapter with Direct Preference Optimization (DPO).
     *
     * Requires a frozen reference model — see [DpoConfig.referenceModelRepo].
     * The PEFT-format adapter is written to `config.core.outputDir` and the
     * returned [TrainedAdapter] descriptor is immediately mountable via
     * [loadAdapter].
     *
     * If [onEvent] is provided it is invoked synchronously for each
     * training-loop transition; throwing from the callback cancels the run.
     */
    public suspend fun trainDpo(
        config: DpoConfig,
        dataset: PreferenceJsonlDataset,
        onEvent: ((TrainingEvent) -> Unit)? = null,
    ): TrainedAdapter {
        val progress = onEvent?.let { TrainingProgressBridge(it) }
        return inner.trainDpo(config.toRecord(), dataset.inner, progress).toDomain()
    }

    /**
     * Train a LoRA adapter with Odds-Ratio Preference Optimization (ORPO).
     *
     * Reference-free; combines an SFT loss on chosen responses with an
     * odds-ratio penalty weighted by [OrpoConfig.lambda].
     */
    public suspend fun trainOrpo(
        config: OrpoConfig,
        dataset: PreferenceJsonlDataset,
        onEvent: ((TrainingEvent) -> Unit)? = null,
    ): TrainedAdapter {
        val progress = onEvent?.let { TrainingProgressBridge(it) }
        return inner.trainOrpo(config.toRecord(), dataset.inner, progress).toDomain()
    }

    /**
     * Train a LoRA adapter with Simple Preference Optimization (SimPO).
     *
     * Reference-free, length-normalized. Defaults follow TRL `main`
     * (`beta = 2.0`, `gamma = 1.0`).
     */
    public suspend fun trainSimpo(
        config: SimpoConfig,
        dataset: PreferenceJsonlDataset,
        onEvent: ((TrainingEvent) -> Unit)? = null,
    ): TrainedAdapter {
        val progress = onEvent?.let { TrainingProgressBridge(it) }
        return inner.trainSimpo(config.toRecord(), dataset.inner, progress).toDomain()
    }

    /**
     * Train a LoRA adapter with Kahneman-Tversky Optimization (KTO).
     *
     * Like DPO, KTO needs a frozen reference model; unlike DPO the dataset
     * is a [RatedJsonlDataset] of `(prompt, completion, desirable)` triples
     * rather than chosen/rejected pairs.
     */
    public suspend fun trainKto(
        config: KtoConfig,
        dataset: RatedJsonlDataset,
        onEvent: ((TrainingEvent) -> Unit)? = null,
    ): TrainedAdapter {
        val progress = onEvent?.let { TrainingProgressBridge(it) }
        return inner.trainKto(config.toRecord(), dataset.inner, progress).toDomain()
    }

    /**
     * Full fine-tune (no PEFT adapter — every parameter trains).
     *
     * Writes the entire model's weights to [FullFineTuneConfig.core]'s
     * `outputDir` and returns a [FullFineTuneResult] descriptor pointing at
     * the resulting directory.
     *
     * `gradient_checkpointing = true` is accepted for forward compatibility
     * but rejected at init time with a `BlazenException.Validation` —
     * candle 0.10.2 has no activation-checkpointing primitive.
     */
    public suspend fun fineTune(
        config: FullFineTuneConfig,
        dataset: JsonlDataset,
        onEvent: ((TrainingEvent) -> Unit)? = null,
    ): FullFineTuneResult {
        val progress = onEvent?.let { TrainingProgressBridge(it) }
        return inner.fineTune(config.toRecord(), dataset.inner, progress).toDomain()
    }

    override fun close() {
        inner.close()
    }
}

/** Per-model state snapshot returned by [ModelManager.status]. */
public data class ModelStatus(
    val id: String,
    val loaded: Boolean,
    val memoryEstimateBytes: ULong,
    val pool: String,
    val adapters: List<AdapterStatus>,
)

/** Per-pool budget snapshot returned by [ModelManager.pools]. */
public data class PoolStatus(
    val pool: String,
    val budgetBytes: ULong,
)

/**
 * Adapter mount options handed to [ModelManager.loadAdapter].
 *
 * [adapterId] defaults to a random UUID so callers can mount one-off
 * adapters without inventing a name; supply an explicit id when you need
 * to unload by name later. [scale] is the per-adapter blend factor (1.0
 * applies the adapter at full strength).
 */
public data class AdapterOptions(
    val adapterId: String = UUID.randomUUID().toString(),
    val scale: Double = 1.0,
)

/** Snapshot of a single mounted adapter. */
public data class AdapterStatus(
    val adapterId: String,
    val scale: Double,
    val sourceDir: String,
    val memoryBytes: ULong,
)

/**
 * Result returned by [ForeignLocalModel.loadAdapter].
 *
 * [mountStrategy] is one of `"attached"`, `"rebuilt"`, `"merged"` — the
 * string mirrors the upstream `AdapterMountStrategy` enum and is kept as
 * a discriminator so adding a new strategy does not break this contract.
 */
public data class AdapterHandle(
    val adapterId: String,
    val memoryBytes: ULong,
    val mountStrategy: String,
)

/**
 * Foreign-language implementation of a local (on-device) model.
 *
 * Mirrors the upstream `blazen_llm::LocalModel` trait in FFI-friendly
 * form: paths are strings and [device] returns a label (`"cpu"`,
 * `"cuda:0"`, `"metal"`, ...) that the Rust side parses back into a
 * `Device`. Implementors that do not care about a verb should return a
 * neutral value (`false` for [isLoaded], `null` for [memoryBytes],
 * `"cpu"` for [device], an empty list for [listAdapters]) or throw
 * [dev.zorpx.blazen.uniffi.BlazenException.Unsupported] from the adapter
 * verbs.
 */
public interface ForeignLocalModel {
    public suspend fun load()

    public suspend fun unload()

    public suspend fun isLoaded(): Boolean

    public fun device(): String

    public suspend fun memoryBytes(): ULong?

    public suspend fun loadAdapter(adapterDir: String, options: AdapterOptions): AdapterHandle

    public suspend fun unloadAdapter(handle: AdapterHandle)

    public suspend fun listAdapters(): List<AdapterStatus>
}

/**
 * Local-inference backend identifier returned by [ModelManager.loadFromHf]
 * and accepted as a forced override on [HfLoadOptions.backendHint].
 *
 * [value] is the lower-case stable string the Rust runtime emits and
 * parses (`"mistralrs"` / `"candle"` / `"llamacpp"`) — kept on the enum so
 * callers can round-trip to logs/config without re-implementing the table.
 */
public enum class BackendHint(public val value: String) {
    MISTRALRS("mistralrs"),
    CANDLE("candle"),
    LLAMACPP("llamacpp"),
    ;

    internal fun toEnum(): BackendHintEnum =
        when (this) {
            MISTRALRS -> BackendHintEnum.MISTRALRS
            CANDLE -> BackendHintEnum.CANDLE
            LLAMACPP -> BackendHintEnum.LLAMACPP
        }

    public companion object {
        internal fun fromWire(s: String): BackendHint =
            values().firstOrNull { it.value == s }
                ?: throw IllegalStateException("unknown backend hint from native side: '$s'")
    }
}

/**
 * Options for [ModelManager.loadFromHf]. Every field is optional and
 * mirrors `blazen_manager::hf_loader::HfLoadOptions`. [pool] is a label
 * (`"cpu"`, `"gpu"`, `"gpu:N"`) and defaults to `"cpu"` on the Rust side
 * when left null.
 */
public data class HfLoadOptions(
    val backendHint: BackendHint? = null,
    val revision: String? = null,
    val hfToken: String? = null,
    val cacheDir: String? = null,
    val device: String? = null,
    val ggufFile: String? = null,
    val memoryEstimateBytes: ULong? = null,
    val pool: String? = null,
)

private fun HfLoadOptions.toRecord(): HfLoadOptionsRecord =
    HfLoadOptionsRecord(
        backendHint = backendHint?.toEnum(),
        revision = revision,
        hfToken = hfToken,
        cacheDir = cacheDir,
        device = device,
        ggufFile = ggufFile,
        memoryEstimateBytes = memoryEstimateBytes,
        pool = pool,
    )

private fun AdapterOptions.toRecord(): AdapterOptionsRecord =
    AdapterOptionsRecord(adapterId = adapterId, scale = scale.toFloat())

private fun AdapterOptionsRecord.toDomain(): AdapterOptions =
    AdapterOptions(adapterId = adapterId, scale = scale.toDouble())

private fun AdapterStatusRecord.toDomain(): AdapterStatus =
    AdapterStatus(
        adapterId = adapterId,
        scale = scale.toDouble(),
        sourceDir = sourceDir,
        memoryBytes = memoryBytes,
    )

private fun AdapterHandle.toRecord(): AdapterHandleRecord =
    AdapterHandleRecord(adapterId = adapterId, memoryBytes = memoryBytes, mountStrategy = mountStrategy)

private fun AdapterHandleRecord.toDomain(): AdapterHandle =
    AdapterHandle(adapterId = adapterId, memoryBytes = memoryBytes, mountStrategy = mountStrategy)

private fun ModelStatusRecord.toDomain(): ModelStatus =
    ModelStatus(
        id = id,
        loaded = loaded,
        memoryEstimateBytes = memoryEstimateBytes,
        pool = pool,
        adapters = adapters.map { it.toDomain() },
    )

@Suppress("unused")
private fun PoolStatusRecord.toDomain(): PoolStatus =
    PoolStatus(pool = pool, budgetBytes = budgetBytes)

/**
 * Learning-rate scheduler shape passed to [SchedulerConfig.kind].
 *
 * [value] is the lower-case stable string the upstream Rust trainer logs
 * and consumes so callers can round-trip to configuration files without
 * re-implementing the table.
 */
public enum class SchedulerKind(public val value: String) {
    CONSTANT("constant"),
    LINEAR("linear"),
    COSINE("cosine"),
    ;

    internal fun toEnum(): SchedulerKindEnum =
        when (this) {
            CONSTANT -> SchedulerKindEnum.CONSTANT
            LINEAR -> SchedulerKindEnum.LINEAR
            COSINE -> SchedulerKindEnum.COSINE
        }
}

/**
 * Mixed-precision strategy passed to [TrainConfig.mixedPrecision].
 *
 * [value] is the lower-case stable string mirrored from the upstream
 * trainer (`"none"` / `"bf16"`).
 */
public enum class MixedPrecision(public val value: String) {
    NONE("none"),
    BF16("bf16"),
    ;

    internal fun toEnum(): MixedPrecisionEnum =
        when (this) {
            NONE -> MixedPrecisionEnum.NONE
            BF16 -> MixedPrecisionEnum.BF16
        }
}

/** LoRA hyperparameters. Defaults target the four attention projections. */
public data class LoraConfig(
    val rank: UInt = 16u,
    val alpha: Float = 32.0f,
    val dropout: Float = 0.0f,
    val targetModules: List<String> = listOf("q_proj", "k_proj", "v_proj", "o_proj"),
)

/** AdamW optimizer hyperparameters. */
public data class OptimConfig(
    val learningRate: Double = 2e-4,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1e-8,
    val weightDecay: Double = 0.0,
    val gradientClip: Float? = 1.0f,
)

/** Learning-rate scheduler configuration. */
public data class SchedulerConfig(
    val kind: SchedulerKind = SchedulerKind.COSINE,
    val warmupSteps: UInt = 0u,
)

/** Full configuration for one [ModelManager.trainLora] run. */
public data class TrainConfig(
    val baseModelRepo: String,
    val outputDir: String,
    val lora: LoraConfig = LoraConfig(),
    val optim: OptimConfig = OptimConfig(),
    val scheduler: SchedulerConfig = SchedulerConfig(),
    val maxSteps: UInt = 1000u,
    val batchSize: UInt = 4u,
    val gradientAccumulationSteps: UInt = 1u,
    val maxSeqLen: UInt = 2048u,
    val evalSteps: UInt? = null,
    val saveSteps: UInt? = null,
    val seed: ULong = 42uL,
    val mixedPrecision: MixedPrecision = MixedPrecision.BF16,
    val device: String? = null,
)

/**
 * Shared training hyperparameters used by every non-SFT trainer (DPO /
 * ORPO / SimPO / KTO / full fine-tune). Mirrors the upstream
 * `TrainCoreConfig` (= [TrainConfig] minus the PEFT-specific [LoraConfig]).
 *
 * Defaults match the upstream Rust struct: smaller per-step batches and
 * larger accumulation than [TrainConfig] because the preference trainers
 * carry two forward passes (chosen + rejected) through the model.
 */
public data class TrainCoreConfig(
    val baseModelRepo: String,
    val outputDir: String,
    val baseModelRevision: String? = null,
    val maxSteps: UInt = 1000u,
    val batchSize: UInt = 1u,
    val gradientAccumulationSteps: UInt = 8u,
    val maxSeqLen: UInt = 1024u,
    val evalSteps: UInt? = null,
    val saveSteps: UInt? = null,
    val seed: ULong = 42uL,
    val mixedPrecision: MixedPrecision = MixedPrecision.BF16,
    val device: String? = null,
    val optim: OptimConfig = OptimConfig(),
    val scheduler: SchedulerConfig = SchedulerConfig(),
)

/**
 * Direct Preference Optimization configuration.
 *
 * Requires a frozen reference model; when [referenceModelRepo] is `null`
 * the trainer reuses `core.baseModelRepo`.
 */
public data class DpoConfig(
    val core: TrainCoreConfig,
    val lora: LoraConfig = LoraConfig(),
    val beta: Float = 0.1f,
    val labelSmoothing: Float = 0.0f,
    val referenceModelRepo: String? = null,
    val referenceModelRevision: String? = null,
)

/**
 * Odds-Ratio Preference Optimization configuration.
 *
 * Reference-free; combines an SFT loss on chosen responses with an
 * odds-ratio penalty weighted by [lambda].
 */
public data class OrpoConfig(
    val core: TrainCoreConfig,
    val lora: LoraConfig = LoraConfig(),
    val lambda: Float = 0.1f,
)

/**
 * Simple Preference Optimization (SimPO) configuration.
 *
 * Reference-free, length-normalized. Defaults follow TRL `main`
 * (`beta = 2.0`, `gamma = 1.0`).
 */
public data class SimpoConfig(
    val core: TrainCoreConfig,
    val lora: LoraConfig = LoraConfig(),
    val beta: Float = 2.0f,
    val gamma: Float = 1.0f,
)

/**
 * Kahneman-Tversky Optimization configuration.
 *
 * Like DPO, KTO needs a frozen reference model (defaults to
 * `core.baseModelRepo`); unlike DPO the dataset schema is a
 * `(prompt, completion, desirable)` triple held in a [RatedJsonlDataset].
 *
 * [lambdaD] and [lambdaU] are the per-sign loss weights — desirable and
 * undesirable rows respectively. The upstream defaults (both `1.0`)
 * weight the two signs equally.
 */
public data class KtoConfig(
    val core: TrainCoreConfig,
    val lora: LoraConfig = LoraConfig(),
    val beta: Float = 0.1f,
    val lambdaD: Float = 1.0f,
    val lambdaU: Float = 1.0f,
    val referenceModelRepo: String? = null,
    val referenceModelRevision: String? = null,
)

/**
 * Full fine-tune configuration (no LoRA — every parameter trains).
 *
 * [gradientCheckpointing] is accepted for forward compatibility but the
 * trainer rejects it at init time with `BlazenException.Validation` —
 * candle 0.10.2 has no activation-checkpointing primitive.
 */
public data class FullFineTuneConfig(
    val core: TrainCoreConfig,
    val gradientCheckpointing: Boolean = false,
)

/** On-disk descriptor returned by [ModelManager.trainLora]. */
public data class TrainedAdapter(
    val adapterDir: String,
    val finalLoss: Float,
    val totalSteps: ULong,
)

/**
 * On-disk descriptor returned by [ModelManager.fineTune].
 *
 * Unlike [TrainedAdapter], no PEFT adapter is written — the entire
 * model's weights are saved to [outputDir] directly.
 */
public data class FullFineTuneResult(
    val outputDir: String,
    val finalLoss: Float,
    val stepsCompleted: ULong,
)

/**
 * One observable event emitted during a [ModelManager.trainLora] run.
 *
 * Sealed so `when`-matching on the callback argument is exhaustive — the
 * compiler will flag a missing branch when the upstream trainer grows a
 * new event variant.
 */
public sealed class TrainingEvent {
    public data class Started(val totalSteps: ULong) : TrainingEvent()

    public data class StepCompleted(
        val step: ULong,
        val loss: Float,
        val learningRate: Double,
        val elapsedMs: ULong,
    ) : TrainingEvent()

    public data class Evaluating(val step: ULong) : TrainingEvent()

    public data class EvalCompleted(val step: ULong, val evalLoss: Float) : TrainingEvent()

    public data class CheckpointSaved(val step: ULong, val path: String) : TrainingEvent()

    public data class Finished(
        val finalLoss: Float,
        val totalSteps: ULong,
        val adapterDir: String,
    ) : TrainingEvent()
}

/**
 * Tokenized JSONL training corpus handed to [ModelManager.trainLora].
 *
 * Construct via [JsonlDataset.fromPath]; the underlying native handle is
 * owned by the Rust runtime and freed when this wrapper is garbage
 * collected (matches the UniFFI-generated object lifecycle).
 */
public class JsonlDataset internal constructor(
    internal val inner: UniffiJsonlDataset,
) {
    public companion object {
        /**
         * Load a JSONL training file using the tokenizer at [tokenizerPath].
         *
         * [chatTemplate] is optional Jinja2 from `tokenizer_config.json`
         * and is required if any row uses the OpenAI `messages` shape.
         * [device] matches the trainer device strings — `"cpu"`,
         * `"cuda"` / `"cuda:N"`, `"metal"` / `"metal:N"` (default `"cpu"`).
         */
        public fun fromPath(
            path: String,
            tokenizerPath: String,
            chatTemplate: String? = null,
            maxSeqLen: UInt = 2048u,
            device: String? = null,
            padTokenId: UInt = 0u,
        ): JsonlDataset =
            JsonlDataset(
                UniffiJsonlDataset.fromPath(
                    path = path,
                    tokenizerPath = tokenizerPath,
                    chatTemplate = chatTemplate,
                    maxSeqLen = maxSeqLen,
                    device = device,
                    padTokenId = padTokenId,
                ),
            )
    }
}

/**
 * Tokenized JSONL preference-pair corpus handed to [ModelManager.trainDpo],
 * [ModelManager.trainOrpo], and [ModelManager.trainSimpo].
 *
 * Each row of the input file must deserialize to one of:
 *   * `{"prompt": "...", "chosen": "...", "rejected": "..."}`
 *   * `{"messages": [...], "chosen": "...", "rejected": "..."}` —
 *     requires [chatTemplate].
 *
 * Construct via [PreferenceJsonlDataset.fromPath]; the underlying native
 * handle is owned by the Rust runtime and freed when this wrapper is
 * garbage collected (UniFFI-managed lifecycle).
 */
public class PreferenceJsonlDataset internal constructor(
    internal val inner: UniffiPreferenceJsonlDataset,
) {
    public companion object {
        /**
         * Load a preference-pair JSONL file. Argument shape mirrors
         * [JsonlDataset.fromPath].
         */
        public fun fromPath(
            path: String,
            tokenizerPath: String,
            chatTemplate: String? = null,
            maxSeqLen: UInt = 2048u,
            device: String? = null,
            padTokenId: UInt = 0u,
        ): PreferenceJsonlDataset =
            PreferenceJsonlDataset(
                UniffiPreferenceJsonlDataset.fromPath(
                    path = path,
                    tokenizerPath = tokenizerPath,
                    chatTemplate = chatTemplate,
                    maxSeqLen = maxSeqLen,
                    device = device,
                    padTokenId = padTokenId,
                ),
            )
    }
}

/**
 * Tokenized JSONL rated single-completion corpus handed to
 * [ModelManager.trainKto].
 *
 * Each row of the input file must deserialize to one of:
 *   * `{"prompt": "...", "completion": "...", "label": true|false}`
 *   * `{"messages": [...], "completion": "...", "label": ...}` —
 *     requires [chatTemplate].
 *
 * The `label` discriminator is the desirability flag KTO requires; rows
 * with `label = true` are "desirable" and contribute to the
 * [KtoConfig.lambdaD]-weighted term, rows with `label = false` to the
 * [KtoConfig.lambdaU]-weighted term.
 */
public class RatedJsonlDataset internal constructor(
    internal val inner: UniffiRatedJsonlDataset,
) {
    public companion object {
        /**
         * Load a rated JSONL file. Argument shape mirrors
         * [JsonlDataset.fromPath].
         */
        public fun fromPath(
            path: String,
            tokenizerPath: String,
            chatTemplate: String? = null,
            maxSeqLen: UInt = 2048u,
            device: String? = null,
            padTokenId: UInt = 0u,
        ): RatedJsonlDataset =
            RatedJsonlDataset(
                UniffiRatedJsonlDataset.fromPath(
                    path = path,
                    tokenizerPath = tokenizerPath,
                    chatTemplate = chatTemplate,
                    maxSeqLen = maxSeqLen,
                    device = device,
                    padTokenId = padTokenId,
                ),
            )
    }
}

private fun LoraConfig.toRecord(): LoraConfigRecord =
    LoraConfigRecord(
        rank = rank,
        alpha = alpha,
        dropout = dropout,
        targetModules = targetModules,
    )

private fun OptimConfig.toRecord(): OptimConfigRecord =
    OptimConfigRecord(
        learningRate = learningRate,
        beta1 = beta1,
        beta2 = beta2,
        epsilon = epsilon,
        weightDecay = weightDecay,
        gradientClip = gradientClip,
    )

private fun SchedulerConfig.toRecord(): SchedulerConfigRecord =
    SchedulerConfigRecord(kind = kind.toEnum(), warmupSteps = warmupSteps)

private fun TrainConfig.toRecord(): TrainConfigRecord =
    TrainConfigRecord(
        baseModelRepo = baseModelRepo,
        outputDir = outputDir,
        lora = lora.toRecord(),
        optim = optim.toRecord(),
        scheduler = scheduler.toRecord(),
        maxSteps = maxSteps,
        batchSize = batchSize,
        gradientAccumulationSteps = gradientAccumulationSteps,
        maxSeqLen = maxSeqLen,
        evalSteps = evalSteps,
        saveSteps = saveSteps,
        seed = seed,
        mixedPrecision = mixedPrecision.toEnum(),
        device = device,
    )

private fun TrainCoreConfig.toRecord(): TrainCoreConfigRecord =
    TrainCoreConfigRecord(
        baseModelRepo = baseModelRepo,
        baseModelRevision = baseModelRevision,
        outputDir = outputDir,
        maxSteps = maxSteps,
        batchSize = batchSize,
        gradientAccumulationSteps = gradientAccumulationSteps,
        maxSeqLen = maxSeqLen,
        evalSteps = evalSteps,
        saveSteps = saveSteps,
        seed = seed,
        mixedPrecision = mixedPrecision.toEnum(),
        device = device,
        optim = optim.toRecord(),
        scheduler = scheduler.toRecord(),
    )

private fun DpoConfig.toRecord(): DpoConfigRecord =
    DpoConfigRecord(
        core = core.toRecord(),
        lora = lora.toRecord(),
        beta = beta,
        labelSmoothing = labelSmoothing,
        referenceModelRepo = referenceModelRepo,
        referenceModelRevision = referenceModelRevision,
    )

private fun OrpoConfig.toRecord(): OrpoConfigRecord =
    OrpoConfigRecord(
        core = core.toRecord(),
        lora = lora.toRecord(),
        lambda = lambda,
    )

private fun SimpoConfig.toRecord(): SimpoConfigRecord =
    SimpoConfigRecord(
        core = core.toRecord(),
        lora = lora.toRecord(),
        beta = beta,
        gamma = gamma,
    )

private fun KtoConfig.toRecord(): KtoConfigRecord =
    KtoConfigRecord(
        core = core.toRecord(),
        lora = lora.toRecord(),
        beta = beta,
        lambdaD = lambdaD,
        lambdaU = lambdaU,
        referenceModelRepo = referenceModelRepo,
        referenceModelRevision = referenceModelRevision,
    )

private fun FullFineTuneConfig.toRecord(): FullFineTuneConfigRecord =
    FullFineTuneConfigRecord(
        core = core.toRecord(),
        gradientCheckpointing = gradientCheckpointing,
    )

private fun FullFineTuneResultRecord.toDomain(): FullFineTuneResult =
    FullFineTuneResult(
        outputDir = outputDir,
        finalLoss = finalLoss,
        stepsCompleted = stepsCompleted,
    )

private fun TrainedAdapterRecord.toDomain(): TrainedAdapter =
    TrainedAdapter(adapterDir = adapterDir, finalLoss = finalLoss, totalSteps = totalSteps)

private fun TrainingEventEnum.toDomain(): TrainingEvent =
    when (this) {
        is TrainingEventEnum.Started -> TrainingEvent.Started(totalSteps)
        is TrainingEventEnum.StepCompleted ->
            TrainingEvent.StepCompleted(
                step = step,
                loss = loss,
                learningRate = learningRate,
                elapsedMs = elapsedMs,
            )
        is TrainingEventEnum.Evaluating -> TrainingEvent.Evaluating(step)
        is TrainingEventEnum.EvalCompleted -> TrainingEvent.EvalCompleted(step = step, evalLoss = evalLoss)
        is TrainingEventEnum.CheckpointSaved -> TrainingEvent.CheckpointSaved(step = step, path = path)
        is TrainingEventEnum.Finished ->
            TrainingEvent.Finished(
                finalLoss = finalLoss,
                totalSteps = totalSteps,
                adapterDir = adapterDir,
            )
    }

private class TrainingProgressBridge(
    private val user: (TrainingEvent) -> Unit,
) : UniffiForeignTrainingProgress {
    override fun onEvent(event: TrainingEventEnum) {
        user(event.toDomain())
    }
}

private class ForeignLocalModelBridge(
    private val user: ForeignLocalModel,
) : UniffiForeignLocalModel {
    override suspend fun load() = user.load()

    override suspend fun unload() = user.unload()

    override suspend fun isLoaded(): Boolean = user.isLoaded()

    override fun device(): String = user.device()

    override suspend fun memoryBytes(): ULong? = user.memoryBytes()

    override suspend fun loadAdapter(
        adapterDir: String,
        options: AdapterOptionsRecord,
    ): AdapterHandleRecord = user.loadAdapter(adapterDir, options.toDomain()).toRecord()

    override suspend fun unloadAdapter(handle: AdapterHandleRecord) {
        user.unloadAdapter(handle.toDomain())
    }

    override suspend fun listAdapters(): List<AdapterStatusRecord> =
        user.listAdapters().map {
            AdapterStatusRecord(
                adapterId = it.adapterId,
                scale = it.scale.toFloat(),
                sourceDir = it.sourceDir,
                memoryBytes = it.memoryBytes,
            )
        }
}
