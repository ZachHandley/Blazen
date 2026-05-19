package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.AdapterHandleRecord
import dev.zorpx.blazen.uniffi.AdapterOptionsRecord
import dev.zorpx.blazen.uniffi.AdapterStatusRecord
import dev.zorpx.blazen.uniffi.ModelStatusRecord
import dev.zorpx.blazen.uniffi.PoolStatusRecord
import dev.zorpx.blazen.uniffi.UniffiModelManager
import java.util.UUID
import dev.zorpx.blazen.uniffi.ForeignLocalModel as UniffiForeignLocalModel

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
