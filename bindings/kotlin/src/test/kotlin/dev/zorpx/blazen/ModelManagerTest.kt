package dev.zorpx.blazen

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import dev.zorpx.blazen.uniffi.BlazenException as UniffiBlazenException

/**
 * Smoke tests for the idiomatic [ModelManager] wrapper.
 *
 * Mirrors the Go binding's `TestModelManager*` shape: exercise the three
 * constructors, snapshot verbs against an empty manager, and assert that
 * verbs operating on unknown ids raise the generated
 * [UniffiBlazenException] sealed exception so callers can `when`-match on
 * the concrete variant.
 */
class ModelManagerTest {
    @Test
    fun `default constructor seeds cpu and gpu0 pools with no budget enforcement`() {
        ModelManager().use { mgr ->
            val pools = mgr.pools().associateBy { it.pool }
            assertNotNull(pools["cpu"], "cpu pool must be present, got: $pools")
            assertNotNull(pools["gpu:0"], "gpu:0 pool must be present, got: $pools")
            assertEquals(ULong.MAX_VALUE, pools.getValue("cpu").budgetBytes)
            assertEquals(ULong.MAX_VALUE, pools.getValue("gpu:0").budgetBytes)
        }
    }

    @Test
    fun `gigabyte budget constructor records both pool budgets`() {
        ModelManager(cpuRamGb = 4.0, gpuVramGb = 8.0).use { mgr ->
            val pools = mgr.pools().associateBy { it.pool }
            // Why: 1 GiB == 1_073_741_824 bytes, matching the Rust constructor's
            // power-of-two conversion (not 1e9). Asserting the exact byte count
            // pins the convention the wrapper inherits from the upstream crate.
            assertEquals(4uL * 1_073_741_824uL, pools.getValue("cpu").budgetBytes)
            assertEquals(8uL * 1_073_741_824uL, pools.getValue("gpu:0").budgetBytes)
        }
    }

    @Test
    fun `per-pool budget constructor accepts custom labels`() {
        ModelManager(poolBudgets = mapOf("cpu" to 2.0, "gpu:1" to 6.0)).use { mgr ->
            val pools = mgr.pools().associateBy { it.pool }
            assertEquals(2uL * 1_073_741_824uL, pools.getValue("cpu").budgetBytes)
            assertEquals(6uL * 1_073_741_824uL, pools.getValue("gpu:1").budgetBytes)
        }
    }

    @Test
    fun `status returns empty list on a fresh manager`() = runBlocking {
        ModelManager().use { mgr ->
            assertEquals(emptyList<ModelStatus>(), mgr.status())
        }
    }

    @Test
    fun `isLoaded returns false for unknown model id`() = runBlocking {
        ModelManager().use { mgr ->
            assertEquals(false, mgr.isLoaded("does-not-exist"))
        }
    }

    @Test
    fun `load on an unregistered model id raises a Blazen exception`() {
        ModelManager().use { mgr ->
            val ex = assertThrows(UniffiBlazenException::class.java) {
                runBlocking { mgr.load("does-not-exist") }
            }
            assertNotNull(ex.message)
            assertTrue(ex.message!!.isNotEmpty(), "exception message must be non-empty")
        }
    }

    @Test
    fun `loadAdapter on an unregistered model id raises a Blazen exception`() {
        ModelManager().use { mgr ->
            val ex = assertThrows(UniffiBlazenException::class.java) {
                runBlocking {
                    mgr.loadAdapter(
                        modelId = "does-not-exist",
                        adapterDir = "/nonexistent/lora",
                        options = AdapterOptions(adapterId = "test-adapter"),
                    )
                }
            }
            assertNotNull(ex.message)
        }
    }

    @Test
    fun `AdapterOptions default adapter id is a non-empty unique string`() {
        val a = AdapterOptions()
        val b = AdapterOptions()
        assertTrue(a.adapterId.isNotEmpty())
        assertTrue(b.adapterId.isNotEmpty())
        // Why: defaults are UUIDs — two adjacent constructions must not
        // collide or the contract that callers can mount adapters without
        // inventing names breaks.
        assertTrue(a.adapterId != b.adapterId, "default adapter ids must be unique")
        assertEquals(1.0, a.scale)
    }

    @Test
    fun `loadFromHf throws on nonexistent repo`() {
        ModelManager().use { mgr ->
            val ex = assertThrows(UniffiBlazenException::class.java) {
                runBlocking {
                    mgr.loadFromHf(
                        id = "ghost",
                        repo = "this-org-does-not-exist/this-repo-also-does-not-exist",
                    )
                }
            }
            assertNotNull(ex.message)
            assertTrue(ex.message!!.isNotEmpty(), "exception message must be non-empty")
        }
    }

    @Test
    fun `loadFromHf with default options does not crash`() {
        // Why: a default-constructed HfLoadOptions must round-trip through
        // the FFI without panicking on the all-null record. We expect a
        // BlazenException (empty repo id), but never a JVM crash or a
        // NullPointerException from the converter — that is the bug we
        // are guarding against.
        ModelManager().use { mgr ->
            val ex = assertThrows(UniffiBlazenException::class.java) {
                runBlocking {
                    mgr.loadFromHf(id = "empty", repo = "")
                }
            }
            assertNotNull(ex.message)
        }
    }

    @Test
    fun `BackendHint values map to stable lower-case wire strings`() {
        assertEquals("mistralrs", BackendHint.MISTRALRS.value)
        assertEquals("candle", BackendHint.CANDLE.value)
        assertEquals("llamacpp", BackendHint.LLAMACPP.value)
    }

    @Test
    fun `trainLora rejects invalid config with maxSteps of zero`() {
        ModelManager().use { mgr ->
            val tokenizer = java.io.File.createTempFile("blazen-train-kotlin-wrap-tok", ".json").apply {
                writeText("{\"version\":\"1.0\",\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]}}")
                deleteOnExit()
            }
            val data = java.io.File.createTempFile("blazen-train-kotlin-wrap-data", ".jsonl").apply {
                writeText("{\"text\":\"hi\"}\n")
                deleteOnExit()
            }
            val outDir = java.nio.file.Files.createTempDirectory("blazen-train-kotlin-wrap-out").toFile().apply {
                deleteOnExit()
            }
            // Why: dataset construction may itself fail before train_lora is reached if
            // the tokenizer is too minimal — both failure modes prove the FFI surface
            // refuses the invalid configuration rather than crashing the JVM.
            val ex = assertThrows(UniffiBlazenException::class.java) {
                runBlocking {
                    val ds = JsonlDataset.fromPath(
                        path = data.absolutePath,
                        tokenizerPath = tokenizer.absolutePath,
                    )
                    mgr.trainLora(
                        config = TrainConfig(
                            baseModelRepo = "hf-internal-testing/tiny-random-gpt2",
                            outputDir = outDir.absolutePath,
                            maxSteps = 0u,
                        ),
                        dataset = ds,
                    )
                }
            }
            assertNotNull(ex.message)
        }
    }

    @Test
    fun `JsonlDataset throws on invalid path`() {
        val ex = assertThrows(UniffiBlazenException::class.java) {
            JsonlDataset.fromPath(
                path = "/nonexistent/blazen-train-kotlin-wrap/missing.jsonl",
                tokenizerPath = "/nonexistent/blazen-train-kotlin-wrap/missing-tokenizer.json",
            )
        }
        assertNotNull(ex.message)
        assertTrue(ex.message!!.isNotEmpty(), "exception message must be non-empty")
    }

    @Test
    fun `TrainingEvent exhaustive when compiles`() {
        // Why: this test exists solely to lock the sealed-class shape in
        // place. If a new variant is added upstream without updating the
        // wrapper's TrainingEvent and conversion, the compiler will flag
        // a missing branch here and fail the test build.
        val events: List<TrainingEvent> = listOf(
            TrainingEvent.Started(totalSteps = 10uL),
            TrainingEvent.StepCompleted(step = 1uL, loss = 0.5f, learningRate = 2e-4, elapsedMs = 100uL),
            TrainingEvent.Evaluating(step = 5uL),
            TrainingEvent.EvalCompleted(step = 5uL, evalLoss = 0.4f),
            TrainingEvent.CheckpointSaved(step = 5uL, path = "/tmp/ckpt"),
            TrainingEvent.Finished(finalLoss = 0.3f, totalSteps = 10uL, adapterDir = "/tmp/adapter"),
        )
        val labels = events.map { ev ->
            when (ev) {
                is TrainingEvent.Started -> "started:${ev.totalSteps}"
                is TrainingEvent.StepCompleted -> "step:${ev.step}"
                is TrainingEvent.Evaluating -> "evaluating:${ev.step}"
                is TrainingEvent.EvalCompleted -> "eval:${ev.step}"
                is TrainingEvent.CheckpointSaved -> "ckpt:${ev.step}"
                is TrainingEvent.Finished -> "finished:${ev.totalSteps}"
            }
        }
        assertEquals(6, labels.size)
        assertEquals("started:10", labels[0])
        assertEquals("finished:10", labels[5])
    }

    @Test
    fun `TrainConfig defaults match documented surface`() {
        val cfg = TrainConfig(baseModelRepo = "repo", outputDir = "/tmp/out")
        assertEquals(1000u, cfg.maxSteps)
        assertEquals(4u, cfg.batchSize)
        assertEquals(2048u, cfg.maxSeqLen)
        assertEquals(42uL, cfg.seed)
        assertEquals(MixedPrecision.BF16, cfg.mixedPrecision)
        assertEquals(SchedulerKind.COSINE, cfg.scheduler.kind)
        assertEquals(16u, cfg.lora.rank)
        assertEquals(32.0f, cfg.lora.alpha)
        assertEquals(listOf("q_proj", "k_proj", "v_proj", "o_proj"), cfg.lora.targetModules)
        assertEquals(2e-4, cfg.optim.learningRate)
        assertEquals(1.0f, cfg.optim.gradientClip)
    }

    @Test
    fun `SchedulerKind and MixedPrecision wire strings are stable`() {
        assertEquals("constant", SchedulerKind.CONSTANT.value)
        assertEquals("linear", SchedulerKind.LINEAR.value)
        assertEquals("cosine", SchedulerKind.COSINE.value)
        assertEquals("none", MixedPrecision.NONE.value)
        assertEquals("bf16", MixedPrecision.BF16.value)
    }

    @Test
    fun `DpoConfig defaults match documented surface`() {
        val cfg = DpoConfig(core = TrainCoreConfig(baseModelRepo = "repo", outputDir = "/tmp/out"))
        assertEquals(0.1f, cfg.beta)
        assertEquals(0.0f, cfg.labelSmoothing)
        assertEquals(null, cfg.referenceModelRepo)
        assertEquals(null, cfg.referenceModelRevision)
        // Core defaults: smaller per-step batches than TrainConfig because
        // each preference step carries two forward passes (chosen + rejected).
        assertEquals(1u, cfg.core.batchSize)
        assertEquals(8u, cfg.core.gradientAccumulationSteps)
        assertEquals(1024u, cfg.core.maxSeqLen)
        assertEquals(MixedPrecision.BF16, cfg.core.mixedPrecision)
        // LoRA defaults reused from PR7's LoraConfig.
        assertEquals(16u, cfg.lora.rank)
        assertEquals(32.0f, cfg.lora.alpha)
    }

    @Test
    fun `OrpoConfig defaults match documented surface`() {
        val cfg = OrpoConfig(core = TrainCoreConfig(baseModelRepo = "repo", outputDir = "/tmp/out"))
        assertEquals(0.1f, cfg.lambda)
        assertEquals(16u, cfg.lora.rank)
        assertEquals(MixedPrecision.BF16, cfg.core.mixedPrecision)
    }

    @Test
    fun `SimpoConfig defaults follow TRL main`() {
        val cfg = SimpoConfig(core = TrainCoreConfig(baseModelRepo = "repo", outputDir = "/tmp/out"))
        // Why: TRL `main` (and the upstream Rust default) sets beta=2.0,
        // gamma=1.0 — pinning these guards against silent drift if the
        // generated record changes its defaults.
        assertEquals(2.0f, cfg.beta)
        assertEquals(1.0f, cfg.gamma)
    }

    @Test
    fun `KtoConfig defaults weight both signs equally`() {
        val cfg = KtoConfig(core = TrainCoreConfig(baseModelRepo = "repo", outputDir = "/tmp/out"))
        assertEquals(0.1f, cfg.beta)
        // Why: lambda_D and lambda_U both default to 1.0, so the desirable
        // and undesirable terms contribute equally before per-sign tuning.
        assertEquals(1.0f, cfg.lambdaD)
        assertEquals(1.0f, cfg.lambdaU)
        assertEquals(null, cfg.referenceModelRepo)
    }

    @Test
    fun `FullFineTuneConfig defaults expose gradient checkpointing flag`() {
        val cfg = FullFineTuneConfig(core = TrainCoreConfig(baseModelRepo = "repo", outputDir = "/tmp/out"))
        // Why: gradient_checkpointing is accepted for forward compatibility
        // but the Rust trainer rejects `true` at init time on candle 0.10.2.
        // Default must be `false` so callers don't accidentally trip that
        // validation.
        assertEquals(false, cfg.gradientCheckpointing)
        assertEquals(1u, cfg.core.batchSize)
    }

    @Test
    fun `PreferenceJsonlDataset throws on invalid path`() {
        val ex = assertThrows(UniffiBlazenException::class.java) {
            PreferenceJsonlDataset.fromPath(
                path = "/nonexistent/blazen-train-kotlin-wrap/missing-pref.jsonl",
                tokenizerPath = "/nonexistent/blazen-train-kotlin-wrap/missing-tokenizer.json",
            )
        }
        assertNotNull(ex.message)
        assertTrue(ex.message!!.isNotEmpty(), "exception message must be non-empty")
    }

    @Test
    fun `RatedJsonlDataset throws on invalid path`() {
        val ex = assertThrows(UniffiBlazenException::class.java) {
            RatedJsonlDataset.fromPath(
                path = "/nonexistent/blazen-train-kotlin-wrap/missing-rated.jsonl",
                tokenizerPath = "/nonexistent/blazen-train-kotlin-wrap/missing-tokenizer.json",
            )
        }
        assertNotNull(ex.message)
        assertTrue(ex.message!!.isNotEmpty(), "exception message must be non-empty")
    }

    @Suppress("UNUSED_VARIABLE", "unused")
    @Test
    fun `trainDpo trainOrpo trainSimpo trainKto fineTune signatures compile`() {
        // Why: pure compile-time guard. We never invoke these references —
        // the suspending native call would block on download/init — but
        // assigning them to typed function values forces the compiler to
        // resolve each overload, so any drift in the wrapper signature
        // (parameter names, optional progress callback, return type)
        // surfaces as a build failure rather than a runtime surprise.
        val mgr: ModelManager? = null
        if (mgr != null) {
            val dpo: suspend (DpoConfig, PreferenceJsonlDataset, ((TrainingEvent) -> Unit)?) -> TrainedAdapter =
                mgr::trainDpo
            val orpo: suspend (OrpoConfig, PreferenceJsonlDataset, ((TrainingEvent) -> Unit)?) -> TrainedAdapter =
                mgr::trainOrpo
            val simpo: suspend (SimpoConfig, PreferenceJsonlDataset, ((TrainingEvent) -> Unit)?) -> TrainedAdapter =
                mgr::trainSimpo
            val kto: suspend (KtoConfig, RatedJsonlDataset, ((TrainingEvent) -> Unit)?) -> TrainedAdapter =
                mgr::trainKto
            val ft: suspend (FullFineTuneConfig, JsonlDataset, ((TrainingEvent) -> Unit)?) -> FullFineTuneResult =
                mgr::fineTune
        }
        // Pure compile-time guard — no runtime assertion needed.
        assertTrue(true)
    }

    @Test
    fun `ForeignLocalModel interface is implementable from Kotlin`() {
        // Why: the interface must be openly implementable (not sealed, not
        // requiring internal types) — exercise that by constructing an
        // anonymous implementation here. The instance is never registered;
        // we are only asserting the interface shape compiles.
        val model = object : ForeignLocalModel {
            override suspend fun load() = Unit
            override suspend fun unload() = Unit
            override suspend fun isLoaded(): Boolean = false
            override fun device(): String = "cpu"
            override suspend fun memoryBytes(): ULong? = null
            override suspend fun loadAdapter(
                adapterDir: String,
                options: AdapterOptions,
            ): AdapterHandle = AdapterHandle(options.adapterId, 0uL, "attached")
            override suspend fun unloadAdapter(handle: AdapterHandle) = Unit
            override suspend fun listAdapters(): List<AdapterStatus> = emptyList()
        }
        assertEquals("cpu", model.device())
    }
}
