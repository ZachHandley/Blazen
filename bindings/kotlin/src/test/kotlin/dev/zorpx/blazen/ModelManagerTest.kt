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
