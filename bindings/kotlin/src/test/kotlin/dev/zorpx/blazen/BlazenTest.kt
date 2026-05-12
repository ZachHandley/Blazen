package dev.zorpx.blazen

import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Smoke + shape tests for the Blazen Kotlin binding.
 *
 * The first test exercises the native FFI surface by reading
 * [Blazen.version] — this confirms JNA found `libblazen_uniffi.{so,dylib,dll}`
 * on the classpath and that the embedded UDL surface dispatches correctly.
 * The rest of the tests pin the pure-Kotlin value record shapes so any
 * accidental field rename / drop produces a failing build.
 */
class BlazenTest {
    @Test
    fun `version is a non-empty semver-ish string`() {
        val v = Blazen.version
        assertNotNull(v)
        assertTrue(v.isNotEmpty(), "version must be non-empty, got: '$v'")
        assertTrue(v.matches(Regex("""^\d+\.\d+\.\d+.*$""")), "version must start with X.Y.Z, got: '$v'")
    }

    @Test
    fun `event round-trips through kotlinx-serialization`() {
        val original = Event(eventType = "StartEvent", dataJson = """{"name":"Zach"}""")
        val encoded = Json.encodeToString(Event.serializer(), original)
        val decoded = Json.decodeFromString(Event.serializer(), encoded)
        assertEquals(original, decoded)
    }

    @Test
    fun `step output sealed hierarchy covers none, single, multiple`() {
        val ev = Event("StopEvent", """{"result":"ok"}""")
        val outputs: List<StepOutput> = listOf(
            StepOutput.None,
            StepOutput.Single(ev),
            StepOutput.Multiple(listOf(ev, ev)),
        )
        assertEquals(3, outputs.size)
        // Exhaustive when() forces the compiler to fail loudly if a variant is added.
        outputs.forEach { out ->
            val kind = when (out) {
                is StepOutput.None -> "none"
                is StepOutput.Single -> "single"
                is StepOutput.Multiple -> "multiple"
            }
            assertNotNull(kind)
        }
    }

    @Test
    fun `chat message defaults are empty collections, not null`() {
        val msg = ChatMessage(role = "user", content = "hi")
        assertEquals(emptyList<Media>(), msg.mediaParts)
        assertEquals(emptyList<ToolCall>(), msg.toolCalls)
    }

    @Test
    fun `BlazenException Provider carries structured payload`() {
        val e = BlazenException.Provider(
            kind = "OpenAIHttp",
            message = "boom",
            providerName = "openai",
            status = 500,
            endpoint = "/v1/chat",
            requestId = "req-123",
            detail = "internal",
            retryAfterMs = 250,
        )
        assertEquals("OpenAIHttp", e.kind)
        assertEquals(500, e.status)
        assertEquals(250, e.retryAfterMs)
    }
}
