package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.AudioResult
import dev.zorpx.blazen.uniffi.BlazenException
import dev.zorpx.blazen.uniffi.ImageRequest
import dev.zorpx.blazen.uniffi.RequestTiming
import dev.zorpx.blazen.uniffi.SpeechRequest
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Tests for the Kotlin `CustomProvider` surface exposed in
 * [`ProvidersCustom.kt`][dev.zorpx.blazen]:
 *
 * 1. The preset factories ([Blazen.ollama], [Blazen.lmStudio]) construct a
 *    [CustomProviderHandle] whose `providerId()` matches the preset name.
 * 2. A [CustomProviderBase] subclass that overrides only `textToSpeech`
 *    has its override invoked when called through the handle.
 * 3. The same subclass surfaces [BlazenException.Unsupported] from any
 *    method it does not override (here: `generateImage`).
 *
 * These mirror the typed-trait override + factory-presets coverage already
 * shipping in the Go/Swift/Python/Node test suites — see Phase F of the
 * CustomProvider rollout.
 */
class CustomProviderTest {
    @Test
    fun `ollama factory constructs provider`() {
        Blazen.ollama(model = "llama3").use { handle ->
            assertEquals("ollama", handle.providerId())
        }
    }

    @Test
    fun `lm studio factory works`() {
        Blazen.lmStudio(model = "qwen-7b").use { handle ->
            assertEquals("lm_studio", handle.providerId())
        }
    }

    /**
     * Subclass that overrides only `textToSpeech`. All other capability
     * methods inherit the throwing `Unsupported` defaults from
     * [CustomProviderBase].
     */
    private class StubTts : CustomProviderBase() {
        override fun providerId(): String = "stub-tts"

        override suspend fun textToSpeech(request: SpeechRequest): AudioResult =
            AudioResult(
                audio = emptyList(),
                timing = RequestTiming(queueMs = null, executionMs = null, totalMs = null),
                cost = null,
                usage = null,
                audioSeconds = 0.0,
                metadata = """{"echoed":"${request.text}"}""",
            )
    }

    @Test
    fun `subclass text-to-speech routes to override`() = runBlocking {
        Blazen.customProvider(StubTts()).use { handle ->
            assertEquals("stub-tts", handle.providerId())
            val result = handle.textToSpeech(
                SpeechRequest(
                    text = "hello",
                    voice = null,
                    voiceUrl = null,
                    language = null,
                    speed = null,
                    model = null,
                    parameters = "",
                ),
            )
            assertTrue(
                result.metadata.contains("hello"),
                "Expected override to echo the request text into metadata, got '${result.metadata}'",
            )
            assertEquals(0.0, result.audioSeconds)
        }
    }

    @Test
    fun `unimplemented method throws Unsupported`() {
        Blazen.customProvider(StubTts()).use { handle ->
            val ex = assertThrows<BlazenException.Unsupported> {
                runBlocking {
                    handle.generateImage(
                        ImageRequest(
                            prompt = "a cat",
                            negativePrompt = null,
                            width = null,
                            height = null,
                            numImages = null,
                            model = null,
                            parameters = "",
                        ),
                    )
                }
            }
            assertTrue(
                ex.message.contains("generateImage"),
                "Expected Unsupported message to name the method, got '${ex.message}'",
            )
            assertTrue(
                ex.message.contains("stub-tts"),
                "Expected Unsupported message to include providerId, got '${ex.message}'",
            )
        }
    }
}
