package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.Event
import dev.zorpx.blazen.uniffi.StepHandler
import dev.zorpx.blazen.uniffi.StepOutput
import dev.zorpx.blazen.uniffi.WorkflowBuilder
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * End-to-end workflow tests for the Blazen Kotlin binding.
 *
 * These mirror the round-trip tests already in the sibling bindings —
 * [Go's `TestEchoWorkflow`][go] and
 * [Swift's `testSingleStepWorkflowRoundTrip`][swift] — so all four
 * bindings (Go / Swift / Kotlin / Ruby) exercise the same FFI shape:
 * register a one-step `StartEvent -> StopEvent` workflow, drive it from
 * the foreign side, and assert the terminal event carries the expected
 * payload.
 *
 * The Rust workflow engine wraps the host-supplied input as a
 * `StartEvent` of the form `{"data": <user-payload>}`, then passes that
 * envelope into the registered handler. The handler emits a
 * `StopEvent`, which becomes the workflow's terminal event.
 *
 * The classes used here come from the UniFFI-generated package
 * (`dev.zorpx.blazen.uniffi.*`); the hand-written wrapper layer in
 * `dev.zorpx.blazen` declares parallel value records of the same names
 * but is not yet wired up to the live FFI surface.
 *
 * [go]: bindings/go/blazen_test.go
 * [swift]: bindings/swift/Tests/BlazenSwiftTests/BlazenSwiftTests.swift
 */
class WorkflowTest {

    /**
     * Build a single-step workflow whose handler echoes the start-event
     * payload back inside a `StopEvent`, run it, and assert the terminal
     * event is a `StopEvent` carrying the echoed `"hello"` value.
     *
     * Mirrors the Go `TestEchoWorkflow` shape: the handler unwraps the
     * `{"data": {"message": ...}}` envelope and emits
     * `{"result": {"echo": ...}}` as the stop payload.
     */
    @Test
    fun `single-step workflow round-trips a JSON payload through Rust`() = runBlocking {
        // Touching `Blazen.version` forces JNA to resolve and load
        // `libblazen_uniffi.{so,dylib,dll}` from the classpath; the
        // hand-written `Blazen` object exposes no explicit `init()`.
        assertNotNull(Blazen.version)

        val handler = object : StepHandler {
            override suspend fun invoke(event: Event): StepOutput {
                // StartEvent wire shape: {"data": {"message": "hello"}}
                val parsed = Json.parseToJsonElement(event.dataJson).jsonObject
                val data = parsed["data"]?.jsonObject ?: JsonObject(emptyMap())
                val message = data["message"]?.jsonPrimitive?.content ?: ""
                // StopEvent wire shape: {"result": <stop-payload>}
                val payload = """{"result":{"echo":"$message"}}"""
                return StepOutput.Single(
                    Event(eventType = "blazen::StopEvent", dataJson = payload),
                )
            }
        }

        val builder = WorkflowBuilder("echo-test")
        try {
            builder.step(
                name = "echo",
                accepts = listOf("blazen::StartEvent"),
                emits = listOf("blazen::StopEvent"),
                handler = handler,
            )
            val workflow = builder.build()
            try {
                val result = workflow.run("""{"message":"hello"}""")
                // Match Swift's `.contains("StopEvent")` check rather than
                // an equality check — the engine may prefix the event type
                // with its module path (e.g. `blazen::StopEvent`).
                assertTrue(
                    result.event.eventType.contains("StopEvent"),
                    "Expected terminal event type to contain 'StopEvent', got '${result.event.eventType}'",
                )
                assertTrue(
                    result.event.dataJson.contains("hello"),
                    "Expected stop-event payload to contain 'hello', got '${result.event.dataJson}'",
                )
            } finally {
                workflow.close()
            }
        } finally {
            builder.close()
        }
    }
}
