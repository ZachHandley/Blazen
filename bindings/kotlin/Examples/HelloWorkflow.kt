package dev.zorpx.blazen.examples

import dev.zorpx.blazen.Blazen
import dev.zorpx.blazen.ChatMessage
import dev.zorpx.blazen.CompletionRequest
import dev.zorpx.blazen.Event
import dev.zorpx.blazen.StepOutput
import kotlinx.serialization.json.Json

/**
 * Minimal Blazen Kotlin binding demo.
 *
 * Prints the resolved native library version, then constructs a few
 * wire-format value records (events, completion requests) and round-trips
 * them through `kotlinx.serialization` to show the expected JSON shape on
 * the FFI boundary.
 *
 * Run with `gradle run -PmainClass=dev.zorpx.blazen.examples.HelloWorkflowKt`
 * once the surrounding module exposes a `main` configuration, or invoke
 * `HelloWorkflow.main()` directly from a Kotlin REPL.
 */
public object HelloWorkflow {
    @JvmStatic
    public fun main(args: Array<String>) {
        println("blazen-uniffi native version: ${Blazen.version}")

        val start = Event(
            eventType = "StartEvent",
            dataJson = """{"name":"Zach"}""",
        )
        println("start event JSON: ${Json.encodeToString(Event.serializer(), start)}")

        val produced = StepOutput.Single(
            Event(
                eventType = "StopEvent",
                dataJson = """{"result":"Hello, Zach!"}""",
            ),
        )
        val emittedTypes: List<String> = when (produced) {
            is StepOutput.None -> emptyList()
            is StepOutput.Single -> listOf(produced.event.eventType)
            is StepOutput.Multiple -> produced.events.map { it.eventType }
        }
        println("emitted event types: $emittedTypes")

        val req = CompletionRequest(
            messages = listOf(
                ChatMessage(role = "user", content = "Say hi to Kotlin."),
            ),
            model = "gpt-4o-mini",
            temperature = 0.7,
        )
        println("request JSON: ${Json.encodeToString(CompletionRequest.serializer(), req)}")
    }
}
