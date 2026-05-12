package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Wire-format event crossed across the Blazen FFI.
 *
 * `eventType` is the event class name (e.g. `"StartEvent"`, `"StopEvent"`,
 * `"MyCustomEvent"`). `dataJson` is the JSON-encoded payload — callers
 * typically marshal it to/from a Kotlin `@Serializable` data class with
 * `Json.encodeToString` / `Json.decodeFromString` at the wrapper boundary.
 */
@Serializable
public data class Event(
    val eventType: String,
    val dataJson: String,
)

/**
 * What a workflow step handler returns: zero, one, or many events to publish.
 */
public sealed class StepOutput {
    /** Step performed work but produced no event. */
    public data object None : StepOutput()

    /** Step produced exactly one event (the common case). */
    public data class Single(val event: Event) : StepOutput()

    /** Step fans out — produced multiple events at once. */
    public data class Multiple(val events: List<Event>) : StepOutput()
}

/**
 * Foreign-implemented step handler.
 *
 * The Rust workflow engine calls [invoke] whenever an event matching the
 * step's `accepts` list arrives, and routes the returned [StepOutput] back
 * into the event queue.
 */
public fun interface StepHandler {
    public suspend fun invoke(event: Event): StepOutput
}

/**
 * Final result of a workflow run, including aggregate LLM token usage and
 * cost across the run.
 */
@Serializable
public data class WorkflowResult(
    val event: Event,
    val totalInputTokens: Long,
    val totalOutputTokens: Long,
    val totalCostUsd: Double,
)
