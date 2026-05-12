package dev.zorpx.blazen

import kotlinx.coroutines.channels.Channel

/**
 * Wire-format terminal frame of a streaming chat completion.
 *
 * The Rust side delivers this once when the stream completes successfully;
 * the idiomatic Kotlin [kotlinx.coroutines.flow.Flow] adapter will surface
 * it by emitting the chunk stream and then closing the flow normally
 * (with [finishReason] and [usage] reachable via a one-element terminal
 * channel — see [StreamTerminator]).
 */
public data class StreamFinish(
    val finishReason: String,
    val usage: TokenUsage,
)

/**
 * Internal coroutine adapter helper bridging a UniFFI callback-style
 * streaming sink to a Kotlin [kotlinx.coroutines.flow.Flow].
 *
 * Holds an unbounded channel for chunks and a one-shot completion signal;
 * downstream consumers usually wrap one of these inside a `callbackFlow {}`
 * builder and forward chunks via `trySend`, then close with the finish
 * frame or an error.
 */
public class StreamTerminator {
    private val finish: Channel<StreamFinish> = Channel(capacity = 1)

    /** Record a successful stream completion. Idempotent if called twice. */
    public fun complete(finishReason: String, usage: TokenUsage) {
        finish.trySend(StreamFinish(finishReason, usage))
        finish.close()
    }

    /** Record a stream failure. Callers receive the exception via [awaitFinish]. */
    public fun fail(cause: Throwable) {
        finish.close(cause)
    }

    /**
     * Suspends until the stream completes, returning the finish reason +
     * usage tally. Throws if the stream failed.
     */
    public suspend fun awaitFinish(): StreamFinish =
        finish.receive()
}
