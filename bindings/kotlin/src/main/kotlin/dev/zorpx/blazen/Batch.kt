package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * A single item to send through a batched chat-completion run.
 *
 * `id` is a caller-chosen key returned on the matching [BatchResult] so
 * callers can correlate inputs to outputs without relying on list order.
 */
@Serializable
public data class BatchItem(
    val id: String,
    val request: CompletionRequest,
)

/**
 * Result of one item in a batch run. Either [response] is set (success)
 * or [errorMessage] is set (failure); never both.
 */
@Serializable
public data class BatchResult(
    val id: String,
    val response: CompletionResponse? = null,
    val errorMessage: String? = null,
)
