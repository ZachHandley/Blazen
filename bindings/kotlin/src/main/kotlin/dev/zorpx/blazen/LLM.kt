package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Multimodal media attached to a [ChatMessage].
 *
 * `kind` is `"image"`, `"audio"`, or `"video"`. `mimeType` is the IANA MIME
 * (`"image/png"`, `"audio/mp3"`, ...). `dataBase64` carries the raw bytes
 * base64-encoded; URL passthrough is intentionally not modelled here —
 * fetch the bytes and base64-encode them at the call site.
 */
@Serializable
public data class Media(
    val kind: String,
    val mimeType: String,
    val dataBase64: String,
)

/**
 * A tool invocation requested by the model.
 *
 * `argumentsJson` is the JSON-encoded arguments object. Foreign callers
 * parse this with `kotlinx.serialization` to access the tool's input
 * parameters.
 */
@Serializable
public data class ToolCall(
    val id: String,
    val name: String,
    val argumentsJson: String,
)

/**
 * A tool that the model may invoke during a completion.
 *
 * `parametersJson` is a JSON Schema describing the tool's inputs; serialise
 * your schema to a JSON string just before constructing the [Tool].
 */
@Serializable
public data class Tool(
    val name: String,
    val description: String,
    val parametersJson: String,
)

/**
 * Token usage statistics for a completion or embedding request.
 *
 * Counters are `Long` (Rust-side `u64` widened) for FFI uniformity. Zero
 * means either "the provider didn't report this counter" or "the counter
 * is genuinely zero" — the wire format does not distinguish.
 */
@Serializable
public data class TokenUsage(
    val promptTokens: Long = 0,
    val completionTokens: Long = 0,
    val totalTokens: Long = 0,
    val cachedInputTokens: Long = 0,
    val reasoningTokens: Long = 0,
)

/**
 * A single message in a chat conversation.
 *
 * `role` is one of `"system"`, `"user"`, `"assistant"`, `"tool"`. `content`
 * is the text payload (empty string when the message carries only tool
 * calls or media).
 */
@Serializable
public data class ChatMessage(
    val role: String,
    val content: String,
    val mediaParts: List<Media> = emptyList(),
    val toolCalls: List<ToolCall> = emptyList(),
    val toolCallId: String? = null,
    val name: String? = null,
)

/**
 * A provider-agnostic chat completion request.
 *
 * `system`, when set, is prepended as a `system`-role message. Provided as
 * a convenience because most foreign callers think of the system prompt as
 * a request-level field, not a message.
 */
@Serializable
public data class CompletionRequest(
    val messages: List<ChatMessage>,
    val tools: List<Tool> = emptyList(),
    val temperature: Double? = null,
    val maxTokens: Int? = null,
    val topP: Double? = null,
    val model: String? = null,
    val responseFormatJson: String? = null,
    val system: String? = null,
)

/**
 * The result of a non-streaming chat completion.
 *
 * `content` is the empty string when the provider returned no text.
 * `finishReason` is the empty string when the provider didn't report one.
 */
@Serializable
public data class CompletionResponse(
    val content: String,
    val toolCalls: List<ToolCall> = emptyList(),
    val finishReason: String,
    val model: String,
    val usage: TokenUsage = TokenUsage(),
)

/**
 * Response from an embedding model.
 *
 * `embeddings[i]` is the vector for the `i`-th input string.
 */
@Serializable
public data class EmbeddingResponse(
    val embeddings: List<List<Double>>,
    val model: String,
    val usage: TokenUsage = TokenUsage(),
)

/**
 * Streaming chunk emitted by a chat completion in streaming mode.
 *
 * `deltaContent` is the incremental text since the last chunk; tool-call
 * deltas surface in [deltaToolCalls]. Streaming finishes via the
 * separate done/error callbacks on the sink interface, not via a sentinel
 * chunk.
 */
@Serializable
public data class StreamChunk(
    val deltaContent: String = "",
    val deltaToolCalls: List<ToolCall> = emptyList(),
)
