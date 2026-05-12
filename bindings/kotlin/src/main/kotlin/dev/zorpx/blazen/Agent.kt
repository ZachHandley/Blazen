package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Foreign-implemented tool handler used by [AgentConfig].
 *
 * The Rust agent loop calls [invoke] when the model issues a tool call;
 * the JSON-encoded arguments come straight from the model, and the
 * returned string is fed back as the tool-result message.
 */
public fun interface ToolHandler {
    public suspend fun invoke(argumentsJson: String): String
}

/**
 * Configuration record for a Blazen agent run.
 *
 * Agents are LLM-driven loops that may issue tool calls, observe results,
 * and iterate up to [maxIterations] turns. `system` sets the system
 * prompt; [tools] declares the tool surface and [toolHandlers] is keyed
 * by tool name to bind a Kotlin handler to each.
 */
public data class AgentConfig(
    val model: String,
    val system: String? = null,
    val tools: List<Tool> = emptyList(),
    val toolHandlers: Map<String, ToolHandler> = emptyMap(),
    val maxIterations: Int = 8,
    val temperature: Double? = null,
)

/** Final outcome of an agent run. */
@Serializable
public data class AgentResult(
    val content: String,
    val iterations: Int,
    val usage: TokenUsage,
)
