package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Static descriptors for the LLM / embedding / TTS / STT / image providers
 * Blazen knows about.
 *
 * These records identify a provider + model pairing in a wire-compatible
 * way; they are what the higher-level idiomatic Kotlin factories
 * (`Providers.openAI(...)`, etc.) pass to the native runtime. The factory
 * functions themselves bind to the generated UniFFI surface (see the
 * UniFFI coverage note on [Blazen] — they will light up once the
 * `#[uniffi::export]` proc-macro surface is wired into the cdylib).
 */
public object Providers {
    /**
     * Identifier + base-URL bundle for an HTTP-backed completion provider.
     *
     * Use [openAI], [anthropic], [google], [mistral] etc. as canonical
     * constructors; passing a custom [baseUrl] lets you target any
     * OpenAI-compatible gateway (vLLM, llama.cpp server, OpenRouter, etc.).
     */
    @Serializable
    public data class CompletionProviderConfig(
        val providerName: String,
        val apiKey: String,
        val model: String,
        val baseUrl: String? = null,
    )

    /** OpenAI (or OpenAI-compatible) chat completion provider config. */
    public fun openAI(apiKey: String, model: String, baseUrl: String? = null): CompletionProviderConfig =
        CompletionProviderConfig("openai", apiKey, model, baseUrl)

    /** Anthropic chat completion provider config. */
    public fun anthropic(apiKey: String, model: String, baseUrl: String? = null): CompletionProviderConfig =
        CompletionProviderConfig("anthropic", apiKey, model, baseUrl)

    /** Google Gemini chat completion provider config. */
    public fun google(apiKey: String, model: String, baseUrl: String? = null): CompletionProviderConfig =
        CompletionProviderConfig("google", apiKey, model, baseUrl)

    /** Mistral chat completion provider config. */
    public fun mistral(apiKey: String, model: String, baseUrl: String? = null): CompletionProviderConfig =
        CompletionProviderConfig("mistral", apiKey, model, baseUrl)

    /** Groq (OpenAI-compatible) chat completion provider config. */
    public fun groq(apiKey: String, model: String, baseUrl: String? = null): CompletionProviderConfig =
        CompletionProviderConfig("groq", apiKey, model, baseUrl)

    /** xAI (Grok) chat completion provider config. */
    public fun xai(apiKey: String, model: String, baseUrl: String? = null): CompletionProviderConfig =
        CompletionProviderConfig("xai", apiKey, model, baseUrl)

    /** Identifier + base-URL bundle for an HTTP-backed embedding provider. */
    @Serializable
    public data class EmbeddingProviderConfig(
        val providerName: String,
        val apiKey: String,
        val model: String,
        val baseUrl: String? = null,
    )

    /** OpenAI embeddings provider config. */
    public fun openAIEmbeddings(apiKey: String, model: String, baseUrl: String? = null): EmbeddingProviderConfig =
        EmbeddingProviderConfig("openai", apiKey, model, baseUrl)

    /** Voyage AI embeddings provider config. */
    public fun voyageEmbeddings(apiKey: String, model: String, baseUrl: String? = null): EmbeddingProviderConfig =
        EmbeddingProviderConfig("voyage", apiKey, model, baseUrl)
}
