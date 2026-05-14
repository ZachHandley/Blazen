@file:Suppress("MatchingDeclarationName", "TooManyFunctions")

package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.ApiProtocol as UniffiApiProtocol
import dev.zorpx.blazen.uniffi.AudioMusicProviderDefaults as UniffiAudioMusicProviderDefaults
import dev.zorpx.blazen.uniffi.AudioResult
import dev.zorpx.blazen.uniffi.AudioSpeechProviderDefaults as UniffiAudioSpeechProviderDefaults
import dev.zorpx.blazen.uniffi.BackgroundRemovalProviderDefaults as UniffiBackgroundRemovalProviderDefaults
import dev.zorpx.blazen.uniffi.BackgroundRemovalRequest
import dev.zorpx.blazen.uniffi.BaseProvider as UniffiBaseProvider
import dev.zorpx.blazen.uniffi.BaseProviderDefaults as UniffiBaseProviderDefaults
import dev.zorpx.blazen.uniffi.BlazenException
import dev.zorpx.blazen.uniffi.CompletionProviderDefaults as UniffiCompletionProviderDefaults
import dev.zorpx.blazen.uniffi.CompletionRequest
import dev.zorpx.blazen.uniffi.CompletionResponse
import dev.zorpx.blazen.uniffi.CompletionStreamSink
import dev.zorpx.blazen.uniffi.CustomProvider as UniffiCustomProvider
import dev.zorpx.blazen.uniffi.CustomProviderHandle as UniffiCustomProviderHandle
import dev.zorpx.blazen.uniffi.EmbeddingProviderDefaults as UniffiEmbeddingProviderDefaults
import dev.zorpx.blazen.uniffi.EmbeddingResponse
import dev.zorpx.blazen.uniffi.ImageGenerationProviderDefaults as UniffiImageGenerationProviderDefaults
import dev.zorpx.blazen.uniffi.ImageRequest
import dev.zorpx.blazen.uniffi.ImageResult
import dev.zorpx.blazen.uniffi.ImageUpscaleProviderDefaults as UniffiImageUpscaleProviderDefaults
import dev.zorpx.blazen.uniffi.MusicRequest
import dev.zorpx.blazen.uniffi.OpenAiCompatConfig as UniffiOpenAiCompatConfig
import dev.zorpx.blazen.uniffi.SpeechRequest
import dev.zorpx.blazen.uniffi.ThreeDProviderDefaults as UniffiThreeDProviderDefaults
import dev.zorpx.blazen.uniffi.ThreeDRequest
import dev.zorpx.blazen.uniffi.ThreeDResult
import dev.zorpx.blazen.uniffi.TranscriptionProviderDefaults as UniffiTranscriptionProviderDefaults
import dev.zorpx.blazen.uniffi.TranscriptionRequest
import dev.zorpx.blazen.uniffi.TranscriptionResult
import dev.zorpx.blazen.uniffi.UpscaleRequest
import dev.zorpx.blazen.uniffi.VideoProviderDefaults as UniffiVideoProviderDefaults
import dev.zorpx.blazen.uniffi.VideoRequest
import dev.zorpx.blazen.uniffi.VideoResult
import dev.zorpx.blazen.uniffi.VoiceCloneRequest
import dev.zorpx.blazen.uniffi.VoiceCloningProviderDefaults as UniffiVoiceCloningProviderDefaults
import dev.zorpx.blazen.uniffi.VoiceHandle
import dev.zorpx.blazen.uniffi.customProviderFromForeign as uniffiCustomProviderFromForeign
import dev.zorpx.blazen.uniffi.lmStudio as uniffiLmStudio
import dev.zorpx.blazen.uniffi.ollama as uniffiOllama
import dev.zorpx.blazen.uniffi.openaiCompat as uniffiOpenaiCompat

// ---------------------------------------------------------------------------
// ApiProtocol
// ---------------------------------------------------------------------------
//
// The UniFFI-generated `ApiProtocol` is already an idiomatic Kotlin sealed
// class (with a `data class OpenAi` variant and a `Custom` object). We
// re-export it as a typealias rather than wrapping — wrapping would force
// a runtime conversion on every `CustomProvider` factory call for zero
// idiomatic gain. Helper factories below sugar the construction down to
// one-liners that match how Python (`ApiProtocol.openai(cfg)`) and Node
// (`ApiProtocol.openAI(cfg)`) build the same value.

/**
 * Selects how a [CustomProvider] talks to its backend for completion calls.
 *
 * Two variants:
 * - [`OpenAi`][UniffiApiProtocol.OpenAi] — framework handles HTTP, SSE
 *   parsing, tool calls, retries. The wrapped [OpenAiCompatConfig]
 *   supplies base URL, model, optional API key, headers, and query
 *   parameters.
 * - [`Custom`][UniffiApiProtocol.Custom] — every completion method is
 *   dispatched to the host-language [CustomProvider] implementation. No
 *   additional transport configuration is required.
 *
 * Use the package-level [openAi] / [custom] factories to build values
 * idiomatically, or construct the sealed-class variants directly.
 */
public typealias ApiProtocol = UniffiApiProtocol

/** Configuration for an OpenAI-compatible provider backend. */
public typealias OpenAiCompatConfig = UniffiOpenAiCompatConfig

/**
 * Sugar constructor: build an `ApiProtocol.OpenAi` variant. Equivalent
 * to `UniffiApiProtocol.OpenAi(config)` — provided so Kotlin code reads
 * like the Python/Node/Swift counterparts.
 *
 * The typealias [ApiProtocol] points at the UniFFI-generated sealed
 * class; reach its variants through the original namespace
 * (`UniffiApiProtocol.OpenAi`, `UniffiApiProtocol.Custom`) since Kotlin
 * typealiases don't re-export nested members.
 */
public fun openAi(config: OpenAiCompatConfig): ApiProtocol = UniffiApiProtocol.OpenAi(config)

/**
 * Sugar constructor: the `ApiProtocol.Custom` singleton. Equivalent to
 * `UniffiApiProtocol.Custom` — provided for symmetry with [openAi].
 */
public fun custom(): ApiProtocol = UniffiApiProtocol.Custom

/**
 * String discriminator for the protocol variant — `"openai"` or `"custom"`.
 *
 * Mirrors the `kind` property exposed by other-language wrappers (Python's
 * `ApiProtocol.kind`, Node's `apiProtocol.kind`). Useful for logging and
 * for foreign callers that need a stable string identifier without
 * pattern-matching on the sealed class.
 */
public val ApiProtocol.kind: String
    get() =
        when (this) {
            is UniffiApiProtocol.OpenAi -> "openai"
            UniffiApiProtocol.Custom -> "custom"
        }

/**
 * The [OpenAiCompatConfig] payload of an `ApiProtocol.OpenAi` value,
 * or `null` for `ApiProtocol.Custom`.
 *
 * Renamed to `openAiConfig` to avoid clashing with the existing `config`
 * field on the sealed-class variant (`(this as UniffiApiProtocol.OpenAi).config`).
 */
public val ApiProtocol.openAiConfig: OpenAiCompatConfig?
    get() =
        when (this) {
            is UniffiApiProtocol.OpenAi -> this.config
            UniffiApiProtocol.Custom -> null
        }

// ---------------------------------------------------------------------------
// Provider defaults (12 records: BaseProviderDefaults + 11 role-specific)
// ---------------------------------------------------------------------------
//
// All defaults records generated by UniFFI are already idiomatic Kotlin
// `data class`es with `companion object` blocks for extension. We
// re-export each one as a typealias and document its current shape.
// V1 carries no functional fields beyond the optional `base` (a hook
// placeholder) and the completion-specific `systemPrompt` /
// `toolsJson` / `responseFormatJson` triple — see CLAUDE.md and the
// upstream `blazen_llm::providers::defaults` module for the design.

/**
 * Universal provider defaults applicable across every provider role.
 *
 * V1 carries no functional fields — the placeholder boolean
 * [`reserved`][UniffiBaseProviderDefaults.reserved] exists only so
 * UniFFI's foreign-language code generator doesn't produce a zero-field
 * record (which generates slightly awkward foreign code). Future
 * versions may attach a role-agnostic `before_request` hook here.
 */
public typealias BaseProviderDefaults = UniffiBaseProviderDefaults

/**
 * Defaults applied to every chat completion call.
 *
 * Per-field semantics:
 * - [`base`][UniffiCompletionProviderDefaults.base] — optional [BaseProviderDefaults]
 *   carried for forward compatibility.
 * - [`systemPrompt`][UniffiCompletionProviderDefaults.systemPrompt] — prepended as a
 *   `system`-role message if the request lacks one.
 * - [`toolsJson`][UniffiCompletionProviderDefaults.toolsJson] — JSON-encoded
 *   `Vec<ToolDefinition>`. Merged into the request's tool list — request-
 *   supplied tools win on name collision.
 * - [`responseFormatJson`][UniffiCompletionProviderDefaults.responseFormatJson] —
 *   JSON-encoded `serde_json::Value` for the OpenAI-style `response_format`
 *   field. Set only if the request lacks one.
 */
public typealias CompletionProviderDefaults = UniffiCompletionProviderDefaults

/** Embedding-role defaults. V1 composes only [BaseProviderDefaults]. */
public typealias EmbeddingProviderDefaults = UniffiEmbeddingProviderDefaults

/** Text-to-speech (audio synthesis) defaults. V1 composes only [BaseProviderDefaults]. */
public typealias AudioSpeechProviderDefaults = UniffiAudioSpeechProviderDefaults

/** Music generation defaults. V1 composes only [BaseProviderDefaults]. */
public typealias AudioMusicProviderDefaults = UniffiAudioMusicProviderDefaults

/** Speech-to-text (transcription) defaults. V1 composes only [BaseProviderDefaults]. */
public typealias TranscriptionProviderDefaults = UniffiTranscriptionProviderDefaults

/** Voice-cloning defaults. V1 composes only [BaseProviderDefaults]. */
public typealias VoiceCloningProviderDefaults = UniffiVoiceCloningProviderDefaults

/** Image generation defaults. V1 composes only [BaseProviderDefaults]. */
public typealias ImageGenerationProviderDefaults = UniffiImageGenerationProviderDefaults

/** Image upscale defaults. V1 composes only [BaseProviderDefaults]. */
public typealias ImageUpscaleProviderDefaults = UniffiImageUpscaleProviderDefaults

/** Background-removal defaults. V1 composes only [BaseProviderDefaults]. */
public typealias BackgroundRemovalProviderDefaults = UniffiBackgroundRemovalProviderDefaults

/** Video generation defaults. V1 composes only [BaseProviderDefaults]. */
public typealias VideoProviderDefaults = UniffiVideoProviderDefaults

/** 3D model generation defaults. V1 composes only [BaseProviderDefaults]. */
public typealias ThreeDProviderDefaults = UniffiThreeDProviderDefaults

// ---------------------------------------------------------------------------
// BaseProvider
// ---------------------------------------------------------------------------

/**
 * A [`CompletionModel`][dev.zorpx.blazen.uniffi.CompletionModel] wrapped
 * with applied [CompletionProviderDefaults].
 *
 * Construct via:
 * - [`BaseProvider.fromCompletionModel(model)`][UniffiBaseProvider.Companion.fromCompletionModel]
 *   — wrap an existing model with empty defaults.
 * - [`BaseProvider.withCompletionDefaults(model, defaults)`][UniffiBaseProvider.Companion.withCompletionDefaults]
 *   — wrap with explicit defaults.
 *
 * Mutate via the `with*` builders, each of which returns a fresh
 * `BaseProvider` (clone-with-mutation, matching Rust's `Arc<BaseProvider>`
 * semantics):
 * - [`withSystemPrompt(prompt)`][UniffiBaseProvider.withSystemPrompt]
 * - [`withToolsJson(json)`][UniffiBaseProvider.withToolsJson]
 * - [`withResponseFormatJson(json)`][UniffiBaseProvider.withResponseFormatJson]
 * - [`withDefaults(defaults)`][UniffiBaseProvider.withDefaults]
 *
 * Inspect with [`defaults()`][UniffiBaseProvider.defaults],
 * [`modelId()`][UniffiBaseProvider.modelId], and unwrap with
 * [`asCompletionModel()`][UniffiBaseProvider.asCompletionModel] for use
 * with APIs that take a generic `CompletionModel`.
 *
 * `BaseProvider` is [AutoCloseable] — close it (or rely on the JVM
 * cleaner) to release the underlying native handle.
 */
public typealias BaseProvider = UniffiBaseProvider

/**
 * Convenience accessor: the default system prompt currently configured
 * on this provider, or `null` if none is set.
 *
 * Equivalent to `defaults().systemPrompt`, but reads more naturally as
 * a property at call sites that only need the system prompt.
 */
public val BaseProvider.systemPrompt: String?
    get() = defaults().systemPrompt

/**
 * Convenience accessor: the JSON-encoded default tool list currently
 * configured on this provider, or `null` if none is set.
 *
 * Equivalent to `defaults().toolsJson`.
 */
public val BaseProvider.toolsJson: String?
    get() = defaults().toolsJson

/**
 * Convenience accessor: the JSON-encoded default `response_format`
 * currently configured on this provider, or `null` if none is set.
 *
 * Equivalent to `defaults().responseFormatJson`.
 */
public val BaseProvider.responseFormatJson: String?
    get() = defaults().responseFormatJson

// ---------------------------------------------------------------------------
// CustomProvider + CustomProviderHandle re-exports
// ---------------------------------------------------------------------------
//
// The UniFFI-generated `CustomProvider` interface declares 16 typed
// suspend methods (plus a sync `providerId()`). Foreign users implement
// it on their own type and pass an instance to [Blazen.customProvider]
// to obtain a [CustomProviderHandle] usable wherever Blazen expects a
// provider. Method signatures, request/result records, and exception
// type all come from the UniFFI surface — we only add the abstract
// base class below, which fills in throwing `Unsupported` defaults for
// each capability so that user subclasses only override the methods
// their provider actually supports.

/**
 * The user-extensible provider trait the host implements directly.
 *
 * Re-exported from the UniFFI-generated package
 * [`dev.zorpx.blazen.uniffi.CustomProvider`][UniffiCustomProvider]. The
 * interface declares 16 typed `suspend` methods covering completion,
 * streaming, embedding, and ten compute/media surfaces, plus one sync
 * accessor [`providerId`][UniffiCustomProvider.providerId].
 *
 * Kotlin interfaces cannot supply default `suspend fun` implementations
 * (the JVM stdlib's `default` method mechanism does not interact with
 * suspend continuations), so any user implementing this interface
 * directly must provide every method by hand. Extend
 * [CustomProviderBase] instead to inherit `Unsupported`-throwing
 * defaults for every capability and override only the methods the
 * provider actually supports.
 */
public typealias CustomProvider = UniffiCustomProvider

/**
 * Handle to a registered [CustomProvider] usable wherever Blazen expects a
 * provider.
 *
 * Returned by the [Blazen.ollama], [Blazen.lmStudio], [Blazen.openaiCompat],
 * and [Blazen.customProvider] factories. Implements the same 16 typed
 * methods as [CustomProvider]; calls dispatch through the inner handle
 * which applies any per-instance defaults attached via the builders
 * (`withSystemPrompt`, `withToolsJson`, ...) before forwarding to the
 * underlying [CustomProvider].
 *
 * The handle owns a native resource — close it (or rely on the JVM
 * cleaner) to release the underlying allocation.
 */
public typealias CustomProviderHandle = UniffiCustomProviderHandle

// ---------------------------------------------------------------------------
// CustomProviderBase — abstract class with Unsupported-throwing defaults
// ---------------------------------------------------------------------------

/**
 * Abstract base class that implements every [CustomProvider] method with
 * `throw BlazenException.Unsupported("...")`.
 *
 * Extend this class and override only the capabilities your provider
 * supports — every other method will surface a typed
 * [`BlazenException.Unsupported`][BlazenException.Unsupported] to the
 * caller, mirroring the Rust `CustomProvider` trait's `Unsupported`
 * default impls.
 *
 * ```kotlin
 * class MyTtsProvider : CustomProviderBase() {
 *     override fun providerId(): String = "my-tts"
 *
 *     override suspend fun textToSpeech(request: SpeechRequest): AudioResult {
 *         // ... call your backend, return AudioResult ...
 *     }
 * }
 *
 * val handle = Blazen.customProvider(MyTtsProvider())
 * val audio = handle.textToSpeech(SpeechRequest(text = "hello", voiceId = "alloy", ...))
 * ```
 *
 * Why an abstract class and not interface default methods? Kotlin
 * interfaces support default implementations for plain `fun`, but not
 * for `suspend fun` — the compiler refuses to emit the JVM bridge
 * because the default-method machinery does not understand suspend
 * continuations. Abstract classes can carry `suspend` bodies, so we
 * route the defaults through one.
 */
public abstract class CustomProviderBase : CustomProvider {
    /**
     * Stable provider identifier surfaced in logs and metrics. Override
     * to return a meaningful string for your provider (e.g.
     * `"my-tts"`). The default returns `"custom"` so existing logs do
     * not explode if a subclass forgets to override.
     */
    override fun providerId(): String = "custom"

    /** Perform a non-streaming chat completion. */
    override suspend fun complete(request: CompletionRequest): CompletionResponse =
        throw BlazenException.Unsupported("complete not supported by ${providerId()}")

    /** Perform a streaming chat completion, pushing chunks into the sink. */
    override suspend fun stream(request: CompletionRequest, sink: CompletionStreamSink): Unit =
        throw BlazenException.Unsupported("stream not supported by ${providerId()}")

    /** Embed one or more texts. */
    override suspend fun embed(texts: List<String>): EmbeddingResponse =
        throw BlazenException.Unsupported("embed not supported by ${providerId()}")

    /** Synthesize speech from text. */
    override suspend fun textToSpeech(request: SpeechRequest): AudioResult =
        throw BlazenException.Unsupported("textToSpeech not supported by ${providerId()}")

    /** Generate music from a prompt. */
    override suspend fun generateMusic(request: MusicRequest): AudioResult =
        throw BlazenException.Unsupported("generateMusic not supported by ${providerId()}")

    /** Generate sound effects from a prompt. */
    override suspend fun generateSfx(request: MusicRequest): AudioResult =
        throw BlazenException.Unsupported("generateSfx not supported by ${providerId()}")

    /** Clone a voice from reference audio. */
    override suspend fun cloneVoice(request: VoiceCloneRequest): VoiceHandle =
        throw BlazenException.Unsupported("cloneVoice not supported by ${providerId()}")

    /** List voices known to the provider. */
    override suspend fun listVoices(): List<VoiceHandle> =
        throw BlazenException.Unsupported("listVoices not supported by ${providerId()}")

    /** Delete a previously-cloned voice. */
    override suspend fun deleteVoice(voice: VoiceHandle): Unit =
        throw BlazenException.Unsupported("deleteVoice not supported by ${providerId()}")

    /** Generate images from a prompt. */
    override suspend fun generateImage(request: ImageRequest): ImageResult =
        throw BlazenException.Unsupported("generateImage not supported by ${providerId()}")

    /** Upscale an existing image. */
    override suspend fun upscaleImage(request: UpscaleRequest): ImageResult =
        throw BlazenException.Unsupported("upscaleImage not supported by ${providerId()}")

    /** Generate a video from a text prompt. */
    override suspend fun textToVideo(request: VideoRequest): VideoResult =
        throw BlazenException.Unsupported("textToVideo not supported by ${providerId()}")

    /** Generate a video from a source image + prompt. */
    override suspend fun imageToVideo(request: VideoRequest): VideoResult =
        throw BlazenException.Unsupported("imageToVideo not supported by ${providerId()}")

    /** Transcribe audio to text. */
    override suspend fun transcribe(request: TranscriptionRequest): TranscriptionResult =
        throw BlazenException.Unsupported("transcribe not supported by ${providerId()}")

    /** Generate a 3D model. */
    override suspend fun generate3d(request: ThreeDRequest): ThreeDResult =
        throw BlazenException.Unsupported("generate3d not supported by ${providerId()}")

    /** Remove the background from an image. */
    override suspend fun removeBackground(request: BackgroundRemovalRequest): ImageResult =
        throw BlazenException.Unsupported("removeBackground not supported by ${providerId()}")
}

// ---------------------------------------------------------------------------
// Idiomatic factories — Blazen.ollama / Blazen.lmStudio / Blazen.openaiCompat /
// Blazen.customProvider
// ---------------------------------------------------------------------------
//
// The UniFFI generator emits four top-level factories (`ollama`,
// `lmStudio`, `openaiCompat`, `customProviderFromForeign`). We surface
// them as methods on [Blazen] so Kotlin callers reach them through a
// single namespace and so the names match the public-facing API in the
// other bindings (`Blazen.ollama(...)`, `Blazen.openaiCompat(...)`).

/**
 * Build a [CustomProviderHandle] targeting a local Ollama server.
 *
 * Equivalent to [Blazen.openaiCompat] with `base_url =
 * http://{host}:{port}/v1` and no API key.
 *
 * @param host hostname or IP of the Ollama server (default: `"localhost"`)
 * @param port TCP port of the Ollama server (default: `11434`)
 * @param model model identifier to request (e.g. `"llama3"`)
 */
public fun Blazen.ollama(
    host: String = "localhost",
    port: UShort = 11_434u,
    model: String,
): CustomProviderHandle = uniffiOllama(model = model, host = host, port = port)

/**
 * Build a [CustomProviderHandle] targeting a local LM Studio server.
 *
 * Equivalent to [Blazen.openaiCompat] with `base_url =
 * http://{host}:{port}/v1` and no API key.
 *
 * @param host hostname or IP of the LM Studio server (default: `"localhost"`)
 * @param port TCP port of the LM Studio server (default: `1234`)
 * @param model model identifier to request
 */
public fun Blazen.lmStudio(
    host: String = "localhost",
    port: UShort = 1_234u,
    model: String,
): CustomProviderHandle = uniffiLmStudio(model = model, host = host, port = port)

/**
 * Build a [CustomProviderHandle] for an arbitrary OpenAI-compatible
 * backend.
 *
 * Use for vLLM, llama.cpp's server, TGI, hosted OpenAI-compat services
 * — anything that speaks the OpenAI chat-completions wire format. The
 * supplied [OpenAiCompatConfig] selects base URL, model, auth method,
 * headers, and query parameters.
 *
 * @param providerId stable identifier surfaced in logs and metrics
 * @param config endpoint configuration
 */
public fun Blazen.openaiCompat(
    providerId: String,
    config: OpenAiCompatConfig,
): CustomProviderHandle = uniffiOpenaiCompat(providerId = providerId, config = config)

/**
 * Wrap a host-implemented [CustomProvider] into a [CustomProviderHandle]
 * usable wherever Blazen expects a provider.
 *
 * The supplied implementation typically extends [CustomProviderBase] so
 * unimplemented methods surface as
 * [`BlazenException.Unsupported`][BlazenException.Unsupported] instead
 * of `NotImplementedError`.
 *
 * @param impl the host-language [CustomProvider] implementation
 */
public fun Blazen.customProvider(impl: CustomProvider): CustomProviderHandle =
    uniffiCustomProviderFromForeign(provider = impl)
