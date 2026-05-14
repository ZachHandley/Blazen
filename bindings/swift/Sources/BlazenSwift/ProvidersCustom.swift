import Foundation
import UniFFIBlazen

// MARK: - ApiProtocol

/// Selects how a `CustomProvider` talks to its backend for completion calls.
///
/// - `.openAi(config:)`: the Blazen framework owns HTTP, SSE parsing, tool
///   call dispatch, and retries. The wrapped `OpenAiCompatConfig` supplies
///   the base URL, model, optional API key, headers, and query parameters.
/// - `.custom`: every completion method dispatches to a foreign-implemented
///   `CustomProvider` instance registered via `Providers.custom(_:)` (or
///   one of the built-in factories like `Providers.ollama`). No transport
///   config is needed — the foreign implementation owns the wire format.
///
/// Media-generation calls always route through the foreign-implemented
/// `CustomProvider` regardless of which protocol is selected.
public typealias ApiProtocol = UniFFIBlazen.ApiProtocol

public extension ApiProtocol {
    /// Build the OpenAI-wire-format protocol from an explicit config.
    static func openAi(_ config: OpenAiCompatConfig) -> ApiProtocol {
        .openAi(config: config)
    }

    /// Convenience name for the foreign-dispatched protocol.
    static var customDispatch: ApiProtocol { .custom }

    /// Discriminator string: `"openai"` for the OpenAI-wire variant,
    /// `"custom"` for the foreign-dispatched variant.
    var kind: String {
        switch self {
        case .openAi: return "openai"
        case .custom: return "custom"
        }
    }

    /// The wrapped `OpenAiCompatConfig` when this is `.openAi(...)`;
    /// otherwise `nil`.
    var config: OpenAiCompatConfig? {
        switch self {
        case .openAi(let cfg): return cfg
        case .custom: return nil
        }
    }
}

// MARK: - OpenAiCompatConfig

/// Configuration for an OpenAI-compatible provider backend, consumed by the
/// `.openAi` variant of `ApiProtocol` and by `Providers.openaiCompat(...)`.
public typealias OpenAiCompatConfig = UniFFIBlazen.OpenAiCompatConfig

/// How a `CustomProvider` authenticates with an OpenAI-compatible backend
/// (e.g. `Authorization: Bearer ...`, `x-api-key: ...`, or none).
public typealias AuthMethod = UniFFIBlazen.AuthMethod

/// A simple `(key, value)` pair used to attach extra HTTP headers and query
/// parameters to a `CustomProvider`'s `.openAi` requests.
public typealias KeyValue = UniFFIBlazen.KeyValue

public extension OpenAiCompatConfig {
    /// Build an `OpenAiCompatConfig` with sensible defaults for the
    /// rarely-used fields. Matches the most common call shape:
    /// `OpenAiCompatConfig.make(providerName: ..., baseUrl: ..., apiKey: ..., defaultModel: ...)`.
    ///
    /// Adding a defaulted-arg `init` on the UniFFI-generated record would
    /// shadow (and infinitely recurse into) the memberwise initialiser, so
    /// we expose this convenience via a static factory instead.
    ///
    /// - Parameters:
    ///   - providerName: Human-readable identifier used in logs and model info.
    ///   - baseUrl: API base URL (e.g. `"https://api.openai.com/v1"`).
    ///   - apiKey: Bearer key; pass `""` for backends that don't require auth.
    ///   - defaultModel: Default model id when a request omits one.
    ///   - authMethod: How to send the key. Defaults to `.bearer`.
    ///   - extraHeaders: Additional headers attached to every request.
    ///   - queryParams: Additional query parameters attached to every request.
    ///   - supportsModelListing: Whether the backend implements `GET /models`.
    static func make(
        providerName: String,
        baseUrl: String,
        apiKey: String,
        defaultModel: String,
        authMethod: AuthMethod = .bearer,
        extraHeaders: [KeyValue] = [],
        queryParams: [KeyValue] = [],
        supportsModelListing: Bool = true
    ) -> OpenAiCompatConfig {
        OpenAiCompatConfig(
            providerName: providerName,
            baseUrl: baseUrl,
            apiKey: apiKey,
            defaultModel: defaultModel,
            authMethod: authMethod,
            extraHeaders: extraHeaders,
            queryParams: queryParams,
            supportsModelListing: supportsModelListing
        )
    }
}

// MARK: - Provider defaults (record types)

/// Provider-role-agnostic defaults applicable across every provider role.
///
/// V1 carries no semantic data — the `before_request` hook is exposed on the
/// foreign-implementable `CustomProvider` protocol directly. The single
/// `reserved` flag exists so the underlying UniFFI record is non-empty.
public typealias BaseProviderDefaults = UniFFIBlazen.BaseProviderDefaults

/// Completion-role defaults: default system prompt, default tools, default
/// `response_format`.
public typealias CompletionProviderDefaults = UniFFIBlazen.CompletionProviderDefaults

/// Embedding-role defaults. V1 composes only `base`.
public typealias EmbeddingProviderDefaults = UniFFIBlazen.EmbeddingProviderDefaults

/// Image-generation-role defaults. V1 composes only `base`.
public typealias ImageGenerationProviderDefaults = UniFFIBlazen.ImageGenerationProviderDefaults

/// Image-upscale-role defaults. V1 composes only `base`.
public typealias ImageUpscaleProviderDefaults = UniFFIBlazen.ImageUpscaleProviderDefaults

/// Background-removal-role defaults. V1 composes only `base`.
public typealias BackgroundRemovalProviderDefaults = UniFFIBlazen.BackgroundRemovalProviderDefaults

/// Video-generation-role defaults. V1 composes only `base`.
public typealias VideoProviderDefaults = UniFFIBlazen.VideoProviderDefaults

/// Text-to-speech (audio speech) role defaults. V1 composes only `base`.
public typealias AudioSpeechProviderDefaults = UniFFIBlazen.AudioSpeechProviderDefaults

/// Music-generation-role defaults. V1 composes only `base`.
public typealias AudioMusicProviderDefaults = UniFFIBlazen.AudioMusicProviderDefaults

/// Speech-to-text (transcription) role defaults. V1 composes only `base`.
public typealias TranscriptionProviderDefaults = UniFFIBlazen.TranscriptionProviderDefaults

/// Voice-cloning role defaults. V1 composes only `base`.
public typealias VoiceCloningProviderDefaults = UniFFIBlazen.VoiceCloningProviderDefaults

/// 3D-asset-generation role defaults. V1 composes only `base`.
public typealias ThreeDProviderDefaults = UniFFIBlazen.ThreeDProviderDefaults

// MARK: - BaseProviderDefaults ergonomics

public extension BaseProviderDefaults {
    /// Convenience zero-argument constructor matching the upstream
    /// `Default` impl. Equivalent to `BaseProviderDefaults(reserved: false)`.
    static var empty: BaseProviderDefaults {
        BaseProviderDefaults(reserved: false)
    }
}

// MARK: - CompletionProviderDefaults ergonomics

public extension CompletionProviderDefaults {
    /// Build completion defaults from individual fields with sensible
    /// `nil` defaults. Matches the most common call shape:
    /// `CompletionProviderDefaults(systemPrompt: "...")`.
    ///
    /// - Parameters:
    ///   - base: Universal defaults; defaults to an empty record.
    ///   - systemPrompt: Prepended as a system message when a request lacks one.
    ///   - toolsJson: JSON-encoded `Vec<ToolDefinition>` merged into request tools.
    ///   - responseFormatJson: JSON-encoded `response_format` value applied when missing.
    static func make(
        base: BaseProviderDefaults? = nil,
        systemPrompt: String? = nil,
        toolsJson: String? = nil,
        responseFormatJson: String? = nil
    ) -> CompletionProviderDefaults {
        CompletionProviderDefaults(
            base: base,
            systemPrompt: systemPrompt,
            toolsJson: toolsJson,
            responseFormatJson: responseFormatJson
        )
    }
}

// MARK: - BaseProvider

/// A `CompletionModel` wrapped with applied `CompletionProviderDefaults`.
///
/// Construct via the static factories `BaseProvider.fromCompletionModel(model:)`
/// (wraps an existing model with no defaults) or
/// `BaseProvider.withCompletionDefaults(model:defaults:)` (wraps with explicit
/// defaults). Mutate via the chainable `with*` builder methods, each of
/// which returns a fresh `BaseProvider` so the original handle is never
/// mutated in place.
public typealias BaseProvider = UniFFIBlazen.BaseProvider

public extension BaseProvider {
    /// The wrapped model's identifier (e.g. `"gpt-4o"`,
    /// `"claude-3-5-sonnet"`). Re-exposed as a property because callers
    /// think of the id as metadata, not a method.
    var id: String { modelId() }

    /// Wrap this provider with a new default system prompt. Returns a
    /// fresh `BaseProvider`; the receiver is unchanged.
    @discardableResult
    func withSystemPrompt(_ prompt: String) -> BaseProvider {
        withSystemPrompt(prompt: prompt)
    }

    /// Wrap this provider with a new default tool list, encoded as a
    /// JSON `Vec<ToolDefinition>`. Returns a fresh `BaseProvider`.
    /// Malformed JSON is treated as an empty tool list.
    @discardableResult
    func withToolsJson(_ toolsJson: String) -> BaseProvider {
        withToolsJson(toolsJson: toolsJson)
    }

    /// Wrap this provider with a new default `response_format` value,
    /// encoded as a JSON `serde_json::Value`. Malformed JSON or an empty
    /// string is treated as JSON null. Returns a fresh `BaseProvider`.
    @discardableResult
    func withResponseFormatJson(_ formatJson: String) -> BaseProvider {
        withResponseFormatJson(fmtJson: formatJson)
    }

    /// Replace the entire `CompletionProviderDefaults` record on this
    /// provider in one call. Returns a fresh `BaseProvider`.
    @discardableResult
    func withDefaults(_ defaults: CompletionProviderDefaults) -> BaseProvider {
        withDefaults(defaults: defaults)
    }
}

// MARK: - CustomProvider protocol

/// The user-extensible provider protocol the foreign side implements
/// directly. Mirrors `blazen_llm::CustomProvider` across the UniFFI
/// boundary: 16 typed `async throws` methods (completion, streaming-via-sink,
/// embeddings, plus 13 compute / media methods) and one synchronous
/// `providerId()` accessor.
///
/// Foreign users conform a class to `CustomProvider` and pass an instance
/// to `Providers.custom(_:)` (or `customProviderFromForeign` directly) to
/// obtain a `CustomProviderHandle` usable wherever Blazen expects a
/// provider.
///
/// **Default implementations.** UniFFI's `with_foreign` traits require
/// every method to be implemented at the foreign-language level — there is
/// no cross-FFI Rust "default impl" fallback. The protocol extension below
/// supplies `Unsupported`-throwing defaults for every async method so
/// conformers only override the capabilities their provider actually
/// supports. Only `providerId()` is mandatory because every provider needs
/// a stable identifier for logs and metrics.
public typealias CustomProvider = UniFFIBlazen.CustomProvider

/// Concrete handle returned by the foreign factories (`Providers.ollama`,
/// `Providers.lmStudio`, `Providers.openaiCompat`, `Providers.custom`).
///
/// Internally backed by `blazen_llm::CustomProviderHandle`; exposes the
/// same 16 typed async methods as `CustomProvider`, plus completion-default
/// builder methods (`withSystemPrompt`, `withToolsJson`,
/// `withResponseFormatJson`).
public typealias CustomProviderHandle = UniFFIBlazen.CustomProviderHandle

// MARK: - CustomProvider default implementations

/// `Unsupported`-throwing defaults for every `async throws` method on
/// `CustomProvider`. Conformers override only the methods their provider
/// supports; everything else surfaces a `BlazenError.Unsupported` with a
/// descriptive message so callers can detect missing capabilities without
/// crashing.
///
/// Swift protocol extensions DO support `async throws` default
/// implementations, which is the idiomatic counterpart to Python's
/// `CustomProviderBase` and Node's `CustomProviderBase` class — there is
/// no need for a separate base class.
public extension CustomProvider {
    func complete(request: UniFFIBlazen.CompletionRequest) async throws -> UniFFIBlazen.CompletionResponse {
        throw BlazenError.Unsupported(message: "CustomProvider.complete not implemented by \(providerId())")
    }

    func stream(
        request: UniFFIBlazen.CompletionRequest,
        sink: UniFFIBlazen.CompletionStreamSink
    ) async throws {
        throw BlazenError.Unsupported(message: "CustomProvider.stream not implemented by \(providerId())")
    }

    func embed(texts: [String]) async throws -> UniFFIBlazen.EmbeddingResponse {
        throw BlazenError.Unsupported(message: "CustomProvider.embed not implemented by \(providerId())")
    }

    func textToSpeech(request: UniFFIBlazen.SpeechRequest) async throws -> UniFFIBlazen.AudioResult {
        throw BlazenError.Unsupported(message: "CustomProvider.textToSpeech not implemented by \(providerId())")
    }

    func generateMusic(request: UniFFIBlazen.MusicRequest) async throws -> UniFFIBlazen.AudioResult {
        throw BlazenError.Unsupported(message: "CustomProvider.generateMusic not implemented by \(providerId())")
    }

    func generateSfx(request: UniFFIBlazen.MusicRequest) async throws -> UniFFIBlazen.AudioResult {
        throw BlazenError.Unsupported(message: "CustomProvider.generateSfx not implemented by \(providerId())")
    }

    func cloneVoice(request: UniFFIBlazen.VoiceCloneRequest) async throws -> UniFFIBlazen.VoiceHandle {
        throw BlazenError.Unsupported(message: "CustomProvider.cloneVoice not implemented by \(providerId())")
    }

    func listVoices() async throws -> [UniFFIBlazen.VoiceHandle] {
        throw BlazenError.Unsupported(message: "CustomProvider.listVoices not implemented by \(providerId())")
    }

    func deleteVoice(voice: UniFFIBlazen.VoiceHandle) async throws {
        throw BlazenError.Unsupported(message: "CustomProvider.deleteVoice not implemented by \(providerId())")
    }

    func generateImage(request: UniFFIBlazen.ImageRequest) async throws -> UniFFIBlazen.ImageResult {
        throw BlazenError.Unsupported(message: "CustomProvider.generateImage not implemented by \(providerId())")
    }

    func upscaleImage(request: UniFFIBlazen.UpscaleRequest) async throws -> UniFFIBlazen.ImageResult {
        throw BlazenError.Unsupported(message: "CustomProvider.upscaleImage not implemented by \(providerId())")
    }

    func textToVideo(request: UniFFIBlazen.VideoRequest) async throws -> UniFFIBlazen.VideoResult {
        throw BlazenError.Unsupported(message: "CustomProvider.textToVideo not implemented by \(providerId())")
    }

    func imageToVideo(request: UniFFIBlazen.VideoRequest) async throws -> UniFFIBlazen.VideoResult {
        throw BlazenError.Unsupported(message: "CustomProvider.imageToVideo not implemented by \(providerId())")
    }

    func transcribe(request: UniFFIBlazen.TranscriptionRequest) async throws -> UniFFIBlazen.TranscriptionResult {
        throw BlazenError.Unsupported(message: "CustomProvider.transcribe not implemented by \(providerId())")
    }

    func generate3d(request: UniFFIBlazen.ThreeDRequest) async throws -> UniFFIBlazen.ThreeDResult {
        throw BlazenError.Unsupported(message: "CustomProvider.generate3d not implemented by \(providerId())")
    }

    func removeBackground(request: UniFFIBlazen.BackgroundRemovalRequest) async throws -> UniFFIBlazen.ImageResult {
        throw BlazenError.Unsupported(message: "CustomProvider.removeBackground not implemented by \(providerId())")
    }
}

// MARK: - CustomProvider factories (idiomatic wrappers)

public extension Providers {
    /// Build a `CustomProviderHandle` for a local Ollama server.
    ///
    /// Equivalent to `Providers.openaiCompat(...)` with
    /// `base_url = http://{host}:{port}/v1` and no API key.
    ///
    /// - Parameters:
    ///   - host: Hostname or IP of the Ollama server. Defaults to `"localhost"`.
    ///   - port: TCP port of the Ollama server. Defaults to `11434`.
    ///   - model: Model identifier to use (e.g. `"llama3"`, `"mistral"`).
    static func ollama(
        host: String = "localhost",
        port: UInt16 = 11434,
        model: String
    ) -> CustomProviderHandle {
        UniFFIBlazen.ollama(model: model, host: host, port: port)
    }

    /// Build a `CustomProviderHandle` for a local LM Studio server.
    ///
    /// Equivalent to `Providers.openaiCompat(...)` with
    /// `base_url = http://{host}:{port}/v1` and no API key.
    ///
    /// - Parameters:
    ///   - host: Hostname or IP of the LM Studio server. Defaults to `"localhost"`.
    ///   - port: TCP port of the LM Studio server. Defaults to `1234`.
    ///   - model: Model identifier to use.
    static func lmStudio(
        host: String = "localhost",
        port: UInt16 = 1234,
        model: String
    ) -> CustomProviderHandle {
        UniFFIBlazen.lmStudio(model: model, host: host, port: port)
    }

    /// Build a `CustomProviderHandle` for an arbitrary OpenAI-compatible
    /// backend (vLLM, llama.cpp's server, TGI, hosted OpenAI-compat
    /// services — anything that speaks the official OpenAI
    /// chat-completions wire format).
    ///
    /// - Parameters:
    ///   - providerId: Stable provider identifier used in logs and metrics.
    ///   - config: Connection details (base URL, model, auth, headers, params).
    static func openaiCompat(
        providerId: String,
        config: OpenAiCompatConfig
    ) -> CustomProviderHandle {
        UniFFIBlazen.openaiCompat(providerId: providerId, config: config)
    }

    /// Wrap a foreign-implemented `CustomProvider` as a
    /// `CustomProviderHandle` usable wherever Blazen expects a provider.
    ///
    /// The handle holds an internal adapter that converts UniFFI records
    /// to upstream `blazen_llm::compute` types on each call, and forwards
    /// every method to the supplied `impl`.
    static func custom(_ impl: CustomProvider) -> CustomProviderHandle {
        customProviderFromForeign(provider: impl)
    }
}
