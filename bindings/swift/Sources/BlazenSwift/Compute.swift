import Foundation
import UniFFIBlazen

/// A text-to-speech model handle. Build one via `Compute.piperTts(...)`
/// (local, feature-gated) or `Compute.falTts(...)` (cloud), then call
/// `synthesize(text:voice:language:)`.
public typealias TtsModel = UniFFIBlazen.TtsModel

/// A speech-to-text model handle. Build one via `Compute.whisperStt(...)`
/// (local, feature-gated) or `Compute.falStt(...)` (cloud), then call
/// `transcribe(audioSource:language:)`.
public typealias SttModel = UniFFIBlazen.SttModel

/// An image-generation model handle. Build one via
/// `Compute.diffusion(...)` (local, feature-gated) or
/// `Compute.falImageGen(...)` (cloud), then call
/// `generate(prompt:...)`.
public typealias ImageGenModel = UniFFIBlazen.ImageGenModel

/// Result of a text-to-speech synthesis call. `audioBase64` is empty when
/// the upstream provider returned a URL only; `mimeType` reflects the
/// upstream media type; `durationMs` is zero when the provider did not
/// report timing.
public typealias TtsResult = UniFFIBlazen.TtsResult

/// Result of a speech-to-text transcription call. `language` is the empty
/// string when the provider did not report a detected language;
/// `durationMs` is zero when the backend did not measure it.
public typealias SttResult = UniFFIBlazen.SttResult

/// Result of an image-generation call. Each `Media` entry's
/// `dataBase64` contains either the raw base64 bytes or, for URL-only
/// providers, the URL string — inspect `mimeType` to decide which.
public typealias ImageGenResult = UniFFIBlazen.ImageGenResult

/// Factory namespace for non-LLM compute modalities (TTS, STT, image
/// generation). Mirrors `Providers` but for the modalities that don't fit
/// the `CompletionModel` / `EmbeddingModel` shape.
public enum Compute {
    /// Build a local Piper text-to-speech model. Available when the
    /// underlying native library was built with the `piper` feature.
    public static func piperTts(
        modelId: String? = nil,
        speakerId: UInt32? = nil,
        sampleRate: UInt32? = nil
    ) throws -> TtsModel {
        try newPiperTtsModel(modelId: modelId, speakerId: speakerId, sampleRate: sampleRate)
    }

    /// Build a fal.ai-backed TTS model. Pass an empty `apiKey` to resolve
    /// it from `FAL_KEY`. `model` overrides the default endpoint
    /// (e.g. `"fal-ai/dia-tts"`).
    public static func falTts(apiKey: String, model: String? = nil) throws -> TtsModel {
        try newFalTtsModel(apiKey: apiKey, model: model)
    }

    /// Build a local whisper.cpp speech-to-text model. Available when
    /// the underlying native library was built with the `whispercpp`
    /// feature.
    public static func whisperStt(
        model: String? = nil,
        device: String? = nil,
        language: String? = nil
    ) throws -> SttModel {
        try newWhisperSttModel(model: model, device: device, language: language)
    }

    /// Build a fal.ai-backed STT model. Pass an empty `apiKey` to resolve
    /// it from `FAL_KEY`.
    public static func falStt(apiKey: String, model: String? = nil) throws -> SttModel {
        try newFalSttModel(apiKey: apiKey, model: model)
    }

    /// Build a local diffusion-rs image-generation model. Available when
    /// the underlying native library was built with the `diffusion`
    /// feature.
    public static func diffusion(
        modelId: String? = nil,
        device: String? = nil,
        width: UInt32? = nil,
        height: UInt32? = nil,
        numInferenceSteps: UInt32? = nil,
        guidanceScale: Float? = nil
    ) throws -> ImageGenModel {
        try newDiffusionModel(
            modelId: modelId,
            device: device,
            width: width,
            height: height,
            numInferenceSteps: numInferenceSteps,
            guidanceScale: guidanceScale
        )
    }

    /// Build a fal.ai-backed image-generation model. Pass an empty
    /// `apiKey` to resolve it from `FAL_KEY`.
    public static func falImageGen(
        apiKey: String,
        model: String? = nil
    ) throws -> ImageGenModel {
        try newFalImageGenModel(apiKey: apiKey, model: model)
    }
}
