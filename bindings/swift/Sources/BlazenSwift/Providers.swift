import Foundation
import UniFFIBlazen

/// Namespace host for `CustomProvider` factories (see `ProvidersCustom.swift`
/// for `Providers.ollama`, `Providers.lmStudio`, `Providers.openaiCompat`,
/// and `Providers.custom`).
///
/// Per-engine LLM / embedding / TTS / STT / image-generation / video / music
/// / voice-cloning / 3D providers are exposed directly from the generated
/// `UniFFIBlazen` module — instantiate them by name (e.g.
/// `UniFFIBlazen.OpenAiProvider(apiKey:model:baseUrl:)`,
/// `UniFFIBlazen.AnthropicProvider(apiKey:model:baseUrl:)`,
/// `UniFFIBlazen.GeminiProvider(apiKey:model:baseUrl:)`,
/// `UniFFIBlazen.PiperProvider(...)`, `UniFFIBlazen.FasterWhisperProvider(...)`,
/// `UniFFIBlazen.DiffusionProvider(...)`, `UniFFIBlazen.FalLlmProvider(...)`,
/// `UniFFIBlazen.FalTtsProvider(...)`, etc.). The previous `Providers.<engine>(...)`
/// static factories all returned a deleted central `Model` / `EmbeddingModel`
/// type; the per-engine constructors replace them one-to-one with the same
/// argument shape.
public enum Providers {}
