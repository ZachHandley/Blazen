import Foundation
import UniFFIBlazen

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

// Per-engine compute provider classes (PiperProvider, FalTtsProvider,
// FasterWhisperProvider, FalSttProvider, DiffusionProvider, FalImageGenProvider,
// ...) are exposed directly from the generated `UniFFIBlazen` module —
// call them by name (e.g. `UniFFIBlazen.PiperProvider(...)`,
// `UniFFIBlazen.FalTtsProvider(...)`). The previous `Compute` factory
// namespace was a thin wrapper over a now-deleted set of `new*Model` free
// functions; the per-engine constructors replace it cleanly with the same
// argument shape.
