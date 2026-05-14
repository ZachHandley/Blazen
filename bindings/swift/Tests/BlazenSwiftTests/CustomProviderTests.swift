import XCTest
@testable import BlazenSwift
import UniFFIBlazen

/// Tests for the Swift `CustomProvider` surface: factory helpers
/// (`Providers.ollama`, `Providers.lmStudio`), the foreign-implemented
/// `Providers.custom(_:)` wrapper, typed override routing, and the
/// `Unsupported`-throwing protocol-extension defaults.
///
/// These tests do not exercise any network path — they only confirm that
/// the FFI surface links, factories construct handles with the expected
/// `providerId`, foreign overrides actually fire on typed dispatch, and
/// unimplemented methods surface `BlazenError.Unsupported` as documented.
final class CustomProviderTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Blazen.initialize()
    }

    /// `Providers.ollama(model:)` must construct a handle whose
    /// `providerId()` matches the documented `"ollama"` discriminator
    /// without touching the network.
    func testOllamaFactoryConstructsProvider() {
        let handle = Providers.ollama(model: "llama3")
        XCTAssertEqual(handle.providerId(), "ollama")
    }

    /// `Providers.lmStudio(model:)` must construct a handle whose
    /// `providerId()` matches the upstream Rust discriminator
    /// (`"lm_studio"`, snake_case to match the OpenAI-compat preset name).
    func testLMStudioFactory() {
        let handle = Providers.lmStudio(model: "qwen2.5-coder")
        XCTAssertEqual(handle.providerId(), "lm_studio")
    }

    /// A foreign subclass that overrides `textToSpeech(request:)` must
    /// have its override invoked when the wrapped `CustomProviderHandle`
    /// is asked to synthesize speech. The marker payload in the returned
    /// `AudioResult.metadata` proves the override (not a Rust built-in)
    /// ran.
    func testSubclassTextToSpeechRoutesToOverride() async throws {
        let stub = StubTts()
        let handle = Providers.custom(stub)
        XCTAssertEqual(handle.providerId(), "stub-tts")

        let request = SpeechRequest(
            text: "hello",
            voice: nil,
            voiceUrl: nil,
            language: nil,
            speed: nil,
            model: nil,
            parameters: ""
        )
        let result = try await handle.textToSpeech(request: request)
        // `audioSeconds` is a plain f64 — proves the foreign override
        // produced the result (Rust built-ins would never set 1.5).
        XCTAssertEqual(result.audioSeconds, 1.5, accuracy: 1e-6)
        XCTAssertEqual(result.audio.count, 1)
        XCTAssertEqual(result.audio.first?.media.mediaType, "audio/wav")
        XCTAssertEqual(result.audio.first?.sampleRate, 44_100)
        XCTAssertEqual(result.audio.first?.channels, 1)
        // `metadata` is a JSON-encoded value at the FFI boundary; the
        // foreign side returned `{"marker":"stub-tts"}`, which the Rust
        // adapter parses and re-emits in canonical JSON form.
        XCTAssertTrue(
            result.metadata.contains("stub-tts"),
            "metadata should contain the override marker, got: \(result.metadata)"
        )
    }

    /// A `CustomProvider` conformer that does NOT override `generateImage`
    /// inherits the protocol-extension default, which must throw
    /// `BlazenError.Unsupported` so callers can detect missing
    /// capabilities without crashing.
    func testUnimplementedMethodThrowsUnsupported() async {
        let stub = StubTts()
        let handle = Providers.custom(stub)
        let request = ImageRequest(
            prompt: "a cat",
            negativePrompt: nil,
            width: nil,
            height: nil,
            numImages: nil,
            model: nil,
            parameters: ""
        )

        do {
            _ = try await handle.generateImage(request: request)
            XCTFail("Expected generateImage to throw BlazenError.Unsupported")
        } catch let error as BlazenError {
            guard case let .Unsupported(message) = error else {
                XCTFail("Expected .Unsupported, got \(error)")
                return
            }
            XCTAssertTrue(
                message.contains("generateImage"),
                "Unsupported message should reference the method name, got: \(message)"
            )
            XCTAssertTrue(
                message.contains("stub-tts"),
                "Unsupported message should reference the provider id, got: \(message)"
            )
        } catch {
            XCTFail("Expected BlazenError, got \(type(of: error)): \(error)")
        }
    }
}

/// Minimal `CustomProvider` conformer that overrides only
/// `textToSpeech(request:)` so we can verify typed dispatch hits the
/// override and the protocol-extension defaults handle every other
/// method.
private final class StubTts: CustomProvider, @unchecked Sendable {
    func providerId() -> String { "stub-tts" }

    func textToSpeech(request: SpeechRequest) async throws -> AudioResult {
        let media = MediaOutput(
            url: nil,
            base64: nil,
            rawContent: nil,
            mediaType: "audio/wav",
            fileSize: nil,
            metadata: ""
        )
        let audio = GeneratedAudio(
            media: media,
            durationSeconds: 1.5,
            sampleRate: 44_100,
            channels: 1
        )
        let timing = RequestTiming(queueMs: nil, executionMs: nil, totalMs: nil)
        return AudioResult(
            audio: [audio],
            timing: timing,
            cost: nil,
            usage: nil,
            audioSeconds: 1.5,
            // `metadata` must be a JSON-encoded value at the FFI boundary
            // — empty / malformed input round-trips to JSON `null`.
            metadata: "{\"marker\":\"stub-tts\"}"
        )
    }
}
