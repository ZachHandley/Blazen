package blazen

// Tests for the typed CustomProvider Go surface. Exercises:
//
//   - Factory functions (Ollama, LMStudio) produce a CustomProviderHandle
//     whose ProviderId() matches the upstream Rust label, without making
//     a network call.
//   - Embedding UnsupportedCustomProvider + overriding a single method
//     (TextToSpeech) routes through the Rust handle, dispatches back into
//     the Go-side override, and returns the override's payload.
//   - Methods left at their UnsupportedCustomProvider defaults return a
//     typed Unsupported error when invoked via the handle round-trip.
//
// The tests do not require any network or model file — they only exercise
// the local FFI plumbing.

import (
	"strings"
	"testing"
)

// TestOllamaConstructsProvider builds an Ollama handle and asserts the
// upstream provider id propagates through the handle without making a
// network call.
func TestOllamaConstructsProvider(t *testing.T) {
	handle := Ollama("localhost", 11434, "llama3")
	if handle == nil {
		t.Fatal("Ollama returned nil handle")
	}
	defer handle.Destroy()
	if id := handle.ProviderId(); id != "ollama" {
		t.Fatalf("Ollama provider id = %q; want %q", id, "ollama")
	}
}

// TestLMStudioFactory builds an LM Studio handle and asserts the upstream
// provider id propagates through the handle.
func TestLMStudioFactory(t *testing.T) {
	handle := LMStudio("localhost", 1234, "qwen2.5")
	if handle == nil {
		t.Fatal("LMStudio returned nil handle")
	}
	defer handle.Destroy()
	if id := handle.ProviderId(); id != "lm_studio" {
		t.Fatalf("LMStudio provider id = %q; want %q", id, "lm_studio")
	}
}

// stubTTS is a host-side CustomProvider that overrides only TextToSpeech;
// every other method falls through to the embedded
// UnsupportedCustomProvider defaults.
type stubTTS struct {
	UnsupportedCustomProvider
}

func (*stubTTS) ProviderId() string { return "stub-tts" }

func (*stubTTS) TextToSpeech(_ SpeechRequest) (AudioResult, error) {
	// Construct via field names; the Timing field defaults to a zero-valued
	// RequestTiming (not re-exported in the public package), which is the
	// "no timing data yet" shape expected by the wire format.
	//
	// Metadata is serialised as `serde_json::Value` on the Rust side; an
	// invalid-JSON string round-trips to "null", so we use a literal JSON
	// string here so the assertion can compare the post-roundtrip value.
	return AudioResult{
		Audio:        nil,
		AudioSeconds: 1.5,
		Metadata:     `"stub-tts-fired"`,
	}, nil
}

// TestSubclassTextToSpeechRoutesToOverride verifies that wrapping a
// foreign CustomProvider via CustomProviderFrom dispatches TextToSpeech
// back into the Go-side override and returns the override's payload.
func TestSubclassTextToSpeechRoutesToOverride(t *testing.T) {
	handle := CustomProviderFrom(&stubTTS{})
	if handle == nil {
		t.Fatal("CustomProviderFrom returned nil handle")
	}
	defer handle.Destroy()

	if id := handle.ProviderId(); id != "stub-tts" {
		t.Fatalf("ProviderId() = %q; want %q", id, "stub-tts")
	}

	result, err := handle.TextToSpeech(SpeechRequest{Text: "hello world"})
	if err != nil {
		t.Fatalf("TextToSpeech returned unexpected error: %v", err)
	}
	if result.Metadata != `"stub-tts-fired"` {
		t.Errorf("AudioResult.Metadata = %q; want %q (override did not fire)",
			result.Metadata, `"stub-tts-fired"`)
	}
	if result.AudioSeconds != 1.5 {
		t.Errorf("AudioResult.AudioSeconds = %v; want 1.5", result.AudioSeconds)
	}
}

// TestUnimplementedMethodReturnsUnsupported invokes a method that the
// stubTTS leaves at the UnsupportedCustomProvider default. The round-trip
// through the Rust handle must surface a typed Unsupported error whose
// message identifies the offending method.
func TestUnimplementedMethodReturnsUnsupported(t *testing.T) {
	handle := CustomProviderFrom(&stubTTS{})
	if handle == nil {
		t.Fatal("CustomProviderFrom returned nil handle")
	}
	defer handle.Destroy()

	_, err := handle.GenerateImage(ImageRequest{Prompt: "a cat"})
	if err == nil {
		t.Fatal("GenerateImage returned nil error; expected Unsupported")
	}
	msg := err.Error()
	if !strings.Contains(msg, "Unsupported") {
		t.Errorf("error message %q does not contain \"Unsupported\"", msg)
	}
	if !strings.Contains(msg, "generate_image") {
		t.Errorf("error message %q does not name the unsupported method", msg)
	}
}
