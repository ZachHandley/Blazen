package blazen

import (
	"context"
	"runtime"
	"sync"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// This file wraps the non-LLM compute modalities exposed by UniFFI:
// text-to-speech, speech-to-text, and image generation. Each model is
// an opaque handle obtained from a provider factory (Fal cloud APIs or
// a local runtime like Whisper / Piper / Diffusion).
//
// The wrappers mirror the [CompletionModel] / [EmbeddingModel] shape:
//
//   - Public Go-native result records (no FFI types in the API).
//   - Pointer-typed handles guarded by sync.Once for idempotent Close().
//   - A finalizer attached at construction as a safety net.
//   - Async methods that run the blocking FFI call on a goroutine and
//     honour [context.Context] cancellation via select, alongside
//     synchronous *Blocking variants for scripts and main() functions.
//   - empty-string-means-unset for optional *string FFI parameters and
//     zero-means-unset for optional *uint32 FFI parameters, matching
//     [optString] and [optUint32].
//
// Cancellation semantics match Complete: ctx fires unblock the caller,
// but the Rust-side request continues until it finishes naturally —
// propagation into the runtime is pending an upstream UniFFI feature.

// ---------------------------------------------------------------------
// Result records
// ---------------------------------------------------------------------

// ImageGenResult is the outcome of an [ImageGenModel.Generate] call.
// Each [Media] entry's DataBase64 contains either the raw base64-encoded
// bytes or, for URL-only providers (some fal endpoints), the URL string
// itself — inspect MimeType to decide which.
type ImageGenResult struct {
	Images []Media
}

// SttResult is the outcome of an [SttModel.Transcribe] call.
//
// Transcript is the decoded text. Language is the detected ISO-639-1
// language code (e.g. "en"); empty when the provider did not report one.
// DurationMs is the audio duration in milliseconds; zero when the
// backend did not measure it.
type SttResult struct {
	Transcript string
	Language   string
	DurationMs uint64
}

// TtsResult is the outcome of a [TtsModel.Synthesize] call.
//
// AudioBase64 is the raw audio bytes base64-encoded; for URL-only
// providers it may instead carry a URL string — inspect MimeType to
// decide. MimeType is the IANA media type (e.g. "audio/mpeg",
// "audio/wav"). DurationMs is the synthesized audio duration in
// milliseconds; zero when the provider did not report timing.
type TtsResult struct {
	AudioBase64 string
	MimeType    string
	DurationMs  uint64
}

// imageGenResultFromFFI converts an FFI ImageGenResult into the wrapper
// form.
func imageGenResultFromFFI(r uniffiblazen.ImageGenResult) ImageGenResult {
	return ImageGenResult{Images: mediaSliceFromFFI(r.Images)}
}

// sttResultFromFFI converts an FFI SttResult into the wrapper form.
func sttResultFromFFI(r uniffiblazen.SttResult) SttResult {
	return SttResult{
		Transcript: r.Transcript,
		Language:   r.Language,
		DurationMs: r.DurationMs,
	}
}

// ttsResultFromFFI converts an FFI TtsResult into the wrapper form.
func ttsResultFromFFI(r uniffiblazen.TtsResult) TtsResult {
	return TtsResult{
		AudioBase64: r.AudioBase64,
		MimeType:    r.MimeType,
		DurationMs:  r.DurationMs,
	}
}

// ---------------------------------------------------------------------
// ImageGenModel
// ---------------------------------------------------------------------

// GenerateOpts configures an [ImageGenModel.Generate] call. Every field
// is optional and follows the empty-string-means-unset / zero-means-unset
// convention used elsewhere in this binding.
//
// NegativePrompt steers the model away from undesired content; empty
// disables it. Width and Height set output dimensions in pixels; zero
// uses the provider default. NumImages requests a batch size; zero uses
// the provider default. Model overrides the model id configured at
// construction; empty keeps the default. fal.ai applies the per-call
// override on top of the value passed to [NewFalImageGen].
type GenerateOpts struct {
	NegativePrompt string
	Width          uint32
	Height         uint32
	NumImages      uint32
	Model          string
}

// ImageGenModel is a handle to an image-generation model. Obtain one
// via [NewFalImageGen] or [NewDiffusion] and call
// [ImageGenModel.Generate] or [ImageGenModel.GenerateBlocking] to
// produce images.
//
// ImageGenModel owns a native handle; call [ImageGenModel.Close] when
// finished to release it. A finalizer is attached as a safety net, but
// explicit Close is preferred for predictable resource release.
type ImageGenModel struct {
	inner *uniffiblazen.ImageGenModel
	once  sync.Once
}

// newImageGenModel wraps a raw FFI handle into the public type and
// installs a finalizer so a forgotten Close() still releases the native
// resource. Intended for use by sibling factories in this file —
// callers in user code obtain a [*ImageGenModel] from a factory rather
// than constructing one directly.
func newImageGenModel(inner *uniffiblazen.ImageGenModel) *ImageGenModel {
	m := &ImageGenModel{inner: inner}
	runtime.SetFinalizer(m, func(g *ImageGenModel) { g.Close() })
	return m
}

// Generate produces one or more images for the supplied prompt. The FFI
// call is blocking on the Rust side; to honour ctx cancellation the
// call runs on a background goroutine and the function returns
// ctx.Err() if ctx fires before the model responds. The Rust-side
// request continues until it finishes naturally — cancellation
// propagation into the runtime is a known gap pending an upstream
// UniFFI feature.
func (m *ImageGenModel) Generate(ctx context.Context, prompt string, opts GenerateOpts) (*ImageGenResult, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "image-gen model has been closed"}
	}
	type generateResult struct {
		res uniffiblazen.ImageGenResult
		err error
	}
	done := make(chan generateResult, 1)
	go func() {
		res, err := m.inner.Generate(
			prompt,
			optString(opts.NegativePrompt),
			optUint32(opts.Width),
			optUint32(opts.Height),
			optUint32(opts.NumImages),
			optString(opts.Model),
		)
		done <- generateResult{res: res, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		out := imageGenResultFromFFI(r.res)
		return &out, nil
	}
}

// GenerateBlocking is the synchronous variant of
// [ImageGenModel.Generate]. It does not accept a [context.Context] and
// blocks the calling goroutine until the model responds. Prefer
// [ImageGenModel.Generate] in long-running services where cancellation
// matters; use this in short scripts or main() functions where the
// async wiring is overkill.
func (m *ImageGenModel) GenerateBlocking(prompt string, opts GenerateOpts) (*ImageGenResult, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "image-gen model has been closed"}
	}
	res, err := m.inner.GenerateBlocking(
		prompt,
		optString(opts.NegativePrompt),
		optUint32(opts.Width),
		optUint32(opts.Height),
		optUint32(opts.NumImages),
		optString(opts.Model),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	out := imageGenResultFromFFI(res)
	return &out, nil
}

// Close releases the underlying native handle. It is safe to call
// Close multiple times and from multiple goroutines; subsequent calls
// are no-ops.
func (m *ImageGenModel) Close() {
	m.once.Do(func() {
		if m.inner != nil {
			m.inner.Destroy()
			m.inner = nil
		}
	})
}

// ---------------------------------------------------------------------
// SttModel
// ---------------------------------------------------------------------

// SttModel is a handle to a speech-to-text model. Obtain one via
// [NewFalStt] or [NewWhisperStt] and call [SttModel.Transcribe] or
// [SttModel.TranscribeBlocking] to decode audio.
//
// SttModel owns a native handle; call [SttModel.Close] when finished to
// release it. A finalizer is attached as a safety net, but explicit
// Close is preferred for predictable resource release.
type SttModel struct {
	inner *uniffiblazen.SttModel
	once  sync.Once
}

// newSttModel wraps a raw FFI handle into the public type and installs
// a finalizer so a forgotten Close() still releases the native
// resource. Intended for use by sibling factories in this file.
func newSttModel(inner *uniffiblazen.SttModel) *SttModel {
	m := &SttModel{inner: inner}
	runtime.SetFinalizer(m, func(s *SttModel) { s.Close() })
	return m
}

// Transcribe decodes audio into text. audioSource is a path or URL; the
// Rust side dispatches on the prefix ("http://", "https://", or a
// filesystem path). language is an ISO-639-1 hint (e.g. "en") that
// overrides the model's configured default; pass the empty string to
// let the backend auto-detect.
//
// Cancellation semantics match [ImageGenModel.Generate] — ctx fires
// unblock the caller but the Rust-side decode continues until it
// finishes naturally.
func (m *SttModel) Transcribe(ctx context.Context, audioSource, language string) (*SttResult, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "stt model has been closed"}
	}
	type transcribeResult struct {
		res uniffiblazen.SttResult
		err error
	}
	done := make(chan transcribeResult, 1)
	go func() {
		res, err := m.inner.Transcribe(audioSource, optString(language))
		done <- transcribeResult{res: res, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		out := sttResultFromFFI(r.res)
		return &out, nil
	}
}

// TranscribeBlocking is the synchronous variant of
// [SttModel.Transcribe]. It does not accept a [context.Context] and
// blocks the calling goroutine until decoding completes.
func (m *SttModel) TranscribeBlocking(audioSource, language string) (*SttResult, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "stt model has been closed"}
	}
	res, err := m.inner.TranscribeBlocking(audioSource, optString(language))
	if err != nil {
		return nil, wrapErr(err)
	}
	out := sttResultFromFFI(res)
	return &out, nil
}

// Close releases the underlying native handle. It is safe to call
// Close multiple times and from multiple goroutines; subsequent calls
// are no-ops.
func (m *SttModel) Close() {
	m.once.Do(func() {
		if m.inner != nil {
			m.inner.Destroy()
			m.inner = nil
		}
	})
}

// ---------------------------------------------------------------------
// TtsModel
// ---------------------------------------------------------------------

// TtsModel is a handle to a text-to-speech model. Obtain one via
// [NewFalTts] or [NewPiperTts] and call [TtsModel.Synthesize] or
// [TtsModel.SynthesizeBlocking] to produce audio.
//
// TtsModel owns a native handle; call [TtsModel.Close] when finished to
// release it. A finalizer is attached as a safety net, but explicit
// Close is preferred for predictable resource release.
type TtsModel struct {
	inner *uniffiblazen.TtsModel
	once  sync.Once
}

// newTtsModel wraps a raw FFI handle into the public type and installs
// a finalizer so a forgotten Close() still releases the native
// resource. Intended for use by sibling factories in this file.
func newTtsModel(inner *uniffiblazen.TtsModel) *TtsModel {
	m := &TtsModel{inner: inner}
	runtime.SetFinalizer(m, func(t *TtsModel) { t.Close() })
	return m
}

// Synthesize converts text to audio. voice selects a voice id (provider
// or model-specific); empty uses the provider default. language is an
// ISO-639-1 hint (e.g. "en"); empty uses the provider default and lets
// the backend decide based on the voice and text.
//
// Cancellation semantics match [ImageGenModel.Generate] — ctx fires
// unblock the caller but the Rust-side synthesis continues until it
// finishes naturally.
func (m *TtsModel) Synthesize(ctx context.Context, text, voice, language string) (*TtsResult, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "tts model has been closed"}
	}
	type synthesizeResult struct {
		res uniffiblazen.TtsResult
		err error
	}
	done := make(chan synthesizeResult, 1)
	go func() {
		res, err := m.inner.Synthesize(text, optString(voice), optString(language))
		done <- synthesizeResult{res: res, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		out := ttsResultFromFFI(r.res)
		return &out, nil
	}
}

// SynthesizeBlocking is the synchronous variant of
// [TtsModel.Synthesize]. It does not accept a [context.Context] and
// blocks the calling goroutine until synthesis completes.
func (m *TtsModel) SynthesizeBlocking(text, voice, language string) (*TtsResult, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "tts model has been closed"}
	}
	res, err := m.inner.SynthesizeBlocking(text, optString(voice), optString(language))
	if err != nil {
		return nil, wrapErr(err)
	}
	out := ttsResultFromFFI(res)
	return &out, nil
}

// Close releases the underlying native handle. It is safe to call
// Close multiple times and from multiple goroutines; subsequent calls
// are no-ops.
func (m *TtsModel) Close() {
	m.once.Do(func() {
		if m.inner != nil {
			m.inner.Destroy()
			m.inner = nil
		}
	})
}

// ---------------------------------------------------------------------
// Image-gen provider factories
// ---------------------------------------------------------------------

// NewFalImageGen creates an [ImageGenModel] backed by fal.ai's image
// generation endpoints.
//
// apiKey is the fal.ai API key; pass the empty string to let the
// runtime resolve it from the FAL_KEY environment variable. model
// overrides the default fal image endpoint (e.g. "fal-ai/flux/dev");
// empty routes to fal's current default image model. The per-call
// Model option on [ImageGenModel.Generate] takes precedence over this
// default when both are set.
func NewFalImageGen(apiKey, model string) (*ImageGenModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFalImageGenModel(apiKey, optString(model))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newImageGenModel(inner), nil
}

// DiffusionOpts configures [NewDiffusion].
//
// ModelID is required and identifies the Hugging Face repo (e.g.
// "stabilityai/stable-diffusion-2-1"). Device is "" for auto-select,
// otherwise "cpu", "cuda:0", "metal", etc. Width and Height set the
// output image size in pixels; zero uses the model default (512x512 is
// typical). NumInferenceSteps controls denoising step count; zero uses
// the provider default. GuidanceScale is a pointer because 0.0 is a
// meaningful value (no guidance); pass nil to use the provider default.
type DiffusionOpts struct {
	ModelID           string
	Device            string
	Width             uint32
	Height            uint32
	NumInferenceSteps uint32
	GuidanceScale     *float32
}

// NewDiffusion creates an [ImageGenModel] running locally via the
// diffusion-rs runtime. Available when the native library was built
// with the "diffusion" feature.
func NewDiffusion(opts DiffusionOpts) (*ImageGenModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewDiffusionModel(
		optString(opts.ModelID),
		optString(opts.Device),
		optUint32(opts.Width),
		optUint32(opts.Height),
		optUint32(opts.NumInferenceSteps),
		opts.GuidanceScale,
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newImageGenModel(inner), nil
}

// ---------------------------------------------------------------------
// STT provider factories
// ---------------------------------------------------------------------

// NewFalStt creates an [SttModel] backed by fal.ai's transcription
// endpoints.
//
// apiKey is the fal.ai API key; pass the empty string to let the
// runtime resolve it from the FAL_KEY environment variable. model
// overrides the default fal transcription endpoint (e.g.
// "fal-ai/whisper"); empty routes to fal's current default Whisper
// endpoint.
func NewFalStt(apiKey, model string) (*SttModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFalSttModel(apiKey, optString(model))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newSttModel(inner), nil
}

// WhisperSttOpts configures [NewWhisperStt].
//
// Model selects a Whisper variant by name (case-insensitive: "tiny",
// "base", "small", "medium", "large-v3"); empty uses "base".
// Unrecognised values default to "small" on the Rust side. Device
// accepts the same format strings as Candle's Device parser ("cpu",
// "cuda", "cuda:N", "metal"); empty auto-selects. Language is an
// optional ISO-639-1 hint (e.g. "en") applied by default at the
// per-model level; the per-call language argument on
// [SttModel.Transcribe] overrides it.
type WhisperSttOpts struct {
	Model    string
	Device   string
	Language string
}

// NewWhisperStt creates an [SttModel] running locally via the
// whisper.cpp runtime. Available when the native library was built
// with the "whispercpp" feature.
func NewWhisperStt(opts WhisperSttOpts) (*SttModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewWhisperSttModel(
		optString(opts.Model),
		optString(opts.Device),
		optString(opts.Language),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newSttModel(inner), nil
}

// ---------------------------------------------------------------------
// TTS provider factories
// ---------------------------------------------------------------------

// NewFalTts creates a [TtsModel] backed by fal.ai's TTS endpoints.
//
// apiKey is the fal.ai API key; pass the empty string to let the
// runtime resolve it from the FAL_KEY environment variable. model
// overrides the default fal TTS endpoint (e.g. "fal-ai/dia-tts");
// empty lets the per-call voice / language arguments decide which
// endpoint fal routes to.
func NewFalTts(apiKey, model string) (*TtsModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFalTtsModel(apiKey, optString(model))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newTtsModel(inner), nil
}

// PiperTtsOpts configures [NewPiperTts].
//
// ModelID selects a Piper voice model — either a Hugging Face repo id
// or a local path (e.g. "en_US-amy-medium"); empty leaves the runtime
// to pick a default. SpeakerID selects a speaker for multi-speaker
// voice models; zero uses the default speaker. SampleRate overrides the
// model's native sample rate in Hz; zero uses the model default.
type PiperTtsOpts struct {
	ModelID    string
	SpeakerID  uint32
	SampleRate uint32
}

// NewPiperTts creates a [TtsModel] running locally via the Piper
// runtime. Available when the native library was built with the
// "piper" feature.
func NewPiperTts(opts PiperTtsOpts) (*TtsModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewPiperTtsModel(
		optString(opts.ModelID),
		optUint32(opts.SpeakerID),
		optUint32(opts.SampleRate),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newTtsModel(inner), nil
}
