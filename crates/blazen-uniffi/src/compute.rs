//! Non-LLM compute surface for the UniFFI bindings.
//!
//! Exposes the four upstream compute modalities that don't fit Blazen's
//! `CompletionModel` / `EmbeddingModel` shape:
//!
//! - **Text-to-speech** — local Piper backend (feature-gated) and fal.ai
//!   cloud TTS, behind a single [`TtsModel`] handle.
//! - **Speech-to-text** — local whisper.cpp backend (feature-gated) and
//!   fal.ai cloud transcription, behind a single [`SttModel`] handle.
//! - **Image generation** — local diffusion-rs backend (feature-gated) and
//!   fal.ai cloud image generation, behind a single [`ImageGenModel`]
//!   handle.
//!
//! The opaque handles parallel [`crate::llm::CompletionModel`] /
//! [`crate::llm::EmbeddingModel`]: foreign callers receive an `Arc<Self>`
//! from a factory function and dispatch through async + `_blocking`
//! methods. Each modality has its own factory per concrete backend
//! ([`new_piper_tts_model`], [`new_fal_tts_model`], ...) so the foreign
//! caller picks the implementation, but downstream code only sees the
//! generic handle.
//!
//! ## Wire-format shape
//!
//! The upstream [`AudioGeneration`](blazen_llm::compute::AudioGeneration) /
//! [`Transcription`](blazen_llm::compute::Transcription) /
//! [`ImageGeneration`](blazen_llm::compute::ImageGeneration) traits carry
//! rich request and response types (`SpeechRequest`, `TranscriptionRequest`,
//! `ImageRequest`, plus `AudioResult` / `TranscriptionResult` / `ImageResult`
//! with timing / cost / metadata / segments). UniFFI's UDL grammar collapses
//! poorly under that level of nesting, so this module flattens the
//! request/response shapes to:
//!
//! - **TTS input**: `text`, `voice`, `language` — provider-specific knobs
//!   (speed, model overrides, ...) are intentionally elided. Callers needing
//!   fine control should drop down to the Rust `blazen-llm` API.
//! - **TTS output** ([`TtsResult`]): `audio_base64` (empty when the provider
//!   only returned a URL), `mime_type`, `duration_ms`.
//! - **STT input**: `audio_source` — interpreted as a local file path by
//!   the whisper.cpp backend, and as a URL or `data:` URI by fal.ai.
//!   `language` is an optional ISO-639-1 hint.
//! - **STT output** ([`SttResult`]): `transcript`, `language` (empty when
//!   the provider didn't report one), `duration_ms`.
//! - **Image input**: `prompt`, `negative_prompt`, `width`, `height`,
//!   `num_images`. `model` overrides the provider's default endpoint.
//! - **Image output** ([`ImageGenResult`]): `Vec<Media>` reusing
//!   [`crate::llm::Media`]. URL-only outputs surface with `data_base64` set
//!   to the URL string and `mime_type` populated from the upstream
//!   [`MediaType`](blazen_llm::MediaType).
//!
//! ## Backend availability
//!
//! Piper and diffusion-rs upstream providers are still pre-engine stubs (see
//! `blazen-audio-piper` and `blazen-image-diffusion` — both crates compile
//! their options structs and validation but don't ship an inference engine
//! yet). The Piper / Diffusion factories here construct the underlying
//! provider and return a working handle, but calls to
//! [`TtsModel::synthesize`] / [`ImageGenModel::generate`] surface as
//! [`BlazenError::Provider`] with the upstream "engine not available"
//! message — matching the parity behaviour `blazen-py` exposes today.
//! Wiring the real engines is tracked in the upstream roadmap (Phase 5.3 /
//! Phase 9) and lights up automatically once the upstream trait impls land.

use std::sync::Arc;

use async_trait::async_trait;
use blazen_llm::MediaOutput as CoreMediaOutput;
use blazen_llm::MediaType as CoreMediaType;
use blazen_llm::compute::{
    AudioGeneration, ImageGeneration, ImageRequest as CoreImageRequest,
    SpeechRequest as CoreSpeechRequest, Transcription,
    TranscriptionRequest as CoreTranscriptionRequest,
};

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::Media;
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// The result of a text-to-speech synthesis call.
///
/// `audio_base64` is the empty string when the upstream provider returned a
/// URL only (the URL travels in the `data_base64` slot of a downstream
/// [`Media`] when callers route through [`crate::llm::Media`]; pure TTS
/// callers should detect the empty `audio_base64` and fall back to fetching
/// the URL themselves). `mime_type` reflects the upstream
/// [`MediaType`](blazen_llm::MediaType); `duration_ms` is zero when the
/// provider didn't report timing.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TtsResult {
    pub audio_base64: String,
    pub mime_type: String,
    pub duration_ms: u64,
}

/// The result of a speech-to-text transcription call.
///
/// `language` is the empty string when the provider didn't report a
/// detected language. `duration_ms` reflects the upstream
/// [`RequestTiming::total_ms`](blazen_llm::RequestTiming) — zero when the
/// backend didn't measure it.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SttResult {
    pub transcript: String,
    pub language: String,
    pub duration_ms: u64,
}

/// The result of an image-generation call.
///
/// `images[i].kind` is always `"image"`. `data_base64` contains either the
/// raw base64 bytes (when the upstream `MediaOutput.base64` field is
/// populated) or the URL string (when only `MediaOutput.url` is set);
/// callers must inspect `mime_type` and treat the field accordingly.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ImageGenResult {
    pub images: Vec<Media>,
}

// ---------------------------------------------------------------------------
// Internal backend traits
// ---------------------------------------------------------------------------

/// Object-safe TTS adapter that unifies upstream `AudioGeneration` impls and
/// the still-stubbed `PiperProvider` behind a single dispatch trait.
#[async_trait]
trait TtsBackend: Send + Sync {
    async fn synthesize(
        &self,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, blazen_llm::BlazenError>;
}

/// Object-safe STT adapter that unifies whisper.cpp and fal.ai `Transcription`
/// impls behind a single dispatch trait.
#[async_trait]
trait SttBackend: Send + Sync {
    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, blazen_llm::BlazenError>;
}

/// Object-safe image-generation adapter that unifies upstream
/// `ImageGeneration` impls and the still-stubbed `DiffusionProvider` behind
/// a single dispatch trait.
#[async_trait]
trait ImageGenBackend: Send + Sync {
    async fn generate(
        &self,
        prompt: String,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_images: Option<u32>,
        model: Option<String>,
    ) -> Result<ImageGenResult, blazen_llm::BlazenError>;
}

// ---------------------------------------------------------------------------
// Upstream → wire-format conversions
// ---------------------------------------------------------------------------

/// Render an upstream [`MediaOutput`](blazen_llm::MediaOutput) as a wire
/// [`Media`] entry tagged with the given `kind` (`"image"`, `"audio"`,
/// `"video"`).
///
/// Prefers the upstream `base64` payload when present; falls back to the
/// upstream `url` so URL-only providers still surface a usable handle (the
/// `mime_type` field tells callers whether to base64-decode or fetch).
fn media_output_to_media(kind: &str, output: &CoreMediaOutput) -> Media {
    let mime_type = output.media_type.mime().to_owned();
    let data_base64 = output
        .base64
        .clone()
        .or_else(|| output.url.clone())
        .or_else(|| output.raw_content.clone())
        .unwrap_or_default();
    Media {
        kind: kind.to_owned(),
        mime_type,
        data_base64,
    }
}

/// Map an upstream [`blazen_llm::BlazenError`] into the UniFFI surface error
/// with a stable `kind` discriminator chosen by the caller.
///
/// Matches the convention `providers.rs` established for local-backend
/// init errors (`"PiperSynthesis"`, `"WhisperTranscribe"`,
/// `"DiffusionGeneration"`, `"FalQueue"`).
fn provider_error(kind: &str, provider: &str, err: blazen_llm::BlazenError) -> BlazenError {
    BlazenError::Provider {
        kind: kind.to_owned(),
        message: err.to_string(),
        provider: Some(provider.to_owned()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

// ---------------------------------------------------------------------------
// FAL backend adapters
// ---------------------------------------------------------------------------

/// Adapter implementing [`TtsBackend`] over fal.ai's
/// [`AudioGeneration::text_to_speech`].
struct FalTtsAdapter {
    inner: Arc<blazen_llm::providers::fal::FalProvider>,
    model: Option<String>,
}

#[async_trait]
impl TtsBackend for FalTtsAdapter {
    async fn synthesize(
        &self,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, blazen_llm::BlazenError> {
        let mut req = CoreSpeechRequest::new(text);
        if let Some(v) = voice {
            req = req.with_voice(v);
        }
        if let Some(l) = language {
            req = req.with_language(l);
        }
        if let Some(m) = self.model.clone() {
            req = req.with_model(m);
        }
        let result = AudioGeneration::text_to_speech(self.inner.as_ref(), req).await?;
        let (audio_base64, mime_type) = result
            .audio
            .first()
            .map(|clip| {
                let mime = clip.media.media_type.mime().to_owned();
                let data = clip
                    .media
                    .base64
                    .clone()
                    .or_else(|| clip.media.url.clone())
                    .unwrap_or_default();
                (data, mime)
            })
            .unwrap_or_default();
        let duration_ms = result.timing.total_ms.unwrap_or(0);
        Ok(TtsResult {
            audio_base64,
            mime_type,
            duration_ms,
        })
    }
}

/// Adapter implementing [`SttBackend`] over fal.ai's
/// [`Transcription::transcribe`].
struct FalSttAdapter {
    inner: Arc<blazen_llm::providers::fal::FalProvider>,
    model: Option<String>,
}

#[async_trait]
impl SttBackend for FalSttAdapter {
    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, blazen_llm::BlazenError> {
        let mut req = CoreTranscriptionRequest::new(audio_source);
        if let Some(l) = language {
            req = req.with_language(l);
        }
        if let Some(m) = self.model.clone() {
            req = req.with_model(m);
        }
        let result = Transcription::transcribe(self.inner.as_ref(), req).await?;
        Ok(SttResult {
            transcript: result.text,
            language: result.language.unwrap_or_default(),
            duration_ms: result.timing.total_ms.unwrap_or(0),
        })
    }
}

/// Adapter implementing [`ImageGenBackend`] over fal.ai's
/// [`ImageGeneration::generate_image`].
struct FalImageGenAdapter {
    inner: Arc<blazen_llm::providers::fal::FalProvider>,
    model: Option<String>,
}

#[async_trait]
impl ImageGenBackend for FalImageGenAdapter {
    async fn generate(
        &self,
        prompt: String,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_images: Option<u32>,
        model: Option<String>,
    ) -> Result<ImageGenResult, blazen_llm::BlazenError> {
        let mut req = CoreImageRequest::new(prompt);
        if let Some(np) = negative_prompt {
            req = req.with_negative_prompt(np);
        }
        if let (Some(w), Some(h)) = (width, height) {
            req = req.with_size(w, h);
        }
        if let Some(n) = num_images {
            req = req.with_count(n);
        }
        if let Some(m) = model.or_else(|| self.model.clone()) {
            req = req.with_model(m);
        }
        let result = ImageGeneration::generate_image(self.inner.as_ref(), req).await?;
        let images = result
            .images
            .iter()
            .map(|img| media_output_to_media("image", &img.media))
            .collect();
        Ok(ImageGenResult { images })
    }
}

// ---------------------------------------------------------------------------
// Piper TTS adapter (feature-gated)
// ---------------------------------------------------------------------------

/// Adapter implementing [`TtsBackend`] over the local Piper provider.
///
/// The upstream `blazen-audio-piper` crate does not yet implement
/// [`AudioGeneration`](blazen_llm::compute::AudioGeneration); calls surface
/// the "engine not available" message via [`BlazenError::Provider`] with
/// `kind = "PiperSynthesis"`. The adapter exists today so foreign callers
/// can wire up Piper construction (and detect engine availability through
/// the eventual real call site) without their code changing when the
/// engine lands.
#[cfg(feature = "piper")]
struct PiperTtsAdapter {
    inner: Arc<blazen_llm::PiperProvider>,
}

#[cfg(feature = "piper")]
#[async_trait]
impl TtsBackend for PiperTtsAdapter {
    async fn synthesize(
        &self,
        _text: String,
        _voice: Option<String>,
        _language: Option<String>,
    ) -> Result<TtsResult, blazen_llm::BlazenError> {
        let detail = if self.inner.engine_available() {
            "piper text-to-speech engine is compiled in but the AudioGeneration trait bridge has not been wired yet"
        } else {
            "piper engine not available: build with the `piper/engine` feature"
        };
        Err(blazen_llm::BlazenError::provider("piper", detail))
    }
}

// ---------------------------------------------------------------------------
// Whisper STT adapter (feature-gated)
// ---------------------------------------------------------------------------

/// Adapter implementing [`SttBackend`] over the local whisper.cpp provider.
///
/// `audio_source` is interpreted as a local filesystem path; remote URLs
/// and base64 payloads are rejected upstream with
/// [`BlazenError::Unsupported`]. The wire-format `SttResult` collapses the
/// upstream segment list down to the full `text` and detected `language`;
/// callers needing time-aligned segments should drop down to the Rust
/// `blazen-llm` API.
#[cfg(feature = "whispercpp")]
struct WhisperSttAdapter {
    inner: Arc<blazen_llm::WhisperCppProvider>,
}

#[cfg(feature = "whispercpp")]
#[async_trait]
impl SttBackend for WhisperSttAdapter {
    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, blazen_llm::BlazenError> {
        let mut req = CoreTranscriptionRequest::from_file(audio_source);
        if let Some(l) = language {
            req = req.with_language(l);
        }
        let result = Transcription::transcribe(self.inner.as_ref(), req).await?;
        Ok(SttResult {
            transcript: result.text,
            language: result.language.unwrap_or_default(),
            duration_ms: result.timing.total_ms.unwrap_or(0),
        })
    }
}

// ---------------------------------------------------------------------------
// Diffusion image-gen adapter (feature-gated)
// ---------------------------------------------------------------------------

/// Adapter implementing [`ImageGenBackend`] over the local diffusion-rs
/// provider.
///
/// The upstream `blazen-image-diffusion` crate does not yet implement
/// [`ImageGeneration`](blazen_llm::compute::ImageGeneration); calls surface
/// a [`BlazenError::Provider`] with `kind = "DiffusionGeneration"` whose
/// message includes the resolved dimensions / steps so foreign callers can
/// confirm their options reached the engine layer. Wired through once the
/// upstream Phase 5.3 work lands.
#[cfg(feature = "diffusion")]
struct DiffusionImageGenAdapter {
    inner: Arc<blazen_llm::DiffusionProvider>,
}

#[cfg(feature = "diffusion")]
#[async_trait]
impl ImageGenBackend for DiffusionImageGenAdapter {
    async fn generate(
        &self,
        _prompt: String,
        _negative_prompt: Option<String>,
        _width: Option<u32>,
        _height: Option<u32>,
        _num_images: Option<u32>,
        _model: Option<String>,
    ) -> Result<ImageGenResult, blazen_llm::BlazenError> {
        let detail = format!(
            "diffusion-rs image generation is not yet wired to the engine pipeline \
             (provider would have rendered {}x{} with {} steps, scheduler={})",
            self.inner.width(),
            self.inner.height(),
            self.inner.num_inference_steps(),
            self.inner.scheduler(),
        );
        Err(blazen_llm::BlazenError::provider("diffusion", detail))
    }
}

// ---------------------------------------------------------------------------
// Opaque model handles
// ---------------------------------------------------------------------------

/// A text-to-speech model.
///
/// Construct via [`new_piper_tts_model`] (local, feature-gated) or
/// [`new_fal_tts_model`] (cloud). Once obtained, call
/// [`synthesize`](Self::synthesize) (async) or
/// [`synthesize_blocking`](Self::synthesize_blocking) (sync) to generate
/// speech.
#[derive(uniffi::Object)]
pub struct TtsModel {
    inner: Arc<dyn TtsBackend>,
}

impl TtsModel {
    /// Wrap an internal TTS backend in the FFI handle.
    ///
    /// Used by the factory functions below; not exposed across the FFI. The
    /// backend trait is private to this module, so `from_arc` is too —
    /// other modules construct `TtsModel` via the public factory functions.
    fn from_arc(inner: Arc<dyn TtsBackend>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl TtsModel {
    /// Synthesize speech from `text` and return the audio payload.
    ///
    /// `voice` selects a provider-specific voice id; `language` is an
    /// optional ISO-639-1 hint. Both are ignored by providers that don't
    /// support them.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> BlazenResult<TtsResult> {
        self.inner
            .synthesize(text, voice, language)
            .await
            .map_err(|e| provider_error("TtsSynthesis", "tts", e))
    }
}

#[uniffi::export]
impl TtsModel {
    /// Synchronous variant of [`synthesize`](Self::synthesize) — blocks on
    /// the shared Tokio runtime.
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> BlazenResult<TtsResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.synthesize(text, voice, language).await })
    }
}

/// A speech-to-text model.
///
/// Construct via [`new_whisper_stt_model`] (local, feature-gated) or
/// [`new_fal_stt_model`] (cloud). Once obtained, call
/// [`transcribe`](Self::transcribe) (async) or
/// [`transcribe_blocking`](Self::transcribe_blocking) (sync) to transcribe
/// audio.
#[derive(uniffi::Object)]
pub struct SttModel {
    inner: Arc<dyn SttBackend>,
}

impl SttModel {
    /// Wrap an internal STT backend in the FFI handle.
    ///
    /// The backend trait is private to this module, so `from_arc` is too;
    /// foreign callers construct via the per-backend factory functions.
    fn from_arc(inner: Arc<dyn SttBackend>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl SttModel {
    /// Transcribe audio at `audio_source` and return the transcript.
    ///
    /// `audio_source` is interpreted per-backend: the whisper.cpp backend
    /// treats it as a local file path (16-bit PCM mono WAV at 16 kHz);
    /// fal.ai treats it as an HTTP(S) URL or a `data:` URI. `language` is
    /// an optional ISO-639-1 hint — when omitted, providers that support
    /// language detection will auto-detect.
    pub async fn transcribe(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> BlazenResult<SttResult> {
        self.inner
            .transcribe(audio_source, language)
            .await
            .map_err(|e| provider_error("SttTranscribe", "stt", e))
    }
}

#[uniffi::export]
impl SttModel {
    /// Synchronous variant of [`transcribe`](Self::transcribe).
    pub fn transcribe_blocking(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> BlazenResult<SttResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.transcribe(audio_source, language).await })
    }
}

/// An image-generation model.
///
/// Construct via [`new_diffusion_model`] (local, feature-gated) or
/// [`new_fal_image_gen_model`] (cloud). Once obtained, call
/// [`generate`](Self::generate) (async) or
/// [`generate_blocking`](Self::generate_blocking) (sync) to render images.
#[derive(uniffi::Object)]
pub struct ImageGenModel {
    inner: Arc<dyn ImageGenBackend>,
}

impl ImageGenModel {
    /// Wrap an internal image-generation backend in the FFI handle.
    ///
    /// The backend trait is private to this module, so `from_arc` is too;
    /// foreign callers construct via the per-backend factory functions.
    fn from_arc(inner: Arc<dyn ImageGenBackend>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl ImageGenModel {
    /// Generate `num_images` images for `prompt` at the given dimensions.
    ///
    /// `negative_prompt` describes content to avoid; `model` overrides the
    /// provider's default endpoint (e.g. a specific fal.ai model id).
    /// Backends ignore knobs they don't support.
    pub async fn generate(
        self: Arc<Self>,
        prompt: String,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_images: Option<u32>,
        model: Option<String>,
    ) -> BlazenResult<ImageGenResult> {
        self.inner
            .generate(prompt, negative_prompt, width, height, num_images, model)
            .await
            .map_err(|e| provider_error("ImageGeneration", "image-gen", e))
    }
}

#[uniffi::export]
impl ImageGenModel {
    /// Synchronous variant of [`generate`](Self::generate).
    pub fn generate_blocking(
        self: Arc<Self>,
        prompt: String,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_images: Option<u32>,
        model: Option<String>,
    ) -> BlazenResult<ImageGenResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move {
            this.generate(prompt, negative_prompt, width, height, num_images, model)
                .await
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: convert a known MIME string to a MediaType for round-tripping
// ---------------------------------------------------------------------------

/// Best-effort parse of an IANA MIME string into an upstream
/// [`MediaType`](blazen_llm::MediaType).
///
/// Unknown MIMEs collapse to [`MediaType::Other`] preserving the raw string.
/// Used by image-result conversion when the upstream provider doesn't
/// pre-populate `MediaOutput.media_type` and the caller wants a stable
/// mime string on the wire.
#[allow(dead_code)]
fn parse_media_type(mime: &str) -> CoreMediaType {
    match mime {
        "image/png" => CoreMediaType::Png,
        "image/jpeg" => CoreMediaType::Jpeg,
        "image/webp" => CoreMediaType::WebP,
        "image/gif" => CoreMediaType::Gif,
        "image/svg+xml" => CoreMediaType::Svg,
        "audio/mpeg" => CoreMediaType::Mp3,
        "audio/wav" => CoreMediaType::Wav,
        "audio/ogg" => CoreMediaType::Ogg,
        "audio/flac" => CoreMediaType::Flac,
        "video/mp4" => CoreMediaType::Mp4,
        "video/webm" => CoreMediaType::WebM,
        other => CoreMediaType::Other {
            mime: other.to_owned(),
        },
    }
}

// ---------------------------------------------------------------------------
// Piper TTS factory
// ---------------------------------------------------------------------------

/// Build a local Piper text-to-speech model.
///
/// `model_id` selects a Piper voice model (e.g. `"en_US-amy-medium"`).
/// `speaker_id` selects a speaker for multi-speaker voice models;
/// `sample_rate` overrides the model's native sample rate. Returns a
/// [`TtsModel`] handle whose [`synthesize`](TtsModel::synthesize) call
/// surfaces the upstream "engine not available" error until the Piper
/// Phase 9 wiring lands — but construction succeeds so foreign callers
/// can wire option plumbing today.
#[cfg(feature = "piper")]
#[uniffi::export]
pub fn new_piper_tts_model(
    model_id: Option<String>,
    speaker_id: Option<u32>,
    sample_rate: Option<u32>,
) -> BlazenResult<Arc<TtsModel>> {
    let opts = blazen_llm::PiperOptions {
        model_id,
        speaker_id,
        sample_rate,
        cache_dir: None,
    };
    let provider =
        blazen_llm::PiperProvider::from_options(opts).map_err(|e| BlazenError::Provider {
            kind: "PiperInit".into(),
            message: e.to_string(),
            provider: Some("piper".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let adapter = PiperTtsAdapter {
        inner: Arc::new(provider),
    };
    Ok(TtsModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// Whisper STT factory
// ---------------------------------------------------------------------------

/// Build a local whisper.cpp speech-to-text model.
///
/// `model` selects a Whisper variant by name (case-insensitive:
/// `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`); unrecognised
/// values default to `Small`. `device` accepts the same format strings as
/// `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`).
/// `language` is an optional default ISO-639-1 hint (overridable per
/// [`SttModel::transcribe`] call).
#[cfg(feature = "whispercpp")]
#[uniffi::export]
pub fn new_whisper_stt_model(
    model: Option<String>,
    device: Option<String>,
    language: Option<String>,
) -> BlazenResult<Arc<SttModel>> {
    let model = match model.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("tiny") => blazen_llm::WhisperModel::Tiny,
        Some("base") => blazen_llm::WhisperModel::Base,
        Some("medium") => blazen_llm::WhisperModel::Medium,
        Some("large-v3" | "largev3" | "large_v3" | "large") => blazen_llm::WhisperModel::LargeV3,
        _ => blazen_llm::WhisperModel::Small,
    };
    let opts = blazen_llm::WhisperOptions {
        model,
        device,
        language,
        diarize: None,
        cache_dir: None,
    };
    let provider = runtime()
        .block_on(async { blazen_llm::WhisperCppProvider::from_options(opts).await })
        .map_err(|e| BlazenError::Provider {
            kind: "WhisperInit".into(),
            message: e.to_string(),
            provider: Some("whispercpp".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let adapter = WhisperSttAdapter {
        inner: Arc::new(provider),
    };
    Ok(SttModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// Diffusion image-gen factory
// ---------------------------------------------------------------------------

/// Build a local diffusion-rs image-generation model.
///
/// `model_id` is the HuggingFace repo id of the Stable Diffusion variant
/// (e.g. `"stabilityai/stable-diffusion-2-1"`). `device` follows the same
/// device-string format as the local-LLM factories. `width` / `height` /
/// `num_inference_steps` / `guidance_scale` set provider defaults applied
/// to every generate call. Calls surface the upstream "engine not yet
/// wired" message until the Phase 5.3 work lands; construction succeeds so
/// foreign callers can plumb their options today.
#[cfg(feature = "diffusion")]
#[uniffi::export]
pub fn new_diffusion_model(
    model_id: Option<String>,
    device: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    num_inference_steps: Option<u32>,
    guidance_scale: Option<f32>,
) -> BlazenResult<Arc<ImageGenModel>> {
    let opts = blazen_llm::DiffusionOptions {
        model_id,
        device,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        scheduler: blazen_llm::DiffusionScheduler::default(),
        cache_dir: None,
        ..blazen_llm::DiffusionOptions::default()
    };
    let provider =
        blazen_llm::DiffusionProvider::from_options(opts).map_err(|e| BlazenError::Provider {
            kind: "DiffusionInit".into(),
            message: e.to_string(),
            provider: Some("diffusion".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let adapter = DiffusionImageGenAdapter {
        inner: Arc::new(provider),
    };
    Ok(ImageGenModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// fal.ai compute factories
// ---------------------------------------------------------------------------

/// Build a fal.ai-backed [`TtsModel`].
///
/// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
/// `model` overrides the default fal TTS endpoint (e.g.
/// `"fal-ai/dia-tts"`); when `None`, the per-call `voice` / `language`
/// arguments decide which endpoint fal routes to.
#[uniffi::export]
pub fn new_fal_tts_model(api_key: String, model: Option<String>) -> BlazenResult<Arc<TtsModel>> {
    let provider = build_fal_provider(api_key)?;
    let adapter = FalTtsAdapter {
        inner: provider,
        model,
    };
    Ok(TtsModel::from_arc(Arc::new(adapter)))
}

/// Build a fal.ai-backed [`SttModel`].
///
/// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
/// `model` overrides the default fal transcription endpoint (e.g.
/// `"fal-ai/whisper"`); when `None`, fal routes to its current default
/// Whisper endpoint.
#[uniffi::export]
pub fn new_fal_stt_model(api_key: String, model: Option<String>) -> BlazenResult<Arc<SttModel>> {
    let provider = build_fal_provider(api_key)?;
    let adapter = FalSttAdapter {
        inner: provider,
        model,
    };
    Ok(SttModel::from_arc(Arc::new(adapter)))
}

/// Build a fal.ai-backed [`ImageGenModel`].
///
/// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
/// `model` overrides the default fal image-gen endpoint (e.g.
/// `"fal-ai/flux/dev"`); when `None`, fal routes to its current default
/// image model. The per-call `model` argument on
/// [`ImageGenModel::generate`] takes precedence over this default when
/// both are set.
#[uniffi::export]
pub fn new_fal_image_gen_model(
    api_key: String,
    model: Option<String>,
) -> BlazenResult<Arc<ImageGenModel>> {
    let provider = build_fal_provider(api_key)?;
    let adapter = FalImageGenAdapter {
        inner: provider,
        model,
    };
    Ok(ImageGenModel::from_arc(Arc::new(adapter)))
}

/// Construct an `Arc<FalProvider>` from a (possibly empty) API key.
///
/// Empty `api_key` falls back to the `FAL_KEY` environment variable via
/// `blazen_llm::keys::resolve_api_key`. Errors are mapped to
/// [`BlazenError::Provider`] with `kind = "FalInit"`.
fn build_fal_provider(
    api_key: String,
) -> BlazenResult<Arc<blazen_llm::providers::fal::FalProvider>> {
    let opts = blazen_llm::types::provider_options::FalOptions {
        base: blazen_llm::types::provider_options::ProviderOptions {
            api_key: if api_key.is_empty() {
                None
            } else {
                Some(api_key)
            },
            model: None,
            base_url: None,
        },
        endpoint: None,
        enterprise: false,
        auto_route_modality: true,
    };
    let provider = blazen_llm::providers::fal::FalProvider::from_options(opts).map_err(|e| {
        BlazenError::Provider {
            kind: "FalInit".into(),
            message: e.to_string(),
            provider: Some("fal".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        }
    })?;
    Ok(Arc::new(provider))
}
