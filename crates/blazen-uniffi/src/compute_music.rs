//! Music + sound-effect generation surface for the UniFFI bindings.
//!
//! Mirrors [`crate::compute`]'s pattern (opaque [`MusicModel`] handle, one
//! factory function per concrete backend, async methods on the handle plus
//! `_blocking` variants for sync callers, plus a foreign-implementable sink
//! trait for streaming).
//!
//! Three native backends ship behind cargo features:
//!
//! - **`audio-music-musicgen`** — Meta's MusicGen text-to-music (32 kHz mono,
//!   small / medium / large checkpoints).
//! - **`audio-music-stable-audio`** — Stability AI's Stable Audio Open
//!   (44.1 kHz stereo; Small or 1.0 variant).
//! - **`audio-music-audiogen`** — Meta's AudioGen text-to-SFX (16 kHz mono).
//!
//! Plus the cloud-side [`new_fal_music_model`] backed by fal.ai. The fal
//! adapter does not support streaming today — `stream_generate_*_to_sink`
//! delivers a single [`BlazenError::Unsupported`] through `on_error`.
//!
//! ## Wire-format shape
//!
//! Streaming chunks carry raw 32-bit float PCM samples at the backend's
//! sample rate. UniFFI marshals `Vec<f32>` natively across Go / Swift /
//! Kotlin / Ruby, so we avoid per-chunk base64 encode / decode and let the
//! foreign side build the playback buffer directly.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_llm::compute::{AudioGeneration, MusicRequest as CoreMusicRequest};
use futures_util::{Stream, StreamExt};

use crate::compute::build_fal_provider;
use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;
use crate::streaming::clone_error;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// One emission from a streaming music backend.
///
/// `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the backend's expected
/// output sample rate (the same `sample_rate` field on the
/// [`MusicResult`] returned by the non-streaming
/// `generate_music` / `generate_sfx` calls).
///
/// `is_final` is `true` for the final chunk of a generation call;
/// implementations should treat it as a UI hint rather than the
/// authoritative completion signal — the sink's `on_done` callback is the
/// canonical end-of-stream marker.
///
/// `latency_seconds`, when present, is the measured latency from the
/// stream's call-start to the moment this chunk was produced — handy for
/// surfacing first-token-latency metrics through the binding.
#[derive(Debug, Clone, uniffi::Record, serde::Serialize, serde::Deserialize)]
pub struct MusicChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the backend's sample rate.
    pub samples: Vec<f32>,
    /// `true` on the final emitted chunk; otherwise `false`.
    pub is_final: bool,
    /// Optional per-chunk latency from call-start in seconds.
    pub latency_seconds: Option<f32>,
}

/// A fully-rendered music / SFX result.
///
/// `bytes` carries the encoded audio (typically a WAV container for the
/// native backends; whatever the cloud provider returned for fal.ai). The
/// non-empty `url` field signals a URL-only response (e.g. fal.ai returning
/// a CDN link without inlining bytes); `bytes` will be empty in that case.
/// Callers should pick whichever payload is present.
#[derive(Debug, Clone, uniffi::Record)]
pub struct MusicResult {
    /// Encoded audio bytes. Empty when the upstream provider only returned
    /// a URL.
    pub bytes: Vec<u8>,
    /// IANA MIME type of `bytes` (e.g. `"audio/wav"`, `"audio/mpeg"`).
    pub mime_type: String,
    /// Sample rate in Hz. Zero when the upstream provider didn't report
    /// one.
    pub sample_rate: u32,
    /// Channel count (1 = mono, 2 = stereo). Zero when the upstream
    /// provider didn't report it.
    pub channels: u32,
    /// Duration of the clip in seconds. Zero when the upstream provider
    /// didn't report a duration.
    pub duration_seconds: f32,
    /// URL of the audio asset when the upstream provider only returned a
    /// link. Empty string for inline-bytes results.
    pub url: String,
}

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
impl From<blazen_llm::MusicChunk> for MusicChunk {
    fn from(chunk: blazen_llm::MusicChunk) -> Self {
        Self {
            samples: chunk.samples,
            is_final: chunk.is_final,
            latency_seconds: chunk.latency_seconds,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a [`blazen_audio::AudioFormat`] (re-exported by `blazen-llm` as
/// [`AudioMusicFormat`](blazen_llm::AudioMusicFormat)) to its IANA MIME
/// string. Mirrors the helper buried inside `compute.rs` for image / TTS
/// payloads; lifted out so both branches (local and fal) share a single
/// canonical mapping.
#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
fn audio_format_to_mime(f: blazen_llm::AudioMusicFormat) -> &'static str {
    match f {
        blazen_llm::AudioMusicFormat::Wav => "audio/wav",
        blazen_llm::AudioMusicFormat::Mp3 => "audio/mpeg",
        blazen_llm::AudioMusicFormat::Flac => "audio/flac",
        blazen_llm::AudioMusicFormat::Opus => "audio/opus",
        blazen_llm::AudioMusicFormat::Pcm => "audio/pcm",
    }
}

/// Convert a [`MusicError`](blazen_llm::MusicError) into the UniFFI surface
/// error with a stable `kind` discriminator. Matches the `provider_error`
/// helper in `compute.rs`.
#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
fn music_error(kind: &str, provider: &str, err: blazen_llm::MusicError) -> BlazenError {
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

/// Map a top-level upstream [`blazen_llm::BlazenError`] into the UniFFI
/// surface error with a stable `kind`. Used by the fal adapter where the
/// upstream provider call signs in `Result<_, blazen_llm::BlazenError>`
/// rather than `MusicError`.
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
// Internal backend trait (private to this module)
// ---------------------------------------------------------------------------

/// Object-safe music-backend adapter that unifies the native MusicGen /
/// Stable Audio / AudioGen impls and the cloud-side fal.ai impl behind a
/// single dispatch trait.
#[async_trait]
trait MusicBackendAdapter: Send + Sync {
    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError>;

    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError>;

    async fn stream_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>;

    async fn stream_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>;
}

// ---------------------------------------------------------------------------
// Local MusicBackend → adapter
// ---------------------------------------------------------------------------

/// Adapter implementing [`MusicBackendAdapter`] over any
/// [`MusicBackend`](blazen_llm::MusicBackend) implementor (MusicGen,
/// Stable Audio, AudioGen, ...).
#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
struct LocalMusicAdapter {
    inner: Arc<dyn blazen_llm::MusicBackend>,
}

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
fn generated_audio_to_result(audio: blazen_llm::MusicGeneratedAudio) -> MusicResult {
    let mime_type = audio_format_to_mime(audio.format).to_owned();
    MusicResult {
        bytes: audio.bytes,
        mime_type,
        sample_rate: audio.sample_rate,
        channels: u32::from(audio.channels),
        duration_seconds: audio.duration_seconds.unwrap_or(0.0),
        url: String::new(),
    }
}

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
#[async_trait]
impl MusicBackendAdapter for LocalMusicAdapter {
    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        let audio = blazen_llm::MusicBackend::generate_music(
            self.inner.as_ref(),
            &prompt,
            duration_seconds,
        )
        .await
        .map_err(|e| music_error("MusicGeneration", "music", e))?;
        Ok(generated_audio_to_result(audio))
    }

    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        let audio =
            blazen_llm::MusicBackend::generate_sfx(self.inner.as_ref(), &prompt, duration_seconds)
                .await
                .map_err(|e| music_error("MusicGeneration", "music", e))?;
        Ok(generated_audio_to_result(audio))
    }

    async fn stream_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        let stream = blazen_llm::MusicBackend::stream_generate_music(
            self.inner.as_ref(),
            &prompt,
            duration_seconds,
        )
        .await
        .map_err(|e| music_error("MusicGeneration", "music", e))?;
        Ok(Box::pin(stream.map(|item| {
            item.map(MusicChunk::from)
                .map_err(|e| music_error("MusicGeneration", "music", e))
        })))
    }

    async fn stream_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        let stream = blazen_llm::MusicBackend::stream_generate_sfx(
            self.inner.as_ref(),
            &prompt,
            duration_seconds,
        )
        .await
        .map_err(|e| music_error("MusicGeneration", "music", e))?;
        Ok(Box::pin(stream.map(|item| {
            item.map(MusicChunk::from)
                .map_err(|e| music_error("MusicGeneration", "music", e))
        })))
    }
}

// ---------------------------------------------------------------------------
// FAL → adapter
// ---------------------------------------------------------------------------

/// Adapter implementing [`MusicBackendAdapter`] over fal.ai's
/// [`AudioGeneration::generate_music`] / `generate_sfx` entry points.
///
/// fal.ai does not expose a chunk-level music stream today, so the two
/// `stream_*` methods surface [`BlazenError::Unsupported`] immediately —
/// foreign callers see the failure delivered through the sink's `on_error`
/// callback.
struct FalMusicAdapter {
    inner: Arc<blazen_llm::providers::fal::FalProvider>,
    model: Option<String>,
}

impl FalMusicAdapter {
    fn build_request(&self, prompt: String, duration_seconds: f32) -> CoreMusicRequest {
        let mut req = CoreMusicRequest::new(prompt).with_duration(duration_seconds);
        if let Some(m) = self.model.clone() {
            req = req.with_model(m);
        }
        req
    }

    fn result_from_audio(result: blazen_llm::compute::AudioResult) -> MusicResult {
        let (bytes, mime_type, sample_rate, channels, duration_seconds, url) = result
            .audio
            .first()
            .map(|clip| {
                let mime = clip.media.media_type.mime().to_owned();
                let bytes = clip
                    .media
                    .base64
                    .as_deref()
                    .map(|s| {
                        use base64::Engine as _;
                        base64::engine::general_purpose::STANDARD
                            .decode(s)
                            .unwrap_or_default()
                    })
                    .unwrap_or_default();
                let url = clip.media.url.clone().unwrap_or_default();
                let sr = clip.sample_rate.unwrap_or(0);
                let ch = u32::from(clip.channels.unwrap_or(0));
                let dur = clip.duration_seconds.unwrap_or(0.0);
                (bytes, mime, sr, ch, dur, url)
            })
            .unwrap_or_else(|| (Vec::new(), String::new(), 0, 0, 0.0, String::new()));
        MusicResult {
            bytes,
            mime_type,
            sample_rate,
            channels,
            duration_seconds,
            url,
        }
    }
}

#[async_trait]
impl MusicBackendAdapter for FalMusicAdapter {
    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        let req = self.build_request(prompt, duration_seconds);
        let result = AudioGeneration::generate_music(self.inner.as_ref(), req)
            .await
            .map_err(|e| provider_error("MusicGeneration", "fal", e))?;
        Ok(Self::result_from_audio(result))
    }

    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        let req = self.build_request(prompt, duration_seconds);
        let result = AudioGeneration::generate_sfx(self.inner.as_ref(), req)
            .await
            .map_err(|e| provider_error("MusicGeneration", "fal", e))?;
        Ok(Self::result_from_audio(result))
    }

    async fn stream_music(
        &self,
        _prompt: String,
        _duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::Unsupported {
            message: "fal music streaming not supported".to_string(),
        })
    }

    async fn stream_sfx(
        &self,
        _prompt: String,
        _duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::Unsupported {
            message: "fal sfx streaming not supported".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Opaque MusicModel handle
// ---------------------------------------------------------------------------

/// A music / sound-effect generation model.
///
/// Construct via one of the per-backend factory functions
/// ([`new_musicgen_model`], [`new_stable_audio_model`],
/// [`new_audiogen_model`], or [`new_fal_music_model`]). Use the async
/// [`generate_music`](Self::generate_music) / [`generate_sfx`](Self::generate_sfx)
/// methods for one-shot rendering, or [`stream_generate_music_to_sink`] /
/// [`stream_generate_sfx_to_sink`] for chunk-level streaming.
#[derive(uniffi::Object)]
pub struct MusicModel {
    inner: Arc<dyn MusicBackendAdapter>,
}

impl MusicModel {
    /// Wrap an internal music adapter in the FFI handle.
    ///
    /// Used by the factory functions below; not exposed across the FFI —
    /// the adapter trait is module-private.
    fn from_arc(inner: Arc<dyn MusicBackendAdapter>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl MusicModel {
    /// Generate `duration_seconds` of music conditioned on `prompt`.
    pub async fn generate_music(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        self.inner.generate_music(prompt, duration_seconds).await
    }

    /// Generate `duration_seconds` of sound-effect audio conditioned on
    /// `prompt`.
    pub async fn generate_sfx(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        self.inner.generate_sfx(prompt, duration_seconds).await
    }
}

#[uniffi::export]
impl MusicModel {
    /// Synchronous variant of [`generate_music`](Self::generate_music).
    pub fn generate_music_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_music(prompt, duration_seconds).await })
    }

    /// Synchronous variant of [`generate_sfx`](Self::generate_sfx).
    pub fn generate_sfx_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_sfx(prompt, duration_seconds).await })
    }
}

// ---------------------------------------------------------------------------
// Streaming sink + pump
// ---------------------------------------------------------------------------

/// Sink for streaming music / SFX output, implemented in foreign code.
///
/// Symmetric to [`crate::streaming::CompletionStreamSink`]: the streaming
/// engine calls [`on_chunk`](Self::on_chunk) for each emitted chunk, then
/// exactly one of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
/// Implementations should treat the terminal callbacks as cleanup hooks
/// (close channels, complete async iterators, signal flow completion, ...).
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait MusicStreamSink: Send + Sync {
    /// Receive a single chunk from the streaming response.
    ///
    /// Returning an `Err` aborts the stream — the engine delivers the error
    /// via [`on_error`](Self::on_error) and stops dispatching further
    /// chunks.
    async fn on_chunk(&self, chunk: MusicChunk) -> BlazenResult<()>;

    /// Receive the terminal completion signal. Called exactly once at the
    /// end of a successful stream.
    async fn on_done(&self) -> BlazenResult<()>;

    /// Receive a fatal error from the stream. Called exactly once when the
    /// stream fails midway (or fails to start at all).
    async fn on_error(&self, cause: BlazenError) -> BlazenResult<()>;
}

async fn drive_music_stream(
    mut stream: Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>,
    sink: Arc<dyn MusicStreamSink>,
) -> BlazenResult<()> {
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                if let Err(sink_err) = sink.on_chunk(chunk).await {
                    let _ = sink.on_error(clone_error(&sink_err)).await;
                    return Ok(());
                }
            }
            Err(err) => {
                let _ = sink.on_error(err).await;
                return Ok(());
            }
        }
    }
    if let Err(sink_err) = sink.on_done().await {
        let _ = sink.on_error(clone_error(&sink_err)).await;
    }
    Ok(())
}

/// Drive a streaming music-generation call, dispatching each chunk to the
/// sink.
///
/// On success, calls `sink.on_done()` exactly once and returns `Ok(())`.
/// On a backend-side or sink-side failure, calls `sink.on_error(...)` and
/// returns `Ok(())` — error delivery is the sink's responsibility, matching
/// the convention `complete_streaming` established for chat completions.
///
/// The only failure mode that propagates back to the caller is a panic in
/// the sink itself or the runtime; init errors (e.g. fal.ai not supporting
/// streaming, MusicGen weight-download failure) are delivered through
/// `on_error`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn stream_generate_music_to_sink(
    model: Arc<MusicModel>,
    prompt: String,
    duration_seconds: f32,
    sink: Arc<dyn MusicStreamSink>,
) -> BlazenResult<()> {
    let stream = match model.inner.stream_music(prompt, duration_seconds).await {
        Ok(s) => s,
        Err(err) => {
            let _ = sink.on_error(err).await;
            return Ok(());
        }
    };
    drive_music_stream(stream, sink).await
}

/// Drive a streaming SFX-generation call, dispatching each chunk to the
/// sink. Same semantics as [`stream_generate_music_to_sink`].
#[uniffi::export(async_runtime = "tokio")]
pub async fn stream_generate_sfx_to_sink(
    model: Arc<MusicModel>,
    prompt: String,
    duration_seconds: f32,
    sink: Arc<dyn MusicStreamSink>,
) -> BlazenResult<()> {
    let stream = match model.inner.stream_sfx(prompt, duration_seconds).await {
        Ok(s) => s,
        Err(err) => {
            let _ = sink.on_error(err).await;
            return Ok(());
        }
    };
    drive_music_stream(stream, sink).await
}

/// Synchronous variant of [`stream_generate_music_to_sink`] — blocks the
/// current thread on the shared Tokio runtime.
#[uniffi::export]
pub fn stream_generate_music_to_sink_blocking(
    model: Arc<MusicModel>,
    prompt: String,
    duration_seconds: f32,
    sink: Arc<dyn MusicStreamSink>,
) -> BlazenResult<()> {
    runtime().block_on(stream_generate_music_to_sink(
        model,
        prompt,
        duration_seconds,
        sink,
    ))
}

/// Synchronous variant of [`stream_generate_sfx_to_sink`] — blocks the
/// current thread on the shared Tokio runtime.
#[uniffi::export]
pub fn stream_generate_sfx_to_sink_blocking(
    model: Arc<MusicModel>,
    prompt: String,
    duration_seconds: f32,
    sink: Arc<dyn MusicStreamSink>,
) -> BlazenResult<()> {
    runtime().block_on(stream_generate_sfx_to_sink(
        model,
        prompt,
        duration_seconds,
        sink,
    ))
}

// ---------------------------------------------------------------------------
// Device helper (native factories)
// ---------------------------------------------------------------------------

/// Parse a device string into a `candle_core::Device` for the native music
/// backends. Mirrors the helper in `manager.rs` (training surface) but
/// lives here so it doesn't require the `training` feature.
///
/// Accepts the same format strings as `blazen_llm::Device::parse`: `"cpu"`,
/// `"cuda"`, `"cuda:N"`, `"metal"`, `"metal:N"`. `None` / unparseable input
/// falls back to `candle_core::Device::Cpu` so foreign callers get a working
/// CPU pipeline by default.
#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
fn parse_music_device(spec: Option<&str>) -> BlazenResult<Option<candle_core::Device>> {
    let Some(raw) = spec else {
        return Ok(None);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Ok(None);
    }
    if normalized == "cpu" {
        return Ok(Some(candle_core::Device::Cpu));
    }
    let (kind, idx) = match normalized.split_once(':') {
        Some((k, rest)) => {
            let parsed = rest.parse::<usize>().map_err(|e| BlazenError::Validation {
                message: format!("music device {raw:?} has non-numeric index {rest:?}: {e}"),
            })?;
            (k, parsed)
        }
        None => (normalized.as_str(), 0),
    };
    match kind {
        "cuda" => {
            candle_core::Device::new_cuda(idx)
                .map(Some)
                .map_err(|e| BlazenError::Validation {
                    message: format!("cuda:{idx} unavailable: {e}"),
                })
        }
        "metal" => {
            candle_core::Device::new_metal(idx)
                .map(Some)
                .map_err(|e| BlazenError::Validation {
                    message: format!("metal:{idx} unavailable: {e}"),
                })
        }
        other => Err(BlazenError::Validation {
            message: format!(
                "unknown music device {other:?} (want one of: cpu, cuda[:N], metal[:N])"
            ),
        }),
    }
}

fn init_error(kind: &str, provider: &str, msg: impl Into<String>) -> BlazenError {
    BlazenError::Provider {
        kind: kind.to_owned(),
        message: msg.into(),
        provider: Some(provider.to_owned()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

// ---------------------------------------------------------------------------
// Native MusicGen factory
// ---------------------------------------------------------------------------

/// Build a native MusicGen-backed [`MusicModel`].
///
/// `variant` selects the MusicGen checkpoint by name (case-insensitive:
/// `"small"`, `"medium"`, `"large"`); unrecognised values default to
/// `Small`. `device` accepts the same format strings as
/// `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`);
/// `None` defers to the backend's auto-detection (CUDA → Metal → CPU).
/// `cache_dir` overrides the Hugging Face Hub cache directory.
/// `max_duration_seconds` overrides the default 30 s per-call safety cap
/// (hard ceiling stays at `MUSICGEN_MAX_DURATION_HARD_LIMIT`).
#[cfg(feature = "audio-music-musicgen")]
#[uniffi::export]
pub fn new_musicgen_model(
    variant: Option<String>,
    device: Option<String>,
    cache_dir: Option<String>,
    max_duration_seconds: Option<f32>,
) -> BlazenResult<Arc<MusicModel>> {
    let variant = match variant.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("medium") => blazen_llm::MusicgenVariant::Medium,
        Some("large") => blazen_llm::MusicgenVariant::Large,
        _ => blazen_llm::MusicgenVariant::Small,
    };
    let device = parse_music_device(device.as_deref())?;
    let config = blazen_llm::MusicgenConfig {
        variant,
        device,
        cache_dir: cache_dir.map(std::path::PathBuf::from),
        max_duration_seconds: max_duration_seconds.unwrap_or(30.0),
    };
    let backend = blazen_llm::MusicgenBackend::new(config);
    let adapter = LocalMusicAdapter {
        inner: Arc::new(backend),
    };
    Ok(MusicModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// Native Stable Audio factory
// ---------------------------------------------------------------------------

/// Build a native Stable Audio Open-backed [`MusicModel`].
///
/// `variant` selects the Stable Audio Open checkpoint by name
/// (case-insensitive: `"small"`, `"open-1.0"` / `"open1.0"`); unrecognised
/// values default to `Small`. `tokenizer_path` must point at the T5
/// SentencePiece `tokenizer.json` shipped with the Stable Audio Open repo
/// — required because Stable Audio's tokenizer is not auto-downloaded by
/// the backend today. `device` follows the same device-string format as
/// the MusicGen factory. `max_duration_seconds` is accepted for API
/// symmetry but Stable Audio enforces its own variant-dependent ceiling
/// internally.
#[cfg(feature = "audio-music-stable-audio")]
#[uniffi::export]
pub fn new_stable_audio_model(
    variant: Option<String>,
    tokenizer_path: String,
    device: Option<String>,
    _max_duration_seconds: Option<f32>,
) -> BlazenResult<Arc<MusicModel>> {
    let variant = match variant.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("open-1.0" | "open1.0" | "open_1_0" | "open" | "1.0") => {
            blazen_llm::StableAudioVariant::Open1_0
        }
        _ => blazen_llm::StableAudioVariant::Small,
    };
    let device = parse_music_device(device.as_deref())?.unwrap_or(candle_core::Device::Cpu);
    let config = blazen_llm::StableAudioConfig {
        hf_repo: variant.hf_repo().to_string(),
        local_weights_path: None,
        tokenizer_path: std::path::PathBuf::from(tokenizer_path),
        device,
        dtype: candle_core::DType::F32,
        variant,
    };
    let backend = runtime()
        .block_on(async { blazen_llm::StableAudioBackend::load(config).await })
        .map_err(|e| init_error("StableAudioInit", "stable-audio", e.to_string()))?;
    let adapter = LocalMusicAdapter {
        inner: Arc::new(backend),
    };
    Ok(MusicModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// Native AudioGen factory
// ---------------------------------------------------------------------------

/// Build a native AudioGen-backed [`MusicModel`].
///
/// `repo_id` overrides the default Hugging Face repo (defaults to
/// `facebook/audiogen-medium`). `revision` pins a specific commit / tag.
/// `device` / `cache_dir` / `max_duration_seconds` follow the MusicGen
/// factory's conventions.
#[cfg(feature = "audio-music-audiogen")]
#[uniffi::export]
pub fn new_audiogen_model(
    repo_id: Option<String>,
    revision: Option<String>,
    device: Option<String>,
    cache_dir: Option<String>,
    max_duration_seconds: Option<f32>,
) -> BlazenResult<Arc<MusicModel>> {
    let device = parse_music_device(device.as_deref())?;
    let config = blazen_llm::AudioGenConfig {
        repo_id: repo_id.unwrap_or_else(|| "facebook/audiogen-medium".to_string()),
        revision,
        device,
        cache_dir: cache_dir.map(std::path::PathBuf::from),
        max_duration_seconds: max_duration_seconds.unwrap_or(30.0),
    };
    let backend = blazen_llm::AudioGenBackend::new(config);
    let adapter = LocalMusicAdapter {
        inner: Arc::new(backend),
    };
    Ok(MusicModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// fal.ai music factory
// ---------------------------------------------------------------------------

/// Build a fal.ai-backed [`MusicModel`].
///
/// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
/// `model` overrides the default fal music / SFX endpoint (the same
/// override is applied to both `generate_music` and `generate_sfx` calls
/// — fal's per-endpoint dispatch handles the routing).
#[uniffi::export]
pub fn new_fal_music_model(
    api_key: String,
    model: Option<String>,
) -> BlazenResult<Arc<MusicModel>> {
    let provider = build_fal_provider(api_key)?;
    let adapter = FalMusicAdapter {
        inner: provider,
        model,
    };
    Ok(MusicModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn new_fal_music_model_constructs_with_dummy_key() {
        // Plumbing test: factory should not contact fal during construction.
        let model = new_fal_music_model("dummy".to_string(), None);
        assert!(model.is_ok(), "factory failed: {:?}", model.err());
    }

    #[test]
    fn music_chunk_roundtrips_through_serde_json() {
        let chunk = MusicChunk {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            is_final: true,
            latency_seconds: Some(0.125),
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        let back: MusicChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.samples, chunk.samples);
        assert_eq!(back.is_final, chunk.is_final);
        assert_eq!(back.latency_seconds, chunk.latency_seconds);
    }

    #[cfg(any(
        feature = "audio-music-musicgen",
        feature = "audio-music-stable-audio",
        feature = "audio-music-audiogen"
    ))]
    #[test]
    fn audio_format_to_mime_covers_all_variants() {
        use blazen_llm::AudioMusicFormat;
        assert_eq!(audio_format_to_mime(AudioMusicFormat::Wav), "audio/wav");
        assert_eq!(audio_format_to_mime(AudioMusicFormat::Mp3), "audio/mpeg");
        assert_eq!(audio_format_to_mime(AudioMusicFormat::Flac), "audio/flac");
        assert_eq!(audio_format_to_mime(AudioMusicFormat::Opus), "audio/opus");
        assert_eq!(audio_format_to_mime(AudioMusicFormat::Pcm), "audio/pcm");
    }
}
