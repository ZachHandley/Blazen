//! Voice-conversion (RVC and friends) surface for the UniFFI bindings.
//!
//! Mirrors [`crate::compute_music`]'s pattern (opaque [`VcModel`] handle,
//! one factory function per concrete backend, async methods on the handle
//! plus `_blocking` variants for sync callers, plus a foreign-implementable
//! sink trait for streaming).
//!
//! One native backend ships behind a cargo feature today:
//!
//! - **`audio-vc-rvc`** — Retrieval-based Voice Conversion (RVC) engine.
//!   Loads weights from `$BLAZEN_RVC_VOICE_DIR/<voice_id>/` on demand.
//!
//! ## Wire-format shape
//!
//! [`convert_voice`](VcModel::convert_voice) returns a self-describing WAV
//! container ([`VcResult::bytes`]) so callers can write the buffer straight
//! to disk or hand it to a decoder. Streaming uses raw 32-bit float PCM
//! ([`VcChunk::samples`]) at the target voice's native sample rate — UniFFI
//! marshals `Vec<f32>` natively across Go / Swift / Kotlin / Ruby, so the
//! foreign side can build the playback buffer directly.

#[cfg(feature = "audio-vc")]
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;
use crate::streaming::clone_error;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// A registered target speaker that a [`VcModel`] can render source audio
/// into.
///
/// Mirrors [`blazen_llm::TargetVoice`] (when the `audio-vc` feature is on)
/// 1:1 across the FFI boundary so foreign code sees a stable record shape
/// regardless of whether the underlying engine is the native RVC backend
/// or a cloud-side provider added later.
#[derive(Debug, Clone, uniffi::Record, serde::Serialize, serde::Deserialize)]
pub struct TargetVoice {
    /// Backend-scoped identifier passed back to
    /// [`VcModel::convert_voice`] / [`VcModel::register_target_voice`].
    pub id: String,
    /// Optional human-readable display name. `None` when the backend did
    /// not record one.
    pub label: Option<String>,
    /// Native sample rate (Hz) the backend renders this voice at.
    pub sample_rate_hz: u32,
}

#[cfg(feature = "audio-vc")]
impl From<blazen_llm::TargetVoice> for TargetVoice {
    fn from(voice: blazen_llm::TargetVoice) -> Self {
        Self {
            id: voice.id,
            label: voice.label,
            sample_rate_hz: voice.sample_rate_hz,
        }
    }
}

/// One emission from a streaming voice-conversion call.
///
/// `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the target voice's
/// native sample rate (see [`TargetVoice::sample_rate_hz`]).
///
/// `is_final` is purely an advisory hint — the sink's `on_done` callback
/// is the canonical end-of-stream signal, matching the contract used by
/// [`crate::compute_music::MusicChunk`].
///
/// `latency_seconds`, when present, is the measured latency from the
/// stream's call-start to the moment this chunk was produced — handy for
/// surfacing first-token-latency metrics through the binding.
#[derive(Debug, Clone, uniffi::Record, serde::Serialize, serde::Deserialize)]
pub struct VcChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the voice's native sample
    /// rate.
    pub samples: Vec<f32>,
    /// `true` on the final emitted chunk; otherwise `false`. Always
    /// `false` for the RVC backend today (end-of-stream is signalled by
    /// the sink's `on_done` callback).
    pub is_final: bool,
    /// Optional per-chunk latency from call-start in seconds.
    pub latency_seconds: Option<f32>,
}

/// A fully-rendered voice-conversion result.
///
/// `bytes` carries a complete WAV (RIFF/`fmt `/`data`) container holding
/// 16-bit signed little-endian PCM samples at the target voice's native
/// sample rate. `sample_rate` echoes that rate for convenience so callers
/// don't have to re-parse the WAV header.
#[derive(Debug, Clone, uniffi::Record)]
pub struct VcResult {
    /// Encoded audio bytes (WAV container, 16-bit signed PCM).
    pub bytes: Vec<u8>,
    /// IANA MIME type of `bytes` (always `"audio/wav"` for the native
    /// backends shipped today).
    pub mime_type: String,
    /// Sample rate in Hz, taken from the target voice's
    /// [`TargetVoice::sample_rate_hz`].
    pub sample_rate: u32,
    /// Duration of the clip in seconds. Zero when the backend did not
    /// report one (no extra WAV header parsing happens here).
    pub duration_seconds: f32,
}

// ---------------------------------------------------------------------------
// Internal backend trait (private to this module)
// ---------------------------------------------------------------------------

/// Object-safe voice-conversion adapter that unifies the native RVC impl
/// (and any future cloud-side providers) behind a single dispatch trait.
#[async_trait]
trait VcBackendAdapter: Send + Sync {
    async fn convert_voice(
        &self,
        input_audio_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError>;

    async fn list_target_voices(&self) -> Result<Vec<TargetVoice>, BlazenError>;

    async fn register_target_voice(
        &self,
        voice_id: String,
        reference_audio_path: String,
    ) -> Result<(), BlazenError>;

    async fn stream_convert_pcm(
        &self,
        input_pcm: Vec<f32>,
        target_voice_id: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<VcChunk, BlazenError>> + Send>>, BlazenError>;
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

/// Map a [`VcError`](blazen_llm::VcError) into the UniFFI surface error
/// with a stable `kind` discriminator. Matches the `provider_error` /
/// `music_error` helpers used in [`crate::compute`] and
/// [`crate::compute_music`].
#[cfg(feature = "audio-vc")]
fn vc_error(kind: &str, provider: &str, err: blazen_llm::VcError) -> BlazenError {
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
// Local VoiceConversionBackend → adapter
// ---------------------------------------------------------------------------

/// Adapter implementing [`VcBackendAdapter`] over any
/// [`VoiceConversionBackend`](blazen_llm::VoiceConversionBackend)
/// implementor (RVC today, future cloud providers tomorrow).
#[cfg(feature = "audio-vc")]
struct LocalVcAdapter {
    inner: Arc<dyn blazen_llm::VoiceConversionBackend>,
}

#[cfg(feature = "audio-vc")]
#[async_trait]
impl VcBackendAdapter for LocalVcAdapter {
    async fn convert_voice(
        &self,
        input_audio_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError> {
        // Look up the voice first so we know what sample rate to stamp on
        // the result. Backends that don't support listing fall back to a
        // zero sample-rate (callers can still parse the WAV header).
        let sample_rate =
            match blazen_llm::VoiceConversionBackend::list_target_voices(self.inner.as_ref()).await
            {
                Ok(voices) => voices
                    .into_iter()
                    .find(|v| v.id == target_voice_id)
                    .map_or(0, |v| v.sample_rate_hz),
                Err(_) => 0,
            };

        let bytes = blazen_llm::VoiceConversionBackend::convert_voice(
            self.inner.as_ref(),
            Path::new(&input_audio_path),
            &target_voice_id,
        )
        .await
        .map_err(|e| vc_error("VoiceConversion", "vc", e))?;

        Ok(VcResult {
            bytes,
            mime_type: "audio/wav".to_owned(),
            sample_rate,
            duration_seconds: 0.0,
        })
    }

    async fn list_target_voices(&self) -> Result<Vec<TargetVoice>, BlazenError> {
        let voices = blazen_llm::VoiceConversionBackend::list_target_voices(self.inner.as_ref())
            .await
            .map_err(|e| vc_error("VoiceConversion", "vc", e))?;
        Ok(voices.into_iter().map(TargetVoice::from).collect())
    }

    async fn register_target_voice(
        &self,
        voice_id: String,
        reference_audio_path: String,
    ) -> Result<(), BlazenError> {
        blazen_llm::VoiceConversionBackend::register_target_voice(
            self.inner.as_ref(),
            &voice_id,
            Path::new(&reference_audio_path),
        )
        .await
        .map_err(|e| vc_error("VoiceConversion", "vc", e))
    }

    async fn stream_convert_pcm(
        &self,
        input_pcm: Vec<f32>,
        target_voice_id: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<VcChunk, BlazenError>> + Send>>, BlazenError> {
        // Wrap the input Vec<f32> in a one-shot stream — the trait takes a
        // boxed Stream<Item = Vec<f32>> so a streaming caller could in
        // principle chunk the input further, but the UniFFI surface today
        // ships one buffer per call to keep the foreign-side API simple.
        let input_stream = futures_util::stream::once(async move { input_pcm });
        let out_stream = blazen_llm::VoiceConversionBackend::stream_convert(
            self.inner.as_ref(),
            Box::pin(input_stream),
            &target_voice_id,
        )
        .await
        .map_err(|e| vc_error("VoiceConversion", "vc", e))?;

        Ok(Box::pin(out_stream.map(|item| {
            item.map(|samples| VcChunk {
                samples,
                is_final: false,
                latency_seconds: None,
            })
            .map_err(|e| vc_error("VoiceConversion", "vc", e))
        })))
    }
}

// ---------------------------------------------------------------------------
// Opaque VcModel handle
// ---------------------------------------------------------------------------

/// A voice-conversion model.
///
/// Construct via one of the per-backend factory functions (currently just
/// [`new_rvc_model`], gated on `audio-vc-rvc`). Use the async
/// [`convert_voice`](Self::convert_voice) method for one-shot rendering,
/// [`list_target_voices`](Self::list_target_voices) /
/// [`register_target_voice`](Self::register_target_voice) for voice
/// management, or [`stream_convert_pcm_to_sink`] for chunk-level
/// streaming.
#[derive(uniffi::Object)]
pub struct VcModel {
    inner: Arc<dyn VcBackendAdapter>,
}

impl VcModel {
    /// Wrap an internal VC adapter in the FFI handle.
    ///
    /// Used by the factory functions below; not exposed across the FFI —
    /// the adapter trait is module-private.
    #[allow(dead_code)] // Only constructed when `audio-vc` features are on.
    fn from_arc(inner: Arc<dyn VcBackendAdapter>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl VcModel {
    /// Convert the source utterance at `input_audio_path` into the voice
    /// of the registered target speaker `target_voice_id`.
    pub async fn convert_voice(
        self: Arc<Self>,
        input_audio_path: String,
        target_voice_id: String,
    ) -> BlazenResult<VcResult> {
        self.inner
            .convert_voice(input_audio_path, target_voice_id)
            .await
    }

    /// List the target voices this backend can currently render.
    pub async fn list_target_voices(self: Arc<Self>) -> BlazenResult<Vec<TargetVoice>> {
        self.inner.list_target_voices().await
    }

    /// Register a new target voice from the reference utterance at
    /// `reference_audio_path`.
    pub async fn register_target_voice(
        self: Arc<Self>,
        voice_id: String,
        reference_audio_path: String,
    ) -> BlazenResult<()> {
        self.inner
            .register_target_voice(voice_id, reference_audio_path)
            .await
    }
}

#[uniffi::export]
impl VcModel {
    /// Synchronous variant of [`convert_voice`](Self::convert_voice).
    pub fn convert_voice_blocking(
        self: Arc<Self>,
        input_audio_path: String,
        target_voice_id: String,
    ) -> BlazenResult<VcResult> {
        let this = Arc::clone(&self);
        runtime()
            .block_on(async move { this.convert_voice(input_audio_path, target_voice_id).await })
    }

    /// Synchronous variant of
    /// [`list_target_voices`](Self::list_target_voices).
    pub fn list_target_voices_blocking(self: Arc<Self>) -> BlazenResult<Vec<TargetVoice>> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.list_target_voices().await })
    }

    /// Synchronous variant of
    /// [`register_target_voice`](Self::register_target_voice).
    pub fn register_target_voice_blocking(
        self: Arc<Self>,
        voice_id: String,
        reference_audio_path: String,
    ) -> BlazenResult<()> {
        let this = Arc::clone(&self);
        runtime().block_on(async move {
            this.register_target_voice(voice_id, reference_audio_path)
                .await
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming sink + pump
// ---------------------------------------------------------------------------

/// Sink for streaming voice-conversion output, implemented in foreign
/// code.
///
/// Symmetric to [`crate::compute_music::MusicStreamSink`] and
/// [`crate::streaming::CompletionStreamSink`]: the streaming engine calls
/// [`on_chunk`](Self::on_chunk) for each emitted chunk, then exactly one
/// of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
/// Implementations should treat the terminal callbacks as cleanup hooks
/// (close channels, complete async iterators, signal flow completion).
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait VcStreamSink: Send + Sync {
    /// Receive a single chunk from the streaming response.
    ///
    /// Returning an `Err` aborts the stream — the engine delivers the
    /// error via [`on_error`](Self::on_error) and stops dispatching
    /// further chunks.
    async fn on_chunk(&self, chunk: VcChunk) -> BlazenResult<()>;

    /// Receive the terminal completion signal. Called exactly once at the
    /// end of a successful stream.
    async fn on_done(&self) -> BlazenResult<()>;

    /// Receive a fatal error from the stream. Called exactly once when
    /// the stream fails midway (or fails to start at all).
    async fn on_error(&self, err: BlazenError) -> BlazenResult<()>;
}

async fn drive_vc_stream(
    mut stream: Pin<Box<dyn Stream<Item = Result<VcChunk, BlazenError>> + Send>>,
    sink: Arc<dyn VcStreamSink>,
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

/// Drive a streaming voice-conversion call, dispatching each chunk to the
/// sink.
///
/// `input_pcm` is the full source utterance as 32-bit float PCM at the
/// backend's expected source sample rate (typically 16 kHz mono for RVC).
///
/// On success, calls `sink.on_done()` exactly once and returns `Ok(())`.
/// On a backend-side or sink-side failure, calls `sink.on_error(...)` and
/// returns `Ok(())` — error delivery is the sink's responsibility, matching
/// the convention `complete_streaming` and `stream_generate_music_to_sink`
/// established.
///
/// The only failure mode that propagates back to the caller is a panic in
/// the sink itself or the runtime; init errors (e.g. voice-not-found,
/// backend-not-built-with-feature) are delivered through `on_error`.
#[cfg(feature = "audio-vc")]
#[uniffi::export(async_runtime = "tokio")]
pub async fn stream_convert_pcm_to_sink(
    model: Arc<VcModel>,
    input_pcm: Vec<f32>,
    target_voice_id: String,
    sink: Arc<dyn VcStreamSink>,
) -> BlazenResult<()> {
    let stream = match model
        .inner
        .stream_convert_pcm(input_pcm, target_voice_id)
        .await
    {
        Ok(s) => s,
        Err(err) => {
            let _ = sink.on_error(err).await;
            return Ok(());
        }
    };
    drive_vc_stream(stream, sink).await
}

/// Synchronous variant of [`stream_convert_pcm_to_sink`] — blocks the
/// current thread on the shared Tokio runtime.
#[cfg(feature = "audio-vc")]
#[uniffi::export]
pub fn stream_convert_pcm_to_sink_blocking(
    model: Arc<VcModel>,
    input_pcm: Vec<f32>,
    target_voice_id: String,
    sink: Arc<dyn VcStreamSink>,
) -> BlazenResult<()> {
    runtime().block_on(stream_convert_pcm_to_sink(
        model,
        input_pcm,
        target_voice_id,
        sink,
    ))
}

// When the `audio-vc` feature is off, the trait is still defined (so the
// scaffolding compiles and foreign code can still import the type), but
// the streaming free functions are unavailable since they need
// `blazen-llm`'s VC re-exports. Provide stub versions that always raise
// `BlazenError::Unsupported` through the sink so callers see a consistent
// surface across feature configurations.
#[cfg(not(feature = "audio-vc"))]
#[uniffi::export(async_runtime = "tokio")]
pub async fn stream_convert_pcm_to_sink(
    _model: Arc<VcModel>,
    _input_pcm: Vec<f32>,
    _target_voice_id: String,
    sink: Arc<dyn VcStreamSink>,
) -> BlazenResult<()> {
    let _ = sink
        .on_error(BlazenError::Unsupported {
            message: "voice conversion not built into this binary (enable audio-vc)".to_string(),
        })
        .await;
    Ok(())
}

#[cfg(not(feature = "audio-vc"))]
#[uniffi::export]
pub fn stream_convert_pcm_to_sink_blocking(
    model: Arc<VcModel>,
    input_pcm: Vec<f32>,
    target_voice_id: String,
    sink: Arc<dyn VcStreamSink>,
) -> BlazenResult<()> {
    runtime().block_on(stream_convert_pcm_to_sink(
        model,
        input_pcm,
        target_voice_id,
        sink,
    ))
}

// ---------------------------------------------------------------------------
// Device helper (native factories)
// ---------------------------------------------------------------------------

/// Parse a device string into a `candle_core::Device` for the native RVC
/// backend. Mirrors [`crate::compute_music::parse_music_device`] but lives
/// here so it doesn't pull in the music feature graph.
///
/// Accepts the same format strings as `blazen_llm::Device::parse`:
/// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`, `"metal:N"`. `None` /
/// unparseable input falls back to `candle_core::Device::Cpu` so foreign
/// callers get a working CPU pipeline by default.
#[cfg(feature = "audio-vc-rvc")]
fn parse_vc_device(spec: Option<&str>) -> BlazenResult<candle_core::Device> {
    let Some(raw) = spec else {
        return Ok(candle_core::Device::Cpu);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Ok(candle_core::Device::Cpu);
    }
    if normalized == "cpu" {
        return Ok(candle_core::Device::Cpu);
    }
    let (kind, idx) = match normalized.split_once(':') {
        Some((k, rest)) => {
            let parsed = rest.parse::<usize>().map_err(|e| BlazenError::Validation {
                message: format!("vc device {raw:?} has non-numeric index {rest:?}: {e}"),
            })?;
            (k, parsed)
        }
        None => (normalized.as_str(), 0),
    };
    match kind {
        "cuda" => candle_core::Device::new_cuda(idx).map_err(|e| BlazenError::Validation {
            message: format!("cuda:{idx} unavailable: {e}"),
        }),
        "metal" => candle_core::Device::new_metal(idx).map_err(|e| BlazenError::Validation {
            message: format!("metal:{idx} unavailable: {e}"),
        }),
        other => Err(BlazenError::Validation {
            message: format!("unknown vc device {other:?} (want one of: cpu, cuda[:N], metal[:N])"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Native RVC factory
// ---------------------------------------------------------------------------

/// Build a native RVC-backed [`VcModel`].
///
/// `voice_dir` overrides the per-process `BLAZEN_RVC_VOICE_DIR`
/// environment variable that the RVC pipeline reads to locate voice
/// profiles on disk (each voice is expected to live at
/// `<voice_dir>/<voice_id>/` with `model.pth`, `index.index`, and
/// `metadata.json`). When `None`, the existing process-environment value
/// is used unchanged. Setting this from inside the factory mutates global
/// process state via `std::env::set_var` — callers running multiple RVC
/// instances in the same process should pick a single voice directory
/// rather than racing factory calls.
///
/// `device` accepts the same format strings as `blazen_llm::Device::parse`
/// (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`); `None` defers to CPU.
#[cfg(feature = "audio-vc-rvc")]
#[uniffi::export]
pub fn new_rvc_model(
    voice_dir: Option<String>,
    device: Option<String>,
) -> BlazenResult<Arc<VcModel>> {
    if let Some(dir) = voice_dir {
        // SAFETY: `set_var` is `unsafe` on edition 2024 because mutating
        // process environment from multi-threaded code can race with
        // concurrent `getenv` calls in C libraries. The RVC pipeline reads
        // this variable lazily on the first conversion call, so callers
        // who construct the model up-front before spinning off threads are
        // safe; multi-RVC-instance setups should pick a single directory.
        unsafe {
            std::env::set_var("BLAZEN_RVC_VOICE_DIR", dir);
        }
    }
    let device = parse_vc_device(device.as_deref())?;
    let backend = blazen_llm::RvcBackend::with_device(device);
    let adapter = LocalVcAdapter {
        inner: Arc::new(backend),
    };
    Ok(VcModel::from_arc(Arc::new(adapter)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vc_chunk_roundtrips_through_serde_json() {
        let chunk = VcChunk {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            is_final: false,
            latency_seconds: Some(0.125),
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        let back: VcChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.samples, chunk.samples);
        assert_eq!(back.is_final, chunk.is_final);
        assert_eq!(back.latency_seconds, chunk.latency_seconds);
    }

    #[test]
    fn target_voice_roundtrips_through_serde_json() {
        let voice = TargetVoice {
            id: "speaker-01".into(),
            label: Some("Alice".into()),
            sample_rate_hz: 40_000,
        };
        let json = serde_json::to_string(&voice).expect("serialize");
        let back: TargetVoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "speaker-01");
        assert_eq!(back.label.as_deref(), Some("Alice"));
        assert_eq!(back.sample_rate_hz, 40_000);
    }

    #[cfg(feature = "audio-vc-rvc")]
    #[test]
    fn new_rvc_model_constructs_on_cpu() {
        // Plumbing test: factory should not contact HF or touch the voice
        // dir during construction. Passing `None` for voice_dir leaves the
        // process environment alone; passing `"cpu"` exercises the device
        // parser.
        let model = new_rvc_model(None, Some("cpu".to_string()));
        assert!(model.is_ok(), "factory failed: {:?}", model.err());
    }
}
