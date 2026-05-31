//! `MusicModel` Python wrapper.
//!
//! Exposes a unified Python class around any
//! [`MusicBackend`](blazen_audio_music::MusicBackend) — MusicGen, AudioGen, or
//! Stable Audio — constructed via per-engine `@staticmethod` factories.
//! The factories are individually feature-gated so a build with only one
//! backend feature compiles in only that constructor.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use blazen_audio_music::MusicBackend;

use crate::error::music_error_to_pyerr;
use crate::music::chunk::PyMusicChunk;
use crate::music::stream::{
    LazyMusicStreamState, MusicStreamKind, PendingMusicStream, PyMusicStream,
};

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-audiogen",
    feature = "audio-music-stable-audio",
))]
use crate::providers::options::PyDevice;

/// Opaque handle around a `dyn MusicBackend`.
///
/// Constructed via one of the feature-gated `@staticmethod` factories
/// (`MusicModel.musicgen(...)`, `MusicModel.audiogen(...)`,
/// `MusicModel.stable_audio(...)`) — there is no public `__init__`.
///
/// The handle is cheap to clone (internally `Arc<dyn MusicBackend>`); the
/// underlying weights are lazily downloaded on the first `generate_*`
/// call.
#[gen_stub_pyclass]
#[pyclass(name = "MusicModel", from_py_object)]
#[derive(Clone)]
pub struct PyMusicModel {
    pub(crate) inner: Arc<dyn MusicBackend>,
    pub(crate) id_str: String,
    pub(crate) sample_rate: u32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMusicModel {
    // -----------------------------------------------------------------
    // MusicGen
    // -----------------------------------------------------------------

    /// Construct a MusicGen backend (`facebook/musicgen-{small,medium,large}`).
    ///
    /// Args:
    ///     variant: One of `"small"` (default), `"medium"`, `"large"`.
    ///     device: Optional [`Device`] override (default: auto-detect
    ///         CUDA -> Metal -> CPU).
    ///     cache_dir: Optional Hugging Face cache directory override.
    ///     max_duration_seconds: Per-backend hard cap on a single call
    ///         (default 30.0; absolute upper bound is 60.0 regardless).
    ///
    /// Output sample rate: 32 kHz mono.
    #[cfg(feature = "audio-music-musicgen")]
    #[staticmethod]
    #[pyo3(signature = (*, variant="small", device=None, cache_dir=None, max_duration_seconds=30.0))]
    fn musicgen(
        variant: &str,
        device: Option<PyRef<'_, PyDevice>>,
        cache_dir: Option<PathBuf>,
        max_duration_seconds: f32,
    ) -> PyResult<Self> {
        use blazen_audio_music::backends::musicgen::{MusicgenBackend, MusicgenConfig};

        let variant = parse_musicgen_variant(variant)?;
        let sample_rate = variant.sample_rate();
        let _ = device; // PyDevice -> candle::Device conversion is feature-gated and
        // currently only wired through the LLM stack; the underlying
        // MusicgenConfig.device default (`None` -> auto-detect) is the
        // right behaviour here. Explicit candle Device routing for the
        // music backends will land alongside the same wiring for the
        // other capability surfaces.
        let cfg = MusicgenConfig {
            variant,
            device: None,
            cache_dir,
            max_duration_seconds,
        };
        let backend = MusicgenBackend::new(cfg);
        let id_str = MusicgenIdLabel::for_variant(variant).to_string();
        Ok(Self {
            inner: Arc::new(backend),
            id_str,
            sample_rate,
        })
    }

    // -----------------------------------------------------------------
    // AudioGen
    // -----------------------------------------------------------------

    /// Construct an AudioGen backend (default `facebook/audiogen-medium`).
    ///
    /// Args:
    ///     repo_id: Hugging Face repo (default `"facebook/audiogen-medium"`).
    ///     revision: Optional pinned revision (commit SHA or tag).
    ///     device: Optional [`Device`] override (default: auto-detect
    ///         CUDA -> Metal -> CPU).
    ///     cache_dir: Optional Hugging Face cache directory override.
    ///     max_duration_seconds: Per-backend hard cap on a single call
    ///         (default 30.0).
    ///
    /// Output sample rate: 16 kHz mono.
    #[cfg(feature = "audio-music-audiogen")]
    #[staticmethod]
    #[pyo3(signature = (*, repo_id=None, revision=None, device=None, cache_dir=None, max_duration_seconds=30.0))]
    fn audiogen(
        repo_id: Option<String>,
        revision: Option<String>,
        device: Option<PyRef<'_, PyDevice>>,
        cache_dir: Option<PathBuf>,
        max_duration_seconds: f32,
    ) -> Self {
        use blazen_audio_music::backends::audiogen::{
            AUDIOGEN_SAMPLE_RATE, AudioGenBackend, AudioGenConfig,
        };

        let _ = device; // see comment on PyMusicModel::musicgen above.
        let repo_id = repo_id.unwrap_or_else(|| "facebook/audiogen-medium".to_string());
        let id_str = format!("audiogen:{repo_id}");
        let cfg = AudioGenConfig {
            repo_id,
            revision,
            device: None,
            cache_dir,
            max_duration_seconds,
        };
        let backend = AudioGenBackend::new(cfg);
        Self {
            inner: Arc::new(backend),
            id_str,
            sample_rate: AUDIOGEN_SAMPLE_RATE,
        }
    }

    // -----------------------------------------------------------------
    // Stable Audio
    // -----------------------------------------------------------------

    /// Construct a Stable Audio Open backend (small or 1.0 variant).
    ///
    /// Args:
    ///     variant: One of `"small"` (default) or `"open-1.0"`.
    ///     tokenizer_path: Path to a local `tokenizer.json` for the T5
    ///         conditioner (Stable Audio Open does not ship one inside
    ///         the model repo).
    ///     device: Optional [`Device`] override (default: auto-detect).
    ///     dtype: Inference dtype — `"f32"` (default) or `"f16"`.
    ///
    /// Output sample rate: 44.1 kHz stereo.
    #[cfg(feature = "audio-music-stable-audio")]
    #[staticmethod]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, MusicModel]", imports = ("typing",)))]
    #[pyo3(signature = (*, variant="small", tokenizer_path, device=None, dtype="f32"))]
    fn stable_audio<'py>(
        py: Python<'py>,
        variant: &str,
        tokenizer_path: PathBuf,
        device: Option<PyRef<'_, PyDevice>>,
        dtype: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        use blazen_audio_music::backends::stable_audio::{
            StableAudioBackend, StableAudioConfig, StableAudioVariant,
        };

        let variant = parse_stable_audio_variant(variant)?;
        let dtype = parse_stable_audio_dtype(dtype)?;
        let _ = device; // see comment on PyMusicModel::musicgen above.
        let hf_repo = variant.hf_repo().to_string();
        let id_str = match variant {
            StableAudioVariant::Small => "stable-audio:small".to_string(),
            StableAudioVariant::Open1_0 => "stable-audio:open-1.0".to_string(),
        };
        let cfg = StableAudioConfig {
            hf_repo,
            variant,
            tokenizer_path,
            local_weights_path: None,
            device: candle_core::Device::Cpu,
            dtype,
        };
        future_into_py(py, async move {
            let backend = StableAudioBackend::load(cfg)
                .await
                .map_err(music_error_to_pyerr)?;
            Ok(Self {
                inner: Arc::new(backend),
                id_str,
                sample_rate: 44_100,
            })
        })
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Stable backend identifier (e.g. `"musicgen-small"`,
    /// `"audiogen:facebook/audiogen-medium"`, `"stable-audio:small"`).
    #[getter]
    fn id(&self) -> &str {
        &self.id_str
    }

    /// Output sample rate in hertz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Return a type-erased :class:`MusicBackendHandle` wrapping this model's
    /// backend. The handle shares the same underlying ``Arc<dyn MusicBackend>``
    /// and forwards ``generate_music`` / ``generate_sfx``.
    #[gen_stub(override_return_type(type_repr = "MusicBackendHandle"))]
    fn backend_handle(&self) -> crate::audio_backends::PyMusicBackendHandle {
        crate::audio_backends::PyMusicBackendHandle {
            inner: self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MusicModel(id={:?}, sample_rate={})",
            self.id_str, self.sample_rate,
        )
    }

    // -----------------------------------------------------------------
    // Non-streaming generation
    // -----------------------------------------------------------------

    /// Generate `duration_seconds` of music conditioned on `prompt`.
    ///
    /// Returns a coroutine that resolves to a final [`MusicChunk`] with
    /// `is_final=True`, `sample_rate=self.sample_rate`, and the full f32
    /// PCM sample buffer.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, MusicChunk]", imports = ("typing",)))]
    fn generate_music<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        duration_seconds: f32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        let sample_rate = self.sample_rate;
        future_into_py(py, async move {
            let audio = backend
                .generate_music(&prompt, duration_seconds)
                .await
                .map_err(music_error_to_pyerr)?;
            let samples = wav_or_pcm_to_f32(&audio)?;
            Ok(PyMusicChunk::from_samples_final(samples, sample_rate))
        })
    }

    /// Generate `duration_seconds` of sound-effect audio conditioned on
    /// `prompt`.
    ///
    /// Returns a coroutine that resolves to a final [`MusicChunk`] with
    /// `is_final=True`.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, MusicChunk]", imports = ("typing",)))]
    fn generate_sfx<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        duration_seconds: f32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        let sample_rate = self.sample_rate;
        future_into_py(py, async move {
            let audio = backend
                .generate_sfx(&prompt, duration_seconds)
                .await
                .map_err(music_error_to_pyerr)?;
            let samples = wav_or_pcm_to_f32(&audio)?;
            Ok(PyMusicChunk::from_samples_final(samples, sample_rate))
        })
    }

    // -----------------------------------------------------------------
    // Streaming generation
    // -----------------------------------------------------------------

    /// Stream music chunks for low-latency progressive playback.
    ///
    /// Returns a [`MusicStream`] async-iterator that yields
    /// [`MusicChunk`] instances (each one a slice of f32 PCM samples)
    /// as the backend produces them. Concatenating all chunks in order
    /// reconstructs the equivalent of a single `generate_music` call.
    fn stream_generate_music<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        duration_seconds: f32,
    ) -> PyResult<Bound<'py, PyMusicStream>> {
        let stream = PyMusicStream {
            state: Arc::new(Mutex::new(LazyMusicStreamState::NotStarted(Box::new(
                PendingMusicStream {
                    backend: self.inner.clone(),
                    prompt: Some(prompt),
                    duration_seconds,
                    kind: MusicStreamKind::Music,
                },
            )))),
        };
        Bound::new(py, stream)
    }

    /// Stream SFX chunks for low-latency progressive playback.
    fn stream_generate_sfx<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        duration_seconds: f32,
    ) -> PyResult<Bound<'py, PyMusicStream>> {
        let stream = PyMusicStream {
            state: Arc::new(Mutex::new(LazyMusicStreamState::NotStarted(Box::new(
                PendingMusicStream {
                    backend: self.inner.clone(),
                    prompt: Some(prompt),
                    duration_seconds,
                    kind: MusicStreamKind::Sfx,
                },
            )))),
        };
        Bound::new(py, stream)
    }
}

// ---------------------------------------------------------------------------
// Variant parsing helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-music-musicgen")]
fn parse_musicgen_variant(
    s: &str,
) -> PyResult<blazen_audio_music::backends::musicgen::MusicgenVariant> {
    use blazen_audio_music::backends::musicgen::MusicgenVariant;
    match s.to_ascii_lowercase().as_str() {
        "small" => Ok(MusicgenVariant::Small),
        "medium" => Ok(MusicgenVariant::Medium),
        "large" => Ok(MusicgenVariant::Large),
        other => Err(PyValueError::new_err(format!(
            "unknown MusicGen variant `{other}` (expected one of: small, medium, large)"
        ))),
    }
}

#[cfg(feature = "audio-music-musicgen")]
struct MusicgenIdLabel;

#[cfg(feature = "audio-music-musicgen")]
impl MusicgenIdLabel {
    fn for_variant(v: blazen_audio_music::backends::musicgen::MusicgenVariant) -> &'static str {
        use blazen_audio_music::backends::musicgen::MusicgenVariant;
        match v {
            MusicgenVariant::Small => "musicgen-small",
            MusicgenVariant::Medium => "musicgen-medium",
            MusicgenVariant::Large => "musicgen-large",
        }
    }
}

#[cfg(feature = "audio-music-stable-audio")]
fn parse_stable_audio_variant(
    s: &str,
) -> PyResult<blazen_audio_music::backends::stable_audio::StableAudioVariant> {
    use blazen_audio_music::backends::stable_audio::StableAudioVariant;
    match s.to_ascii_lowercase().as_str() {
        "small" | "stable-audio-open-small" => Ok(StableAudioVariant::Small),
        "open-1.0" | "1.0" | "stable-audio-open-1.0" => Ok(StableAudioVariant::Open1_0),
        other => Err(PyValueError::new_err(format!(
            "unknown Stable Audio variant `{other}` \
             (expected one of: small, open-1.0)"
        ))),
    }
}

#[cfg(feature = "audio-music-stable-audio")]
fn parse_stable_audio_dtype(s: &str) -> PyResult<candle_core::DType> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "float32" => Ok(candle_core::DType::F32),
        "f16" | "float16" => Ok(candle_core::DType::F16),
        "bf16" | "bfloat16" => Ok(candle_core::DType::BF16),
        other => Err(PyValueError::new_err(format!(
            "unknown dtype `{other}` (expected one of: f32, f16, bf16)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Audio bytes -> f32 PCM conversion
// ---------------------------------------------------------------------------

/// Decode the bytes carried by a [`GeneratedAudio`] to interleaved f32 PCM
/// samples.
///
/// MusicGen / AudioGen / Stable Audio all encode their non-streaming
/// output as a 16-bit-PCM WAV container (see `blazen-audio-music`'s
/// `wav::pcm_to_wav`). This helper handles that case plus the trivial
/// `AudioFormat::Pcm` passthrough; non-Wav non-Pcm containers (mp3, flac,
/// opus) are surfaced as a clear `ValueError` rather than silently
/// dropping the audio.
fn wav_or_pcm_to_f32(audio: &blazen_audio::GeneratedAudio) -> PyResult<Vec<f32>> {
    use blazen_audio::{AudioFormat, SampleFormat};

    match audio.format {
        AudioFormat::Wav => decode_wav_to_f32(&audio.bytes),
        AudioFormat::Pcm => {
            // Raw PCM — interpret per the metadata `sample_format` would
            // imply, but the music backends always emit i16 in WAV (we
            // never hit the Pcm arm in practice today). Default to i16
            // for safety; a future backend that emits raw f32 PCM can
            // tag it via a thin wrapper.
            decode_i16_le_to_f32(&audio.bytes)
        }
        AudioFormat::Mp3 | AudioFormat::Flac | AudioFormat::Opus => {
            Err(PyValueError::new_err(format!(
                "MusicModel backends are expected to emit Wav or raw Pcm, got {:?}; \
                 this is a music-backend bug — please file an issue",
                audio.format
            )))
        }
        #[allow(unreachable_patterns)]
        _ => {
            let _ = SampleFormat::I16; // keep the import alive for future arms
            Err(PyValueError::new_err(format!(
                "unsupported audio format: {:?}",
                audio.format
            )))
        }
    }
}

/// Decode a minimal RIFF/WAVE container containing a single 16-bit-PCM
/// `data` chunk (the format emitted by `blazen-audio-music`'s
/// `pcm_to_wav`).
fn decode_wav_to_f32(bytes: &[u8]) -> PyResult<Vec<f32>> {
    // Header layout: RIFF(4) + size(4) + WAVE(4) + "fmt "(4) + 16(4) +
    // PCM(2) + channels(2) + sample_rate(4) + byte_rate(4) +
    // block_align(2) + bits_per_sample(2) + "data"(4) + data_size(4).
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(PyValueError::new_err(
            "GeneratedAudio.bytes is not a RIFF/WAVE container",
        ));
    }
    if &bytes[36..40] != b"data" {
        return Err(PyValueError::new_err(
            "GeneratedAudio.bytes WAV missing `data` chunk at expected offset \
             (non-canonical WAV layout)",
        ));
    }
    let data_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    let data_start = 44_usize;
    let data_end = data_start
        .checked_add(data_size)
        .ok_or_else(|| PyValueError::new_err("WAV data_size overflow"))?;
    if data_end > bytes.len() {
        return Err(PyValueError::new_err("WAV data_size exceeds buffer length"));
    }
    decode_i16_le_to_f32(&bytes[data_start..data_end])
}

/// Decode interleaved 16-bit little-endian PCM bytes to f32 samples in
/// `[-1.0, 1.0]`.
fn decode_i16_le_to_f32(bytes: &[u8]) -> PyResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(PyValueError::new_err(
            "i16 PCM payload length is not a multiple of 2",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f32::from(s) / f32::from(i16::MAX));
    }
    Ok(out)
}
