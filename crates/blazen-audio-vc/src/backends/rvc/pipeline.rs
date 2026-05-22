//! End-to-end RVC voice-conversion pipeline.
//!
//! [`RvcBackend`] implements
//! [`crate::VoiceConversionBackend`] by composing the four sibling
//! modules:
//!
//! 1. [`super::f0::RmvpeF0`]: extract a 10 ms-frame pitch contour from
//!    the source 16 kHz audio.
//! 2. [`super::content::ContentEncoder`]: encode the same audio into
//!    per-frame `HuBERT`/`ContentVec` content features.
//! 3. [`super::retrieval::FeatureIndex`]: blend the content features
//!    toward the target voice's training-time feature distribution.
//! 4. [`super::generator::NsfHifiGan`]: synthesise the target-voice
//!    waveform conditioned on the blended content + pitch + speaker id.
//!
//! Voice profiles live on disk under `$BLAZEN_RVC_VOICE_DIR/<voice_id>/`
//! and are loaded lazily on first conversion request. The heavy
//! components (the RMVPE F0 extractor and the `HuBERT` content encoder)
//! are shared across voices and lazily initialised through `OnceCell`.
//!
//! # Current limitations
//!
//! - The `HuBERT` content encoder loader is deferred upstream
//!   (`candle-transformers` lacks a stock `HuBERT` model). Until that
//!   lands, [`RvcBackend::convert_voice`] surfaces a clear
//!   [`VcError::ModelLoad`] error sourced from
//!   [`super::content::ContentEncoder::load`] and the pipeline does
//!   *not* panic.
//! - On-the-fly voice registration is not supported: RVC voice profiles
//!   are trained offline. [`RvcBackend::register_target_voice`] returns
//!   [`VcError::Unsupported`] explaining how to place a precomputed
//!   profile under `$BLAZEN_RVC_VOICE_DIR/<voice_id>/`.
//! - Streaming conversion ([`RvcBackend::stream_convert`]) buffers
//!   2-second windows (32 000 samples at the 16 kHz source rate) and
//!   runs the full convert pipeline per window. Lower-latency
//!   real-time conversion with overlap-add crossfading is a follow-up
//!   enhancement.

#![cfg(feature = "rvc")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError};
use candle_core::{DType, Device, Tensor};
use futures_core::Stream;
use tokio::sync::{OnceCell, RwLock};

use crate::error::VcError;
use crate::traits::{TargetVoice, VoiceConversionBackend};

use super::content::{ContentEncoder, ContentError, RvcVersion, SAMPLE_RATE_HZ as HUBERT_SR};
use super::f0::{F0_MAX, F0_MIN, F0Error, PITCH_COARSE_BINS, RmvpeF0, pitch_to_coarse};
use super::generator::GeneratorError;
use super::retrieval::RetrievalError;
use super::weights::{
    META_FILENAME, RvcVoiceProfile, VoiceMeta, hf_download, load_voice_profile, voice_root_dir,
};

// ---------------------------------------------------------------------------
// Default retrieval / model-cache knobs
// ---------------------------------------------------------------------------

/// Default top-`k` neighbours per query frame.
pub const DEFAULT_TOP_K: usize = 8;

/// Default retrieval blend factor (RVC's `index_rate`).
pub const DEFAULT_RETRIEVAL_BLEND: f32 = 0.75;

/// HF repo holding the canonical RVC support models (RMVPE F0 ONNX
/// graph, `HuBERT` `ContentVec` checkpoint).
pub const RVC_SUPPORT_REPO: &str = "lj1995/VoiceConversionWebUI";

/// File name of the RMVPE ONNX graph in [`RVC_SUPPORT_REPO`].
pub const RMVPE_FILENAME: &str = "rmvpe.onnx";

/// File name of the v2 `HuBERT` content encoder in [`RVC_SUPPORT_REPO`].
pub const HUBERT_FILENAME: &str = "hubert_base.pt";

/// Stable backend identifier surfaced by [`AudioBackend::id`].
pub const BACKEND_ID: &str = "rvc";

/// Streaming buffer size in 16 kHz source samples. 32 000 samples =
/// 2 seconds of audio; this is the chunking granularity for
/// [`RvcBackend::stream_convert`]. Lower-latency real-time conversion
/// with overlap-add crossfading is a follow-up enhancement.
pub const STREAM_BUFFER_SAMPLES: usize = 32_000;

/// Bounded capacity for the mpsc channel that carries converted
/// chunks from the background streaming task back to the caller. A
/// small capacity provides natural backpressure so the producer can't
/// outrun the consumer when conversion is faster than consumption.
const STREAM_CHANNEL_CAPACITY: usize = 8;

// ---------------------------------------------------------------------------
// Error conversions
// ---------------------------------------------------------------------------

impl From<F0Error> for VcError {
    fn from(err: F0Error) -> Self {
        match err {
            F0Error::ModelLoad(msg) => Self::ModelLoad(format!("rmvpe: {msg}")),
            F0Error::Inference(msg) | F0Error::InvalidInput(msg) => {
                Self::Conversion(format!("rmvpe: {msg}"))
            }
        }
    }
}

impl From<ContentError> for VcError {
    fn from(err: ContentError) -> Self {
        match err {
            ContentError::ModelLoad(msg) => Self::ModelLoad(format!("hubert: {msg}")),
            ContentError::Inference(msg) => Self::Conversion(format!("hubert: {msg}")),
            ContentError::Io(e) => Self::Io(e),
            ContentError::Candle(e) => Self::Conversion(format!("hubert candle: {e}")),
        }
    }
}

impl From<RetrievalError> for VcError {
    fn from(err: RetrievalError) -> Self {
        match err {
            RetrievalError::Build(msg) => Self::Conversion(format!("retrieval: {msg}")),
            RetrievalError::Io(e) => Self::Io(e),
            RetrievalError::Serde(msg) => Self::ModelLoad(format!("retrieval serde: {msg}")),
            RetrievalError::Candle(e) => Self::Conversion(format!("retrieval candle: {e}")),
            RetrievalError::EmptyIndex => Self::Conversion("retrieval index is empty".into()),
        }
    }
}

impl From<GeneratorError> for VcError {
    fn from(err: GeneratorError) -> Self {
        match err {
            GeneratorError::ModelLoad(msg) => Self::ModelLoad(format!("nsf-hifigan: {msg}")),
            GeneratorError::Inference(msg) => Self::Conversion(format!("nsf-hifigan: {msg}")),
            GeneratorError::Io(e) => Self::Io(e),
            GeneratorError::Candle(e) => Self::Conversion(format!("nsf-hifigan candle: {e}")),
        }
    }
}

impl From<candle_core::Error> for VcError {
    fn from(err: candle_core::Error) -> Self {
        Self::Conversion(format!("candle: {err}"))
    }
}

// ---------------------------------------------------------------------------
// RvcBackend
// ---------------------------------------------------------------------------

/// Retrieval-based Voice Conversion backend.
///
/// Cheap to clone (all heavy state lives behind `Arc`s) and safe to
/// share across tokio tasks.
#[derive(Clone)]
pub struct RvcBackend {
    device: Device,
    rmvpe: Arc<OnceCell<Arc<RmvpeF0>>>,
    hubert: Arc<OnceCell<Arc<ContentEncoder>>>,
    voices: Arc<RwLock<HashMap<String, Arc<RvcVoiceProfile>>>>,
    top_k: usize,
    retrieval_blend: f32,
    rvc_version: RvcVersion,
}

impl std::fmt::Debug for RvcBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvcBackend")
            .field("top_k", &self.top_k)
            .field("retrieval_blend", &self.retrieval_blend)
            .field("rvc_version", &self.rvc_version)
            .finish_non_exhaustive()
    }
}

impl Default for RvcBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RvcBackend {
    /// Construct a CPU-backed RVC backend with default retrieval
    /// settings. No I/O is performed.
    #[must_use]
    pub fn new() -> Self {
        Self::with_device(Device::Cpu)
    }

    /// Construct an RVC backend pinned to the given candle device.
    #[must_use]
    pub fn with_device(device: Device) -> Self {
        Self {
            device,
            rmvpe: Arc::new(OnceCell::new()),
            hubert: Arc::new(OnceCell::new()),
            voices: Arc::new(RwLock::new(HashMap::new())),
            top_k: DEFAULT_TOP_K,
            retrieval_blend: DEFAULT_RETRIEVAL_BLEND,
            rvc_version: RvcVersion::V2,
        }
    }

    /// Override the top-`k` neighbour count for retrieval. Clamped into
    /// `[1, usize::MAX]` at query time.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Override the retrieval blend factor (`index_rate` upstream).
    /// Clamped into `[0, 1]`.
    #[must_use]
    pub fn with_retrieval_blend(mut self, blend: f32) -> Self {
        self.retrieval_blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Force a specific `ContentVec` family for the shared `HuBERT`
    /// encoder. Defaults to [`RvcVersion::V2`].
    #[must_use]
    pub const fn with_rvc_version(mut self, version: RvcVersion) -> Self {
        self.rvc_version = version;
        self
    }

    /// Lazily download + load the shared RMVPE pitch extractor.
    async fn ensure_rmvpe(&self) -> Result<Arc<RmvpeF0>, VcError> {
        let device = self.device.clone();
        let arc = self
            .rmvpe
            .get_or_try_init(|| async move {
                let path = hf_download(RVC_SUPPORT_REPO, RMVPE_FILENAME, None).await?;
                let model = tokio::task::spawn_blocking(move || RmvpeF0::load(&path, &device))
                    .await
                    .map_err(|e| VcError::ModelLoad(format!("rmvpe spawn_blocking: {e}")))??;
                Ok::<Arc<RmvpeF0>, VcError>(Arc::new(model))
            })
            .await?;
        Ok(arc.clone())
    }

    /// Lazily download + load the shared `HuBERT` content encoder.
    ///
    /// At the time of writing this returns
    /// [`VcError::ModelLoad`] with the upstream "pending" message --
    /// see [`super::content`].
    async fn ensure_hubert(&self) -> Result<Arc<ContentEncoder>, VcError> {
        let device = self.device.clone();
        let version = self.rvc_version;
        let arc = self
            .hubert
            .get_or_try_init(|| async move {
                let path = hf_download(RVC_SUPPORT_REPO, HUBERT_FILENAME, None).await?;
                let model = tokio::task::spawn_blocking(move || {
                    ContentEncoder::load(&path, &device, version)
                })
                .await
                .map_err(|e| VcError::ModelLoad(format!("hubert spawn_blocking: {e}")))??;
                Ok::<Arc<ContentEncoder>, VcError>(Arc::new(model))
            })
            .await?;
        Ok(arc.clone())
    }

    /// Lazily load (and cache) a target-voice profile.
    async fn ensure_voice(&self, voice_id: &str) -> Result<Arc<RvcVoiceProfile>, VcError> {
        if let Some(cached) = self.voices.read().await.get(voice_id).cloned() {
            return Ok(cached);
        }
        let profile = load_voice_profile(voice_id, &self.device).await?;
        let arc = Arc::new(profile);
        let mut guard = self.voices.write().await;
        // Another caller may have raced us; honour the first insert.
        let entry = guard
            .entry(voice_id.to_owned())
            .or_insert_with(|| arc.clone());
        Ok(entry.clone())
    }

    /// Read + decode a 16-bit PCM WAV file into mono 16 kHz f32 samples.
    fn read_wav_to_mono_16khz(raw: &[u8]) -> Result<Vec<f32>, VcError> {
        if raw.len() < 44 {
            return Err(VcError::Conversion(
                "input WAV too small (need at least 44-byte header)".into(),
            ));
        }
        if &raw[..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
            return Err(VcError::Conversion("input is not a RIFF/WAVE file".into()));
        }

        // Walk the chunks past `WAVE` to find `fmt ` + `data` instead of
        // assuming a fixed 44-byte header (real-world WAVs can have
        // intervening LIST / bext chunks that throw off a fixed offset).
        let mut cursor = 12_usize;
        let mut channels: Option<u16> = None;
        let mut sample_rate: Option<u32> = None;
        let mut bits_per_sample: Option<u16> = None;
        let mut audio_format: Option<u16> = None;
        let mut data: Option<&[u8]> = None;
        while cursor + 8 <= raw.len() {
            let id = &raw[cursor..cursor + 4];
            let size = u32::from_le_bytes([
                raw[cursor + 4],
                raw[cursor + 5],
                raw[cursor + 6],
                raw[cursor + 7],
            ]) as usize;
            let body_start = cursor + 8;
            let body_end = body_start.saturating_add(size).min(raw.len());
            let body = &raw[body_start..body_end];
            match id {
                b"fmt " => {
                    if body.len() < 16 {
                        return Err(VcError::Conversion(
                            "WAV fmt chunk shorter than 16 bytes".into(),
                        ));
                    }
                    audio_format = Some(u16::from_le_bytes([body[0], body[1]]));
                    channels = Some(u16::from_le_bytes([body[2], body[3]]));
                    sample_rate = Some(u32::from_le_bytes([body[4], body[5], body[6], body[7]]));
                    bits_per_sample = Some(u16::from_le_bytes([body[14], body[15]]));
                }
                b"data" => {
                    data = Some(body);
                }
                _ => {}
            }
            // Chunks are 2-byte aligned per the RIFF spec.
            cursor = body_end + (body_end & 1);
            if data.is_some() && channels.is_some() {
                break;
            }
        }

        let channels =
            channels.ok_or_else(|| VcError::Conversion("WAV missing fmt chunk".into()))?;
        let sample_rate =
            sample_rate.ok_or_else(|| VcError::Conversion("WAV missing fmt chunk".into()))?;
        let bits_per_sample =
            bits_per_sample.ok_or_else(|| VcError::Conversion("WAV missing fmt chunk".into()))?;
        let data = data.ok_or_else(|| VcError::Conversion("WAV missing data chunk".into()))?;
        let audio_format =
            audio_format.ok_or_else(|| VcError::Conversion("WAV missing fmt chunk".into()))?;

        if audio_format != 1 {
            return Err(VcError::Conversion(format!(
                "WAV audio_format = {audio_format}; only PCM (1) is supported"
            )));
        }
        if bits_per_sample != 16 {
            return Err(VcError::Conversion(format!(
                "WAV bits_per_sample = {bits_per_sample}; only 16-bit PCM is supported"
            )));
        }
        if channels == 0 {
            return Err(VcError::Conversion("WAV has zero channels".into()));
        }

        // i16 -> f32 in [-1, 1]; mix down to mono on the fly.
        let bytes_per_sample = 2_usize;
        let frame_stride = bytes_per_sample * channels as usize;
        if !data.len().is_multiple_of(frame_stride) {
            return Err(VcError::Conversion(format!(
                "WAV data length {} not a multiple of frame stride {}",
                data.len(),
                frame_stride
            )));
        }
        let n_frames = data.len() / frame_stride;
        let mut mono = Vec::with_capacity(n_frames);
        let inv_norm = 1.0_f32 / f32::from(i16::MAX);
        let inv_channels = 1.0_f32 / f32::from(channels);
        for f in 0..n_frames {
            let start = f * frame_stride;
            let mut acc = 0.0_f32;
            for c in 0..channels as usize {
                let off = start + c * bytes_per_sample;
                let s = i16::from_le_bytes([data[off], data[off + 1]]);
                acc += f32::from(s) * inv_norm;
            }
            mono.push(acc * inv_channels);
        }

        // Resample to 16 kHz if needed via linear interpolation. The
        // upstream RVC reference uses librosa's polyphase resample but
        // a linear pass is good enough for the HuBERT content encoder
        // (which is robust to ~kHz of resampling artifact) and avoids
        // pulling in `rubato`.
        let target_sr = HUBERT_SR;
        if sample_rate == target_sr {
            return Ok(mono);
        }
        Ok(linear_resample(&mono, sample_rate, target_sr))
    }

    /// Run the F0 + content + retrieval + generator flow on a mono
    /// 16 kHz `f32` PCM buffer, returning the converted PCM at the
    /// voice's native sample rate.
    ///
    /// Shared by [`Self::convert_voice`] (which decodes a WAV first)
    /// and [`Self::stream_convert`] (which buffers raw samples from the
    /// input stream). The split keeps the inference logic
    /// deduplicated.
    async fn convert_buffer(
        &self,
        samples_16k: &[f32],
        target_voice_id: &str,
    ) -> Result<Vec<f32>, VcError> {
        // 1. Voice profile (cheap if cached; otherwise hits disk).
        let voice = self.ensure_voice(target_voice_id).await?;
        if samples_16k.is_empty() {
            return Err(VcError::Conversion("source audio is empty".into()));
        }

        // 2. Pitch extraction.
        let rmvpe = self.ensure_rmvpe().await?;
        let pitch_hz_vec = {
            let rmvpe_clone = rmvpe.clone();
            let samples = samples_16k.to_vec();
            tokio::task::spawn_blocking(move || rmvpe_clone.extract(&samples))
                .await
                .map_err(|e| VcError::Conversion(format!("rmvpe spawn_blocking: {e}")))??
        };
        let pitch_coarse_vec = pitch_to_coarse(&pitch_hz_vec, F0_MIN, F0_MAX, PITCH_COARSE_BINS);

        // 3. Content extraction (currently propagates the upstream
        //    "pending" error verbatim through the `ContentError -> VcError`
        //    conversion; callers see a clear `VcError::ModelLoad`).
        let hubert = self.ensure_hubert().await?;
        let content = hubert.encode(samples_16k)?;

        // 4. Retrieval blend.
        //    content is (1, n_frames, hidden_dim); FeatureIndex::retrieve
        //    takes and returns the same layout.
        let blended = voice
            .index
            .retrieve(&content, self.top_k, self.retrieval_blend)?;

        // 5. Transpose to channels-first for the generator.
        //    (1, n_frames, hidden) -> (1, hidden, n_frames).
        let blended_cf = blended.transpose(1, 2)?.contiguous()?;

        // Align frame counts: the content encoder operates at 50 Hz
        // (16 kHz / 320 hop) while RMVPE emits at 100 Hz (16 kHz /
        // 160 hop). Take the shorter of the two so every per-frame
        // tensor lines up.
        let (_, _, n_frames_content) = blended_cf.dims3()?;
        let n_frames_pitch = pitch_hz_vec.len();
        let n_frames = n_frames_content.min(n_frames_pitch);
        if n_frames == 0 {
            return Err(VcError::Conversion("no frames to synthesise".into()));
        }
        let content_aligned = blended_cf.narrow(2, 0, n_frames)?;
        let pitch_hz_aligned: Vec<f32> = pitch_hz_vec.into_iter().take(n_frames).collect();
        let pitch_coarse_aligned: Vec<i64> = pitch_coarse_vec
            .into_iter()
            .take(n_frames)
            .map(i64::from)
            .collect();

        // 6. Build pitch + speaker tensors on the configured device.
        let pitch_coarse_t = Tensor::from_vec(pitch_coarse_aligned, (1, n_frames), &self.device)?
            .to_dtype(DType::I64)?;
        let pitch_hz_t = Tensor::from_vec(pitch_hz_aligned, (1, n_frames), &self.device)?;
        let speaker_id_t = Tensor::from_vec(vec![i64::from(voice.speaker_id)], (1,), &self.device)?;

        // 7. Synthesise → PCM at the voice's native rate.
        Ok(voice.generator.synthesize(
            &content_aligned,
            &pitch_coarse_t,
            &pitch_hz_t,
            &speaker_id_t,
        )?)
    }

    /// Encode mono f32 PCM as a 16-bit little-endian WAV byte buffer.
    fn encode_wav_16bit_mono(samples: &[f32], sample_rate_hz: u32) -> Vec<u8> {
        let channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let data_size = (samples.len() * 2) as u32;
        let byte_rate = sample_rate_hz * u32::from(channels) * u32::from(bits_per_sample) / 8;
        let block_align = channels * bits_per_sample / 8;

        let mut out = Vec::with_capacity(44 + samples.len() * 2);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_size).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16_u32.to_le_bytes());
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate_hz.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits_per_sample.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i = (clamped * f32::from(i16::MAX)) as i16;
            out.extend_from_slice(&i.to_le_bytes());
        }
        out
    }
}

/// Linear-interpolation resample of mono `f32` audio. Good enough for
/// the `HuBERT` content encoder; not bit-exact against `librosa` but
/// well within the encoder's robustness envelope.
fn linear_resample(samples: &[f32], src_hz: u32, dst_hz: u32) -> Vec<f32> {
    if samples.is_empty() || src_hz == 0 || dst_hz == 0 || src_hz == dst_hz {
        return samples.to_vec();
    }
    let src_len = samples.len() as f64;
    let dst_len = (src_len * f64::from(dst_hz) / f64::from(src_hz)).round() as usize;
    if dst_len <= 1 {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(dst_len);
    let step = (src_len - 1.0) / (dst_len as f64 - 1.0);
    for i in 0..dst_len {
        let x = i as f64 * step;
        let lo = x.floor() as usize;
        let hi = (lo + 1).min(samples.len() - 1);
        let frac = (x - lo as f64) as f32;
        out.push(samples[lo] * (1.0 - frac) + samples[hi] * frac);
    }
    out
}

// ---------------------------------------------------------------------------
// AudioBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioBackend for RvcBackend {
    fn id(&self) -> &str {
        BACKEND_ID
    }

    fn provider_kind(&self) -> &'static str {
        // Voice conversion is its own capability; reuse the umbrella
        // "voice" kind so the manager / router can group it alongside
        // TTS voice management without colliding with the "tts" kind.
        "voice-conversion"
    }
}

// ---------------------------------------------------------------------------
// VoiceConversionBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl VoiceConversionBackend for RvcBackend {
    async fn convert_voice(
        &self,
        input_audio_path: &Path,
        target_voice_id: &str,
    ) -> Result<Vec<u8>, VcError> {
        // 1. Voice profile (cheap if cached; otherwise hits disk).
        let voice = self.ensure_voice(target_voice_id).await?;

        // 2. Decode + resample the source audio.
        let raw = tokio::fs::read(input_audio_path).await?;
        let samples_16k = Self::read_wav_to_mono_16khz(&raw)?;
        if samples_16k.is_empty() {
            return Err(VcError::Conversion("source audio is empty".into()));
        }

        // 3-8. Run the shared F0+content+retrieval+generator flow.
        let pcm = self.convert_buffer(&samples_16k, target_voice_id).await?;

        // 9. Encode as 16-bit mono WAV at the voice's native rate.
        Ok(Self::encode_wav_16bit_mono(&pcm, voice.sample_rate_hz))
    }

    async fn list_target_voices(&self) -> Result<Vec<TargetVoice>, VcError> {
        let root = voice_root_dir();
        if !root.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut entries = tokio::fs::read_dir(&root).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let meta_path = path.join(META_FILENAME);
            let meta = if meta_path.is_file() {
                let raw = tokio::fs::read_to_string(&meta_path).await?;
                VoiceMeta::parse(&raw).unwrap_or_default()
            } else {
                VoiceMeta::default()
            };
            out.push(TargetVoice {
                id: name.to_owned(),
                label: None,
                sample_rate_hz: meta.sample_rate_hz,
            });
        }
        Ok(out)
    }

    /// Registering RVC voices at runtime is intentionally unsupported.
    ///
    /// RVC voice profiles bundle three trained-offline artefacts
    /// (NSF-HiFi-GAN generator weights, the kNN retrieval index over
    /// training-time content features, and the speaker-embedding
    /// index) that cannot be synthesised from a short reference clip
    /// in real time -- the upstream pipeline requires a training run
    /// (1+ hour even on a single-speaker dataset). Place a precomputed
    /// profile under `$BLAZEN_RVC_VOICE_DIR/<voice_id>/` and the new
    /// voice will surface through [`Self::list_target_voices`] /
    /// [`Self::convert_voice`] on the next call.
    async fn register_target_voice(
        &self,
        _voice_id: &str,
        _reference_audio_path: &Path,
    ) -> Result<(), VcError> {
        Err(VcError::Unsupported(
            "on-the-fly RVC voice registration requires training; precomputed voice profiles \
             must be placed under $BLAZEN_RVC_VOICE_DIR/<voice_id>/ (containing model.pth, \
             index.bin, and optional meta.toml)"
                .into(),
        ))
    }

    /// Chunked streaming voice conversion.
    ///
    /// Buffers incoming `f32` source samples until at least
    /// [`STREAM_BUFFER_SAMPLES`] (2 seconds at the 16 kHz source rate)
    /// are available, then runs the full F0 + content + retrieval +
    /// generator pipeline (shared with [`Self::convert_voice`] through
    /// [`Self::convert_buffer`]) on the buffered window and yields the
    /// converted PCM at the voice's native sample rate. The cycle
    /// repeats until the input stream ends; any non-empty remainder is
    /// flushed as a final partial chunk.
    ///
    /// # Latency tradeoff
    ///
    /// The current implementation buffers 2-second windows, so end-to-
    /// end latency is bounded below by ~2 s plus per-window inference
    /// time. Lower-latency real-time conversion with overlap-add
    /// crossfading (so the generator's `ConvTranspose1d` boundary
    /// artefacts don't surface as audible clicks at chunk seams) is a
    /// follow-up enhancement.
    ///
    /// # Backpressure
    ///
    /// The background task forwards converted chunks through a bounded
    /// `tokio::sync::mpsc::channel` of capacity
    /// [`STREAM_CHANNEL_CAPACITY`]. If the consumer falls behind the
    /// task suspends on `send().await` until capacity frees up, giving
    /// natural backpressure all the way to the input stream.
    async fn stream_convert(
        &self,
        mut audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        target_voice_id: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<f32>, VcError>> + Send>>, VcError> {
        use futures_util::StreamExt;

        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<Vec<f32>, VcError>>(STREAM_CHANNEL_CAPACITY);

        let backend = self.clone();
        let voice_id = target_voice_id.to_owned();
        tokio::spawn(async move {
            let mut buffer: Vec<f32> = Vec::with_capacity(STREAM_BUFFER_SAMPLES);

            while let Some(chunk) = audio.next().await {
                buffer.extend_from_slice(&chunk);
                while buffer.len() >= STREAM_BUFFER_SAMPLES {
                    let window: Vec<f32> = buffer.drain(..STREAM_BUFFER_SAMPLES).collect();
                    let result = backend.convert_buffer(&window, &voice_id).await;
                    if tx.send(result).await.is_err() {
                        // Consumer went away; stop pulling input.
                        return;
                    }
                }
            }

            // Input stream ended; flush any partial remainder as a
            // final chunk so callers see the tail of the conversion.
            if !buffer.is_empty() {
                let result = backend.convert_buffer(&buffer, &voice_id).await;
                let _ = tx.send(result).await;
            }
        });

        // Adapt the mpsc receiver into a `Stream`. We can't use
        // `tokio_stream::wrappers::ReceiverStream` here because
        // `tokio-stream` isn't a direct dep of this crate (gating it on
        // the `rvc` feature would require a Cargo.toml edit that's
        // out of scope for this change). `stream::unfold` does the job
        // with no extra deps.
        let stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// `From<VcError>` shim for callers that want to upcast into the
// crate's `AudioError`. Already provided by `error.rs`; we use it for
// the tests below so we don't need to manually project errors.
// ---------------------------------------------------------------------------

#[doc(hidden)]
fn _audio_error_upcast(err: VcError) -> AudioError {
    err.into()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(unsafe_code, reason = "tests mutate process env in a serialised block")]
mod tests {
    use super::*;

    /// Serialise env-mutating tests on a process-wide async mutex so
    /// they don't race with each other under cargo's parallel test
    /// runner. A tokio mutex (rather than a std one) is required
    /// because tests below hold the guard across `.await` points.
    fn env_lock() -> &'static tokio::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
    }

    async fn set_voice_dir(p: &Path) -> tokio::sync::MutexGuard<'static, ()> {
        let guard = env_lock().lock().await;
        // SAFETY: env_lock() serialises the whole "set, observe,
        // restore" cycle so no other thread in this test binary can
        // be mid-`std::env::var` at the same moment.
        unsafe {
            std::env::set_var(super::super::weights::VOICE_DIR_ENV, p);
        }
        guard
    }

    fn clear_voice_dir() {
        // SAFETY: callers hold the env_lock() guard returned by
        // set_voice_dir(); see that fn's safety note.
        unsafe {
            std::env::remove_var(super::super::weights::VOICE_DIR_ENV);
        }
    }

    #[test]
    fn rvc_backend_new_smoke() {
        let backend = RvcBackend::new();
        assert_eq!(backend.id(), BACKEND_ID);
        assert_eq!(backend.provider_kind(), "voice-conversion");
        assert_eq!(backend.top_k, DEFAULT_TOP_K);
        assert!((backend.retrieval_blend - DEFAULT_RETRIEVAL_BLEND).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_setters_clamp() {
        let backend = RvcBackend::new().with_top_k(0).with_retrieval_blend(2.0);
        assert_eq!(backend.top_k, 1);
        assert!((backend.retrieval_blend - 1.0).abs() < f32::EPSILON);

        let backend = RvcBackend::new().with_retrieval_blend(-0.5);
        assert!((backend.retrieval_blend - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn list_voices_empty_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _g = set_voice_dir(tmp.path()).await;
        let backend = RvcBackend::new();
        let voices = backend.list_target_voices().await.expect("list");
        assert!(voices.is_empty(), "expected empty, got {voices:?}");
        clear_voice_dir();
    }

    #[tokio::test]
    async fn register_voice_returns_unsupported_with_clear_message() {
        let backend = RvcBackend::new();
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let err = backend
            .register_target_voice("new-voice", tmp.path())
            .await
            .expect_err("should fail");
        match err {
            VcError::Unsupported(msg) => {
                assert!(
                    msg.contains("training") || msg.contains("trained"),
                    "message should mention trained voice profiles, got: {msg}"
                );
                assert!(
                    msg.contains("BLAZEN_RVC_VOICE_DIR"),
                    "message should mention the voice-dir env var, got: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_convert_yields_chunks_for_buffered_input() {
        use futures_util::StreamExt;
        use futures_util::stream;

        // Tempdir keeps `ensure_voice` from finding any voice on disk
        // so we get a clean `VoiceNotFound` error per buffered window.
        // That's enough to confirm the streaming wiring is live --
        // we're asserting the task runs and the channel forwards
        // results, not that inference succeeds (it can't: HuBERT load
        // is pending upstream, and no voice profile exists either).
        let tmp = tempfile::tempdir().expect("tempdir");
        let _g = set_voice_dir(tmp.path()).await;

        let backend = RvcBackend::new();
        // Three 16 000-sample chunks → 48 000 samples total. With a
        // 32 000-sample buffer threshold the task should produce at
        // least one full-window chunk plus a flushed remainder.
        let input: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>> = Box::pin(stream::iter(vec![
            vec![0.0_f32; 16_000],
            vec![0.0_f32; 16_000],
            vec![0.0_f32; 16_000],
        ]));

        let out_stream = backend
            .stream_convert(input, "missing-voice")
            .await
            .expect("stream_convert returns Ok");
        let chunks: Vec<Result<Vec<f32>, VcError>> = out_stream.collect().await;

        assert!(
            !chunks.is_empty(),
            "stream_convert must produce at least one chunk attempt; got 0"
        );
        // Since the voice is missing every window converts into an
        // error -- assert at least one of them is the expected
        // VoiceNotFound (proves convert_buffer ran end-to-end).
        let saw_voice_not_found = chunks
            .iter()
            .any(|r| matches!(r, Err(VcError::VoiceNotFound(_))));
        assert!(
            saw_voice_not_found,
            "expected at least one VoiceNotFound chunk, got: {chunks:?}"
        );

        clear_voice_dir();
    }

    #[tokio::test]
    async fn convert_voice_voice_not_found() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _g = set_voice_dir(tmp.path()).await;
        let backend = RvcBackend::new();
        let audio = tempfile::NamedTempFile::new().expect("audio tmp");
        let err = backend
            .convert_voice(audio.path(), "missing")
            .await
            .expect_err("should fail");
        match err {
            VcError::VoiceNotFound(msg) => {
                assert!(msg.contains("missing"), "msg: {msg}");
            }
            other => panic!("expected VoiceNotFound, got {other:?}"),
        }
        clear_voice_dir();
    }

    #[test]
    fn linear_resample_passthrough_when_rates_match() {
        let s = vec![0.1_f32, 0.2, 0.3];
        let out = linear_resample(&s, 16_000, 16_000);
        assert_eq!(out, s);
    }

    #[test]
    fn linear_resample_changes_length_proportionally() {
        let s = vec![0.0_f32; 320]; // 20 ms at 16 kHz
        let out = linear_resample(&s, 16_000, 8_000);
        // 160 samples at 8 kHz; ±1 for rounding.
        assert!(out.len() == 160 || out.len() == 161, "len = {}", out.len());
    }

    #[test]
    fn read_wav_rejects_short_input() {
        let err = RvcBackend::read_wav_to_mono_16khz(&[0; 8]).expect_err("too short");
        assert!(matches!(err, VcError::Conversion(_)));
    }

    #[test]
    fn read_wav_rejects_non_riff() {
        let mut bad = vec![0u8; 64];
        bad[0..4].copy_from_slice(b"NOPE");
        let err = RvcBackend::read_wav_to_mono_16khz(&bad).expect_err("bad magic");
        assert!(matches!(err, VcError::Conversion(_)));
    }

    #[test]
    fn read_wav_decodes_minimal_mono_16khz() {
        // Build a 4-sample mono 16-bit PCM WAV at 16 kHz, decode it,
        // and assert the values round-trip.
        let samples_i16: [i16; 4] = [0, 8000, -8000, i16::MAX];
        let wav = RvcBackend::encode_wav_16bit_mono(
            &samples_i16
                .iter()
                .map(|&i| f32::from(i) / f32::from(i16::MAX))
                .collect::<Vec<_>>(),
            16_000,
        );
        let mono = RvcBackend::read_wav_to_mono_16khz(&wav).expect("decode");
        assert_eq!(mono.len(), 4);
        for (got, want) in mono.iter().zip(samples_i16.iter()) {
            let want_f = f32::from(*want) / f32::from(i16::MAX);
            assert!((got - want_f).abs() < 1e-4, "got {got}, want {want_f}");
        }
    }
}
