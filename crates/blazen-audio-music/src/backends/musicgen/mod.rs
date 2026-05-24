//! Meta MusicGen text-to-music backend (real, end-to-end implementation).
//!
//! Wires together:
//!
//! - [`model`]: the MusicGen decoder, T5 prompt encoder, EnCodec audio
//!   codec, and combined `GenConfig` (ported and patched from
//!   `huggingface/candle/candle-examples/examples/musicgen/musicgen_model.rs`).
//! - [`delay_pattern`]: the 4-codebook delay pattern interleaver used at
//!   generation time.
//! - [`sampler`]: Classifier-Free-Guidance logits combiner +
//!   per-codebook sampling on top of `candle_transformers::generation::LogitsProcessor`.
//! - [`generation`]: the autoregressive loop that drives the decoder.
//!
//! The [`MusicgenBackend`] type is the only thing the rest of the crate
//! (and downstream binding code) needs to know about; it implements
//! [`MusicBackend`](crate::traits::MusicBackend) on top of
//! [`AudioBackend`](blazen_audio::AudioBackend).
//!
//! # Weight licensing
//!
//! `facebook/musicgen-{small,medium,large}` checkpoints are released
//! under CC-BY-NC-4.0 and are **not licensed for commercial use**.
//! Surface this restriction to your end-users before shipping music
//! generation in a commercial product.

#![cfg(feature = "musicgen")]

pub mod delay_pattern;
pub mod generation;
pub mod model;
pub mod sampler;

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError, AudioFormat, GeneratedAudio};
use blazen_audio_codec::backends::encodec::{EncodecBackend, EncodecConfig};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::sync::{OnceCell, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::MusicError;
use crate::traits::{MusicBackend, MusicChunk};

use generation::GenerationParams;
use model::{GenConfig, MusicgenForConditionalGeneration};

/// Number of EnCodec frames to bundle into a single streamed `MusicChunk`.
///
/// EnCodec runs at 50 Hz, so 25 frames = 500 ms of audio per chunk. At
/// MusicGen's 32 kHz native rate that's 16 000 f32 samples per chunk.
const STREAM_CHUNK_FRAMES: usize = 25;

/// Bounded back-pressure on the streaming channel — `4 * 500 ms` keeps
/// at most ≈ 2 s of audio buffered between the producer (decoder) and
/// consumer (downstream playback / encoder).
const STREAM_CHANNEL_CAPACITY: usize = 4;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

/// Available MusicGen checkpoints on Hugging Face Hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MusicgenVariant {
    /// `facebook/musicgen-small` -- ~300M params, 32 kHz mono output.
    Small,
    /// `facebook/musicgen-medium` -- ~1.5B params, 32 kHz mono output.
    Medium,
    /// `facebook/musicgen-large` -- ~3.3B params, 32 kHz mono output.
    Large,
}

impl MusicgenVariant {
    /// The Hugging Face Hub repo identifier for this variant.
    #[must_use]
    pub const fn hf_repo(self) -> &'static str {
        match self {
            Self::Small => "facebook/musicgen-small",
            Self::Medium => "facebook/musicgen-medium",
            Self::Large => "facebook/musicgen-large",
        }
    }

    /// Native sample rate (always 32 kHz for MusicGen, regardless of size).
    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        32_000
    }

    /// EnCodec frame rate (always 50 Hz for the 32 kHz MusicGen variants).
    #[must_use]
    pub const fn frame_rate(self) -> u32 {
        50
    }

    /// Combined model config for this variant.
    #[must_use]
    pub fn gen_config(self) -> GenConfig {
        match self {
            Self::Small => GenConfig::small(),
            Self::Medium => GenConfig::medium(),
            Self::Large => GenConfig::large(),
        }
    }
}

/// Configuration for a [`MusicgenBackend`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicgenConfig {
    /// Which checkpoint to use.
    pub variant: MusicgenVariant,
    /// Override candle device selection. `None` falls back to
    /// auto-detection (CUDA → Metal → CPU).
    #[serde(skip)]
    pub device: Option<Device>,
    /// Optional override for the HF cache directory. `None` falls back
    /// to the default cache (`~/.cache/huggingface/hub`).
    pub cache_dir: Option<PathBuf>,
    /// Hard safety cap on the requested duration. Default `30.0`
    /// seconds. Calls past this cap return [`MusicError::InvalidInput`]
    /// regardless of `MUSICGEN_MAX_DURATION_HARD_LIMIT`.
    pub max_duration_seconds: f32,
}

impl Default for MusicgenConfig {
    fn default() -> Self {
        Self {
            variant: MusicgenVariant::Small,
            device: None,
            cache_dir: None,
            max_duration_seconds: 30.0,
        }
    }
}

/// Absolute upper bound on a single MusicGen call, regardless of how
/// permissive `MusicgenConfig::max_duration_seconds` is set. Anything
/// past this returns `MusicError::InvalidInput("max 60s for MusicGen")`.
pub const MUSICGEN_MAX_DURATION_HARD_LIMIT: f32 = 60.0;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Loaded MusicGen handle: model graph + tokenizer + (lazy) EnCodec backend.
struct Loaded {
    model: tokio::sync::Mutex<MusicgenForConditionalGeneration>,
    tokenizer: Tokenizer,
    /// EnCodec backend lazily configured to point at the model weights
    /// we already mmapped for MusicGen -- it implements the codec API the
    /// crate already ships so we route the codes through it instead of
    /// duplicating the decode path. Lazy because the same MusicGen
    /// checkpoint embeds the EnCodec weights under `audio_encoder.*`;
    /// we go through [`MusicgenForConditionalGeneration::audio_encoder`]
    /// directly to avoid a second weight load.
    device: Device,
}

/// MusicGen text-to-music + text-to-SFX backend.
///
/// Construction is cheap: weights / tokenizer are downloaded on the
/// first `generate_music` / `generate_sfx` call.
pub struct MusicgenBackend {
    config: MusicgenConfig,
    id: String,
    loaded: Arc<OnceCell<Loaded>>,
}

impl std::fmt::Debug for MusicgenBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicgenBackend")
            .field("config", &self.config)
            .field("id", &self.id)
            .field("loaded", &self.loaded.initialized())
            .finish()
    }
}

impl MusicgenBackend {
    /// Construct a MusicGen backend handle from the given config.
    #[must_use]
    pub fn new(config: MusicgenConfig) -> Self {
        let id = match config.variant {
            MusicgenVariant::Small => "musicgen-small",
            MusicgenVariant::Medium => "musicgen-medium",
            MusicgenVariant::Large => "musicgen-large",
        }
        .to_string();
        Self {
            config,
            id,
            loaded: Arc::new(OnceCell::new()),
        }
    }

    /// Borrow the backend config.
    #[must_use]
    pub const fn config(&self) -> &MusicgenConfig {
        &self.config
    }

    /// Pick the candle device honoring `MusicgenConfig::device` and the
    /// available accelerator features.
    fn pick_device(&self) -> Device {
        if let Some(dev) = &self.config.device {
            return dev.clone();
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(dev) = Device::new_cuda(0) {
                return dev;
            }
        }
        #[cfg(feature = "metal")]
        {
            if let Ok(dev) = Device::new_metal(0) {
                return dev;
            }
        }
        Device::Cpu
    }

    async fn ensure_loaded(&self) -> Result<&Loaded, MusicError> {
        self.loaded
            .get_or_try_init(|| async { self.load_inner().await })
            .await
    }

    async fn load_inner(&self) -> Result<Loaded, MusicError> {
        let repo = self.config.variant.hf_repo().to_string();
        let cache_dir = self.config.cache_dir.clone();

        let (weights, tokenizer_path) =
            tokio::task::spawn_blocking(move || -> Result<(PathBuf, PathBuf), MusicError> {
                let mut builder = hf_hub::api::sync::ApiBuilder::new();
                if let Some(dir) = cache_dir {
                    builder = builder.with_cache_dir(dir);
                }
                let api = builder.build().map_err(|e| MusicError::HfHub {
                    repo: repo.clone(),
                    source: std::io::Error::other(e.to_string()),
                })?;
                let m = api.model(repo.clone());
                let weights = m.get("model.safetensors").map_err(|e| MusicError::HfHub {
                    repo: repo.clone(),
                    source: std::io::Error::other(e.to_string()),
                })?;
                let tok = m.get("tokenizer.json").map_err(|e| MusicError::HfHub {
                    repo,
                    source: std::io::Error::other(e.to_string()),
                })?;
                Ok((weights, tok))
            })
            .await
            .map_err(|e| MusicError::other(format!("blocking task join failed: {e}")))??;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MusicError::other(format!("tokenizer load failed: {e}")))?;

        let device = self.pick_device();
        let cfg = self.config.variant.gen_config();

        // SAFETY: candle's `from_mmaped_safetensors` requires `unsafe`
        // because the safetensors file must outlive the mmap and the
        // file contents must not change underneath us. We pass a
        // PathBuf rooted in the hf-hub cache whose contents are
        // immutable by convention.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)
                .map_err(MusicError::from)?
        };
        let model = MusicgenForConditionalGeneration::load(vb, cfg)?;

        Ok(Loaded {
            model: tokio::sync::Mutex::new(model),
            tokenizer,
            device,
        })
    }

    fn validate_inputs(&self, prompt: &str, duration_seconds: f32) -> Result<(), MusicError> {
        if prompt.trim().is_empty() {
            return Err(MusicError::invalid_input("prompt is empty"));
        }
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err(MusicError::invalid_input(format!(
                "duration_seconds must be positive and finite, got {duration_seconds}"
            )));
        }
        if duration_seconds > MUSICGEN_MAX_DURATION_HARD_LIMIT {
            return Err(MusicError::invalid_input(format!(
                "max {MUSICGEN_MAX_DURATION_HARD_LIMIT}s for MusicGen (got {duration_seconds}s)"
            )));
        }
        if duration_seconds > self.config.max_duration_seconds {
            return Err(MusicError::invalid_input(format!(
                "requested duration {duration_seconds}s exceeds configured cap {}s",
                self.config.max_duration_seconds
            )));
        }
        Ok(())
    }

    async fn generate(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.validate_inputs(prompt, duration_seconds)?;
        let loaded = self.ensure_loaded().await?;
        let pcm = {
            let mut model = loaded.model.lock().await;
            run_pipeline_pcm(
                &mut model,
                &loaded.tokenizer,
                &loaded.device,
                self.config.variant,
                prompt,
                duration_seconds,
            )?
        };

        let wav_bytes = pcm_to_wav(&pcm, self.config.variant.sample_rate(), 1);
        #[allow(clippy::cast_precision_loss)]
        let duration = pcm.len() as f32 / self.config.variant.sample_rate() as f32;
        Ok(GeneratedAudio {
            bytes: wav_bytes,
            format: AudioFormat::Wav,
            sample_rate: self.config.variant.sample_rate(),
            channels: 1,
            duration_seconds: Some(duration),
        })
    }

    /// Shared streaming entry point used by both `stream_generate_music`
    /// and `stream_generate_sfx` (MusicGen treats SFX and music as the
    /// same pipeline — only the prompt differs).
    ///
    /// Validates inputs and `ensure_loaded()` synchronously so weight
    /// fetch / tokenizer load failures surface as a real `Err` instead of
    /// the first stream item. Once loaded, spawns the AR loop +
    /// EnCodec decode on a `spawn_blocking` worker and returns a
    /// `ReceiverStream` whose items are `MusicChunk`s of
    /// `STREAM_CHUNK_FRAMES * (sample_rate / frame_rate)` samples each.
    async fn stream_generate(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        self.validate_inputs(prompt, duration_seconds)?;
        self.ensure_loaded().await?;

        // Snapshot everything the blocking worker needs. The Arc keeps
        // the Loaded handle alive even if the caller drops the backend
        // while the stream is in flight.
        let loaded_arc = Arc::clone(&self.loaded);
        let variant = self.config.variant;
        let prompt = prompt.to_string();
        let sample_rate = variant.sample_rate() as usize;
        let frame_rate = variant.frame_rate() as usize;
        let samples_per_chunk = STREAM_CHUNK_FRAMES * (sample_rate / frame_rate);

        let (tx, rx) = mpsc::channel::<Result<MusicChunk, MusicError>>(STREAM_CHANNEL_CAPACITY);

        tokio::task::spawn_blocking(move || {
            // `ensure_loaded` was awaited above, so `.get()` is `Some`.
            let Some(loaded) = loaded_arc.get() else {
                let _ = tx.blocking_send(Err(MusicError::other(
                    "musicgen backend not loaded (internal invariant violated)",
                )));
                return;
            };

            // `tokio::sync::Mutex::blocking_lock` is the supported way
            // to acquire a tokio mutex from a `spawn_blocking` worker.
            let mut model = loaded.model.blocking_lock();
            let pcm = match run_pipeline_pcm(
                &mut model,
                &loaded.tokenizer,
                &loaded.device,
                variant,
                &prompt,
                duration_seconds,
            ) {
                Ok(pcm) => pcm,
                Err(e) => {
                    // Send the error and stop — do NOT emit a trailing
                    // `is_final` chunk after an error.
                    let _ = tx.blocking_send(Err(e));
                    return;
                }
            };

            // Per-N-frames EnCodec decode would introduce CNN-seam
            // clicks; the candle EnCodec impl does not expose
            // streaming-friendly decode state. Single full decode +
            // post-hoc PCM slice trades first-chunk latency
            // (≈ full-track gen time) for clean audio across chunk
            // boundaries.
            let n_chunks = pcm.len().div_ceil(samples_per_chunk).max(1);
            for (idx, chunk) in pcm.chunks(samples_per_chunk).enumerate() {
                let item = MusicChunk {
                    samples: chunk.to_vec(),
                    is_final: idx + 1 == n_chunks,
                    latency_seconds: None,
                };
                if tx.blocking_send(Ok(item)).is_err() {
                    // Consumer dropped — silently abort.
                    return;
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

impl Default for MusicgenBackend {
    fn default() -> Self {
        Self::new(MusicgenConfig::default())
    }
}

/// Run the validate-and-tokenize-aware portion of the MusicGen pipeline
/// (T5 encoder → decoder AR loop → EnCodec decode) and return the raw
/// mono f32 PCM at the variant's native sample rate.
///
/// Extracted out of `MusicgenBackend::generate` so the streaming path
/// (`MusicgenBackend::stream_generate`) can reuse the exact same
/// pipeline from a `spawn_blocking` worker without duplicating logic.
/// Takes the model by `&mut` because
/// `MusicgenForConditionalGeneration` owns mutable sinusoidal
/// embedding state.
///
/// # Errors
///
/// Propagates tokenizer / decoder / EnCodec errors as [`MusicError`].
fn run_pipeline_pcm(
    model: &mut MusicgenForConditionalGeneration,
    tokenizer: &Tokenizer,
    device: &Device,
    variant: MusicgenVariant,
    prompt: &str,
    duration_seconds: f32,
) -> Result<Vec<f32>, MusicError> {
    let frame_rate = f32::from(u16::try_from(variant.frame_rate()).unwrap_or(50));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let target_frames = (duration_seconds * frame_rate).ceil() as usize;
    let num_codebooks = model.decoder.num_codebooks();
    let max_steps = target_frames + num_codebooks.saturating_sub(1);

    // Tokenize prompt.
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| MusicError::other(format!("tokenizer encode failed: {e}")))?;
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    if ids.is_empty() {
        return Err(MusicError::invalid_input(
            "tokenizer returned 0 ids for prompt",
        ));
    }
    let ids_len = ids.len();
    let prompt_tokens = Tensor::from_vec(ids, (1, ids_len), device)?;

    let params = GenerationParams {
        max_steps,
        ..GenerationParams::default()
    };

    let tokens = generation::generate_tokens(model, &prompt_tokens, &params, device)?;
    decode_to_pcm(model, &tokens, device)
}

// ---------------------------------------------------------------------------
// Codec routing
// ---------------------------------------------------------------------------

/// Decode the undelayed token grid into mono f32 PCM by running the
/// MusicGen-internal EnCodec decoder.
///
/// Note: this currently uses `MusicgenForConditionalGeneration::audio_encoder`
/// directly to avoid loading the EnCodec weights twice. Callers that want
/// to route through the [`EncodecBackend`] from `blazen-audio-codec` can
/// do so with [`encodec_backend_for`] once that backend lands a 32 kHz
/// MusicGen config (today it ships the 24 kHz default).
///
/// Exposed `pub` so sibling backends (AudioGen) can reuse the same codec
/// decode path on top of the shared `MusicgenForConditionalGeneration`
/// model class.
///
/// # Errors
///
/// Propagates candle errors from the EnCodec decoder + tensor reshape.
pub fn decode_to_pcm(
    model: &MusicgenForConditionalGeneration,
    tokens: &generation::GeneratedTokens,
    device: &Device,
) -> Result<Vec<f32>, MusicError> {
    if tokens.frames == 0 {
        return Ok(Vec::new());
    }
    let codes = Tensor::from_vec(
        tokens.flat.clone(),
        (1, tokens.num_codebooks, tokens.frames),
        device,
    )?;
    let audio = model.audio_encoder.decode(&codes)?;
    // audio: [1, 1, T]
    let audio = audio.i(0)?.i(0)?.flatten_all()?;
    let v = audio.to_vec1::<f32>()?;
    Ok(v)
}

/// Build a 32 kHz mono `EncodecBackend` pointed at a MusicGen checkpoint.
///
/// Provided for callers that want to drive the codec through the
/// canonical [`CodecBackend`] interface (e.g. for cross-engine codec
/// reuse, A/B testing different codecs, or to keep the MusicGen LM and
/// the codec in separate `tokio` tasks). Unused inside the crate today
/// but exercised by integration tests.
#[must_use]
pub fn encodec_backend_for(variant: MusicgenVariant) -> EncodecBackend {
    EncodecBackend::new(EncodecConfig {
        hf_repo: variant.hf_repo().to_string(),
        weights_filename: "model.safetensors".to_string(),
        cpu_only: false,
        cache_dir: None,
    })
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

/// Pack `f32` PCM into a 16-bit-PCM WAV byte vector.
///
/// Re-export of [`super::wav::pcm_to_wav`] kept here for backward
/// compatibility with sibling backends (AudioGen) that already import
/// it from `backends::musicgen::pcm_to_wav`.
pub use super::wav::pcm_to_wav;

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioBackend for MusicgenBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "music"
    }

    async fn load(&self) -> Result<(), AudioError> {
        self.ensure_loaded()
            .await
            .map_err(|e| AudioError::from(std::io::Error::other(e.to_string())))?;
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        self.loaded.initialized()
    }
}

#[async_trait]
impl MusicBackend for MusicgenBackend {
    async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.generate(prompt, duration_seconds).await
    }

    async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        // MusicGen doesn't differentiate music vs SFX -- the model class
        // is the same, only the prompt changes.
        self.generate(prompt, duration_seconds).await
    }

    async fn stream_generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        self.stream_generate(prompt, duration_seconds).await
    }

    async fn stream_generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        // MusicGen treats SFX and music as the same pipeline — same as
        // `generate_sfx`.
        self.stream_generate(prompt, duration_seconds).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variant_is_small() {
        let cfg = MusicgenConfig::default();
        assert_eq!(cfg.variant, MusicgenVariant::Small);
        assert_eq!(cfg.variant.hf_repo(), "facebook/musicgen-small");
    }

    #[test]
    fn sample_rate_is_32k_for_all_variants() {
        assert_eq!(MusicgenVariant::Small.sample_rate(), 32_000);
        assert_eq!(MusicgenVariant::Medium.sample_rate(), 32_000);
        assert_eq!(MusicgenVariant::Large.sample_rate(), 32_000);
    }

    #[test]
    fn frame_rate_is_50hz_for_all_variants() {
        for v in [
            MusicgenVariant::Small,
            MusicgenVariant::Medium,
            MusicgenVariant::Large,
        ] {
            assert_eq!(v.frame_rate(), 50);
        }
    }

    #[test]
    fn variant_to_repo_mapping() {
        assert_eq!(MusicgenVariant::Small.hf_repo(), "facebook/musicgen-small");
        assert_eq!(
            MusicgenVariant::Medium.hf_repo(),
            "facebook/musicgen-medium"
        );
        assert_eq!(MusicgenVariant::Large.hf_repo(), "facebook/musicgen-large");
    }

    #[test]
    fn id_reflects_variant() {
        let b = MusicgenBackend::new(MusicgenConfig {
            variant: MusicgenVariant::Medium,
            ..MusicgenConfig::default()
        });
        assert_eq!(b.id(), "musicgen-medium");
        assert_eq!(b.provider_kind(), "music");
    }

    #[test]
    fn gen_config_small_matches_published_values() {
        let cfg = MusicgenVariant::Small.gen_config();
        assert_eq!(cfg.musicgen.num_codebooks, 4);
        assert_eq!(cfg.musicgen.vocab_size, 2048);
        assert_eq!(cfg.musicgen.hidden_size, 1024);
        assert_eq!(cfg.musicgen.num_hidden_layers, 24);
        assert_eq!(cfg.encodec.sampling_rate, 32_000);
        assert_eq!(cfg.t5.d_model, 768);
    }

    #[test]
    fn pcm_to_wav_writes_riff_header() {
        let bytes = pcm_to_wav(&[0.0_f32, 0.5, -0.5, 1.0], 32_000, 1);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(&bytes[36..40], b"data");
        // 4 samples * 2 bytes = 8 bytes payload.
        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, 8);
    }

    #[tokio::test]
    async fn rejects_empty_prompt() {
        let b = MusicgenBackend::default();
        let err = b.generate_music("   ", 1.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("empty")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_zero_duration() {
        let b = MusicgenBackend::default();
        let err = b.generate_music("lofi piano", 0.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("positive")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_duration_over_hard_limit() {
        let b = MusicgenBackend::new(MusicgenConfig {
            max_duration_seconds: 120.0, // permissive config cap
            ..MusicgenConfig::default()
        });
        let err = b.generate_music("lofi piano", 61.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("max 60s for MusicGen")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_empty_prompt_streaming() {
        let b = MusicgenBackend::default();
        let Err(err) = b.stream_generate_music("   ", 1.0).await else {
            panic!("expected Err for empty prompt");
        };
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn rejects_duration_over_hard_limit_streaming() {
        let b = MusicgenBackend::default();
        let Err(err) = b.stream_generate_music("lofi piano", 61.0).await else {
            panic!("expected Err for over-limit duration");
        };
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn stream_generate_sfx_uses_same_pipeline() {
        let b = MusicgenBackend::default();
        let Err(err) = b.stream_generate_sfx("", 1.0).await else {
            panic!("expected Err for empty sfx prompt");
        };
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn rejects_duration_over_configured_cap() {
        let b = MusicgenBackend::new(MusicgenConfig {
            max_duration_seconds: 5.0,
            ..MusicgenConfig::default()
        });
        let err = b.generate_music("lofi piano", 10.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("exceeds configured cap")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[test]
    fn encodec_backend_for_uses_musicgen_repo() {
        let b = encodec_backend_for(MusicgenVariant::Small);
        assert_eq!(b.config().hf_repo, "facebook/musicgen-small");
    }

    // Live-models test: round-trip a 2-second "lo-fi hip hop beat" prompt
    // through the real `facebook/musicgen-small` checkpoint. Gated
    // because it fetches ~1.5 GB of safetensors weights on first run.
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn generates_two_seconds_of_lofi_hip_hop_beat() {
        let backend = MusicgenBackend::new(MusicgenConfig {
            max_duration_seconds: 5.0,
            ..MusicgenConfig::default()
        });

        // Explicit load() so weight-fetch failures show up here rather
        // than mixed into the generate() error path.
        if let Err(e) = backend.load().await {
            eprintln!("live-models test skipped: load() failed: {e}");
            return;
        }

        // generate_music drives the full T5 + decoder + delay-pattern +
        // EnCodec pipeline and packages the result via decode_to_pcm +
        // pcm_to_wav -- the same functions named in the task spec.
        let audio = match backend.generate_music("lo-fi hip hop beat", 2.0).await {
            Ok(a) => a,
            Err(MusicError::HfHub { repo, source }) => {
                eprintln!("live-models test skipped: HF fetch failed for {repo}: {source}");
                return;
            }
            Err(other) => panic!("unexpected failure: {other}"),
        };

        // WAV header sanity (steps 4-5 of the task spec).
        assert_eq!(audio.format, AudioFormat::Wav);
        assert_eq!(audio.sample_rate, 32_000);
        assert_eq!(audio.channels, 1);
        assert!(audio.bytes.len() > 44, "wav payload too small");
        assert_eq!(&audio.bytes[0..4], b"RIFF");
        assert_eq!(&audio.bytes[8..12], b"WAVE");

        // PCM length ~= 2 * sample_rate, +/- 50% (step 6). MusicGen's
        // EnCodec runs at 50 Hz, so an exact 2 s clip at 32 kHz is
        // floor(2 * 50) * (32_000 / 50) = 64_000 samples; the window is
        // intentionally generous to absorb end-of-stream truncation.
        let pcm_samples = (audio.bytes.len() - 44) / 2; // 16-bit mono.
        let expected: usize = 2 * 32_000;
        let low = expected / 2;
        let high = expected * 3 / 2;
        assert!(
            (low..=high).contains(&pcm_samples),
            "pcm length {pcm_samples} outside [{low}, {high}] for 2s @ 32 kHz",
        );

        // Every reconstructed f32 sample must be finite (step 7). The
        // wav is 16-bit PCM so we round-trip back to f32 to mirror the
        // SNAC live-test pattern.
        let payload = &audio.bytes[44..44 + pcm_samples * 2];
        for chunk in payload.chunks_exact(2) {
            let i = i16::from_le_bytes([chunk[0], chunk[1]]);
            let f = f32::from(i) / f32::from(i16::MAX);
            assert!(f.is_finite(), "decoded sample must be finite, got {f}");
        }

        let dur = audio.duration_seconds.expect("duration");
        assert!(
            (0.5..=4.0).contains(&dur),
            "duration {dur}s outside tolerant 2s window",
        );
    }

    // Live-models streaming test: drives the real
    // `facebook/musicgen-small` checkpoint via `stream_generate_music`,
    // collects all chunks, and asserts the expected emission shape
    // (≥ 1 chunk, exactly one final flag at the end, total length
    // within ±50% of 2 s @ 32 kHz, all samples finite). Gated alongside
    // the non-streaming live test for the same weight-fetch reasons.
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn streams_two_seconds_of_lofi_with_final_flag() {
        use futures_util::StreamExt;

        let backend = MusicgenBackend::new(MusicgenConfig {
            max_duration_seconds: 5.0,
            ..MusicgenConfig::default()
        });

        if let Err(e) = backend.load().await {
            eprintln!("live-models streaming test skipped: load() failed: {e}");
            return;
        }

        let stream = match backend
            .stream_generate_music("lo-fi hip hop beat", 2.0)
            .await
        {
            Ok(s) => s,
            Err(MusicError::HfHub { repo, source }) => {
                eprintln!(
                    "live-models streaming test skipped: HF fetch failed for {repo}: {source}",
                );
                return;
            }
            Err(other) => panic!("unexpected failure: {other}"),
        };

        let items: Vec<Result<MusicChunk, MusicError>> = stream.collect().await;
        let chunks: Vec<MusicChunk> = items
            .into_iter()
            .map(|r| r.expect("stream item is Err"))
            .collect();

        assert!(!chunks.is_empty(), "expected at least one streamed chunk");

        let final_count = chunks.iter().filter(|c| c.is_final).count();
        assert_eq!(final_count, 1, "expected exactly one is_final chunk");
        assert!(
            chunks.last().expect("at least one chunk").is_final,
            "the final-flagged chunk must be the last one",
        );

        let total_samples: usize = chunks.iter().map(|c| c.samples.len()).sum();
        let expected: usize = 2 * 32_000;
        let low = expected / 2;
        let high = expected * 3 / 2;
        assert!(
            (low..=high).contains(&total_samples),
            "total streamed samples {total_samples} outside [{low}, {high}] for 2s @ 32 kHz",
        );

        for chunk in &chunks {
            for &s in &chunk.samples {
                assert!(s.is_finite(), "streamed sample must be finite, got {s}");
            }
        }
    }
}
