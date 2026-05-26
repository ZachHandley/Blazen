//! End-to-end orchestration for the Stable Audio Open native candle port.
//!
//! Wires the four sibling modules
//! ([`super::conditioner::Conditioner`], [`super::dit::DiT`],
//! [`super::oobleck::OobleckVAE`], [`super::sampler::Sampler`]) into a
//! single text-to-audio pipeline and exposes a [`StableAudioBackend`] that
//! implements [`blazen_audio::AudioBackend`] + [`crate::traits::MusicBackend`].
//!
//! # Pipeline flow
//!
//! ```text
//! prompt (&str)
//!     │
//!     ▼  conditioner.encode_text(prompt)
//!   (t5_tokens, t5_mask)                          shapes: (1, T_text, 768), (1, T_text)
//!     │
//!     ▼  build_numeric_conds(0.0, duration_s, 1)
//!   (seconds_start, seconds_total)                shapes: (1,), (1,)
//!     │
//!     ▼  sampler.sample((1, 64, T_latent), seed, denoise=|x, σ| DiT.forward(...))
//!   denoised latent                               shape:  (1, 64, T_latent)
//!     │
//!     ▼  vae.decode(latent)
//!   stereo waveform                               shape:  (1, 2, T_audio)
//!     │
//!     ▼  pcm_to_wav(interleaved, 44_100, 2)
//!   GeneratedAudio { Wav, 44_100 Hz, 2ch }
//! ```
//!
//! `T_latent = ceil(duration_seconds * 86.0)` for the Small variant
//! (`86.13 Hz = 44_100 / 512` latent frame rate; see [`OobleckConfig::stable_audio_small`]).
//!
//! # Status
//!
//! Wave 3.5 of the Stable Audio port: the module is registered in
//! [`super`] and [`StableAudioBackend`] is re-exported as the real
//! `StableAudioBackend` whenever the `stable-audio` cargo feature is on.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::needless_pass_by_value)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioFormat, GeneratedAudio};
use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_transformers::models::t5::Config as T5Config;
use futures_core::Stream;
use tokio::sync::{Mutex, mpsc};

use crate::backends::wav::pcm_to_wav;
use crate::error::MusicError;
use crate::traits::{MusicBackend, MusicChunk};

use super::conditioner::{Conditioner, SMALL_MAX_TEXT_TOKENS, build_numeric_conds};
use super::dit::{DiT, DiTConfig};
use super::oobleck::{OobleckConfig, OobleckVAE};
use super::sampler::{DistilledSolver, DpmSolverPlusPlus, Sampler};
use super::weights::StableAudioWeights;

/// Integer latent frame rate of the Small variant Oobleck VAE
/// (`floor(44_100 / 512) ≈ 86 Hz`; the exact float is `86.13`, but we
/// round down to integer for deterministic latent-length math).
const SMALL_LATENT_FRAME_RATE: f32 = 86.0;

/// Audio sample rate of every Stable Audio Open variant.
const STABLE_AUDIO_SAMPLE_RATE: u32 = 44_100;

/// Output channel count (stereo) for every Stable Audio Open variant.
const STABLE_AUDIO_CHANNELS: u16 = 2;

/// Hard upper bound on `duration_seconds`. The Small variant's distilled
/// sampler is trained for ≤11 s; 1.0 caps at ~47 s. Clamp at 60 to leave
/// headroom while still rejecting obviously-invalid inputs.
const MAX_DURATION_SECONDS: f32 = 60.0;

/// Per-emitted-chunk duration in milliseconds. 250 ms is the sweet spot:
/// long enough to amortise tokio task wake overhead and keep per-chunk
/// allocations modest, short enough that a downstream player can begin
/// playback within one chunk of the final VAE decode completing.
const STREAM_CHUNK_MS: u32 = 250;

/// Bounded mpsc capacity feeding the output stream. Four chunks ≈ 1 s of
/// buffered audio back-pressure headroom; mirrors Spark-TTS's
/// `STREAM_CHANNEL_CAPACITY=4`.
const STREAM_CHANNEL_CAPACITY: usize = 4;

/// Hyperparameter pack describing which Stable Audio Open weights are
/// being loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableAudioVariant {
    /// `stabilityai/stable-audio-open-small` — 341 M params, 8-step
    /// distilled sampler, 11 s output cap.
    Small,
    /// `stabilityai/stable-audio-open-1.0` — 1.21 B params, 100-step
    /// DPM-Solver++, 47 s output cap.
    Open1_0,
}

impl StableAudioVariant {
    /// Default Hugging Face repo for the variant.
    #[must_use]
    pub const fn hf_repo(self) -> &'static str {
        match self {
            Self::Small => "stabilityai/stable-audio-open-small",
            Self::Open1_0 => "stabilityai/stable-audio-open-1.0",
        }
    }

    /// DiT hyperparameters for the variant.
    #[must_use]
    pub fn dit_config(self) -> DiTConfig {
        match self {
            Self::Small => DiTConfig::stable_audio_small(),
            Self::Open1_0 => DiTConfig::stable_audio_open_1_0(),
        }
    }

    /// VAE hyperparameters. The Oobleck config is shared between Small
    /// and 1.0 (both use `channels=[128,256,512,1024,1024]`,
    /// `strides=[2,4,8,8]`, `latent_dim=64`, stereo, 44.1 kHz).
    #[must_use]
    pub fn vae_config(self) -> OobleckConfig {
        OobleckConfig::stable_audio_small()
    }

    /// Build the appropriate sampler for the variant.
    fn build_sampler(self) -> Arc<dyn Sampler> {
        match self {
            Self::Small => Arc::new(DistilledSolver::stable_audio_small()),
            Self::Open1_0 => Arc::new(DpmSolverPlusPlus::stable_audio_1_0()),
        }
    }
}

/// Construction parameters for [`StableAudioPipeline::load`].
#[derive(Debug, Clone)]
pub struct StableAudioConfig {
    /// Hugging Face repo id (e.g. `"stabilityai/stable-audio-open-small"`).
    /// Used as the cache key when `local_weights_path` is `None`.
    pub hf_repo: String,
    /// Optional override for the model safetensors path. If `Some`, the
    /// HF download step is skipped and the file is mmap-loaded directly.
    pub local_weights_path: Option<PathBuf>,
    /// Path to the T5 SentencePiece `tokenizer.json` shipped with the
    /// model repo.
    pub tokenizer_path: PathBuf,
    /// Inference device. CPU works for the Small variant in ~30-60 s;
    /// CUDA / Metal recommended for 1.0.
    pub device: Device,
    /// Compute precision for the DiT and VAE forward passes. F32 is the
    /// safest baseline; BF16 trades ~2× memory for some numerical drift.
    pub dtype: DType,
    /// Variant — drives sampler choice and DiT hyperparameters.
    pub variant: StableAudioVariant,
}

impl StableAudioConfig {
    /// Construct a Small-variant config rooted at the default HF repo,
    /// with no local-weights override and a CPU/F32 default device pair.
    #[must_use]
    pub fn small(tokenizer_path: PathBuf) -> Self {
        Self {
            hf_repo: StableAudioVariant::Small.hf_repo().to_string(),
            local_weights_path: None,
            tokenizer_path,
            device: Device::Cpu,
            dtype: DType::F32,
            variant: StableAudioVariant::Small,
        }
    }
}

/// Loaded text+diffusion+VAE stack ready to serve `generate(prompt, ...)`
/// calls.
pub struct StableAudioPipeline {
    conditioner: Mutex<Conditioner>,
    dit: Arc<DiT>,
    vae: Arc<OobleckVAE>,
    sampler: Arc<dyn Sampler>,
    sample_rate: u32,
    device: Device,
    dtype: DType,
}

impl StableAudioPipeline {
    /// Build the pipeline by downloading + mmap-loading the weights.
    ///
    /// # Errors
    ///
    /// Returns [`MusicError::HfHub`] on download failure,
    /// [`MusicError::Io`] on local-file IO failure, and
    /// [`MusicError::Candle`] on safetensors parse / weight-shape errors.
    pub async fn load(config: StableAudioConfig) -> Result<Self, MusicError> {
        let device = config.device.clone();
        let dtype = config.dtype;
        let variant = config.variant;
        let tokenizer_path = config.tokenizer_path.clone();
        let local_weights = config.local_weights_path.clone();

        // All compute on the safetensors mmap + module construction runs
        // on a blocking thread — file IO and tens of megabytes of tensor
        // shape inference are not friendly to the tokio runtime.
        let (conditioner, dit, vae) =
            tokio::task::spawn_blocking(move || -> Result<_, MusicError> {
                // The remap-aware loader lives in `weights.rs`. Use
                // `load_from_hf` by default; honour the local-path
                // override when present (for hermetic CI runs).
                let weights = match local_weights {
                    Some(p) => StableAudioWeights::load(&p, &device, dtype)?,
                    None => StableAudioWeights::load_from_hf(&device, dtype)?,
                };
                let vb = weights.var_builder().clone();

                // Build T5 config. The Stable Audio Open release ships
                // `t5-base` (12 layers, 12 heads, d_model=768, d_ff=3072,
                // d_kv=64). We mirror that inline rather than pulling in
                // a JSON dependency — the values are stable across both
                // open variants.
                let t5_cfg = T5Config {
                    vocab_size: 32_128,
                    d_model: 768,
                    d_kv: 64,
                    d_ff: 3_072,
                    num_layers: 12,
                    num_heads: 12,
                    pad_token_id: 0,
                    eos_token_id: 1,
                    ..T5Config::default()
                };

                let conditioner = Conditioner::new(
                    &vb.pp("conditioner"),
                    &t5_cfg,
                    &tokenizer_path,
                    SMALL_MAX_TEXT_TOKENS,
                )?;
                let dit = DiT::new(vb.pp("dit"), variant.dit_config())?;
                let vae = OobleckVAE::new(vb.pp("oobleck"), &variant.vae_config())?;

                Ok((conditioner, dit, vae))
            })
            .await
            .map_err(|e| MusicError::other(format!("blocking task join failed: {e}")))??;

        Ok(Self {
            conditioner: Mutex::new(conditioner),
            dit: Arc::new(dit),
            vae: Arc::new(vae),
            sampler: variant.build_sampler(),
            sample_rate: STABLE_AUDIO_SAMPLE_RATE,
            device: config.device,
            dtype: config.dtype,
        })
    }

    /// Configured output sample rate (Hz).
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Inference device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Compute precision.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Compute the latent time-axis length for `duration_seconds` at the
    /// Small variant's integer-86 Hz latent frame rate. Always rounds up
    /// so the produced audio is at least `duration_seconds` long.
    #[must_use]
    pub fn latent_frames_for(duration_seconds: f32) -> usize {
        (duration_seconds * SMALL_LATENT_FRAME_RATE).ceil() as usize
    }

    /// Validate caller inputs (non-empty prompt, finite & positive
    /// duration within the configured cap).
    fn validate_inputs(prompt: &str, duration_seconds: f32) -> Result<(), MusicError> {
        if prompt.trim().is_empty() {
            return Err(MusicError::invalid_input("prompt is empty"));
        }
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err(MusicError::invalid_input(format!(
                "duration_seconds must be positive and finite, got {duration_seconds}"
            )));
        }
        if duration_seconds > MAX_DURATION_SECONDS {
            return Err(MusicError::invalid_input(format!(
                "duration_seconds {duration_seconds} exceeds Stable Audio cap of {MAX_DURATION_SECONDS}s"
            )));
        }
        Ok(())
    }

    /// Run the diffusion + VAE pipeline end-to-end and return the
    /// interleaved stereo PCM (`Vec<f32>` of length `T_audio * 2` at
    /// `self.sample_rate`).
    ///
    /// Shared by both [`Self::generate`] (which wraps the PCM in a WAV
    /// container) and [`Self::generate_stream`] (which chunks the PCM
    /// into [`MusicChunk`] emissions). Both callers go through this
    /// helper so the inference path can never drift between them.
    ///
    /// # Errors
    ///
    /// - [`MusicError::InvalidInput`] for malformed inputs.
    /// - [`MusicError::Candle`] for any tensor-level failure inside the
    ///   conditioner, DiT, sampler, or VAE.
    async fn generate_pcm(
        &self,
        prompt: &str,
        duration_seconds: f32,
        seed: u64,
    ) -> Result<Vec<f32>, MusicError> {
        Self::validate_inputs(prompt, duration_seconds)?;

        // 1. Text conditioning. The Conditioner is `&mut`-bound (the T5
        //    encoder mutates its KV cache between calls); acquire the
        //    mutex inside the async stack, then release before kicking
        //    off `spawn_blocking` so the lock isn't held across the
        //    inference critical section.
        let (t5_tokens, _t5_mask) = {
            let mut cond = self.conditioner.lock().await;
            cond.encode_text(prompt)?
        };

        let dit = Arc::clone(&self.dit);
        let vae = Arc::clone(&self.vae);
        let sampler = Arc::clone(&self.sampler);
        let device = self.device.clone();
        let dtype = self.dtype;

        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, MusicError> {
            // 2. Numeric conditioning + closure capture. The DiT consumes
            //    `seconds_start` (where in the original clip this
            //    generation sits) and `seconds_total` (full clip length).
            //    For pure text-to-audio we anchor `start=0` and the
            //    caller's requested `duration` as the total.
            let (seconds_start, seconds_total) =
                build_numeric_conds(0.0, duration_seconds, 1, &device)?;
            let t5_tokens = t5_tokens.to_dtype(dtype)?;

            // 3. Sampler. The closure clones `Arc<DiT>` and the
            //    conditioning tensors so it owns everything it needs for
            //    the lifetime of the loop.
            let t_latent = Self::latent_frames_for(duration_seconds);
            let latent_shape = [1_usize, 64, t_latent];

            let dit_for_closure = dit;
            let t5_for_closure = t5_tokens;
            let start_for_closure = seconds_start;
            let total_for_closure = seconds_total;
            let device_for_closure = device.clone();

            let denoise_fn = move |latent: &Tensor, sigma: f32| -> CandleResult<Tensor> {
                // Stable Audio uses `t = σ / (1 + σ)` for the DiT's
                // continuous timestep input. The sampler hands us a
                // sigma here; map it into the DiT's `t` domain.
                let timestep_value = sigma / (1.0 + sigma);
                let timestep = Tensor::new(&[timestep_value], &device_for_closure)?;
                let latent_dt = latent.to_dtype(dtype)?;
                dit_for_closure.forward(
                    &latent_dt,
                    &t5_for_closure,
                    &timestep,
                    &start_for_closure,
                    &total_for_closure,
                )
            };

            let latent = sampler
                .sample(&latent_shape, seed, &device, &denoise_fn)
                .map_err(MusicError::from)?;

            // 4. VAE decode. `(1, 2, T_audio)` stereo waveform; `T_audio
            //    = t_latent * 512` (the Oobleck downsample ratio).
            let latent = latent.to_dtype(dtype)?;
            let waveform = vae.decode(&latent)?;

            // 5. Interleave L/R into a single `Vec<f32>` of length
            //    `T_audio * 2` for `pcm_to_wav`. The (1, 2, T) shape
            //    means: batch 0, channel 0 is left; channel 1 is right.
            let waveform = waveform.to_dtype(DType::F32)?;
            let chans = waveform.dim(1)?;
            let t_audio = waveform.dim(2)?;
            let mut interleaved = Vec::with_capacity(t_audio * chans);
            let left = waveform.i(0)?.i(0)?.to_vec1::<f32>()?;
            let right = if chans >= 2 {
                waveform.i(0)?.i(1)?.to_vec1::<f32>()?
            } else {
                left.clone()
            };
            for i in 0..t_audio {
                interleaved.push(left[i]);
                interleaved.push(right[i]);
            }
            Ok(interleaved)
        })
        .await
        .map_err(|e| MusicError::other(format!("blocking task join failed: {e}")))?
    }

    /// Generate `duration_seconds` of audio for `prompt`.
    ///
    /// # Errors
    ///
    /// - [`MusicError::InvalidInput`] for malformed inputs.
    /// - [`MusicError::Candle`] for any tensor-level failure inside the
    ///   conditioner, DiT, sampler, or VAE.
    pub async fn generate(
        &self,
        prompt: &str,
        duration_seconds: f32,
        seed: u64,
    ) -> Result<GeneratedAudio, MusicError> {
        let pcm = self.generate_pcm(prompt, duration_seconds, seed).await?;
        let wav_bytes = pcm_to_wav(&pcm, self.sample_rate, STABLE_AUDIO_CHANNELS);
        let duration_seconds_actual =
            pcm.len() as f32 / (self.sample_rate as f32 * f32::from(STABLE_AUDIO_CHANNELS));
        Ok(GeneratedAudio {
            bytes: wav_bytes,
            format: AudioFormat::Wav,
            sample_rate: self.sample_rate,
            channels: STABLE_AUDIO_CHANNELS,
            duration_seconds: Some(duration_seconds_actual),
        })
    }

    /// Streaming variant of [`Self::generate`].
    ///
    /// Stable Audio is a diffusion + VAE pipeline; mid-denoising latents
    /// are pure noise and the VAE produces meaningful audio only when
    /// given the fully-denoised latent. We therefore run the full
    /// generation and chunk the resulting PCM into ~250 ms slices.
    /// The mpsc + `spawn` + `receiver_into_stream` shape matches
    /// Spark-TTS for API uniformity across the audio crates.
    ///
    /// Validation errors (empty prompt, non-finite or out-of-range
    /// duration) surface as the first stream item rather than at the
    /// call site — this matches the Spark-TTS error-channel convention.
    pub fn generate_stream(
        self: &Arc<Self>,
        prompt: &str,
        duration_seconds: f32,
        seed: u64,
    ) -> Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>> {
        let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let pipeline = Arc::clone(self);
        let prompt = prompt.to_owned();

        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let pcm = match pipeline.generate_pcm(&prompt, duration_seconds, seed).await {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            let chunk_len =
                (STABLE_AUDIO_SAMPLE_RATE * u32::from(STABLE_AUDIO_CHANNELS) * STREAM_CHUNK_MS)
                    / 1000;
            let chunk_len = chunk_len as usize;
            let total = pcm.len();

            // Edge case: empty PCM (shouldn't happen because
            // `validate_inputs` rejects duration <= 0, but guard for
            // completeness — every successful stream must terminate
            // with at least one `is_final = true` chunk).
            if total == 0 {
                let _ = tx
                    .send(Ok(MusicChunk {
                        samples: Vec::new(),
                        is_final: true,
                        latency_seconds: Some(started.elapsed().as_secs_f32()),
                    }))
                    .await;
                return;
            }

            let mut offset = 0;
            while offset < total {
                let end = (offset + chunk_len).min(total);
                let is_final = end == total;
                let chunk = MusicChunk {
                    samples: pcm[offset..end].to_vec(),
                    is_final,
                    latency_seconds: Some(started.elapsed().as_secs_f32()),
                };
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
                offset = end;
            }
        });

        receiver_into_stream(rx)
    }
}

/// Stable Audio backend implementing [`AudioBackend`] + [`MusicBackend`].
///
/// Re-exported by [`super`] as `StableAudioBackend` whenever the
/// `stable-audio` cargo feature is on; otherwise [`super`] re-exports a
/// stub of the same name that surfaces
/// [`MusicError::NotYetImplemented`] from every entry point.
pub struct StableAudioBackend {
    pipeline: Arc<StableAudioPipeline>,
}

impl StableAudioBackend {
    /// Load the backend from `config`.
    ///
    /// # Errors
    ///
    /// Propagates any [`MusicError`] surfaced by
    /// [`StableAudioPipeline::load`].
    pub async fn load(config: StableAudioConfig) -> Result<Self, MusicError> {
        let pipeline = StableAudioPipeline::load(config).await?;
        Ok(Self {
            pipeline: Arc::new(pipeline),
        })
    }

    /// Borrow the loaded pipeline.
    #[must_use]
    pub fn pipeline(&self) -> &StableAudioPipeline {
        &self.pipeline
    }
}

#[async_trait]
impl AudioBackend for StableAudioBackend {
    fn id(&self) -> &'static str {
        "stable-audio"
    }

    fn provider_kind(&self) -> &'static str {
        "music"
    }
}

#[async_trait]
impl MusicBackend for StableAudioBackend {
    async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        // Default to seed 0 — a future wave can plumb seed selection
        // through the trait or a request-level config.
        self.pipeline.generate(prompt, duration_seconds, 0).await
    }

    async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        // Stable Audio Open uses the same model for music and SFX —
        // the prompt is the only discriminator.
        self.pipeline.generate(prompt, duration_seconds, 0).await
    }

    async fn stream_generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        Ok(self.pipeline.generate_stream(prompt, duration_seconds, 0))
    }

    async fn stream_generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        // Same model serves both music and SFX; the prompt is the only
        // discriminator (matches the non-streaming path).
        Ok(self.pipeline.generate_stream(prompt, duration_seconds, 0))
    }
}

/// Convert an mpsc receiver into a `Pin<Box<dyn Stream>>` of its items.
/// Mirrors the helper that lives in the Spark-TTS streaming pipeline —
/// same shape, kept local to avoid a cross-crate re-export for a
/// four-line helper.
fn receiver_into_stream<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| (v, rx))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_construction() {
        let cfg = StableAudioConfig::small(PathBuf::from("/var/empty/tokenizer.json"));
        assert_eq!(cfg.variant, StableAudioVariant::Small);
        assert_eq!(cfg.hf_repo, "stabilityai/stable-audio-open-small");
        assert!(cfg.local_weights_path.is_none());
        assert!(matches!(cfg.device, Device::Cpu));
        assert!(matches!(cfg.dtype, DType::F32));
        assert_eq!(
            cfg.tokenizer_path,
            PathBuf::from("/var/empty/tokenizer.json")
        );
    }

    #[test]
    fn variant_repo_strings() {
        assert_eq!(
            StableAudioVariant::Small.hf_repo(),
            "stabilityai/stable-audio-open-small"
        );
        assert_eq!(
            StableAudioVariant::Open1_0.hf_repo(),
            "stabilityai/stable-audio-open-1.0"
        );
    }

    #[test]
    fn variant_dit_configs_distinct() {
        let small = StableAudioVariant::Small.dit_config();
        let big = StableAudioVariant::Open1_0.dit_config();
        assert_eq!(small.depth, 12);
        assert_eq!(big.depth, 24);
        assert_eq!(small.embed_dim, 768);
        assert_eq!(big.embed_dim, 1536);
    }

    #[test]
    fn t_latent_math() {
        // `ceil(10.5 * 86.0) = ceil(903.0) = 903`. The task spec quotes
        // 904 against the floating 86.13 Hz frame rate; we use the
        // integer 86 Hz so the latent-length math is bit-exact across
        // runs. Either constant gives audio at least `duration` seconds
        // long after Oobleck's 512× upsample.
        assert_eq!(StableAudioPipeline::latent_frames_for(10.5), 903);
        assert_eq!(StableAudioPipeline::latent_frames_for(11.0), 946);
        assert_eq!(StableAudioPipeline::latent_frames_for(1.0), 86);
        // 0.5s rounds up to 43 frames.
        assert_eq!(StableAudioPipeline::latent_frames_for(0.5), 43);
    }

    #[test]
    fn validate_inputs_rejects_empty_prompt() {
        let err = StableAudioPipeline::validate_inputs("   ", 5.0).unwrap_err();
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[test]
    fn validate_inputs_rejects_zero_duration() {
        let err = StableAudioPipeline::validate_inputs("foo", 0.0).unwrap_err();
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[test]
    fn validate_inputs_rejects_nan_duration() {
        let err = StableAudioPipeline::validate_inputs("foo", f32::NAN).unwrap_err();
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[test]
    fn validate_inputs_rejects_overlong_duration() {
        let err = StableAudioPipeline::validate_inputs("foo", 120.0).unwrap_err();
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[test]
    fn validate_inputs_accepts_normal_request() {
        assert!(StableAudioPipeline::validate_inputs("128 BPM house loop", 8.0).is_ok());
    }

    /// Integration smoke test — needs actual safetensors weights + T5
    /// tokenizer downloaded. Set `BLAZEN_STABLE_AUDIO_DIR` to a
    /// directory containing both files to enable.
    #[tokio::test]
    #[ignore = "requires stable-audio-open-small weights + tokenizer; set BLAZEN_STABLE_AUDIO_DIR to enable"]
    async fn pipeline_load_smoke() {
        let Ok(dir) = std::env::var("BLAZEN_STABLE_AUDIO_DIR") else {
            return;
        };
        let dir = PathBuf::from(dir);
        let cfg = StableAudioConfig {
            hf_repo: StableAudioVariant::Small.hf_repo().to_string(),
            local_weights_path: Some(dir.join("model.safetensors")),
            tokenizer_path: dir.join("tokenizer.json"),
            device: Device::Cpu,
            dtype: DType::F32,
            variant: StableAudioVariant::Small,
        };
        let backend = StableAudioBackend::load(cfg).await.expect("load pipeline");
        assert_eq!(backend.id(), "stable-audio");
        assert_eq!(backend.provider_kind(), "music");
    }

    #[test]
    fn stream_chunk_len_matches_250ms() {
        // 44_100 Hz × 2 channels × 250 ms / 1000 = 22_050 interleaved
        // samples per chunk. Bit-exact: any change to the constants
        // (sample rate, channel count, chunk duration) must update this
        // assertion in lockstep.
        let chunk_len = ((STABLE_AUDIO_SAMPLE_RATE
            * u32::from(STABLE_AUDIO_CHANNELS)
            * STREAM_CHUNK_MS)
            / 1000) as usize;
        assert_eq!(chunk_len, 22_050);
    }

    /// Live streaming smoke test — needs actual safetensors weights + T5
    /// tokenizer downloaded. Set `BLAZEN_STABLE_AUDIO_DIR` to a
    /// directory containing both files to enable.
    #[tokio::test]
    #[ignore = "requires stable-audio-open-small weights + tokenizer; set BLAZEN_STABLE_AUDIO_DIR to enable"]
    async fn stream_generate_music_emits_chunks_with_final_flag() {
        use futures_util::StreamExt;

        let Ok(dir) = std::env::var("BLAZEN_STABLE_AUDIO_DIR") else {
            return;
        };
        let dir = PathBuf::from(dir);
        let cfg = StableAudioConfig {
            hf_repo: StableAudioVariant::Small.hf_repo().to_string(),
            local_weights_path: Some(dir.join("model.safetensors")),
            tokenizer_path: dir.join("tokenizer.json"),
            device: Device::Cpu,
            dtype: DType::F32,
            variant: StableAudioVariant::Small,
        };
        let backend = StableAudioBackend::load(cfg).await.expect("load pipeline");
        let mut stream = backend
            .stream_generate_music("ambient pad", 2.0)
            .await
            .expect("stream constructed");

        let mut total = 0usize;
        let mut chunk_count = 0usize;
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            let c = item.expect("chunk ok");
            total += c.samples.len();
            chunk_count += 1;
            if c.is_final {
                assert!(!saw_final, "more than one is_final=true chunk emitted");
                saw_final = true;
            }
        }
        assert!(saw_final, "stream ended without is_final=true");
        assert!(
            chunk_count >= 2,
            "expected multiple chunks for 2s @ 250ms, got {chunk_count}"
        );

        let expected_samples =
            (STABLE_AUDIO_SAMPLE_RATE as usize) * (STABLE_AUDIO_CHANNELS as usize) * 2;
        // 10% tolerance — VAE upsample is exact at 512× the latent
        // length, but `latent_frames_for` rounds up so the total can
        // exceed the requested duration by a fraction of a frame.
        let tolerance = (STABLE_AUDIO_SAMPLE_RATE as usize) * (STABLE_AUDIO_CHANNELS as usize) / 10;
        assert!(
            total.abs_diff(expected_samples) <= tolerance,
            "total samples {total} too far from expected {expected_samples} (tolerance {tolerance})"
        );
    }
}
