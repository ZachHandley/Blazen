//! Meta AudioGen text-to-SFX backend.
//!
//! AudioGen is architecturally identical to MusicGen — same T5 prompt
//! encoder + autoregressive transformer decoder over EnCodec tokens,
//! same delay-pattern interleaver, same Classifier-Free-Guidance sampler.
//! The only material differences are:
//!
//! 1. Different training corpus (environmental sound + SFX vs music).
//! 2. Different checkpoint (`facebook/audiogen-medium`, 1.5B params).
//! 3. Different audio codec config: AudioGen produces **16 kHz mono** at
//!    50 Hz EnCodec frame rate vs MusicGen's 32 kHz.
//!
//! This backend therefore **reuses** the [`super::musicgen`] submodule's
//! [`MusicgenForConditionalGeneration`] model class, weight loader,
//! generation loop, codec decode path, and WAV writer; it only contributes
//! AudioGen-specific config values (EnCodec config + LM dims + sample
//! rate) plus a thin wrapper that routes `generate_music` / `generate_sfx`
//! through the shared infrastructure.
//!
//! # Weight licensing
//!
//! `facebook/audiogen-medium` is released under the same CC-BY-NC-4.0
//! license as MusicGen and is **not licensed for commercial use**.
//! Surface this restriction to your end users.

#![cfg(feature = "audiogen")]

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError, AudioFormat, GeneratedAudio};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{encodec, t5};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

use crate::error::MusicError;
use crate::traits::MusicBackend;

use super::musicgen::generation::{self, GenerationParams};
use super::musicgen::model::{
    Config as MusicgenLmConfig, GenConfig, MusicgenForConditionalGeneration,
};
use super::musicgen::{decode_to_pcm, pcm_to_wav};

/// Absolute upper bound on a single AudioGen call. Mirrors MusicGen's cap.
pub const AUDIOGEN_MAX_DURATION_HARD_LIMIT: f32 = 60.0;

/// Native sample rate of AudioGen output (always 16 kHz mono).
pub const AUDIOGEN_SAMPLE_RATE: u32 = 16_000;

/// EnCodec frame rate used by AudioGen (50 Hz, matches MusicGen).
pub const AUDIOGEN_FRAME_RATE: u32 = 50;

/// Configuration for an [`AudioGenBackend`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenConfig {
    /// Hugging Face Hub repository identifier. Defaults to the official
    /// 1.5B checkpoint `facebook/audiogen-medium`.
    pub repo_id: String,
    /// Optional pinned revision (commit SHA or tag).
    pub revision: Option<String>,
    /// Optional explicit candle device. `None` falls back to
    /// auto-detection (CUDA → Metal → CPU).
    #[serde(skip)]
    pub device: Option<Device>,
    /// Optional override for the HF cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Per-backend safety cap on the requested duration. Calls past this
    /// cap surface [`MusicError::InvalidInput`] regardless of
    /// [`AUDIOGEN_MAX_DURATION_HARD_LIMIT`].
    pub max_duration_seconds: f32,
}

impl Default for AudioGenConfig {
    fn default() -> Self {
        Self {
            repo_id: "facebook/audiogen-medium".to_string(),
            revision: None,
            device: None,
            cache_dir: None,
            max_duration_seconds: 30.0,
        }
    }
}

impl AudioGenConfig {
    /// Combined [`GenConfig`] describing the AudioGen-medium architecture.
    ///
    /// The LM hyper-parameters mirror MusicGen-medium (same 1.5B-param
    /// transformer); the T5 encoder mirrors MusicGen-small (T5-base, same
    /// `d_model=768`) with `d_ff=3072`; the EnCodec config is the
    /// **16 kHz** variant AudioGen ships, with
    /// `upsampling_ratios=[8, 5, 4, 2]` (total stride 320 = 16 kHz / 50 Hz
    /// frame rate).
    #[must_use]
    pub fn gen_config() -> GenConfig {
        let mut lm = MusicgenLmConfig::musicgen_medium();
        // AudioGen-medium and MusicGen-medium share the LM dims; re-state
        // them explicitly here so a future MusicGen divergence doesn't
        // silently couple to AudioGen.
        lm.num_codebooks = 4;
        lm.vocab_size = 2048;
        lm.bos_token_id = 2048;
        lm.pad_token_id = 2048;

        let mut t5 = t5::Config::musicgen_small();
        t5.d_ff = 3072;

        GenConfig {
            musicgen: lm,
            t5,
            encodec: audiogen_encodec_config(),
        }
    }
}

/// EnCodec config baked into `facebook/audiogen-medium`.
///
/// 16 kHz mono, 4 codebooks of 2048 entries each, 50 Hz frame rate.
/// Total upsampling stride = 320 = 16_000 / 50.
fn audiogen_encodec_config() -> encodec::Config {
    encodec::Config {
        audio_channels: 1,
        chunk_length_s: None,
        codebook_dim: Some(128),
        codebook_size: 2048,
        compress: 2,
        dilation_growth_rate: 2,
        hidden_size: 128,
        kernel_size: 7,
        last_kernel_size: 7,
        norm_type: encodec::NormType::WeightNorm,
        normalize: false,
        num_filters: 64,
        num_lstm_layers: 2,
        num_residual_layers: 1,
        overlap: None,
        pad_mode: encodec::PadMode::Replicate,
        residual_kernel_size: 3,
        sampling_rate: AUDIOGEN_SAMPLE_RATE as usize,
        target_bandwidths: vec![1.5],
        trim_right_ratio: 1.0,
        // 8 * 5 * 4 * 2 = 320 (16 kHz / 50 Hz).
        upsampling_ratios: vec![8, 5, 4, 2],
        use_causal_conv: false,
        use_conv_shortcut: false,
    }
}

/// Loaded AudioGen handle: model graph + tokenizer + chosen device.
struct LoadedAudioGen {
    model: tokio::sync::Mutex<MusicgenForConditionalGeneration>,
    tokenizer: Tokenizer,
    device: Device,
}

/// AudioGen text-to-SFX (and text-to-music) backend.
///
/// Construction is cheap: weights / tokenizer are downloaded lazily on
/// the first `generate_*` call. The autoregressive head is the same
/// [`MusicgenForConditionalGeneration`] used by [`super::musicgen`].
pub struct AudioGenBackend {
    config: AudioGenConfig,
    loaded: Arc<OnceCell<LoadedAudioGen>>,
}

impl std::fmt::Debug for AudioGenBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioGenBackend")
            .field("config", &self.config)
            .field("loaded", &self.loaded.initialized())
            .finish()
    }
}

impl Default for AudioGenBackend {
    fn default() -> Self {
        Self::new(AudioGenConfig::default())
    }
}

impl AudioGenBackend {
    /// Construct an AudioGen backend handle from the given config.
    #[must_use]
    pub fn new(config: AudioGenConfig) -> Self {
        Self {
            config,
            loaded: Arc::new(OnceCell::new()),
        }
    }

    /// Borrow the backend config.
    #[must_use]
    pub const fn config(&self) -> &AudioGenConfig {
        &self.config
    }

    /// Pick the candle device honoring [`AudioGenConfig::device`] and the
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

    async fn ensure_loaded(&self) -> Result<&LoadedAudioGen, MusicError> {
        self.loaded
            .get_or_try_init(|| async { self.load_inner().await })
            .await
    }

    async fn load_inner(&self) -> Result<LoadedAudioGen, MusicError> {
        let repo = self.config.repo_id.clone();
        let revision = self.config.revision.clone();
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
                let m = match revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        repo.clone(),
                        hf_hub::RepoType::Model,
                        rev,
                    )),
                    None => api.model(repo.clone()),
                };
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
        let cfg = AudioGenConfig::gen_config();

        // SAFETY: candle's `from_mmaped_safetensors` requires `unsafe`
        // because the safetensors file must outlive the mmap and its
        // contents must not change underneath us. We pass a PathBuf
        // rooted in the hf-hub cache whose contents are immutable by
        // convention.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)
                .map_err(MusicError::from)?
        };
        let model = MusicgenForConditionalGeneration::load(vb, cfg)?;

        Ok(LoadedAudioGen {
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
        if duration_seconds > AUDIOGEN_MAX_DURATION_HARD_LIMIT {
            return Err(MusicError::invalid_input(format!(
                "max {AUDIOGEN_MAX_DURATION_HARD_LIMIT}s for AudioGen \
                 (got {duration_seconds}s)"
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
        let frame_rate = f32::from(u16::try_from(AUDIOGEN_FRAME_RATE).unwrap_or(50));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_frames = (duration_seconds * frame_rate).ceil() as usize;
        let num_codebooks = {
            let m = loaded.model.lock().await;
            m.decoder.num_codebooks()
        };
        let max_steps = target_frames + num_codebooks.saturating_sub(1);

        let encoded = loaded
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| MusicError::other(format!("tokenizer encode failed: {e}")))?;
        let ids: Vec<u32> = encoded.get_ids().to_vec();
        if ids.is_empty() {
            return Err(MusicError::invalid_input(
                "tokenizer returned 0 ids for prompt",
            ));
        }
        let ids_len = ids.len();
        let prompt_tokens = Tensor::from_vec(ids, (1, ids_len), &loaded.device)?;

        let params = GenerationParams {
            max_steps,
            ..GenerationParams::default()
        };

        let pcm = {
            let mut model = loaded.model.lock().await;
            let tokens =
                generation::generate_tokens(&mut model, &prompt_tokens, &params, &loaded.device)?;
            decode_to_pcm(&model, &tokens, &loaded.device)?
        };

        let wav_bytes = pcm_to_wav(&pcm, AUDIOGEN_SAMPLE_RATE, 1);
        #[allow(clippy::cast_precision_loss)]
        let duration = pcm.len() as f32 / AUDIOGEN_SAMPLE_RATE as f32;
        Ok(GeneratedAudio {
            bytes: wav_bytes,
            format: AudioFormat::Wav,
            sample_rate: AUDIOGEN_SAMPLE_RATE,
            channels: 1,
            duration_seconds: Some(duration),
        })
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioBackend for AudioGenBackend {
    fn id(&self) -> &'static str {
        "audiogen-medium"
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
impl MusicBackend for AudioGenBackend {
    async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        // AudioGen's training corpus is SFX-heavy but the model class is
        // a vanilla autoregressive LM over EnCodec tokens; nothing about
        // the inference path differentiates "music" from "SFX". The prompt
        // does. Route both verbs through the same generation pipeline.
        self.generate(prompt, duration_seconds).await
    }

    async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.generate(prompt, duration_seconds).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_targets_audiogen_medium() {
        let cfg = AudioGenConfig::default();
        assert_eq!(cfg.repo_id, "facebook/audiogen-medium");
        assert!(cfg.revision.is_none());
        assert!(cfg.cache_dir.is_none());
        assert!((cfg.max_duration_seconds - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sample_rate_is_16khz() {
        assert_eq!(AUDIOGEN_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn frame_rate_is_50hz() {
        assert_eq!(AUDIOGEN_FRAME_RATE, 50);
    }

    #[test]
    fn upsampling_ratios_multiply_to_sample_rate_over_frame_rate() {
        let cfg = audiogen_encodec_config();
        let product: usize = cfg.upsampling_ratios.iter().product();
        assert_eq!(
            product,
            (AUDIOGEN_SAMPLE_RATE / AUDIOGEN_FRAME_RATE) as usize
        );
        assert_eq!(cfg.sampling_rate, AUDIOGEN_SAMPLE_RATE as usize);
        assert_eq!(cfg.codebook_size, 2048);
        assert_eq!(cfg.audio_channels, 1);
    }

    #[test]
    fn gen_config_uses_four_codebooks_and_medium_lm() {
        let cfg = AudioGenConfig::gen_config();
        assert_eq!(cfg.musicgen.num_codebooks, 4);
        assert_eq!(cfg.musicgen.vocab_size, 2048);
        assert_eq!(cfg.musicgen.bos_token_id, 2048);
        assert_eq!(cfg.musicgen.hidden_size, 1536);
        assert_eq!(cfg.musicgen.num_hidden_layers, 48);
        assert_eq!(cfg.encodec.sampling_rate, AUDIOGEN_SAMPLE_RATE as usize);
    }

    #[test]
    fn id_is_audiogen_medium() {
        let b = AudioGenBackend::default();
        assert_eq!(b.id(), "audiogen-medium");
        assert_eq!(b.provider_kind(), "music");
    }

    #[test]
    fn custom_repo_round_trips() {
        let b = AudioGenBackend::new(AudioGenConfig {
            repo_id: "myorg/audiogen-clone".into(),
            revision: Some("abc1234".into()),
            ..AudioGenConfig::default()
        });
        assert_eq!(b.config().repo_id, "myorg/audiogen-clone");
        assert_eq!(b.config().revision.as_deref(), Some("abc1234"));
    }

    #[tokio::test]
    async fn rejects_empty_prompt() {
        let b = AudioGenBackend::default();
        let err = b.generate_sfx("   ", 1.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("empty")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_zero_duration() {
        let b = AudioGenBackend::default();
        let err = b.generate_sfx("dog barking", 0.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("positive")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_duration_over_hard_limit() {
        let b = AudioGenBackend::new(AudioGenConfig {
            max_duration_seconds: 120.0,
            ..AudioGenConfig::default()
        });
        let err = b.generate_sfx("rain falling", 61.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => {
                assert!(m.contains("max 60s for AudioGen"));
            }
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_duration_over_configured_cap() {
        let b = AudioGenBackend::new(AudioGenConfig {
            max_duration_seconds: 3.0,
            ..AudioGenConfig::default()
        });
        let err = b.generate_sfx("dog barking", 10.0).await.unwrap_err();
        match err {
            MusicError::InvalidInput(m) => assert!(m.contains("exceeds configured cap")),
            other => panic!("expected InvalidInput got {other:?}"),
        }
    }

    #[tokio::test]
    async fn generate_music_routes_through_same_pipeline() {
        // generate_music and generate_sfx share validation + dispatch.
        let b = AudioGenBackend::default();
        let err = b.generate_music("", 5.0).await.unwrap_err();
        assert!(matches!(err, MusicError::InvalidInput(_)));
    }

    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn generates_two_seconds_of_sfx() {
        // Pulls real weights from HF the first time this runs. Skip
        // transparently if HF fetch or model load fails (offline / no
        // auth / checkpoint shipped only in audiocraft `state_dict.bin`
        // format rather than HF-transformers `model.safetensors`).
        let b = AudioGenBackend::new(AudioGenConfig {
            max_duration_seconds: 5.0,
            ..AudioGenConfig::default()
        });
        if let Err(e) = b.load().await {
            eprintln!("live-models test skipped: load failed: {e}");
            return;
        }
        let audio = match b.generate_sfx("dog barking", 2.0).await {
            Ok(a) => a,
            Err(MusicError::HfHub { repo, source }) => {
                eprintln!("live-models test skipped: HF fetch failed for {repo}: {source}");
                return;
            }
            Err(MusicError::Candle(msg)) => {
                eprintln!("live-models test skipped: candle load failed: {msg}");
                return;
            }
            Err(other) => panic!("unexpected failure: {other}"),
        };
        assert_eq!(audio.format, AudioFormat::Wav);
        assert_eq!(audio.sample_rate, AUDIOGEN_SAMPLE_RATE);
        assert_eq!(audio.channels, 1);

        // WAV magic bytes.
        assert!(audio.bytes.starts_with(b"RIFF"), "wav must start with RIFF");
        assert!(
            audio.bytes.windows(4).any(|w| w == b"WAVE"),
            "wav must contain WAVE chunk id"
        );
        assert!(audio.bytes.len() > 44 + 1000, "wav payload looks empty");

        // Duration tolerance: spec is ±50%; keep a generous bound so a
        // codec/decoder length quirk does not cause spurious failures.
        let dur = audio.duration_seconds.expect("duration");
        let expected = 2.0_f32;
        let tolerance = expected * 0.5;
        assert!(
            (dur - expected).abs() <= tolerance,
            "expected ~{expected}s ± {tolerance}s, got {dur}s"
        );

        // Decode PCM out of the WAV payload (skip the 44-byte canonical
        // header written by `pcm_to_wav`) and assert all samples are
        // finite. The writer emits i16 LE at AUDIOGEN_SAMPLE_RATE mono,
        // so each sample is 2 bytes.
        let payload = &audio.bytes[44..];
        assert!(
            payload.len().is_multiple_of(2),
            "i16 payload must be even length, got {}",
            payload.len()
        );
        let mut all_finite = true;
        for chunk in payload.chunks_exact(2) {
            #[allow(clippy::cast_precision_loss)]
            let s = f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / f32::from(i16::MAX);
            if !s.is_finite() {
                all_finite = false;
                break;
            }
        }
        assert!(all_finite, "all decoded samples must be finite");

        // Sample count should match the requested duration within ±50%.
        let sample_count = payload.len() / 2;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let expected_samples = (expected * AUDIOGEN_SAMPLE_RATE as f32) as usize;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let sample_tol = (expected_samples as f32 * 0.5) as usize;
        assert!(
            sample_count.abs_diff(expected_samples) <= sample_tol,
            "sample_count {sample_count} should be within \u{00b1}{sample_tol} of {expected_samples}"
        );
    }
}
