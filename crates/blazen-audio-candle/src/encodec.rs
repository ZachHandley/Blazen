//! EnCodec neural audio codec wrapper.
//!
//! Wraps [`candle_transformers::models::encodec::Model`] with a friendlier
//! API: load weights from Hugging Face Hub on demand, then `encode_pcm` /
//! `decode_tokens` against `&[f32]` / `&[u32]` slices instead of raw
//! candle tensors.
//!
//! The crate's default checkpoint is `facebook/encodec_24khz` (24 kHz
//! mono, 4 codebooks of 1024 entries each at 6 kbps target bandwidth).
//! Both the encoder and decoder run on the device picked by
//! [`pick_device`]; CUDA / Metal are used automatically when the matching
//! cargo feature is enabled.
//!
//! Adapted from the official `candle-examples/examples/encodec/main.rs`
//! reference. See `/home/zach/.cache/blazen-musicgen-research/encodec.rs`
//! for the upstream source pulled in by `candle-transformers`.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::encodec as upstream;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::error::{CandleAudioError, Result};
use crate::model::AudioModel;

// ---------------------------------------------------------------------------
// Config / device helpers
// ---------------------------------------------------------------------------

/// User-facing config for the [`EncodecModel`] wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodecConfig {
    /// Hugging Face Hub repo identifier (default: `facebook/encodec_24khz`).
    pub hf_repo: String,
    /// Safetensors filename inside the repo (default: `model.safetensors`).
    pub weights_filename: String,
    /// Force CPU even when CUDA/Metal features are enabled.
    pub cpu_only: bool,
    /// Optional override for the hf-hub cache directory. `None` falls back
    /// to the default cache (`~/.cache/huggingface/hub`).
    pub cache_dir: Option<PathBuf>,
}

impl Default for EncodecConfig {
    fn default() -> Self {
        Self {
            hf_repo: "facebook/encodec_24khz".to_string(),
            weights_filename: "model.safetensors".to_string(),
            cpu_only: false,
            cache_dir: None,
        }
    }
}

/// Pick the best available candle [`Device`] honoring [`EncodecConfig::cpu_only`].
///
/// Falls back to [`Device::Cpu`] silently if the requested accelerator is
/// unavailable.
fn pick_device(cpu_only: bool) -> Device {
    if cpu_only {
        return Device::Cpu;
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

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Loaded EnCodec model + its device.
struct LoadedModel {
    inner: upstream::Model,
    device: Device,
    /// Sampling rate baked into the config (Hz).
    sample_rate: u32,
}

/// EnCodec wrapper. Cheap to construct (no I/O); the model is lazily
/// loaded on the first `encode_pcm` / `decode_tokens` / `generate` call.
pub struct EncodecModel {
    config: EncodecConfig,
    loaded: Arc<OnceCell<LoadedModel>>,
}

impl std::fmt::Debug for EncodecModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodecModel")
            .field("config", &self.config)
            .field("loaded", &self.loaded.initialized())
            .finish()
    }
}

impl EncodecModel {
    /// Construct a new wrapper with the given config. Does not load weights.
    #[must_use]
    pub fn new(config: EncodecConfig) -> Self {
        Self {
            config,
            loaded: Arc::new(OnceCell::new()),
        }
    }

    /// Construct using the default `facebook/encodec_24khz` config.
    #[must_use]
    pub fn default_24khz() -> Self {
        Self::new(EncodecConfig::default())
    }

    /// Borrow the config.
    #[must_use]
    pub fn config(&self) -> &EncodecConfig {
        &self.config
    }

    /// Get the model's sample rate. Returns `None` until the model is
    /// loaded (call `ensure_loaded` first if you need it eagerly).
    #[must_use]
    pub fn sample_rate_loaded(&self) -> Option<u32> {
        self.loaded.get().map(|m| m.sample_rate)
    }

    /// Lazily load weights from the Hugging Face Hub.
    async fn ensure_loaded(&self) -> Result<&LoadedModel> {
        self.loaded
            .get_or_try_init(|| async { self.load_inner().await })
            .await
    }

    async fn load_inner(&self) -> Result<LoadedModel> {
        let repo = self.config.hf_repo.clone();
        let filename = self.config.weights_filename.clone();
        let cache_dir = self.config.cache_dir.clone();

        // hf-hub's async API is callback-heavy; the canonical pattern in
        // candle-examples is to spawn-blocking the sync API instead. The
        // sync API blocks on the runtime's IO thread, so we offload it to
        // a blocking worker explicitly.
        let weights_path = tokio::task::spawn_blocking(move || -> Result<PathBuf> {
            let mut builder = hf_hub::api::sync::ApiBuilder::new();
            if let Some(dir) = cache_dir {
                builder = builder.with_cache_dir(dir);
            }
            let api = builder.build().map_err(|e| CandleAudioError::HfHub {
                repo: repo.clone(),
                source: std::io::Error::other(e.to_string()),
            })?;
            api.model(repo.clone())
                .get(&filename)
                .map_err(|e| CandleAudioError::HfHub {
                    repo,
                    source: std::io::Error::other(e.to_string()),
                })
        })
        .await
        .map_err(|e| CandleAudioError::other(format!("blocking task join failed: {e}")))??;

        let device = pick_device(self.config.cpu_only);
        let cfg = upstream::Config::default();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let sample_rate = cfg.sampling_rate as u32;

        // SAFETY: candle's `from_mmaped_safetensors` requires `unsafe`
        // because the safetensors file must outlive the mmap and the file
        // contents must not change underneath us. We pass a PathBuf rooted
        // in the hf-hub cache whose contents are immutable by convention.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                .map_err(CandleAudioError::from)?
        };
        let inner = upstream::Model::new(&cfg, vb).map_err(CandleAudioError::from)?;

        Ok(LoadedModel {
            inner,
            device,
            sample_rate,
        })
    }

    /// Encode mono PCM samples (`f32` in `[-1.0, 1.0]`) into discrete
    /// EnCodec codebook tokens.
    ///
    /// The input sample rate must match the model's native sample rate
    /// (24 kHz for the default checkpoint). Resample upstream if needed.
    ///
    /// Returns a flat token vector laid out as
    /// `[codebook_0_t0, codebook_0_t1, ..., codebook_1_t0, ...]` — the
    /// caller can reshape with `tokens.len() / num_codebooks` once they
    /// know the codebook count for their config.
    ///
    /// # Errors
    ///
    /// - [`CandleAudioError::InvalidInput`] when `samples` is empty or
    ///   `sample_rate` does not match the model's native rate.
    /// - [`CandleAudioError::HfHub`] / [`CandleAudioError::Io`] on first
    ///   call if the weights download fails.
    /// - [`CandleAudioError::Candle`] for inference failures.
    pub async fn encode_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<u32>> {
        if samples.is_empty() {
            return Err(CandleAudioError::invalid_input("PCM input is empty"));
        }
        let model = self.ensure_loaded().await?;
        if sample_rate != model.sample_rate {
            return Err(CandleAudioError::invalid_input(format!(
                "expected sample rate {} Hz, got {} Hz -- resample first",
                model.sample_rate, sample_rate
            )));
        }

        // Shape: [1 (batch), 1 (channels), seqlen]
        let xs = Tensor::from_slice(samples, (1, 1, samples.len()), &model.device)
            .map_err(CandleAudioError::from)?;
        let codes = model.inner.encode(&xs).map_err(CandleAudioError::from)?;

        // `codes` shape: [batch, codebooks, seqlen]. Flatten batch + codebook +
        // seqlen into a single vector of u32 tokens.
        let codes = codes
            .i(0)
            .map_err(CandleAudioError::from)?
            .flatten_all()
            .map_err(CandleAudioError::from)?;
        codes.to_vec1::<u32>().map_err(CandleAudioError::from)
    }

    /// Decode codebook tokens back into mono PCM samples.
    ///
    /// `tokens` must be a flat row-major `[codebooks * seqlen]` vector
    /// (the layout returned by [`encode_pcm`](Self::encode_pcm)) with
    /// `num_codebooks` equal to the model's quantizer count
    /// (4 for `encodec_24khz` at the default bandwidth).
    ///
    /// # Errors
    ///
    /// - [`CandleAudioError::InvalidInput`] when `tokens.len()` is not a
    ///   multiple of `num_codebooks`.
    /// - [`CandleAudioError::HfHub`] / [`CandleAudioError::Io`] on first
    ///   call if the weights download fails.
    /// - [`CandleAudioError::Candle`] for inference failures.
    pub async fn decode_tokens(&self, tokens: &[u32], num_codebooks: usize) -> Result<Vec<f32>> {
        if num_codebooks == 0 {
            return Err(CandleAudioError::invalid_input("num_codebooks must be > 0"));
        }
        if tokens.is_empty() || !tokens.len().is_multiple_of(num_codebooks) {
            return Err(CandleAudioError::invalid_input(format!(
                "token count {} is not a positive multiple of num_codebooks {}",
                tokens.len(),
                num_codebooks
            )));
        }
        let model = self.ensure_loaded().await?;

        let seqlen = tokens.len() / num_codebooks;
        let codes = Tensor::from_slice(tokens, (1, num_codebooks, seqlen), &model.device)
            .map_err(CandleAudioError::from)?;
        let audio = model.inner.decode(&codes).map_err(CandleAudioError::from)?;
        // `audio` shape: [batch, channels, seqlen]; squeeze and flatten.
        let audio = audio
            .i(0)
            .map_err(CandleAudioError::from)?
            .i(0)
            .map_err(CandleAudioError::from)?
            .flatten_all()
            .map_err(CandleAudioError::from)?;
        audio.to_vec1::<f32>().map_err(CandleAudioError::from)
    }
}

#[async_trait]
impl AudioModel for EncodecModel {
    fn name(&self) -> &str {
        // `&'static str` is technically more precise, but the trait pins
        // the return type to `&str` borrowed from `&self` to leave room
        // for impls (like MusicGen) that store the name as a `String`.
        #[allow(clippy::unnecessary_literal_bound)]
        {
            "encodec"
        }
    }

    fn sample_rate(&self) -> u32 {
        // Returns the *config* sample rate even when not yet loaded — the
        // upstream `Config::default()` is 24 kHz.
        self.sample_rate_loaded().unwrap_or(24_000)
    }

    /// `EnCodec` is a *codec*, not a generative model. Calling `generate` on
    /// it is a category error — the caller wants `MusicGen` or `AudioGen`.
    /// Surfaces a clear, actionable error rather than silently returning
    /// silence or noise.
    async fn generate(&self, _prompt: &str, _duration_seconds: f32) -> Result<Vec<f32>> {
        Err(CandleAudioError::not_yet_implemented(
            "EnCodec is a neural audio codec, not a text-to-audio generator -- \
             use `encode_pcm` / `decode_tokens` for codec round-tripping, or \
             pick a generative model (e.g. MusicGen, AudioGen) once those land",
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_24khz_repo() {
        let cfg = EncodecConfig::default();
        assert_eq!(cfg.hf_repo, "facebook/encodec_24khz");
        assert_eq!(cfg.weights_filename, "model.safetensors");
        assert!(!cfg.cpu_only);
        assert!(cfg.cache_dir.is_none());
    }

    #[test]
    fn default_24khz_constructor_matches_default_config() {
        let model = EncodecModel::default_24khz();
        assert_eq!(model.config().hf_repo, "facebook/encodec_24khz");
        // No weights loaded yet — sample_rate falls back to the upstream default.
        assert_eq!(model.sample_rate(), 24_000);
        assert!(model.sample_rate_loaded().is_none());
    }

    #[tokio::test]
    async fn generate_is_unsupported_on_codec() {
        let model = EncodecModel::default_24khz();
        let err = model.generate("anything", 1.0).await.unwrap_err();
        matches!(err, CandleAudioError::NotYetImplemented(_));
    }

    #[tokio::test]
    async fn encode_rejects_empty_pcm() {
        let model = EncodecModel::default_24khz();
        let err = model.encode_pcm(&[], 24_000).await.unwrap_err();
        match err {
            CandleAudioError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_misaligned_tokens() {
        let model = EncodecModel::default_24khz();
        // 5 tokens cannot be reshaped as 4 codebooks × n.
        let err = model.decode_tokens(&[1, 2, 3, 4, 5], 4).await.unwrap_err();
        match err {
            CandleAudioError::InvalidInput(msg) => {
                assert!(msg.contains("multiple"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}
