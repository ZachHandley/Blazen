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
//! reference and the previous home of this code in
//! `crates/blazen-audio-candle/src/encodec.rs`.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::encodec as upstream;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::error::{CodecError, Result};
use crate::traits::CodecBackend;

// ---------------------------------------------------------------------------
// Config / device helpers
// ---------------------------------------------------------------------------

/// User-facing config for the [`EncodecBackend`] wrapper.
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

/// EnCodec backend. Cheap to construct (no I/O); the model is lazily
/// loaded on the first `encode_pcm` / `decode_tokens` / `load` call.
pub struct EncodecBackend {
    id: String,
    config: EncodecConfig,
    loaded: Arc<OnceCell<LoadedModel>>,
}

impl std::fmt::Debug for EncodecBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodecBackend")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("loaded", &self.loaded.initialized())
            .finish()
    }
}

impl EncodecBackend {
    /// Construct a new backend with the given config. Does not load weights.
    #[must_use]
    pub fn new(config: EncodecConfig) -> Self {
        let id = format!("encodec:{}", config.hf_repo);
        Self {
            id,
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
    /// loaded.
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
            let api = builder.build().map_err(|e| CodecError::HfHub {
                repo: repo.clone(),
                source: std::io::Error::other(e.to_string()),
            })?;
            api.model(repo.clone())
                .get(&filename)
                .map_err(|e| CodecError::HfHub {
                    repo,
                    source: std::io::Error::other(e.to_string()),
                })
        })
        .await
        .map_err(|e| CodecError::other(format!("blocking task join failed: {e}")))??;

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
                .map_err(CodecError::from)?
        };
        let inner = upstream::Model::new(&cfg, vb).map_err(CodecError::from)?;

        Ok(LoadedModel {
            inner,
            device,
            sample_rate,
        })
    }
}

#[async_trait]
impl AudioBackend for EncodecBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "codec"
    }

    async fn load(&self) -> std::result::Result<(), blazen_audio::AudioError> {
        self.ensure_loaded()
            .await
            .map_err(blazen_audio::AudioError::from)?;
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        self.loaded.initialized()
    }
}

#[async_trait]
impl CodecBackend for EncodecBackend {
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
    async fn encode_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<u32>> {
        if samples.is_empty() {
            return Err(CodecError::invalid_input("PCM input is empty"));
        }
        let model = self.ensure_loaded().await?;
        if sample_rate != model.sample_rate {
            return Err(CodecError::invalid_input(format!(
                "expected sample rate {} Hz, got {} Hz -- resample first",
                model.sample_rate, sample_rate
            )));
        }

        // Shape: [1 (batch), 1 (channels), seqlen]
        let xs = Tensor::from_slice(samples, (1, 1, samples.len()), &model.device)
            .map_err(CodecError::from)?;
        let codes = model.inner.encode(&xs).map_err(CodecError::from)?;

        // `codes` shape: [batch, codebooks, seqlen]. Flatten batch + codebook +
        // seqlen into a single vector of u32 tokens.
        let codes = codes
            .i(0)
            .map_err(CodecError::from)?
            .flatten_all()
            .map_err(CodecError::from)?;
        codes.to_vec1::<u32>().map_err(CodecError::from)
    }

    /// Decode codebook tokens back into mono PCM samples.
    ///
    /// `tokens` must be a flat row-major `[codebooks * seqlen]` vector
    /// (the layout returned by [`Self::encode_pcm`]) with
    /// `num_codebooks` equal to the model's quantizer count
    /// (4 for `encodec_24khz` at the default bandwidth).
    async fn decode_tokens(&self, tokens: &[u32], num_codebooks: usize) -> Result<Vec<f32>> {
        if num_codebooks == 0 {
            return Err(CodecError::invalid_input("num_codebooks must be > 0"));
        }
        if tokens.is_empty() || !tokens.len().is_multiple_of(num_codebooks) {
            return Err(CodecError::invalid_input(format!(
                "token count {} is not a positive multiple of num_codebooks {}",
                tokens.len(),
                num_codebooks
            )));
        }
        let model = self.ensure_loaded().await?;

        let seqlen = tokens.len() / num_codebooks;
        let codes = Tensor::from_slice(tokens, (1, num_codebooks, seqlen), &model.device)
            .map_err(CodecError::from)?;
        let audio = model.inner.decode(&codes).map_err(CodecError::from)?;
        // `audio` shape: [batch, channels, seqlen]; squeeze and flatten.
        let audio = audio
            .i(0)
            .map_err(CodecError::from)?
            .i(0)
            .map_err(CodecError::from)?
            .flatten_all()
            .map_err(CodecError::from)?;
        audio.to_vec1::<f32>().map_err(CodecError::from)
    }

    fn sample_rate(&self) -> u32 {
        // Returns the *config* sample rate even when not yet loaded — the
        // upstream `Config::default()` is 24 kHz.
        self.sample_rate_loaded().unwrap_or(24_000)
    }

    fn num_codebooks(&self) -> usize {
        // EnCodec 24 kHz at the default 6 kbps target bandwidth ships 4
        // codebooks of 1024 entries each. Configurable codecs would override.
        4
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
        let backend = EncodecBackend::default_24khz();
        assert_eq!(backend.config().hf_repo, "facebook/encodec_24khz");
        // No weights loaded yet — sample_rate falls back to the upstream default.
        assert_eq!(CodecBackend::sample_rate(&backend), 24_000);
        assert_eq!(CodecBackend::num_codebooks(&backend), 4);
        assert!(backend.sample_rate_loaded().is_none());
    }

    #[test]
    fn id_includes_repo_name() {
        let backend = EncodecBackend::default_24khz();
        assert_eq!(backend.id(), "encodec:facebook/encodec_24khz");
        assert_eq!(backend.provider_kind(), "codec");
    }

    #[tokio::test]
    async fn encode_rejects_empty_pcm() {
        let backend = EncodecBackend::default_24khz();
        let err = backend.encode_pcm(&[], 24_000).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_misaligned_tokens() {
        let backend = EncodecBackend::default_24khz();
        // 5 tokens cannot be reshaped as 4 codebooks × n.
        let err = backend
            .decode_tokens(&[1, 2, 3, 4, 5], 4)
            .await
            .unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => {
                assert!(msg.contains("multiple"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_zero_codebooks() {
        let backend = EncodecBackend::default_24khz();
        let err = backend.decode_tokens(&[1, 2, 3], 0).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("num_codebooks")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn is_loaded_starts_false() {
        let backend = EncodecBackend::default_24khz();
        assert!(!AudioBackend::is_loaded(&backend).await);
    }

    // Live-models test: round-trip a 1-second sine wave at 24 kHz
    // through the real `facebook/encodec_24khz` checkpoint. Gated
    // because it fetches ~90 MB of safetensors weights on first run.
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn live_round_trip_sine_wave_24khz() {
        let backend = EncodecBackend::default_24khz();
        AudioBackend::load(&backend)
            .await
            .expect("load default 24 kHz EnCodec weights");

        let sample_rate = backend.sample_rate_loaded().expect("loaded sample rate");
        assert_eq!(sample_rate, 24_000);

        // 1 second of a 440 Hz sine wave at 24 kHz, mono, amplitude 0.5.
        // The cast-precision-loss allow is fine here: `i` runs over
        // [0, 24_000) which is well within f32's exact-integer range
        // (<= 2^23 ≈ 1.6e7).
        let len = 24_000usize;
        let mut pcm: Vec<f32> = Vec::with_capacity(len);
        for i in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 24_000.0;
            pcm.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        let codes = backend
            .encode_pcm(&pcm, 24_000)
            .await
            .expect("encode 24 kHz sine wave");
        assert!(!codes.is_empty(), "encoded codes must not be empty");

        let num_codebooks = CodecBackend::num_codebooks(&backend);
        let decoded = backend
            .decode_tokens(&codes, num_codebooks)
            .await
            .expect("decode EnCodec codes");
        assert!(!decoded.is_empty(), "decoded PCM must not be empty");
        for s in &decoded {
            assert!(s.is_finite(), "decoded sample must be finite, got {s}");
        }
        // EnCodec is windowed; small drift on either side is normal.
        let tolerance = 2048usize;
        let diff = decoded.len().abs_diff(len);
        assert!(
            diff <= tolerance,
            "decoded len {} differs from input len {len} by more than {tolerance}",
            decoded.len()
        );
    }
}
