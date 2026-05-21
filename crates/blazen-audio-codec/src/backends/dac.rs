//! Descript Audio Codec (DAC) wrapper.
//!
//! Wraps [`candle_transformers::models::dac::Model`] with a friendlier
//! API: load the model config + safetensors weights from Hugging Face Hub
//! on demand, then `encode_pcm` / `decode_tokens` against `&[f32]` /
//! `&[u32]` slices instead of raw candle tensors.
//!
//! ## Canonical checkpoint
//!
//! The default repo is [`descript/dac_44khz`] — 44.1 kHz mono, 9 codebooks
//! of 1024 entries at the 8 kbps reference bandwidth. This is the only
//! published Descript checkpoint whose `[2, 4, 8, 8]` downsampling rates
//! match the hard-coded architecture that `candle_transformers::models::dac`
//! ships on 0.10.x. The 24 kHz checkpoint
//! ([`descript/dac_24khz`]) uses `[2, 4, 5, 8]` strides and will fail to
//! load against the upstream `Model::new` until candle parameterises the
//! encoder/decoder rates.
//!
//! ## Encode is not yet supported upstream
//!
//! `candle_transformers::models::dac::Model` exposes
//! [`decode_codes`](candle_transformers::models::dac::Model::decode_codes)
//! but **no** public encode-to-codes path: both the residual vector
//! quantiser's projection layers and its codebook embedding are private
//! fields, so external crates cannot run nearest-neighbour quantisation
//! against the quantiser without forking the upstream module. Until
//! candle exposes an `encode` helper, [`DacBackend::encode_pcm`] returns
//! [`CodecError::NotYetImplemented`] with a pointer to this limitation.
//! Decode is fully wired through.
//!
//! [`descript/dac_44khz`]: https://huggingface.co/descript/dac_44khz
//! [`descript/dac_24khz`]: https://huggingface.co/descript/dac_24khz

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::dac as upstream;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::error::{CodecError, Result};
use crate::traits::CodecBackend;

// ---------------------------------------------------------------------------
// Config / device helpers
// ---------------------------------------------------------------------------

/// User-facing config for the [`DacBackend`] wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DacConfig {
    /// Hugging Face Hub repo identifier
    /// (default: [`descript/dac_44khz`](https://huggingface.co/descript/dac_44khz)).
    pub repo_id: String,
    /// Optional Hugging Face revision (branch, tag, or commit SHA).
    /// `None` uses the repo's default branch.
    pub revision: Option<String>,
    /// Safetensors filename inside the repo (default: `model.safetensors`).
    pub weights_filename: String,
    /// Config filename inside the repo (default: `config.json`).
    pub config_filename: String,
    /// Force CPU even when CUDA/Metal features are enabled.
    pub cpu_only: bool,
    /// Optional override for the hf-hub cache directory. `None` falls back
    /// to the default cache (`~/.cache/huggingface/hub`).
    pub cache_dir: Option<PathBuf>,
}

impl Default for DacConfig {
    fn default() -> Self {
        Self {
            repo_id: "descript/dac_44khz".to_string(),
            revision: None,
            weights_filename: "model.safetensors".to_string(),
            config_filename: "config.json".to_string(),
            cpu_only: false,
            cache_dir: None,
        }
    }
}

/// Pick the best available candle [`Device`] honouring [`DacConfig::cpu_only`].
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
// HF config-json shim
// ---------------------------------------------------------------------------

/// Subset of fields published in `descript/dac_*` `config.json` files that
/// we need to construct the candle [`upstream::Config`].
///
/// Hugging Face's transformers-format config uses different field names
/// than candle's struct, so we deserialise into this shim and then map
/// across.
#[derive(Debug, Clone, Deserialize)]
struct HfDacConfig {
    sampling_rate: u32,
    n_codebooks: usize,
    codebook_size: usize,
    hidden_size: usize,
    #[serde(default = "default_hop_length")]
    hop_length: u32,
}

const fn default_hop_length() -> u32 {
    512
}

impl HfDacConfig {
    /// Convert the on-disk HF config into the candle-internal [`upstream::Config`].
    ///
    /// `model_bitrate` is informational only — the upstream `Model::new`
    /// does not consume it for graph construction, just stores it. We
    /// derive it from `n_codebooks * frame_rate * 10` bits (≈10 bits per
    /// 1024-entry codebook) to keep the field honest.
    fn into_candle(self) -> upstream::Config {
        let frame_rate = self.sampling_rate / self.hop_length;
        // Safe to clamp — n_codebooks is at most 32 for any published DAC
        // checkpoint, well below u32::MAX. We saturate defensively in case
        // a future config ships a pathological value rather than panic.
        let codebooks_u32 = u32::try_from(self.n_codebooks).unwrap_or(u32::MAX);
        let model_bitrate = codebooks_u32.saturating_mul(frame_rate).saturating_mul(10);
        upstream::Config {
            num_codebooks: self.n_codebooks,
            model_bitrate,
            codebook_size: self.codebook_size,
            latent_dim: self.hidden_size,
            frame_rate,
            sampling_rate: self.sampling_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Loaded DAC model + its device + the config it was built from.
struct LoadedDac {
    inner: upstream::Model,
    device: Device,
    sample_rate: u32,
    num_codebooks: usize,
}

/// DAC backend. Cheap to construct (no I/O); the model is lazily loaded
/// on the first `encode_pcm` / `decode_tokens` / `load` call.
pub struct DacBackend {
    id: String,
    config: DacConfig,
    loaded: Arc<OnceCell<LoadedDac>>,
}

impl std::fmt::Debug for DacBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DacBackend")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("loaded", &self.loaded.initialized())
            .finish()
    }
}

impl DacBackend {
    /// Construct a new backend with the given config. Does not load weights.
    #[must_use]
    pub fn new(config: DacConfig) -> Self {
        let id = format!("dac:{}", config.repo_id);
        Self {
            id,
            config,
            loaded: Arc::new(OnceCell::new()),
        }
    }

    /// Construct using the default `descript/dac_44khz` config — the only
    /// published Descript checkpoint whose architecture matches the
    /// upstream candle hard-coded stride layout.
    #[must_use]
    pub fn default_44khz() -> Self {
        Self::new(DacConfig::default())
    }

    /// Borrow the config.
    #[must_use]
    pub fn config(&self) -> &DacConfig {
        &self.config
    }

    /// Get the model's sample rate. Returns `None` until the model is
    /// loaded.
    #[must_use]
    pub fn sample_rate_loaded(&self) -> Option<u32> {
        self.loaded.get().map(|m| m.sample_rate)
    }

    /// Get the model's codebook count. Returns `None` until the model is
    /// loaded.
    #[must_use]
    pub fn num_codebooks_loaded(&self) -> Option<usize> {
        self.loaded.get().map(|m| m.num_codebooks)
    }

    /// Lazily load weights from the Hugging Face Hub.
    async fn ensure_loaded(&self) -> Result<&LoadedDac> {
        self.loaded
            .get_or_try_init(|| async { self.load_inner().await })
            .await
    }

    async fn load_inner(&self) -> Result<LoadedDac> {
        let repo = self.config.repo_id.clone();
        let revision = self.config.revision.clone();
        let weights_filename = self.config.weights_filename.clone();
        let config_filename = self.config.config_filename.clone();
        let cache_dir = self.config.cache_dir.clone();

        // hf-hub's async API is callback-heavy; the canonical pattern in
        // candle-examples is to spawn-blocking the sync API instead.
        let (weights_path, config_path) =
            tokio::task::spawn_blocking(move || -> Result<(PathBuf, PathBuf)> {
                let mut builder = hf_hub::api::sync::ApiBuilder::new();
                if let Some(dir) = cache_dir {
                    builder = builder.with_cache_dir(dir);
                }
                let api = builder.build().map_err(|e| CodecError::HfHub {
                    repo: repo.clone(),
                    source: std::io::Error::other(e.to_string()),
                })?;
                let model_repo = match revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        repo.clone(),
                        hf_hub::RepoType::Model,
                        rev,
                    )),
                    None => api.model(repo.clone()),
                };
                let weights = model_repo
                    .get(&weights_filename)
                    .map_err(|e| CodecError::HfHub {
                        repo: repo.clone(),
                        source: std::io::Error::other(e.to_string()),
                    })?;
                let cfg = model_repo
                    .get(&config_filename)
                    .map_err(|e| CodecError::HfHub {
                        repo,
                        source: std::io::Error::other(e.to_string()),
                    })?;
                Ok((weights, cfg))
            })
            .await
            .map_err(|e| CodecError::other(format!("blocking task join failed: {e}")))??;

        // Parse the HF-format config.json and map into candle's struct.
        let config_bytes = std::fs::read(&config_path).map_err(CodecError::Io)?;
        let hf_config: HfDacConfig = serde_json::from_slice(&config_bytes).map_err(|e| {
            CodecError::other(format!(
                "failed to parse DAC config.json at {}: {e}",
                config_path.display()
            ))
        })?;
        let sample_rate = hf_config.sampling_rate;
        let num_codebooks = hf_config.n_codebooks;
        let candle_cfg = hf_config.into_candle();

        let device = pick_device(self.config.cpu_only);

        // SAFETY: candle's `from_mmaped_safetensors` requires `unsafe`
        // because the safetensors file must outlive the mmap and the
        // file contents must not change underneath us. We pass a PathBuf
        // rooted in the hf-hub cache whose contents are immutable by
        // convention.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                .map_err(CodecError::from)?
        };
        let inner = upstream::Model::new(&candle_cfg, vb).map_err(CodecError::from)?;

        Ok(LoadedDac {
            inner,
            device,
            sample_rate,
            num_codebooks,
        })
    }
}

#[async_trait]
impl AudioBackend for DacBackend {
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
impl CodecBackend for DacBackend {
    /// Encode mono PCM samples into discrete DAC codebook tokens.
    ///
    /// **Not yet implemented.** `candle_transformers::models::dac::Model`
    /// on 0.10.x exposes [`decode_codes`](upstream::Model::decode_codes)
    /// but no public encode path: the residual vector quantiser's
    /// `in_proj` / `out_proj` / `codebook` fields are private, so external
    /// crates cannot run nearest-neighbour quantisation against the
    /// quantiser without forking the upstream module. We surface this as
    /// [`CodecError::NotYetImplemented`] rather than silently returning
    /// garbage; once upstream exposes an `encode` helper this method
    /// will be wired through without changing the public surface.
    async fn encode_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<u32>> {
        if samples.is_empty() {
            return Err(CodecError::invalid_input("PCM input is empty"));
        }
        // Eagerly load + validate the sample rate so the caller sees a
        // recognisable rate-mismatch error before hitting the encode gap.
        let model = self.ensure_loaded().await?;
        if sample_rate != model.sample_rate {
            return Err(CodecError::invalid_input(format!(
                "expected sample rate {} Hz, got {} Hz -- resample first",
                model.sample_rate, sample_rate
            )));
        }
        Err(CodecError::not_yet_implemented(
            "DAC encode is not yet exposed by candle_transformers::models::dac \
             (the residual vector quantiser's projections and codebook are \
             private fields on 0.10.x); decode_tokens works fully. Track \
             huggingface/candle upstream or contribute an encode_to_codes \
             helper to unblock encode_pcm.",
        ))
    }

    /// Decode flat row-major codebook tokens back into mono PCM samples.
    ///
    /// `tokens` must be `[codebook_0_t0, codebook_0_t1, ..., codebook_1_t0, ...]`
    /// with `num_codebooks` equal to the model's quantiser count
    /// (9 for `descript/dac_44khz` at the default 8 kbps bandwidth).
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
        if num_codebooks != model.num_codebooks {
            return Err(CodecError::invalid_input(format!(
                "expected num_codebooks {} (from loaded model), got {num_codebooks}",
                model.num_codebooks
            )));
        }

        let seqlen = tokens.len() / num_codebooks;
        // Candle's quantizer expects [batch, codebooks, seqlen] integer
        // codes; use U32 to match `decode_code`'s embedding lookup
        // signature.
        let codes = Tensor::from_slice(tokens, (1, num_codebooks, seqlen), &model.device)
            .map_err(CodecError::from)?;
        let audio = model.inner.decode_codes(&codes).map_err(CodecError::from)?;
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
        // Returns the *loaded* sample rate when available, else the
        // default-checkpoint rate (44.1 kHz for `descript/dac_44khz`).
        self.sample_rate_loaded().unwrap_or(44_100)
    }

    fn num_codebooks(&self) -> usize {
        // DAC 44.1 kHz at 8 kbps ships 9 quantiser codebooks. Returns the
        // *loaded* count when available, else 9.
        self.num_codebooks_loaded().unwrap_or(9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_44khz_repo() {
        let cfg = DacConfig::default();
        assert_eq!(cfg.repo_id, "descript/dac_44khz");
        assert_eq!(cfg.weights_filename, "model.safetensors");
        assert_eq!(cfg.config_filename, "config.json");
        assert!(cfg.revision.is_none());
        assert!(!cfg.cpu_only);
        assert!(cfg.cache_dir.is_none());
    }

    #[test]
    fn default_44khz_constructor_matches_default_config() {
        let backend = DacBackend::default_44khz();
        assert_eq!(backend.config().repo_id, "descript/dac_44khz");
        // No weights loaded yet — sample_rate / num_codebooks fall back
        // to the upstream default values.
        assert_eq!(CodecBackend::sample_rate(&backend), 44_100);
        assert_eq!(CodecBackend::num_codebooks(&backend), 9);
        assert!(backend.sample_rate_loaded().is_none());
        assert!(backend.num_codebooks_loaded().is_none());
    }

    #[test]
    fn id_includes_repo_name() {
        let backend = DacBackend::default_44khz();
        assert_eq!(backend.id(), "dac:descript/dac_44khz");
        assert_eq!(backend.provider_kind(), "codec");
    }

    #[test]
    fn custom_repo_round_trips_through_config() {
        let cfg = DacConfig {
            repo_id: "descript/dac_24khz".to_string(),
            revision: Some("main".to_string()),
            ..Default::default()
        };
        let backend = DacBackend::new(cfg);
        assert_eq!(backend.id(), "dac:descript/dac_24khz");
        assert_eq!(backend.config().revision.as_deref(), Some("main"));
    }

    #[test]
    fn hf_config_maps_into_candle_config() {
        let raw = serde_json::json!({
            "sampling_rate": 44100,
            "n_codebooks": 9,
            "codebook_size": 1024,
            "hidden_size": 1024,
            "hop_length": 512
        });
        let hf: HfDacConfig = serde_json::from_value(raw).unwrap();
        let candle = hf.into_candle();
        assert_eq!(candle.sampling_rate, 44_100);
        assert_eq!(candle.num_codebooks, 9);
        assert_eq!(candle.codebook_size, 1024);
        assert_eq!(candle.latent_dim, 1024);
        assert_eq!(candle.frame_rate, 44_100 / 512);
        // bitrate = codebooks * frame_rate * ~10 bits per 1024-entry codebook
        assert_eq!(candle.model_bitrate, 9 * (44_100 / 512) * 10);
    }

    #[tokio::test]
    async fn encode_rejects_empty_pcm() {
        let backend = DacBackend::default_44khz();
        let err = backend.encode_pcm(&[], 44_100).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_misaligned_tokens() {
        let backend = DacBackend::default_44khz();
        // 5 tokens cannot be reshaped as 9 codebooks × n.
        let err = backend
            .decode_tokens(&[1, 2, 3, 4, 5], 9)
            .await
            .unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("multiple")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_zero_codebooks() {
        let backend = DacBackend::default_44khz();
        let err = backend.decode_tokens(&[1, 2, 3], 0).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("num_codebooks")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn is_loaded_starts_false() {
        let backend = DacBackend::default_44khz();
        assert!(!AudioBackend::is_loaded(&backend).await);
    }

    #[tokio::test]
    async fn provider_kind_dispatches_through_audio_backend_trait() {
        // Confirm AudioBackend trait dispatch works for downstream
        // dynamic-dispatch consumers (Arc<dyn AudioBackend>).
        let backend: Arc<dyn AudioBackend> = Arc::new(DacBackend::default_44khz());
        assert_eq!(backend.provider_kind(), "codec");
        assert!(backend.id().starts_with("dac:"));
    }

    // Live-models test: round-trip a 1-second sine wave at 44.1 kHz
    // through the real `descript/dac_44khz` checkpoint. Gated because
    // it fetches ~300 MB of safetensors weights on first run.
    //
    // NOTE: encode is not yet implemented upstream, so this test only
    // exercises decode against a deterministic zero-token stream — the
    // shape and finite-ness of the output are checked. A full
    // encode→decode round-trip will be added once candle exposes an
    // encode_to_codes API.
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn live_round_trip_sine_wave_24khz() {
        let backend = DacBackend::default_44khz();
        backend
            .load()
            .await
            .expect("load default 44 kHz DAC weights");

        let num_codebooks = backend
            .num_codebooks_loaded()
            .expect("loaded codebook count");
        let sample_rate = backend.sample_rate_loaded().expect("loaded sample rate");
        assert_eq!(sample_rate, 44_100);
        assert_eq!(num_codebooks, 9);

        // Encode-to-codes is not yet supported. Once candle exposes an
        // encode helper, replace the zero-token fixture below with a
        // genuine encode of a 1-second 440 Hz sine wave and assert the
        // decoded length matches the input length within ±frame_rate.
        let one_second = vec![0.1_f32; 44_100];
        let err = backend.encode_pcm(&one_second, 44_100).await.unwrap_err();
        assert!(matches!(err, CodecError::NotYetImplemented(_)));

        // Decode a short zero-token stream as a smoke test for the
        // decode_codes wiring. ~75 frames ≈ 1 second of audio in the
        // 44 kHz checkpoint's frame space.
        let seqlen = 86; // 44100 / 512 ≈ 86 frames per second
        let tokens = vec![0_u32; seqlen * num_codebooks];
        let pcm = backend
            .decode_tokens(&tokens, num_codebooks)
            .await
            .expect("decode zero-token stream");
        assert!(!pcm.is_empty(), "decoded PCM must not be empty");
        for s in &pcm {
            assert!(s.is_finite(), "decoded sample must be finite, got {s}");
        }
        // Decoded length should be approximately seqlen * hop_length
        // (= 86 * 512 ≈ 44032 samples for a 1-second window).
        let expected = seqlen * 512;
        let tolerance = 512;
        let diff = pcm.len().abs_diff(expected);
        assert!(
            diff <= tolerance,
            "decoded len {} differs from expected {expected} by more than {tolerance}",
            pcm.len()
        );
    }
}
