//! Multi-Scale Neural Audio Codec (SNAC) wrapper.
//!
//! Wraps [`candle_transformers::models::snac::Model`] with a friendlier
//! API: load the model config + pytorch weights from Hugging Face Hub
//! on demand, then `encode_pcm` / `decode_tokens` against `&[f32]` /
//! `&[u32]` slices instead of raw candle tensors.
//!
//! ## Canonical checkpoint
//!
//! The default repo is [`hubertsiuzdak/snac_24khz`] — 24 kHz mono,
//! **3 multi-scale codebooks** at `vq_strides = [4, 2, 1]`, codebook size
//! 4096, ~3 kbps. Other published checkpoints (`hubertsiuzdak/snac_32khz`,
//! `hubertsiuzdak/snac_44khz`) use different stride / dim combinations
//! and load through the same config-driven path — flip `repo_id` and
//! the wrapper picks up the new architecture from `config.json`.
//!
//! ## Token layout (multi-scale)
//!
//! SNAC is **not** a residual VQ stack — its quantisers operate at
//! *different* temporal strides, so the per-codebook token count
//! differs. For `vq_strides = [4, 2, 1]` and a base latent length `T`,
//! the three codebooks emit `T/4`, `T/2`, and `T` tokens respectively.
//!
//! Despite the per-codebook shape mismatch, both [`SnacBackend::encode_pcm`]
//! and [`SnacBackend::decode_tokens`] use the same flat row-major
//! layout as every other [`CodecBackend`]:
//!
//! ```text
//! [cb0_t0, cb0_t1, ..., cb0_t_{T/s0 - 1},
//!  cb1_t0, cb1_t1, ..., cb1_t_{T/s1 - 1},
//!  cb2_t0, cb2_t1, ..., cb2_t_{T/s2 - 1}]
//! ```
//!
//! The wrapper splits / concatenates against the loaded model's
//! `vq_strides` automatically. Callers do not need to know the per-codebook
//! lengths up front — they only need a `tokens.len()` that's a positive
//! multiple of the smallest valid round-trip step
//! (`sum(lcm / s_i)` for the loaded strides; 7 for the canonical
//! `[4, 2, 1]` checkpoint).
//!
//! ## Weight format
//!
//! `hubertsiuzdak/snac_*` only ships `pytorch_model.bin` (pickle). We
//! load via [`candle_nn::VarBuilder::from_pth`]; no safetensors
//! conversion is required.
//!
//! [`hubertsiuzdak/snac_24khz`]: https://huggingface.co/hubertsiuzdak/snac_24khz

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::snac as upstream;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::error::{CodecError, Result};
use crate::traits::CodecBackend;

// ---------------------------------------------------------------------------
// Config / device helpers
// ---------------------------------------------------------------------------

/// User-facing config for the [`SnacBackend`] wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnacConfig {
    /// Hugging Face Hub repo identifier
    /// (default: [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz)).
    pub repo_id: String,
    /// Optional Hugging Face revision (branch, tag, or commit SHA).
    /// `None` uses the repo's default branch.
    pub revision: Option<String>,
    /// Pickle weight filename inside the repo (default: `pytorch_model.bin`).
    /// SNAC checkpoints have not been republished as safetensors on the
    /// Hub, so the wrapper loads via `VarBuilder::from_pth`.
    pub weights_filename: String,
    /// Config filename inside the repo (default: `config.json`).
    pub config_filename: String,
    /// Force CPU even when CUDA/Metal features are enabled.
    pub cpu_only: bool,
    /// Optional override for the hf-hub cache directory. `None` falls back
    /// to the default cache (`~/.cache/huggingface/hub`).
    pub cache_dir: Option<PathBuf>,
}

impl Default for SnacConfig {
    fn default() -> Self {
        Self {
            repo_id: "hubertsiuzdak/snac_24khz".to_string(),
            revision: None,
            weights_filename: "pytorch_model.bin".to_string(),
            config_filename: "config.json".to_string(),
            cpu_only: false,
            cache_dir: None,
        }
    }
}

/// Pick the best available candle [`Device`] honouring [`SnacConfig::cpu_only`].
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
// Multi-scale token layout helpers
// ---------------------------------------------------------------------------

/// Greatest common divisor (Euclid's algorithm). Used only to derive the
/// least-common-multiple of the loaded model's `vq_strides`.
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Least-common-multiple of all entries in `strides`. Panics on empty
/// input (the wrapper only invokes this with the loaded model's
/// non-empty `vq_strides`).
fn lcm_all(strides: &[usize]) -> usize {
    let mut acc = strides[0];
    for &s in &strides[1..] {
        acc = acc / gcd(acc, s) * s;
    }
    acc
}

/// Per-codebook token counts for a SNAC stream of `base_len` latent
/// frames. `strides[i]` is the temporal stride of codebook `i`; the
/// codebook emits `base_len / strides[i]` tokens.
///
/// `base_len` MUST be a multiple of [`lcm_all`]`(strides)` for the
/// division to be exact.
fn per_codebook_lens(strides: &[usize], base_len: usize) -> Vec<usize> {
    strides.iter().map(|s| base_len / *s).collect()
}

/// Total tokens emitted across all codebooks for `base_len` latent frames.
fn total_tokens(strides: &[usize], base_len: usize) -> usize {
    per_codebook_lens(strides, base_len).iter().sum()
}

/// Given the loaded model's `strides` and a flat token count, recover
/// the base latent length `T` such that
/// `total_tokens(strides, T) == token_count`.
///
/// Returns `None` if `token_count` does not correspond to any valid
/// integer `T` (i.e. is not a positive multiple of the smallest valid
/// step size).
fn base_len_from_token_count(strides: &[usize], token_count: usize) -> Option<usize> {
    if token_count == 0 {
        return None;
    }
    let lcm = lcm_all(strides);
    let step = total_tokens(strides, lcm);
    if !token_count.is_multiple_of(step) {
        return None;
    }
    Some((token_count / step) * lcm)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Loaded SNAC model + its device + the config it was built from.
struct LoadedSnac {
    inner: upstream::Model,
    device: Device,
    sample_rate: u32,
    num_codebooks: usize,
    vq_strides: Vec<usize>,
}

/// SNAC backend. Cheap to construct (no I/O); the model is lazily loaded
/// on the first `encode_pcm` / `decode_tokens` / `load` call.
pub struct SnacBackend {
    id: String,
    config: SnacConfig,
    loaded: Arc<OnceCell<LoadedSnac>>,
}

impl std::fmt::Debug for SnacBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnacBackend")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("loaded", &self.loaded.initialized())
            .finish()
    }
}

impl SnacBackend {
    /// Construct a new backend with the given config. Does not load weights.
    #[must_use]
    pub fn new(config: SnacConfig) -> Self {
        let id = format!("snac:{}", config.repo_id);
        Self {
            id,
            config,
            loaded: Arc::new(OnceCell::new()),
        }
    }

    /// Construct using the default `hubertsiuzdak/snac_24khz` config.
    #[must_use]
    pub fn default_24khz() -> Self {
        Self::new(SnacConfig::default())
    }

    /// Borrow the config.
    #[must_use]
    pub fn config(&self) -> &SnacConfig {
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

    /// Get the loaded model's per-codebook `vq_strides`. Returns `None`
    /// until the model is loaded.
    #[must_use]
    pub fn vq_strides_loaded(&self) -> Option<&[usize]> {
        self.loaded.get().map(|m| m.vq_strides.as_slice())
    }

    /// Lazily load weights from the Hugging Face Hub.
    async fn ensure_loaded(&self) -> Result<&LoadedSnac> {
        self.loaded
            .get_or_try_init(|| async { self.load_inner().await })
            .await
    }

    async fn load_inner(&self) -> Result<LoadedSnac> {
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

        // SNAC's published `config.json` uses field names that match
        // `upstream::Config` verbatim, so a direct serde deserialise
        // round-trips without an intermediate shim.
        let config_bytes = std::fs::read(&config_path).map_err(CodecError::Io)?;
        let candle_cfg: upstream::Config = serde_json::from_slice(&config_bytes).map_err(|e| {
            CodecError::other(format!(
                "failed to parse SNAC config.json at {}: {e}",
                config_path.display()
            ))
        })?;

        // Defensive sanity-check — SNAC's quantiser counts on the trait
        // bound `num_codebooks() > 0`, and empty `vq_strides` would
        // trip a division-by-zero in our layout helpers.
        if candle_cfg.vq_strides.is_empty() {
            return Err(CodecError::other(
                "SNAC config.json has empty vq_strides; checkpoint is malformed",
            ));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let sample_rate = candle_cfg.sampling_rate as u32;
        let num_codebooks = candle_cfg.vq_strides.len();
        let vq_strides = candle_cfg.vq_strides.clone();

        let device = pick_device(self.config.cpu_only);

        // SNAC weights ship as pickle (`pytorch_model.bin`) on the Hub,
        // so use `VarBuilder::from_pth` rather than the safetensors mmap
        // path used by EnCodec / DAC.
        let vb =
            VarBuilder::from_pth(&weights_path, DType::F32, &device).map_err(CodecError::from)?;
        let inner = upstream::Model::new(&candle_cfg, vb).map_err(CodecError::from)?;

        Ok(LoadedSnac {
            inner,
            device,
            sample_rate,
            num_codebooks,
            vq_strides,
        })
    }
}

#[async_trait]
impl AudioBackend for SnacBackend {
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
impl CodecBackend for SnacBackend {
    /// Encode mono PCM samples into discrete SNAC codebook tokens.
    ///
    /// `sample_rate` MUST match the loaded model's native rate (24 kHz
    /// for `hubertsiuzdak/snac_24khz`); resample upstream otherwise.
    ///
    /// Returns a flat row-major token vector with all of codebook 0,
    /// then codebook 1, then codebook 2 — see the module-level docs for
    /// the exact layout and per-codebook lengths.
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

        // Shape: [1 (batch), 1 (channels), seqlen]. SNAC's `preprocess`
        // hook right-pads to a multiple of `hop_length * lcm(vq_strides)`
        // internally, so we don't need to round the input length up here.
        let xs = Tensor::from_slice(samples, (1, 1, samples.len()), &model.device)
            .map_err(CodecError::from)?;
        let codes_per_cb = model.inner.encode(&xs).map_err(CodecError::from)?;

        // `codes_per_cb` is a `Vec<Tensor>` with one tensor per codebook,
        // each shaped `[batch, t / stride_i]` (u32 indices from
        // `argmin`). Flatten into the public row-major
        // `[cb0_all, cb1_all, ...]` layout.
        let mut flat = Vec::with_capacity(
            codes_per_cb
                .iter()
                .try_fold(0usize, |acc, t| Ok::<_, CodecError>(acc + t.elem_count()))?,
        );
        for codes in &codes_per_cb {
            let codes_vec = codes
                .i(0)
                .map_err(CodecError::from)?
                .flatten_all()
                .map_err(CodecError::from)?
                .to_vec1::<u32>()
                .map_err(CodecError::from)?;
            flat.extend_from_slice(&codes_vec);
        }
        Ok(flat)
    }

    /// Decode multi-scale codebook tokens back into mono PCM samples.
    ///
    /// `tokens` must be the flat row-major
    /// `[cb0_all, cb1_all, ..., cb{N-1}_all]` vector produced by
    /// [`Self::encode_pcm`], with `num_codebooks` equal to the loaded
    /// model's `vq_strides.len()` (3 for `hubertsiuzdak/snac_24khz`).
    ///
    /// `tokens.len()` MUST be a positive multiple of the multi-scale
    /// step size (`sum(lcm / s_i)`; 7 for the canonical `[4, 2, 1]`
    /// strides). Mismatches surface as
    /// [`CodecError::InvalidInput`].
    async fn decode_tokens(&self, tokens: &[u32], num_codebooks: usize) -> Result<Vec<f32>> {
        if num_codebooks == 0 {
            return Err(CodecError::invalid_input("num_codebooks must be > 0"));
        }
        if tokens.is_empty() {
            return Err(CodecError::invalid_input("tokens input is empty"));
        }
        let model = self.ensure_loaded().await?;
        if num_codebooks != model.num_codebooks {
            return Err(CodecError::invalid_input(format!(
                "expected num_codebooks {} (from loaded model), got {num_codebooks}",
                model.num_codebooks
            )));
        }

        // Recover the base latent length T from the flat token count
        // using the loaded model's vq_strides.
        let base_len =
            base_len_from_token_count(&model.vq_strides, tokens.len()).ok_or_else(|| {
                let lcm = lcm_all(&model.vq_strides);
                let step = total_tokens(&model.vq_strides, lcm);
                CodecError::invalid_input(format!(
                    "token count {} is not a positive multiple of the SNAC multi-scale step {step} \
                     (vq_strides = {:?}, lcm = {lcm})",
                    tokens.len(),
                    model.vq_strides,
                ))
            })?;

        // Split the flat vector into per-codebook tensors of shape
        // `[1, t / stride_i]` and pass them to `Model::decode`.
        let lens = per_codebook_lens(&model.vq_strides, base_len);
        let mut tensors = Vec::with_capacity(num_codebooks);
        let mut offset = 0usize;
        for len in &lens {
            let slice = &tokens[offset..offset + *len];
            let tensor =
                Tensor::from_slice(slice, (1, *len), &model.device).map_err(CodecError::from)?;
            tensors.push(tensor);
            offset += *len;
        }
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let audio = model.inner.decode(&refs).map_err(CodecError::from)?;

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
        // default-checkpoint rate (24 kHz for `hubertsiuzdak/snac_24khz`).
        self.sample_rate_loaded().unwrap_or(24_000)
    }

    fn num_codebooks(&self) -> usize {
        // SNAC ships 3 multi-scale codebooks at every published rate.
        // Returns the *loaded* count when available, else 3.
        self.num_codebooks_loaded().unwrap_or(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_24khz_repo() {
        let cfg = SnacConfig::default();
        assert_eq!(cfg.repo_id, "hubertsiuzdak/snac_24khz");
        assert_eq!(cfg.weights_filename, "pytorch_model.bin");
        assert_eq!(cfg.config_filename, "config.json");
        assert!(cfg.revision.is_none());
        assert!(!cfg.cpu_only);
        assert!(cfg.cache_dir.is_none());
    }

    #[test]
    fn default_24khz_constructor_matches_default_config() {
        let backend = SnacBackend::default_24khz();
        assert_eq!(backend.config().repo_id, "hubertsiuzdak/snac_24khz");
        // No weights loaded yet — sample_rate / num_codebooks fall back
        // to the upstream default values.
        assert_eq!(CodecBackend::sample_rate(&backend), 24_000);
        assert_eq!(CodecBackend::num_codebooks(&backend), 3);
        assert!(backend.sample_rate_loaded().is_none());
        assert!(backend.num_codebooks_loaded().is_none());
        assert!(backend.vq_strides_loaded().is_none());
    }

    #[test]
    fn id_includes_repo_name() {
        let backend = SnacBackend::default_24khz();
        assert_eq!(backend.id(), "snac:hubertsiuzdak/snac_24khz");
        assert_eq!(backend.provider_kind(), "codec");
    }

    #[test]
    fn custom_repo_round_trips_through_config() {
        let cfg = SnacConfig {
            repo_id: "hubertsiuzdak/snac_32khz".to_string(),
            revision: Some("main".to_string()),
            ..Default::default()
        };
        let backend = SnacBackend::new(cfg);
        assert_eq!(backend.id(), "snac:hubertsiuzdak/snac_32khz");
        assert_eq!(backend.config().revision.as_deref(), Some("main"));
    }

    #[test]
    fn lcm_all_matches_known_snac_strides() {
        assert_eq!(lcm_all(&[4, 2, 1]), 4);
        assert_eq!(lcm_all(&[8, 4, 2, 1]), 8);
        assert_eq!(lcm_all(&[3, 5]), 15);
    }

    #[test]
    fn per_codebook_lens_split_evenly_at_step_boundary() {
        let strides = vec![4, 2, 1];
        // base_len = 4 (one lcm step) -> [1, 2, 4]
        assert_eq!(per_codebook_lens(&strides, 4), vec![1, 2, 4]);
        // base_len = 8 -> [2, 4, 8]
        assert_eq!(per_codebook_lens(&strides, 8), vec![2, 4, 8]);
    }

    #[test]
    fn total_tokens_for_canonical_24khz_strides() {
        // For [4, 2, 1], the per-step token count is 1 + 2 + 4 = 7.
        assert_eq!(total_tokens(&[4, 2, 1], 4), 7);
        assert_eq!(total_tokens(&[4, 2, 1], 8), 14);
        assert_eq!(total_tokens(&[4, 2, 1], 12), 21);
    }

    #[test]
    fn base_len_from_token_count_round_trips_for_valid_inputs() {
        let strides = vec![4, 2, 1];
        // 7 -> base_len 4; 14 -> 8; 21 -> 12.
        assert_eq!(base_len_from_token_count(&strides, 7), Some(4));
        assert_eq!(base_len_from_token_count(&strides, 14), Some(8));
        assert_eq!(base_len_from_token_count(&strides, 21), Some(12));
    }

    #[test]
    fn base_len_from_token_count_rejects_non_step_multiples() {
        let strides = vec![4, 2, 1];
        // 7 is the smallest valid step; 6, 8, 13 are not multiples of 7.
        assert_eq!(base_len_from_token_count(&strides, 6), None);
        assert_eq!(base_len_from_token_count(&strides, 8), None);
        assert_eq!(base_len_from_token_count(&strides, 13), None);
        // Zero tokens is never valid.
        assert_eq!(base_len_from_token_count(&strides, 0), None);
    }

    #[tokio::test]
    async fn encode_rejects_empty_pcm() {
        let backend = SnacBackend::default_24khz();
        let err = backend.encode_pcm(&[], 24_000).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_empty_tokens() {
        let backend = SnacBackend::default_24khz();
        let err = backend.decode_tokens(&[], 3).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decode_rejects_zero_codebooks() {
        let backend = SnacBackend::default_24khz();
        let err = backend.decode_tokens(&[1, 2, 3], 0).await.unwrap_err();
        match err {
            CodecError::InvalidInput(msg) => assert!(msg.contains("num_codebooks")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn is_loaded_starts_false() {
        let backend = SnacBackend::default_24khz();
        assert!(!AudioBackend::is_loaded(&backend).await);
    }

    #[tokio::test]
    async fn provider_kind_dispatches_through_audio_backend_trait() {
        // Confirm AudioBackend trait dispatch works for downstream
        // dynamic-dispatch consumers (Arc<dyn AudioBackend>).
        let backend: Arc<dyn AudioBackend> = Arc::new(SnacBackend::default_24khz());
        assert_eq!(backend.provider_kind(), "codec");
        assert!(backend.id().starts_with("snac:"));
    }

    // Live-models test: round-trip a 1-second sine wave at 24 kHz
    // through the real `hubertsiuzdak/snac_24khz` checkpoint. Gated
    // because it fetches ~80 MB of pickle weights on first run.
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn live_round_trip_sine_wave_24khz() {
        let backend = SnacBackend::default_24khz();
        backend
            .load()
            .await
            .expect("load default 24 kHz SNAC weights");

        let num_codebooks = backend
            .num_codebooks_loaded()
            .expect("loaded codebook count");
        let sample_rate = backend.sample_rate_loaded().expect("loaded sample rate");
        let strides = backend
            .vq_strides_loaded()
            .expect("loaded vq_strides")
            .to_vec();
        assert_eq!(sample_rate, 24_000);
        assert_eq!(num_codebooks, 3);
        assert_eq!(strides, vec![4, 2, 1]);

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
        let step = total_tokens(&strides, lcm_all(&strides));
        assert!(
            codes.len().is_multiple_of(step),
            "encoded token count {} should be a multiple of the SNAC step {step}",
            codes.len()
        );

        let decoded = backend
            .decode_tokens(&codes, num_codebooks)
            .await
            .expect("decode SNAC codes");
        assert!(!decoded.is_empty(), "decoded PCM must not be empty");
        for s in &decoded {
            assert!(s.is_finite(), "decoded sample must be finite, got {s}");
        }
        // Decoded length should be approximately the input length
        // (SNAC right-pads the encoder input to the next multiple of
        // `hop_length * lcm(vq_strides)`; the canonical 24 kHz
        // checkpoint has hop_length = 512 and lcm = 4, so the tolerance
        // is at most 2048 samples on either side).
        let tolerance = 4096usize;
        let diff = decoded.len().abs_diff(len);
        assert!(
            diff <= tolerance,
            "decoded len {} differs from input len {len} by more than {tolerance}",
            decoded.len()
        );
    }
}
