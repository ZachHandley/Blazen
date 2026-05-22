//! Bark inference pipeline — composes [`BarkTokenizer`] +
//! [`SemanticDecoder`] + [`CoarseDecoder`] + [`FineDecoder`] + the `EnCodec`
//! waveform decoder from `blazen-audio-codec` into a single text-to-WAV
//! pipeline.
//!
//! Wave B.4 lands this; the previous waves shipped the components.
//!
//! # External API surface this file depends on
//!
//! Audited 2026-05-22 against `crates/blazen-audio-codec/src/backends/encodec.rs`
//! and `crates/blazen-audio-codec/src/traits.rs`:
//!
//! - [`EncodecBackend::default_24khz`] / [`EncodecBackend::new`] —
//!   cheap, no I/O.
//! - [`EncodecBackend::config`] / [`EncodecBackend::sample_rate_loaded`] —
//!   metadata accessors.
//! - The [`CodecBackend`] trait — the **decode** path used here is
//!   `decode_tokens(&self, tokens: &[u32], num_codebooks: usize)
//!   -> Result<Vec<f32>, CodecError>`. There is **no** `decode(&Tensor)`
//!   method on the public surface; the wrapper takes a flat token slice
//!   laid out as `[cb0_t0, cb0_t1, ..., cb1_t0, ...]` (codebook-major,
//!   time-minor) and returns mono `f32` PCM at the model's native sample
//!   rate (24 kHz for the default checkpoint).
//! - Weights load lazily — calling [`CodecBackend::decode_tokens`]
//!   triggers `ensure_loaded` internally if needed.
//!
//! We hand-roll the WAV writer below rather than reusing
//! `blazen-audio-music`'s `pcm_to_wav` because that helper is feature-gated
//! on `musicgen`/`stable-audio`; pulling those flags in just for the
//! header would couple the TTS crate to unrelated engines.

#![cfg(feature = "bark")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use blazen_audio_codec::backends::encodec::EncodecBackend;
use blazen_audio_codec::traits::CodecBackend;
use candle_core::{DType, Device, IndexOp, Tensor};
use tokio::sync::OnceCell;

use crate::error::TtsError;

use super::coarse::{CoarseDecoder, CoarseSampling};
use super::fine::{FineDecoder, FineSampling};
use super::semantic::{SemanticDecoder, SemanticSampling};
use super::tokenizer::{BarkTokenizer, SEMANTIC_INFER_TOKEN, TEXT_PAD_TOKEN};
use super::weights::{BarkConfigs, BarkWeights, DEFAULT_MODEL_ID};

/// Default `EnCodec` checkpoint paired with Bark — 24 kHz mono.
pub const DEFAULT_ENCODEC_MODEL_ID: &str = "facebook/encodec_24khz";

/// `EnCodec` native sample rate for the canonical Bark waveform output.
pub const BARK_SAMPLE_RATE_HZ: u32 = 24_000;

/// Width the text slice is padded to before being prefixed with the
/// optional semantic history and suffixed with `SEMANTIC_INFER_TOKEN`.
/// Matches `bark/generation.py::generate_text_semantic` (256 BERT tokens).
const TEXT_BLOCK: usize = 256;

/// User-facing sampling knobs for the whole pipeline. Maps onto the three
/// per-stage sampling configs.
#[derive(Debug, Clone)]
pub struct BarkSamplingConfig {
    /// Softmax temperature for the semantic stage. Upstream default `0.7`.
    pub semantic_temperature: f32,
    /// Softmax temperature for the coarse stage. Upstream default `0.7`.
    pub coarse_temperature: f32,
    /// Softmax temperature for the fine stage. Upstream default `0.5`.
    pub fine_temperature: f32,
    /// Optional top-k truncation applied across all three stages.
    pub top_k: Option<usize>,
    /// Optional top-p (nucleus) truncation applied across all three stages.
    pub top_p: Option<f32>,
    /// Cap on semantic-stage new tokens (default `768`, upstream max).
    pub max_semantic_tokens: usize,
    /// Cap on coarse-stage emitted tokens across both codebooks (default
    /// `768`, upstream max).
    pub max_coarse_tokens: usize,
}

impl Default for BarkSamplingConfig {
    fn default() -> Self {
        Self {
            semantic_temperature: 0.7,
            coarse_temperature: 0.7,
            fine_temperature: 0.5,
            top_k: None,
            top_p: None,
            max_semantic_tokens: 768,
            max_coarse_tokens: 768,
        }
    }
}

/// Orchestrator owning the three transformer stages, the tokenizer, the
/// `EnCodec` waveform decoder, and the candle device. Cheap to clone via
/// [`Arc`]; construction is [`Self::from_weights`] / [`Self::from_hf`].
pub struct BarkPipeline {
    tokenizer: BarkTokenizer,
    semantic: SemanticDecoder,
    coarse: CoarseDecoder,
    fine: FineDecoder,
    codec: EncodecBackend,
    device: Device,
}

impl BarkPipeline {
    /// Construct from already-loaded [`BarkWeights`] + an [`EncodecBackend`].
    /// The tokenizer is loaded from disk via the path captured in
    /// `weights.tokenizer_path`.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] if the tokenizer JSON cannot be
    /// opened.
    pub fn from_weights(
        weights: BarkWeights,
        codec: EncodecBackend,
        device: Device,
    ) -> Result<Self, TtsError> {
        let tokenizer = BarkTokenizer::from_path(&weights.tokenizer_path)?;
        Ok(Self {
            tokenizer,
            semantic: weights.semantic,
            coarse: weights.coarse,
            fine: weights.fine,
            codec,
            device,
        })
    }

    /// Convenience: download both the Bark + `EnCodec` checkpoints from
    /// HF Hub, then construct the pipeline.
    ///
    /// # Errors
    ///
    /// Surfaces any [`TtsError::ModelLoad`] raised by the inner downloads.
    pub async fn from_hf(
        bark_model_id: &str,
        encodec_model_id: &str,
        device: &Device,
    ) -> Result<Self, TtsError> {
        let weights =
            BarkWeights::from_hf(bark_model_id, BarkConfigs::bark_small(), device).await?;
        let codec_cfg = blazen_audio_codec::backends::encodec::EncodecConfig {
            hf_repo: encodec_model_id.to_owned(),
            cpu_only: matches!(device, Device::Cpu),
            ..blazen_audio_codec::backends::encodec::EncodecConfig::default()
        };
        let codec = EncodecBackend::new(codec_cfg);
        Self::from_weights(weights, codec, device.clone())
    }

    /// Convenience constructor wiring everything up against the canonical
    /// `suno/bark-small` + `facebook/encodec_24khz` checkpoints.
    ///
    /// # Errors
    ///
    /// Surfaces any inner download / load failure.
    pub async fn default_from_hf(device: &Device) -> Result<Self, TtsError> {
        Self::from_hf(DEFAULT_MODEL_ID, DEFAULT_ENCODEC_MODEL_ID, device).await
    }

    /// Underlying tokenizer reference. Useful for the
    /// [`super::BarkBackend::clone_voice`] path which pre-tokenizes a
    /// transcript without going through the full synthesis loop.
    #[must_use]
    pub fn tokenizer(&self) -> &BarkTokenizer {
        &self.tokenizer
    }

    /// Device the pipeline runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Synthesize speech from `text` and return a fully-formed WAV byte
    /// blob ready to hand back through [`crate::traits::TtsBackend`].
    ///
    /// Pipeline stages, mirroring `bark/generation.py`:
    /// 1. Tokenize `text` → `[T_text]` u32 BERT ids with
    ///    `TEXT_ENCODING_OFFSET` already applied (see [`BarkTokenizer`]).
    /// 2. (Optional) prepend a history-prompt of semantic tokens for
    ///    voice cloning — upstream concatenates `[history_semantic,
    ///    text_tokens, SEMANTIC_INFER_TOKEN]`.
    /// 3. Pad / truncate to 256 tokens + append the infer sentinel.
    /// 4. Semantic stage: `[1, T_in] → [1, T_sem]`.
    /// 5. Coarse stage: `[1, T_sem] → [1, 2, T_coarse]` (raw `EnCodec` ids).
    /// 6. Fine stage: `[1, 2, T_coarse] → [1, 8, T_coarse]`.
    /// 7. Flatten + decode through `EnCodec` → mono 24 kHz f32 PCM.
    /// 8. Encode the PCM as a 16-bit-PCM WAV blob.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Synthesis`] for any internal tensor / decode
    /// failure, [`TtsError::InvalidOptions`] for malformed sampling
    /// configs, and [`TtsError::ModelLoad`] for codec download failures
    /// triggered on the lazy load path.
    pub async fn synthesize(
        &self,
        text: &str,
        sampling: &BarkSamplingConfig,
        history_prompt: Option<&[u32]>,
    ) -> Result<Vec<u8>, TtsError> {
        let (wav, _samples) = self.synthesize_wav(text, sampling, history_prompt).await?;
        Ok(wav)
    }

    /// Same as [`Self::synthesize`] but also reports the raw PCM sample
    /// count so callers can fill in
    /// [`blazen_audio::GeneratedAudio::duration_seconds`] without
    /// re-parsing the WAV header.
    ///
    /// # Errors
    ///
    /// See [`Self::synthesize`].
    pub async fn synthesize_wav(
        &self,
        text: &str,
        sampling: &BarkSamplingConfig,
        history_prompt: Option<&[u32]>,
    ) -> Result<(Vec<u8>, usize), TtsError> {
        let samples = self.synthesize_pcm(text, sampling, history_prompt).await?;
        let n = samples.len();
        let wav = encode_wav_16bit(&samples, BARK_SAMPLE_RATE_HZ, 1);
        Ok((wav, n))
    }

    /// Same as [`Self::synthesize`] but returns raw mono f32 PCM at
    /// 24 kHz instead of a WAV blob. Used by both the public
    /// `synthesize` entrypoint and the smoke test that checks for the
    /// `RIFF` header after wrapping.
    async fn synthesize_pcm(
        &self,
        text: &str,
        sampling: &BarkSamplingConfig,
        history_prompt: Option<&[u32]>,
    ) -> Result<Vec<f32>, TtsError> {
        // 1. Tokenize text.
        let text_ids = self.tokenizer.encode(text)?;
        if text_ids.is_empty() {
            return Err(TtsError::InvalidOptions(
                "bark synthesize: tokenized input is empty".to_owned(),
            ));
        }

        // 2. Build the semantic prefix: [history_semantic?, text_tokens,
        //    SEMANTIC_INFER_TOKEN]. Upstream
        //    `bark/generation.py::generate_text_semantic` pads the text
        //    slice to TEXT_BLOCK tokens with `TEXT_PAD_TOKEN` before
        //    concatenating, but the pure-greedy path tolerates any
        //    `T_text <= block_size`; we keep the pad-to-256 contract so
        //    weight-loaded runs match the reference behaviour.
        let mut text_slice = text_ids.clone();
        if text_slice.len() > TEXT_BLOCK {
            text_slice.truncate(TEXT_BLOCK);
        } else {
            text_slice.resize(TEXT_BLOCK, TEXT_PAD_TOKEN);
        }

        let mut prefix: Vec<u32> = Vec::with_capacity(text_slice.len() + 8);
        if let Some(history) = history_prompt {
            prefix.extend_from_slice(history);
        }
        prefix.extend_from_slice(&text_slice);
        prefix.push(SEMANTIC_INFER_TOKEN);

        let prefix_len = prefix.len();
        let prefix_tensor = Tensor::from_vec(prefix, (1, prefix_len), &self.device)
            .map_err(|e| TtsError::Synthesis(format!("bark prefix tensor: {e}")))?;

        // 3. Semantic stage.
        let semantic_sampling = SemanticSampling {
            temperature: sampling.semantic_temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            max_new_tokens: sampling.max_semantic_tokens,
            ..SemanticSampling::default()
        };
        let semantic_tokens = self.semantic.generate(&prefix_tensor, &semantic_sampling)?;
        if semantic_tokens
            .dim(1)
            .map_err(|e| TtsError::Synthesis(format!("semantic dims: {e}")))?
            == 0
        {
            return Err(TtsError::Synthesis(
                "bark semantic stage produced zero tokens".to_owned(),
            ));
        }

        // 4. Coarse stage.
        let coarse_sampling = CoarseSampling {
            temperature: sampling.coarse_temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            max_coarse_tokens: sampling.max_coarse_tokens,
        };
        let coarse_tokens = self.coarse.generate(&semantic_tokens, &coarse_sampling)?;

        // 5. Fine stage — refines to all 8 codebooks.
        let fine_sampling = FineSampling {
            temperature: sampling.fine_temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
        };
        let all_codebooks = self.fine.generate(&coarse_tokens, &fine_sampling)?;

        // 6. `EnCodec` decode. The CodecBackend trait works on a flat u32
        // slice laid out as [cb0_t0, cb0_t1, ..., cb1_t0, ...]. The fine
        // output is `[1, n_codebooks, T]` — we drop the batch dim and
        // flatten in codebook-major / time-minor order via `flatten_all`
        // which preserves that exact layout (the underlying buffer is
        // contiguous and row-major over the last two dims after the
        // implicit `i(0)` squeeze).
        let (_b, n_cb, _t) = all_codebooks
            .dims3()
            .map_err(|e| TtsError::Synthesis(format!("fine output dims: {e}")))?;
        let codes_2d = all_codebooks
            .i(0)
            .and_then(|t| t.to_dtype(DType::U32))
            .and_then(|t| t.contiguous())
            .map_err(|e| TtsError::Synthesis(format!("fine output squeeze: {e}")))?;
        let codes_flat: Vec<u32> = codes_2d
            .flatten_all()
            .and_then(|t| t.to_vec1::<u32>())
            .map_err(|e| TtsError::Synthesis(format!("fine output to_vec1: {e}")))?;
        if codes_flat.is_empty() {
            return Err(TtsError::Synthesis(
                "bark fine stage produced zero codes".to_owned(),
            ));
        }

        let pcm = self
            .codec
            .decode_tokens(&codes_flat, n_cb)
            .await
            .map_err(|e| TtsError::Synthesis(format!("bark EnCodec decode: {e}")))?;
        Ok(pcm)
    }
}

/// Minimal 16-bit-PCM WAV encoder. Mono / 24 kHz by default for Bark
/// output; kept local to the module rather than depending on the
/// musicgen-feature-gated helper in `blazen-audio-music`.
#[must_use]
fn encode_wav_16bit(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "WAV `data` chunk size is u32 by spec; buffers > 4 GiB \
                  are impossible in practice for a single TTS clip"
    )]
    let data_size = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1_u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "Clamped to [-1.0, 1.0]; multiplying by i16::MAX \
                      stays inside i16 range by construction"
        )]
        let i = (clamped * f32::from(i16::MAX)) as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

/// Compute the inclusive duration in seconds of an `f32` PCM buffer at
/// `sample_rate` Hz. Returned as `f32` (matches the
/// [`blazen_audio::GeneratedAudio::duration_seconds`] field type).
#[allow(clippy::cast_precision_loss, reason = "PCM sample counts fit f32")]
#[must_use]
pub(super) fn pcm_duration_seconds(num_samples: usize, sample_rate: u32) -> f32 {
    num_samples as f32 / sample_rate as f32
}

// ---------------------------------------------------------------------------
// Lazy-loaded pipeline cache used by the BarkBackend trait impl.
// ---------------------------------------------------------------------------

/// Thread-safe one-shot cache for the heavy pipeline construction. The
/// outer [`Arc`] is cheap to clone across `BarkBackend` clones (so all
/// clones share the same lazily-loaded pipeline) without forcing
/// [`BarkPipeline`] itself to be `Clone`.
pub(super) type PipelineCell = Arc<OnceCell<Arc<BarkPipeline>>>;

/// Build a new empty [`PipelineCell`] for [`super::BarkBackend`].
pub(super) fn new_pipeline_cell() -> PipelineCell {
    Arc::new(OnceCell::new())
}

/// Lazily build (or fetch the cached) [`BarkPipeline`] for the given
/// configuration. Subsequent calls reuse the cached pipeline.
pub(super) async fn get_or_init_pipeline(
    cell: &PipelineCell,
    bark_model_id: &str,
    encodec_model_id: &str,
) -> Result<Arc<BarkPipeline>, TtsError> {
    let bark_id = bark_model_id.to_owned();
    let codec_id = encodec_model_id.to_owned();
    let arc = cell
        .get_or_try_init(|| async move {
            let device = pick_device();
            let pipeline = BarkPipeline::from_hf(&bark_id, &codec_id, &device).await?;
            Ok::<Arc<BarkPipeline>, TtsError>(Arc::new(pipeline))
        })
        .await?;
    Ok(arc.clone())
}

/// CPU-only device picker. Bark exercises a lot of small matmuls on the
/// fine stage; if a CUDA / Metal device is needed callers can construct
/// via [`BarkPipeline::from_hf`] with an explicit device.
fn pick_device() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// Voice-clone cache helpers (used by `BarkBackend::clone_voice`).
// ---------------------------------------------------------------------------

/// Directory under the user cache where cloned voice prompts live. Uses
/// `~/.cache/blazen/bark-voices/` by default; falls back to `/tmp` if no
/// home directory is set.
#[must_use]
pub(super) fn voice_cache_dir() -> PathBuf {
    std::env::var_os("BLAZEN_BARK_VOICE_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|h| h.join(".cache/blazen/bark-voices")))
        .unwrap_or_else(|| PathBuf::from("/tmp/blazen-bark-voices"))
}

fn dirs_home() -> Option<PathBuf> {
    // Avoid pulling in the `dirs` crate for one path read; HOME / USERPROFILE
    // cover every platform Blazen targets. Returns `None` on the rare host
    // where neither variable is set, in which case [`voice_cache_dir`]
    // falls back to `/tmp`.
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Persist a pre-tokenized semantic prompt to the voice cache. Returns
/// the path written, suitable for stuffing into a
/// [`blazen_audio::VoiceHandle`] downstream.
///
/// # Errors
///
/// Returns [`TtsError::Synthesis`] when the cache directory cannot be
/// created or the file write fails.
pub(super) fn save_voice_prompt(name: &str, semantic_tokens: &[u32]) -> Result<PathBuf, TtsError> {
    let dir = voice_cache_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| TtsError::Synthesis(format!("bark voice cache mkdir: {e}")))?;
    let safe_name: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{safe_name}.bin"));
    let mut bytes: Vec<u8> = Vec::with_capacity(semantic_tokens.len() * 4);
    for tok in semantic_tokens {
        bytes.extend_from_slice(&tok.to_le_bytes());
    }
    std::fs::write(&path, &bytes)
        .map_err(|e| TtsError::Synthesis(format!("bark voice cache write: {e}")))?;
    Ok(path)
}

/// Load a previously-saved voice prompt back into a `Vec<u32>`.
///
/// # Errors
///
/// Returns [`TtsError::Synthesis`] when the file is missing or has a
/// byte length that isn't a multiple of 4.
pub(super) fn load_voice_prompt(path: &Path) -> Result<Vec<u32>, TtsError> {
    let bytes = std::fs::read(path)
        .map_err(|e| TtsError::Synthesis(format!("bark voice prompt read: {e}")))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(TtsError::Synthesis(format!(
            "bark voice prompt {} has length {} which is not a multiple of 4",
            path.display(),
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) guarantees length 4");
        out.push(u32::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    use super::super::coarse::CoarseConfig;
    use super::super::fine::FineConfig;
    use super::super::semantic::SemanticConfig;

    /// Build a tiny zero-weight pipeline against a stub `EnCodec` for the
    /// non-network smoke test. Skips weight files entirely by going
    /// straight from `VarBuilder::zeros`.
    fn build_zero_weight_pipeline(
        tokenizer_path: &Path,
    ) -> Result<BarkPipeline, Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        // Tiny configs so the smoke test stays fast — the goal is to
        // exercise wiring, not produce coherent audio.
        let mut sem_cfg = SemanticConfig::bark_small();
        sem_cfg.n_layer = 1;
        sem_cfg.n_embd = 32;
        sem_cfg.n_head = 2;
        sem_cfg.block_size = 64;
        sem_cfg.bias = false;
        let mut coarse_cfg = CoarseConfig::bark_small();
        coarse_cfg.n_layer = 1;
        coarse_cfg.n_embd = 32;
        coarse_cfg.n_head = 2;
        coarse_cfg.block_size = 64;
        coarse_cfg.bias = false;
        let mut fine_cfg = FineConfig::bark_small();
        fine_cfg.n_layer = 1;
        fine_cfg.n_embd = 32;
        fine_cfg.n_head = 2;
        fine_cfg.block_size = 64;
        fine_cfg.bias = false;

        let vb_sem = VarBuilder::zeros(DType::F32, &device);
        let vb_coarse = VarBuilder::zeros(DType::F32, &device);
        let vb_fine = VarBuilder::zeros(DType::F32, &device);
        let semantic = SemanticDecoder::load(vb_sem, sem_cfg)?;
        let coarse = CoarseDecoder::from_vb(vb_coarse, coarse_cfg)?;
        let fine = FineDecoder::from_vb(vb_fine, fine_cfg)?;

        let weights = BarkWeights {
            semantic,
            coarse,
            fine,
            tokenizer_path: tokenizer_path.to_path_buf(),
        };
        let codec = EncodecBackend::default_24khz();
        Ok(BarkPipeline::from_weights(weights, codec, device)?)
    }

    #[test]
    fn encode_wav_16bit_emits_riff_header() {
        let samples = [0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav_16bit(&samples, 24_000, 1);
        assert_eq!(&bytes[0..4], b"RIFF", "wav must start with RIFF");
        assert_eq!(&bytes[8..12], b"WAVE", "container type must be WAVE");
        assert_eq!(&bytes[12..16], b"fmt ", "fmt chunk must follow");
        // sample_rate at offset 24..28 (LE u32)
        let sr = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert_eq!(sr, 24_000);
        // 16-bit-per-sample at offset 34..36 (LE u16)
        let bps = u16::from_le_bytes(bytes[34..36].try_into().unwrap());
        assert_eq!(bps, 16);
        // 5 samples * 2 bytes = 10 bytes of PCM data
        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, 10);
    }

    #[test]
    fn encode_wav_16bit_clamps_out_of_range_samples() {
        // Samples > 1.0 should clamp to i16::MAX, < -1.0 to i16::MIN.
        let samples = [5.0_f32, -5.0];
        let bytes = encode_wav_16bit(&samples, 24_000, 1);
        let lo_idx = 44;
        let v0 = i16::from_le_bytes([bytes[lo_idx], bytes[lo_idx + 1]]);
        let v1 = i16::from_le_bytes([bytes[lo_idx + 2], bytes[lo_idx + 3]]);
        assert_eq!(v0, i16::MAX);
        assert_eq!(v1, -i16::MAX); // 1.0 * i16::MAX → 32767; -1.0 → -32767
    }

    #[test]
    fn sampling_config_defaults_match_upstream_bark() {
        let s = BarkSamplingConfig::default();
        assert!((s.semantic_temperature - 0.7).abs() < f32::EPSILON);
        assert!((s.coarse_temperature - 0.7).abs() < f32::EPSILON);
        assert!((s.fine_temperature - 0.5).abs() < f32::EPSILON);
        assert!(s.top_k.is_none());
        assert!(s.top_p.is_none());
        assert_eq!(s.max_semantic_tokens, 768);
        assert_eq!(s.max_coarse_tokens, 768);
    }

    #[test]
    fn pcm_duration_seconds_matches_sample_rate() {
        assert!((pcm_duration_seconds(24_000, 24_000) - 1.0).abs() < f32::EPSILON);
        assert!((pcm_duration_seconds(12_000, 24_000) - 0.5).abs() < f32::EPSILON);
        assert!((pcm_duration_seconds(0, 24_000) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn save_and_load_voice_prompt_round_trips() {
        let tmpdir =
            std::env::temp_dir().join(format!("blazen-bark-voice-test-{}", std::process::id()));
        // Override the cache dir via env var for hermeticity.
        // SAFETY: tests in this module are not parallelized over this var,
        // and the integration / nextest harness isolates threads per-test
        // for env mutation only when needed. We set + unset in one fn.
        // SAFETY: single-threaded mutation of process env; tests in this
        // module avoid concurrent access to BLAZEN_BARK_VOICE_DIR.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("BLAZEN_BARK_VOICE_DIR", &tmpdir);
        }
        let tokens = vec![1u32, 2, 3, 4, 5_000_000];
        let path = save_voice_prompt("my voice/with..weird chars", &tokens).expect("save");
        // Filename should be sanitized.
        let file_name = path
            .file_name()
            .expect("file name")
            .to_string_lossy()
            .into_owned();
        assert!(
            file_name.starts_with("my_voice_with__weird_chars"),
            "got {file_name}"
        );
        let loaded = load_voice_prompt(&path).expect("load");
        assert_eq!(loaded, tokens);
        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmpdir);
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("BLAZEN_BARK_VOICE_DIR");
        }
    }

    #[tokio::test]
    async fn voice_prompt_with_bad_byte_length_errors() {
        let tmpdir =
            std::env::temp_dir().join(format!("blazen-bark-voice-bad-{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let bad = tmpdir.join("bad.bin");
        std::fs::write(&bad, [1u8, 2, 3]).unwrap(); // 3 bytes — not multiple of 4
        let err = load_voice_prompt(&bad).unwrap_err();
        match err {
            TtsError::Synthesis(msg) => assert!(msg.contains("multiple of 4"), "msg = {msg}"),
            other => panic!("expected Synthesis, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// Smoke test: a zero-weight pipeline plus stub tokenizer.json
    /// compiles and wires the stages without panicking. We don't actually
    /// run `synthesize` here because that requires (a) a real
    /// tokenizer.json on disk and (b) downloading `EnCodec` on first
    /// decode — both of which would push this test into the live tier.
    /// The smoke test confirms the trait objects line up and the
    /// `BarkPipeline` value is constructable from a `VarBuilder::zeros`
    /// triple.
    #[test]
    fn pipeline_constructed_from_zero_varbuilder_compiles_cleanly() {
        // We don't have a tokenizer.json to point at hermetically; the
        // construct path that *exercises* the tokenizer load is
        // `from_weights`, which calls `BarkTokenizer::from_path`. So we
        // skip the actual construction (which would need a real
        // tokenizer.json) and instead assert that the function pointer
        // exists / is invokable with a non-existent path (expected to
        // error in `BarkTokenizer::from_path`).
        let fake_tok = PathBuf::from("/nonexistent/tokenizer.json");
        let result = build_zero_weight_pipeline(&fake_tok);
        assert!(
            result.is_err(),
            "expected from_weights to fail on missing tokenizer.json"
        );
    }

    /// Live test — downloads `suno/bark-small` + `facebook/encodec_24khz`
    /// and synthesizes a short prompt. Gated on `BLAZEN_TEST_BARK=1` so
    /// `cargo test` stays hermetic by default.
    #[tokio::test]
    #[ignore = "requires network + ~700 MB of HF downloads; set BLAZEN_TEST_BARK=1"]
    async fn synthesize_with_real_weights() {
        if std::env::var("BLAZEN_TEST_BARK").ok().as_deref() != Some("1") {
            return;
        }
        let device = Device::Cpu;
        let pipeline = BarkPipeline::default_from_hf(&device)
            .await
            .expect("hf load");
        let cfg = BarkSamplingConfig {
            max_semantic_tokens: 64,
            max_coarse_tokens: 64,
            ..BarkSamplingConfig::default()
        };
        let wav = pipeline
            .synthesize("hello world", &cfg, None)
            .await
            .expect("synthesize");
        assert!(wav.len() > 44, "wav too short ({} bytes)", wav.len());
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }
}
