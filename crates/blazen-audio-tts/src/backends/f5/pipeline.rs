//! End-to-end F5-TTS synthesis pipeline.
//!
//! Wave F.4 wires the components landed in F.1-F.3 into a single
//! text → 24 kHz WAV path:
//!
//! 1. Tokenize the target text with [`F5Tokenizer`].
//! 2. Build a target mel shape `(B, T_mel, mel_dim)` from a duration
//!    heuristic (1.5 mel-frames per character, with a 64-frame
//!    floor).
//! 3. Initialise `x_0` as Gaussian noise of that shape inside
//!    [`super::sampler::sample`].
//! 4. Run the Euler flow-matching sampler for [`F5Sampling::n_steps`]
//!    steps, invoking [`F5Dit::forward`] each step with
//!    classifier-free guidance.
//! 5. Decode the predicted mel through [`VocosBackbone`] +
//!    [`super::vocos::istft`] into a 24 kHz waveform.
//! 6. Encode the waveform as a 16-bit PCM WAV (mono / 24 kHz).
//!
//! # Why the text encoder isn't wired yet
//!
//! Upstream F5-TTS embeds text via a small `nn.Embedding` +
//! [`conv_layers`](super::dit_wrapper::F5DitConfig::conv_layers)
//! `ConvNeXtV2` stack before the `DiT` consumes it. The standalone
//! text-encoder module is intentionally out of scope for the Wave
//! F.4 commit (it lives in a follow-up wave); to keep the pipeline
//! callable end-to-end we synthesise a **zero** `text_embed` tensor
//! of the expected `(B, T_mel, text_dim)` shape. This is enough to
//! exercise the wiring + shape contract; with real weights and a
//! real text encoder the same call site stays unchanged.
//!
//! Voice-clone reference audio is similarly accepted in the API but
//! routed through the same zero-conditioning path until the F5 audio
//! encoder lands.

#![cfg(feature = "f5-tts")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use tokio::sync::OnceCell;

use crate::error::TtsError;

use super::dit_wrapper::{F5Dit, F5DitConfig};
use super::sampler::{F5Sampling, sample};
use super::tokenizer::F5Tokenizer;
use super::vocos::{VocosBackbone, VocosConfig, istft};
use super::weights::F5Weights;

/// Vocos default output sample rate.
pub const F5_SAMPLE_RATE_HZ: u32 = 24_000;

/// Approximate ratio of mel-spectrogram frames to input characters
/// used by upstream `infer_process` (50 mel-frames per second / 32
/// characters per second of speech).
const MEL_FRAMES_PER_CHAR_NUM: usize = 50;
const MEL_FRAMES_PER_CHAR_DEN: usize = 32;
/// Floor on the synthesised mel length so the iSTFT has enough
/// frames to overlap-add.
const MIN_MEL_FRAMES: usize = 64;

/// End-to-end F5-TTS synthesis pipeline.
///
/// Owns the tokenizer, the `DiT` velocity-predictor, the Vocos vocoder
/// (+ its config for [`istft`]), the sampler config, and the candle
/// device. Cheap to wrap in [`Arc`] for cross-clone sharing — the
/// internal modules borrow weights immutably during `forward` calls.
pub struct F5Pipeline {
    tokenizer: F5Tokenizer,
    dit: F5Dit,
    vocos: VocosBackbone,
    vocos_cfg: VocosConfig,
    sampling: F5Sampling,
    device: Device,
}

impl std::fmt::Debug for F5Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Surface a summary instead of the full module weights — the
        // candle modules don't impl Debug usefully and dumping a
        // gigabyte of tensors at trace time would be useless. Members
        // omitted from the debug surface are intentional.
        f.debug_struct("F5Pipeline")
            .field("vocab_size", &self.tokenizer.vocab_size())
            .field("dit_depth", &self.dit.config().depth)
            .field("vocos_layers", &self.vocos.config().n_layers)
            .field("sampling_steps", &self.sampling.n_steps)
            .field("device", &format_args!("{:?}", self.device))
            .finish_non_exhaustive()
    }
}

impl F5Pipeline {
    /// Construct from already-loaded weights + tokenizer.
    ///
    /// # Errors
    ///
    /// Surfaces tokenizer / vocab-load errors from
    /// [`F5Tokenizer::from_vocab_path`].
    pub fn from_weights(
        weights: F5Weights,
        sampling: F5Sampling,
        device: Device,
    ) -> Result<Self, TtsError> {
        let tokenizer = F5Tokenizer::from_vocab_path(&weights.vocab_path)?;
        Ok(Self {
            tokenizer,
            dit: weights.dit,
            vocos: weights.vocos,
            vocos_cfg: VocosConfig::vocos_24khz(),
            sampling,
            device,
        })
    }

    /// Convenience: download from HF + construct.
    ///
    /// Uses [`F5DitConfig::f5_base`] + [`VocosConfig::vocos_24khz`].
    ///
    /// # Errors
    ///
    /// Propagates any [`TtsError::ModelLoad`] from
    /// [`F5Weights::from_hf`].
    pub async fn from_hf(
        f5_model_id: &str,
        vocos_model_id: &str,
        sampling: F5Sampling,
        device: Device,
    ) -> Result<Self, TtsError> {
        let weights = F5Weights::from_hf(
            f5_model_id,
            vocos_model_id,
            F5DitConfig::f5_base(),
            VocosConfig::vocos_24khz(),
            &device,
        )
        .await?;
        Self::from_weights(weights, sampling, device)
    }

    /// Read-only access to the tokenizer (mostly for tests + tracing).
    #[must_use]
    pub fn tokenizer(&self) -> &F5Tokenizer {
        &self.tokenizer
    }

    /// Read-only access to the Vocos backbone config.
    #[must_use]
    pub fn vocos_config(&self) -> &VocosConfig {
        &self.vocos_cfg
    }

    /// Compute the target mel-frame count for an input character count.
    ///
    /// Returns `max(text_len * 50 / 32, MIN_MEL_FRAMES)`. Public for
    /// tests + tooling.
    #[must_use]
    pub fn compute_mel_frames(text_len: usize) -> usize {
        (text_len * MEL_FRAMES_PER_CHAR_NUM / MEL_FRAMES_PER_CHAR_DEN).max(MIN_MEL_FRAMES)
    }

    /// Synthesize speech from `text`.
    ///
    /// `reference_audio` + `reference_text` are accepted for the
    /// voice-clone path; they currently route through the same zero
    /// conditioning bias as unconditioned synthesis (see module
    /// docstring for the rationale).
    ///
    /// Returns the raw 16-bit PCM / 24 kHz mono WAV bytes alongside
    /// the rendered PCM sample count (the caller wraps these in
    /// [`blazen_audio::GeneratedAudio`]).
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Synthesis`] when any tensor op fails or
    /// when the input text is empty (the iSTFT requires a non-empty
    /// mel).
    //
    // The function is `async` even though the current body doesn't
    // await — the audio-encoder + text-encoder passes that land in
    // follow-up waves are spawn-blocking calls. Keeping the signature
    // async now avoids a churning API break later.
    #[allow(clippy::unused_async)]
    pub async fn synthesize_wav(
        &self,
        text: &str,
        _reference_audio: Option<&Path>,
        _reference_text: Option<&str>,
    ) -> Result<(Vec<u8>, usize), TtsError> {
        if text.is_empty() {
            return Err(TtsError::Synthesis(
                "f5-tts synthesize_wav: empty input text".to_owned(),
            ));
        }

        // 1. Tokenize input text.
        //    The DiT's text-id pathway is i64 in the upstream Python
        //    (`torch.LongTensor`). We don't actually feed token ids
        //    through the DiT yet (see module docs), but we still
        //    build the tensor here for the sampler signature and so
        //    the call shape is stable when the text encoder lands.
        let raw_ids: Vec<u32> = self.tokenizer.encode(text);
        #[allow(
            clippy::cast_possible_wrap,
            reason = "tokenizer ids never exceed i64::MAX in practice"
        )]
        let long_ids: Vec<i64> = raw_ids.iter().map(|&id| i64::from(id)).collect();
        let t_text = long_ids.len();
        let text_tensor = Tensor::from_vec(long_ids, (1, t_text), &self.device)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: text id tensor: {e}")))?;

        // 2. Null text for CFG: pad-tokens of the same length.
        #[allow(clippy::cast_possible_wrap)]
        let pad = i64::from(self.tokenizer.pad_token());
        let null_ids: Vec<i64> = vec![pad; t_text];
        let null_tensor = Tensor::from_vec(null_ids, (1, t_text), &self.device)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: null id tensor: {e}")))?;

        // 3. Target mel shape.
        let mel_dim = self.dit.config().mel_dim;
        let text_dim = self.dit.config().text_dim;
        let t_mel = Self::compute_mel_frames(t_text);
        let batch = 1_usize;

        // 4. Zero-conditioning placeholders for the DiT.
        //    `cond` mirrors the reference mel (shape `(B, T_mel, mel_dim)`);
        //    `text_embed` is the per-frame text feature stream
        //    (shape `(B, T_mel, text_dim)`). Both are stubbed as zeros
        //    pending the standalone text + audio encoders.
        let cond = Tensor::zeros((batch, t_mel, mel_dim), DType::F32, &self.device)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: cond zeros: {e}")))?;
        let text_embed = Tensor::zeros((batch, t_mel, text_dim), DType::F32, &self.device)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: text_embed zeros: {e}")))?;

        // 5. Run the Euler sampler.
        let forward = |x: &Tensor, _ids: &Tensor, t: &Tensor| -> Result<Tensor, TtsError> {
            // The sampler's `text_ids` argument is currently unused —
            // the text features arrive via the closure-captured
            // `text_embed`. When the text encoder lands the closure
            // will derive `text_embed` from `_ids` per step (it's
            // independent of the sampler's `t`, so an outer-scope
            // cache works just as well).
            self.dit
                .forward(x, &cond, &text_embed, t)
                .map_err(|e| TtsError::Synthesis(format!("f5-tts: dit forward: {e}")))
        };

        let mel = sample(
            &forward,
            &text_tensor,
            &null_tensor,
            (batch, t_mel, mel_dim),
            &self.sampling,
            &self.device,
        )?;

        // 6. Vocos expects `[B, mel_dim, T]`; the sampler returns
        //    `[B, T, mel_dim]` — transpose the time/feature axes.
        let mel_for_vocos = mel
            .transpose(1, 2)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: vocos transpose: {e}")))?;
        let spec = self
            .vocos
            .forward(&mel_for_vocos)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: vocos forward: {e}")))?;

        // 7. iSTFT → mono waveform.
        let waveform = istft(&spec, &self.vocos_cfg)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: istft: {e}")))?;
        let samples: Vec<f32> = waveform
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: waveform to vec: {e}")))?;
        let n_samples = samples.len();

        // 8. WAV encode (16-bit / 24 kHz / mono).
        let wav_bytes = encode_wav_16bit(&samples, F5_SAMPLE_RATE_HZ, 1);
        Ok((wav_bytes, n_samples))
    }
}

/// Minimal 16-bit-PCM WAV encoder. Mirrors the local helper inside
/// the Bark pipeline (deliberately duplicated — the two files share
/// only this ~25-line writer and we keep them decoupled rather than
/// hoist a `wav.rs` module just for two callers).
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

/// Compute the inclusive PCM duration in seconds. Public to the
/// parent `mod.rs` so it can populate
/// [`blazen_audio::GeneratedAudio::duration_seconds`] without
/// re-computing the ratio.
#[allow(clippy::cast_precision_loss, reason = "PCM sample counts fit f32")]
#[must_use]
pub(super) fn pcm_duration_seconds(num_samples: usize, sample_rate: u32) -> f32 {
    num_samples as f32 / sample_rate as f32
}

// ---------------------------------------------------------------------------
// Lazy-loaded pipeline cache used by the F5Backend trait impl.
// ---------------------------------------------------------------------------

/// Thread-safe one-shot cache for the heavy pipeline construction.
/// Cloned `F5Backend` values share the same `Arc<OnceCell>` so the
/// lazily-loaded pipeline is built once across clones.
pub(super) type PipelineCell = Arc<OnceCell<Arc<F5Pipeline>>>;

/// Build a new empty [`PipelineCell`] for [`super::F5Backend`].
#[must_use]
pub(super) fn new_pipeline_cell() -> PipelineCell {
    Arc::new(OnceCell::new())
}

/// Lazily build (or fetch the cached) [`F5Pipeline`] for the given
/// configuration. Subsequent calls reuse the cached pipeline.
pub(super) async fn get_or_init_pipeline(
    cell: &PipelineCell,
    f5_model_id: &str,
    vocos_model_id: &str,
    sampling: F5Sampling,
) -> Result<Arc<F5Pipeline>, TtsError> {
    let f5_id = f5_model_id.to_owned();
    let vocos_id = vocos_model_id.to_owned();
    let arc = cell
        .get_or_try_init(|| async move {
            let device = Device::Cpu;
            let pipeline = F5Pipeline::from_hf(&f5_id, &vocos_id, sampling, device).await?;
            Ok::<Arc<F5Pipeline>, TtsError>(Arc::new(pipeline))
        })
        .await?;
    Ok(arc.clone())
}

// ---------------------------------------------------------------------------
// Voice-clone reference cache helpers (used by `F5Backend::clone_voice`).
// ---------------------------------------------------------------------------

/// Directory under the user cache where cloned-voice reference clips
/// (`.wav` + `.txt` pairs) live. Uses `~/.cache/blazen/f5-voices/`
/// by default; falls back to `/tmp` if no home is set.
#[must_use]
pub(super) fn voice_cache_dir() -> PathBuf {
    std::env::var_os("BLAZEN_F5_VOICE_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|h| h.join(".cache/blazen/f5-voices")))
        .unwrap_or_else(|| PathBuf::from("/tmp/blazen-f5-voices"))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Persist a reference clip + its transcript to the voice cache.
/// Returns the `.wav` path; the transcript lives at the same stem
/// with a `.txt` suffix.
///
/// # Errors
///
/// Returns [`TtsError::Synthesis`] when the cache directory cannot be
/// created or the file writes fail.
pub(super) fn save_voice_reference(
    name: &str,
    audio_bytes: &[u8],
    transcript: Option<&str>,
) -> Result<PathBuf, TtsError> {
    let dir = voice_cache_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| TtsError::Synthesis(format!("f5-tts voice cache mkdir: {e}")))?;
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
    let wav_path = dir.join(format!("{safe_name}.wav"));
    let txt_path = dir.join(format!("{safe_name}.txt"));
    std::fs::write(&wav_path, audio_bytes)
        .map_err(|e| TtsError::Synthesis(format!("f5-tts voice cache wav write: {e}")))?;
    if let Some(t) = transcript {
        std::fs::write(&txt_path, t)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts voice cache txt write: {e}")))?;
    }
    Ok(wav_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    /// Tiny vocab fixture: a single space token + lowercase ASCII +
    /// punctuation. Satisfies the upstream `vocab[0] == " "`
    /// invariant.
    fn fixture_vocab() -> String {
        let mut s = String::from(" \n");
        for c in 'a'..='z' {
            s.push(c);
            s.push('\n');
        }
        s.push_str(".\n");
        s
    }

    /// Tiny `DiT` config so the smoke test stays fast (depth 1,
    /// hidden 16, `mel_dim` 8, `text_dim` 8). The Vocos config gets
    /// matching tiny dimensions in [`tiny_vocos_cfg`]. Both modules
    /// load from [`VarBuilder::zeros`] — no real weights, no network.
    fn tiny_dit_cfg() -> F5DitConfig {
        F5DitConfig {
            mel_dim: 8,
            hidden_dim: 16,
            depth: 1,
            heads: 2,
            head_dim: 8,
            ff_mult: 2,
            text_dim: 8,
            conv_layers: 0,
            text_num_embeds: 32,
            freq_embed_dim: 8,
            conv_pos_kernel: 3,
            conv_pos_groups: 2,
            max_seq_len: 256,
            long_skip_connection: false,
            qk_norm: false,
        }
    }

    fn tiny_vocos_cfg() -> VocosConfig {
        // Keep n_fft small + win == n_fft + hop divides n_fft so the
        // overlap-add math has enough frames at MIN_MEL_FRAMES=64.
        VocosConfig {
            mel_dim: 8,
            hidden_dim: 8,
            n_layers: 1,
            intermediate_dim: 16,
            n_fft: 16,
            hop_length: 4,
            win_length: 16,
            sample_rate: F5_SAMPLE_RATE_HZ,
        }
    }

    fn build_zero_weight_pipeline(vocab_path: &Path) -> Result<F5Pipeline, TtsError> {
        let device = Device::Cpu;
        let dit_cfg = tiny_dit_cfg();
        let vocos_cfg = tiny_vocos_cfg();
        let vb_dit = VarBuilder::zeros(DType::F32, &device);
        let vb_vocos = VarBuilder::zeros(DType::F32, &device);
        let dit = F5Dit::new(vb_dit, dit_cfg)
            .map_err(|e| TtsError::ModelLoad(format!("tiny dit: {e}")))?;
        let vocos_backbone = VocosBackbone::load(vb_vocos, vocos_cfg.clone())
            .map_err(|e| TtsError::ModelLoad(format!("tiny vocos: {e}")))?;
        // Use a fast sampling config so the smoke test finishes in
        // sub-second on a debug build.
        let sampling = F5Sampling {
            n_steps: 2,
            cfg_strength: 0.0, // skip the unconditional pass
            sway_sampling_coef: None,
            seed: None,
        };
        let tokenizer = F5Tokenizer::from_vocab_path(vocab_path)?;
        Ok(F5Pipeline {
            tokenizer,
            dit,
            vocos: vocos_backbone,
            vocos_cfg,
            sampling,
            device,
        })
    }

    #[test]
    fn compute_mel_duration_heuristic_matches_upstream() {
        // 50 / 32 ratio with a 64-frame floor.
        // text_len = 1 -> floor (64).
        assert_eq!(F5Pipeline::compute_mel_frames(1), 64);
        // text_len = 64 -> floor still wins (64 * 50 / 32 = 100,
        // exceeds floor, so we get the ratio).
        assert_eq!(F5Pipeline::compute_mel_frames(64), 100);
        // text_len = 128 -> 128 * 50 / 32 = 200.
        assert_eq!(F5Pipeline::compute_mel_frames(128), 200);
        // text_len = 0 -> floor (the empty-text case is rejected at
        // the synthesize entry point anyway, but the helper itself
        // is total).
        assert_eq!(F5Pipeline::compute_mel_frames(0), MIN_MEL_FRAMES);
    }

    #[test]
    fn encode_wav_16bit_emits_riff_header() {
        let samples = [0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav_16bit(&samples, F5_SAMPLE_RATE_HZ, 1);
        assert_eq!(&bytes[0..4], b"RIFF", "wav must start with RIFF");
        assert_eq!(&bytes[8..12], b"WAVE", "container type must be WAVE");
        assert_eq!(&bytes[12..16], b"fmt ", "fmt chunk must follow");
        assert_eq!(&bytes[36..40], b"data", "data chunk must follow fmt");
    }

    #[test]
    fn pcm_duration_seconds_at_24khz() {
        // 24 000 samples / 24 kHz = 1.0 s.
        assert!((pcm_duration_seconds(24_000, 24_000) - 1.0).abs() < f32::EPSILON);
        assert!((pcm_duration_seconds(12_000, 24_000) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn voice_cache_dir_honors_env_override() {
        // SAFETY: tests are single-threaded by harness convention for
        // this crate; no other thread reads BLAZEN_F5_VOICE_DIR
        // concurrently. We restore the env at the end of the test.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("BLAZEN_F5_VOICE_DIR", "/tmp/blazen-test-f5-voices");
        }
        let dir = voice_cache_dir();
        assert_eq!(dir, PathBuf::from("/tmp/blazen-test-f5-voices"));
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("BLAZEN_F5_VOICE_DIR");
        }
    }

    #[test]
    fn save_voice_reference_writes_wav_and_transcript() {
        let tmp = std::env::temp_dir().join(format!("blazen-f5-voice-test-{}", std::process::id()));
        // SAFETY: see `voice_cache_dir_honors_env_override`.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("BLAZEN_F5_VOICE_DIR", &tmp);
        }
        let wav_path = save_voice_reference("my voice!", b"RIFF...fake...", Some("hello"))
            .expect("save_voice_reference");
        assert!(wav_path.exists(), "wav file must exist at {wav_path:?}");
        assert_eq!(
            wav_path.file_name().and_then(|s| s.to_str()),
            Some("my_voice_.wav"),
            "non-alphanumeric chars must collapse to '_'",
        );
        let txt_path = wav_path.with_extension("txt");
        let txt = std::fs::read_to_string(&txt_path).expect("read transcript");
        assert_eq!(txt, "hello");
        let _ = std::fs::remove_dir_all(&tmp);
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("BLAZEN_F5_VOICE_DIR");
        }
    }

    #[tokio::test]
    async fn pipeline_constructs_from_zero_varbuilder_and_synthesizes_short_text() {
        // Write a tiny vocab to a temp file the tokenizer can read.
        let tmp =
            std::env::temp_dir().join(format!("blazen-f5-vocab-test-{}.txt", std::process::id()));
        std::fs::write(&tmp, fixture_vocab()).expect("write vocab fixture");
        let pipeline = build_zero_weight_pipeline(&tmp).expect("zero-weight pipeline");
        let (wav, n_samples) = pipeline
            .synthesize_wav("hello", None, None)
            .await
            .expect("synthesize hello");
        assert_eq!(&wav[0..4], b"RIFF", "wav must start with RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert!(n_samples > 0, "rendered PCM must have samples");
        // Vocos PCM byte count == 2 * n_samples (16-bit) and the WAV
        // header adds 44 bytes.
        assert_eq!(wav.len(), 44 + n_samples * 2);
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn pipeline_rejects_empty_text() {
        let tmp = std::env::temp_dir().join(format!(
            "blazen-f5-vocab-test-empty-{}.txt",
            std::process::id()
        ));
        std::fs::write(&tmp, fixture_vocab()).expect("write vocab fixture");
        let pipeline = build_zero_weight_pipeline(&tmp).expect("zero-weight pipeline");
        let err = pipeline
            .synthesize_wav("", None, None)
            .await
            .expect_err("empty text must be rejected");
        match err {
            TtsError::Synthesis(msg) => assert!(msg.contains("empty input text"), "msg = {msg}"),
            other => panic!("expected Synthesis error, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Live download test — requires HF cache + network. Off by
    /// default to keep CI fast and offline-clean.
    #[tokio::test]
    #[ignore = "network: downloads ~1.4 GB of F5-TTS + Vocos weights"]
    async fn synthesize_with_real_weights() {
        let pipeline = F5Pipeline::from_hf(
            "SWivid/F5-TTS",
            "charactr/vocos-mel-24khz",
            F5Sampling::default(),
            Device::Cpu,
        )
        .await
        .expect("download F5 + Vocos from HF");
        let (wav, n_samples) = pipeline
            .synthesize_wav("Hello world.", None, None)
            .await
            .expect("synthesize hello world");
        assert_eq!(&wav[0..4], b"RIFF");
        assert!(n_samples > 0);
    }
}
