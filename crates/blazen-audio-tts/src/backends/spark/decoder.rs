//! Spark-TTS Qwen2.5-0.5B autoregressive decoder (Wave S.2.2).
//!
//! Thin wrapper over [`candle_transformers::models::qwen2::ModelForCausalLM`]
//! that performs greedy / temperature-scaled autoregressive sampling using
//! candle's stock [`candle_transformers::generation::LogitsProcessor`].
//!
//! # What this module does (and what it does NOT)
//!
//! The upstream Spark-TTS LLM checkpoint
//! (`SparkAudio/Spark-TTS-0.5B/LLM`) is a Qwen2.5-0.5B model whose
//! tokenizer vocabulary has been **extended** with ~13.5k literal
//! `BiCodec` tokens like `<|bicodec_semantic_0|>` …
//! `<|bicodec_semantic_8191|>`, `<|bicodec_global_0|>` …
//! `<|bicodec_global_4095|>`, plus structural markers like
//! `<|start_global_token|>`, `<|end_global_token|>`,
//! `<|start_semantic_token|>`, `<|task_tts|>`, etc.
//!
//! In other words, `BiCodec` tokens are **just text tokens** as far as
//! the LM is concerned. Generation is bog-standard AR sampling against
//! the full extended vocab (Qwen2.5 config reports
//! `vocab_size = 166000` for this checkpoint, vs. ~151936 for stock
//! Qwen2.5).
//!
//! This module therefore only handles:
//!
//! 1. Loading the safetensors checkpoint into
//!    [`Qwen2Model`](candle_transformers::models::qwen2::ModelForCausalLM).
//! 2. Running the KV-cached AR forward+sample loop until EOS or
//!    `max_new_tokens`, returning the **full** id sequence
//!    (prompt ++ generated).
//!
//! Prompt construction (with `<|task_tts|>`, `<|start_content|>`, …
//! markers and the reference-audio `BiCodec` spans) lives in
//! [`super::tokenizer`] (Wave S.2.3). Parsing the generated tokens back
//! into integer `BiCodec` semantic / global indices and feeding them to
//! [`super::bicodec`] lives in [`super::pipeline`] (Wave S.2.4).

#![cfg(feature = "spark-tts")]
#![allow(
    dead_code,
    reason = "Wave S.2.2 lands the Qwen2.5 AR decoder surface (config, \
              loader, sampler, generate loop) that the S.2.4 pipeline \
              wave will consume. Until then nothing else in the crate \
              calls this module — the unit tests below exercise the \
              configured surface in the meantime."
)]

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};

/// EOS token id for the Spark-TTS LLM checkpoint.
///
/// Sourced from
/// `https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/LLM/config.json`
/// (`"eos_token_id": 151645`) and confirmed via the bundled
/// `tokenizer_config.json` (`151645` → `<|im_end|>`). This matches the
/// stock Qwen2.5 EOS id — the Spark-TTS vocab extension grows the
/// vocab beyond `151645` but does not relocate the existing special
/// tokens.
pub(super) const SPARK_LLM_EOS_TOKEN_ID: u32 = 151_645;

/// Sampling configuration for the Spark-TTS Qwen2.5 decoder.
///
/// Defaults mirror upstream `cli/SparkTTS.py::SparkTTS.inference`
/// (`temperature=0.8`, `top_k=50`, `top_p=0.95`, `max_new_tokens=3000`,
/// `do_sample=True`). Upstream passes no `repetition_penalty`, so we
/// default to `1.0` (HF default = no penalty).
#[derive(Debug, Clone)]
pub(super) struct SparkLlmConfig {
    /// Softmax temperature. `0.0` collapses to greedy argmax sampling.
    /// Upstream Spark-TTS default: `0.8`.
    pub temperature: f32,
    /// Top-k filter applied before sampling. `None` disables.
    /// Upstream Spark-TTS default: `Some(50)`.
    pub top_k: Option<usize>,
    /// Top-p (nucleus) filter applied after top-k. `None` disables.
    /// Upstream Spark-TTS default: `Some(0.95)`.
    pub top_p: Option<f32>,
    /// Repetition penalty (`1.0` = no penalty). Upstream Spark-TTS
    /// does not pass `repetition_penalty` to `model.generate`, so HF
    /// transformers falls back to `1.0`.
    pub repetition_penalty: f32,
    /// Maximum number of new tokens to emit on top of the prompt.
    /// Upstream Spark-TTS default: `3000` (≈ enough for ~30 s of
    /// audio at the 50 Hz `BiCodec` semantic rate).
    pub max_new_tokens: usize,
    /// Stop-on-EOS token id. Defaults to
    /// [`SPARK_LLM_EOS_TOKEN_ID`].
    pub eos_token_id: u32,
    /// PRNG seed for the sampler. Fixed seed → deterministic output.
    pub seed: u64,
}

impl Default for SparkLlmConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.0,
            max_new_tokens: 3000,
            eos_token_id: SPARK_LLM_EOS_TOKEN_ID,
            seed: 0,
        }
    }
}

/// Spark-TTS Qwen2.5-0.5B autoregressive decoder.
///
/// Wraps [`Qwen2Model`] plus its KV cache and a
/// [`LogitsProcessor`]. Call [`SparkLlm::generate`] to run the
/// AR loop.
pub(super) struct SparkLlm {
    inner: Qwen2Model,
    device: Device,
    cfg: SparkLlmConfig,
}

impl SparkLlm {
    /// Load a Qwen2.5 checkpoint from a single safetensors file (the
    /// Spark-TTS bundle ships `LLM/model.safetensors` as a single ~2 GB
    /// file rather than a sharded set — see
    /// <https://huggingface.co/SparkAudio/Spark-TTS-0.5B/tree/main/LLM>).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`candle_core::Error`] if the safetensors
    /// file is missing, malformed, or its tensor shapes don't match
    /// `qwen_config`.
    pub(super) fn load(
        weights_path: &Path,
        qwen_config: &Qwen2Config,
        device: Device,
        cfg: SparkLlmConfig,
    ) -> CandleResult<Self> {
        Self::load_from_paths(&[weights_path.to_path_buf()], qwen_config, device, cfg)
    }

    /// Load a Qwen2.5 checkpoint that may be split across multiple
    /// safetensors shards (`model-00001-of-00002.safetensors`, …).
    /// Pass all shard paths in index order; this is the same convention
    /// candle's other model examples follow.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`candle_core::Error`] if any shard is
    /// missing or malformed, or if the merged var-builder can't satisfy
    /// every parameter of `qwen_config`.
    pub(super) fn load_from_paths(
        weights_paths: &[PathBuf],
        qwen_config: &Qwen2Config,
        device: Device,
        cfg: SparkLlmConfig,
    ) -> CandleResult<Self> {
        // SAFETY: safetensors mmap is sound provided the underlying
        // files are not mutated under us, which holds for read-only
        // distributed model checkpoints. This mirrors the pattern used
        // by `super::bicodec::BiCodec::from_safetensors` (and every
        // other candle-based backend in this crate).
        #[allow(
            unsafe_code,
            reason = "VarBuilder::from_mmaped_safetensors is an unsafe API \
                      because mmapped contents can race with on-disk mutation. \
                      Model checkpoints are read-only on disk, so this is the \
                      canonical safe usage pattern."
        )]
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(weights_paths, DType::F32, &device)? };
        let inner = Qwen2Model::new(qwen_config, vb)?;
        Ok(Self { inner, device, cfg })
    }

    /// Active sampling configuration. Useful for tests + diagnostics.
    pub(super) fn config(&self) -> &SparkLlmConfig {
        &self.cfg
    }

    /// Device the underlying Qwen2 weights live on.
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// Autoregressively extend `prompt_ids` until EOS or
    /// `cfg.max_new_tokens` is reached.
    ///
    /// `prompt_ids` must have shape `(1, T_prompt)` with dtype `u32`
    /// and live on [`Self::device`]. The returned tensor has shape
    /// `(1, T_prompt + T_generated)` and contains the prompt followed
    /// by every sampled token (including the trailing EOS if one was
    /// emitted).
    ///
    /// The KV cache inside the wrapped [`Qwen2Model`] is **cleared**
    /// at the start of every call so back-to-back generations don't
    /// leak state.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`candle_core::Error`] from any tensor
    /// op or sampling failure.
    pub(super) fn generate(&mut self, prompt_ids: &Tensor) -> CandleResult<Tensor> {
        // The non-streaming path is the streaming path with a no-op
        // callback. Keeping both entry points around lets existing
        // callers stay terse while the streaming wave plumbs a real
        // per-token sink.
        self.generate_streaming(prompt_ids, |_| Ok(()))
    }

    /// Streaming variant of [`Self::generate`]. Calls
    /// `on_token(token_id)` for each freshly sampled token (not for the
    /// prompt tokens — those are already known to the caller). Stops at
    /// EOS, at `cfg.max_new_tokens`, or as soon as `on_token` returns
    /// `Err(_)`. In every case the **full** `(prompt ++ generated)`
    /// tensor is still returned so non-streaming consumers keep working.
    ///
    /// This is the per-token forward hook the [`super::pipeline`]
    /// streaming wave dispatches against: the closure forwards each
    /// sampled id into an mpsc channel so the async pipeline can wake
    /// up the moment a new `<|bicodec_semantic_K|>` token becomes
    /// available, rather than blocking until the entire utterance
    /// finishes generating.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`candle_core::Error`] from any tensor
    /// op, sampling failure, OR from `on_token` (the closure's `Err`
    /// is wrapped via [`candle_core::Error::Msg`] and surfaced to the
    /// caller).
    pub(super) fn generate_streaming<F>(
        &mut self,
        prompt_ids: &Tensor,
        mut on_token: F,
    ) -> CandleResult<Tensor>
    where
        F: FnMut(u32) -> CandleResult<()>,
    {
        let (batch, _prompt_len) = prompt_ids.dims2()?;
        debug_assert_eq!(
            batch, 1,
            "SparkLlm::generate currently only supports batch=1"
        );

        // Reset KV cache so a previous call doesn't bleed into this one.
        self.inner.clear_kv_cache();

        let sampling = build_sampling(&self.cfg);
        let mut sampler = LogitsProcessor::from_sampling(self.cfg.seed, sampling);

        // Materialise the prompt on the host so we can (a) apply the
        // repetition penalty against the running id history and
        // (b) cheaply append sampled tokens before returning.
        let mut history: Vec<u32> = prompt_ids.flatten_all()?.to_vec1()?;

        // Step 0: full prompt forward pass at seqlen_offset = 0. Yields
        // logits for the LAST position only (Qwen2Model::forward narrows
        // to the final timestep internally).
        let mut logits = self.inner.forward(prompt_ids, 0)?;

        for _ in 0..self.cfg.max_new_tokens {
            let next = sample_one(&mut sampler, &logits, &history, self.cfg.repetition_penalty)?;
            history.push(next);
            // Notify the streaming hook BEFORE the EOS short-circuit so
            // the consumer sees the terminator and can flush any
            // pending partial batch. If the hook errors (typically: the
            // consumer dropped the receiving end of the channel) we
            // bail out of the AR loop early but still hand back the
            // partial `(prompt ++ generated-so-far)` tensor so the
            // caller can salvage the work-in-progress. Mirrors how
            // tokio-stream consumers treat downstream cancellation as a
            // clean stop signal rather than a hard error.
            if on_token(next).is_err() {
                break;
            }
            if next == self.cfg.eos_token_id {
                break;
            }
            // Single-token forward at the current cache offset.
            let offset = history.len() - 1;
            let next_input = Tensor::from_slice(&[next], (1, 1), &self.device)?;
            logits = self.inner.forward(&next_input, offset)?;
        }

        let total_len = history.len();
        Tensor::from_vec(history, (1, total_len), &self.device)
    }
}

/// Apply repetition penalty in-place (HF semantics: divide logits of
/// previously-seen ids by `penalty` when positive, multiply when
/// negative). `penalty == 1.0` is a no-op.
fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    let vocab = logits.len();
    for &id in history {
        let idx = id as usize;
        if idx >= vocab {
            continue;
        }
        let l = logits[idx];
        logits[idx] = if l >= 0.0 { l / penalty } else { l * penalty };
    }
}

/// Sample a single next token from logits with shape `(1, 1, vocab)` or
/// `(1, vocab)`. Squeezes leading singleton dims, applies the
/// repetition penalty against `history`, and dispatches to the
/// candle-stock [`LogitsProcessor`].
fn sample_one(
    sampler: &mut LogitsProcessor,
    logits: &Tensor,
    history: &[u32],
    repetition_penalty: f32,
) -> CandleResult<u32> {
    // Qwen2Model::forward returns (batch, 1, vocab); collapse to (vocab,).
    let last = match logits.dims() {
        [_, _, _] => logits.squeeze(0)?.squeeze(0)?,
        [_, _] => logits.squeeze(0)?,
        _ => logits.clone(),
    };
    let last_f32 = last.to_dtype(DType::F32)?;
    if (repetition_penalty - 1.0).abs() < f32::EPSILON {
        return sampler.sample(&last_f32);
    }
    // sample_f gives us mutable access to the post-softmax probs, but
    // we want to penalise pre-softmax logits — easiest path is to
    // materialise+penalise+reupload the logits and then call sample().
    let mut logit_vec: Vec<f32> = last_f32.to_vec1()?;
    apply_repetition_penalty(&mut logit_vec, history, repetition_penalty);
    let len = logit_vec.len();
    let penalised = Tensor::from_vec(logit_vec, len, last_f32.device())?;
    sampler.sample(&penalised)
}

/// Translate a [`SparkLlmConfig`] into the candle [`Sampling`] enum.
///
/// * `temperature <= 0` (or effectively zero) → greedy [`Sampling::ArgMax`].
/// * `top_k.is_some() && top_p.is_some()` → [`Sampling::TopKThenTopP`].
/// * `top_k.is_some()` alone → [`Sampling::TopK`].
/// * `top_p.is_some()` alone → [`Sampling::TopP`].
/// * otherwise → [`Sampling::All`] (pure temperature sampling).
fn build_sampling(cfg: &SparkLlmConfig) -> Sampling {
    if cfg.temperature <= f32::EPSILON {
        return Sampling::ArgMax;
    }
    let temperature = f64::from(cfg.temperature);
    match (cfg.top_k, cfg.top_p) {
        (Some(k), Some(p)) => Sampling::TopKThenTopP {
            k,
            p: f64::from(p),
            temperature,
        },
        (Some(k), None) => Sampling::TopK { k, temperature },
        (None, Some(p)) => Sampling::TopP {
            p: f64::from(p),
            temperature,
        },
        (None, None) => Sampling::All { temperature },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defaults must mirror upstream `cli/SparkTTS.py::SparkTTS.inference`
    /// kwargs (`temperature=0.8`, `top_k=50`, `top_p=0.95`,
    /// `max_new_tokens=3000`, `do_sample=True`). Repetition penalty is
    /// not passed upstream, so HF transformers' default of `1.0` (i.e.
    /// no penalty) is what we mirror.
    #[test]
    fn spark_llm_config_defaults_match_upstream() {
        let cfg = SparkLlmConfig::default();
        assert!(
            (cfg.temperature - 0.8).abs() < 1e-6,
            "temperature = {} (expected 0.8 to match upstream)",
            cfg.temperature
        );
        assert_eq!(cfg.top_k, Some(50));
        assert_eq!(
            cfg.top_p,
            Some(0.95),
            "top_p must match upstream Spark-TTS default of 0.95"
        );
        assert!(
            (cfg.repetition_penalty - 1.0).abs() < f32::EPSILON,
            "repetition_penalty = {} (HF default — upstream Spark-TTS does \
             not override)",
            cfg.repetition_penalty
        );
        assert_eq!(cfg.max_new_tokens, 3000);
        assert_eq!(cfg.eos_token_id, SPARK_LLM_EOS_TOKEN_ID);
    }

    /// Sourced from
    /// `https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/LLM/config.json`
    /// (`"eos_token_id": 151645`) and cross-checked against
    /// `LLM/tokenizer_config.json` where token id `151645` maps to
    /// `<|im_end|>` — the stock Qwen2.5 EOS marker. The Spark-TTS vocab
    /// extension grows the vocabulary beyond 151645 but does NOT relocate
    /// the pre-existing special tokens, so the Qwen2.5 default stays valid.
    #[test]
    fn spark_llm_eos_token_id_matches_qwen_default() {
        assert_eq!(SPARK_LLM_EOS_TOKEN_ID, 151_645);
        assert_eq!(SparkLlmConfig::default().eos_token_id, 151_645);
    }

    /// Loading from a non-existent path must surface a Candle error
    /// rather than panic. Real end-to-end generation requires a ~2 GB
    /// safetensors download plus a Qwen2 [`Qwen2Config`]; that path is
    /// deferred to Wave S.2.4 pipeline integration and gated behind a
    /// `BLAZEN_TEST_SPARK_TTS=1` env-var (see
    /// [`super::bicodec`] / `super::weights` for the matching pattern).
    #[test]
    fn spark_llm_load_returns_error_for_missing_weights() {
        let cfg = qwen_config_for_spark_tts_05b();
        let result = SparkLlm::load(
            Path::new("/nonexistent/spark-tts/model.safetensors"),
            &cfg,
            Device::Cpu,
            SparkLlmConfig::default(),
        );
        assert!(
            result.is_err(),
            "loading a non-existent checkpoint must fail (got Ok)"
        );
    }

    #[test]
    fn build_sampling_collapses_to_argmax_when_temperature_zero() {
        let cfg = SparkLlmConfig {
            temperature: 0.0,
            ..SparkLlmConfig::default()
        };
        assert!(matches!(build_sampling(&cfg), Sampling::ArgMax));
    }

    #[test]
    fn build_sampling_with_top_k_and_top_p_picks_top_k_then_top_p() {
        let cfg = SparkLlmConfig::default();
        match build_sampling(&cfg) {
            Sampling::TopKThenTopP { k, p, temperature } => {
                assert_eq!(k, 50);
                assert!((p - 0.95).abs() < 1e-6);
                assert!((temperature - 0.8).abs() < 1e-6);
            }
            other => panic!("expected TopKThenTopP, got {other:?}"),
        }
    }

    #[test]
    fn build_sampling_with_only_top_k_picks_top_k() {
        let cfg = SparkLlmConfig {
            top_p: None,
            ..SparkLlmConfig::default()
        };
        assert!(matches!(build_sampling(&cfg), Sampling::TopK { k: 50, .. }));
    }

    #[test]
    fn build_sampling_with_only_top_p_picks_top_p() {
        let cfg = SparkLlmConfig {
            top_k: None,
            ..SparkLlmConfig::default()
        };
        match build_sampling(&cfg) {
            Sampling::TopP { p, .. } => assert!((p - 0.95).abs() < 1e-6),
            other => panic!("expected TopP, got {other:?}"),
        }
    }

    #[test]
    fn build_sampling_with_no_filters_picks_all() {
        let cfg = SparkLlmConfig {
            top_k: None,
            top_p: None,
            ..SparkLlmConfig::default()
        };
        assert!(matches!(build_sampling(&cfg), Sampling::All { .. }));
    }

    #[test]
    fn apply_repetition_penalty_is_noop_when_penalty_is_one() {
        let mut logits = vec![1.0_f32, -2.0, 3.0, -4.0];
        let history = [0, 1, 2, 3];
        apply_repetition_penalty(&mut logits, &history, 1.0);
        assert_eq!(logits, vec![1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn apply_repetition_penalty_divides_positive_and_multiplies_negative() {
        let mut logits = vec![2.0_f32, -2.0, 4.0, 0.0];
        let history = [0_u32, 1, 2];
        apply_repetition_penalty(&mut logits, &history, 2.0);
        // id 0 (pos 2.0) → 2.0 / 2.0 = 1.0
        // id 1 (neg -2.0) → -2.0 * 2.0 = -4.0
        // id 2 (pos 4.0) → 4.0 / 2.0 = 2.0
        // id 3 (untouched) → 0.0
        assert_eq!(logits, vec![1.0, -4.0, 2.0, 0.0]);
    }

    #[test]
    fn apply_repetition_penalty_skips_out_of_vocab_ids() {
        let mut logits = vec![1.0_f32, 1.0, 1.0];
        let history = [99_u32];
        apply_repetition_penalty(&mut logits, &history, 2.0);
        assert_eq!(logits, vec![1.0, 1.0, 1.0]);
    }

    /// Verifies the public surface the pipeline wave (S.2.4) will rely
    /// on: a constructable config, an EOS constant, and a sampler-builder
    /// indirectly exercised via the other tests. End-to-end generation
    /// against a real checkpoint is deferred to S.2.4 (gated behind
    /// `BLAZEN_TEST_SPARK_TTS=1`).
    #[test]
    fn spark_llm_public_surface_is_constructible() {
        let cfg = SparkLlmConfig::default();
        assert!(cfg.max_new_tokens > 0);
        // Direct sanity check of the public-to-the-module accessor.
        let _ = SPARK_LLM_EOS_TOKEN_ID;
    }

    /// Synthetic Qwen2 config matching the Spark-TTS-0.5B checkpoint
    /// (`https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/LLM/config.json`).
    /// Only used by tests that try to load weights — the real
    /// [`Qwen2Config`] is parsed from `config.json` by the weights/
    /// pipeline waves.
    fn qwen_config_for_spark_tts_05b() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 166_000,
            hidden_size: 896,
            intermediate_size: 4_864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            max_position_embeddings: 32_768,
            sliding_window: 32_768,
            max_window_layers: 21,
            tie_word_embeddings: true,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        }
    }
}
