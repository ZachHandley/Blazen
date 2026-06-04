//! Bark coarse acoustic stage — semantic tokens → first 2 `EnCodec` codebooks.
//!
//! GPT-style autoregressive decoder that consumes the semantic-token
//! sequence emitted by [`super::semantic`] and produces the leading two
//! codebooks of `EnCodec` acoustic tokens by interleaved sampling. The
//! remaining six codebooks are filled in by the non-autoregressive
//! [`super::fine`] stage.
//!
//! # Architecture
//!
//! Identical block shape to the semantic stage (`suno-ai/bark/bark/model.py`
//! `class GPT(nn.Module)`, mirrored in HF transformers at
//! `models/bark/modeling_bark.py:361` `BarkCausalModel`):
//!
//! - Token embedding `input_embeds_layer: Embedding(input_vocab_size, n_embd)`
//! - Learned positional embedding
//!   `position_embeds_layer: Embedding(block_size, n_embd)`
//! - `n_layer` × [`Block`] from [`super::gpt_block`] (pre-LN, causal
//!   self-attention, MLP)
//! - Final `LayerNorm` (`layernorm_final`)
//! - Single `lm_head: Linear(n_embd, output_vocab_size, bias=false)` —
//!   produces logits across the **combined** vocabulary covering both
//!   codebooks. Alternation is enforced at sampling time by masking
//!   logits outside the active codebook's slice (matches upstream
//!   `AlternatingCodebooksLogitsProcessor` in
//!   `transformers/generation/logits_process.py:2175`).
//!
//! # Vocab offset trick
//!
//! To share a single embedding table across heterogeneous token types,
//! Bark packs the vocabularies sequentially:
//!
//! ```text
//!   [0 .. semantic_vocab_size)                                 -> semantic tokens
//!   [semantic_vocab_size      .. + codebook_size)              -> codebook 0
//!   [semantic_vocab_size + 1*codebook_size .. + codebook_size) -> codebook 1
//!   semantic_vocab_size + n_codebooks*codebook_size            -> coarse_semantic_pad_token
//!   + 2                                                        -> coarse_infer_token
//! ```
//!
//! See [`Self::vocab_offset_for_codebook`] for the exact arithmetic and
//! `BarkCoarseGenerationConfig` defaults in
//! `transformers/models/bark/generation_configuration_bark.py:120` for the
//! `coarse_semantic_pad_token=12_048` / `coarse_infer_token=12_050` /
//! `semantic_vocab_size=10_000` / `codebook_size=1024` numbers.
//!
//! # Deviations from upstream
//!
//! - **Single LM head, masked alternation** (matches HF transformers /
//!   upstream `suno-ai/bark`).
//! - **Dropout is omitted at inference time** (upstream sets `p=0.0` in
//!   eval mode; we never train).
//! - **KV-cache is not implemented in this wave.** Each `generate` step
//!   re-runs the prefix through the full transformer. Wave B.4 will add
//!   incremental decoding to match the upstream speed.

#![cfg(feature = "bark")]
#![allow(clippy::needless_pass_by_value)]

use candle_core::{D, DType, IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, linear_no_bias, ops};

use crate::error::TtsError;

use super::gpt_block::{Block, BlockConfig, load_layer_norm};

/// Configuration for the Bark coarse-stage transformer.
///
/// Defaults mirror the `suno/bark-small` checkpoint as published in
/// `suno-ai/bark/bark/model.py` (and cross-checked against HF transformers
/// `models/bark/configuration_bark.py:74`).
#[derive(Debug, Clone)]
pub struct CoarseConfig {
    /// Size of the semantic-token vocabulary slice in the packed
    /// embedding table. Bark uses `10_000`.
    pub semantic_vocab_size: usize,
    /// Number of entries per `EnCodec` codebook. Bark uses `1024`.
    pub codebook_size: usize,
    /// Number of codebooks the coarse stage emits. `2` for Bark — the
    /// remaining 6 are produced by the fine stage.
    pub n_codebooks: usize,
    /// Total input vocabulary size of the embedding table. Sized to cover
    /// `semantic_vocab_size + n_codebooks * codebook_size` plus a small
    /// pad for the coarse-specific special tokens (pad / infer).
    pub input_vocab_size: usize,
    /// Output vocabulary size of the LM head. Same shape as
    /// `input_vocab_size` in upstream Bark (embeddings are **not** tied;
    /// the LM head is a separate `Linear` per upstream `model.py`).
    pub output_vocab_size: usize,
    /// Number of stacked transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads per block. `n_embd` must be divisible
    /// by `n_head`.
    pub n_head: usize,
    /// Hidden / embedding dimension.
    pub n_embd: usize,
    /// Maximum sequence length the learned positional embedding can index
    /// (the upstream `block_size`).
    pub block_size: usize,
    /// Whether the `Linear` / `LayerNorm` layers carry a bias. HF
    /// transformers' published `bark-small` ships with `bias=false` for
    /// the coarse module; older Suno-format checkpoints used `bias=true`.
    pub bias: bool,
}

impl CoarseConfig {
    /// `suno/bark-small` coarse-stage configuration.
    ///
    /// 12 layers × hidden 768 × 12 heads. Matches `bark/model.py`'s
    /// `GPTConfig` defaults for the coarse module.
    #[must_use]
    pub fn bark_small() -> Self {
        // 10_000 semantic + 2 * 1024 codebooks = 12_048; +48 slop covers
        // the two coarse special tokens (pad=12_048, infer=12_050) and
        // rounds to a friendlier shape.
        let input_vocab_size = 10_000 + 2 * 1024 + 48;
        Self {
            semantic_vocab_size: 10_000,
            codebook_size: 1024,
            n_codebooks: 2,
            input_vocab_size,
            output_vocab_size: input_vocab_size,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            block_size: 1024,
            bias: false,
        }
    }

    /// `suno/bark` (full-size) coarse-stage configuration.
    ///
    /// 24 layers × hidden 1024 × 16 heads.
    #[must_use]
    pub fn bark() -> Self {
        let input_vocab_size = 10_000 + 2 * 1024 + 48;
        Self {
            semantic_vocab_size: 10_000,
            codebook_size: 1024,
            n_codebooks: 2,
            input_vocab_size,
            output_vocab_size: input_vocab_size,
            n_layer: 24,
            n_head: 16,
            n_embd: 1024,
            block_size: 1024,
            bias: false,
        }
    }

    /// First valid token index for codebook `k` (0-indexed) in the packed
    /// vocabulary. See module docs for the layout.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `k >= n_codebooks`.
    #[must_use]
    pub fn vocab_offset_for_codebook(&self, k: usize) -> usize {
        debug_assert!(k < self.n_codebooks, "codebook index {k} out of range");
        self.semantic_vocab_size + k * self.codebook_size
    }

    fn block_cfg(&self) -> BlockConfig {
        BlockConfig {
            n_embd: self.n_embd,
            n_head: self.n_head,
            block_size: self.block_size,
            bias: self.bias,
            is_causal: true,
        }
    }
}

/// Sampling knobs for [`CoarseDecoder::generate`].
#[derive(Debug, Clone)]
pub struct CoarseSampling {
    /// Softmax temperature. Bark default `0.7`.
    pub temperature: f32,
    /// Top-k filter applied before softmax. `None` disables.
    pub top_k: Option<usize>,
    /// Top-p (nucleus) filter applied after top-k. `None` disables.
    pub top_p: Option<f32>,
    /// Maximum number of coarse tokens to emit **in total across both
    /// codebooks** (i.e. `n_codebooks * frames_per_codebook`). The
    /// upstream default is `768` (≈ 5 s of 75 Hz audio × 2 codebooks).
    pub max_coarse_tokens: usize,
}

impl Default for CoarseSampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: None,
            top_p: None,
            max_coarse_tokens: 768,
        }
    }
}

/// Bark coarse-stage decoder.
///
/// See module docs for the architecture and vocab-packing layout.
pub struct CoarseDecoder {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: CoarseConfig,
}

impl CoarseDecoder {
    /// Build the decoder by reading weights from `vb`. The expected
    /// param-path layout matches `BarkCausalModel` in HF transformers:
    /// `input_embeds_layer`, `position_embeds_layer`, `layers.{i}.*`,
    /// `layernorm_final`, `lm_head`.
    ///
    /// # Errors
    ///
    /// Propagates any candle tensor-load failure.
    pub fn from_vb(vb: VarBuilder, config: CoarseConfig) -> CandleResult<Self> {
        let wte = embedding(
            config.input_vocab_size,
            config.n_embd,
            vb.pp("input_embeds_layer"),
        )?;
        let wpe = embedding(
            config.block_size,
            config.n_embd,
            vb.pp("position_embeds_layer"),
        )?;
        let blocks = (0..config.n_layer)
            .map(|i| Block::load(vb.pp(format!("layers.{i}")), config.block_cfg()))
            .collect::<CandleResult<Vec<_>>>()?;
        let ln_f = load_layer_norm(config.n_embd, config.bias, vb.pp("layernorm_final"))?;
        // `lm_head` is bias-free per upstream `modeling_bark.py:379`.
        let lm_head = linear_no_bias(config.n_embd, config.output_vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            config,
        })
    }

    /// Configuration this decoder was loaded against.
    #[must_use]
    pub fn config(&self) -> &CoarseConfig {
        &self.config
    }

    /// Forward pass.
    ///
    /// - `input_ids`: `[B, T]` packed-vocabulary token ids (must be in
    ///   `[0, input_vocab_size)`).
    /// - `codebook_idx`: which codebook the **next** sampled token should
    ///   come from. Drives the logit-mask that restricts the LM-head
    ///   output to the relevant `[offset, offset+codebook_size)` slice.
    ///   Pass `usize::MAX` to skip masking (e.g. for unit tests that
    ///   inspect raw logits).
    ///
    /// Returns the masked logits `[B, T, output_vocab_size]`.
    ///
    /// # Errors
    ///
    /// Propagates any candle tensor failure (shape mismatch, OOM, …).
    pub fn forward(&self, input_ids: &Tensor, codebook_idx: usize) -> CandleResult<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        assert!(
            t <= self.config.block_size,
            "sequence length {t} exceeds block_size {}",
            self.config.block_size
        );

        let tok_emb = self.wte.forward(input_ids)?;
        // Positions: `[0, 1, ..., t-1]` broadcast across the batch.
        let positions = Tensor::arange(
            0u32,
            u32::try_from(t).unwrap_or(u32::MAX),
            input_ids.device(),
        )?;
        let pos_emb = self.wpe.forward(&positions)?;
        // [T, E] + [B, T, E] via broadcast on dim 0.
        let mut xs = tok_emb.broadcast_add(&pos_emb)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let xs = self.ln_f.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?;

        if codebook_idx == usize::MAX || codebook_idx >= self.config.n_codebooks {
            return Ok(logits);
        }
        self.mask_logits_to_codebook(&logits, codebook_idx)
    }

    /// Restrict logits to the slice belonging to `codebook_idx`,
    /// setting all out-of-range entries to `-inf`. Mirrors
    /// `AlternatingCodebooksLogitsProcessor` in upstream HF transformers
    /// (`generation/logits_process.py:2175`).
    fn mask_logits_to_codebook(
        &self,
        logits: &Tensor,
        codebook_idx: usize,
    ) -> CandleResult<Tensor> {
        let vocab = self.config.output_vocab_size;
        let start = self.config.vocab_offset_for_codebook(codebook_idx);
        let end = start + self.config.codebook_size;

        let mut mask = vec![f32::NEG_INFINITY; vocab];
        for entry in mask.iter_mut().take(end).skip(start) {
            *entry = 0.0;
        }
        let mask = Tensor::from_slice(&mask, (vocab,), logits.device())?;
        logits.broadcast_add(&mask)
    }

    /// Generate the first 2 codebooks autoregressively.
    ///
    /// `semantic_tokens`: `[B, T_sem]` semantic tokens emitted by the
    /// semantic stage. Returns `[B, n_codebooks, T_coarse]` where
    /// `T_coarse = sampling.max_coarse_tokens / n_codebooks`.
    ///
    /// Wave B.2 ships the algorithmic skeleton (greedy / multinomial with
    /// temperature + top-k + top-p, alternating-codebook masking via
    /// [`Self::forward`]). Wave B.4 will add the sliding-window context
    /// trimming + KV cache that the upstream Bark coarse generator uses
    /// for long utterances (`modeling_bark.py:850` `for _ in range(n_window_steps)`).
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Synthesis`] for invalid sampling configs
    /// (zero temperature, top-k > vocab, …) and for any internal candle
    /// tensor failure.
    pub fn generate(
        &self,
        semantic_tokens: &Tensor,
        sampling: &CoarseSampling,
    ) -> std::result::Result<Tensor, TtsError> {
        if sampling.temperature <= 0.0 {
            return Err(TtsError::Synthesis(format!(
                "coarse sampling temperature must be > 0, got {}",
                sampling.temperature
            )));
        }
        if !sampling
            .max_coarse_tokens
            .is_multiple_of(self.config.n_codebooks)
        {
            return Err(TtsError::Synthesis(format!(
                "max_coarse_tokens ({}) must be a multiple of n_codebooks ({})",
                sampling.max_coarse_tokens, self.config.n_codebooks
            )));
        }

        let device = semantic_tokens.device().clone();
        let (b, _t_sem) = semantic_tokens
            .dims2()
            .map_err(|e| TtsError::Synthesis(format!("semantic_tokens must be [B, T]: {e}")))?;

        // Working sequence — seed with the semantic prefix.
        let mut tokens = semantic_tokens
            .to_dtype(DType::U32)
            .map_err(|e| TtsError::Synthesis(format!("semantic dtype cast: {e}")))?;

        let mut emitted: Vec<u32> = Vec::with_capacity(sampling.max_coarse_tokens * b);

        for step in 0..sampling.max_coarse_tokens {
            let codebook_idx = step % self.config.n_codebooks;

            // Trim to block_size from the right if the prefix grew too big.
            let prefix_len = tokens
                .dims2()
                .map_err(|e| TtsError::Synthesis(format!("prefix shape: {e}")))?
                .1;
            let tokens_view = if prefix_len > self.config.block_size {
                let offset = prefix_len - self.config.block_size;
                tokens
                    .narrow(1, offset, self.config.block_size)
                    .map_err(|e| TtsError::Synthesis(format!("prefix narrow: {e}")))?
            } else {
                tokens.clone()
            };

            let logits = self
                .forward(&tokens_view, codebook_idx)
                .map_err(|e| TtsError::Synthesis(format!("coarse forward: {e}")))?;

            // [B, T, V] -> [B, V] (last position only).
            let last = tokens_view
                .dims2()
                .map_err(|e| TtsError::Synthesis(format!("view dims: {e}")))?
                .1
                - 1;
            let step_logits = logits
                .i((.., last, ..))
                .map_err(|e| TtsError::Synthesis(format!("logits index: {e}")))?;

            let next = sample_token(&step_logits, sampling)
                .map_err(|e| TtsError::Synthesis(format!("sample: {e}")))?;

            emitted.extend_from_slice(&next);

            // Append the sampled tokens column-wise — shape `[B, 1]`.
            let next_col = Tensor::from_vec(next.clone(), (b, 1), &device)
                .map_err(|e| TtsError::Synthesis(format!("next col tensor: {e}")))?;
            tokens = Tensor::cat(&[&tokens, &next_col], 1)
                .map_err(|e| TtsError::Synthesis(format!("token cat: {e}")))?;
        }

        // Reshape emitted (interleaved cb0, cb1, cb0, cb1, …) into
        // `[B, n_codebooks, T_coarse]`. `emitted` is stored as
        // step-major then batch-major: `[step0_b0, step0_b1, ..., step1_b0, ...]`.
        let frames_per_book = sampling.max_coarse_tokens / self.config.n_codebooks;
        let mut out = vec![0u32; b * self.config.n_codebooks * frames_per_book];
        for step in 0..sampling.max_coarse_tokens {
            let cb = step % self.config.n_codebooks;
            let frame = step / self.config.n_codebooks;
            for bi in 0..b {
                let src = step * b + bi;
                let dst = ((bi * self.config.n_codebooks) + cb) * frames_per_book + frame;
                // Subtract the vocab offset so the output is a raw
                // EnCodec codebook index in `[0, codebook_size)`.
                let raw = emitted[src].saturating_sub(
                    u32::try_from(self.config.vocab_offset_for_codebook(cb)).unwrap_or(0),
                );
                out[dst] = raw;
            }
        }
        Tensor::from_vec(out, (b, self.config.n_codebooks, frames_per_book), &device)
            .map_err(|e| TtsError::Synthesis(format!("output tensor: {e}")))
    }
}

/// Pick one token per batch row from a `[B, V]` logits tensor under the
/// given sampling configuration.
fn sample_token(logits: &Tensor, sampling: &CoarseSampling) -> CandleResult<Vec<u32>> {
    let (b, v) = logits.dims2()?;
    let temperature = f64::from(sampling.temperature);
    let scaled = (logits / temperature)?;
    let probs = ops::softmax(&scaled, D::Minus1)?;
    let probs_vec: Vec<Vec<f32>> = probs.to_vec2()?;

    let mut out = Vec::with_capacity(b);
    for row in probs_vec {
        let mut indexed: Vec<(usize, f32)> = row.into_iter().enumerate().collect();
        // top-k filter
        if let Some(k) = sampling.top_k
            && k < v
        {
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(k);
        }
        // top-p filter (nucleus)
        if let Some(p) = sampling.top_p {
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, (_, prob)) in indexed.iter().enumerate() {
                cum += *prob;
                if cum >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }
        // Take argmax of the surviving set. Wave B.4 will swap in a
        // proper multinomial draw against an RNG; greedy is sufficient
        // for the deterministic shape-correctness path this wave needs.
        let (best_idx, _) = indexed
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        out.push(u32::try_from(best_idx).unwrap_or(0));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn bark_small_coarse_config_matches_upstream() {
        let cfg = CoarseConfig::bark_small();
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.block_size, 1024);
        assert_eq!(cfg.semantic_vocab_size, 10_000);
        assert_eq!(cfg.codebook_size, 1024);
        assert_eq!(cfg.n_codebooks, 2);
        assert_eq!(cfg.input_vocab_size, 12_096);
        assert_eq!(cfg.output_vocab_size, 12_096);
        // HF `suno/bark*` coarse checkpoints are bias-free.
        assert!(!cfg.bias);
    }

    #[test]
    fn vocab_offset_arithmetic_correct() {
        let cfg = CoarseConfig::bark_small();
        assert_eq!(cfg.vocab_offset_for_codebook(0), 10_000);
        assert_eq!(cfg.vocab_offset_for_codebook(1), 10_000 + 1024);
        assert_eq!(
            cfg.vocab_offset_for_codebook(0) + cfg.codebook_size,
            10_000 + 1024
        );
        assert_eq!(cfg.vocab_offset_for_codebook(1) + cfg.codebook_size, 12_048);
    }

    fn tiny_cfg() -> CoarseConfig {
        CoarseConfig {
            semantic_vocab_size: 8,
            codebook_size: 4,
            n_codebooks: 2,
            input_vocab_size: 8 + 2 * 4 + 2,
            output_vocab_size: 8 + 2 * 4 + 2,
            n_layer: 1,
            n_head: 2,
            n_embd: 16,
            block_size: 32,
            bias: true,
        }
    }

    #[test]
    fn forward_pass_shape_correct() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg();
        let decoder = CoarseDecoder::from_vb(vb, cfg.clone()).expect("load tiny coarse");

        let ids =
            Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7, 0, 1], (2, 5), &device).expect("ids");
        let logits = decoder
            .forward(&ids, /* codebook_idx */ 0)
            .expect("forward");
        assert_eq!(logits.dims(), &[2, 5, cfg.output_vocab_size]);

        let raw = decoder.forward(&ids, usize::MAX).expect("forward raw");
        assert_eq!(raw.dims(), &[2, 5, cfg.output_vocab_size]);
    }

    #[test]
    fn forward_masks_to_active_codebook() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg();
        let decoder = CoarseDecoder::from_vb(vb, cfg.clone()).expect("load tiny coarse");

        let ids = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device).expect("ids");
        let logits = decoder
            .forward(&ids, /* codebook_idx */ 1)
            .expect("forward");
        let row: Vec<f32> = logits
            .i((0, 2, ..))
            .expect("last step")
            .to_vec1()
            .expect("to_vec1");

        let start = cfg.vocab_offset_for_codebook(1);
        let end = start + cfg.codebook_size;
        for (i, v) in row.iter().enumerate() {
            if (start..end).contains(&i) {
                assert!(v.is_finite(), "in-range logit {i} should be finite: {v}");
            } else {
                assert!(
                    v.is_infinite() && *v < 0.0,
                    "out-of-range logit {i} should be -inf: {v}"
                );
            }
        }
    }

    #[test]
    fn generate_produces_expected_codebook_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg();
        let decoder = CoarseDecoder::from_vb(vb, cfg.clone()).expect("load tiny coarse");

        let semantic = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).expect("semantic");
        let sampling = CoarseSampling {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            max_coarse_tokens: 8,
        };
        let out = decoder.generate(&semantic, &sampling).expect("generate");
        assert_eq!(out.dims(), &[1, cfg.n_codebooks, 4]);

        let flat: Vec<u32> = out
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");
        for v in flat {
            assert!(
                (v as usize) < cfg.codebook_size,
                "token {v} not in codebook range {}",
                cfg.codebook_size
            );
        }
    }

    #[test]
    fn generate_rejects_zero_temperature() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let decoder = CoarseDecoder::from_vb(vb, tiny_cfg()).expect("load tiny coarse");
        let semantic = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).expect("semantic");
        let sampling = CoarseSampling {
            temperature: 0.0,
            ..CoarseSampling::default()
        };
        let err = decoder
            .generate(&semantic, &sampling)
            .expect_err("zero temp must be rejected");
        match err {
            TtsError::Synthesis(msg) => assert!(msg.contains("temperature"), "msg = {msg}"),
            other => panic!("expected Synthesis, got {other:?}"),
        }
    }
}
