//! Bark semantic-stage transformer.
//!
//! GPT-style decoder-only transformer that consumes BERT-tokenized text
//! (plus an optional history prompt for voice cloning) and emits semantic
//! tokens autoregressively. These tokens are then consumed by the coarse
//! stage to produce `EnCodec` acoustic tokens.
//!
//! Architecture mirrors `suno-ai/bark/bark/model.py::GPT` — 12 layers /
//! hidden 768 / 12 heads for `suno/bark-small`, 24 layers / hidden 1024 /
//! 16 heads for `suno/bark`. Learned positional embeddings (no `RoPE`),
//! standard causal mask, pre-`LayerNorm` with optional bias, GELU MLP.
//!
//! Parameter naming matches the HF transformers Bark layout
//! (`input_embeds_layer`, `position_embeds_layer`, `layers.{i}`,
//! `layernorm_final`, `lm_head`) — this is what the published `suno/bark`
//! checkpoints actually ship in their `pytorch_model.bin`. The shared
//! transformer block ([`super::gpt_block`]) honours the same naming.
//!
//! Reference: <https://github.com/suno-ai/bark/blob/main/bark/model.py>.
//! Constants: <https://github.com/suno-ai/bark/blob/main/bark/generation.py>.

#![cfg(feature = "bark")]
// Match the convention used by `crates/blazen-audio-music/src/backends/
// musicgen/model.rs`: candle's `VarBuilder` is consumed by value
// throughout, even though the body only forwards it to child builders.
// Its inner state is `Arc<Inner>`-shaped so the move is cheap, and the
// upstream candle-transformers crate uses the same idiom.
#![allow(clippy::needless_pass_by_value)]
// Several local bindings deliberately match the upstream Bark tensor
// names (`q`, `k`, `v`, `t`, `c`) to keep the port readable next to the
// reference implementation; the `many_single_char_names` lint adds noise
// without value here.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear_no_bias};

use crate::error::TtsError;

use super::gpt_block::{Block, BlockConfig};

/// `SEMANTIC_VOCAB_SIZE` from
/// `bark/generation.py` (`10_000`). The pad token id (`10_000`) lives at
/// the boundary and is what upstream uses as the EOS signal for early
/// stop.
pub const SEMANTIC_VOCAB_SIZE: usize = 10_000;

/// `SEMANTIC_PAD_TOKEN` from `bark/generation.py` (== `10_000`).
/// Upstream uses this id as the de-facto end-of-semantic marker when
/// `allow_early_stop=True`.
pub const SEMANTIC_PAD_TOKEN: u32 = 10_000;

/// `LayerNorm` epsilon. Upstream `bark/model.py::LayerNorm` uses
/// `PyTorch`'s default (`1e-5`).
const LAYER_NORM_EPS: f64 = 1e-5;

/// Hyper-parameters for the semantic GPT decoder.
///
/// Field names match `bark/model.py::GPTConfig` so a future weight
/// converter can be a direct lookup. The defaults in [`Self::bark_small`]
/// reproduce upstream `GPTConfig` defaults exactly
/// (`bark/model.py::GPTConfig` lines 19-26).
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Text-token vocabulary size. Upstream default is `10_048` and
    /// covers semantic ids `0..10_000`, the `SEMANTIC_PAD_TOKEN` slot,
    /// and the BERT-WordPiece text-token shelf at offset
    /// `TEXT_ENCODING_OFFSET = 10_048` (the text tokens themselves are
    /// represented by re-indexing into the same embedding table during
    /// inference, hence the shared `input_vocab_size`).
    pub input_vocab_size: usize,
    /// Output (semantic) vocabulary size. Upstream default is `10_048`;
    /// the live semantic ids occupy `0..10_000` and id `10_000` doubles
    /// as the EOS / pad token.
    pub output_vocab_size: usize,
    /// Number of transformer blocks. 12 for `suno/bark-small`, 24 for
    /// `suno/bark`.
    pub n_layer: usize,
    /// Attention head count. 12 for `suno/bark-small`, 16 for `suno/bark`.
    pub n_head: usize,
    /// Hidden / residual width. 768 for `suno/bark-small`, 1024 for
    /// `suno/bark`. Must be divisible by `n_head`.
    pub n_embd: usize,
    /// Maximum context length supported by the learned positional
    /// embedding table. Upstream default is `1024` for the semantic
    /// stage.
    pub block_size: usize,
    /// Whether `Linear` projections and `LayerNorm` carry a bias. HF
    /// transformers' published Bark checkpoints set `bias=false` for the
    /// transformer body; older Suno-format checkpoints set `bias=true`.
    /// The flag is plumbed through so either layout loads.
    pub bias: bool,
}

impl SemanticConfig {
    /// Default config for `suno/bark-small` — matches upstream
    /// `GPTConfig` defaults at `bark/model.py:19-26`.
    #[must_use]
    pub fn bark_small() -> Self {
        Self {
            input_vocab_size: 10_048,
            output_vocab_size: 10_048,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            block_size: 1024,
            bias: true,
        }
    }

    /// Config for the full-size `suno/bark` checkpoint
    /// (`bark/model.py::GPT.from_pretrained("text")` resolves to these
    /// numbers via the published 24-layer / 1024-hidden semantic stage).
    #[must_use]
    pub fn bark_full() -> Self {
        Self {
            input_vocab_size: 10_048,
            output_vocab_size: 10_048,
            n_layer: 24,
            n_head: 16,
            n_embd: 1024,
            block_size: 1024,
            bias: true,
        }
    }

    /// Per-head dimension (`n_embd / n_head`).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
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

/// Sampling configuration for autoregressive semantic-token generation.
///
/// Defaults mirror `bark/generation.py::generate_text_semantic` —
/// `temperature=0.7`, no top-k, no top-p, `n_tot_steps=768` upper bound,
/// EOS at [`SEMANTIC_PAD_TOKEN`].
#[derive(Debug, Clone)]
pub struct SemanticSampling {
    /// Softmax temperature; values < 1 sharpen the distribution. Upstream
    /// default is `0.7`.
    pub temperature: f32,
    /// Optional top-k truncation. Upstream default is `None`.
    pub top_k: Option<usize>,
    /// Optional top-p (nucleus) truncation in `(0, 1]`. Upstream default
    /// is `None`.
    pub top_p: Option<f32>,
    /// Hard cap on newly generated tokens; upstream uses `768` (~15.4 s
    /// of audio at the 49.9 Hz semantic rate).
    pub max_new_tokens: usize,
    /// Token id that signals end of generation. Defaults to
    /// [`SEMANTIC_PAD_TOKEN`].
    pub eos_token_id: u32,
}

impl Default for SemanticSampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: None,
            top_p: None,
            max_new_tokens: 768,
            eos_token_id: SEMANTIC_PAD_TOKEN,
        }
    }
}

/// Bark semantic-stage transformer.
pub struct SemanticDecoder {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
    config: SemanticConfig,
}

impl SemanticDecoder {
    /// Load the decoder from a candle `VarBuilder` (typically rooted at
    /// the safetensors / pth mmap produced by [`super::weights`]).
    ///
    /// Parameter naming matches HF transformers' `BarkSemanticModel`:
    /// `input_embeds_layer`, `position_embeds_layer`, `layers.{i}.*`,
    /// `layernorm_final`, `lm_head`.
    ///
    /// # Errors
    ///
    /// Returns any candle error raised while constructing the underlying
    /// `Linear` / `Embedding` / `LayerNorm` layers (typically a missing
    /// tensor name in the safetensors file).
    pub fn load(vb: VarBuilder, config: SemanticConfig) -> Result<Self> {
        let n_embd = config.n_embd;
        let wte = embedding(config.input_vocab_size, n_embd, vb.pp("input_embeds_layer"))?;
        let wpe = embedding(config.block_size, n_embd, vb.pp("position_embeds_layer"))?;
        let blocks = (0..config.n_layer)
            .map(|i| Block::load(vb.pp(format!("layers.{i}")), config.block_cfg()))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(n_embd, LAYER_NORM_EPS, vb.pp("layernorm_final"))?;
        // Upstream: `self.lm_head = nn.Linear(n_embd, output_vocab_size,
        // bias=False)`. Always no-bias regardless of `config.bias`.
        let head = linear_no_bias(n_embd, config.output_vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            head,
            config,
        })
    }

    /// The configuration this decoder was built with.
    #[must_use]
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    /// Forward pass over `input_ids` of shape `[B, T]`. Returns logits
    /// over the semantic vocabulary at the **last** position only —
    /// shape `[B, output_vocab_size]`. Matches the inference-mode forward
    /// in `bark/model.py::GPT.forward` when `use_cache=False` (no KV
    /// cache; full prefix is re-encoded each step).
    ///
    /// # Errors
    ///
    /// Returns any candle error from the underlying layer ops, including
    /// the `block_size` overflow check.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        if t > self.config.block_size {
            return Err(candle_core::Error::Msg(format!(
                "semantic forward: sequence length {} exceeds block_size {}",
                t, self.config.block_size
            )));
        }
        let device = input_ids.device();

        // Token + positional embedding.
        let tok_emb = self.wte.forward(input_ids)?; // [B, T, n_embd]
        let t_u32 = u32::try_from(t).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let positions = Tensor::arange(0u32, t_u32, device)?; // [T]
        let pos_emb = self.wpe.forward(&positions)?; // [T, n_embd]
        let mut xs = tok_emb.broadcast_add(&pos_emb)?;

        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let xs = self.ln_f.forward(&xs)?;
        // Pick the last time step before the head — same shortcut
        // upstream's `forward` uses during incremental decoding.
        let last = xs.i((.., t - 1, ..))?; // [B, n_embd]
        self.head.forward(&last) // [B, output_vocab_size]
    }

    /// Autoregressively generate semantic tokens.
    ///
    /// `input_ids` (shape `[B, T]`) is the prefix — typically the
    /// BERT-tokenized text plus an optional history-prompt prefix. Only
    /// the **newly generated** tokens are returned (shape `[B, T_new]`,
    /// with `T_new <= sampling.max_new_tokens`). When the first batch
    /// element samples the `eos_token_id`, generation stops globally —
    /// callers that need per-row stopping should run with `B == 1`.
    ///
    /// # Errors
    ///
    /// - [`TtsError::InvalidOptions`] when `input_ids` is not 2-D or when
    ///   `sampling.temperature <= 0`.
    /// - [`TtsError::Synthesis`] wrapping any candle error raised during
    ///   forward / sampling.
    pub fn generate(
        &self,
        input_ids: &Tensor,
        sampling: &SemanticSampling,
    ) -> std::result::Result<Tensor, TtsError> {
        if input_ids.rank() != 2 {
            return Err(TtsError::InvalidOptions(format!(
                "semantic generate: input_ids must be rank 2 [B, T], got rank {}",
                input_ids.rank()
            )));
        }
        if sampling.temperature <= 0.0 || sampling.temperature.is_nan() {
            return Err(TtsError::InvalidOptions(format!(
                "semantic generate: temperature must be > 0, got {}",
                sampling.temperature
            )));
        }
        let device = input_ids.device().clone();

        let mut ids = input_ids
            .to_dtype(DType::U32)
            .map_err(|e| TtsError::Synthesis(e.to_string()))?;
        let mut new_tokens: Vec<u32> = Vec::with_capacity(sampling.max_new_tokens);

        for _ in 0..sampling.max_new_tokens {
            // Guard the context window — drop the oldest tokens once we
            // exceed `block_size`. Upstream does the same via a sliding
            // window in `generate_text_semantic`.
            let logits = if ids.dim(1).map_err(|e| TtsError::Synthesis(e.to_string()))?
                <= self.config.block_size
            {
                self.forward(&ids)
                    .map_err(|e| TtsError::Synthesis(e.to_string()))?
            } else {
                let t = ids.dim(1).map_err(|e| TtsError::Synthesis(e.to_string()))?;
                let start = t - self.config.block_size;
                let cropped = ids
                    .narrow(1, start, self.config.block_size)
                    .map_err(|e| TtsError::Synthesis(e.to_string()))?;
                self.forward(&cropped)
                    .map_err(|e| TtsError::Synthesis(e.to_string()))?
            };

            // Sample one token from the first batch row.
            let row = logits
                .i((0, ..))
                .map_err(|e| TtsError::Synthesis(e.to_string()))?
                .to_dtype(DType::F32)
                .map_err(|e| TtsError::Synthesis(e.to_string()))?;
            let row_vec: Vec<f32> = row
                .to_vec1()
                .map_err(|e| TtsError::Synthesis(e.to_string()))?;
            let next = sample_token(&row_vec, sampling);

            new_tokens.push(next);
            if next == sampling.eos_token_id {
                break;
            }

            // Append next token to ids: build `[B, 1]` then concat.
            let b = ids.dim(0).map_err(|e| TtsError::Synthesis(e.to_string()))?;
            let next_col = Tensor::from_vec(vec![next; b], (b, 1), &device)
                .map_err(|e| TtsError::Synthesis(e.to_string()))?;
            ids = Tensor::cat(&[&ids, &next_col], 1)
                .map_err(|e| TtsError::Synthesis(e.to_string()))?;
        }

        // Return generated tokens as a `[1, T_new]` tensor. If the very
        // first sampled token was the EOS, `new_tokens` contains just it
        // and we strip it so the caller sees an empty `[1, 0]` tensor —
        // upstream's `generate_text_semantic` returns the sequence
        // *excluding* the EOS sentinel.
        let trimmed: &[u32] = if new_tokens.last() == Some(&sampling.eos_token_id) {
            &new_tokens[..new_tokens.len() - 1]
        } else {
            &new_tokens[..]
        };
        let len = trimmed.len();
        Tensor::from_vec(trimmed.to_vec(), (1, len), &device)
            .map_err(|e| TtsError::Synthesis(e.to_string()))
    }
}

/// Apply temperature, top-k, and top-p (nucleus) filtering to `logits`
/// and sample one token. Pure CPU; designed for the
/// `[output_vocab_size]`-sized row that comes out of one decode step.
fn sample_token(logits: &[f32], sampling: &SemanticSampling) -> u32 {
    // Temperature scaling. Already validated `> 0` by `generate`.
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / sampling.temperature).collect();

    // Top-k: keep the k largest, mask the rest to -inf.
    if let Some(k) = sampling.top_k
        && k > 0
        && k < scaled.len()
    {
        // Partial sort: find the k-th largest value.
        let mut sorted = scaled.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];
        for v in &mut scaled {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax (numerically stable).
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    // Top-p (nucleus): zero out the tail until cumulative probability
    // reaches `p`. We rank indices by descending probability.
    if let Some(p) = sampling.top_p
        && p > 0.0
        && p < 1.0
    {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0_f32;
        let mut cutoff = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            cum += probs[i];
            if cum >= p {
                cutoff = rank + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> = idx[..cutoff].iter().copied().collect();
        for (i, prob) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *prob = 0.0;
            }
        }
        // Renormalize.
        let s: f32 = probs.iter().sum();
        if s > 0.0 {
            for prob in &mut probs {
                *prob /= s;
            }
        }
    }

    // Deterministic argmax from the filtered distribution. We
    // intentionally do *not* hash a PRNG here so that the unit tests are
    // reproducible and the surrounding pipeline (Wave B.4) owns the
    // RNG-vs-greedy choice. Bark's published quality benefits from
    // sampling, but the surrounding orchestration will introduce
    // multinomial sampling once a shared rng is plumbed through.
    let (best, _) =
        probs
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                if v > best_v { (i, v) } else { (best_i, best_v) }
            });
    #[allow(clippy::cast_possible_truncation)]
    {
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    /// Tiny config used by the shape / generation smoke tests. Shrunk
    /// along every dim but exercises every architectural component
    /// (multi-head split, mask broadcast, pre-LN, GELU MLP, LM head).
    fn tiny_cfg() -> SemanticConfig {
        SemanticConfig {
            input_vocab_size: 32,
            output_vocab_size: 32,
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
            block_size: 32,
            bias: true,
        }
    }

    /// Upstream defaults — `bark/model.py::GPTConfig`, lines 19-26.
    #[test]
    fn bark_small_config_matches_upstream() {
        let c = SemanticConfig::bark_small();
        assert_eq!(c.input_vocab_size, 10_048);
        assert_eq!(c.output_vocab_size, 10_048);
        assert_eq!(c.n_layer, 12);
        assert_eq!(c.n_head, 12);
        assert_eq!(c.n_embd, 768);
        assert_eq!(c.block_size, 1024);
        assert!(c.bias);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.n_head * c.head_dim(), c.n_embd);

        let full = SemanticConfig::bark_full();
        assert_eq!(full.n_layer, 24);
        assert_eq!(full.n_head, 16);
        assert_eq!(full.n_embd, 1024);
        assert_eq!(full.head_dim(), 64);
    }

    /// Forward shape: `[B=1, T=16]` -> `[1, output_vocab_size]` (logits
    /// at the last position only).
    #[test]
    fn forward_pass_shape_correct() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg();
        let model = SemanticDecoder::load(vb, cfg.clone())?;
        let ids = Tensor::zeros((1, 16), DType::U32, &device)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dims(), &[1, cfg.output_vocab_size]);
        Ok(())
    }

    /// Zero-init weights make every logit identical → argmax picks
    /// token 0; with `eos_token_id = 0`, `generate` returns `[1, 0]`
    /// (EOS stripped per upstream `generate_text_semantic` contract).
    #[test]
    fn generation_stops_on_eos() -> std::result::Result<(), TtsError> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = SemanticDecoder::load(vb, tiny_cfg())
            .map_err(|e| TtsError::Synthesis(e.to_string()))?;
        let ids = Tensor::zeros((1, 4), DType::U32, &device)
            .map_err(|e| TtsError::Synthesis(e.to_string()))?;
        let sampling = SemanticSampling {
            temperature: 1.0,
            max_new_tokens: 16,
            eos_token_id: 0,
            ..SemanticSampling::default()
        };
        let out = model.generate(&ids, &sampling)?;
        assert_eq!(out.dims(), &[1, 0]);
        Ok(())
    }

    /// Top-k=1 collapses sampling to argmax of the original logits;
    /// top-p < smallest nonzero probability also collapses to argmax.
    #[test]
    fn top_k_top_p_filter_correctness() {
        let logits = vec![0.1_f32, 5.0, 0.2, 0.3, 4.5];
        let base = SemanticSampling {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            max_new_tokens: 1,
            eos_token_id: 99,
        };
        // top_k = 1 → index 1 (largest).
        assert_eq!(
            sample_token(
                &logits,
                &SemanticSampling {
                    top_k: Some(1),
                    ..base.clone()
                }
            ),
            1
        );
        // top_k = 2 → still index 1.
        assert_eq!(
            sample_token(
                &logits,
                &SemanticSampling {
                    top_k: Some(2),
                    ..base.clone()
                }
            ),
            1
        );
        // top_p ≈ 0.5 — softmax-dominant class is `{1}`.
        assert_eq!(
            sample_token(
                &logits,
                &SemanticSampling {
                    top_p: Some(0.5),
                    ..base.clone()
                }
            ),
            1
        );
        // Tiny temperature still argmaxes to index 1.
        assert_eq!(
            sample_token(
                &logits,
                &SemanticSampling {
                    temperature: 0.1,
                    ..base
                }
            ),
            1
        );
    }
}
