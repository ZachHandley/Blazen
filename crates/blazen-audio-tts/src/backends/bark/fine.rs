//! Bark fine acoustic stage — non-autoregressive iterative refinement of
//! `EnCodec` codebooks 2..7 from the two coarse codebooks (and any prior
//! fine-history prompt) emitted by [`super::coarse`].
//!
//! Architecture mirrors `suno-ai/bark/bark/model_fine.py:FineGPT`:
//!
//! * **N codebook embedding tables** (one per `EnCodec` codebook, 8 total
//!   for `bark-small`). At forward time the embeddings for codebooks
//!   `0..=pred_idx` are *summed* per token position
//!   (`bark/model_fine.py:115`: `tok_emb[..., :pred_idx+1].sum(dim=-1)`).
//! * **Learned absolute positional embedding** of length `block_size`
//!   (1024 — matches `bark/model_fine.py:90`).
//! * **`n_layer` GPT-style blocks** with *non-causal* self-attention
//!   (`bark/model_fine.py:NonCausalSelfAttention` calls
//!   `scaled_dot_product_attention(..., is_causal=False)` and the manual
//!   fallback omits the causal mask). Bidirectional attention is the
//!   single architectural difference vs. the semantic/coarse stages and
//!   maps to [`super::gpt_block::BlockConfig::is_causal`] = `false`.
//! * **Per-output-codebook LM heads** (one `Linear(n_embd ->
//!   output_vocab_size)` per codebook in `[n_codes_given, n_codes_total)`
//!   — i.e. heads for codebooks 1..=7 in stock `bark-small`).
//!
//! Generation follows `bark/generation.py:generate_fine`:
//!
//! 1. Stack the (B=)1 sequence of `[n_coarse, T]` coarse tokens with a
//!    zero-padded block of `[N_FINE_CODEBOOKS - n_coarse, T]` rows.
//! 2. Slide a length-`block_size` (=1024) window with stride 512 over the
//!    time axis. For each window, iterate `nn` from `n_coarse` to
//!    `N_FINE_CODEBOOKS - 1` and call `forward(pred_idx=nn, idx=window)`
//!    to obtain logits for that codebook over the window's `T` positions,
//!    then sample. The sampled tokens overwrite the relevant rows in the
//!    in-place input buffer so subsequent `nn` iterations see the freshly
//!    predicted higher codebooks.
//! 3. Strip any short-input padding from the tail and return the
//!    `[B, N_FINE_CODEBOOKS, T]` result.

#![cfg(feature = "bark")]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

use candle_core::{D, IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear_no_bias,
    ops::softmax_last_dim,
};

use crate::error::TtsError;

use super::gpt_block::{Block, BlockConfig};

/// `LayerNorm` epsilon — `PyTorch` default, matches upstream Bark.
const LAYER_NORM_EPS: f64 = 1e-5;

/// Static config for the Bark fine GPT.
///
/// Field names mirror `bark/model_fine.py:FineGPTConfig`:
///
/// * `codebook_size` — `EnCodec` vocab per codebook (1024 for stock Bark).
/// * `n_codebooks` — total `EnCodec` codebooks (8).
/// * `n_codebooks_input` — codebooks supplied by the coarse stage (2 in
///   the default API; the underlying `n_codes_given` config is 1, but the
///   API always feeds two coarse codebooks — see
///   `bark/generation.py:719`).
/// * `n_layer` / `n_head` / `n_embd` — transformer geometry (12 / 12 /
///   768 for `bark-small`; the full `suno/bark` checkpoint bumps these to
///   24 / 16 / 1024 but is not the default config here).
/// * `block_size` — max sequence length (1024). Generation requires a
///   full 1024-token window, padding short inputs (`generate_fine:751`).
#[derive(Debug, Clone)]
pub struct FineConfig {
    pub codebook_size: usize,
    pub n_codebooks: usize,
    pub n_codebooks_input: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub bias: bool,
}

impl FineConfig {
    /// `bark-small` defaults: 12 layers, 12 heads, hidden 768, 8
    /// codebooks of 1024 tokens, 2 coarse-input codebooks, block size
    /// 1024. Matches `bark/model_fine.py:FineGPTConfig` with the
    /// `bark/generation.py` default `n_coarse=2`.
    #[must_use]
    pub fn bark_small() -> Self {
        Self {
            codebook_size: 1024,
            n_codebooks: 8,
            n_codebooks_input: 2,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            block_size: 1024,
            bias: true,
        }
    }

    fn block_cfg(&self) -> BlockConfig {
        BlockConfig {
            n_embd: self.n_embd,
            n_head: self.n_head,
            block_size: self.block_size,
            bias: self.bias,
            is_causal: false,
        }
    }
}

/// Sampling knobs for [`FineDecoder::generate`].
///
/// Upstream defaults `temperature=0.5` (`bark/generation.py:694`) with no
/// top-k / top-p; we expose both so callers can fall back to argmax-style
/// behaviour (`temperature=0.0`) or constrain the distribution.
#[derive(Debug, Clone)]
pub struct FineSampling {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for FineSampling {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            top_k: None,
            top_p: None,
        }
    }
}

/// Bark fine-stage non-autoregressive transformer.
///
/// See module docs for the upstream mapping and architectural rationale.
pub struct FineDecoder {
    config: FineConfig,
    /// One embedding per codebook. HF transformers names this
    /// `input_embeds_layers` (plural — fine stage is the odd one out;
    /// semantic + coarse use the singular `input_embeds_layer`).
    wtes: Vec<Embedding>,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    /// LM heads, one per codebook in `[n_codes_given, n_codebooks)`. In
    /// upstream `n_codes_given=1` so heads cover codebooks 1..=7
    /// (length 7). The API in practice only invokes heads `2..=7` when
    /// `n_coarse=2`; we keep all heads materialised to preserve weight
    /// loading from the upstream checkpoint.
    lm_heads: Vec<Linear>,
    /// Lower bound of the head index space — i.e. codebook 0's logits
    /// are *not* produced (`n_codes_given` in upstream config). Fixed
    /// at 1 for stock Bark; surfaced as a field so weight loading can
    /// remain symmetric with upstream naming.
    n_codes_given: usize,
}

impl FineDecoder {
    /// Construct a [`FineDecoder`] reading weights through `vb`.
    ///
    /// Sub-paths used (mirroring HF transformers `BarkFineModel` state-dict):
    ///
    /// * `input_embeds_layers.<i>` for `i in 0..n_codebooks`
    /// * `position_embeds_layer`
    /// * `layers.<i>` for `i in 0..n_layer`
    /// * `layernorm_final`
    /// * `lm_heads.<i>` for `i in 0..(n_codebooks - n_codes_given)`
    pub fn from_vb(vb: VarBuilder, config: FineConfig) -> CandleResult<Self> {
        let n_codes_given = 1; // upstream FineGPTConfig default
        let mut wtes = Vec::with_capacity(config.n_codebooks);
        for i in 0..config.n_codebooks {
            wtes.push(embedding(
                config.codebook_size,
                config.n_embd,
                vb.pp(format!("input_embeds_layers.{i}")),
            )?);
        }
        let wpe = embedding(
            config.block_size,
            config.n_embd,
            vb.pp("position_embeds_layer"),
        )?;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block::load(
                vb.pp(format!("layers.{i}")),
                config.block_cfg(),
            )?);
        }
        let ln_f = layer_norm(config.n_embd, LAYER_NORM_EPS, vb.pp("layernorm_final"))?;
        let mut lm_heads = Vec::with_capacity(config.n_codebooks - n_codes_given);
        for i in 0..(config.n_codebooks - n_codes_given) {
            // Heads have no bias in upstream (`bias=False`).
            lm_heads.push(linear_no_bias(
                config.n_embd,
                config.codebook_size,
                vb.pp(format!("lm_heads.{i}")),
            )?);
        }
        Ok(Self {
            config,
            wtes,
            wpe,
            blocks,
            ln_f,
            lm_heads,
            n_codes_given,
        })
    }

    /// Forward pass for one codebook.
    ///
    /// * `codebook_inputs`: `[B, n_codebooks, T]` integer tensor (dtype
    ///   u32). Codebooks `0..=pred_codebook_idx` must contain the
    ///   currently-known tokens; codebooks above `pred_codebook_idx` are
    ///   ignored (their embeddings are not summed in).
    /// * `pred_codebook_idx`: index in `[n_codes_given, n_codebooks)`.
    ///
    /// Returns `[B, T, codebook_size]` logits for codebook
    /// `pred_codebook_idx`.
    pub fn forward(
        &self,
        codebook_inputs: &Tensor,
        pred_codebook_idx: usize,
    ) -> CandleResult<Tensor> {
        let (b, n_cb, t) = codebook_inputs.dims3()?;
        assert!(
            pred_codebook_idx >= self.n_codes_given,
            "cannot predict codebook < n_codes_given"
        );
        assert!(
            pred_codebook_idx < self.config.n_codebooks,
            "pred_codebook_idx out of range"
        );
        assert_eq!(
            n_cb, self.config.n_codebooks,
            "codebook dim must equal n_codebooks"
        );
        assert!(
            t <= self.config.block_size,
            "sequence length exceeds block_size"
        );

        // Sum embeddings for codebooks 0..=pred_codebook_idx (upstream
        // `tok_emb[..., :pred_idx+1].sum(dim=-1)` —
        // `bark/model_fine.py:115`).
        let mut x: Option<Tensor> = None;
        for cb in 0..=pred_codebook_idx {
            let toks = codebook_inputs.i((.., cb, ..))?; // [B, T]
            let e = self.wtes[cb].forward(&toks)?; // [B, T, n_embd]
            x = Some(match x {
                None => e,
                Some(acc) => (acc + e)?,
            });
        }
        let mut x = x.expect("pred_codebook_idx >= 0 guarantees at least one embedding");

        // Add absolute positional embedding (1, T, n_embd) -> broadcast.
        #[allow(clippy::cast_possible_truncation)]
        let t_u32 = t as u32;
        let pos = Tensor::arange(0u32, t_u32, x.device())?;
        let pos_emb = self.wpe.forward(&pos)?; // [T, n_embd]
        let pos_emb = pos_emb.unsqueeze(0)?; // [1, T, n_embd]
        x = x.broadcast_add(&pos_emb)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = x.apply(&self.ln_f)?;
        let head_idx = pred_codebook_idx - self.n_codes_given;
        let logits = self.lm_heads[head_idx].forward(&x)?; // [B, T, codebook_size]
        let _ = b; // silences unused — shape is implicit
        Ok(logits)
    }

    /// Iterative non-autoregressive generation: predict codebooks
    /// `n_codebooks_input..n_codebooks` for the given coarse tokens.
    ///
    /// * `coarse_codebooks`: `[B, n_codebooks_input, T]` u32 tensor from
    ///   the coarse stage. `T` may be any positive length; we internally
    ///   pad-then-window to the model's `block_size`
    ///   (`bark/generation.py:751`) and strip the tail when returning.
    /// * `sampling`: temperature + optional top-k / top-p.
    ///
    /// Returns `[B, n_codebooks, T]` u32 tensor (concatenation of the
    /// input coarse codebooks and the predicted fine ones).
    pub fn generate(
        &self,
        coarse_codebooks: &Tensor,
        sampling: &FineSampling,
    ) -> std::result::Result<Tensor, TtsError> {
        self.generate_inner(coarse_codebooks, sampling)
            .map_err(|e| TtsError::Synthesis(format!("bark fine generate failed: {e}")))
    }

    fn generate_inner(&self, coarse: &Tensor, sampling: &FineSampling) -> CandleResult<Tensor> {
        let (b, n_in, t) = coarse.dims3()?;
        assert_eq!(n_in, self.config.n_codebooks_input);
        let n_total = self.config.n_codebooks;
        let block_size = self.config.block_size;
        let device = coarse.device();
        // Pad rows for the un-predicted codebooks with zeros — the
        // sentinel gets overwritten on the very first prediction loop
        // anyway, so the seed value is inert.
        let pad_rows = Tensor::zeros((b, n_total - n_in, t), candle_core::DType::U32, device)?;
        let buf = Tensor::cat(&[coarse.clone(), pad_rows], 1)?; // [B, n_total, T]

        // If T < block_size, append padding columns so we can run a
        // single window. We track `n_remove_from_end` to strip them
        // before returning.
        let (mut buf, n_remove_from_end) = if t < block_size {
            let pad = block_size - t;
            let pad_cols = Tensor::zeros((b, n_total, pad), candle_core::DType::U32, device)?;
            (Tensor::cat(&[buf, pad_cols], 2)?, pad)
        } else {
            (buf, 0)
        };

        // Number of sliding windows (`generation.py:760`). With stride
        // 512 and a forced full first window, `ceil((T-block_size)/512)
        // + 1`, floored at 1.
        let usable_t = buf.dim(2)?;
        let n_loops = if usable_t <= block_size {
            1
        } else {
            ((usable_t - block_size).div_ceil(512)) + 1
        };

        for n in 0..n_loops {
            let start = (n * 512).min(usable_t.saturating_sub(block_size));
            // window: [B, n_total, block_size]
            let window = buf.narrow(2, start, block_size)?;
            let mut window = window.contiguous()?;
            // Predict each non-coarse codebook in turn.
            for pred_idx in n_in..n_total {
                let logits = self.forward(&window, pred_idx)?; // [B, block_size, codebook_size]
                let sampled = sample_tokens(&logits, sampling)?; // [B, block_size]
                // Overwrite the predicted row of the window with the
                // sampled tokens.
                let pre = window.narrow(1, 0, pred_idx)?;
                let mid = sampled.unsqueeze(1)?; // [B, 1, block_size]
                let new_window = if pred_idx + 1 < n_total {
                    let post = window.narrow(1, pred_idx + 1, n_total - pred_idx - 1)?;
                    Tensor::cat(&[pre, mid, post], 1)?
                } else {
                    Tensor::cat(&[pre, mid], 1)?
                };
                window = new_window.contiguous()?;
            }
            // Splice the window back into `buf`.
            let pre = buf.narrow(2, 0, start)?;
            let tail_start = start + block_size;
            let post = if tail_start < usable_t {
                Some(buf.narrow(2, tail_start, usable_t - tail_start)?)
            } else {
                None
            };
            buf = match post {
                Some(p) => Tensor::cat(&[pre, window, p], 2)?,
                None => Tensor::cat(&[pre, window], 2)?,
            };
        }

        // Strip short-input padding columns (`generation.py:780`).
        let out_t = usable_t - n_remove_from_end;
        let out = buf.narrow(2, 0, out_t)?;
        Ok(out)
    }
}

/// Sample one token per `(batch, time)` position from a `[B, T, V]`
/// logit tensor. Returns `[B, T]` u32 tokens.
///
/// Implements temperature scaling followed by optional top-k / top-p
/// truncation, then multinomial sampling. For deterministic / argmax
/// behaviour pass `temperature == 0.0`.
fn sample_tokens(logits: &Tensor, sampling: &FineSampling) -> CandleResult<Tensor> {
    let (b, t, v) = logits.dims3()?;
    // Argmax fast path.
    if sampling.temperature <= 0.0 {
        return logits.argmax(D::Minus1)?.to_dtype(candle_core::DType::U32);
    }
    let scale = 1.0f64 / f64::from(sampling.temperature);
    let scaled = (logits * scale)?;
    let probs = softmax_last_dim(&scaled)?;
    let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0u32; b * t];
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
    let top_k = sampling.top_k;
    let top_p = sampling.top_p;
    for (slot, out_cell) in out.iter_mut().enumerate() {
        let row_start = slot * v;
        let row = &probs_vec[row_start..row_start + v];
        let mut pairs: Vec<(f32, u32)> = row
            .iter()
            .enumerate()
            .map(|(k, &p)| (p, u32::try_from(k).unwrap_or(0)))
            .collect();
        if let Some(k) = top_k
            && k < v
        {
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            pairs.truncate(k);
        }
        if let Some(p_thresh) = top_p {
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (j, (p, _)) in pairs.iter().enumerate() {
                cum += *p;
                keep = j + 1;
                if cum >= p_thresh {
                    break;
                }
            }
            pairs.truncate(keep.max(1));
        }
        let total: f32 = pairs.iter().map(|(p, _)| *p).sum();
        if !(total > 0.0 && total.is_finite()) {
            let (best_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            *out_cell = u32::try_from(best_idx).unwrap_or(0);
            continue;
        }
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        #[allow(clippy::cast_precision_loss)]
        let u_num = (z >> 40) as f32;
        #[allow(clippy::cast_precision_loss)]
        let u_den = (1u64 << 24) as f32;
        let u = u_num / u_den;
        let mut acc = 0.0f32;
        let mut chosen = pairs.last().map_or(0, |(_, idx)| *idx);
        for (p, idx) in &pairs {
            acc += *p / total;
            if u <= acc {
                chosen = *idx;
                break;
            }
        }
        *out_cell = chosen;
    }
    Tensor::from_vec(out, (b, t), logits.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    fn tiny_config() -> FineConfig {
        FineConfig {
            codebook_size: 8,
            n_codebooks: 4,
            n_codebooks_input: 2,
            n_layer: 1,
            n_head: 2,
            n_embd: 8,
            block_size: 16,
            bias: true,
        }
    }

    #[test]
    fn bark_small_fine_config_matches_upstream() {
        let cfg = FineConfig::bark_small();
        assert_eq!(cfg.codebook_size, 1024);
        assert_eq!(cfg.n_codebooks, 8);
        assert_eq!(cfg.n_codebooks_input, 2);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.block_size, 1024);
        assert!(cfg.bias);
    }

    #[test]
    fn forward_pass_shape_correct() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_config();
        let model = FineDecoder::from_vb(vb, cfg.clone()).expect("build");
        let input = Tensor::zeros((1, cfg.n_codebooks, 8), DType::U32, &device).unwrap();
        let logits = model.forward(&input, 2).expect("forward");
        let dims = logits.dims3().unwrap();
        assert_eq!(dims, (1, 8, cfg.codebook_size));
    }

    #[test]
    fn generate_returns_all_codebooks() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_config();
        let model = FineDecoder::from_vb(vb, cfg.clone()).expect("build");
        let coarse = Tensor::zeros((1, cfg.n_codebooks_input, 8), DType::U32, &device).unwrap();
        let out = model
            .generate(
                &coarse,
                &FineSampling {
                    temperature: 0.0,
                    top_k: None,
                    top_p: None,
                },
            )
            .expect("generate");
        let dims = out.dims3().unwrap();
        assert_eq!(dims, (1, cfg.n_codebooks, 8));
    }

    #[test]
    fn fine_blocks_are_non_causal() {
        // Build a tiny model and confirm the block stores is_causal=false.
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_config();
        let model = FineDecoder::from_vb(vb, cfg.clone()).expect("build");
        for block in &model.blocks {
            assert!(
                !block.attn.is_causal,
                "fine stage blocks MUST be non-causal"
            );
        }
    }
}
