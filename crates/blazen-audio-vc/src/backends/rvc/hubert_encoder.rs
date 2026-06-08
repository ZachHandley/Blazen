//! HuBERT-base transformer encoder layer + relative-positional conv.
//!
//! This module lands the per-layer transformer block of HuBERT-base
//! (`facebook/hubert-base-ls960` / the fairseq Wav2Vec2 family) along
//! with the relative-position conv that fairseq prepends to the
//! encoder stack as `encoder.pos_conv`. Together with the 7-layer
//! `FeatureExtractor` from [`super::feature_extractor`] and the
//! [`super::hubert::HubertBase`] model wrapper these compose the full
//! ContentVec front-end for the RVC backend.
//!
//! # Topology
//!
//! HuBERT-base ships 12 stacked [`HubertEncoderLayer`]s, each carrying:
//!
//! - `self_attn_layer_norm` -- pre-norm before attention
//! - `self_attn` -- 12-head bidirectional self-attention,
//!   hidden 768, head-dim 64, separate `q_proj` / `k_proj` / `v_proj` /
//!   `out_proj` linears (fairseq layout, not the combined-QKV Bark
//!   layout)
//! - `final_layer_norm` -- pre-norm before FFN
//! - `fc1` / `fc2` -- 768 -> 3072 -> 768 FFN with `GELU` (erf form)
//!
//! Forward (pre-norm):
//!
//! ```text
//! h = self_attn_layer_norm(xs)
//! xs = xs + self_attn(h)
//! h = final_layer_norm(xs)
//! xs = xs + fc2(gelu(fc1(h)))
//! ```
//!
//! The activation is the erf-based `gelu_erf`, matching the
//! [`super::feature_extractor::FeatureExtractor`] activation and the
//! fairseq `F.gelu` default. There is no dropout: ContentVec is
//! inference-only inside RVC.
//!
//! # PosConv
//!
//! Before the first encoder layer, HuBERT adds a learned relative-position
//! signal produced by a single depthwise 1-D conv (`groups = 16`, kernel
//! `128`, padding `64`) followed by `GELU`. The conv is stored with
//! `torch.nn.utils.weight_norm` applied -- the checkpoint carries
//! `weight_g` (gain) and `weight_v` (direction) instead of a single
//! `weight`, and we synthesize the dense kernel at load time:
//!
//! `weight = weight_g * (weight_v / ||weight_v||_{axes (1, 2)})`
//!
//! with the norm taken jointly over the input-channel and kernel axes,
//! keeping the output-channel axis 0 independent. This is PyTorch's
//! `weight_norm(dim=0)` convention. The synthesized dense tensor is then
//! handed to a hand-constructed [`Conv1d`] (we bypass `candle_nn`'s
//! builders because they expect a single `weight` key in the
//! [`VarBuilder`], which the weight-normalised checkpoint does not
//! carry).
//!
//! Because the conv kernel is even (`k = 128`), `padding = k / 2 = 64`
//! makes the output exactly one position longer than the input.
//! Fairseq's reference implementation drops the trailing frame in that
//! case (`x = x[:, :, :-1] if k % 2 == 0 else x`) and we mirror that
//! exactly inside [`PosConv::forward`].
//!
//! # Weight layout (fairseq state-dict)
//!
//! Per-layer keys, under `encoder.layers.{i}`:
//!
//! - `self_attn.{q,k,v,out}_proj.{weight,bias}`
//! - `self_attn_layer_norm.{weight,bias}`
//! - `fc1.{weight,bias}`, `fc2.{weight,bias}`
//! - `final_layer_norm.{weight,bias}`
//!
//! PosConv keys, under `encoder.pos_conv.0` (fairseq wraps it in a
//! one-element `Sequential`, hence the `.0`):
//!
//! - `weight_g` -- typically `[768, 1, 1]`, occasionally `[768]`
//!   depending on how the checkpoint was serialised. [`PosConv::load`]
//!   accepts both and reshapes as needed.
//! - `weight_v` -- `[768, 48, 128]` (`out`, `in_per_group = 768 / 16`,
//!   `kernel`).
//! - `bias` -- `[768]`.
//!
//! See [`super::content`]'s state-dict remapping table for the full
//! key list (and the HF transformers counterparts).

#![cfg(feature = "rvc")]
// The architecture docs are dense with proper nouns (`HuBERT`,
// `Wav2Vec2`, `PyTorch`, `GELU`, `LayerNorm`, `PosConv`, `RVC`,
// `ContentVec`); match the file-level allow in `feature_extractor.rs`
// and `content.rs` so the prose reads cleanly.
#![allow(clippy::doc_markdown)]
// `VarBuilder` is consumed by value -- the candle convention. Its inner
// state is `Arc<Inner>`-shaped so the move is cheap, and the upstream
// candle-nn crate uses the same idiom (see `bark/gpt_block.rs` and the
// sibling `feature_extractor.rs`).
#![allow(clippy::needless_pass_by_value)]
// Transformer math is full of (b, t, c) / (q, k, v) tensor names; the
// short single-character bindings match upstream and the math.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use candle_core::{Module, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder, layer_norm, linear};

/// `LayerNorm` epsilon. Fairseq's `LayerNorm`s default to PyTorch's
/// standard `1e-5`, matching [`super::feature_extractor::NORM_EPS`].
const NORM_EPS: f64 = 1e-5;

/// PosConv kernel size (matches fairseq `conv_pos = 128`).
const POS_CONV_KERNEL: usize = 128;

/// PosConv group count (matches fairseq `conv_pos_groups = 16`).
const POS_CONV_GROUPS: usize = 16;

/// Configuration for one [`HubertEncoderLayer`] / [`HubertSelfAttention`]
/// pair. The default for HuBERT-base is exposed via [`Self::HUBERT_BASE`].
#[derive(Debug, Clone, Copy)]
pub(super) struct HubertConfig {
    /// Hidden / residual width (must be divisible by `n_heads`).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// FFN inner dimension (`fc1` out, `fc2` in).
    pub ffn_dim: usize,
}

impl HubertConfig {
    /// Canonical HuBERT-base: `768` hidden, `12` heads, `3072` FFN.
    pub(super) const HUBERT_BASE: Self = Self {
        embed_dim: 768,
        n_heads: 12,
        ffn_dim: 3072,
    };
}

/// Multi-head bidirectional self-attention with **separate** Q/K/V/out
/// projections.
///
/// Mirrors fairseq's `MultiheadAttention` / HF transformers'
/// `Wav2Vec2Attention`. Note that unlike Bark's combined-QKV `att_proj`
/// (see `bark/gpt_block.rs::CausalSelfAttention`), HuBERT carries four
/// separate linears -- the layout the published checkpoint expects.
///
/// All four projections have bias (fairseq Wav2Vec2's default
/// `qkv_bias = True`).
pub(super) struct HubertSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    /// Scale `1 / sqrt(head_dim)` applied to the QK dot product.
    scale: f64,
}

impl HubertSelfAttention {
    /// Load from a [`VarBuilder`] already rooted at the attention
    /// sub-module (the caller does `vb.pp("self_attn")` before invoking
    /// this). Within that root, the four projections live at
    /// `q_proj`, `k_proj`, `v_proj`, `out_proj`.
    ///
    /// # Errors
    ///
    /// Propagates any [`candle_core::Error`] surfaced by the underlying
    /// [`VarBuilder`] (missing tensor, dtype / shape mismatch, etc.).
    pub(super) fn load(vb: VarBuilder, cfg: HubertConfig) -> Result<Self> {
        assert!(
            cfg.embed_dim.is_multiple_of(cfg.n_heads),
            "embed_dim ({}) must be divisible by n_heads ({})",
            cfg.embed_dim,
            cfg.n_heads
        );
        let head_dim = cfg.embed_dim / cfg.n_heads;
        let q_proj = linear(cfg.embed_dim, cfg.embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.embed_dim, cfg.embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.embed_dim, cfg.embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(cfg.embed_dim, cfg.embed_dim, vb.pp("out_proj"))?;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads: cfg.n_heads,
            head_dim,
            embed_dim: cfg.embed_dim,
            scale,
        })
    }

    /// Forward `(B, T, embed_dim) -> (B, T, embed_dim)`.
    ///
    /// Non-causal (bidirectional) -- HuBERT's self-attention sees the
    /// full sequence at every position, so we never construct or apply
    /// a causal mask. Otherwise the math mirrors
    /// `bark/gpt_block.rs::CausalSelfAttention::forward`.
    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, t, c) = xs.dims3()?;
        debug_assert_eq!(c, self.embed_dim);

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        let reshape_heads = |t_in: Tensor| -> Result<Tensor> {
            t_in.reshape((b_sz, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = reshape_heads(q)?;
        let k = reshape_heads(k)?;
        let v = reshape_heads(v)?;

        // Scaled dot-product. (B, n_heads, T, head_dim) x (B, n_heads, head_dim, T)
        // -> (B, n_heads, T, T).
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn = softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;

        // (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, C)
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, t, self.embed_dim))?;
        self.out_proj.forward(&out)
    }
}

/// One HuBERT-base transformer encoder layer (pre-norm).
///
/// See the module-level docs for the four-step pre-norm forward
/// pattern, the fairseq state-dict key layout, and the
/// `gelu_erf` activation choice.
pub(super) struct HubertEncoderLayer {
    self_attn: HubertSelfAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl HubertEncoderLayer {
    /// Load from a [`VarBuilder`] already rooted at the per-layer path
    /// (the caller does `vb.pp("encoder").pp("layers").pp(i)` -- or the
    /// equivalent string form -- before invoking this). Within that
    /// root we walk into `self_attn`, `self_attn_layer_norm`, `fc1`,
    /// `fc2`, and `final_layer_norm`.
    ///
    /// # Errors
    ///
    /// Propagates any [`candle_core::Error`] surfaced by the underlying
    /// [`VarBuilder`] (missing tensor, dtype / shape mismatch, etc.).
    pub(super) fn load(vb: VarBuilder, cfg: HubertConfig) -> Result<Self> {
        let self_attn = HubertSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let self_attn_layer_norm =
            layer_norm(cfg.embed_dim, NORM_EPS, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(cfg.embed_dim, cfg.ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.ffn_dim, cfg.embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.embed_dim, NORM_EPS, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    /// Forward `(B, T, embed_dim) -> (B, T, embed_dim)` (pre-norm).
    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.self_attn_layer_norm.forward(xs)?;
        let h = self.self_attn.forward(&h)?;
        let xs = (xs + h)?;
        let h = self.final_layer_norm.forward(&xs)?;
        // FFN with the erf-based GELU (`F.gelu` upstream), matching the
        // sibling `FeatureExtractor` activation choice.
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        &xs + h
    }
}

/// Relative-position conv prepended to the HuBERT encoder stack
/// (`encoder.pos_conv`).
///
/// See the module-level docs for the weight-norm composition formula,
/// the `weight_g` shape fallback, and the even-kernel slicing policy.
pub(super) struct PosConv {
    /// Synthesised dense conv kernel `[embed_dim, embed_dim / groups, kernel]`.
    /// Reconstructed from the checkpoint's `weight_g` * `weight_v` /
    /// `||weight_v||` at [`PosConv::load`] time.
    conv: Conv1d,
}

impl PosConv {
    /// Load from a [`VarBuilder`] already rooted at the pos-conv path
    /// (the caller does `vb.pp("encoder").pp("pos_conv").pp("0")` --
    /// the trailing `.0` reflects fairseq's one-element `Sequential`
    /// wrapper -- before invoking this). Within that root we read
    /// `weight_g`, `weight_v`, and `bias`.
    ///
    /// # Errors
    ///
    /// Propagates any [`candle_core::Error`] surfaced by the underlying
    /// [`VarBuilder`] (missing tensor, dtype / shape mismatch, etc.) or
    /// the weight-norm tensor arithmetic.
    pub(super) fn load(vb: VarBuilder, embed_dim: usize) -> Result<Self> {
        let in_per_group = embed_dim / POS_CONV_GROUPS;

        // weight_v has the canonical conv shape (out, in/groups, kernel).
        // For HuBERT-base: (768, 48, 128).
        let weight_v = vb.get((embed_dim, in_per_group, POS_CONV_KERNEL), "weight_v")?;

        // fairseq's pos-conv uses `weight_norm(conv, name="weight",
        // dim=2)`, so `weight_g` is a *per-kernel-position* gain of shape
        // [1, 1, kernel] (= [1, 1, 128]) — NOT the per-output-channel
        // [out, 1, 1] of the default `dim=0`. Try that canonical shape,
        // falling back to a flat [kernel] serialisation.
        let weight_g = vb.get((1, 1, POS_CONV_KERNEL), "weight_g").or_else(|_| {
            vb.get(POS_CONV_KERNEL, "weight_g")?
                .reshape((1, 1, POS_CONV_KERNEL))
        })?;

        // Bias is a normal per-output-channel tensor.
        let bias = vb.get(embed_dim, "bias")?;

        // PyTorch `weight_norm(dim=2)`: synthesize
        //   weight = weight_g * weight_v / ||weight_v||
        // where the L2 norm is taken jointly over axes (0, 1) -- the
        // output-channel and input-channel dims -- keeping the kernel
        // position (axis 2) independent. `sum_keepdim((0, 1))` preserves
        // the trailing axis so the broadcast against the [1, 1, kernel]
        // `weight_g` works element-wise.
        let norm = weight_v.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
        let weight = weight_g.broadcast_mul(&weight_v.broadcast_div(&norm)?)?;

        // Construct the Conv1d by hand: `candle_nn::conv1d` expects a
        // single `weight` key in the VarBuilder, which we don't have --
        // we just synthesized the dense tensor.
        let cfg = Conv1dConfig {
            padding: POS_CONV_KERNEL / 2,
            stride: 1,
            dilation: 1,
            groups: POS_CONV_GROUPS,
            ..Conv1dConfig::default()
        };
        let conv = Conv1d::new(weight, Some(bias), cfg);
        Ok(Self { conv })
    }

    /// Forward `(B, T, embed_dim) -> (B, T, embed_dim)`.
    ///
    /// Transposes to `(B, embed_dim, T)` for the depthwise-grouped
    /// 1-D conv, applies `GELU(erf)`, slices off the trailing frame
    /// produced by the even-kernel `padding = kernel / 2` asymmetry,
    /// and transposes back. Matches fairseq's
    /// `x = x[:, :, :-1] if k % 2 == 0 else x` truncation exactly.
    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, t_in, _c) = xs.dims3()?;
        let h = xs.transpose(1, 2)?.contiguous()?;
        let h = self.conv.forward(&h)?;
        let h = h.gelu_erf()?;
        // Even kernel + `padding = k / 2` produces an output that is
        // exactly one frame longer than the input. Drop the trailing
        // frame so the residual `xs + pos_conv(xs)` lines up
        // element-wise upstream.
        let t_out = h.dim(2)?;
        let h = if POS_CONV_KERNEL.is_multiple_of(2) {
            debug_assert_eq!(
                t_out,
                t_in + 1,
                "PosConv even-kernel invariant violated: in {t_in}, out {t_out}"
            );
            h.narrow(2, 0, t_in)?
        } else {
            h
        };
        h.transpose(1, 2)?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn encoder_layer_forward_shape_preserves_btc() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = HubertConfig::HUBERT_BASE;
        let layer = HubertEncoderLayer::load(vb, cfg)?;
        let xs = Tensor::zeros((2, 16, cfg.embed_dim), DType::F32, &device)?;
        let out = layer.forward(&xs)?;
        assert_eq!(out.dims(), &[2, 16, cfg.embed_dim]);
        Ok(())
    }

    #[test]
    fn pos_conv_forward_preserves_seq_len() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let pos = PosConv::load(vb, HubertConfig::HUBERT_BASE.embed_dim)?;
        let xs = Tensor::zeros(
            (1, 20, HubertConfig::HUBERT_BASE.embed_dim),
            DType::F32,
            &device,
        )?;
        // Validates both the (B, T, C) -> (B, C, T) round-trip and the
        // even-kernel trailing-frame truncation: without the truncation
        // the output would be (1, 21, 768).
        let out = pos.forward(&xs)?;
        assert_eq!(out.dims(), &[1, 20, HubertConfig::HUBERT_BASE.embed_dim]);
        Ok(())
    }

    #[test]
    fn hubert_base_config_matches_spec() {
        let cfg = HubertConfig::HUBERT_BASE;
        assert_eq!(cfg.embed_dim, 768);
        assert_eq!(cfg.n_heads, 12);
        assert_eq!(cfg.ffn_dim, 3072);
        // Sanity-check that the head split divides cleanly.
        assert_eq!(cfg.embed_dim % cfg.n_heads, 0);
        assert_eq!(cfg.embed_dim / cfg.n_heads, 64);
    }
}
