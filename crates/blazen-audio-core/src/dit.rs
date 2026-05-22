//! Generic DiT primitives: multi-head attention (self + cross) and a
//! SwiGLU feed-forward block.
//!
//! These mirror the building blocks used by the Stable Audio Open Small
//! reference DiT, but are intentionally agnostic about the surrounding
//! block recipe (whether AdaLN is applied, whether RoPE is supplied,
//! whether a cross-attention pass is interleaved, etc.). Consumers
//! (audio + 3D DiT ports) assemble these into the specific recipe
//! their model uses.
//!
//! # What's intentionally NOT here
//!
//! - Block-level recipes (e.g. `pre_norm → modulate → self_attn →
//!   gate-blend → residual + cross_attn + AdaLN-FF`). Those bake in
//!   backend-specific gating choices (`sigmoid(1 - gate)` vs
//!   `tanh(gate)` vs plain `gate`) and should live with the consuming
//!   backend.
//! - Backend-specific hyperparameter packs and the I/O projections
//!   sized to specific latent shapes.

#![allow(clippy::similar_names)]
#![allow(clippy::module_name_repetitions)]
// `VarBuilder` is the candle convention for module construction — every
// existing backend takes it by value, and the cost of a move is one
// cheap clone of an `Arc`.
#![allow(clippy::needless_pass_by_value)]
// `b`, `t`, `c`, `h`, `d` are the canonical batch/time/channel/head/dim
// names used by every PyTorch port; renaming them would diverge from
// references and hurt cross-checking.
#![allow(clippy::many_single_char_names)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};

use crate::rope::apply_rope;

/// Bias-less LayerNorm with a learnable per-channel `gamma` only.
///
/// The Stable Audio reference uses `F.layer_norm(x, x.shape[-1:],
/// weight=gamma, bias=None)`. `candle_nn::LayerNorm::new_no_bias` is
/// the candle equivalent. Exposed here so DiT backends can share the
/// same factory.
pub fn layer_norm_no_bias(dim: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let gamma = vb.get(dim, "gamma")?;
    Ok(candle_nn::LayerNorm::new_no_bias(gamma, eps))
}

/// Fully-affine LayerNorm with learnable `gamma` (weight) and `beta`
/// (bias). The Stable Audio reference only uses bias-less LayerNorms
/// but the helper is kept for future DiT variants that need it.
pub fn layer_norm_affine(dim: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let gamma = vb.get(dim, "gamma")?;
    let beta = vb.get(dim, "beta")?;
    Ok(candle_nn::LayerNorm::new(gamma, beta, eps))
}

/// Multi-head attention block (self or cross). Bias-less Q/K/V/output
/// projections. Self-attention uses a fused QKV linear and accepts
/// RoPE inputs; cross-attention uses a split `q` plus fused `kv`
/// linear and does *not* apply RoPE.
#[derive(Debug)]
pub struct Attention {
    /// Self-attn QKV fused projection (`embed_dim → 3 * embed_dim`).
    /// `None` for cross-attention.
    qkv: Option<Linear>,
    /// Cross-attn Q projection (`embed_dim → embed_dim`). `None` for
    /// self-attention.
    q_proj: Option<Linear>,
    /// Cross-attn KV fused projection (`kv_dim → 2 * kv_dim`). `None`
    /// for self-attention.
    kv_proj: Option<Linear>,
    /// Output projection (`embed_dim → embed_dim`).
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    is_cross: bool,
    qk_norm: bool,
    /// 1.0 / sqrt(head_dim), pre-computed.
    scale: f64,
}

impl Attention {
    /// Construct a self-attention block.
    ///
    /// The fused QKV weight lives at `vb / to_qkv` and the output
    /// projection at `vb / to_out` — matching the Stable Audio
    /// reference naming.
    pub fn new_self(
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert_eq!(embed_dim, num_heads * head_dim);
        let qkv = linear_no_bias(embed_dim, embed_dim * 3, vb.pp("to_qkv"))?;
        let out_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("to_out"))?;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            qkv: Some(qkv),
            q_proj: None,
            kv_proj: None,
            out_proj,
            num_heads,
            head_dim,
            is_cross: false,
            qk_norm,
            scale,
        })
    }

    /// Construct a cross-attention block. `kv_dim` is the context hidden
    /// dim (e.g. the T5 hidden size for Stable Audio).
    ///
    /// Weights live at `vb / to_q`, `vb / to_kv`, and `vb / to_out`.
    pub fn new_cross(
        embed_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        head_dim: usize,
        qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert_eq!(embed_dim, num_heads * head_dim);
        let q_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("to_q"))?;
        let kv_proj = linear_no_bias(kv_dim, kv_dim * 2, vb.pp("to_kv"))?;
        let out_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("to_out"))?;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            qkv: None,
            q_proj: Some(q_proj),
            kv_proj: Some(kv_proj),
            out_proj,
            num_heads,
            head_dim,
            is_cross: true,
            qk_norm,
            scale,
        })
    }

    /// Reshape `(B, T, H*D)` → `(B, H, T, D)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        x.reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Reshape `(B, H, T, D)` → `(B, T, H*D)`.
    fn merge_heads(x: &Tensor) -> Result<Tensor> {
        let (b, h, t, d) = x.dims4()?;
        x.transpose(1, 2)?.contiguous()?.reshape((b, t, h * d))
    }

    /// L2-normalize along the last axis.
    fn l2_normalize(t: &Tensor) -> Result<Tensor> {
        let sq = t.sqr()?;
        let sum = sq.sum_keepdim(D::Minus1)?;
        let norm = (sum + 1e-12)?.sqrt()?;
        t.broadcast_div(&norm)
    }

    /// Forward pass.
    ///
    /// - Self-attn: `kv_input` should be `None`. RoPE is applied to Q/K
    ///   if `rope` is `Some((cos, sin))`.
    /// - Cross-attn: `kv_input` is the context `(B, T_ctx, kv_dim)`.
    ///   RoPE tables are ignored.
    pub fn forward(
        &self,
        x: &Tensor,
        kv_input: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, t_q, _) = x.dims3()?;
        let (q, k, v) = if self.is_cross {
            let kv = kv_input.expect("cross-attention requires a kv_input");
            let q = self
                .q_proj
                .as_ref()
                .expect("cross-attn has q_proj")
                .forward(x)?;
            let kv = self
                .kv_proj
                .as_ref()
                .expect("cross-attn has kv_proj")
                .forward(kv)?;
            let kv_dim = kv.dim(D::Minus1)? / 2;
            let k = kv.narrow(D::Minus1, 0, kv_dim)?;
            let v = kv.narrow(D::Minus1, kv_dim, kv_dim)?;
            (q, k, v)
        } else {
            let qkv = self.qkv.as_ref().expect("self-attn has qkv").forward(x)?;
            let d = qkv.dim(D::Minus1)? / 3;
            let q = qkv.narrow(D::Minus1, 0, d)?;
            let k = qkv.narrow(D::Minus1, d, d)?;
            let v = qkv.narrow(D::Minus1, 2 * d, d)?;
            (q, k, v)
        };

        let mut q = self.split_heads(&q)?;
        let mut k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        if self.qk_norm {
            q = Self::l2_normalize(&q)?;
            k = Self::l2_normalize(&k)?;
        }

        if !self.is_cross
            && let Some((cos, sin)) = rope
        {
            // RoPE is cast to f32 inside in the Stable Audio reference;
            // honour that for numerical stability.
            let orig_dtype = q.dtype();
            let q_f = q.to_dtype(DType::F32)?;
            let k_f = k.to_dtype(DType::F32)?;
            let q_rot = apply_rope(&q_f, cos, sin)?.to_dtype(orig_dtype)?;
            let k_rot = apply_rope(&k_f, cos, sin)?.to_dtype(orig_dtype)?;
            q = q_rot;
            k = k_rot;
        }

        // Scaled dot-product attention. (B, H, T_q, D) @ (B, H, D, T_k)
        // → (B, H, T_q, T_k).
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_scores = q.matmul(&k_t)?;
        let attn_scores = (attn_scores * self.scale)?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let out = attn_probs.matmul(&v.contiguous()?)?;

        let out = Self::merge_heads(&out)?;
        debug_assert_eq!(out.dim(1)?, t_q);
        debug_assert_eq!(out.dim(0)?, b);
        self.out_proj.forward(&out)
    }
}

/// SwiGLU feed-forward block: `Linear(d, 2*inner) -> chunk -> x *
/// silu(g) -> Linear(inner, d)`.
///
/// The Stable Audio reference's GLU module fuses the chunk +
/// activation; we replicate the parameter layout exactly so the
/// safetensors keys line up. Sub-tensor names are `ff.0.proj` and
/// `ff.2` — these correspond to the Sequential indices in the upstream
/// PyTorch module and are baked into the existing safetensors weight
/// dumps, so we keep them stable here.
#[derive(Debug)]
pub struct FeedForward {
    /// `embed_dim → 2 * inner_dim`. Stored as `ff.0.proj` in the
    /// reference.
    glu_proj: Linear,
    /// `inner_dim → embed_dim`. Stored as `ff.2` in the reference (the
    /// `nn.Sequential` index for the final linear).
    out_proj: Linear,
}

impl FeedForward {
    /// Construct from a `VarBuilder`. The sub-paths `ff.0.proj` and
    /// `ff.2` are baked in to preserve safetensors key compatibility
    /// with the Stable Audio reference.
    pub fn new(embed_dim: usize, inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        let glu_proj = linear_no_bias(embed_dim, inner_dim * 2, vb.pp("ff.0.proj"))?;
        let out_proj = linear_no_bias(inner_dim, embed_dim, vb.pp("ff.2"))?;
        Ok(Self { glu_proj, out_proj })
    }

    /// Forward: `(B, T, embed_dim) → (B, T, embed_dim)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.glu_proj.forward(x)?;
        let half = projected.dim(D::Minus1)? / 2;
        let value = projected.narrow(D::Minus1, 0, half)?;
        let gate = projected.narrow(D::Minus1, half, half)?;
        let activated = (value * candle_nn::ops::silu(&gate)?)?;
        self.out_proj.forward(&activated)
    }
}
