//! Diffusion Transformer (DiT) for Stable Audio Open Small.
//!
//! Native candle port of the latent-space diffusion transformer used by
//! `stabilityai/stable-audio-open-small`. Mirrors the reference
//! implementation in
//! [`stable-audio-tools/stable_audio_tools/models/dit.py`][saoref] (and the
//! cleaned ComfyUI mirror at `comfy/ldm/audio/dit.py`) tensor-for-tensor so
//! that block-level numerical-parity tests against a Python dump harness
//! can validate every sub-component.
//!
//! [saoref]: https://github.com/Stability-AI/stable-audio-tools
//!
//! # Architecture summary
//!
//! - Latent IO: `(B, 64, T)` audio latents at ~86 Hz (Small variant).
//! - Embed dim 768, depth 12, 8 heads (head_dim 96). Stable Audio uses
//!   `dim_heads // 2` rotary dims, so RoPE operates on the first 48
//!   components of each head's Q/K (partial RoPE, GPT-J style).
//! - Conditioning: T5-base text tokens (cross-attention) plus a global
//!   `(timestep + seconds_start + seconds_total)` embedding injected via
//!   AdaLN modulation on the self-attn and feed-forward pre-norms.
//! - Each block: `(self_attn with AdaLN + RoPE) + (cross_attn, no AdaLN)
//!   + (SwiGLU FFN with AdaLN)`. Bias-less LayerNorms throughout.
//! - The gating is `sigmoid(1 - gate)`, NOT `tanh(gate)` or plain `gate`
//!   — this matches the upstream reference exactly and is a common
//!   silent-drift trap.
//! - V-objective: the model returns velocity, converted by the sampler
//!   via `x0 = alpha*x_t - sigma*v` / `eps = sigma*x_t + alpha*v`.
//!
//! # Status
//!
//! Wave 3.1 of the Stable Audio port (PR-AUDIO follow-up). This module is
//! *not yet* wired into `super::mod`; that happens in Wave 3.5 once the
//! Oobleck VAE, conditioner, sampler, and pipeline land.

#![cfg(feature = "stable-audio")]
#![allow(clippy::similar_names)] // shift_msa / scale_msa / gate_msa etc.
#![allow(clippy::module_name_repetitions)]
// `VarBuilder` is the candle convention for module construction — every
// existing backend in this crate takes it by value, and the cost of a
// move is one cheap clone of an `Arc`. Mirror that here.
#![allow(clippy::needless_pass_by_value)]
// `b`, `t`, `c`, `h`, `d` are the canonical batch/time/channel/head/dim
// names used by every PyTorch port in the crate; renaming them to
// `batch`, `time`, `channels` etc. would diverge from the reference and
// hurt readability for anyone cross-checking against `stable-audio-tools`.
#![allow(clippy::many_single_char_names)]
// Many casts here are between small `usize` (≤ a few thousand) and `f32`
// for math constants — precision loss is not a real concern in those
// spots and the explicit `#[allow]` on each call site would be noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use candle_core::{D, DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};

/// Configuration for the Stable Audio DiT.
#[derive(Debug, Clone)]
pub struct DiTConfig {
    /// Number of transformer blocks.
    pub depth: usize,
    /// Hidden width per token.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension (must equal `embed_dim / num_heads`).
    pub head_dim: usize,
    /// MLP expansion ratio (Stable Audio's SwiGLU expansion is 4x).
    pub mlp_ratio: f64,
    /// Latent channel count (64 for both Small and 1.0).
    pub latent_channels: usize,
    /// T5 hidden dim (768 for T5-base).
    pub t5_dim: usize,
    /// Width of the Fourier feature buffer used to embed the three
    /// scalars (timestep / seconds_start / seconds_total). 256 in the
    /// reference.
    pub fourier_features: usize,
    /// Max sequence length used to pre-compute the RoPE cos/sin tables.
    /// 1032 covers the Small variant's 11s @ 86 Hz comfortably plus
    /// headroom for the prepend cond token.
    pub max_seq_len: usize,
    /// Whether to apply `qk_norm` (L2 normalize Q and K). Off for Small.
    pub qk_norm: bool,
}

impl DiTConfig {
    /// Hyperparameters for `stabilityai/stable-audio-open-small`.
    #[must_use]
    pub fn stable_audio_small() -> Self {
        Self {
            depth: 12,
            embed_dim: 768,
            num_heads: 8,
            head_dim: 96,
            mlp_ratio: 4.0,
            latent_channels: 64,
            t5_dim: 768,
            fourier_features: 256,
            max_seq_len: 1032,
            qk_norm: false,
        }
    }

    /// Hyperparameters for `stabilityai/stable-audio-open-1.0` (24 blocks,
    /// 24 heads, 1536 embed dim). Provided for completeness; the Small
    /// variant is the supported target for the initial port.
    #[must_use]
    pub fn stable_audio_open_1_0() -> Self {
        Self {
            depth: 24,
            embed_dim: 1536,
            num_heads: 24,
            head_dim: 64,
            mlp_ratio: 4.0,
            latent_channels: 64,
            t5_dim: 768,
            fourier_features: 256,
            max_seq_len: 2048,
            qk_norm: false,
        }
    }

    /// Number of rotary embedding dimensions per head. Stable Audio uses
    /// partial rotary (GPT-J style) over `head_dim / 2` components — the
    /// remainder is left un-rotated.
    #[must_use]
    pub fn rope_dim(&self) -> usize {
        self.head_dim.max(64) / 2
    }

    /// MLP hidden width (with SwiGLU, the gate projection produces
    /// `2 * inner_dim` and the activated half feeds the down projection).
    #[must_use]
    pub fn mlp_inner(&self) -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let inner = (self.embed_dim as f64 * self.mlp_ratio) as usize;
        inner
    }
}

/// Bias-less LayerNorm with a learnable per-channel `gamma` only.
///
/// The Stable Audio reference uses `F.layer_norm(x, x.shape[-1:],
/// weight=gamma, bias=None)`. `candle_nn::LayerNorm::new_no_bias` is the
/// candle equivalent.
fn layer_norm_no_bias(dim: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let gamma = vb.get(dim, "gamma")?;
    Ok(candle_nn::LayerNorm::new_no_bias(gamma, eps))
}

/// Fully-affine LayerNorm with learnable `gamma` (weight) and `beta`
/// (bias). The reference only uses bias-less LayerNorms but the helper
/// is kept for future variants.
#[allow(dead_code)]
fn layer_norm_affine(dim: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let gamma = vb.get(dim, "gamma")?;
    let beta = vb.get(dim, "beta")?;
    Ok(candle_nn::LayerNorm::new(gamma, beta, eps))
}

/// Fourier feature embedding for continuous scalars (timestep,
/// seconds_start, seconds_total).
///
/// The reference parameterizes a `(out_features / 2, in_features)` weight
/// matrix `W` and maps a scalar `x` to
/// `cat([cos(2π x Wᵀ), sin(2π x Wᵀ)])`. The input arrives as a `(B,)`
/// vector and is unsqueezed to `(B, 1)` inside `forward`.
#[derive(Debug)]
pub struct FourierFeatures {
    /// `(out_features / 2, in_features)` weight matrix.
    weight: Tensor,
    #[allow(
        dead_code,
        reason = "Surfaced via out_features() for downstream wiring; the \
                  forward pass derives the output shape from the weight \
                  tensor directly."
    )]
    out_features: usize,
}

impl FourierFeatures {
    /// Construct from a `VarBuilder`. The weight tensor lives at `weight`.
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        assert!(
            out_features.is_multiple_of(2),
            "fourier feature out_features must be even (got {out_features})"
        );
        let weight = vb.get((out_features / 2, in_features), "weight")?;
        Ok(Self {
            weight,
            out_features,
        })
    }

    /// `out_features` getter for downstream wiring code.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Public accessor reserved for downstream consumers that \
                  introspect the Fourier-feature width."
    )]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward: `(B, in_features)` → `(B, out_features)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // f = 2π * x @ Wᵀ      shape (B, out/2)
        let weight_t = self.weight.t()?.contiguous()?;
        let weight_t = weight_t.to_dtype(x.dtype())?;
        let two_pi = std::f64::consts::TAU;
        let f = x.matmul(&weight_t)?;
        let f = (f * two_pi)?;
        let cos = f.cos()?;
        let sin = f.sin()?;
        Tensor::cat(&[&cos, &sin], D::Minus1)
    }
}

/// AdaLN modulation projector: takes the global conditioning vector and
/// produces six chunks per block: `(scale_self, shift_self, gate_self,
/// scale_ff, shift_ff, gate_ff)`. The reference uses a `SiLU → Linear`
/// stack and chunks the output along the last axis after unsqueezing on
/// the sequence dimension.
#[derive(Debug)]
pub struct AdaLNModulation {
    /// Linear projection from `embed_dim` to `6 * embed_dim` (no bias in
    /// the reference). Note the SiLU is applied to the *input*, not the
    /// output, of this linear.
    proj: Linear,
    embed_dim: usize,
}

impl AdaLNModulation {
    /// Construct from a `VarBuilder`. The Linear weight lives at `weight`
    /// (the reference's `self.to_scale_shift_gate[1].weight`).
    pub fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(embed_dim, embed_dim * 6, vb)?;
        Ok(Self { proj, embed_dim })
    }

    /// Apply `SiLU` to the global conditioning, project to 6 chunks, and
    /// split. Returns `(scale_self, shift_self, gate_self, scale_ff,
    /// shift_ff, gate_ff)`, each shaped `(B, 1, embed_dim)` so they
    /// broadcast over the audio sequence.
    pub fn forward(
        &self,
        global_cond: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // global_cond is (B, embed_dim).
        let activated = candle_nn::ops::silu(global_cond)?;
        let projected = self.proj.forward(&activated)?;
        // Insert the sequence axis so we can broadcast over T_audio later.
        let projected = projected.unsqueeze(1)?; // (B, 1, 6*embed_dim)
        // Split along the last axis into 6 equal-size chunks.
        let mut chunks = Vec::with_capacity(6);
        for i in 0..6 {
            let start = i * self.embed_dim;
            let chunk = projected.narrow(D::Minus1, start, self.embed_dim)?;
            chunks.push(chunk);
        }
        // Reverse-pop is awkward; use indexed swap_remove from the end.
        let gate_ff = chunks.remove(5);
        let shift_ff = chunks.remove(4);
        let scale_ff = chunks.remove(3);
        let gate_self = chunks.remove(2);
        let shift_self = chunks.remove(1);
        let scale_self = chunks.remove(0);
        Ok((
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff,
        ))
    }
}

/// AdaLN modulation: `(1 + scale) * x + shift`, broadcasting `scale` and
/// `shift` from `(B, 1, D)` over the sequence axis of `x = (B, T, D)`.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(scale)?;
    let scaled = x.broadcast_mul(&(scale + one)?)?;
    scaled.broadcast_add(shift)
}

/// Apply RoPE (partial, rotate-half / GPT-J variant) to a `(B, H, T, D)`
/// tensor of attention queries or keys.
///
/// `freqs_cos` and `freqs_sin` are pre-computed tables of shape
/// `(max_seq_len, rope_dim / 2)`. We slice the leading `T` rows for the
/// current sequence and broadcast over `(B, H)`.
///
/// Only the first `rope_dim` components of each head are rotated; the
/// remainder is concatenated back unchanged. This matches the reference's
/// `t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]` split.
pub fn apply_rope(qk: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq_len, head_dim) = qk.dims4()?;
    let rope_dim = freqs_cos.dim(D::Minus1)? * 2;
    if rope_dim > head_dim {
        candle_core::bail!("rope_dim {} cannot exceed head_dim {}", rope_dim, head_dim);
    }
    if rope_dim == head_dim {
        // Full rotary — slice freq tables to seq_len and apply.
        let cos = freqs_cos.narrow(0, 0, seq_len)?.contiguous()?;
        let sin = freqs_sin.narrow(0, 0, seq_len)?.contiguous()?;
        return candle_nn::rotary_emb::rope(&qk.contiguous()?, &cos, &sin);
    }
    // Partial rotary: split, rotate the leading slice, concat back.
    let rotated = qk.narrow(D::Minus1, 0, rope_dim)?.contiguous()?;
    let unrotated = qk.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?;
    let cos = freqs_cos.narrow(0, 0, seq_len)?.contiguous()?;
    let sin = freqs_sin.narrow(0, 0, seq_len)?.contiguous()?;
    let rotated = candle_nn::rotary_emb::rope(&rotated, &cos, &sin)?;
    Tensor::cat(&[&rotated, &unrotated.contiguous()?], D::Minus1)
}

/// Pre-compute `(cos, sin)` RoPE tables of shape
/// `(max_seq_len, rope_dim / 2)`. Matches the reference's
/// `inv_freq = base ** -(arange(rope_dim/2) / (rope_dim/2))` followed by
/// `freqs = einsum('i,j->ij', positions, inv_freq)` — i.e. the candle
/// `rope` op's expected layout for `dim_heads/2`-sized tables.
pub fn precompute_rope_freqs(
    max_seq_len: usize,
    rope_dim: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    assert!(
        rope_dim.is_multiple_of(2),
        "rope_dim must be even (got {rope_dim})"
    );
    let half = rope_dim / 2;
    let base = 10_000f32;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let exp = i as f32 / half as f32;
            base.powf(-exp)
        })
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, half, device)?;
    let positions: Vec<f32> = (0..max_seq_len)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f32;
            v
        })
        .collect();
    let positions = Tensor::from_vec(positions, max_seq_len, device)?;
    // (T, half)
    let freqs = positions
        .unsqueeze(1)?
        .broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok((cos, sin))
}

/// Multi-head attention block (self or cross). Bias-less Q/K/V/output
/// projections. Self-attention uses a fused QKV linear and accepts RoPE
/// inputs; cross-attention uses a split `q` plus fused `kv` linear and
/// does *not* apply RoPE.
#[derive(Debug)]
pub struct Attention {
    /// Self-attn QKV fused projection (`embed_dim → 3 * embed_dim`).
    /// `None` for cross-attention.
    qkv: Option<Linear>,
    /// Cross-attn Q projection (`embed_dim → embed_dim`). `None` for
    /// self-attention.
    q_proj: Option<Linear>,
    /// Cross-attn KV fused projection (`t5_dim → 2 * t5_dim`). `None`
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

    /// Construct a cross-attention block. `kv_dim` is the T5 hidden dim.
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
    /// - Self-attn: `kv_input` should be `None`. RoPE is applied to Q/K.
    /// - Cross-attn: `kv_input` is the T5 context `(B, T_ctx, t5_dim)`.
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
            // RoPE is cast to f32 inside in the reference; honour
            // that for numerical stability.
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

/// SwiGLU feed-forward block: `Linear(d, 2*inner) -> chunk -> x * silu(g)
/// -> Linear(inner, d)`. The reference's GLU module fuses the chunk +
/// activation; we replicate the parameter layout exactly so the
/// safetensors keys line up.
#[derive(Debug)]
pub struct FeedForward {
    /// `embed_dim → 2 * inner_dim`. Stored as `ff.0.proj` in the reference.
    glu_proj: Linear,
    /// `inner_dim → embed_dim`. Stored as `ff.2` in the reference (the
    /// `nn.Sequential` index for the final linear).
    out_proj: Linear,
}

impl FeedForward {
    pub fn new(embed_dim: usize, inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        let glu_proj = linear_no_bias(embed_dim, inner_dim * 2, vb.pp("ff.0.proj"))?;
        let out_proj = linear_no_bias(inner_dim, embed_dim, vb.pp("ff.2"))?;
        Ok(Self { glu_proj, out_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.glu_proj.forward(x)?;
        let half = projected.dim(D::Minus1)? / 2;
        let value = projected.narrow(D::Minus1, 0, half)?;
        let gate = projected.narrow(D::Minus1, half, half)?;
        let activated = (value * candle_nn::ops::silu(&gate)?)?;
        self.out_proj.forward(&activated)
    }
}

/// One DiT block: AdaLN-modulated self-attn (with RoPE), then plain
/// cross-attn against T5, then AdaLN-modulated SwiGLU FFN. All three
/// sub-blocks have additive residual paths. The two AdaLN paths use
/// `sigmoid(1 - gate)` to blend the residual back in — this matches
/// `stable-audio-tools` exactly.
#[derive(Debug)]
pub struct DiTBlock {
    pre_norm: candle_nn::LayerNorm,
    self_attn: Attention,
    cross_attend_norm: candle_nn::LayerNorm,
    cross_attn: Attention,
    ff_norm: candle_nn::LayerNorm,
    feed_forward: FeedForward,
    adaln: AdaLNModulation,
}

impl DiTBlock {
    /// Construct a block from a `VarBuilder` rooted at `layers.<idx>`.
    pub fn new(config: &DiTConfig, vb: VarBuilder) -> Result<Self> {
        let eps = 1e-5;
        let pre_norm = layer_norm_no_bias(config.embed_dim, eps, vb.pp("pre_norm"))?;
        let self_attn = Attention::new_self(
            config.embed_dim,
            config.num_heads,
            config.head_dim,
            config.qk_norm,
            vb.pp("self_attn"),
        )?;
        let cross_attend_norm =
            layer_norm_no_bias(config.embed_dim, eps, vb.pp("cross_attend_norm"))?;
        let cross_attn = Attention::new_cross(
            config.embed_dim,
            config.t5_dim,
            config.num_heads,
            config.head_dim,
            config.qk_norm,
            vb.pp("cross_attn"),
        )?;
        let ff_norm = layer_norm_no_bias(config.embed_dim, eps, vb.pp("ff_norm"))?;
        let feed_forward = FeedForward::new(config.embed_dim, config.mlp_inner(), vb.pp("ff"))?;
        let adaln = AdaLNModulation::new(config.embed_dim, vb.pp("to_scale_shift_gate.1"))?;
        Ok(Self {
            pre_norm,
            self_attn,
            cross_attend_norm,
            cross_attn,
            ff_norm,
            feed_forward,
            adaln,
        })
    }

    /// Forward pass through one block.
    ///
    /// `x` is the audio token sequence `(B, T_audio, embed_dim)`.
    /// `cond` is the T5 cross-attention context `(B, T_text, t5_dim)`.
    /// `global_cond` is the combined `(timestep + seconds)` embedding
    /// `(B, embed_dim)` used to drive AdaLN.
    pub fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
        global_cond: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let (scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff) =
            self.adaln.forward(global_cond)?;

        // Self-attention path with AdaLN modulation. Mirrors the
        // reference exactly:
        //     residual = x
        //     x = pre_norm(x)
        //     x = x * (1 + scale_self) + shift_self
        //     x = self_attn(x, rope=rope)
        //     x = x * sigmoid(1 - gate_self)
        //     x = x + residual
        let residual = x.clone();
        let h = self.pre_norm.forward(x)?;
        let h = modulate(&h, &shift_self, &scale_self)?;
        let h = self
            .self_attn
            .forward(&h, None, Some((rope_cos, rope_sin)))?;
        let one = Tensor::ones_like(&gate_self)?;
        let blend_self = candle_nn::ops::sigmoid(&(one - &gate_self)?)?;
        let h = h.broadcast_mul(&blend_self)?;
        let x = (h + residual)?;

        // Cross-attention path: plain residual, no AdaLN gating in the
        // reference. `cross_attend_norm` is applied to the *query*
        // stream, then we attend over the T5 context (which is already
        // pre-projected to `t5_dim`).
        let norm_x = self.cross_attend_norm.forward(&x)?;
        let cross = self.cross_attn.forward(&norm_x, Some(cond), None)?;
        let x = (x + cross)?;

        // Feed-forward path with AdaLN modulation, same recipe.
        let residual = x.clone();
        let h = self.ff_norm.forward(&x)?;
        let h = modulate(&h, &shift_ff, &scale_ff)?;
        let h = self.feed_forward.forward(&h)?;
        let blend_ff = candle_nn::ops::sigmoid(&(Tensor::ones_like(&gate_ff)? - &gate_ff)?)?;
        let h = h.broadcast_mul(&blend_ff)?;
        h + residual
    }
}

/// Full Stable Audio DiT.
///
/// Input/output shape: `(B, latent_channels, T_audio)` (channels-first).
/// Internally the audio is transposed to `(B, T_audio, embed_dim)` for
/// the transformer stack and transposed back at the end.
#[derive(Debug)]
pub struct DiT {
    /// 1×1 conv-equivalent linear: `latent_channels → latent_channels`,
    /// with an additive residual (`preprocess_conv(x) + x` in the
    /// reference). We model the conv as a `Linear` since the kernel is 1.
    preprocess_conv: Linear,
    /// `latent_channels → embed_dim` token projection.
    project_in: Linear,
    /// `embed_dim → latent_channels` token un-projection.
    project_out: Linear,
    /// Post-process additive residual conv (`postprocess_conv(out) + out`).
    postprocess_conv: Linear,
    /// 12 transformer blocks for Small (24 for 1.0).
    blocks: Vec<DiTBlock>,
    /// Fourier features for the three input scalars.
    fourier_features: FourierFeatures,
    /// Timestep embedding head (Linear → SiLU → Linear) producing
    /// `(B, embed_dim)`.
    timestep_proj1: Linear,
    timestep_proj2: Linear,
    /// Cross-attention conditioning projection (Linear → SiLU → Linear).
    /// The reference exposes this as `to_cond_embed` and applies it to
    /// the T5 tokens before they reach the cross-attention blocks.
    cond_proj1: Linear,
    cond_proj2: Linear,
    /// RoPE tables sized to `config.max_seq_len`.
    rope_cos: Tensor,
    rope_sin: Tensor,
    config: DiTConfig,
}

impl DiT {
    /// Construct from a `VarBuilder` rooted at the DiT submodule's
    /// safetensors prefix.
    pub fn new(vb: VarBuilder, config: DiTConfig) -> Result<Self> {
        let device = vb.device().clone();

        // The reference uses 1×1 Conv1d; with kernel=1 those are
        // mathematically equivalent to a Linear, so we model them as
        // such for simplicity. The safetensors loader will need to
        // squeeze the conv weight from `(C_out, C_in, 1)` to
        // `(C_out, C_in)` to land here — that's handled in the (yet to
        // be written) `weights.rs` mapping layer.
        let preprocess_conv = linear_no_bias(
            config.latent_channels,
            config.latent_channels,
            vb.pp("preprocess_conv"),
        )?;
        let postprocess_conv = linear_no_bias(
            config.latent_channels,
            config.latent_channels,
            vb.pp("postprocess_conv"),
        )?;

        let project_in = linear_no_bias(
            config.latent_channels,
            config.embed_dim,
            vb.pp("transformer.project_in"),
        )?;
        let project_out = linear_no_bias(
            config.embed_dim,
            config.latent_channels,
            vb.pp("transformer.project_out"),
        )?;

        // 12 transformer blocks. The reference's prefix is
        // `transformer.layers.<i>` — keep that mapping exact so weights
        // load cleanly.
        let blocks_vb = vb.pp("transformer.layers");
        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            blocks.push(DiTBlock::new(&config, blocks_vb.pp(i))?);
        }

        let fourier_features =
            FourierFeatures::new(1, config.fourier_features, vb.pp("timestep_features"))?;

        // `to_timestep_embed` in the reference is a 2-layer MLP with
        // bias=True, SiLU in between. We don't fold the SiLU into a
        // module since candle doesn't have an `nn.Sequential` parallel;
        // it's applied inline in `forward`.
        let timestep_proj1 = linear(
            config.fourier_features,
            config.embed_dim,
            vb.pp("to_timestep_embed.0"),
        )?;
        let timestep_proj2 = linear(
            config.embed_dim,
            config.embed_dim,
            vb.pp("to_timestep_embed.2"),
        )?;

        // `to_cond_embed` is bias-less in the reference; same width in,
        // same width out (cond_embed_dim == cond_token_dim == 768 when
        // `project_cond_tokens=False`, which is the Small variant's
        // setting).
        let cond_proj1 = linear_no_bias(config.t5_dim, config.t5_dim, vb.pp("to_cond_embed.0"))?;
        let cond_proj2 = linear_no_bias(config.t5_dim, config.t5_dim, vb.pp("to_cond_embed.2"))?;

        let (rope_cos, rope_sin) =
            precompute_rope_freqs(config.max_seq_len, config.rope_dim(), &device)?;

        Ok(Self {
            preprocess_conv,
            project_in,
            project_out,
            postprocess_conv,
            blocks,
            fourier_features,
            timestep_proj1,
            timestep_proj2,
            cond_proj1,
            cond_proj2,
            rope_cos,
            rope_sin,
            config,
        })
    }

    /// Access the active config.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Public accessor surfaced for downstream wiring + \
                  parity-test harnesses that need to inspect the active \
                  hyperparameter pack."
    )]
    pub fn config(&self) -> &DiTConfig {
        &self.config
    }

    /// Build the global conditioning vector. The reference embeds each
    /// of `timestep`, `seconds_start`, and `seconds_total` through the
    /// *same* `FourierFeatures` block (different inputs, shared weight),
    /// projects through the timestep MLP, and sums the three resulting
    /// embed-dim vectors into a single `(B, embed_dim)` global cond.
    ///
    /// `seconds_*` are passed as fractions of the model's `max_seq_len`
    /// window in the upstream pipeline; here we just take whatever the
    /// caller hands us.
    fn global_conditioning(
        &self,
        timestep: &Tensor,
        seconds_start: &Tensor,
        seconds_total: &Tensor,
    ) -> Result<Tensor> {
        // Each scalar arrives as `(B,)`; unsqueeze to `(B, 1)` for the
        // Fourier projection.
        let t = self.embed_scalar(timestep)?;
        let s_start = self.embed_scalar(seconds_start)?;
        let s_total = self.embed_scalar(seconds_total)?;
        (&t + &s_start)?.add(&s_total)
    }

    /// Single-scalar Fourier+MLP embed: `(B,)` → `(B, embed_dim)`.
    fn embed_scalar(&self, scalar: &Tensor) -> Result<Tensor> {
        let scalar = scalar.unsqueeze(D::Minus1)?; // (B, 1)
        let scalar = scalar.to_dtype(DType::F32)?;
        let f = self.fourier_features.forward(&scalar)?;
        let h = self.timestep_proj1.forward(&f)?;
        let h = candle_nn::ops::silu(&h)?;
        self.timestep_proj2.forward(&h)
    }

    /// Cross-attention conditioning preprocessor. Mirrors the
    /// reference's `to_cond_embed` 2-layer MLP (Linear → SiLU → Linear),
    /// applied to the T5 token stream before it reaches the
    /// cross-attention blocks.
    fn project_cond(&self, t5_tokens: &Tensor) -> Result<Tensor> {
        let h = self.cond_proj1.forward(t5_tokens)?;
        let h = candle_nn::ops::silu(&h)?;
        self.cond_proj2.forward(&h)
    }

    /// Full forward pass.
    ///
    /// - `latent`: `(B, latent_channels, T_audio)` diffusion latent at
    ///   the current sampler step.
    /// - `t5_tokens`: `(B, T_text, t5_dim)` text encoder output.
    /// - `timestep`: `(B,)` continuous timestep in `[0, 1]`.
    /// - `seconds_start`: `(B,)` start-of-audio second (typically 0).
    /// - `seconds_total`: `(B,)` total audio length in seconds.
    ///
    /// Returns the predicted velocity, shape `(B, latent_channels,
    /// T_audio)` (channels-first, same as the input).
    pub fn forward(
        &self,
        latent: &Tensor,
        t5_tokens: &Tensor,
        timestep: &Tensor,
        seconds_start: &Tensor,
        seconds_total: &Tensor,
    ) -> Result<Tensor> {
        let (b, c, t_audio) = latent.dims3()?;
        debug_assert_eq!(c, self.config.latent_channels);

        // Pre-process conv: `(B, C, T)` → flatten to per-token Linear,
        // residual add.
        let pre_in = latent.transpose(1, 2)?.contiguous()?; // (B, T, C)
        let pre_out = self.preprocess_conv.forward(&pre_in)?;
        let x = (pre_in + pre_out)?;

        // Project to embed_dim and pass through the transformer stack.
        let x = self.project_in.forward(&x)?; // (B, T, embed_dim)
        let cond = self.project_cond(t5_tokens)?;
        let global_cond = self.global_conditioning(timestep, seconds_start, seconds_total)?;

        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h, &cond, &global_cond, &self.rope_cos, &self.rope_sin)?;
        }

        // Project back to `(B, T, latent_channels)`, transpose, post
        // residual conv.
        let h = self.project_out.forward(&h)?;
        let post_out = self.postprocess_conv.forward(&h)?;
        let h = (h + post_out)?;
        let h = h.transpose(1, 2)?.contiguous()?; // (B, C, T)
        debug_assert_eq!(h.dim(0)?, b);
        debug_assert_eq!(h.dim(1)?, self.config.latent_channels);
        debug_assert_eq!(h.dim(2)?, t_audio);
        Ok(h)
    }

    /// Convenience: convert a velocity prediction to (`x0`, `eps`) at the
    /// given `(alpha, sigma)` schedule point. The Stable Audio few-step
    /// distilled sampler consumes velocity directly, but the helper is
    /// useful for sanity-checking against PyTorch dumps that report `eps`
    /// or `x0` from the same forward pass.
    #[allow(
        dead_code,
        reason = "Diagnostic helper exercised by parity tests + reserved \
                  for the next sampler wave that consumes eps directly."
    )]
    pub fn velocity_to_x0_eps(
        x_t: &Tensor,
        velocity: &Tensor,
        alpha: f64,
        sigma: f64,
    ) -> Result<(Tensor, Tensor)> {
        let x0 = ((x_t * alpha)? - (velocity * sigma)?)?;
        let eps = ((x_t * sigma)? + (velocity * alpha)?)?;
        Ok((x0, eps))
    }

    /// Helper for callers that want to inspect a specific block's
    /// activations (for parity tests). Returns the activations *after*
    /// `block_idx`. Out-of-range indices return an error.
    #[allow(
        dead_code,
        reason = "Reserved for block-level parity tests against a Python \
                  dump harness (introduced in a later wave); not yet \
                  exercised by the in-tree test suite."
    )]
    pub fn forward_through(
        &self,
        latent: &Tensor,
        t5_tokens: &Tensor,
        timestep: &Tensor,
        seconds_start: &Tensor,
        seconds_total: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        if block_idx >= self.blocks.len() {
            candle_core::bail!(
                "block_idx {} out of range (depth={})",
                block_idx,
                self.blocks.len()
            );
        }
        let pre_in = latent.transpose(1, 2)?.contiguous()?;
        let pre_out = self.preprocess_conv.forward(&pre_in)?;
        let x = (pre_in + pre_out)?;
        let x = self.project_in.forward(&x)?;
        let cond = self.project_cond(t5_tokens)?;
        let global_cond = self.global_conditioning(timestep, seconds_start, seconds_total)?;

        let mut h = x;
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &cond, &global_cond, &self.rope_cos, &self.rope_sin)?;
            if i == block_idx {
                return Ok(h);
            }
        }
        // Unreachable given the bounds check above, but the borrow
        // checker doesn't know that.
        Ok(h)
    }

    /// Read-only view of the first-token activation of the
    /// pre-transformer embedding, useful for smoke tests that need to
    /// confirm `project_in` was wired correctly without paying for a
    /// full forward.
    #[allow(
        dead_code,
        reason = "Smoke-test helper used by the in-tree parity tests; the \
                  release build only calls forward(), so the lib build \
                  reports this as dead."
    )]
    pub fn token_embed_zero(&self, latent: &Tensor) -> Result<Tensor> {
        let pre_in = latent.transpose(1, 2)?.contiguous()?;
        let pre_out = self.preprocess_conv.forward(&pre_in)?;
        let x = (pre_in + pre_out)?;
        let x = self.project_in.forward(&x)?;
        x.i((.., 0, ..))?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    /// Build a `VarBuilder` backed by zeros for shape-only tests.
    fn zeros_vb() -> (VarBuilder<'static>, Device) {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        (vb, device)
    }

    #[test]
    fn config_small_invariants() {
        let c = DiTConfig::stable_audio_small();
        assert_eq!(c.depth, 12);
        assert_eq!(c.embed_dim, 768);
        assert_eq!(c.num_heads, 8);
        assert_eq!(c.head_dim, 96);
        assert_eq!(c.embed_dim, c.num_heads * c.head_dim);
        // SwiGLU expansion 4x → inner is 3072.
        assert_eq!(c.mlp_inner(), 3072);
        // Partial rotary: max(96, 64) / 2 = 48.
        assert_eq!(c.rope_dim(), 48);
    }

    #[test]
    fn config_open_1_0_invariants() {
        let c = DiTConfig::stable_audio_open_1_0();
        assert_eq!(c.depth, 24);
        assert_eq!(c.embed_dim, c.num_heads * c.head_dim);
    }

    #[test]
    fn modulate_broadcasts_over_sequence() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 4, 8), DType::F32, &device)?;
        let shift = Tensor::ones((2, 1, 8), DType::F32, &device)?;
        let scale = Tensor::zeros((2, 1, 8), DType::F32, &device)?;
        let out = modulate(&x, &shift, &scale)?;
        // (1 + 0) * 0 + 1 = 1 everywhere.
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(flat.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn precompute_rope_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (cos, sin) = precompute_rope_freqs(1032, 48, &device)?;
        assert_eq!(cos.dims(), &[1032, 24]);
        assert_eq!(sin.dims(), &[1032, 24]);
        Ok(())
    }

    #[test]
    fn fourier_features_doubles_dim() -> Result<()> {
        let (vb, device) = zeros_vb();
        let ff = FourierFeatures::new(1, 256, vb.pp("ff"))?;
        let x = Tensor::zeros((3, 1), DType::F32, &device)?;
        let out = ff.forward(&x)?;
        assert_eq!(out.dims(), &[3, 256]);
        Ok(())
    }

    #[test]
    fn apply_rope_preserves_unrotated_tail() -> Result<()> {
        let device = Device::Cpu;
        // (B=1, H=2, T=4, D=8). Rope dim 4 → first 4 rotated, last 4 left alone.
        let q = Tensor::ones((1, 2, 4, 8), DType::F32, &device)?;
        let (cos, sin) = precompute_rope_freqs(4, 4, &device)?;
        let out = apply_rope(&q, &cos, &sin)?;
        assert_eq!(out.dims(), q.dims());
        // The trailing slice (indices 4..8) must equal the input there.
        let tail_in = q.narrow(D::Minus1, 4, 4)?.flatten_all()?.to_vec1::<f32>()?;
        let tail_out = out
            .narrow(D::Minus1, 4, 4)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(tail_in, tail_out);
        Ok(())
    }

    #[test]
    fn dit_constructs_under_zeros() -> Result<()> {
        let (vb, _device) = zeros_vb();
        let config = DiTConfig::stable_audio_small();
        let dit = DiT::new(vb, config)?;
        assert_eq!(dit.blocks.len(), 12);
        assert_eq!(dit.config.embed_dim, 768);
        Ok(())
    }

    #[test]
    fn dit_forward_shape_preserved() -> Result<()> {
        let (vb, device) = zeros_vb();
        let config = DiTConfig::stable_audio_small();
        let dit = DiT::new(vb, config)?;
        let latent = Tensor::zeros((1, 64, 860), DType::F32, &device)?;
        let t5 = Tensor::zeros((1, 64, 768), DType::F32, &device)?;
        let timestep = Tensor::zeros(1, DType::F32, &device)?;
        let s_start = Tensor::zeros(1, DType::F32, &device)?;
        let s_total = Tensor::zeros(1, DType::F32, &device)?;
        let out = dit.forward(&latent, &t5, &timestep, &s_start, &s_total)?;
        assert_eq!(out.dims(), &[1, 64, 860]);
        Ok(())
    }

    #[test]
    fn dit_forward_through_matches_block_count() -> Result<()> {
        let (vb, device) = zeros_vb();
        let config = DiTConfig::stable_audio_small();
        let dit = DiT::new(vb, config)?;
        let latent = Tensor::zeros((1, 64, 128), DType::F32, &device)?;
        let t5 = Tensor::zeros((1, 8, 768), DType::F32, &device)?;
        let timestep = Tensor::zeros(1, DType::F32, &device)?;
        let s_start = Tensor::zeros(1, DType::F32, &device)?;
        let s_total = Tensor::zeros(1, DType::F32, &device)?;
        let h = dit.forward_through(&latent, &t5, &timestep, &s_start, &s_total, 0)?;
        assert_eq!(h.dims(), &[1, 128, 768]);
        let h = dit.forward_through(&latent, &t5, &timestep, &s_start, &s_total, 11)?;
        assert_eq!(h.dims(), &[1, 128, 768]);
        assert!(
            dit.forward_through(&latent, &t5, &timestep, &s_start, &s_total, 12)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn velocity_to_x0_eps_consistent() -> Result<()> {
        let device = Device::Cpu;
        let x_t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let v = Tensor::from_vec(vec![0.5_f32, 0.5, 0.5, 0.5], (2, 2), &device)?;
        let (x0, eps) = DiT::velocity_to_x0_eps(&x_t, &v, 0.8, 0.6)?;
        // Reconstruction: alpha*x0 + sigma*eps should equal x_t (since
        // alpha^2 + sigma^2 = 1).
        let recon = ((&x0 * 0.8)? + (&eps * 0.6)?)?;
        let diff = ((recon - &x_t)?.abs()?.flatten_all()?.to_vec1::<f32>()?)
            .into_iter()
            .fold(0.0_f32, f32::max);
        assert!(diff < 1e-5, "reconstruction error too high: {diff}");
        Ok(())
    }

    #[test]
    fn token_embed_zero_extracts_first_token() -> Result<()> {
        let (vb, device) = zeros_vb();
        let config = DiTConfig::stable_audio_small();
        let dit = DiT::new(vb, config)?;
        let latent = Tensor::zeros((2, 64, 32), DType::F32, &device)?;
        let out = dit.token_embed_zero(&latent)?;
        assert_eq!(out.dims(), &[2, 768]);
        Ok(())
    }

    // Numerical-parity tests against the Wave 3.1 Python dump harness.
    // These are `#[ignore]` so a normal `cargo test` run doesn't fail on
    // a fresh checkout where the .npz dumps don't exist. The dedicated
    // `tests/stable_audio_block_compare.rs` integration test (added in
    // Wave 3.5) will exercise these with real safetensors weights.
    #[test]
    #[ignore = "requires Wave 3.1 python dumps at ~/.cache/blazen-stableaudio-research/dumps/"]
    fn dit_block_parity_against_python_dump() {
        let dump_root = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".cache/blazen-stableaudio-research/dumps");
        assert!(
            dump_root.exists(),
            "python dump dir missing: {}",
            dump_root.display()
        );
        // Concrete parity comparison lives in
        // crates/blazen-audio-music/tests/stable_audio_block_compare.rs
        // — this test just asserts the dump directory is present so
        // CI's audit job can flag a regenerate-needed state.
    }
}
