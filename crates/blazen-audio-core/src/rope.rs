//! Rotary Position Embedding (RoPE) primitives + Fourier-feature
//! embedding for continuous scalars.
//!
//! Both the partial-rotary [`apply_rope`] (GPT-J style — only the
//! leading `rope_dim` components of each head are rotated) and the
//! `(cos, sin)` table precomputation are exposed so DiT backends with
//! different head-dim / rope-dim ratios can share the same plumbing.
//!
//! The [`FourierFeatures`] block is the standard
//! `cat([cos(2π x Wᵀ), sin(2π x Wᵀ)])` embedding used by diffusion
//! models to lift continuous scalars (timestep, seconds, etc.) into a
//! high-dimensional space before passing them through an MLP.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::needless_pass_by_value)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use candle_core::{D, Result, Tensor};
use candle_nn::VarBuilder;

/// Apply RoPE (partial, rotate-half / GPT-J variant) to a
/// `(B, H, T, D)` tensor of attention queries or keys.
///
/// `freqs_cos` and `freqs_sin` are pre-computed tables of shape
/// `(max_seq_len, rope_dim / 2)`. We slice the leading `T` rows for
/// the current sequence and broadcast over `(B, H)`.
///
/// Only the first `rope_dim` components of each head are rotated; the
/// remainder is concatenated back unchanged. This matches the
/// reference's `t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]`
/// split.
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
/// `(max_seq_len, rope_dim / 2)`.
///
/// Matches the canonical
/// `inv_freq = base ** -(arange(rope_dim/2) / (rope_dim/2))` recipe
/// followed by `freqs = einsum('i,j->ij', positions, inv_freq)` —
/// i.e. the candle `rope` op's expected layout for `rope_dim/2`-sized
/// tables.
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

/// Fourier feature embedding for continuous scalars (timestep,
/// seconds_start, seconds_total, etc.).
///
/// The reference parameterizes a `(out_features / 2, in_features)`
/// weight matrix `W` and maps a scalar `x` to
/// `cat([cos(2π x Wᵀ), sin(2π x Wᵀ)])`. The input arrives as a
/// `(B, in_features)` tensor (typically `in_features = 1`).
#[derive(Debug)]
pub struct FourierFeatures {
    /// `(out_features / 2, in_features)` weight matrix.
    weight: Tensor,
    out_features: usize,
}

impl FourierFeatures {
    /// Construct from a `VarBuilder`. The weight tensor lives at
    /// `weight`.
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
