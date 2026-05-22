//! Temporal multi-head self-attention block.
//!
//! Modern video diffusion architectures (CogVideoX, HunyuanVideo,
//! Mochi, …) bolt a "temporal" attention pass on top of their spatial
//! attention pass: the spatial pass treats each frame independently
//! (folding `T` into the batch axis), and the temporal pass mixes
//! information *across* frames at every spatial location (folding
//! `H*W` into the batch axis).
//!
//! This module provides the latter — a bias-less, RoPE-free,
//! mask-free multi-head self-attention block over the `frames` axis
//! that is intentionally agnostic about the surrounding block recipe
//! (whether AdaLN is applied, whether a cross-attention pass is
//! interleaved, etc.). Consuming backends assemble this into the
//! specific recipe their model uses.
//!
//! # Layout convention
//!
//! The forward pass accepts an already-flattened input tensor of
//! shape `(batch, frames, channels)` — callers are expected to fuse
//! their `(height, width)` axes into the batch axis (i.e.
//! `B*H*W, T, C`) before calling [`TemporalAttention::forward`]. The
//! output has the same shape as the input.

#![allow(clippy::similar_names)]
#![allow(clippy::module_name_repetitions)]
// `VarBuilder` is the candle convention for module construction; the
// move cost is one cheap Arc clone.
#![allow(clippy::needless_pass_by_value)]
// `b`, `t`, `c`, `h`, `d` are the canonical batch / time / channel /
// head / dim names used by every PyTorch port; renaming them would
// diverge from references and hurt cross-checking.
#![allow(clippy::many_single_char_names)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use candle_core::{D, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use thiserror::Error;

/// Errors emitted by [`TemporalAttention`].
#[derive(Debug, Error)]
pub enum TemporalAttentionError {
    /// The input tensor did not satisfy the documented rank or
    /// channel-count constraints. The string carries a
    /// human-readable explanation suitable for surfacing in the
    /// consuming backend's error chain.
    #[error("invalid temporal attention input shape: {0}")]
    InvalidShape(String),

    /// A candle tensor / kernel operation failed. Most commonly this
    /// is a shape mismatch inside a `matmul` or `reshape` — wrap it
    /// up and let the caller deal with it.
    #[error("candle tensor op failed: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Temporal multi-head self-attention block.
///
/// Applies multi-head attention along the `frames` axis while
/// keeping the `(height, width)` dimensions fused into the batch
/// axis. Q / K / V and output projections are bias-less linear
/// layers. The attention pattern is unmasked and includes no
/// position encoding — RoPE / ALiBi / etc. is the consuming
/// backend's responsibility for v1.
///
/// # Shape
///
/// - Input: `(batch, frames, embed_dim)`.
/// - Output: `(batch, frames, embed_dim)` (same shape).
///
/// `embed_dim` must equal `num_heads * head_dim`.
#[derive(Debug)]
pub struct TemporalAttention {
    /// Fused QKV projection (`embed_dim → 3 * embed_dim`).
    qkv: Linear,
    /// Output projection (`embed_dim → embed_dim`).
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    /// `1.0 / sqrt(head_dim)`, pre-computed.
    scale: f64,
}

impl TemporalAttention {
    /// Construct a temporal attention block.
    ///
    /// `embed_dim` must equal `num_heads * head_dim`. The fused QKV
    /// weight lives at `vb / to_qkv` and the output projection at
    /// `vb / to_out` — matching the naming convention used by the
    /// audio DiT primitives in `blazen-audio-core`.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self, TemporalAttentionError> {
        if embed_dim != num_heads * head_dim {
            return Err(TemporalAttentionError::InvalidShape(format!(
                "embed_dim ({embed_dim}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
            )));
        }
        let qkv = linear_no_bias(embed_dim, embed_dim * 3, vb.pp("to_qkv"))?;
        let out_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("to_out"))?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            qkv,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Reshape `(B, T, H*D)` → `(B, H, T, D)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, t, _) = x.dims3()?;
        x.reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Reshape `(B, H, T, D)` → `(B, T, H*D)`.
    fn merge_heads(x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, h, t, d) = x.dims4()?;
        x.transpose(1, 2)?.contiguous()?.reshape((b, t, h * d))
    }

    /// Forward pass over the time axis.
    ///
    /// Accepts `(batch, frames, embed_dim)` and returns a tensor of
    /// the same shape. The attention is computed over the `frames`
    /// axis using standard scaled dot-product self-attention with
    /// softmax along the key axis.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TemporalAttentionError> {
        let (b, t, c) = match x.dims3() {
            Ok(dims) => dims,
            Err(e) => {
                return Err(TemporalAttentionError::InvalidShape(format!(
                    "expected rank-3 input (batch, frames, channels), got: {e}"
                )));
            }
        };
        if c != self.num_heads * self.head_dim {
            return Err(TemporalAttentionError::InvalidShape(format!(
                "channel dim ({c}) does not match num_heads*head_dim ({})",
                self.num_heads * self.head_dim
            )));
        }

        let qkv = self.qkv.forward(x)?;
        let d = qkv.dim(D::Minus1)? / 3;
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, d)?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?;

        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        // (B, H, T_q, D) @ (B, H, D, T_k) → (B, H, T_q, T_k).
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_scores = q.matmul(&k_t)?;
        let attn_scores = (attn_scores * self.scale)?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let out = attn_probs.matmul(&v.contiguous()?)?;

        let out = Self::merge_heads(&out)?;
        debug_assert_eq!(out.dim(0)?, b);
        debug_assert_eq!(out.dim(1)?, t);
        Ok(self.out_proj.forward(&out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn forward_shape_preserved() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let embed_dim = 64usize;
        let num_heads = 4usize;
        let head_dim = 16usize;
        let attn = TemporalAttention::new(embed_dim, num_heads, head_dim, vb)
            .expect("construct temporal attention");

        // (B=2, T=8, D=64) — two short clips, eight frames each.
        let x = Tensor::randn(0f32, 1f32, (2usize, 8usize, 64usize), &device)
            .expect("randn input tensor");
        let y = attn.forward(&x).expect("forward pass");
        assert_eq!(y.dims(), &[2, 8, 64]);
    }
}
