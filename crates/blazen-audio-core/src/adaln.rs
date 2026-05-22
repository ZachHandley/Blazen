//! AdaLN-Zero modulation primitives.
//!
//! Adaptive LayerNorm (with "Zero" referring to the original
//! initialization scheme) is the standard conditioning mechanism for
//! Diffusion Transformers: a global conditioning vector is projected
//! into per-block `(scale, shift, gate)` triples that modulate the
//! normalized activations on each residual path.
//!
//! [`AdaLNModulation`] produces the six-chunk projection used by a
//! two-residual block (one for the attention path, one for the
//! feed-forward path). [`modulate`] applies a single `(shift, scale)`
//! pair to a sequence of activations with the canonical
//! `(1 + scale) * x + shift` formula.

#![allow(clippy::similar_names)]
#![allow(clippy::needless_pass_by_value)]

use candle_core::{D, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};

/// AdaLN modulation projector: takes the global conditioning vector
/// and produces six chunks per block: `(scale_self, shift_self,
/// gate_self, scale_ff, shift_ff, gate_ff)`.
///
/// The Stable Audio reference uses a `SiLU → Linear` stack and chunks
/// the output along the last axis after unsqueezing on the sequence
/// dimension. We preserve that exact layout so safetensors weight
/// dumps land cleanly.
#[derive(Debug)]
pub struct AdaLNModulation {
    /// Linear projection from `embed_dim` to `6 * embed_dim` (no bias
    /// in the reference). Note the SiLU is applied to the *input*, not
    /// the output, of this linear.
    proj: Linear,
    embed_dim: usize,
}

impl AdaLNModulation {
    /// Construct from a `VarBuilder`. The Linear weight lives at
    /// `weight` (the reference's
    /// `self.to_scale_shift_gate[1].weight`).
    pub fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(embed_dim, embed_dim * 6, vb)?;
        Ok(Self { proj, embed_dim })
    }

    /// Apply `SiLU` to the global conditioning, project to 6 chunks,
    /// and split. Returns `(scale_self, shift_self, gate_self,
    /// scale_ff, shift_ff, gate_ff)`, each shaped `(B, 1, embed_dim)`
    /// so they broadcast over the audio sequence.
    pub fn forward(
        &self,
        global_cond: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // global_cond is (B, embed_dim).
        let activated = candle_nn::ops::silu(global_cond)?;
        let projected = self.proj.forward(&activated)?;
        // Insert the sequence axis so we can broadcast over T_audio.
        let projected = projected.unsqueeze(1)?; // (B, 1, 6*embed_dim)
        // Split along the last axis into 6 equal-size chunks.
        let mut chunks = Vec::with_capacity(6);
        for i in 0..6 {
            let start = i * self.embed_dim;
            let chunk = projected.narrow(D::Minus1, start, self.embed_dim)?;
            chunks.push(chunk);
        }
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

/// AdaLN modulation: `(1 + scale) * x + shift`, broadcasting `scale`
/// and `shift` from `(B, 1, D)` over the sequence axis of `x = (B, T,
/// D)`.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(scale)?;
    let scaled = x.broadcast_mul(&(scale + one)?)?;
    scaled.broadcast_add(shift)
}
