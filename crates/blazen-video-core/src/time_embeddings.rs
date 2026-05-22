//! Sinusoidal diffusion timestep embeddings.
//!
//! This is the standard "DDPM-style" sinusoidal positional embedding
//! applied to a scalar timestep `t` to produce a `dim`-wide feature
//! vector that conditions the denoiser. Every modern video diffusion
//! backend (CogVideoX, HunyuanVideo, Mochi) wraps this in a small
//! MLP head, but the sinusoidal embedding itself is identical across
//! them and lives here.
//!
//! # Formula
//!
//! For an output dimension `dim` and half-dim `half = dim / 2`:
//!
//! ```text
//! freqs       = exp(-ln(10000) * arange(0, half) / half)   // (half,)
//! angles      = timesteps[:, None] * freqs[None, :]        // (B, half)
//! embeddings  = concat(sin(angles), cos(angles), dim=-1)    // (B, 2*half)
//! ```
//!
//! If `dim` is odd, a single trailing zero column is appended so the
//! returned tensor has exactly `dim` channels.

#![allow(clippy::module_name_repetitions)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use candle_core::{D, DType, Tensor};
use thiserror::Error;

/// Errors emitted by [`SinusoidalTimeEmbedding`].
#[derive(Debug, Error)]
pub enum TimeEmbeddingError {
    /// A candle tensor / kernel operation failed.
    #[error("candle tensor op failed: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Sinusoidal timestep embedding module.
///
/// Stateless aside from the output `dim`. Pre-computed frequency
/// tables are intentionally *not* cached on the struct because the
/// caller's device may differ between forward calls (CPU for
/// preprocessing, CUDA for the actual denoise step); building the
/// frequency tensor on the timesteps' device per-call costs a
/// vanishingly small fraction of the model's compute and keeps the
/// API trivial.
#[derive(Debug, Clone, Copy)]
pub struct SinusoidalTimeEmbedding {
    dim: usize,
}

impl SinusoidalTimeEmbedding {
    /// Construct a sinusoidal timestep embedding producing
    /// `dim`-channel feature vectors.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// The output channel dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Forward pass.
    ///
    /// - Input `timesteps`: rank-1 `f32` tensor of shape `(B,)`
    ///   (continuous timesteps; the formula works for both integer
    ///   and fractional `t`).
    /// - Output: rank-2 `f32` tensor of shape `(B, dim)`.
    ///
    /// If `dim` is odd, a single zero-valued trailing column is
    /// appended after the sin/cos concat.
    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor, TimeEmbeddingError> {
        let device = timesteps.device();
        let dtype = DType::F32;
        let timesteps = timesteps.to_dtype(dtype)?;
        let b = timesteps.dim(0)?;

        let half = self.dim / 2;
        let trailing_zero = !self.dim.is_multiple_of(2);

        // Edge case: zero-channel output. Return an empty (B, 0)
        // tensor rather than crash.
        if self.dim == 0 {
            return Ok(Tensor::zeros((b, 0usize), dtype, device)?);
        }

        // freqs = exp(-ln(10000) * arange(0, half) / half).
        // When half == 0 (dim == 1), we only have the trailing zero
        // column, so skip the sin/cos path entirely.
        let sin_cos = if half > 0 {
            let arange = Tensor::arange(0u32, half as u32, device)?.to_dtype(dtype)?;
            let scale = -f64::ln(10_000.0) / (half as f64);
            let freqs = (arange * scale)?.exp()?; // (half,)

            // angles = timesteps[:, None] * freqs[None, :]  →  (B, half)
            let ts_col = timesteps.unsqueeze(D::Minus1)?; // (B, 1)
            let freqs_row = freqs.unsqueeze(0)?; // (1, half)
            let angles = ts_col.broadcast_mul(&freqs_row)?; // (B, half)

            let sin = angles.sin()?;
            let cos = angles.cos()?;
            Some(Tensor::cat(&[&sin, &cos], D::Minus1)?) // (B, 2*half)
        } else {
            None
        };

        let result = match (sin_cos, trailing_zero) {
            (Some(sc), false) => sc,
            (Some(sc), true) => {
                let pad = Tensor::zeros((b, 1usize), dtype, device)?;
                Tensor::cat(&[&sc, &pad], D::Minus1)?
            }
            (None, true) => Tensor::zeros((b, 1usize), dtype, device)?,
            (None, false) => unreachable!("dim == 0 was handled above"),
        };

        debug_assert_eq!(result.dims(), &[b, self.dim]);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn shape_matches_dim() {
        let device = Device::Cpu;
        let emb = SinusoidalTimeEmbedding::new(128);
        let t = Tensor::from_slice(&[0.0f32, 1.0, 10.0, 999.0], (4,), &device)
            .expect("timesteps tensor");
        let y = emb.forward(&t).expect("forward pass");
        assert_eq!(y.dims(), &[4, 128]);
    }

    #[test]
    fn deterministic_for_same_timestep() {
        let device = Device::Cpu;
        let emb = SinusoidalTimeEmbedding::new(128);
        let t = Tensor::from_slice(&[0.0f32, 1.0, 10.0, 999.0], (4,), &device)
            .expect("timesteps tensor");
        let y1 = emb.forward(&t).expect("first forward");
        let y2 = emb.forward(&t).expect("second forward");

        let v1 = y1
            .flatten_all()
            .and_then(|x| x.to_vec1::<f32>())
            .expect("flatten y1");
        let v2 = y2
            .flatten_all()
            .and_then(|x| x.to_vec1::<f32>())
            .expect("flatten y2");
        assert_eq!(v1.len(), v2.len());
        // Byte-identical: every value must match bit-for-bit, not
        // just within a floating-point epsilon.
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "expected bit-identical outputs");
        }
    }
}
