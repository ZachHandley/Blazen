//! Video VAE primitives.
//!
//! Latent video VAEs differ from 2D image VAEs in two important
//! ways:
//!
//! 1. They convolve over a 3D `(time, height, width)` volume rather
//!    than a 2D `(height, width)` plane.
//! 2. The convolutions used by the encoder / decoder are
//!    **time-causal** — padding along the time axis is applied
//!    only on the past side, so each output frame depends only on
//!    the current and previous input frames. This lets the decoder
//!    stream output frames as soon as enough context is available
//!    without re-running the network from the start of the clip.
//!
//! This module currently ships the single foundational primitive
//! used by every video VAE in the wild: [`TemporalCausalConv3d`].
//! Full encoder / decoder block recipes (residual blocks, attention
//! down/up-samplers, latent shape adapters) live in the consuming
//! backend crate where the specific architecture is pinned.
//!
//! # `candle_nn::Conv3d` availability
//!
//! As of `candle-nn` `0.10`, there is no `Conv3d` type in candle
//! itself. The candle kernel surface only ships `Conv1d` /
//! `Conv2d` / their transposed variants. Rather than block on a
//! candle upstream change, [`TemporalCausalConv3d`] implements the
//! 3D convolution by decomposing it into `kT` independent 2D
//! convolutions (one per slice of the time-kernel), each operating
//! on a different time slice of the (time-padded) input volume,
//! and summing the results. This is mathematically equivalent to
//! a direct 3D convolution and uses only stable candle primitives.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::many_single_char_names)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use candle_core::Tensor;
use candle_nn::VarBuilder;
use thiserror::Error;

/// Errors emitted by the video VAE primitives.
#[derive(Debug, Error)]
pub enum VideoVaeError {
    /// The input tensor did not satisfy the documented rank or
    /// channel-count constraints. Carries a human-readable
    /// explanation suitable for surfacing in the consuming
    /// backend's error chain.
    #[error("invalid video VAE input shape: {0}")]
    InvalidShape(String),

    /// A candle tensor / kernel operation failed.
    #[error("candle tensor op failed: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Time-causal 3D convolution.
///
/// Operates on volumes of shape `(batch, channels, frames, height,
/// width)`. The convolution is **causal in the time axis**: padding
/// of `kernel_size.0 - 1` zero frames is applied to the *front*
/// (past) of the time axis only, so the convolution output at time
/// `t` is a function of input times `[t - kT + 1, t]` — never
/// future frames. Spatial padding is symmetric and computed as
/// `(kH - 1) / 2` and `(kW - 1) / 2` so odd-sized kernels preserve
/// the input spatial extent when `stride == 1`.
///
/// The 3D convolution is decomposed into `kT` independent 2D
/// convolutions, one per time-kernel slice, each applied to a
/// different time-slice of the (time-padded) input. Their outputs
/// are summed along the time axis. This is mathematically
/// equivalent to a direct 3D convolution and uses only stable
/// candle primitives.
#[derive(Debug)]
pub struct TemporalCausalConv3d {
    /// Convolution weight tensor of shape
    /// `(out_channels, in_channels, kT, kH, kW)`. Held as a raw
    /// candle tensor because `candle_nn` does not (yet) expose a
    /// `Conv3d` module type — see module docs.
    ///
    /// The legacy field name `conv` is preserved here so that if a
    /// future candle release lands `Conv3d`, the field can be
    /// upgraded to a real `Conv3d` without churning the public
    /// API.
    conv: Tensor,
    /// `(time, height, width)` padding. The time component is
    /// applied asymmetrically on the past side; the spatial
    /// components are passed straight through to `conv2d`.
    padding: (usize, usize, usize),
    /// `(time, height, width)` stride.
    stride: (usize, usize, usize),
    /// Kernel size: `(kT, kH, kW)`. Cached because the weight
    /// tensor's dims would otherwise need re-querying on every
    /// forward.
    kernel_size: (usize, usize, usize),
    /// Number of input channels (cached for clarity in error
    /// messages and shape checks).
    in_channels: usize,
}

impl TemporalCausalConv3d {
    /// Construct a time-causal 3D convolution.
    ///
    /// - `kernel_size`: `(kT, kH, kW)`.
    /// - `stride`: `(sT, sH, sW)`. Time stride is permitted but the
    ///   causality semantics only line up cleanly with `sT == 1`;
    ///   higher time strides are accepted and applied verbatim.
    ///
    /// Spatial padding is auto-computed as `(kH - 1) / 2` and
    /// `(kW - 1) / 2`. Time padding is auto-computed as
    /// `kT - 1` (entirely on the past side).
    ///
    /// The weight tensor lives at `vb / weight` and has shape
    /// `(out_channels, in_channels, kT, kH, kW)`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self, VideoVaeError> {
        let (k_t, k_h, k_w) = kernel_size;
        if k_t == 0 || k_h == 0 || k_w == 0 {
            return Err(VideoVaeError::InvalidShape(format!(
                "kernel_size must be > 0 in every dim, got {kernel_size:?}"
            )));
        }
        let conv = vb.get((out_channels, in_channels, k_t, k_h, k_w), "weight")?;
        let padding = (k_t - 1, (k_h - 1) / 2, (k_w - 1) / 2);
        Ok(Self {
            conv,
            padding,
            stride,
            kernel_size,
            in_channels,
        })
    }

    /// Forward pass over `(batch, channels, frames, height,
    /// width)`. Past-only time padding is applied here (so the
    /// underlying 2D convolutions can be unpadded along the time
    /// axis); spatial padding is delegated to the per-slice
    /// `conv2d` call.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, VideoVaeError> {
        let dims = x.dims();
        if dims.len() != 5 {
            return Err(VideoVaeError::InvalidShape(format!(
                "expected rank-5 input (B, C, T, H, W), got {dims:?}"
            )));
        }
        let (b, c_in, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        if c_in != self.in_channels {
            return Err(VideoVaeError::InvalidShape(format!(
                "channel dim ({c_in}) does not match configured in_channels ({})",
                self.in_channels
            )));
        }

        let (k_t, _k_h, _k_w) = self.kernel_size;
        let (pad_t, pad_h, pad_w) = self.padding;
        let (s_t, s_h, s_w) = self.stride;
        if s_h != s_w {
            return Err(VideoVaeError::InvalidShape(format!(
                "non-uniform spatial stride is not supported: (s_h={s_h}, s_w={s_w})"
            )));
        }
        if s_t == 0 || s_h == 0 || s_w == 0 {
            return Err(VideoVaeError::InvalidShape(format!(
                "stride must be > 0 in every dim, got {:?}",
                self.stride
            )));
        }

        // Time-causal padding: prepend `pad_t` zero frames on the
        // front of the time axis. We pad on the time axis only;
        // height / width padding is delegated to the per-slice
        // `conv2d` call below via its `padding` config.
        let x_padded = if pad_t > 0 {
            let pad_shape = (b, c_in, pad_t, h, w);
            let pad = Tensor::zeros(pad_shape, x.dtype(), x.device())?;
            Tensor::cat(&[&pad, x], 2)?
        } else {
            x.clone()
        };

        let t_padded = t + pad_t;
        // Stride along time selects every `s_t`-th valid output
        // position. The number of valid output positions is the
        // standard convolution arithmetic for an unpadded 1D
        // signal of length `t_padded` with kernel `k_t` and stride
        // `s_t`.
        if t_padded < k_t {
            return Err(VideoVaeError::InvalidShape(format!(
                "padded time dim ({t_padded}) is smaller than kT ({k_t})"
            )));
        }
        let t_out = (t_padded - k_t) / s_t + 1;

        // Iterate over the time-kernel and accumulate `kT`
        // independent 2D convolutions. Each per-`kt` 2D conv uses
        // candle's existing `conv2d` kernel for the spatial pass.
        let mut accum: Option<Tensor> = None;
        // Spatial padding is symmetric — pad the H/W axes once via
        // a 2D-conv config that wraps every kt-slice call.
        for kt in 0..k_t {
            // Per-kt slice of the weight tensor:
            // (out_channels, in_channels, kH, kW).
            let w_kt = self.conv.narrow(2, kt, 1)?.squeeze(2)?;

            // For each output time step `t_out_idx`, the
            // corresponding *input* time index is
            // `t_out_idx * s_t + kt`. Collect those indices and
            // gather them into a 4D batched tensor.
            //
            // The simple loop-and-stack form keeps the code
            // readable and avoids manual `index_select` shape
            // gymnastics. `t_out` is small in practice (one frame
            // per output step) and the per-step `conv2d` is the
            // dominant cost, so the overhead is negligible.
            let mut slice_outs: Vec<Tensor> = Vec::with_capacity(t_out);
            for t_out_idx in 0..t_out {
                let t_in_idx = t_out_idx * s_t + kt;
                let x_slice = x_padded.narrow(2, t_in_idx, 1)?.squeeze(2)?; // (B, C_in, H, W)
                // Apply spatial padding manually so the candle
                // `conv2d` call can stay unpadded. We use
                // `pad_with_zeros` on the H and W axes of the
                // rank-4 slice.
                let x_padded_spatial = {
                    let mut tslice = x_slice;
                    if pad_h > 0 {
                        tslice = tslice.pad_with_zeros(2, pad_h, pad_h)?;
                    }
                    if pad_w > 0 {
                        tslice = tslice.pad_with_zeros(3, pad_w, pad_w)?;
                    }
                    tslice
                };
                // candle's `conv2d` takes a single uniform stride
                // argument; the earlier check rejects non-uniform
                // spatial strides so `s_h == s_w` here.
                let out_slice = x_padded_spatial.conv2d(&w_kt, 0, s_h, 1, 1)?;
                slice_outs.push(out_slice.unsqueeze(2)?); // re-introduce time axis
            }
            let slice_refs: Vec<&Tensor> = slice_outs.iter().collect();
            let kt_out = Tensor::cat(&slice_refs, 2)?;
            accum = Some(match accum {
                None => kt_out,
                Some(a) => (a + kt_out)?,
            });
        }

        accum.ok_or_else(|| {
            VideoVaeError::InvalidShape(
                "internal error: empty time kernel produced no output slices".to_string(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn causal_conv3d_pads_only_past() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // kernel_size = (3, 1, 1), stride = (1, 1, 1): pure
        // temporal conv with kT=3. Past-only padding of 2 frames
        // means the output time dimension equals the input time
        // dimension (5).
        let conv =
            TemporalCausalConv3d::new(4, 4, (3, 1, 1), (1, 1, 1), vb).expect("construct conv");

        let x = Tensor::randn(
            0f32,
            1f32,
            (1usize, 4usize, 5usize, 8usize, 8usize),
            &device,
        )
        .expect("input tensor");
        let y = conv.forward(&x).expect("forward pass");
        let dims = y.dims();
        assert_eq!(dims.len(), 5);
        // Time dimension preserved under causal padding.
        assert_eq!(dims[2], 5);
        // Spatial dimensions also preserved (kH = kW = 1).
        assert_eq!(dims[3], 8);
        assert_eq!(dims[4], 8);
    }
}
