//! Shared low-level primitives for the `Spark-TTS` `BiCodec` codec.
//!
//! This is sub-wave **S.2.1.a** of the [`super`] `BiCodec` port: every
//! subsequent sub-wave (vocos, sampler, quantizer, speaker, decoder,
//! top-level) consumes one or more of these building blocks. Keeping them
//! in a single module mirrors the upstream
//! `sparktts/modules/blocks/layers.py` + `…/vocos.py` organization and
//! lets future waves pull `use super::primitives::{…};` without churn.
//!
//! # Primitives
//!
//! * [`Snake1d`] — DAC-style learnable activation
//!   `x + (1/alpha) * sin(alpha * x)^2` with a per-channel learnable
//!   `alpha` (initial value `1.0` upstream).
//! * [`WeightNormConv1d`] / [`WeightNormConvTranspose1d`] — load-time
//!   fold of `PyTorch` `torch.nn.utils.weight_norm` parametrised
//!   conv kernels (`weight_g` gain × `weight_v` direction / `||weight_v||`)
//!   into a single dense kernel handed to [`candle_nn::Conv1d`] /
//!   [`candle_nn::ConvTranspose1d`].
//! * [`AdaLayerNorm`] — vocos-style adaptive layer norm with
//!   condition-projected scale + shift.
//! * [`ResidualUnit`] — DAC residual unit used by every `BiCodec`
//!   encoder/decoder block.
//! * [`repeat_interleave_dim2`] — `torch.repeat_interleave(x, n, dim=2)`
//!   shim (candle has no native equivalent).
//!
//! # Weight-norm fold convention
//!
//! Both `Conv1d` and `ConvTranspose1d` are wrapped with
//! `torch.nn.utils.weight_norm` in the upstream Python — which stores
//! the weight as `weight_g` (a per-channel gain along `PyTorch`'s `dim=0`)
//! plus `weight_v` (an unnormalised direction tensor). At load time we
//! synthesize the dense kernel:
//!
//! ```text
//!   weight = weight_g * weight_v / ||weight_v||_{axes (1, 2)}
//! ```
//!
//! For `Conv1d`, `PyTorch`'s weight tensor is shaped
//! `(out_channels, in_channels / groups, kernel_size)` and `weight_norm`
//! defaults to `dim=0`, so `weight_g` is `(out_channels, 1, 1)`.
//!
//! For `ConvTranspose1d`, `PyTorch`'s weight tensor is shaped
//! `(in_channels, out_channels / groups, kernel_size)` — the *input*
//! channels are dim 0. `weight_norm(dim=0)` therefore makes `weight_g`
//! shape `(in_channels, 1, 1)`, NOT `(out_channels, 1, 1)`. This matches
//! [`candle_transformers::models::encodec::conv_transpose1d_weight_norm`]
//! which uses `(in_c, 1, 1)` for `weight_g` and `(in_c, out_c, k)` for
//! `weight_v`. Getting this axis wrong loads the conv successfully but
//! produces silently-garbage outputs (per the upstream `PyTorch`
//! `WeightNorm.dim` documentation).
//!
//! # Numerical stability
//!
//! Upstream `snake` adds `+ 1e-9` to `alpha` before reciprocation so a
//! zero-initialized `alpha` (or a random-init under a fresh
//! [`candle_nn::VarMap`] in tests) doesn't blow up. We replicate that
//! guard verbatim.

// `VarBuilder` is the canonical "consume by value" handle in candle: every
// load method takes ownership so callers can chain `vb.pp(...)` cleanly.
// The clippy lint here would force us to take `&VarBuilder` everywhere,
// which breaks the established pattern used by every other Blazen
// `*::load(vb, ...)` constructor.
#![allow(clippy::needless_pass_by_value)]

use candle_core::{DType, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder,
    linear,
};

// ---------------------------------------------------------------------------
// Snake1d
// ---------------------------------------------------------------------------

/// DAC-style snake activation:
/// `x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2` with a per-channel
/// learnable `alpha` of shape `(1, C, 1)`.
///
/// Upstream: `sparktts/modules/blocks/layers.py::Snake1d`.
#[derive(Debug, Clone)]
pub struct Snake1d {
    /// Learnable per-channel alpha, shape `(1, channels, 1)`.
    /// Initialised to `1.0` in the upstream Python `__init__`.
    alpha: Tensor,
}

impl Snake1d {
    /// Load the activation's `alpha` tensor under `vb / "alpha"`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the underlying
    /// [`VarBuilder::get`] (missing key, dtype / shape mismatch).
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    /// Forward `(B, C, T) -> (B, C, T)`.
    ///
    /// # Errors
    ///
    /// Propagates any [`candle_core::Error`] from the underlying tensor
    /// arithmetic.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let alpha = self.alpha.to_dtype(dtype)?;
        // `+1e-9` clamp matches the upstream `(alpha + 1e-9).reciprocal()`
        // in `@torch.jit.script snake`. Without it a zero-initialized
        // alpha (random VarBuilder under tests) divides by zero. The
        // pretrained checkpoint never ships alpha ≈ 0 so the guard is
        // effectively a no-op in production.
        let eps = Tensor::new(1e-9_f32, x.device())?.to_dtype(dtype)?;
        let alpha_safe = alpha.broadcast_add(&eps)?;
        let inv_alpha = alpha_safe.recip()?;
        let scaled = x.broadcast_mul(&alpha)?;
        let sin_sq = scaled.sin()?.sqr()?;
        let bump = sin_sq.broadcast_mul(&inv_alpha)?;
        x + bump
    }
}

impl Module for Snake1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Snake1d::forward(self, x)
    }
}

// ---------------------------------------------------------------------------
// Weight-norm conv folders
// ---------------------------------------------------------------------------

/// Numerical clamp added to the `||weight_v||` denominator so a fresh
/// zero-initialized [`candle_nn::VarBuilder`] (random-init in unit
/// tests) doesn't divide by zero. The pretrained Spark-TTS checkpoint
/// always has non-zero `weight_v`, so production loads see no
/// detectable difference.
const NORM_EPS: f64 = 1e-12;

/// 1-D conv whose checkpoint stores `PyTorch` `weight_norm`-parametrised
/// kernels (`weight_g` + `weight_v`). The dense kernel is synthesized at
/// load time and the runtime wrapper is a plain [`candle_nn::Conv1d`].
///
/// Upstream Python wraps every conv in `WNConv1d = weight_norm(nn.Conv1d(…))`,
/// which means the state-dict keys under this `VarBuilder` are
/// `weight_g`, `weight_v`, and (optionally) `bias`.
#[derive(Debug, Clone)]
pub struct WeightNormConv1d {
    inner: Conv1d,
}

impl WeightNormConv1d {
    /// Synthesize the dense kernel from `weight_g` × `weight_v` /
    /// `||weight_v||` and wrap it as a [`candle_nn::Conv1d`].
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from missing tensors or shape
    /// mismatches.
    #[allow(
        clippy::too_many_arguments,
        reason = "matches PyTorch Conv1d's surface"
    )]
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
    ) -> Result<Self> {
        // PyTorch Conv1d weight shape: (out, in / groups, kernel).
        // `weight_norm(dim=0)` makes `weight_g` shape (out, 1, 1).
        let in_per_group = in_channels / groups;
        let weight_g = vb.get((out_channels, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_channels, in_per_group, kernel_size), "weight_v")?;

        let norm = weight_v.sqr()?.sum_keepdim((1, 2))?;
        let norm = (norm + NORM_EPS)?.sqrt()?;
        let weight = weight_g.broadcast_mul(&weight_v.broadcast_div(&norm)?)?;

        let bias = if with_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };

        let cfg = Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
            ..Conv1dConfig::default()
        };
        let inner = Conv1d::new(weight, bias, cfg);
        Ok(Self { inner })
    }

    /// Forward through the synthesized dense conv.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the underlying conv.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

impl Module for WeightNormConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        WeightNormConv1d::forward(self, x)
    }
}

/// 1-D transposed conv whose checkpoint stores `PyTorch` `weight_norm`-
/// parametrised kernels.
///
/// Crucially the `weight_g` axis convention differs from
/// [`WeightNormConv1d`]: `PyTorch`'s `ConvTranspose1d` weight is shaped
/// `(in_channels, out_channels / groups, kernel_size)`, so
/// `weight_norm(dim=0)` yields `weight_g` of shape `(in_channels, 1, 1)`
/// — *not* `(out_channels, 1, 1)`. The L2 norm of `weight_v` is taken
/// over axes (1, 2) just like the conv case, preserving the input-
/// channel axis. Mirrors
/// [`candle_transformers::models::encodec::conv_transpose1d_weight_norm`].
#[derive(Debug, Clone)]
pub struct WeightNormConvTranspose1d {
    inner: ConvTranspose1d,
}

impl WeightNormConvTranspose1d {
    /// Synthesize the dense kernel and wrap it as a
    /// [`candle_nn::ConvTranspose1d`].
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from missing tensors or shape
    /// mismatches.
    #[allow(
        clippy::too_many_arguments,
        reason = "matches PyTorch ConvTranspose1d's surface"
    )]
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        with_bias: bool,
    ) -> Result<Self> {
        // PyTorch ConvTranspose1d weight shape: (in, out / groups, kernel).
        // `weight_norm(dim=0)` makes `weight_g` shape (in, 1, 1).
        let out_per_group = out_channels / groups;
        let weight_g = vb.get((in_channels, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_channels, out_per_group, kernel_size), "weight_v")?;

        let norm = weight_v.sqr()?.sum_keepdim((1, 2))?;
        let norm = (norm + NORM_EPS)?.sqrt()?;
        let weight = weight_g.broadcast_mul(&weight_v.broadcast_div(&norm)?)?;

        let bias = if with_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };

        let cfg = ConvTranspose1dConfig {
            padding,
            output_padding: 0,
            stride,
            dilation: 1,
            groups,
        };
        let inner = ConvTranspose1d::new(weight, bias, cfg);
        Ok(Self { inner })
    }

    /// Forward through the synthesized dense transposed conv.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the underlying conv.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

impl Module for WeightNormConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        WeightNormConvTranspose1d::forward(self, x)
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNorm
// ---------------------------------------------------------------------------

/// Adaptive layer norm: projects a per-batch condition vector into a
/// scale + shift, then applies
/// `LayerNorm(x) * scale.unsqueeze(1) + shift.unsqueeze(1)`.
///
/// Layout matches upstream `sparktts/modules/blocks/vocos.py::AdaLayerNorm`:
///
/// * Input `x` is shaped `(B, T, C)` (channels-last; `LayerNorm` is over
///   the last dim).
/// * Condition `cond` is shaped `(B, condition_dim)`.
/// * Output is `(B, T, C)`.
///
/// Note the formula is `x * scale + shift` (not `x * (1 + scale) + shift`).
/// Upstream initialises `scale.weight = ones` and `shift.weight = zeros`
/// so that at init `scale ≈ cond * 1 + scale.bias_random` and
/// `shift ≈ shift.bias_random` — the +1 trick is *not* used here.
#[derive(Debug, Clone)]
pub struct AdaLayerNorm {
    scale: Linear,
    shift: Linear,
    normalized_shape: usize,
    eps: f64,
}

impl AdaLayerNorm {
    /// Load the two projection Linears under `vb / "scale"` and
    /// `vb / "shift"`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from missing tensors or shape
    /// mismatches.
    pub fn load(
        vb: VarBuilder,
        condition_dim: usize,
        normalized_shape: usize,
        eps: f64,
    ) -> Result<Self> {
        let scale = linear(condition_dim, normalized_shape, vb.pp("scale"))?;
        let shift = linear(condition_dim, normalized_shape, vb.pp("shift"))?;
        Ok(Self {
            scale,
            shift,
            normalized_shape,
            eps,
        })
    }

    /// Forward: `x: (B, T, C)`, `condition: (B, condition_dim)` ->
    /// `(B, T, C)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the projection or the
    /// underlying [`candle_nn::ops::layer_norm`] call.
    pub fn forward(&self, x: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let (b, _t, c) = x.dims3()?;
        if c != self.normalized_shape {
            candle_core::bail!(
                "AdaLayerNorm: last dim mismatch, expected {}, got {c}",
                self.normalized_shape
            );
        }
        let (cb, _cd) = condition.dims2()?;
        if cb != b {
            candle_core::bail!("AdaLayerNorm: batch mismatch between x ({b}) and condition ({cb})");
        }

        // LayerNorm along the last (channel) dim. We don't load
        // affine weights here because the upstream Python uses the
        // *unaffine* `F.layer_norm(x, (dim,), eps=eps)` followed by
        // the conditional scale/shift — no learnable gamma/beta on the
        // LN itself.
        let xn = layer_norm_no_affine(x, self.eps)?;

        // (B, C) → (B, 1, C) so it broadcasts over T.
        let scale = self.scale.forward(condition)?.unsqueeze(1)?;
        let shift = self.shift.forward(condition)?.unsqueeze(1)?;

        xn.broadcast_mul(&scale)?.broadcast_add(&shift)
    }
}

/// Affine-free `LayerNorm` over the last dim of `x`. Mirrors
/// `torch.nn.functional.layer_norm(x, (C,), eps=eps)` — there is no
/// learnable gamma/beta because `AdaLayerNorm`'s scale/shift come from
/// the projected condition, not from a stored affine.
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    // Use candle_nn::ops::rms_norm style: subtract mean, divide by
    // sqrt(var + eps). Done in f32 for numerical stability and cast
    // back to the input dtype.
    let in_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
    let centered = x_f32.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let inv_std = (var + eps)?.recip()?.sqrt()?;
    let normalized = centered.broadcast_mul(&inv_std)?;
    normalized.to_dtype(in_dtype)
}

// ---------------------------------------------------------------------------
// ResidualUnit
// ---------------------------------------------------------------------------

/// DAC-style residual unit:
///
/// ```text
///   y = WNConv1d(1, 1)( Snake1d( WNConv1d(7, dil)( Snake1d(x) ) ) )
///   out = center_crop(x, len(y)) + y
/// ```
///
/// Upstream: `sparktts/modules/blocks/layers.py::ResidualUnit`. The
/// upstream wraps the four sub-ops in `nn.Sequential(Snake1d, WNConv1d,
/// Snake1d, WNConv1d)`, so the state-dict keys nest under `block.0`,
/// `block.1`, `block.2`, `block.3`. We follow that path verbatim so
/// official checkpoints load with no renaming.
#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: WeightNormConv1d,
    snake2: Snake1d,
    conv2: WeightNormConv1d,
}

impl ResidualUnit {
    /// Load a residual unit. Upstream uses 7-tap dilated conv +
    /// 1-tap pointwise conv with `padding = ((7 - 1) * dilation) / 2`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any of the four child
    /// loaders.
    pub fn load(vb: VarBuilder, dim: usize, dilation: usize) -> Result<Self> {
        const KERNEL: usize = 7;
        let pad = ((KERNEL - 1) * dilation) / 2;
        let block = vb.pp("block");
        let snake1 = Snake1d::load(block.pp("0"), dim)?;
        let conv1 = WeightNormConv1d::load(
            block.pp("1"),
            dim,
            dim,
            KERNEL,
            /* stride */ 1,
            pad,
            dilation,
            /* groups */ 1,
            /* with_bias */ true,
        )?;
        let snake2 = Snake1d::load(block.pp("2"), dim)?;
        let conv2 = WeightNormConv1d::load(
            block.pp("3"),
            dim,
            dim,
            /* kernel */ 1,
            /* stride */ 1,
            /* padding */ 0,
            /* dilation */ 1,
            /* groups */ 1,
            /* with_bias */ true,
        )?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }

    /// Forward `(B, C, T) -> (B, C, T')` where `T'` is the
    /// centre-cropped temporal length of `x` to match `y`'s length when
    /// upstream truncates the residual.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the sub-modules or tensor
    /// arithmetic.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.snake2.forward(&h)?;
        let y = self.conv2.forward(&h)?;
        // Upstream centre-crops `x` to match `y`'s length when same-
        // padding rounding makes them differ:
        //   pad = (x.shape[-1] - y.shape[-1]) // 2
        //   if pad > 0: x = x[..., pad:-pad]
        let x_t = x.dim(2)?;
        let y_t = y.dim(2)?;
        let residual = if x_t > y_t {
            let pad = (x_t - y_t) / 2;
            x.narrow(2, pad, y_t)?
        } else {
            x.clone()
        };
        residual + y
    }
}

// ---------------------------------------------------------------------------
// repeat_interleave_dim2
// ---------------------------------------------------------------------------

/// `torch.repeat_interleave(x, repeats, dim=2)` shim.
///
/// Expands `(B, C, T)` to `(B, C, T * repeats)` by repeating each
/// element along the time axis `repeats` times consecutively. Matches
/// the `PyTorch` semantics used inside Spark-TTS's `SamplingBlock` skip
/// connections.
///
/// # Errors
///
/// Returns [`candle_core::Error`] if `repeats == 0`, if `x` doesn't have
/// at least 3 dims, or if the underlying reshape/expand fails.
pub fn repeat_interleave_dim2(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 0 {
        candle_core::bail!("repeat_interleave_dim2: repeats must be >= 1");
    }
    if repeats == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims();
    if dims.len() < 3 {
        candle_core::bail!("repeat_interleave_dim2: expected >=3D tensor, got {dims:?}");
    }
    let b = dims[0];
    let c = dims[1];
    let t = dims[2];

    // (B, C, T) → (B, C, T, 1) → broadcast/expand to (B, C, T, repeats)
    // → reshape to (B, C, T * repeats).
    let x4 = x.unsqueeze(3)?;
    let expanded = x4.expand(&[b, c, t, repeats])?.contiguous()?;
    expanded.reshape(&[b, c, t * repeats])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "tests construct small deterministic vectors via `usize as f32` \
              indices; ranges are tiny (< 2^23) so precision is exact."
)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    /// Approximate-equality helper for flat f32 vectors.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "mismatch at index {i}: got {a}, expected {e} (tol {tol})"
            );
        }
    }

    fn vb_from(map: HashMap<String, Tensor>, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_tensors(map, DType::F32, dev)
    }

    #[test]
    fn snake1d_forward_shape_preserved() -> Result<()> {
        let dev = Device::Cpu;
        let channels = 16;
        // alpha = 1.0 → snake(x) = x + sin(x)^2 (modulo the 1e-9 clamp).
        let alpha = Tensor::ones((1, channels, 1), DType::F32, &dev)?;
        let mut map = HashMap::new();
        map.insert("alpha".to_owned(), alpha);
        let vb = vb_from(map, &dev);

        let snake = Snake1d::load(vb, channels)?;

        // Mildly random-looking input: arithmetic sequence reshaped to
        // (2, 16, 32). We compare element-wise to `x + sin(x)^2`.
        let n = 2 * channels * 32;
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let x = Tensor::from_vec(xs.clone(), (2, channels, 32), &dev)?;

        let y = snake.forward(&x)?;
        assert_eq!(y.dims(), &[2, channels, 32]);

        // Expected: x + sin(x)^2 (the 1e-9 clamp on alpha=1.0 is well
        // below the tolerance).
        let expected: Vec<f32> = xs.iter().map(|v| v + v.sin().powi(2)).collect();
        let got = y.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&got, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn weight_norm_conv1d_load_fold_matches_dense() -> Result<()> {
        let dev = Device::Cpu;
        let out_c = 4;
        let in_c = 3;
        let k = 5;

        // Deterministic synthetic weights.
        let g_vec: Vec<f32> = (0..out_c).map(|i| 0.5 + (i as f32) * 0.25).collect();
        let weight_g = Tensor::from_vec(g_vec.clone(), (out_c, 1, 1), &dev)?;
        let v_vec: Vec<f32> = (0..(out_c * in_c * k))
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let weight_v = Tensor::from_vec(v_vec.clone(), (out_c, in_c, k), &dev)?;
        let bias_vec: Vec<f32> = (0..out_c).map(|i| (i as f32) * 0.1).collect();
        let bias = Tensor::from_vec(bias_vec.clone(), out_c, &dev)?;

        // Manual reference weight = weight_g * weight_v / ||weight_v||_(1,2)
        let norm = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let manual_w = weight_g.broadcast_mul(&weight_v.broadcast_div(&norm)?)?;
        let cfg = Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Conv1dConfig::default()
        };
        let manual_conv = Conv1d::new(manual_w, Some(bias.clone()), cfg);

        let mut map = HashMap::new();
        map.insert("weight_g".to_owned(), weight_g);
        map.insert("weight_v".to_owned(), weight_v);
        map.insert("bias".to_owned(), bias);
        let vb = vb_from(map, &dev);

        let wnc = WeightNormConv1d::load(
            vb, in_c, out_c, k, /* stride */ 1, /* padding */ 1, /* dilation */ 1,
            /* groups */ 1, /* with_bias */ true,
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, in_c, 17), &dev)?;
        let got = wnc.forward(&x)?;
        let expected = manual_conv.forward(&x)?;
        assert_eq!(got.dims(), expected.dims());

        let diff = (&got - &expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max abs diff {diff} exceeds 1e-5");
        Ok(())
    }

    #[test]
    fn weight_norm_conv_transpose1d_load_fold_matches_dense() -> Result<()> {
        let dev = Device::Cpu;
        // Note the axis convention: weight_g is (in_c, 1, 1), NOT
        // (out_c, 1, 1).
        let in_c = 3;
        let out_c = 4;
        let k = 5;

        let g_vec: Vec<f32> = (0..in_c).map(|i| 0.7 + (i as f32) * 0.2).collect();
        let weight_g = Tensor::from_vec(g_vec.clone(), (in_c, 1, 1), &dev)?;
        let v_vec: Vec<f32> = (0..(in_c * out_c * k))
            .map(|i| ((i as f32) * 0.17).cos())
            .collect();
        let weight_v = Tensor::from_vec(v_vec.clone(), (in_c, out_c, k), &dev)?;
        let bias_vec: Vec<f32> = (0..out_c).map(|i| (i as f32) * 0.05 - 0.1).collect();
        let bias = Tensor::from_vec(bias_vec.clone(), out_c, &dev)?;

        let norm = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let manual_w = weight_g.broadcast_mul(&weight_v.broadcast_div(&norm)?)?;
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride: 2,
            dilation: 1,
            groups: 1,
        };
        let manual_conv = ConvTranspose1d::new(manual_w, Some(bias.clone()), cfg);

        let mut map = HashMap::new();
        map.insert("weight_g".to_owned(), weight_g);
        map.insert("weight_v".to_owned(), weight_v);
        map.insert("bias".to_owned(), bias);
        let vb = vb_from(map, &dev);

        let wnc = WeightNormConvTranspose1d::load(
            vb, in_c, out_c, k, /* stride */ 2, /* padding */ 0, /* groups */ 1,
            /* with_bias */ true,
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, in_c, 8), &dev)?;
        let got = wnc.forward(&x)?;
        let expected = manual_conv.forward(&x)?;
        assert_eq!(got.dims(), expected.dims());

        let diff = (&got - &expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max abs diff {diff} exceeds 1e-5");
        Ok(())
    }

    #[test]
    fn ada_layernorm_zero_condition_is_layernorm() -> Result<()> {
        let dev = Device::Cpu;
        let cond_dim = 8;
        let dim = 6;
        let eps = 1e-6;

        // Build weights so that condition = zeros yields scale = ones,
        // shift = zeros: scale.weight = zeros, scale.bias = ones,
        // shift.weight = zeros, shift.bias = zeros.
        let mut map = HashMap::new();
        map.insert(
            "scale.weight".to_owned(),
            Tensor::zeros((dim, cond_dim), DType::F32, &dev)?,
        );
        map.insert(
            "scale.bias".to_owned(),
            Tensor::ones(dim, DType::F32, &dev)?,
        );
        map.insert(
            "shift.weight".to_owned(),
            Tensor::zeros((dim, cond_dim), DType::F32, &dev)?,
        );
        map.insert(
            "shift.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        let vb = vb_from(map, &dev);

        let ada = AdaLayerNorm::load(vb, cond_dim, dim, eps)?;

        // Channels-last layout (B, T, C).
        let x = Tensor::randn(0.0_f32, 1.0, (2, 5, dim), &dev)?;
        let cond = Tensor::zeros((2, cond_dim), DType::F32, &dev)?;

        let got = ada.forward(&x, &cond)?;
        let expected = layer_norm_no_affine(&x, eps)?;

        let diff = (&got - &expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max abs diff {diff} exceeds 1e-5");
        Ok(())
    }

    #[test]
    fn residual_unit_zero_input_is_zero_after_residual() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 4;
        let dilation = 1;

        // Weights chosen so that block(zeros) = zeros:
        // * Snake1d alpha = 1 (output of snake on zero = 0).
        // * Conv weight_v non-zero (avoid the eps-clamp dominating), but
        //   bias = 0 so conv(zero) = 0.
        let mut map = HashMap::new();
        // block.0 (Snake1d): alpha = ones.
        map.insert(
            "block.0.alpha".to_owned(),
            Tensor::ones((1, dim, 1), DType::F32, &dev)?,
        );
        // block.1 (WeightNormConv1d, kernel=7): weight_g = ones,
        // weight_v = arbitrary non-zero, bias = zeros.
        map.insert(
            "block.1.weight_g".to_owned(),
            Tensor::ones((dim, 1, 1), DType::F32, &dev)?,
        );
        let v1_vec: Vec<f32> = (0..(dim * dim * 7))
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        map.insert(
            "block.1.weight_v".to_owned(),
            Tensor::from_vec(v1_vec, (dim, dim, 7), &dev)?,
        );
        map.insert(
            "block.1.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        // block.2 (Snake1d): alpha = ones.
        map.insert(
            "block.2.alpha".to_owned(),
            Tensor::ones((1, dim, 1), DType::F32, &dev)?,
        );
        // block.3 (WeightNormConv1d, kernel=1): weight_g = ones, etc.
        map.insert(
            "block.3.weight_g".to_owned(),
            Tensor::ones((dim, 1, 1), DType::F32, &dev)?,
        );
        let v3_vec: Vec<f32> = (0..(dim * dim)).map(|i| (i as f32) * 0.02 + 0.2).collect();
        map.insert(
            "block.3.weight_v".to_owned(),
            Tensor::from_vec(v3_vec, (dim, dim, 1), &dev)?,
        );
        map.insert(
            "block.3.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );

        let vb = vb_from(map, &dev);
        let unit = ResidualUnit::load(vb, dim, dilation)?;

        let x = Tensor::zeros((1, dim, 16), DType::F32, &dev)?;
        let y = unit.forward(&x)?;
        assert_eq!(y.dims(), &[1, dim, 16]);

        let max_abs = y.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max_abs < 1e-6, "max abs {max_abs} on zero-input residual");
        Ok(())
    }

    #[test]
    fn repeat_interleave_dim2_matches_torch_semantics() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (1, 1, 3), &dev)?;
        let y = repeat_interleave_dim2(&x, 2)?;
        assert_eq!(y.dims(), &[1, 1, 6]);
        let flat = y.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(flat, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

        // Also check repeats=1 round-trips.
        let y1 = repeat_interleave_dim2(&x, 1)?;
        assert_eq!(y1.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0]);

        // And a 3x repeat on a 2-channel tensor.
        let x2 = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 2, 2), &dev)?;
        let y2 = repeat_interleave_dim2(&x2, 3)?;
        assert_eq!(y2.dims(), &[1, 2, 6]);
        assert_eq!(
            y2.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
        Ok(())
    }
}
