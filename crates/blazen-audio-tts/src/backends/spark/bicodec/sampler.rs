//! Depthwise up/down sampler + the `BiCodec` feature `Encoder` that wraps
//! it around a stack of [`VocosBackbone`]s (sub-wave **S.2.1.c** of the
//! [`super`] `BiCodec` port).
//!
//! Mirrors `sparktts/modules/blocks/samper.py::SamplingBlock` and
//! `sparktts/modules/encoder_decoder/feat_encoder.py::Encoder`.
//!
//! # Channels-first I/O convention (diverges from upstream)
//!
//! Upstream `SamplingBlock.forward` starts with `x = x.transpose(1, 2)`
//! because `VocosBackbone` returns channels-last `(B, T, dim)` in
//! Python. Our [`super::vocos::VocosBackbone::forward`] returns
//! channels-first `(B, dim, T)` already (see the rationale in
//! [`super::vocos`]'s module docstring), so we skip that initial
//! transpose entirely and run the depthwise convs on `(B, C, T)`
//! directly. End-to-end semantics are unchanged.
//!
//! # State-dict layout
//!
//! Upstream wraps each conv/transpose conv in `nn.Sequential(LeakyReLU,
//! nn.Conv1d/ConvTranspose1d)`, so the state-dict keys are:
//!
//! * `de_conv_upsampler.1.weight`, `de_conv_upsampler.1.bias` —
//!   only present when `upsample_scale > 1`.
//! * `conv_downsampler.1.weight`, `conv_downsampler.1.bias` —
//!   only present when `downsample_scale > 1`.
//!
//! When `upsample_scale == 1` AND `downsample_scale == 1` (the standard
//! Spark-TTS checkpoint's `sample_ratios = [1, 1]` case), upstream
//! *does not instantiate* either child, so neither key prefix appears
//! in the checkpoint. We mirror that with `Option`-typed fields.
//!
//! # The two-scales-are-one case is NOT a no-op
//!
//! It is tempting to assume `SamplingBlock(upsample=1, downsample=1)`
//! returns `x` unchanged. It does **not**: upstream's forward returns
//! `conv_res + skip1_res + skip2_res`, and when both scales are 1
//! `conv_res = skip1_res = skip2_res = x`, so the output is `3 * x`.
//! This is preserved verbatim here — getting it wrong silently scales
//! every Spark-TTS encoder forward by `1 / 3`.

// `VarBuilder` is the canonical "consume by value" handle in candle.
// See primitives.rs for the same lint waiver and rationale.
#![allow(clippy::needless_pass_by_value)]

use candle_core::{Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder,
    linear,
};

use super::vocos::VocosBackbone;

/// `LeakyReLU` slope used by both `de_conv_upsampler` and
/// `conv_downsampler` upstream. Mirrors `nn.LeakyReLU(0.2)`.
const LEAKY_SLOPE: f64 = 0.2;

// ---------------------------------------------------------------------------
// SamplingBlock
// ---------------------------------------------------------------------------

/// Depthwise up/down sampler with `repeat_interleave` / `avg_pool1d`
/// skip connections.
///
/// Upstream signature (paraphrased):
///
/// ```text
///   if upsample_scale > 1:
///       upmerge = de_conv_upsampler(x) + repeat_interleave(x, upsample_scale)
///       repeat  = repeat_interleave(x, upsample_scale)
///   else:
///       upmerge = x; repeat = x
///   if downsample_scale > 1:
///       conv_res  = conv_downsampler(upmerge)
///       skip2_res = avg_pool1d(upmerge, downsample_scale)
///       skip1_res = avg_pool1d(repeat,  downsample_scale)
///   else:
///       conv_res = upmerge; skip1_res = repeat; skip2_res = upmerge
///   return conv_res + skip1_res + skip2_res
/// ```
///
/// The two `de_conv_upsampler` / `conv_downsampler` sub-modules are
/// `Optional` because upstream only instantiates them when the relevant
/// scale is `> 1` — see the module docstring.
pub(super) struct SamplingBlock {
    /// Transposed conv that drives the upsample. Wrapped in `Option`
    /// because upstream does not instantiate it when
    /// `upsample_scale == 1`.
    de_conv_upsampler: Option<ConvTranspose1d>,
    /// Strided conv that drives the downsample. Wrapped in `Option`
    /// because upstream does not instantiate it when
    /// `downsample_scale == 1`.
    conv_downsampler: Option<Conv1d>,
    upsample_scale: usize,
    downsample_scale: usize,
}

impl SamplingBlock {
    /// Load a `SamplingBlock`. Only consumes child keys under
    /// `vb / "de_conv_upsampler" / "1"` when `upsample_scale > 1` and
    /// under `vb / "conv_downsampler" / "1"` when `downsample_scale > 1`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from missing tensors or shape
    /// mismatches in the optional child convs.
    pub(super) fn load(
        vb: VarBuilder,
        dim: usize,
        groups: usize,
        upsample_scale: usize,
        downsample_scale: usize,
    ) -> Result<Self> {
        let de_conv_upsampler = if upsample_scale > 1 {
            // Upstream:
            //   ConvTranspose1d(dim, dim,
            //     kernel_size=upsample_scale*2,
            //     stride=upsample_scale,
            //     padding=upsample_scale // 2 + upsample_scale % 2,
            //     output_padding=upsample_scale % 2,
            //     groups=groups)
            let kernel = upsample_scale * 2;
            let padding = upsample_scale / 2 + upsample_scale % 2;
            let output_padding = upsample_scale % 2;
            let cfg = ConvTranspose1dConfig {
                padding,
                output_padding,
                stride: upsample_scale,
                dilation: 1,
                groups,
            };
            // `nn.Sequential(LeakyReLU, ConvTranspose1d)` — params live
            // under `de_conv_upsampler.1.{weight,bias}`.
            Some(candle_nn::conv_transpose1d(
                dim,
                dim,
                kernel,
                cfg,
                vb.pp("de_conv_upsampler").pp("1"),
            )?)
        } else {
            None
        };

        let conv_downsampler = if downsample_scale > 1 {
            // Upstream:
            //   Conv1d(dim, dim,
            //     kernel_size=2 * downsample_scale,
            //     stride=downsample_scale,
            //     padding=downsample_scale // 2 + downsample_scale % 2,
            //     groups=groups)
            let kernel = 2 * downsample_scale;
            let padding = downsample_scale / 2 + downsample_scale % 2;
            let cfg = Conv1dConfig {
                padding,
                stride: downsample_scale,
                dilation: 1,
                groups,
                ..Conv1dConfig::default()
            };
            // `nn.Sequential(LeakyReLU, Conv1d)` — params under
            // `conv_downsampler.1.{weight,bias}`.
            Some(candle_nn::conv1d(
                dim,
                dim,
                kernel,
                cfg,
                vb.pp("conv_downsampler").pp("1"),
            )?)
        } else {
            None
        };

        Ok(Self {
            de_conv_upsampler,
            conv_downsampler,
            upsample_scale,
            downsample_scale,
        })
    }

    /// Forward `(B, C, T) -> (B, C, T')`.
    ///
    /// `T'` is `T * upsample_scale` (when upsample) or
    /// `(T + 2 * pad - 2 * downsample_scale) / downsample_scale + 1`
    /// (when downsample). The standard checkpoint config `(1, 1)`
    /// returns `3 * x` — see the module docstring.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the optional sub-convs or
    /// from the manual `avg_pool1d` shim (via `avg_pool2d` over an
    /// inserted height-1 axis).
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Upstream's leading `x = x.transpose(1, 2)` is intentionally
        // skipped — our VocosBackbone returns `(B, C, T)` already.
        let (upmerge_res, repeat_res) = if self.upsample_scale > 1 {
            // Both skip and main path operate on the *original* `x`
            // (not the leaky-relu'd one — upstream's LeakyReLU lives
            // inside the Sequential and only affects `deconv_res`).
            let repeat_res = super::primitives::repeat_interleave_dim2(x, self.upsample_scale)?;
            let deconv = self.de_conv_upsampler.as_ref().ok_or_else(|| {
                candle_core::Error::Msg(
                    "SamplingBlock: de_conv_upsampler missing despite upsample_scale > 1".into(),
                )
            })?;
            let pre_act = candle_nn::ops::leaky_relu(x, LEAKY_SLOPE)?;
            let deconv_res = deconv.forward(&pre_act)?;
            // Upstream sums the deconv branch onto the
            // repeat_interleave skip *before* feeding the result into
            // the downsample stage. Shapes must match exactly; the
            // padding/output_padding upstream picked guarantees this
            // for every supported `upsample_scale`.
            let upmerge_res = (&repeat_res + &deconv_res)?;
            (upmerge_res, repeat_res)
        } else {
            (x.clone(), x.clone())
        };

        let (conv_res, skip2_res, skip1_res) = if self.downsample_scale > 1 {
            let conv = self.conv_downsampler.as_ref().ok_or_else(|| {
                candle_core::Error::Msg(
                    "SamplingBlock: conv_downsampler missing despite downsample_scale > 1".into(),
                )
            })?;
            let pre_act = candle_nn::ops::leaky_relu(&upmerge_res, LEAKY_SLOPE)?;
            let conv_res = conv.forward(&pre_act)?;
            let skip2_res = avg_pool1d_dim2(&upmerge_res, self.downsample_scale)?;
            let skip1_res = avg_pool1d_dim2(&repeat_res, self.downsample_scale)?;
            (conv_res, skip2_res, skip1_res)
        } else {
            (upmerge_res.clone(), upmerge_res, repeat_res)
        };

        // `conv_res + skip1_res + skip2_res`. When both scales are 1
        // this is `3 * x`, not `x` — see module docstring.
        ((&conv_res + &skip1_res)? + &skip2_res)?.contiguous()
    }
}

/// `F.avg_pool1d(x, kernel_size=k, stride=k)` shim — candle doesn't
/// expose a 1-D avg pool, so we insert a height-1 axis, run
/// `avg_pool2d_with_stride((1, k), (1, k))`, and squeeze.
///
/// Matches `torch.nn.functional.avg_pool1d` with kernel == stride and
/// no padding (which is exactly upstream's call site).
fn avg_pool1d_dim2(x: &Tensor, kernel: usize) -> Result<Tensor> {
    let (batch, channels, time) = x.dims3()?;
    // (B, C, T) -> (B, C, 1, T)
    let x4 = x.unsqueeze(2)?;
    // avg_pool2d_with_stride takes ((kh, kw), (sh, sw)).
    let pooled = x4.avg_pool2d_with_stride((1, kernel), (1, kernel))?;
    // (B, C, 1, T') -> (B, C, T')
    let time_out = (time - kernel) / kernel + 1;
    pooled.reshape((batch, channels, time_out))
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// One `(SamplingBlock, VocosBackbone)` pair from
/// `feat_encoder.py::Encoder.downsample`. Upstream wraps both in
/// `nn.Sequential`, so the state-dict keys nest under
/// `downsample.{stage_idx}.0.…` (the sampler) and
/// `downsample.{stage_idx}.1.…` (the per-stage `VocosBackbone`).
struct EncoderStage {
    sampler: SamplingBlock,
    backbone: VocosBackbone,
}

impl EncoderStage {
    /// Forward through sampler → 2-layer `VocosBackbone`.
    ///
    /// `x` is channels-first `(B, vocos_dim, T)`. Output is
    /// `(B, vocos_dim, T / ratio)`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.sampler.forward(x)?;
        // Per-stage VocosBackbones are loaded with `condition_dim =
        // None`, so we pass `None` for the condition tensor.
        self.backbone.forward(&x, None)
    }
}

/// `BiCodec` feature encoder: a [`VocosBackbone`] trunk followed by a
/// sequence of `(SamplingBlock, VocosBackbone)` stages and a final
/// `Linear` projection.
///
/// Mirrors upstream `sparktts/modules/encoder_decoder/feat_encoder.py::
/// Encoder`. Each downsample stage halves (or otherwise rescales) the
/// temporal length according to its `sample_ratios[i]`. The standard
/// Spark-TTS checkpoint uses `sample_ratios = [1, 1]` — both stages are
/// loaded but the samplers run in their "scales-are-one" path
/// (output = `3 * input`).
#[allow(
    clippy::struct_field_names,
    reason = "`encoder` matches upstream's `self.encoder` attribute name \
              (the trunk `VocosBackbone`) — renaming would break the \
              state-dict key path `encoder.…`."
)]
pub(super) struct Encoder {
    /// Initial trunk — upstream `self.encoder` with `vocos_num_layers`
    /// blocks and no conditioning.
    encoder: VocosBackbone,
    /// Per-`sample_ratios` downsample stages, each one being a pair of
    /// `(SamplingBlock, 2-layer VocosBackbone)`.
    stages: Vec<EncoderStage>,
    /// Final `Linear(vocos_dim, out_channels)` applied in channels-last
    /// layout before a transpose back to channels-first.
    project: Linear,
}

impl Encoder {
    /// Load the full encoder. State-dict key layout:
    ///
    /// * `encoder.…` — the trunk, see [`VocosBackbone::load`].
    /// * `downsample.{i}.0.…` — the `i`-th stage's sampler
    ///   (see [`SamplingBlock::load`]; possibly empty when
    ///   `sample_ratios[i] == 1`).
    /// * `downsample.{i}.1.…` — the `i`-th stage's 2-layer
    ///   `VocosBackbone`.
    /// * `project.{weight,bias}` — `Linear(vocos_dim, out_channels)`.
    ///
    /// Per-stage `VocosBackbone`s are loaded with
    /// `layer_scale_init_value = Some(1.0 / num_layers)` to match
    /// upstream's `vocos.py` default (`num_layers` is the per-stage
    /// count, not the trunk's `vocos_num_layers`). Same trunk default.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-load.
    pub(super) fn load(
        vb: VarBuilder,
        input_channels: usize,
        vocos_dim: usize,
        vocos_intermediate_dim: usize,
        vocos_num_layers: usize,
        out_channels: usize,
        sample_ratios: &[usize],
    ) -> Result<Self> {
        // Upstream hard-codes the per-stage VocosBackbone to 2 layers.
        const STAGE_LAYERS: usize = 2;

        // Trunk: `self.encoder` upstream, no conditioning.
        let trunk_lsi = if vocos_num_layers > 0 {
            #[allow(
                clippy::cast_precision_loss,
                reason = "num_layers is a small usize (<= ~64) — exact in f64."
            )]
            let lsi = 1.0_f64 / (vocos_num_layers as f64);
            Some(lsi)
        } else {
            None
        };
        let encoder = VocosBackbone::load(
            vb.pp("encoder"),
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            trunk_lsi,
            /* condition_dim */ None,
        )?;

        // Per-ratio downsample stages.
        #[allow(
            clippy::cast_precision_loss,
            reason = "STAGE_LAYERS is a const (2) — exact in f64."
        )]
        let stage_lsi = Some(1.0_f64 / (STAGE_LAYERS as f64));
        let downsample_vb = vb.pp("downsample");
        let mut stages = Vec::with_capacity(sample_ratios.len());
        for (i, &ratio) in sample_ratios.iter().enumerate() {
            let stage_vb = downsample_vb.pp(i.to_string());
            let sampler = SamplingBlock::load(
                stage_vb.pp("0"),
                vocos_dim,
                /* groups */ vocos_dim,
                /* upsample_scale */ 1,
                /* downsample_scale */ ratio,
            )?;
            let backbone = VocosBackbone::load(
                stage_vb.pp("1"),
                vocos_dim,
                vocos_dim,
                vocos_intermediate_dim,
                STAGE_LAYERS,
                stage_lsi,
                /* condition_dim */ None,
            )?;
            stages.push(EncoderStage { sampler, backbone });
        }

        let project = linear(vocos_dim, out_channels, vb.pp("project"))?;

        Ok(Self {
            encoder,
            stages,
            project,
        })
    }

    /// Forward `(B, input_channels, T) -> (B, out_channels, T / prod(sample_ratios))`.
    ///
    /// The downsample stages each divide `T` by their `sample_ratios[i]`
    /// when `ratio > 1`. When all ratios are 1 the trailing
    /// time-length matches the input length (modulo the `3 * x` factor
    /// inside the sampler — see [`SamplingBlock`]).
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the trunk, any stage, or
    /// the final projection / transpose.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.encoder.forward(x, None)?;
        for stage in &self.stages {
            h = stage.forward(&h)?;
        }
        // (B, dim, T) -> (B, T, dim) for the channel-axis Linear.
        let h_tc = h.transpose(1, 2)?.contiguous()?;
        let h_tc = self.project.forward(&h_tc)?;
        // (B, T, out_channels) -> (B, out_channels, T).
        h_tc.transpose(1, 2)?.contiguous()
    }
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

    fn vb_from(map: HashMap<String, Tensor>, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_tensors(map, DType::F32, dev)
    }

    /// Plain `LayerNorm(dim)` under `prefix`.
    fn put_plain_ln(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.weight"),
            Tensor::ones(dim, DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// Depthwise conv (`groups=dim`, kernel=7, padding=3) under `prefix`.
    fn put_dwconv(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        let w_vec: Vec<f32> = (0..(dim * 7)).map(|i| (i as f32) * 0.005).collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(w_vec, (dim, 1, 7), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    fn put_linear(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        dev: &Device,
    ) -> Result<()> {
        let w_vec: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| ((i as f32) * 0.011).sin() * 0.05)
            .collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(w_vec, (out_dim, in_dim), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// One full `ConvNeXt` block (no `AdaLN`, with gamma) under `prefix`.
    fn put_convnext_block(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        intermediate_dim: usize,
        gamma: Option<f32>,
        dev: &Device,
    ) -> Result<()> {
        put_dwconv(map, &format!("{prefix}.dwconv"), dim, dev)?;
        put_plain_ln(map, &format!("{prefix}.norm"), dim, dev)?;
        put_linear(
            map,
            &format!("{prefix}.pwconv1"),
            dim,
            intermediate_dim,
            dev,
        )?;
        put_linear(
            map,
            &format!("{prefix}.pwconv2"),
            intermediate_dim,
            dim,
            dev,
        )?;
        if let Some(g) = gamma {
            let gamma_vec = vec![g; dim];
            map.insert(
                format!("{prefix}.gamma"),
                Tensor::from_vec(gamma_vec, dim, dev)?,
            );
        }
        Ok(())
    }

    /// One full `VocosBackbone` (plain LN, no `AdaLN`) under `prefix`.
    fn put_vocos_backbone(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        dev: &Device,
    ) -> Result<()> {
        let embed_w_vec: Vec<f32> = (0..(dim * input_channels * 7))
            .map(|i| ((i as f32) * 0.0007).cos() * 0.02)
            .collect();
        map.insert(
            format!("{prefix}.embed.weight"),
            Tensor::from_vec(embed_w_vec, (dim, input_channels, 7), dev)?,
        );
        map.insert(
            format!("{prefix}.embed.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        put_plain_ln(map, &format!("{prefix}.norm"), dim, dev)?;
        let lsi_f32 = 1.0_f32 / (num_layers as f32);
        for i in 0..num_layers {
            put_convnext_block(
                map,
                &format!("{prefix}.convnext.{i}"),
                dim,
                intermediate_dim,
                Some(lsi_f32),
                dev,
            )?;
        }
        put_plain_ln(map, &format!("{prefix}.final_layer_norm"), dim, dev)?;
        Ok(())
    }

    #[test]
    fn sampling_block_scales_both_one_returns_three_x() -> Result<()> {
        // Upstream's forward returns `conv_res + skip1_res + skip2_res`,
        // which collapses to `3 * x` when both scales are 1 — NOT `x`.
        // This test pins that behaviour to catch regressions that
        // silently drop the multi-skip sum.
        let dev = Device::Cpu;
        let dim = 8;
        let map: HashMap<String, Tensor> = HashMap::new();
        let vb = vb_from(map, &dev);
        let block = SamplingBlock::load(
            vb, dim, /* groups */ dim, /* upsample_scale */ 1,
            /* downsample_scale */ 1,
        )?;
        // No child VarBuilder keys should be consumed — the empty map
        // suffices because both Optionals stay `None`.

        let n = 2 * dim * 12;
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 0.5).collect();
        let x = Tensor::from_vec(xs.clone(), (2, dim, 12), &dev)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims(), &[2, dim, 12]);

        let expected: Vec<f32> = xs.iter().map(|v| v * 3.0).collect();
        let got = y.flatten_all()?.to_vec1::<f32>()?;
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
        Ok(())
    }

    #[test]
    fn sampling_block_downsample_halves_length() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 4;
        let downsample_scale = 2_usize;
        // conv_downsampler is Conv1d(dim, dim, k=4, stride=2, p=1,
        // groups=dim). With groups=dim, weight shape is (dim, 1, 4).
        let kernel = 2 * downsample_scale;
        let w_vec: Vec<f32> = (0..(dim * kernel)).map(|i| (i as f32) * 0.01).collect();
        let mut map = HashMap::new();
        map.insert(
            "conv_downsampler.1.weight".to_owned(),
            Tensor::from_vec(w_vec, (dim, 1, kernel), &dev)?,
        );
        map.insert(
            "conv_downsampler.1.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        let vb = vb_from(map, &dev);
        let block = SamplingBlock::load(
            vb,
            dim,
            /* groups */ dim,
            /* upsample_scale */ 1,
            downsample_scale,
        )?;

        let n = 2 * dim * 16;
        let xs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
        let x = Tensor::from_vec(xs, (2, dim, 16), &dev)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims(), &[2, dim, 8]);
        Ok(())
    }

    #[test]
    fn sampling_block_upsample_doubles_length() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 4;
        let upsample_scale = 2_usize;
        // de_conv_upsampler.1 is ConvTranspose1d(dim, dim,
        //   k=4, stride=2, padding=1, output_padding=0, groups=dim).
        // ConvTranspose1d weight shape: (in, out/groups, k) =
        // (dim, 1, 4).
        let kernel = upsample_scale * 2;
        let w_vec: Vec<f32> = (0..(dim * kernel)).map(|i| (i as f32) * 0.01).collect();
        let mut map = HashMap::new();
        map.insert(
            "de_conv_upsampler.1.weight".to_owned(),
            Tensor::from_vec(w_vec, (dim, 1, kernel), &dev)?,
        );
        map.insert(
            "de_conv_upsampler.1.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        let vb = vb_from(map, &dev);
        let block = SamplingBlock::load(
            vb,
            dim,
            /* groups */ dim,
            upsample_scale,
            /* downsample_scale */ 1,
        )?;

        let n = 2 * dim * 8;
        let xs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos()).collect();
        let x = Tensor::from_vec(xs, (2, dim, 8), &dev)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims(), &[2, dim, 16]);
        Ok(())
    }

    /// Build a synthetic Encoder for the shape-tests below.
    fn build_encoder_for_test(
        dev: &Device,
        input_channels: usize,
        vocos_dim: usize,
        vocos_intermediate_dim: usize,
        vocos_num_layers: usize,
        out_channels: usize,
        sample_ratios: &[usize],
    ) -> Result<Encoder> {
        let mut map = HashMap::new();
        put_vocos_backbone(
            &mut map,
            "encoder",
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            dev,
        )?;
        for (i, &ratio) in sample_ratios.iter().enumerate() {
            if ratio > 1 {
                // conv_downsampler.1: Conv1d(dim, dim, k=2*ratio,
                //   stride=ratio, groups=dim) -> weight (dim, 1, 2*ratio).
                let k = 2 * ratio;
                let w_vec: Vec<f32> = (0..(vocos_dim * k)).map(|j| (j as f32) * 0.001).collect();
                map.insert(
                    format!("downsample.{i}.0.conv_downsampler.1.weight"),
                    Tensor::from_vec(w_vec, (vocos_dim, 1, k), dev)?,
                );
                map.insert(
                    format!("downsample.{i}.0.conv_downsampler.1.bias"),
                    Tensor::zeros(vocos_dim, DType::F32, dev)?,
                );
            }
            put_vocos_backbone(
                &mut map,
                &format!("downsample.{i}.1"),
                vocos_dim,
                vocos_dim,
                vocos_intermediate_dim,
                2,
                dev,
            )?;
        }
        put_linear(&mut map, "project", vocos_dim, out_channels, dev)?;

        let vb = vb_from(map, dev);
        Encoder::load(
            vb,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            sample_ratios,
        )
    }

    #[test]
    fn encoder_forward_shape_no_downsample() -> Result<()> {
        // Standard Spark-TTS config: sample_ratios = [1, 1] is a
        // length-preserving encoder (modulo the inner 3x sampler).
        // Use smaller dims than the production 1024/384/1152 so tests
        // stay fast — the *shape* contract is what we're pinning.
        let dev = Device::Cpu;
        let input_channels = 32;
        let vocos_dim = 16;
        let vocos_intermediate_dim = 24;
        let vocos_num_layers = 2;
        let out_channels = 16;
        let encoder = build_encoder_for_test(
            &dev,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            &[1, 1],
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 48), &dev)?;
        let y = encoder.forward(&x)?;
        assert_eq!(y.dims(), &[2, out_channels, 48]);
        Ok(())
    }

    #[test]
    fn encoder_forward_shape_with_2x2_downsample() -> Result<()> {
        // 2x2 downsample: 48 -> 24 -> 12.
        let dev = Device::Cpu;
        let input_channels = 32;
        let vocos_dim = 16;
        let vocos_intermediate_dim = 24;
        let vocos_num_layers = 2;
        let out_channels = 8;
        let encoder = build_encoder_for_test(
            &dev,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            &[2, 2],
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 48), &dev)?;
        let y = encoder.forward(&x)?;
        assert_eq!(y.dims(), &[2, out_channels, 12]);
        Ok(())
    }
}
