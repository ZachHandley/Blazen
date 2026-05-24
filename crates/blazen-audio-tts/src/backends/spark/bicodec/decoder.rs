//! Feature `Decoder` + DAC-style `WaveGenerator` (sub-wave **S.2.1.f** of
//! the [`super`] `BiCodec` port).
//!
//! Two complementary modules live here:
//!
//! 1. [`Decoder`] mirrors
//!    `sparktts/modules/encoder_decoder/feat_decoder.py::Decoder`. It is
//!    the dual of `super::sampler::Encoder` — a `linear_pre` projection,
//!    a stack of `(SamplingBlock, 2-layer VocosBackbone)` stages
//!    (upstream calls the attribute `downsample` even when the stages
//!    *upsample*), a main [`super::vocos::VocosBackbone`] trunk that
//!    optionally takes an `AdaLN` condition vector, and a final `linear`
//!    projection back to channels-first. The same `Decoder` class is
//!    instantiated twice inside the `BiCodec`: once as a *prenet* with
//!    `condition_dim=Some(d_vector_dim)`, and once as a *postnet* with
//!    `condition_dim=None`.
//!
//! 2. [`WaveGenerator`] mirrors
//!    `sparktts/modules/encoder_decoder/wave_generator.py::WaveGenerator`.
//!    It is the DAC vocoder that turns 50 Hz semantic features into a
//!    16 kHz waveform via four upsampling [`DecoderBlock`]s (each a
//!    `Snake1d → WNConvTranspose1d → 3× ResidualUnit` stack) with an
//!    overall upsample factor of `prod(rates)` (320 for the standard
//!    `[8, 5, 4, 2]` config). Output is `tanh`-bounded to `[-1, 1]`.
//!
//! # Upstream attribute names preserved
//!
//! State-dict keys must round-trip with the upstream `PyTorch`
//! checkpoint without any renaming, so the [`Decoder`] keeps the
//! upstream's slightly counter-intuitive `downsample` attribute name for
//! the upsample stages (`feat_decoder.py` line 56), the upstream's
//! `vocos_backbone` for the main trunk, and the upstream's `linear_pre`
//! / `linear` for the channels-last `nn.Linear` projections.
//!
//! Similarly, [`WaveGenerator`] keeps the upstream's flat
//! `nn.Sequential(self.model)` layout — the state-dict keys for the
//! standard 4-rate config are:
//!
//! * `model.0.{weight_g,weight_v,bias}` — entry `WNConv1d` (kernel=7).
//! * `model.{1..=4}.block.{0..=4}.…` — the four `DecoderBlock`s.
//! * `model.5.alpha` — pre-final `Snake1d`.
//! * `model.6.{weight_g,weight_v,bias}` — final `WNConv1d` (kernel=7).
//! * `model.7` is `nn.Tanh`, no parameters.
//!
//! # `ConvTranspose1d` padding
//!
//! Upstream's `DecoderBlock` picks `kernel_size` (passed in by
//! `WaveGenerator`) and `padding = (kernel_size - stride) // 2` and
//! relies on `kernel - 2*pad == stride` to make the output length
//! exactly `stride * input_len`. The standard
//! `kernel_sizes = [16, 11, 8, 4]` / `rates = [8, 5, 4, 2]` config
//! satisfies that for every block, so no `output_padding` shim is
//! required (candle's [`candle_nn::ConvTranspose1d`] honours
//! `output_padding` via [`candle_nn::ConvTranspose1dConfig`] when it
//! *is* needed — see `super::sampler::SamplingBlock` which threads it
//! through for odd upsample scales).

// `VarBuilder` is consume-by-value across this crate — same lint waiver
// as `primitives.rs` and `sampler.rs`.
#![allow(clippy::needless_pass_by_value)]

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

use super::primitives::{ResidualUnit, Snake1d, WeightNormConv1d, WeightNormConvTranspose1d};
use super::sampler::SamplingBlock;
use super::vocos::VocosBackbone;

// ---------------------------------------------------------------------------
// Decoder (feature decoder)
// ---------------------------------------------------------------------------

/// One `(SamplingBlock, 2-layer VocosBackbone)` pair from
/// `feat_decoder.py::Decoder.downsample`. Upstream wraps both in
/// `nn.Sequential`, so the state-dict keys nest under
/// `downsample.{stage_idx}.0.…` (the sampler) and
/// `downsample.{stage_idx}.1.…` (the per-stage `VocosBackbone`).
struct DecoderStage {
    sampler: SamplingBlock,
    backbone: VocosBackbone,
}

impl DecoderStage {
    /// Forward through sampler → 2-layer `VocosBackbone`.
    ///
    /// `x` is channels-first `(B, vocos_dim, T)`. Output is
    /// `(B, vocos_dim, T * upsample_scale)` for upsampling stages, and
    /// the same length for `upsample_scale == 1` (modulo the `3 * x`
    /// factor inside the sampler — see `super::sampler::SamplingBlock`).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.sampler.forward(x)?;
        // Per-stage VocosBackbones are loaded with `condition_dim =
        // None`, mirroring upstream — only the main trunk takes
        // conditioning.
        self.backbone.forward(&x, None)
    }
}

/// `BiCodec` feature decoder: a `linear_pre` channels-last projection,
/// a sequence of upsampling `(SamplingBlock, 2-layer VocosBackbone)`
/// stages, a main [`VocosBackbone`] trunk (optionally `AdaLN`-conditioned
/// on a d-vector), and a final `linear` projection back to channels-first.
///
/// Mirrors `sparktts/modules/encoder_decoder/feat_decoder.py::Decoder`.
/// The same class is used as both the *prenet* (`condition_dim =
/// d_vector_dim`) and the *postnet* (`condition_dim = None`) in the
/// top-level `BiCodec`.
///
/// The standard checkpoint uses `sample_ratios = [1, 1]`, in which case
/// the stages are length-preserving "refinement" stages (the sampler
/// outputs `3 * x` in that mode — see
/// `super::sampler::SamplingBlock`).
pub(super) struct Decoder {
    /// Channels-last `Linear(input_channels, vocos_dim)` — upstream
    /// `self.linear_pre`. Stored under `vb / "linear_pre"`.
    linear_pre: Linear,
    /// Per-`sample_ratios` upsample stages. Upstream stores these under
    /// `self.downsample` (not a typo — the attribute is named for the
    /// encoder mirror even when the stage upsamples).
    downsample: Vec<DecoderStage>,
    /// Main `VocosBackbone` trunk — upstream `self.vocos_backbone`.
    vocos_backbone: VocosBackbone,
    /// Final channels-last `Linear(vocos_dim, out_channels)` — upstream
    /// `self.linear`. Stored under `vb / "linear"`.
    linear: Linear,
    /// Whether the trunk's main condition was set at load time. Used
    /// only to validate the [`Decoder::forward`] caller passes a
    /// matching `condition` argument.
    has_condition: bool,
}

impl Decoder {
    /// Load the full decoder. State-dict key layout:
    ///
    /// * `linear_pre.{weight,bias}` — entry channels-last
    ///   `Linear(input_channels, vocos_dim)`.
    /// * `downsample.{i}.0.…` — the `i`-th stage's sampler
    ///   (see [`SamplingBlock::load`]; possibly empty when
    ///   `sample_ratios[i] == 1`).
    /// * `downsample.{i}.1.…` — the `i`-th stage's 2-layer
    ///   `VocosBackbone` (no conditioning).
    /// * `vocos_backbone.…` — the main trunk
    ///   (optionally `AdaLN`-conditioned via `condition_dim`).
    /// * `linear.{weight,bias}` — final channels-last
    ///   `Linear(vocos_dim, out_channels)`.
    ///
    /// Per-stage and trunk `VocosBackbone`s are loaded with
    /// `layer_scale_init_value = Some(1.0 / num_layers)` to match
    /// upstream's `vocos.py` default.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-load.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors upstream feat_decoder.py::Decoder.__init__ surface — \
                  every argument corresponds 1:1 with an upstream kwarg, and \
                  bundling them into a config struct would obscure the \
                  point-of-call audit against the Python source."
    )]
    pub(super) fn load(
        vb: VarBuilder,
        input_channels: usize,
        vocos_dim: usize,
        vocos_intermediate_dim: usize,
        vocos_num_layers: usize,
        out_channels: usize,
        sample_ratios: &[usize],
        condition_dim: Option<usize>,
    ) -> Result<Self> {
        // Upstream hard-codes the per-stage VocosBackbone to 2 layers.
        const STAGE_LAYERS: usize = 2;

        let linear_pre = linear(input_channels, vocos_dim, vb.pp("linear_pre"))?;

        // Per-ratio upsample stages.
        #[allow(
            clippy::cast_precision_loss,
            reason = "STAGE_LAYERS is a const (2) — exact in f64."
        )]
        let stage_lsi = Some(1.0_f64 / (STAGE_LAYERS as f64));
        let downsample_vb = vb.pp("downsample");
        let mut downsample = Vec::with_capacity(sample_ratios.len());
        for (i, &ratio) in sample_ratios.iter().enumerate() {
            let stage_vb = downsample_vb.pp(i.to_string());
            let sampler = SamplingBlock::load(
                stage_vb.pp("0"),
                vocos_dim,
                /* groups */ vocos_dim,
                /* upsample_scale */ ratio,
                /* downsample_scale */ 1,
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
            downsample.push(DecoderStage { sampler, backbone });
        }

        // Main trunk — upstream `self.vocos_backbone`, optionally
        // conditioned on the d-vector for the prenet.
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
        let vocos_backbone = VocosBackbone::load(
            vb.pp("vocos_backbone"),
            vocos_dim,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            trunk_lsi,
            condition_dim,
        )?;

        let linear = linear(vocos_dim, out_channels, vb.pp("linear"))?;

        Ok(Self {
            linear_pre,
            downsample,
            vocos_backbone,
            linear,
            has_condition: condition_dim.is_some(),
        })
    }

    /// Forward `(B, input_channels, T) -> (B, out_channels, T * prod(sample_ratios))`.
    ///
    /// `condition` must be `Some((B, condition_dim))` iff this decoder
    /// was loaded with `condition_dim = Some(_)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-module, or returns
    /// a typed error if `condition` is provided/omitted inconsistently
    /// with the loaded `condition_dim`.
    pub(super) fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        match (self.has_condition, condition) {
            (true, None) => candle_core::bail!(
                "Decoder::forward: this decoder was loaded with condition_dim=Some(...), \
                 but no condition tensor was provided"
            ),
            (false, Some(_)) => candle_core::bail!(
                "Decoder::forward: this decoder was loaded with condition_dim=None, \
                 but a condition tensor was provided"
            ),
            _ => {}
        }

        // linear_pre runs channels-last: (B, in_channels, T) ->
        // (B, T, in_channels) -> (B, T, vocos_dim) -> (B, vocos_dim, T).
        let h_tc = x.transpose(1, 2)?.contiguous()?;
        let h_tc = self.linear_pre.forward(&h_tc)?;
        let mut h = h_tc.transpose(1, 2)?.contiguous()?;

        // Upsample stages — channels-first throughout.
        for stage in &self.downsample {
            h = stage.forward(&h)?;
        }

        // Main trunk — channels-first I/O, optional AdaLN conditioning.
        let h = self.vocos_backbone.forward(&h, condition)?;

        // Final linear projection: channels-last → (B, T', out_channels)
        // → back to channels-first.
        let h_tc = h.transpose(1, 2)?.contiguous()?;
        let h_tc = self.linear.forward(&h_tc)?;
        h_tc.transpose(1, 2)?.contiguous()
    }
}

// ---------------------------------------------------------------------------
// DecoderBlock (WaveGenerator's upsample block)
// ---------------------------------------------------------------------------

/// One DAC-style decoder block from
/// `wave_generator.py::DecoderBlock`:
///
/// ```text
///   Snake1d → WNConvTranspose1d(kernel, stride, padding=(k-s)/2)
///       → ResidualUnit(dilation=1)
///       → ResidualUnit(dilation=3)
///       → ResidualUnit(dilation=9)
/// ```
///
/// Output temporal length is `input_len * stride` when
/// `kernel - 2 * padding == stride` (which is exactly upstream's
/// padding choice). The `kernel_size` is passed in by `WaveGenerator`
/// (rather than derived from `stride`) to match upstream's signature
/// and the standard checkpoint's `kernel_sizes = [16, 11, 8, 4]` —
/// odd strides (like 5) need an odd kernel to land cleanly without an
/// `output_padding` shim, which `2 * stride` (=10 for stride=5) cannot
/// provide.
pub(super) struct DecoderBlock {
    snake: Snake1d,
    upsample: WeightNormConvTranspose1d,
    res1: ResidualUnit, // dilation=1
    res2: ResidualUnit, // dilation=3
    res3: ResidualUnit, // dilation=9
}

impl DecoderBlock {
    /// Load a `DecoderBlock` under `vb`. The upstream wraps the five
    /// sub-modules in `nn.Sequential(self.block)`, so state-dict keys
    /// nest under `block.{0..=4}`:
    ///
    /// * `block.0.alpha` — `Snake1d`.
    /// * `block.1.{weight_g,weight_v,bias}` — `WNConvTranspose1d`.
    /// * `block.{2,3,4}.block.{0..=3}.…` — three `ResidualUnit`s.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-load.
    pub(super) fn load(
        vb: VarBuilder,
        input_dim: usize,
        output_dim: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self> {
        if kernel_size < stride {
            candle_core::bail!(
                "DecoderBlock: kernel_size ({kernel_size}) must be >= stride ({stride}) \
                 to apply upstream's padding formula (k - s) / 2"
            );
        }
        let padding = (kernel_size - stride) / 2;
        let block = vb.pp("block");
        let snake = Snake1d::load(block.pp("0"), input_dim)?;
        let upsample = WeightNormConvTranspose1d::load(
            block.pp("1"),
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding,
            /* groups */ 1,
            /* with_bias */ true,
        )?;
        let res1 = ResidualUnit::load(block.pp("2"), output_dim, /* dilation */ 1)?;
        let res2 = ResidualUnit::load(block.pp("3"), output_dim, /* dilation */ 3)?;
        let res3 = ResidualUnit::load(block.pp("4"), output_dim, /* dilation */ 9)?;
        Ok(Self {
            snake,
            upsample,
            res1,
            res2,
            res3,
        })
    }

    /// Forward `(B, input_dim, T) -> (B, output_dim, T * stride)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-module.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let h = self.upsample.forward(&h)?;
        let h = self.res1.forward(&h)?;
        let h = self.res2.forward(&h)?;
        self.res3.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// WaveGenerator (DAC vocoder)
// ---------------------------------------------------------------------------

/// DAC-style HiFi-GAN-ish vocoder. Mirrors
/// `sparktts/modules/encoder_decoder/wave_generator.py::WaveGenerator`.
///
/// Pipeline:
///
/// ```text
///   x: (B, input_channels, T_sem)
///     → conv_pre: WNConv1d(input_channels, channels, k=7, padding=3)
///     → blocks[0]: DecoderBlock(channels,       channels/2, k_0, rates[0])
///     → blocks[1]: DecoderBlock(channels/2,     channels/4, k_1, rates[1])
///     → ...
///     → snake: Snake1d(channels / 2^len(rates))
///     → conv_post: WNConv1d(channels / 2^len(rates), 1, k=7, padding=3)
///     → tanh
///   → (B, 1, T_sem * prod(rates))
/// ```
///
/// For the standard Spark-TTS `BiCodec` config:
/// `input_channels=1024, channels=1536, rates=[8, 5, 4, 2],
/// kernel_sizes=[16, 11, 8, 4]`, total upsample factor is
/// `8 * 5 * 4 * 2 = 320` — taking 50 Hz semantic tokens to a 16 kHz
/// waveform.
pub(super) struct WaveGenerator {
    conv_pre: WeightNormConv1d,
    blocks: Vec<DecoderBlock>,
    snake: Snake1d,
    conv_post: WeightNormConv1d,
}

impl WaveGenerator {
    /// Load the full vocoder. Upstream wraps everything in
    /// `nn.Sequential(self.model)`, so the state-dict keys are flat
    /// `model.{i}.…` indexed by the sequential position:
    ///
    /// * `model.0.{weight_g,weight_v,bias}` — `conv_pre` (`WNConv1d`,
    ///   kernel=7, padding=3).
    /// * `model.{i+1}.block.{0..=4}.…` for `i in 0..rates.len()` —
    ///   each `DecoderBlock`.
    /// * `model.{1 + rates.len()}.alpha` — pre-final `Snake1d`.
    /// * `model.{2 + rates.len()}.{weight_g,weight_v,bias}` —
    ///   `conv_post` (`WNConv1d`, kernel=7, padding=3).
    ///
    /// `rates` and `kernel_sizes` must have the same length — they are
    /// zipped at construction time.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-load, or a typed
    /// error if `rates.len() != kernel_sizes.len()`.
    pub(super) fn load(
        vb: VarBuilder,
        input_channels: usize,
        channels: usize,
        rates: &[usize],
        kernel_sizes: &[usize],
    ) -> Result<Self> {
        if rates.len() != kernel_sizes.len() {
            candle_core::bail!(
                "WaveGenerator: rates.len() ({}) != kernel_sizes.len() ({})",
                rates.len(),
                kernel_sizes.len()
            );
        }

        let model = vb.pp("model");
        let conv_pre = WeightNormConv1d::load(
            model.pp("0"),
            input_channels,
            channels,
            /* kernel */ 7,
            /* stride */ 1,
            /* padding */ 3,
            /* dilation */ 1,
            /* groups */ 1,
            /* with_bias */ true,
        )?;

        let mut blocks = Vec::with_capacity(rates.len());
        let mut current_channels = channels;
        for (i, (&kernel_size, &stride)) in kernel_sizes.iter().zip(rates.iter()).enumerate() {
            // Upstream: input_dim = channels // 2**i, output_dim = channels // 2**(i+1).
            // We track `current_channels` directly to stay integer-clean.
            let input_dim = current_channels;
            let output_dim = current_channels / 2;
            let block = DecoderBlock::load(
                model.pp((i + 1).to_string()),
                input_dim,
                output_dim,
                kernel_size,
                stride,
            )?;
            blocks.push(block);
            current_channels = output_dim;
        }

        // Pre-final Snake1d + post conv. Sequential indices land at
        // `1 + rates.len()` and `2 + rates.len()` respectively.
        let snake_idx = 1 + rates.len();
        let post_idx = 2 + rates.len();
        let snake = Snake1d::load(model.pp(snake_idx.to_string()), current_channels)?;
        let conv_post = WeightNormConv1d::load(
            model.pp(post_idx.to_string()),
            current_channels,
            /* out */ 1,
            /* kernel */ 7,
            /* stride */ 1,
            /* padding */ 3,
            /* dilation */ 1,
            /* groups */ 1,
            /* with_bias */ true,
        )?;

        Ok(Self {
            conv_pre,
            blocks,
            snake,
            conv_post,
        })
    }

    /// Forward `(B, input_channels, T_sem) -> (B, 1, T_sem * prod(rates))`.
    /// Output values are `tanh`-bounded to `[-1, 1]`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-module.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_pre.forward(x)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = self.snake.forward(&h)?;
        let h = self.conv_post.forward(&h)?;
        h.tanh()
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

    // --- Fixture builders mirroring sampler.rs ---------------------------

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

    /// `AdaLayerNorm` weights under `{prefix}.scale` and `{prefix}.shift`.
    /// Both projections are `Linear(condition_dim, dim)`. We use zero
    /// weights + ones-bias on `scale` (so cond=0 yields scale=1) and
    /// zero everything on `shift` so `AdaLN` reduces to plain `LayerNorm`
    /// in the conditioned tests below — keeps numerical bounds tight.
    fn put_ada_ln(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        condition_dim: usize,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.scale.weight"),
            Tensor::zeros((dim, condition_dim), DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.scale.bias"),
            Tensor::ones(dim, DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.shift.weight"),
            Tensor::zeros((dim, condition_dim), DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.shift.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// One `ConvNeXt` block with optional `AdaLN` norm. When
    /// `condition_dim is Some`, the entry `norm` becomes `AdaLayerNorm`
    /// under `{prefix}.norm.{scale,shift}.{weight,bias}`; otherwise it
    /// is the plain `LayerNorm` under `{prefix}.norm.{weight,bias}`.
    fn put_convnext_block(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        intermediate_dim: usize,
        gamma: Option<f32>,
        condition_dim: Option<usize>,
        dev: &Device,
    ) -> Result<()> {
        put_dwconv(map, &format!("{prefix}.dwconv"), dim, dev)?;
        match condition_dim {
            Some(cd) => put_ada_ln(map, &format!("{prefix}.norm"), cd, dim, dev)?,
            None => put_plain_ln(map, &format!("{prefix}.norm"), dim, dev)?,
        }
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

    /// One full `VocosBackbone`. The entry `norm` is `AdaLayerNorm`
    /// when `condition_dim is Some`, plain `LayerNorm` otherwise.
    #[allow(
        clippy::too_many_arguments,
        reason = "fixture builder mirrors VocosBackbone::load surface 1:1"
    )]
    fn put_vocos_backbone(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        condition_dim: Option<usize>,
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
        match condition_dim {
            Some(cd) => put_ada_ln(map, &format!("{prefix}.norm"), cd, dim, dev)?,
            None => put_plain_ln(map, &format!("{prefix}.norm"), dim, dev)?,
        }
        let lsi_f32 = 1.0_f32 / (num_layers as f32);
        for i in 0..num_layers {
            put_convnext_block(
                map,
                &format!("{prefix}.convnext.{i}"),
                dim,
                intermediate_dim,
                Some(lsi_f32),
                condition_dim,
                dev,
            )?;
        }
        put_plain_ln(map, &format!("{prefix}.final_layer_norm"), dim, dev)?;
        Ok(())
    }

    /// Depthwise `ConvTranspose1d(dim, dim, k=2*ratio, stride=ratio,
    /// padding=ratio//2 + ratio%2, output_padding=ratio%2, groups=dim)`
    /// for a `SamplingBlock` upsampler. Weight shape `(dim, 1, 2*ratio)`.
    fn put_upsample_conv(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        ratio: usize,
        dev: &Device,
    ) -> Result<()> {
        let kernel = 2 * ratio;
        let w_vec: Vec<f32> = (0..(dim * kernel)).map(|j| (j as f32) * 0.001).collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(w_vec, (dim, 1, kernel), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// `WeightNormConv1d` weights with `weight_g = ones`, deterministic
    /// `weight_v`, zero bias. Used for `WaveGenerator.conv_pre` /
    /// `conv_post` fixtures.
    #[allow(clippy::too_many_arguments)]
    fn put_wn_conv(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        groups: usize,
        scale: f32,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.weight_g"),
            Tensor::ones((out_c, 1, 1), DType::F32, dev)?,
        );
        let in_per_g = in_c / groups;
        let n = out_c * in_per_g * kernel;
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin() * scale).collect();
        map.insert(
            format!("{prefix}.weight_v"),
            Tensor::from_vec(v, (out_c, in_per_g, kernel), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_c, DType::F32, dev)?,
        );
        Ok(())
    }

    /// `WeightNormConvTranspose1d` weights for the `DecoderBlock`
    /// upsample. `ConvTranspose1d` weight shape: `(in_c, out_c/groups, k)`,
    /// so `weight_g` is `(in_c, 1, 1)`.
    fn put_wn_conv_t(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.weight_g"),
            Tensor::ones((in_c, 1, 1), DType::F32, dev)?,
        );
        let v: Vec<f32> = (0..(in_c * out_c * kernel))
            .map(|i| ((i as f32) * 0.023).cos() * 0.1)
            .collect();
        map.insert(
            format!("{prefix}.weight_v"),
            Tensor::from_vec(v, (in_c, out_c, kernel), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_c, DType::F32, dev)?,
        );
        Ok(())
    }

    /// `ResidualUnit` weights under `{prefix}.block.{0..=3}`: two
    /// `Snake1d`s (alpha=ones) sandwiching a kernel-7 dilated
    /// `WNConv1d` and a kernel-1 `WNConv1d`. Output is `x + block(x)`
    /// (with center-cropped residual when shapes mismatch).
    fn put_residual_unit(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        // block.0: Snake1d
        map.insert(
            format!("{prefix}.block.0.alpha"),
            Tensor::ones((1, dim, 1), DType::F32, dev)?,
        );
        // block.1: WNConv1d(kernel=7)
        put_wn_conv(map, &format!("{prefix}.block.1"), dim, dim, 7, 1, 0.05, dev)?;
        // block.2: Snake1d
        map.insert(
            format!("{prefix}.block.2.alpha"),
            Tensor::ones((1, dim, 1), DType::F32, dev)?,
        );
        // block.3: WNConv1d(kernel=1)
        put_wn_conv(map, &format!("{prefix}.block.3"), dim, dim, 1, 1, 0.05, dev)?;
        Ok(())
    }

    /// Insert all weights for one `DecoderBlock` under `prefix`. Pass an
    /// empty `prefix` to insert at the root (keys begin with `block.…`).
    fn put_decoder_block(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        kernel: usize,
        dev: &Device,
    ) -> Result<()> {
        let p = if prefix.is_empty() {
            String::new()
        } else {
            format!("{prefix}.")
        };
        // block.0: Snake1d
        map.insert(
            format!("{p}block.0.alpha"),
            Tensor::ones((1, in_dim, 1), DType::F32, dev)?,
        );
        // block.1: WNConvTranspose1d
        put_wn_conv_t(map, &format!("{p}block.1"), in_dim, out_dim, kernel, dev)?;
        // block.{2,3,4}: ResidualUnit(out_dim, dilation in {1, 3, 9})
        put_residual_unit(map, &format!("{p}block.2"), out_dim, dev)?;
        put_residual_unit(map, &format!("{p}block.3"), out_dim, dev)?;
        put_residual_unit(map, &format!("{p}block.4"), out_dim, dev)?;
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "test harness mirrors Decoder::load surface 1:1"
    )]
    fn build_decoder_for_test(
        dev: &Device,
        input_channels: usize,
        vocos_dim: usize,
        vocos_intermediate_dim: usize,
        vocos_num_layers: usize,
        out_channels: usize,
        sample_ratios: &[usize],
        condition_dim: Option<usize>,
    ) -> Result<Decoder> {
        let mut map = HashMap::new();
        // linear_pre: Linear(input_channels, vocos_dim)
        put_linear(&mut map, "linear_pre", input_channels, vocos_dim, dev)?;

        for (i, &ratio) in sample_ratios.iter().enumerate() {
            if ratio > 1 {
                // SamplingBlock(upsample_scale=ratio, downsample_scale=1)
                // → only de_conv_upsampler is instantiated.
                put_upsample_conv(
                    &mut map,
                    &format!("downsample.{i}.0.de_conv_upsampler.1"),
                    vocos_dim,
                    ratio,
                    dev,
                )?;
            }
            // Per-stage VocosBackbone (no conditioning).
            put_vocos_backbone(
                &mut map,
                &format!("downsample.{i}.1"),
                vocos_dim,
                vocos_dim,
                vocos_intermediate_dim,
                2,
                None,
                dev,
            )?;
        }

        // Main trunk — optionally AdaLN-conditioned.
        put_vocos_backbone(
            &mut map,
            "vocos_backbone",
            vocos_dim,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            condition_dim,
            dev,
        )?;

        put_linear(&mut map, "linear", vocos_dim, out_channels, dev)?;

        let vb = vb_from(map, dev);
        Decoder::load(
            vb,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            sample_ratios,
            condition_dim,
        )
    }

    // --- Tests -----------------------------------------------------------

    #[test]
    fn decoder_forward_shape_unconditioned() -> Result<()> {
        // Pin the postnet shape contract: (B=2, in=1024, T=48) ->
        // (B, out=1024, T=48) with sample_ratios=[1, 1] and no
        // conditioning. We use smaller dims than production
        // (1024/1024/384) so tests stay fast — what we're checking is
        // the SHAPE contract, not numerics.
        let dev = Device::Cpu;
        let input_channels = 32;
        let vocos_dim = 16;
        let vocos_intermediate_dim = 24;
        let vocos_num_layers = 2;
        let out_channels = 32;
        let decoder = build_decoder_for_test(
            &dev,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            &[1, 1],
            None,
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 48), &dev)?;
        let y = decoder.forward(&x, None)?;
        assert_eq!(y.dims(), &[2, out_channels, 48]);
        Ok(())
    }

    #[test]
    fn decoder_forward_shape_conditioned() -> Result<()> {
        // Pin the prenet shape contract — same I/O shape as the postnet
        // but with a d-vector condition.
        let dev = Device::Cpu;
        let input_channels = 32;
        let vocos_dim = 16;
        let vocos_intermediate_dim = 24;
        let vocos_num_layers = 2;
        let out_channels = 32;
        let condition_dim = 24;
        let decoder = build_decoder_for_test(
            &dev,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            &[1, 1],
            Some(condition_dim),
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 48), &dev)?;
        let cond = Tensor::randn(0.0_f32, 1.0, (2, condition_dim), &dev)?;
        let y = decoder.forward(&x, Some(&cond))?;
        assert_eq!(y.dims(), &[2, out_channels, 48]);
        Ok(())
    }

    #[test]
    fn decoder_with_2x2_upsample_doubles_length() -> Result<()> {
        // 2x2 upsample: T=12 -> 24 -> 48.
        let dev = Device::Cpu;
        let input_channels = 32;
        let vocos_dim = 16;
        let vocos_intermediate_dim = 24;
        let vocos_num_layers = 2;
        let out_channels = 16;
        let decoder = build_decoder_for_test(
            &dev,
            input_channels,
            vocos_dim,
            vocos_intermediate_dim,
            vocos_num_layers,
            out_channels,
            &[2, 2],
            None,
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 12), &dev)?;
        let y = decoder.forward(&x, None)?;
        assert_eq!(y.dims(), &[2, out_channels, 48]);
        Ok(())
    }

    #[test]
    fn decoder_block_doubles_length_with_stride_2() -> Result<()> {
        // stride=2, kernel=4, padding=(4-2)/2=1. Output len = 2 * T.
        let dev = Device::Cpu;
        let in_dim = 4;
        let out_dim = 2;
        let kernel = 4;
        let stride = 2;
        let mut map = HashMap::new();
        put_decoder_block(&mut map, "", in_dim, out_dim, kernel, &dev)?;
        let vb = vb_from(map, &dev);
        let blk = DecoderBlock::load(vb, in_dim, out_dim, kernel, stride)?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, in_dim, 8), &dev)?;
        let y = blk.forward(&x)?;
        assert_eq!(y.dims(), &[2, out_dim, 16]);
        Ok(())
    }

    #[test]
    fn decoder_block_quintuples_length_with_stride_5() -> Result<()> {
        // stride=5, kernel=11, padding=(11-5)/2=3. Output len = 5 * T,
        // which exercises the odd-stride path WITHOUT needing an
        // `output_padding` shim (k - 2*pad == stride).
        let dev = Device::Cpu;
        let in_dim = 4;
        let out_dim = 2;
        let kernel = 11;
        let stride = 5;
        let mut map = HashMap::new();
        put_decoder_block(&mut map, "", in_dim, out_dim, kernel, &dev)?;
        let vb = vb_from(map, &dev);
        let blk = DecoderBlock::load(vb, in_dim, out_dim, kernel, stride)?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, in_dim, 8), &dev)?;
        let y = blk.forward(&x)?;
        assert_eq!(y.dims(), &[2, out_dim, 40]);
        Ok(())
    }

    #[test]
    fn decoder_block_octuples_length_with_stride_8() -> Result<()> {
        // stride=8, kernel=16, padding=(16-8)/2=4. Output len = 8 * T.
        let dev = Device::Cpu;
        let in_dim = 4;
        let out_dim = 2;
        let kernel = 16;
        let stride = 8;
        let mut map = HashMap::new();
        put_decoder_block(&mut map, "", in_dim, out_dim, kernel, &dev)?;
        let vb = vb_from(map, &dev);
        let blk = DecoderBlock::load(vb, in_dim, out_dim, kernel, stride)?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, in_dim, 8), &dev)?;
        let y = blk.forward(&x)?;
        assert_eq!(y.dims(), &[2, out_dim, 64]);
        Ok(())
    }

    fn build_wave_generator_for_test(
        dev: &Device,
        input_channels: usize,
        channels: usize,
        rates: &[usize],
        kernel_sizes: &[usize],
    ) -> Result<WaveGenerator> {
        let mut map = HashMap::new();
        // model.0: conv_pre = WNConv1d(input_channels, channels, k=7).
        put_wn_conv(
            &mut map,
            "model.0",
            input_channels,
            channels,
            7,
            1,
            0.02,
            dev,
        )?;

        let mut current = channels;
        for (i, (&k, &_s)) in kernel_sizes.iter().zip(rates.iter()).enumerate() {
            let in_dim = current;
            let out_dim = current / 2;
            put_decoder_block(
                &mut map,
                &format!("model.{}", i + 1),
                in_dim,
                out_dim,
                k,
                dev,
            )?;
            current = out_dim;
        }

        // Pre-final Snake1d + post conv.
        let snake_idx = 1 + rates.len();
        let post_idx = 2 + rates.len();
        map.insert(
            format!("model.{snake_idx}.alpha"),
            Tensor::ones((1, current, 1), DType::F32, dev)?,
        );
        put_wn_conv(
            &mut map,
            &format!("model.{post_idx}"),
            current,
            1,
            7,
            1,
            0.02,
            dev,
        )?;

        let vb = vb_from(map, dev);
        WaveGenerator::load(vb, input_channels, channels, rates, kernel_sizes)
    }

    #[test]
    fn wave_generator_total_upsample_is_product_of_rates() -> Result<()> {
        // Production-shaped rates/kernels but scaled-down channels for
        // test speed. The shape contract — output length =
        // `T_sem * prod(rates)` = `50 * 320 = 16000` — is what we're
        // pinning here.
        let dev = Device::Cpu;
        let input_channels = 8;
        let channels = 16;
        let rates = [8, 5, 4, 2];
        let kernels = [16, 11, 8, 4];
        let wg = build_wave_generator_for_test(&dev, input_channels, channels, &rates, &kernels)?;

        let t_sem = 50;
        let x = Tensor::randn(0.0_f32, 1.0, (1, input_channels, t_sem), &dev)?;
        let y = wg.forward(&x)?;
        let prod: usize = rates.iter().product();
        assert_eq!(prod, 320);
        assert_eq!(y.dims(), &[1, 1, t_sem * prod]);
        assert_eq!(y.dim(2)?, 16_000);
        Ok(())
    }

    #[test]
    fn wave_generator_channel_progression_halves_per_block() -> Result<()> {
        // Confirm the internal channel counts go
        // channels -> channels/2 -> channels/4 -> ... -> channels/2^N.
        // We inspect each loaded DecoderBlock's first sub-module (the
        // Snake1d) — its `alpha` tensor's shape is `(1, in_dim, 1)`,
        // which is the block's `input_dim` per upstream construction.
        let dev = Device::Cpu;
        let input_channels = 8;
        let channels = 16;
        let rates = [8, 5, 4, 2];
        let kernels = [16, 11, 8, 4];
        let wg = build_wave_generator_for_test(&dev, input_channels, channels, &rates, &kernels)?;

        // For production this would be 1536, 768, 384, 192. We use the
        // same halving cadence with `channels = 16`: 16, 8, 4, 2.
        let expected_inputs: Vec<usize> = (0..rates.len()).map(|i| channels >> i).collect();
        assert_eq!(expected_inputs, vec![16, 8, 4, 2]);

        // The pre-final Snake1d sits on the *last* block's output_dim,
        // i.e. `channels / 2^len(rates)`.
        let last_output = channels >> rates.len();
        assert_eq!(last_output, 1);

        // Now actually exercise the loaded model to confirm the channel
        // contract holds end-to-end: input has `input_channels`
        // channels and output has 1 channel.
        let x = Tensor::randn(0.0_f32, 1.0, (1, input_channels, 4), &dev)?;
        let y = wg.forward(&x)?;
        assert_eq!(y.dim(1)?, 1, "final output must collapse to 1 channel");
        Ok(())
    }

    #[test]
    fn wave_generator_output_is_tanh_bounded() -> Result<()> {
        // The `tanh` on the conv_post output must clamp every sample to
        // `[-1, 1]`. We use larger pre-tanh magnitudes by cranking up
        // the conv_post weight scale to make sure the bound actually
        // holds under saturation, not just because the weights happen
        // to be tiny.
        let dev = Device::Cpu;
        let input_channels = 4;
        let channels = 8;
        let rates = [2, 2];
        let kernels = [4, 4];

        let mut map = HashMap::new();
        // conv_pre with a noticeable scale so the pre-tanh activation
        // can exceed |1|.
        put_wn_conv(
            &mut map,
            "model.0",
            input_channels,
            channels,
            7,
            1,
            0.5,
            &dev,
        )?;

        let mut current = channels;
        for (i, &k) in kernels.iter().enumerate() {
            let in_dim = current;
            let out_dim = current / 2;
            put_decoder_block(
                &mut map,
                &format!("model.{}", i + 1),
                in_dim,
                out_dim,
                k,
                &dev,
            )?;
            current = out_dim;
        }
        let snake_idx = 1 + rates.len();
        let post_idx = 2 + rates.len();
        map.insert(
            format!("model.{snake_idx}.alpha"),
            Tensor::ones((1, current, 1), DType::F32, &dev)?,
        );
        // Use a LARGE post-conv weight scale so pre-tanh easily exceeds
        // 1 — proves the tanh bound is what's clamping the output.
        put_wn_conv(
            &mut map,
            &format!("model.{post_idx}"),
            current,
            1,
            7,
            1,
            5.0,
            &dev,
        )?;

        let vb = vb_from(map, &dev);
        let wg = WaveGenerator::load(vb, input_channels, channels, &rates, &kernels)?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 16), &dev)?;
        let y = wg.forward(&x)?;
        let prod: usize = rates.iter().product();
        assert_eq!(y.dims(), &[2, 1, 16 * prod]);

        let max = y.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max <= 1.0 + 1e-6, "output max {max} exceeds tanh bound 1.0");
        Ok(())
    }
}
