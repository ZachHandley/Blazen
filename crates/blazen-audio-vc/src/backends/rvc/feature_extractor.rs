//! HuBERT-base 7-layer convolutional feature extractor.
//!
//! This is the front-end of `facebook/hubert-base-ls960` (and the
//! `lj1995/VoiceConversionWebUI/hubert_base.pt` checkpoint that RVC
//! ships): a stack of seven 1-D convolutions that maps a 16 kHz mono
//! waveform into a 50 Hz / 512-channel feature sequence
//! (`320x` combined stride).
//!
//! # Topology
//!
//! Per the fairseq `Wav2VecModel` defaults (and the equivalent
//! `Wav2Vec2FeatureEncoder` in HuggingFace `transformers`):
//!
//! | layer | in -> out  | kernel | stride | norm                  |
//! |------:|------------|-------:|-------:|-----------------------|
//! | 0     | 1 -> 512   | 10     | 5      | `GroupNorm(32, 512)`  |
//! | 1     | 512 -> 512 | 3      | 2      | none                  |
//! | 2     | 512 -> 512 | 3      | 2      | none                  |
//! | 3     | 512 -> 512 | 3      | 2      | none                  |
//! | 4     | 512 -> 512 | 3      | 2      | none                  |
//! | 5     | 512 -> 512 | 2      | 2      | none                  |
//! | 6     | 512 -> 512 | 2      | 2      | none                  |
//!
//! This is fairseq's `extractor_mode="default"` (what `hubert_base.pt`
//! ships): layer 0 is `Conv1d(no bias) -> GroupNorm -> GELU(erf)`, layers
//! 1..=6 are `Conv1d(no bias) -> GELU(erf)` with no normalization. The
//! fairseq reference uses `nn.functional.gelu` (the exact, erf-based
//! form), so we match that with `Tensor::gelu_erf` rather than the tanh
//! approximation.
//!
//! The combined stride is `5 * 2 * 2 * 2 * 2 * 2 * 2 = 320`, which is
//! exactly `FEATURE_DOWNSAMPLE` from the parent [`super::content`]
//! module (16 kHz / 320 = 50 Hz frame rate).
//!
//! # Shapes
//!
//! - Input:  `(B, 1, T_samples)` -- mono PCM. The caller is responsible
//!   for inserting the channel dim before invoking [`Module::forward`];
//!   see the `ContentEncoder::encode` contract in [`super::content`].
//! - Output: `(B, 512, T_frames)`, where `T_frames` follows the standard
//!   convolution-output formula `floor((L - k) / s) + 1` accumulated
//!   across all seven layers (approximately `T_samples / 320`).
//!
//! # Weight layout (fairseq state-dict)
//!
//! The supplied [`VarBuilder`] must already be rooted at
//! `feature_extractor` (i.e. the caller does `vb.pp("feature_extractor")`
//! before calling [`FeatureExtractor::load`]). Within that root, layer
//! `i` lives at:
//!
//! - `conv_layers.{i}.0.weight` -- conv kernel `(out, in, k)`, no bias
//! - Layer 0 norm: `conv_layers.0.2.{weight,bias}` -- `GroupNorm(32, 512)`
//! - Layers 1..=6: no norm tensors (bare conv + GELU)
//!
//! This matches the table at the top of [`super::content`] (the
//! fairseq -> candle remap that [`super::hubert::HubertBase::load`]
//! honors).

#![cfg(feature = "rvc")]
// The architecture docs are dense with proper nouns (`HuBERT`,
// `Wav2Vec2`, `PyTorch`, `GELU`, `LayerNorm`, `GroupNorm`, `RVC`); match
// the file-level allow in `content.rs` so the docs read cleanly.
#![allow(clippy::doc_markdown)]
// `VarBuilder` is consumed by value -- the candle convention. Its inner
// state is `Arc<Inner>`-shaped so the move is cheap, and the upstream
// candle-nn crate uses the same idiom (see `bark/gpt_block.rs`).
#![allow(clippy::needless_pass_by_value)]

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, VarBuilder, conv1d_no_bias, group_norm};

/// LayerNorm / GroupNorm epsilon. Fairseq's `Fp32LayerNorm` and
/// `Fp32GroupNorm` both default to PyTorch's `1e-5`.
const NORM_EPS: f64 = 1e-5;

/// Output channel count of every layer in the stack (and the input to
/// every layer except the first).
const HIDDEN_CHANNELS: usize = 512;

/// Number of groups for layer 0's `GroupNorm` (fairseq default for the
/// HuBERT-base feature extractor: 32 groups across 512 channels = 16
/// channels per group).
const GROUP_NORM_GROUPS: usize = 32;

/// `(kernel, stride)` for each of the seven conv layers, in order. The
/// kernels match fairseq's
/// `[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2` -- HuBERT-base's
/// canonical `conv_feature_layers` default.
const LAYER_SHAPES: [(usize, usize); 7] = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)];

/// One conv-stack layer: `Conv1d -> (GroupNorm?) -> GELU(erf)`.
///
/// `hubert_base.pt` ships fairseq's `extractor_mode="default"` topology:
/// only layer 0 carries a `GroupNorm`; layers 1..=6 are bare
/// `Conv1d -> Dropout(paramless) -> GELU` with no normalization.
struct FeatureLayer {
    conv: Conv1d,
    gn: Option<GroupNorm>,
}

impl FeatureLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = match &self.gn {
            Some(gn) => gn.forward(&xs)?,
            None => xs,
        };
        xs.gelu_erf()
    }
}

/// HuBERT-base 7-layer feature extractor: `(B, 1, T_samples) -> (B, 512, T_frames)`.
///
/// See the module-level docs for the per-layer shape table, the
/// 320x combined stride, and the fairseq state-dict layout the
/// [`FeatureExtractor::load`] entry-point consumes.
pub(super) struct FeatureExtractor {
    layers: Vec<FeatureLayer>,
}

impl FeatureExtractor {
    /// Load the 7-layer conv stack from a [`VarBuilder`] already rooted
    /// at `feature_extractor` (the caller does
    /// `vb.pp("feature_extractor")` before invoking this).
    ///
    /// # Errors
    ///
    /// Propagates any [`candle_core::Error`] surfaced by the underlying
    /// `VarBuilder` (missing tensor, dtype/shape mismatch, etc.).
    pub(super) fn load(vb: VarBuilder) -> Result<Self> {
        let conv_layers = vb.pp("conv_layers");
        let mut layers = Vec::with_capacity(LAYER_SHAPES.len());
        for (i, &(kernel, stride)) in LAYER_SHAPES.iter().enumerate() {
            let in_channels = if i == 0 { 1 } else { HIDDEN_CHANNELS };
            let layer_vb = conv_layers.pp(i.to_string());
            let cfg = Conv1dConfig {
                stride,
                ..Conv1dConfig::default()
            };
            // Fairseq stores the conv as `conv_layers.{i}.0.weight`
            // (sequential index `0` inside the per-layer module), with
            // no bias.
            let conv = conv1d_no_bias(in_channels, HIDDEN_CHANNELS, kernel, cfg, layer_vb.pp("0"))?;
            let gn = if i == 0 {
                // Layer 0 only: `conv_layers.0.2.{weight,bias}` is the
                // `GroupNorm(32, 512)`. Sequential index `2` lands
                // directly on the `GroupNorm` module (no inner wrapper).
                Some(group_norm(
                    GROUP_NORM_GROUPS,
                    HIDDEN_CHANNELS,
                    NORM_EPS,
                    layer_vb.pp("2"),
                )?)
            } else {
                // Layers 1..=6 carry no normalization in fairseq's
                // `extractor_mode="default"` — the per-layer module is
                // just `Conv1d -> Dropout -> GELU`.
                None
            };
            layers.push(FeatureLayer { conv, gn });
        }
        Ok(Self { layers })
    }
}

impl Module for FeatureExtractor {
    /// Forward `(B, 1, T_samples)` -> `(B, 512, T_frames)`.
    ///
    /// `T_frames` is determined by sequentially applying each layer's
    /// `floor((L - kernel) / stride) + 1` (no padding, no dilation), so
    /// the output length is approximately `T_samples / 320` for inputs
    /// that are an exact multiple of the combined stride.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Smoke-test that the stack assembles against a zero-initialised
    /// `VarBuilder` and produces the documented output shape. Numerical
    /// correctness against a real checkpoint is validated by the
    /// `#[ignore]`'d integration test in [`super::content`].
    #[test]
    fn forward_shape_matches_320x_downsample() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let extractor = FeatureExtractor::load(vb)?;

        // 16 000 samples (= 1 s @ 16 kHz) is comfortably above the
        // minimum receptive field. The exact output length depends on
        // the per-layer floor formula; we just sanity-check the rank
        // and channel count, plus that the time dim is in the expected
        // ballpark (50 frames / s, give or take edge effects).
        let samples = 16_000;
        let xs = Tensor::zeros((1, 1, samples), DType::F32, &device)?;
        let out = extractor.forward(&xs)?;
        let dims = out.dims();
        assert_eq!(dims.len(), 3, "expected (B, C, T), got {dims:?}");
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], HIDDEN_CHANNELS);
        // 16 000 / 320 = 50; the conv-output floor formula shaves a
        // handful of frames off the edges, so accept the closed
        // interval [40, 50].
        assert!(
            (40..=50).contains(&dims[2]),
            "expected ~50 frames for 1 s @ 16 kHz, got {}",
            dims[2]
        );
        Ok(())
    }

    #[test]
    fn layer_count_and_shapes_match_hubert_base() {
        assert_eq!(LAYER_SHAPES.len(), 7);
        // Combined stride must be exactly 320 (16 kHz -> 50 Hz).
        let combined: usize = LAYER_SHAPES.iter().map(|&(_, s)| s).product();
        assert_eq!(combined, 320);
        // First layer kernel must be 10 (fairseq HuBERT-base default).
        assert_eq!(LAYER_SHAPES[0].0, 10);
    }
}
