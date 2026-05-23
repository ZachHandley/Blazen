//! HuBERT-base model wrapper for the RVC content encoder.
//!
//! Composes the [`FeatureExtractor`] (7-layer conv front-end), the
//! [`PosConv`] (relative-position depthwise conv) and 12x
//! [`HubertEncoderLayer`] (pre-norm transformer blocks), plus the
//! small [`FeatureProjection`] (`LayerNorm(512) + Linear(512 -> 768)`)
//! and the single pre-encoder `LayerNorm(768)` that fairseq's
//! `layer_norm_first = True` HuBERT-base applies before the stacked
//! encoder layers.
//!
//! # Forward path
//!
//! ```text
//! samples (T_samples,) f32 in [-1, 1]
//!   -> reshape (1, 1, T_samples)
//!   -> FeatureExtractor          (1, 512, T_frames)     [320x downsample]
//!   -> transpose to              (1, T_frames, 512)
//!   -> FeatureProjection:
//!        LayerNorm(512) + Linear(512 -> 768)
//!                                (1, T_frames, 768)
//!   -> h + PosConv(h)            (1, T_frames, 768)     [residual]
//!   -> pre_encoder_layer_norm    (1, T_frames, 768)     [applied ONCE,
//!                                                        BEFORE the layers]
//!   -> for i in 0..12:           (1, T_frames, 768)
//!        h = encoder.layers[i](h)
//!        hidden_states.push(h)
//! ```
//!
//! Because HuBERT-base uses `layer_norm_first = True` (the fairseq /
//! HF transformers `Wav2Vec2 do_stable_layer_norm = True` default for
//! the `-base` variant), `encoder.layer_norm` is applied ONCE BEFORE the
//! stack -- NOT after it. The per-layer hidden states captured by
//! [`HubertBase::forward_layers`] are therefore already
//! post-encoder-layer-norm and match what fairseq's
//! `extract_features(output_layer = N)` returns to RVC at readout time.
//!
//! # Hidden-state semantics
//!
//! [`HubertBase::forward_layers`] returns a `Vec<Tensor>` of length 12,
//! where index `i` is the activation produced by encoder layer `i + 1`
//! (1-indexed in upstream docs). That is:
//!
//! - index `0` = post-layer-1
//! - index `8` = post-layer-9 (`RvcVersion::V1` readout)
//! - index `11` = post-layer-12 (`RvcVersion::V2` readout)
//!
//! We eagerly accumulate all 12 entries rather than short-circuiting at
//! the requested layer: the cost is `12 * (1, T_frames, 768) * 4 bytes`
//! (a few MB for typical RVC clip lengths) and the API stays simple for
//! callers that may want multiple readouts (e.g. layer ablation).
//!
//! # Weight layout (fairseq state-dict)
//!
//! Loaded from the ROOT [`VarBuilder`] (no `.pp(...)` prefix applied by
//! the caller); each sub-module walks into its own subtree:
//!
//! - `feature_extractor.*` -- see [`super::feature_extractor`] docs
//! - `layer_norm.{weight,bias}` -- `LayerNorm(512)` of the feature
//!   projection (top-level in fairseq -- NOT nested under a
//!   `feature_projection` module)
//! - `post_extract_proj.{weight,bias}` -- `Linear(512 -> 768)` of the
//!   feature projection (top-level)
//! - `encoder.pos_conv.0.{weight_g, weight_v, bias}` -- see
//!   [`super::hubert_encoder::PosConv`]
//! - `encoder.layer_norm.{weight,bias}` -- pre-encoder `LayerNorm(768)`
//! - `encoder.layers.{0..11}.*` -- per-layer transformer blocks, see
//!   [`super::hubert_encoder::HubertEncoderLayer`]
//!
//! See [`super::content`]'s state-dict remapping table for the full
//! list and the HuggingFace `transformers` counterparts.

#![cfg(feature = "rvc")]
// The architecture docs are dense with proper nouns (`HuBERT`,
// `Wav2Vec2`, `PyTorch`, `LayerNorm`, `PosConv`, `RVC`, `ContentVec`);
// match the file-level allow in `feature_extractor.rs`,
// `hubert_encoder.rs` and `content.rs` so the prose reads cleanly.
#![allow(clippy::doc_markdown)]
// `VarBuilder` is consumed by value -- the candle convention. Its inner
// state is `Arc<Inner>`-shaped so the move is cheap, and the upstream
// candle-nn crate uses the same idiom (see `bark/gpt_block.rs` and the
// sibling `feature_extractor.rs` / `hubert_encoder.rs` files).
#![allow(clippy::needless_pass_by_value)]

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};

use super::content::ContentError;
use super::feature_extractor::FeatureExtractor;
use super::hubert_encoder::{HubertConfig, HubertEncoderLayer, PosConv};

/// `LayerNorm` epsilon. Fairseq's `LayerNorm`s default to PyTorch's
/// standard `1e-5`, matching the sibling files.
const NORM_EPS: f64 = 1e-5;

/// Feature-extractor output channel count (= feature-projection input).
const FEATURE_DIM: usize = 512;

/// Number of stacked transformer encoder layers in HuBERT-base.
const NUM_ENCODER_LAYERS: usize = 12;

/// Feature projection: `LayerNorm(512) + Linear(512 -> 768)`.
///
/// Applied between the 7-layer conv [`FeatureExtractor`] and the
/// transformer encoder stack. The fairseq state-dict layout splits
/// these across two TOP-LEVEL keys (`layer_norm.*` and
/// `post_extract_proj.*`) rather than nesting them under a single
/// `feature_projection` module -- see [`FeatureProjection::load`]. The
/// HuggingFace `transformers` port re-groups them under
/// `feature_projection.{layer_norm, projection}`; we follow the fairseq
/// layout because that is what `hubert_base.pt` ships.
pub(super) struct FeatureProjection {
    /// `LayerNorm(512)`, loaded from the top-level `layer_norm.*` keys.
    /// Applied BEFORE the projection (HF `feature_projection.layer_norm`).
    /// Not to be confused with the pre-encoder `encoder.layer_norm` --
    /// that's a separate, 768-wide LN owned by [`HubertBase`].
    layer_norm: LayerNorm,
    /// `Linear(512 -> 768)`, loaded from the top-level
    /// `post_extract_proj.*` keys (HF
    /// `feature_projection.projection`).
    projection: Linear,
}

impl FeatureProjection {
    /// Load from the ROOT [`VarBuilder`] (not `pp`'d into anything --
    /// the two pieces live at the top-level `layer_norm` and
    /// `post_extract_proj` keys in the fairseq checkpoint).
    ///
    /// # Errors
    ///
    /// Propagates any [`candle_core::Error`] surfaced by the underlying
    /// [`VarBuilder`] (missing tensor, dtype / shape mismatch, etc.).
    pub(super) fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let layer_norm = layer_norm(FEATURE_DIM, NORM_EPS, vb.pp("layer_norm"))?;
        let projection = linear(
            FEATURE_DIM,
            HubertConfig::HUBERT_BASE.embed_dim,
            vb.pp("post_extract_proj"),
        )?;
        Ok(Self {
            layer_norm,
            projection,
        })
    }

    /// Forward `(B, T, 512) -> (B, T, 768)`.
    pub(super) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.layer_norm.forward(xs)?;
        self.projection.forward(&h)
    }
}

/// HuBERT-base model wrapper.
///
/// Composes [`FeatureExtractor`] + [`FeatureProjection`] + [`PosConv`] +
/// the pre-encoder `LayerNorm(768)` + 12x [`HubertEncoderLayer`] into the
/// full ContentVec front-end. See the module-level docs for the forward
/// path, the `layer_norm_first = True` rationale, and the per-layer
/// hidden-state semantics returned by [`Self::forward_layers`].
pub(super) struct HubertBase {
    feature_extractor: FeatureExtractor,
    feature_projection: FeatureProjection,
    pos_conv: PosConv,
    /// `encoder.layer_norm` from the fairseq state dict. Applied ONCE,
    /// BEFORE the 12 encoder layers, per HuBERT-base's
    /// `layer_norm_first = True`. There is NO matching post-stack
    /// `LayerNorm`.
    pre_encoder_layer_norm: LayerNorm,
    layers: Vec<HubertEncoderLayer>,
    device: Device,
}

impl HubertBase {
    /// Load HuBERT-base from a fairseq `.pt` checkpoint
    /// (`hubert_base.pt`).
    ///
    /// # Errors
    ///
    /// - [`ContentError::ModelLoad`] if the `.pt` pickle cannot be
    ///   parsed by candle (corrupted file, wrong dtype, missing tensor,
    ///   etc.).
    /// - [`ContentError::Candle`] if any sub-module's `load` surfaces a
    ///   shape / dtype mismatch against the canonical HuBERT-base
    ///   topology.
    pub(super) fn load(weights_path: &Path, device: &Device) -> Result<Self, ContentError> {
        let vb = VarBuilder::from_pth(weights_path, DType::F32, device).map_err(|e| {
            ContentError::ModelLoad(format!(
                "hubert load: VarBuilder::from_pth({}) failed: {e}",
                weights_path.display()
            ))
        })?;
        Self::from_vb(vb, device)
    }

    /// Assemble the model from an existing [`VarBuilder`].
    ///
    /// Split out from [`Self::load`] so tests can exercise the full
    /// composition against a `VarBuilder::zeros` without needing real
    /// `.pt` weights on disk. The `#[ignore]`'d integration test in
    /// [`super::content`] covers the real-weights path.
    ///
    /// # Errors
    ///
    /// Propagates [`ContentError::Candle`] for any per-sub-module shape
    /// / dtype mismatch surfaced by the underlying [`VarBuilder`].
    pub(super) fn from_vb(vb: VarBuilder, device: &Device) -> Result<Self, ContentError> {
        let cfg = HubertConfig::HUBERT_BASE;

        let feature_extractor = FeatureExtractor::load(vb.pp("feature_extractor"))?;
        let feature_projection = FeatureProjection::load(vb.clone())?;
        let pos_conv = PosConv::load(vb.pp("encoder").pp("pos_conv").pp("0"), cfg.embed_dim)?;
        let pre_encoder_layer_norm =
            layer_norm(cfg.embed_dim, NORM_EPS, vb.pp("encoder").pp("layer_norm"))?;

        let mut layers = Vec::with_capacity(NUM_ENCODER_LAYERS);
        let encoder_layers = vb.pp("encoder").pp("layers");
        for i in 0..NUM_ENCODER_LAYERS {
            let layer = HubertEncoderLayer::load(encoder_layers.pp(i.to_string()), cfg)?;
            layers.push(layer);
        }

        Ok(Self {
            feature_extractor,
            feature_projection,
            pos_conv,
            pre_encoder_layer_norm,
            layers,
            device: device.clone(),
        })
    }

    /// Forward a mono 16 kHz f32 PCM clip into per-layer hidden states.
    ///
    /// Returns a `Vec` of 12 tensors, each of shape
    /// `(1, T_frames, 768)`, where:
    ///
    /// - index `0` = post-layer-1 activation
    /// - index `8` = post-layer-9 activation (`RvcVersion::V1` readout)
    /// - index `11` = post-layer-12 activation (`RvcVersion::V2` readout)
    ///
    /// `T_frames` follows the standard convolution-output formula
    /// accumulated across the 7-layer feature extractor (approximately
    /// `samples_16khz.len() / 320`, the 50 Hz HuBERT frame rate).
    ///
    /// # Errors
    ///
    /// Returns [`ContentError::Inference`] for any failure inside the
    /// forward pass (tensor construction, conv, attention, layer norm,
    /// residual add, etc.); the underlying candle error message is
    /// preserved in the wrapped string.
    pub(super) fn forward_layers(
        &self,
        samples_16khz: &[f32],
    ) -> Result<Vec<Tensor>, ContentError> {
        // Inner helper so we can do the candle work with the ergonomic
        // `?` operator on `candle_core::Result`, then map every error
        // through a single `ContentError::Inference` wrap. This keeps
        // the call-site (and the error context) uniform across the
        // entire forward pass.
        let inner = || -> candle_core::Result<Vec<Tensor>> {
            let xs = Tensor::from_slice(samples_16khz, (1, 1, samples_16khz.len()), &self.device)?;

            // 1. Conv feature extractor: (1, 1, T_samples) -> (1, 512, T_frames).
            let h = self.feature_extractor.forward(&xs)?;

            // 2. Transpose to (1, T_frames, 512) for the
            //    sequence-major projection / encoder stack.
            let h = h.transpose(1, 2)?.contiguous()?;

            // 3. FeatureProjection: (1, T_frames, 512) -> (1, T_frames, 768).
            let h = self.feature_projection.forward(&h)?;

            // 4. Relative-position residual: h + PosConv(h).
            //    PosConv preserves the (B, T, C) shape, including the
            //    even-kernel trailing-frame truncation -- see
            //    `super::hubert_encoder::PosConv`.
            let pos = self.pos_conv.forward(&h)?;
            let h = (&h + &pos)?;

            // 5. Pre-encoder LayerNorm: applied ONCE, BEFORE the 12
            //    encoder layers, per HuBERT-base's
            //    `layer_norm_first = True`. There is no matching
            //    post-stack LayerNorm.
            let mut h = self.pre_encoder_layer_norm.forward(&h)?;

            // 6. Stacked encoder layers, accumulating per-layer
            //    outputs. After the loop, `h` is identical to
            //    `hidden_states[11]`.
            let mut hidden_states = Vec::with_capacity(NUM_ENCODER_LAYERS);
            for layer in &self.layers {
                h = layer.forward(&h)?;
                hidden_states.push(h.clone());
            }
            Ok(hidden_states)
        };

        inner().map_err(|e| ContentError::Inference(format!("hubert forward: {e}")))
    }

    /// Device this model lives on. All tensors returned from
    /// [`Self::forward_layers`] are on this device.
    ///
    /// Currently consumed only by the in-file shape tests; production
    /// callers read the device via [`super::content::ContentEncoder::device`]
    /// which mirrors the same handle.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test the full composition: load the model against a
    /// zero-initialised `VarBuilder`, push a 1 s @ 16 kHz clip through,
    /// and assert that the returned hidden-state vector has the
    /// documented shape contract. Numerical correctness against real
    /// weights is covered by the `#[ignore]`'d integration test in
    /// [`super::content`].
    #[test]
    fn forward_layers_returns_12_tensors_of_correct_shape() -> Result<(), ContentError> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = HubertBase::from_vb(vb, &device)?;

        // 16 000 samples (= 1 s @ 16 kHz) is comfortably above the
        // minimum receptive field of the 7-layer conv stack.
        let samples = vec![0.0_f32; 16_000];
        let hidden = model.forward_layers(&samples)?;

        assert_eq!(
            hidden.len(),
            NUM_ENCODER_LAYERS,
            "expected exactly 12 per-layer hidden states, got {}",
            hidden.len()
        );

        let embed_dim = HubertConfig::HUBERT_BASE.embed_dim;
        for (i, h) in hidden.iter().enumerate() {
            let dims = h.dims();
            assert_eq!(dims.len(), 3, "layer {i}: expected (B, T, C), got {dims:?}");
            assert_eq!(dims[0], 1, "layer {i}: expected batch 1, got {}", dims[0]);
            assert_eq!(
                dims[2], embed_dim,
                "layer {i}: expected hidden dim {embed_dim}, got {}",
                dims[2]
            );
            // 16 000 / 320 = 50; the conv-output floor formula shaves
            // a handful of frames off the edges, so accept the closed
            // interval [40, 50] -- matches the `FeatureExtractor` test
            // bound in the sibling `feature_extractor.rs`.
            assert!(
                (40..=50).contains(&dims[1]),
                "layer {i}: expected ~50 frames for 1 s @ 16 kHz, got {}",
                dims[1]
            );
        }
        Ok(())
    }

    /// Sanity-check that the full composite assembles against a
    /// `VarBuilder::zeros` without errors -- catches sub-module key
    /// path / shape regressions independently of the forward pass.
    #[test]
    fn from_vb_succeeds_on_zero_initialized_builder() -> Result<(), ContentError> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = HubertBase::from_vb(vb, &device)?;

        assert_eq!(model.layers.len(), NUM_ENCODER_LAYERS);
        // Device round-trip: `Device::Cpu` doesn't implement `PartialEq`
        // directly, so check the discriminant via `is_cpu()`.
        assert!(model.device().is_cpu());
        Ok(())
    }
}
