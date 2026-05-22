//! ContentVec content encoder for the RVC voice-conversion backend.
//!
//! ContentVec is a fairseq HuBERT-base checkpoint fine-tuned for
//! speaker-disentangled content extraction. RVC uses it as the linguistic
//! front-end: 16 kHz mono PCM in, dense per-frame content features out,
//! which the generator (Wave D.3) consumes alongside F0 and (optionally)
//! retrieved index features.
//!
//! # Architecture
//!
//! HuBERT-base, identical topology to `facebook/hubert-base-ls960`:
//!
//! - **Feature extractor**: 7-layer 1-D conv stack at the raw audio,
//!   strides `(5, 2, 2, 2, 2, 2, 2)` for a combined 320x downsample
//!   (16 kHz -> 50 Hz frame rate). Each layer is `Conv1d -> GroupNorm`
//!   (first layer) or `Conv1d -> LayerNorm` (remaining six) with GELU.
//! - **Feature projection**: `LayerNorm -> Linear(512 -> 768) -> Dropout`.
//! - **Transformer encoder**: 12 layers, 768 hidden, 12 heads,
//!   3072 FFN with GELU, pre-norm. Relative positional bias is applied
//!   through a 1-D depthwise conv (`conv_pos`, kernel 128, groups 16)
//!   added to the input embeddings before the first attention block.
//!
//! For RVC the two versions diverge only in which transformer layer's
//! activations are read out:
//!
//! - [`RvcVersion::V1`]: layer-9 output, no projection, 256 channels.
//!   (The v1 checkpoint actually ships as a 256-dim feature variant --
//!   the public `checkpoint_best_legacy_500.pt`.)
//! - [`RvcVersion::V2`]: layer-12 (final) output, fed through a learned
//!   `Linear(768 -> 768)` projection. Most current RVC models target v2.
//!
//! # Weight provenance
//!
//! The canonical v2 checkpoint is `hubert_base.pt` from
//! `https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt`.
//! This is a **fairseq** state-dict, not a HuggingFace `transformers`
//! HuBERT -- the key prefixes differ:
//!
//! | fairseq                                              | hf transformers                                     |
//! |------------------------------------------------------|-----------------------------------------------------|
//! | `feature_extractor.conv_layers.{i}.0.weight`         | `feature_extractor.conv_layers.{i}.conv.weight`     |
//! | `feature_extractor.conv_layers.0.2.{weight,bias}`    | `feature_extractor.conv_layers.0.layer_norm.{w,b}`  |
//! | `feature_extractor.conv_layers.{i>=1}.2.1.{w,b}`     | `feature_extractor.conv_layers.{i}.layer_norm.{w,b}`|
//! | `post_extract_proj.{weight,bias}`                    | `feature_projection.projection.{weight,bias}`       |
//! | `layer_norm.{weight,bias}`                           | `feature_projection.layer_norm.{weight,bias}`       |
//! | `encoder.pos_conv.0.{weight_g,weight_v,bias}`        | `encoder.pos_conv_embed.conv.{w_g,w_v,bias}`        |
//! | `encoder.layer_norm.{weight,bias}`                   | `encoder.layer_norm.{weight,bias}`                  |
//! | `encoder.layers.{i}.self_attn.{q,k,v,out}_proj.*`    | `encoder.layers.{i}.attention.{q,k,v,out}_proj.*`   |
//! | `encoder.layers.{i}.self_attn_layer_norm.*`          | `encoder.layers.{i}.layer_norm.*`                   |
//! | `encoder.layers.{i}.fc1.*` / `fc2.*`                 | `encoder.layers.{i}.feed_forward.{int,out}_dense.*` |
//! | `encoder.layers.{i}.final_layer_norm.*`              | `encoder.layers.{i}.final_layer_norm.*`             |
//!
//! `pos_conv` uses PyTorch weight-norm -- `weight_g` (gain, shape
//! `[768]`) and `weight_v` (direction, shape `[768, 48, 128]`) must be
//! combined into a single dense `[768, 48, 128]` kernel via
//! `weight = weight_g * weight_v / ||weight_v||` along the input dims.
//!
//! # Deferred: load() implementation
//!
//! `candle-transformers` 0.10 does not yet ship a HuBERT or Wav2Vec2
//! model. Implementing the full feature extractor + transformer encoder
//! from scratch -- together with the fairseq -> candle key remapping
//! and the weight-norm composition -- is a multi-hundred-LOC effort
//! that belongs in its own change. Wave D.2 therefore lands the public
//! contract and architecture documentation; [`ContentEncoder::load`]
//! returns
//! `Err(ContentError::ModelLoad("ContentVec loader pending upstream ..."))`.
//! The pipeline (Wave D.3) is expected to surface this as a backend
//! capability error and refuse to construct an RVC session until the
//! loader lands.
//!
//! When the loader is implemented, only [`ContentEncoder::load`] needs
//! to change -- the [`ContentEncoder::encode`] signature, the input
//! contract (mono 16 kHz f32 PCM), and the output contract
//! (`(1, n_frames, hidden_dim)` f32 tensor on the configured device)
//! are stable.

#![cfg(feature = "rvc")]
// The architecture docs above and on each item are dense with proper
// nouns (`HuBERT`, `ContentVec`, `PyTorch`, `HuggingFace`, `Wav2Vec2`,
// `RVC`) that would clutter the prose if backticked at every mention.
// Disable the doc-markdown lint at the file level.
#![allow(clippy::doc_markdown)]

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use thiserror::Error;

/// Sample rate, in Hz, that ContentVec / HuBERT-base operates on.
/// Any other input rate must be resampled by the caller before
/// invoking [`ContentEncoder::encode`].
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Combined stride of the 7-layer convolutional feature extractor.
/// 16 kHz audio is downsampled to a 50 Hz frame rate, so
/// `n_frames ~= samples.len() / FEATURE_DOWNSAMPLE`.
pub const FEATURE_DOWNSAMPLE: usize = 320;

/// Hidden width used by [`RvcVersion::V1`] readouts (HuBERT layer 9,
/// no learned projection -- the published v1 ContentVec ships a
/// 256-channel variant).
pub const V1_HIDDEN_DIM: usize = 256;

/// Hidden width used by [`RvcVersion::V2`] readouts (HuBERT layer 12,
/// followed by a learned `Linear(768 -> 768)` projection).
pub const V2_HIDDEN_DIM: usize = 768;

/// Which RVC checkpoint family the loaded weights belong to.
///
/// The choice changes the readout layer (HuBERT layer 9 vs. layer 12)
/// and the output hidden dimension (256 vs. 768) but not the underlying
/// HuBERT-base topology -- so the same model code serves both versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RvcVersion {
    /// HuBERT layer-9 readout, no projection, 256-dim per-frame features.
    /// Targets the legacy v1 ContentVec checkpoint
    /// (`checkpoint_best_legacy_500.pt`).
    V1,
    /// HuBERT layer-12 (final) readout, fed through the learned
    /// `Linear(768 -> 768)` projection. Targets `hubert_base.pt`.
    V2,
}

impl RvcVersion {
    /// Output hidden dimension for this version's content features.
    #[must_use]
    pub const fn hidden_dim(self) -> usize {
        match self {
            Self::V1 => V1_HIDDEN_DIM,
            Self::V2 => V2_HIDDEN_DIM,
        }
    }

    /// Index of the HuBERT transformer layer whose activations are
    /// returned as the content features (1-indexed, matching upstream).
    #[must_use]
    pub const fn readout_layer(self) -> usize {
        match self {
            Self::V1 => 9,
            Self::V2 => 12,
        }
    }
}

/// Errors that can surface from the ContentVec encoder.
#[derive(Debug, Error)]
pub enum ContentError {
    /// Failed to load or parse the model weights.
    #[error("ContentVec model load failed: {0}")]
    ModelLoad(String),

    /// A forward / inference error from candle.
    #[error("ContentVec inference failed: {0}")]
    Inference(String),

    /// Underlying I/O error (file open, read, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Underlying candle tensor error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// ContentVec content-feature encoder.
///
/// Wraps the HuBERT-base architecture with the readout / projection
/// behavior selected by [`RvcVersion`]. See the module-level docs for
/// the full architecture description and the fairseq -> candle key
/// remapping table.
#[derive(Debug)]
pub struct ContentEncoder {
    /// Device the model lives on; all returned tensors will be on this
    /// device.
    device: Device,
    /// Which checkpoint family this encoder was loaded against.
    version: RvcVersion,
    /// Path the weights were loaded from. Retained for diagnostics and
    /// for the deferred loader to consume once it lands.
    weights_path: PathBuf,
}

impl ContentEncoder {
    /// Load a ContentVec encoder from a fairseq HuBERT `.pt` checkpoint
    /// onto the given device.
    ///
    /// # Errors
    ///
    /// Currently returns [`ContentError::ModelLoad`] unconditionally:
    /// the candle HuBERT model + fairseq state-dict remapping is
    /// deferred (see the module-level docs). Once the loader lands,
    /// this will return errors from missing/mismatched tensors, file
    /// I/O, and candle failures.
    pub fn load(
        weights_path: &Path,
        device: &Device,
        rvc_version: RvcVersion,
    ) -> Result<Self, ContentError> {
        // Validate up-front so that, when the loader lands, callers
        // that pass a bogus path still see a fast, clear error.
        if !weights_path.exists() {
            return Err(ContentError::ModelLoad(format!(
                "weights path does not exist: {}",
                weights_path.display()
            )));
        }

        let _ = (device, rvc_version);

        Err(ContentError::ModelLoad(
            "ContentVec loader pending upstream candle-transformers HuBERT support \
             (Wave D.2 lands the architecture + contract; the fairseq state-dict \
             remapping is a follow-up). See backends/rvc/content.rs module docs."
                .to_string(),
        ))
    }

    /// Device the encoder's tensors live on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Which RVC checkpoint family this encoder is configured for.
    #[must_use]
    pub fn version(&self) -> RvcVersion {
        self.version
    }

    /// Output hidden dimension of the per-frame content features.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.version.hidden_dim()
    }

    /// Path the weights were loaded from. Useful for diagnostics.
    #[must_use]
    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }

    /// Encode a mono 16 kHz PCM waveform into per-frame content features.
    ///
    /// # Input
    ///
    /// `samples_16khz` is mono PCM at 16 kHz, `f32` in `[-1.0, 1.0]`.
    /// The caller is responsible for resampling and channel mixing
    /// upstream; supplying any other rate yields silently wrong
    /// linguistic content.
    ///
    /// # Output
    ///
    /// A tensor of shape `(1, n_frames, hidden_dim)`, dtype `f32`, on
    /// the encoder's configured device, where:
    ///
    /// - `n_frames ~= samples_16khz.len() / 320` (HuBERT's feature
    ///   extractor downsamples 16 kHz audio by 320x to a 50 Hz frame
    ///   rate; the exact count follows the standard convolution-output
    ///   formula `floor((L - k) / s) + 1` accumulated across all seven
    ///   conv layers).
    /// - `hidden_dim = 256` for [`RvcVersion::V1`] (HuBERT layer-9
    ///   readout, no projection).
    /// - `hidden_dim = 768` for [`RvcVersion::V2`] (HuBERT layer-12
    ///   readout fed through the learned 768->768 projection).
    ///
    /// # Errors
    ///
    /// Returns [`ContentError::Inference`] if the input is empty (the
    /// HuBERT conv stack needs at least one frame's worth of samples),
    /// or [`ContentError::Candle`] / [`ContentError::Inference`] from
    /// the forward pass once the loader lands.
    pub fn encode(&self, samples_16khz: &[f32]) -> Result<Tensor, ContentError> {
        if samples_16khz.len() < FEATURE_DOWNSAMPLE {
            return Err(ContentError::Inference(format!(
                "input too short: {} samples < one HuBERT frame ({} samples / 20 ms at 16 kHz)",
                samples_16khz.len(),
                FEATURE_DOWNSAMPLE
            )));
        }

        // Once the loader lands, the forward pass goes here:
        //   1. `Tensor::from_slice(samples_16khz, (1, samples_16khz.len()), &self.device)`
        //   2. 7-layer conv feature extractor -> `(1, 512, n_frames)`
        //   3. Transpose to `(1, n_frames, 512)` + feature-projection LN/Linear -> 768
        //   4. Add `pos_conv` relative-pos embedding -> encoder LN
        //   5. Run 12 transformer blocks, capturing the readout layer's
        //      activations (`self.version.readout_layer()`).
        //   6. For v2, apply the learned 768->768 projection. For v1,
        //      take the 256-dim readout as-is.
        //   7. Return as `(1, n_frames, hidden_dim)` f32.

        Err(ContentError::Inference(
            "ContentVec encode unavailable: model not loaded (loader pending -- see \
             ContentEncoder::load and module docs)."
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rvc_version_hidden_dims_match_spec() {
        assert_eq!(RvcVersion::V1.hidden_dim(), 256);
        assert_eq!(RvcVersion::V2.hidden_dim(), 768);
    }

    #[test]
    fn rvc_version_readout_layers_match_spec() {
        assert_eq!(RvcVersion::V1.readout_layer(), 9);
        assert_eq!(RvcVersion::V2.readout_layer(), 12);
    }

    #[test]
    fn load_missing_path_returns_clear_error() {
        let device = Device::Cpu;
        let result = ContentEncoder::load(
            Path::new("/nonexistent/path/to/hubert_base.pt"),
            &device,
            RvcVersion::V2,
        );
        match result {
            Err(ContentError::ModelLoad(msg)) => {
                assert!(
                    msg.contains("does not exist"),
                    "expected path-missing error, got: {msg}"
                );
            }
            other => panic!("expected ModelLoad error, got: {other:?}"),
        }
    }

    #[test]
    fn load_returns_pending_upstream_error_for_existing_path() {
        // Use a path that *does* exist so we exercise the deferred-loader
        // branch, not the path-missing guard. Once the real loader lands,
        // this test should be replaced with a fixture-based round-trip.
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let device = Device::Cpu;
        let result = ContentEncoder::load(tmp.path(), &device, RvcVersion::V2);
        match result {
            Err(ContentError::ModelLoad(msg)) => {
                assert!(
                    msg.contains("pending upstream"),
                    "expected pending-upstream error, got: {msg}"
                );
            }
            other => panic!("expected ModelLoad pending-upstream error, got: {other:?}"),
        }
    }
}
