//! ContentVec content encoder for the RVC voice-conversion backend.
//!
//! ContentVec is a fairseq HuBERT-base checkpoint fine-tuned for
//! speaker-disentangled content extraction. RVC uses it as the linguistic
//! front-end: 16 kHz mono PCM in, dense per-frame content features out,
//! which the [`super::generator`] consumes alongside F0 and
//! (optionally) retrieved index features.
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
//! # Loader status
//!
//! [`ContentEncoder::load`] consumes the [`super::hubert::HubertBase`]
//! composite (feature extractor + feature projection + pos-conv +
//! 12x transformer encoder layer):
//!
//! - [`RvcVersion::V2`] -- supported. The fairseq `hubert_base.pt`
//!   pickle is parsed via `candle_nn::VarBuilder::from_pth`, the
//!   feature-extractor / feature-projection / pos-conv / 12x encoder
//!   stack is assembled, and [`ContentEncoder::encode`] returns the
//!   post-layer-12 hidden state.
//! - [`RvcVersion::V1`] -- rejected at load time with
//!   [`ContentError::ModelLoad`]. The public v1 ContentVec
//!   (`checkpoint_best_legacy_500.pt`) ships a 256-dim variant whose
//!   model topology differs from the standard fairseq HuBERT-base, so it
//!   cannot be loaded into the [`super::hubert::HubertBase`] composite.
//!   A parallel loader for the v1 topology is deliberately deferred
//!   to a future wave; v2 covers the overwhelming majority of
//!   contemporary RVC models.

#![cfg(feature = "rvc")]
// The architecture docs above and on each item are dense with proper
// nouns (`HuBERT`, `ContentVec`, `PyTorch`, `HuggingFace`, `Wav2Vec2`,
// `RVC`) that would clutter the prose if backticked at every mention.
// Disable the doc-markdown lint at the file level.
#![allow(clippy::doc_markdown)]

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{Device, Tensor};
use thiserror::Error;

use super::hubert::HubertBase;

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
pub struct ContentEncoder {
    /// Device the model lives on; all returned tensors will be on this
    /// device.
    device: Device,
    /// Which checkpoint family this encoder was loaded against.
    version: RvcVersion,
    /// Path the weights were loaded from. Retained for diagnostics.
    weights_path: PathBuf,
    /// Loaded HuBERT-base model. Only populated for [`RvcVersion::V2`];
    /// v1 (the legacy 256-dim ContentVec) requires a different model
    /// topology and is rejected at [`ContentEncoder::load`] time.
    ///
    /// Wrapped in [`Arc`] so cloning an encoder (e.g. for sharing across
    /// pipeline sessions) doesn't deep-copy the model weights -- mirrors
    /// the `Arc<OnceCell<Arc<ContentEncoder>>>` pattern in `pipeline.rs`.
    hubert: Arc<HubertBase>,
}

// Manual `Debug` impl: `HubertBase` doesn't itself derive `Debug` (its
// inner `candle_nn` modules don't either), so we elide the model
// internals and surface only the diagnostically useful fields.
impl fmt::Debug for ContentEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContentEncoder")
            .field("device", &self.device)
            .field("version", &self.version)
            .field("weights_path", &self.weights_path)
            .field("hubert", &"<HubertBase>")
            .finish()
    }
}

impl ContentEncoder {
    /// Load a ContentVec encoder from a fairseq HuBERT `.pt` checkpoint
    /// onto the given device.
    ///
    /// Only [`RvcVersion::V2`] is supported by this loader: it consumes
    /// `hubert_base.pt`-compatible checkpoints and reads out the
    /// post-layer-12 hidden state. [`RvcVersion::V1`] requires a
    /// different (256-dim) topology and is rejected up front.
    ///
    /// # Errors
    ///
    /// - [`ContentError::ModelLoad`] if `weights_path` does not exist,
    ///   if `rvc_version` is [`RvcVersion::V1`] (unsupported topology),
    ///   or if the underlying pickle parse fails.
    /// - [`ContentError::Candle`] if any sub-module's load surfaces a
    ///   shape / dtype mismatch against the canonical HuBERT-base
    ///   topology.
    pub fn load(
        weights_path: &Path,
        device: &Device,
        rvc_version: RvcVersion,
    ) -> Result<Self, ContentError> {
        if !weights_path.exists() {
            return Err(ContentError::ModelLoad(format!(
                "weights path does not exist: {}",
                weights_path.display()
            )));
        }

        // v1 uses a different 256-dim ContentVec checkpoint
        // (`checkpoint_best_legacy_500.pt`) with a different model
        // topology than the standard fairseq HuBERT-base. The current
        // loader targets the v2 case (`hubert_base.pt` -> layer-12
        // readout) which covers the overwhelming majority of
        // contemporary RVC models. v1 support requires a parallel
        // loader that we deliberately defer.
        if matches!(rvc_version, RvcVersion::V1) {
            return Err(ContentError::ModelLoad(
                "RVC v1 (256-dim legacy ContentVec) requires a different model \
                 topology than fairseq HuBERT-base; the current loader supports \
                 v2 only. Pass RvcVersion::V2 and a hubert_base.pt-compatible \
                 checkpoint, or open an issue if v1 support is required."
                    .to_string(),
            ));
        }

        let hubert = HubertBase::load(weights_path, device)?;
        Ok(Self {
            device: device.clone(),
            version: rvc_version,
            weights_path: weights_path.to_path_buf(),
            hubert: Arc::new(hubert),
        })
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
    /// - `hidden_dim = 768` for [`RvcVersion::V2`] (HuBERT layer-12
    ///   readout). The loader does not apply a learned 768->768
    ///   projection: the `hubert_base.pt` state-dict does not carry
    ///   one (the architecture doc above predated checkpoint
    ///   inspection -- `post_extract_proj` is the 512->768 projection
    ///   INSIDE the feature-projection module, not a post-readout one).
    ///
    /// The forward path runs the full HuBERT-base composite
    /// ([`super::hubert::HubertBase::forward_layers`]) and returns the
    /// hidden state at `Vec` index 11 -- the post-layer-12 activation.
    ///
    /// # Errors
    ///
    /// - [`ContentError::Inference`] if the input is shorter than one
    ///   HuBERT frame (320 samples), if the configured version is
    ///   somehow [`RvcVersion::V1`] (defensive -- [`Self::load`] rejects
    ///   v1 up front), or if the underlying HuBERT forward returns
    ///   fewer hidden states than expected.
    /// - [`ContentError::Candle`] from any tensor operation inside the
    ///   forward pass (re-wrapped through
    ///   [`super::hubert::HubertBase::forward_layers`]).
    pub fn encode(&self, samples_16khz: &[f32]) -> Result<Tensor, ContentError> {
        if samples_16khz.len() < FEATURE_DOWNSAMPLE {
            return Err(ContentError::Inference(format!(
                "input too short: {} samples < one HuBERT frame ({} samples / 20 ms at 16 kHz)",
                samples_16khz.len(),
                FEATURE_DOWNSAMPLE
            )));
        }

        // V2 reads post-layer-12 (Vec index 11). V1 was rejected at
        // load time, so this match is exhaustive but defensive in case
        // of future version variants.
        let layer_idx = match self.version {
            RvcVersion::V2 => 11,
            RvcVersion::V1 => {
                return Err(ContentError::Inference(
                    "RVC v1 unsupported at this loader; load() should have rejected it."
                        .to_string(),
                ));
            }
        };

        let hidden_states = self.hubert.forward_layers(samples_16khz)?;

        hidden_states.get(layer_idx).cloned().ok_or_else(|| {
            ContentError::Inference(format!(
                "hubert produced {} hidden states; expected at least {}",
                hidden_states.len(),
                layer_idx + 1
            ))
        })
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
    fn load_v1_returns_unsupported_topology_error() {
        // V1 uses a different 256-dim ContentVec checkpoint with a
        // different model topology than the standard fairseq
        // HuBERT-base; the current loader rejects it at load time with
        // a clear message.
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let device = Device::Cpu;
        let result = ContentEncoder::load(tmp.path(), &device, RvcVersion::V1);
        match result {
            Err(ContentError::ModelLoad(msg)) => {
                assert!(
                    msg.contains("v1"),
                    "expected v1-unsupported error, got: {msg}"
                );
            }
            other => panic!("expected ModelLoad v1-unsupported error, got: {other:?}"),
        }
    }

    #[test]
    fn load_v2_against_garbage_file_returns_clean_load_error() {
        // An empty (zero-byte) tempfile is not a valid PyTorch pickle,
        // so `VarBuilder::from_pth` must fail with a structured error
        // rather than panic. Either `ModelLoad` (wrapped by
        // `HubertBase::load`) or `Candle` (raw candle error if it
        // surfaces unwrapped) is acceptable.
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let device = Device::Cpu;
        let result = ContentEncoder::load(tmp.path(), &device, RvcVersion::V2);
        assert!(
            matches!(
                result,
                Err(ContentError::ModelLoad(_) | ContentError::Candle(_))
            ),
            "expected clean load error, got: {result:?}"
        );
    }

    /// End-to-end smoke against the real `hubert_base.pt` checkpoint.
    ///
    /// Downloads the v2 ContentVec weights from
    /// `lj1995/VoiceConversionWebUI` via `hf-hub`, assembles the full
    /// [`ContentEncoder`], pushes a 1 s @ 16 kHz silence buffer through
    /// [`ContentEncoder::encode`], and asserts the documented
    /// `(1, ~50, 768)` output shape. This is the only test in the file
    /// that exercises real `.pt` parsing and the full forward path.
    ///
    /// `#[ignore]`'d because it requires network access (the first run
    /// pulls ~360 MB into `~/.cache/huggingface/`) and several seconds
    /// of CPU. Run explicitly with:
    ///
    /// ```bash
    /// cargo nextest run -p blazen-audio-vc --features rvc \
    ///     --run-ignored only encode_against_real_hubert_base
    /// ```
    ///
    /// Uses `#[tokio::test]` rather than a hand-rolled
    /// `Runtime::new().block_on(...)` because `tokio = { features =
    /// ["macros", "rt"] }` is already in `[dev-dependencies]` and the
    /// macro keeps the test signature symmetric with the other async
    /// integration tests in the crate (see `pipeline.rs`).
    #[tokio::test]
    #[ignore = "downloads ~360 MB; run explicitly with --run-ignored only"]
    async fn encode_against_real_hubert_base() {
        let path = super::super::weights::hf_download(
            "lj1995/VoiceConversionWebUI",
            "hubert_base.pt",
            None,
        )
        .await
        .expect("hf-hub download of hubert_base.pt should succeed");

        let device = Device::Cpu;
        let encoder = ContentEncoder::load(&path, &device, RvcVersion::V2)
            .expect("ContentEncoder::load against real hubert_base.pt should succeed");

        // 16 000 samples = 1 s at 16 kHz; silence is enough -- this
        // test validates the loader + forward shape contract, not
        // audio quality.
        let samples = vec![0.0_f32; 16_000];
        let out = encoder
            .encode(&samples)
            .expect("encode of 1 s silence should succeed");

        let dims = out.dims();
        assert_eq!(dims.len(), 3, "expected (B, T, C), got {dims:?}");
        assert_eq!(dims[0], 1, "expected batch 1, got {}", dims[0]);
        assert_eq!(
            dims[2],
            RvcVersion::V2.hidden_dim(),
            "expected hidden dim {}, got {}",
            RvcVersion::V2.hidden_dim(),
            dims[2]
        );
        // 16 000 / 320 = 50; the conv-output floor formula shaves a
        // handful of frames off the edges. Same closed interval the
        // synthetic-weights tests use.
        assert!(
            (40..=50).contains(&dims[1]),
            "expected ~50 frames for 1 s @ 16 kHz, got {}",
            dims[1]
        );
    }
}
