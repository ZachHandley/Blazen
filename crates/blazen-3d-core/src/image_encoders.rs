//! Image encoders for 3D conditioning.
//!
//! The first 3D port (TripoSR) conditions on per-patch DINOv2-base
//! features. This module hosts a thin wrapper around
//! `candle_transformers::models::dinov2::DinoVisionTransformer` so the
//! consuming backend doesn't need to depend on candle-transformers
//! directly and so future encoder swaps (CLIP, SigLIP, …) have a place
//! to land.

use candle_core::Tensor;
use candle_nn::Module;
use candle_nn::VarBuilder;
use candle_transformers::models::dinov2::DinoVisionTransformer;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Configuration for the [`DinoV2Encoder`] wrapper.
///
/// `patch_size` is reported back to callers for documentation purposes
/// (the upstream candle implementation hard-codes a `14`-pixel patch),
/// but the other fields are passed through to
/// [`DinoVisionTransformer::new`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DinoV2Config {
    /// Patch edge length in pixels. Must be `14` to match the upstream
    /// candle implementation; other values are rejected at
    /// [`DinoV2Encoder::load`].
    pub patch_size: usize,
    /// Transformer hidden dimension. `768` for `dinov2-base`.
    pub hidden_dim: usize,
    /// Number of transformer blocks. `12` for `dinov2-base`.
    pub num_layers: usize,
    /// Number of attention heads per block. `12` for `dinov2-base`.
    pub num_heads: usize,
}

impl DinoV2Config {
    /// Default config for the `dinov2-base` checkpoint
    /// (`patch_size=14`, `hidden_dim=768`, `num_layers=12`,
    /// `num_heads=12`).
    #[must_use]
    pub fn base_default() -> Self {
        Self {
            patch_size: 14,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
        }
    }
}

/// DINOv2 vision-transformer wrapper backed by
/// `candle_transformers::models::dinov2`.
///
/// Loaded weights stay on the device they were materialised on; the
/// wrapper is otherwise stateless.
#[derive(Debug)]
pub struct DinoV2Encoder {
    model: DinoVisionTransformer,
    config: DinoV2Config,
}

impl DinoV2Encoder {
    /// Load a DINOv2 encoder from the supplied `VarBuilder` and
    /// configuration.
    ///
    /// The upstream candle implementation hard-codes a patch size of
    /// `14`. Configurations that disagree are rejected with
    /// [`ImageEncoderError::ModelLoad`] so the caller is not surprised
    /// by silently-ignored hyperparameters.
    pub fn load(weights: &VarBuilder, config: DinoV2Config) -> Result<Self, ImageEncoderError> {
        if config.patch_size != 14 {
            return Err(ImageEncoderError::ModelLoad(format!(
                "DinoV2Config.patch_size must be 14 to match candle's hard-coded \
                 PATCH_SIZE; got {}",
                config.patch_size
            )));
        }
        let model = DinoVisionTransformer::new(
            weights.clone(),
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
        )
        .map_err(|e| ImageEncoderError::ModelLoad(format!("DinoVisionTransformer::new: {e}")))?;
        Ok(Self { model, config })
    }

    /// Encode a `(1, 3, H, W)` image tensor where `H` and `W` are
    /// multiples of `patch_size`.
    ///
    /// Output: the raw classifier logits `(1, 1000)` returned by the
    /// upstream candle DINOv2 head. Callers that want per-patch
    /// features should drive a separate forward via
    /// `get_intermediate_layers` on the underlying model — that path
    /// is reserved for a follow-up wrapper once a concrete
    /// 3D-backend consumer pins down which layer + reshape it needs.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor, ImageEncoderError> {
        let dims = image.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
            return Err(ImageEncoderError::ModelLoad(format!(
                "image tensor must be (1, 3, H, W); got shape {dims:?}"
            )));
        }
        let (h, w) = (dims[2], dims[3]);
        if h % self.config.patch_size != 0 || w % self.config.patch_size != 0 {
            return Err(ImageEncoderError::ModelLoad(format!(
                "image H and W must be multiples of patch_size={}, got ({h}, {w})",
                self.config.patch_size
            )));
        }
        let out = self.model.forward(image)?;
        Ok(out)
    }

    /// The config this encoder was loaded with.
    #[must_use]
    pub fn config(&self) -> &DinoV2Config {
        &self.config
    }
}

/// Errors raised by image-encoder construction and inference.
#[derive(Debug, Error)]
pub enum ImageEncoderError {
    /// A capability that is not yet implemented (e.g. an alternate
    /// encoder family) was requested.
    #[error("image encoder feature not supported yet: {0}")]
    Unsupported(String),
    /// Model construction / weight loading failed (bad shapes,
    /// unsupported config, missing keys, …).
    #[error("model load failed: {0}")]
    ModelLoad(String),
    /// A candle tensor operation failed during the forward pass.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_default_matches_dinov2_base() {
        let cfg = DinoV2Config::base_default();
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.hidden_dim, 768);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.num_heads, 12);
    }

    #[test]
    fn load_rejects_non_canonical_patch_size() {
        // Build an empty VarBuilder backed by zeros so we get past
        // dtype/device requirements; the patch-size check rejects
        // before any tensor is materialised.
        use candle_core::{DType, Device};
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let bad_cfg = DinoV2Config {
            patch_size: 16,
            ..DinoV2Config::base_default()
        };
        let err = DinoV2Encoder::load(&vb, bad_cfg).unwrap_err();
        match err {
            ImageEncoderError::ModelLoad(msg) => {
                assert!(
                    msg.contains("patch_size") && msg.contains("14"),
                    "error should call out the patch-size constraint; got: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_variant_carries_actionable_message() {
        // Sanity-check that the Unsupported variant exists and round-trips
        // a meaningful message — the rest of the crate uses this pattern.
        let err = ImageEncoderError::Unsupported(
            "CLIP encoder landing in follow-up commit; see image_encoders.rs".to_string(),
        );
        let msg = format!("{err}");
        assert!(msg.contains("follow-up commit"));
    }
}
