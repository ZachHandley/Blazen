//! Gaussian-splat rasterization primitives.
//!
//! Defines the public surface for 3D Gaussian-splat rendering:
//! the [`GaussianSplat`] storage layout, [`CameraIntrinsics`] /
//! [`CameraExtrinsics`] structs, and the [`GsplatRasterizer`] entry
//! point. The native tile-based rasterizer (sort-by-depth → tile bin →
//! alpha-blend) is large (~1k+ LOC) and lands in a follow-up commit;
//! [`GsplatRasterizer::rasterize`] currently returns
//! [`GsplatError::Unsupported`] with a clear pointer.

use candle_core::Tensor;
use thiserror::Error;

/// A batch of 3D Gaussian splats — the standard representation popularised
/// by Kerbl et al. (SIGGRAPH 2023, "3D Gaussian Splatting for Real-Time
/// Radiance Field Rendering").
///
/// All tensors share the same leading `N` dimension (number of
/// splats). All tensors must live on the same device.
#[derive(Debug, Clone)]
pub struct GaussianSplat {
    /// Per-splat means in world space, shape `(N, 3)`.
    pub means: Tensor,
    /// Per-splat log-space scales along the local axes, shape
    /// `(N, 3)`. The actual scale is `exp(scales)`.
    pub scales: Tensor,
    /// Per-splat orientations as unit quaternions in `wxyz` order,
    /// shape `(N, 4)`.
    pub rotations: Tensor,
    /// Per-splat colors. Shape `(N, 3)` for plain RGB or
    /// `(N, sh_dim)` for spherical-harmonics coefficients (the
    /// rasterizer documents which it accepts).
    pub colors: Tensor,
    /// Per-splat scalar opacities in `[0, 1]`, shape `(N, 1)`.
    pub opacities: Tensor,
}

/// Pinhole camera intrinsics in pixel units.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Focal length along the x axis (pixels).
    pub fx: f32,
    /// Focal length along the y axis (pixels).
    pub fy: f32,
    /// Principal point x coordinate (pixels).
    pub cx: f32,
    /// Principal point y coordinate (pixels).
    pub cy: f32,
    /// Output image width (pixels).
    pub width: u32,
    /// Output image height (pixels).
    pub height: u32,
}

/// Rigid-body world-to-camera transform.
#[derive(Debug, Clone)]
pub struct CameraExtrinsics {
    /// 4x4 world-to-camera matrix as a candle tensor of shape `(4, 4)`
    /// in row-major form. Caller is responsible for ensuring the
    /// rotation block is orthonormal.
    pub world_to_cam: Tensor,
}

/// Gaussian-splat rasterizer entry point.
///
/// Stateless wrapper around the (future) native tile-based renderer.
pub struct GsplatRasterizer;

impl GsplatRasterizer {
    /// Render the given splat cloud through the supplied pinhole
    /// camera into an RGB image tensor of shape `(3, height, width)`
    /// with f32 values in `[0, 1]`.
    ///
    /// The native rasterizer is not yet implemented — see
    /// [`GsplatError::Unsupported`].
    pub fn rasterize(
        splats: &GaussianSplat,
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
    ) -> Result<Tensor, GsplatError> {
        // Cheap shape validation so callers wiring this up still get a
        // useful error when they hand in the wrong shapes — the
        // Unsupported branch comes after.
        let (n, three) = splats.means.dims2()?;
        if three != 3 {
            return Err(GsplatError::InvalidShape(format!(
                "splats.means last dim must be 3, got {three}"
            )));
        }
        if splats.scales.dims2()? != (n, 3) {
            return Err(GsplatError::InvalidShape(format!(
                "splats.scales must be (N={n}, 3), got {:?}",
                splats.scales.dims()
            )));
        }
        if splats.rotations.dims2()? != (n, 4) {
            return Err(GsplatError::InvalidShape(format!(
                "splats.rotations must be (N={n}, 4), got {:?}",
                splats.rotations.dims()
            )));
        }
        if splats.opacities.dims2()? != (n, 1) {
            return Err(GsplatError::InvalidShape(format!(
                "splats.opacities must be (N={n}, 1), got {:?}",
                splats.opacities.dims()
            )));
        }
        if splats.colors.dims().len() != 2 || splats.colors.dim(0)? != n {
            return Err(GsplatError::InvalidShape(format!(
                "splats.colors must be (N={n}, C), got {:?}",
                splats.colors.dims()
            )));
        }
        if extrinsics.world_to_cam.dims() != [4, 4] {
            return Err(GsplatError::InvalidShape(format!(
                "extrinsics.world_to_cam must be (4, 4), got {:?}",
                extrinsics.world_to_cam.dims()
            )));
        }
        if intrinsics.width == 0 || intrinsics.height == 0 {
            return Err(GsplatError::InvalidShape(
                "intrinsics.width and intrinsics.height must be > 0".to_string(),
            ));
        }

        Err(GsplatError::Unsupported(
            "native gsplat rasterizer landing in follow-up commit; see \
             blazen-3d-core/src/gsplat.rs for the public API contract"
                .into(),
        ))
    }
}

/// Errors raised by [`GsplatRasterizer`] (and shape-validation on the
/// associated splat / camera structs).
#[derive(Debug, Error)]
pub enum GsplatError {
    /// The native rasterizer is not yet wired up. The error message
    /// names the file containing the public API contract that the
    /// follow-up commit will fill in.
    #[error("gsplat operation not supported yet: {0}")]
    Unsupported(String),
    /// A splat / camera tensor had the wrong rank or mismatched
    /// dimensions.
    #[error("invalid gsplat shape: {0}")]
    InvalidShape(String),
    /// A candle tensor operation failed during shape inspection.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_splat(n: usize, dev: &Device) -> GaussianSplat {
        GaussianSplat {
            means: Tensor::zeros((n, 3), candle_core::DType::F32, dev).unwrap(),
            scales: Tensor::zeros((n, 3), candle_core::DType::F32, dev).unwrap(),
            rotations: Tensor::zeros((n, 4), candle_core::DType::F32, dev).unwrap(),
            colors: Tensor::zeros((n, 3), candle_core::DType::F32, dev).unwrap(),
            opacities: Tensor::zeros((n, 1), candle_core::DType::F32, dev).unwrap(),
        }
    }

    #[test]
    fn rasterize_returns_unsupported_with_actionable_message() {
        let dev = Device::Cpu;
        let splats = make_splat(4, &dev);
        let intrinsics = CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 256.0,
            cy: 256.0,
            width: 512,
            height: 512,
        };
        let extrinsics = CameraExtrinsics {
            world_to_cam: Tensor::zeros((4, 4), candle_core::DType::F32, &dev).unwrap(),
        };
        let err = GsplatRasterizer::rasterize(&splats, &intrinsics, &extrinsics).unwrap_err();
        match err {
            GsplatError::Unsupported(msg) => {
                assert!(
                    msg.contains("follow-up commit"),
                    "message should point at the follow-up commit, got: {msg}"
                );
                assert!(
                    msg.contains("blazen-3d-core/src/gsplat.rs"),
                    "message should name the API contract file, got: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
