//! Triplane feature decoder primitives.
//!
//! Triplane decoders represent a 3D feature volume as three orthogonal
//! 2D feature planes — one for each cardinal plane (XY, YZ, XZ). At
//! query time, a 3D point `(x, y, z) ∈ [-1, 1]³` is projected onto each
//! plane, the corresponding plane is sampled, and the three per-plane
//! feature vectors are concatenated into the final per-point feature
//! consumed by a downstream MLP (typically an NeRF-style density/color
//! head).
//!
//! This is the representation used by EG3D / TripoSR / Stable
//! Zero123-XL-Triplane and many subsequent ports.
//!
//! # Sampling notes
//!
//! Candle 0.10 does not expose a `grid_sample` analogue, so the
//! implementation here uses nearest-neighbour sampling. Bilinear is a
//! follow-up — see the doc comment on
//! [`Triplane::sample`]. For TripoSR-class resolutions
//! (`64²`–`128²` planes) the nearest-neighbour error is small enough
//! to validate the rest of the pipeline against the reference
//! implementation, but production runs should land bilinear sampling
//! before declaring the backend reference-accurate.

use candle_core::{IndexOp, Tensor};
use thiserror::Error;

/// A triplane feature container: three orthogonal feature planes
/// covering the XY, YZ and XZ axes of the unit cube.
///
/// Each plane has shape `(1, channels_per_plane, resolution,
/// resolution)`. The order is fixed as `[xy, yz, xz]` so consumers can
/// assume a stable concatenation order.
#[derive(Debug, Clone)]
pub struct Triplane {
    /// Per-plane feature tensors of shape
    /// `(1, channels_per_plane, resolution, resolution)`.
    ///
    /// Order: `[xy, yz, xz]`.
    planes: [Tensor; 3],
    channels_per_plane: usize,
    resolution: usize,
}

impl Triplane {
    /// Build a [`Triplane`] from three feature planes.
    ///
    /// All three planes must share the same shape
    /// `(1, channels_per_plane, resolution, resolution)`.
    pub fn new(planes: [Tensor; 3]) -> Result<Self, TriplaneError> {
        let (b0, c0, h0, w0) = planes[0].dims4().map_err(|e| {
            TriplaneError::InvalidShape(format!("plane[0] must be 4-D (1, C, H, W); got {e}"))
        })?;
        if b0 != 1 {
            return Err(TriplaneError::InvalidShape(format!(
                "plane[0] batch dim must be 1, got {b0}"
            )));
        }
        if h0 != w0 {
            return Err(TriplaneError::InvalidShape(format!(
                "plane[0] must be square; got {h0}x{w0}"
            )));
        }
        for (i, plane) in planes.iter().enumerate().skip(1) {
            let (b, c, h, w) = plane.dims4().map_err(|e| {
                TriplaneError::InvalidShape(format!("plane[{i}] must be 4-D (1, C, H, W); got {e}"))
            })?;
            if (b, c, h, w) != (b0, c0, h0, w0) {
                return Err(TriplaneError::InvalidShape(format!(
                    "plane[{i}] shape ({b}, {c}, {h}, {w}) does not match plane[0] \
                     ({b0}, {c0}, {h0}, {w0})"
                )));
            }
        }
        Ok(Self {
            planes,
            channels_per_plane: c0,
            resolution: h0,
        })
    }

    /// Sample per-point features at `points: (N, 3)` in `[-1, 1]³`.
    ///
    /// Returns a tensor of shape
    /// `(N, channels_per_plane * 3)` — the three per-plane feature
    /// vectors concatenated along the channel axis in `[xy, yz, xz]`
    /// order.
    ///
    /// Points are projected onto each plane as follows:
    /// - XY plane uses `(x, y)`.
    /// - YZ plane uses `(y, z)`.
    /// - XZ plane uses `(x, z)`.
    ///
    /// The plane coordinates are linearly remapped from `[-1, 1]` to
    /// `[0, resolution - 1]` and rounded to the nearest integer
    /// (nearest-neighbour sampling). Out-of-range coordinates are
    /// clamped to the plane bounds. Bilinear interpolation is a
    /// follow-up; see the module-level documentation.
    // `xy_stack` / `yz_stack` / `xz_stack` (and the matching `feats_*`
    // buffers) are intentional terms of art — they name the three
    // canonical triplane projections. Renaming them to satisfy
    // `clippy::similar_names` would obscure the geometry; allow the
    // lint here only.
    #[allow(clippy::similar_names)]
    pub fn sample(&self, points: &Tensor) -> Result<Tensor, TriplaneError> {
        let (n, three) = points.dims2().map_err(|e| {
            TriplaneError::InvalidShape(format!("points must be 2-D (N, 3); got {e}"))
        })?;
        if three != 3 {
            return Err(TriplaneError::InvalidShape(format!(
                "points last dim must be 3, got {three}"
            )));
        }

        let device = points.device();
        let pts = points.to_vec2::<f32>()?;
        let res = self.resolution;
        #[allow(clippy::cast_precision_loss)]
        let res_minus_one = (res as f32 - 1.0).max(0.0);

        // Compute integer plane coordinates for all N points.
        // For each plane we gather a (N, channels_per_plane) slice and
        // concatenate them at the end.
        let mut feats_axial = Vec::with_capacity(n);
        let mut feats_yz = Vec::with_capacity(n);
        let mut feats_lateral = Vec::with_capacity(n);

        // Squeeze the leading batch dim once so per-point indexing is
        // straightforward; resulting shape is (C, H, W).
        let xy = self.planes[0].i(0)?;
        let yz = self.planes[1].i(0)?;
        let xz = self.planes[2].i(0)?;

        let to_idx = |coord: f32| -> usize {
            // Remap [-1, 1] → [0, res - 1], clamp, then nearest-neighbour.
            let mapped = (coord + 1.0) * 0.5 * res_minus_one;
            let rounded = mapped.round();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let idx = rounded.clamp(0.0, res_minus_one) as usize;
            idx
        };

        for p in &pts {
            let (x, y, z) = (p[0], p[1], p[2]);
            let (ix, iy, iz) = (to_idx(x), to_idx(y), to_idx(z));
            // candle index order on (C, H, W): take (.., row, col) with
            // row = y/z (vertical) and col = x/y (horizontal).
            feats_axial.push(xy.i((.., iy, ix))?);
            feats_yz.push(yz.i((.., iz, iy))?);
            feats_lateral.push(xz.i((.., iz, ix))?);
        }

        let xy_stack = Tensor::stack(&feats_axial, 0)?;
        let yz_stack = Tensor::stack(&feats_yz, 0)?;
        let xz_stack = Tensor::stack(&feats_lateral, 0)?;
        let out = Tensor::cat(&[&xy_stack, &yz_stack, &xz_stack], 1)?;
        // Make sure the output lives on the same device the caller's
        // points came from; stack/cat already does this in candle but
        // we re-anchor defensively.
        let out = out.to_device(device)?;
        Ok(out)
    }

    /// Number of feature channels stored in a single plane.
    ///
    /// The concatenated per-point output of [`Triplane::sample`] has
    /// `channels_per_plane() * 3` channels.
    #[must_use]
    pub fn channels_per_plane(&self) -> usize {
        self.channels_per_plane
    }

    /// Edge length of each square feature plane (in feature pixels).
    #[must_use]
    pub fn resolution(&self) -> usize {
        self.resolution
    }
}

/// Errors raised by [`Triplane`] construction and sampling.
#[derive(Debug, Error)]
pub enum TriplaneError {
    /// A plane tensor had the wrong rank, mismatched dimensions, a
    /// non-unit batch, or non-square spatial dims; or `points` was not
    /// `(N, 3)`.
    #[error("invalid triplane shape: {0}")]
    InvalidShape(String),
    /// A candle tensor operation (slice / stack / cat / device move)
    /// failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_plane(channels: usize, res: usize, dev: &Device) -> Tensor {
        // Deterministic ramp so test results are reproducible without
        // pulling rand into a dev-dep.
        let total = channels * res * res;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        Tensor::from_vec(data, (1, channels, res, res), dev).unwrap()
    }

    #[test]
    fn sample_shape_matches_query() {
        let dev = Device::Cpu;
        let channels = 4;
        let res = 8;
        let planes = [
            make_plane(channels, res, &dev),
            make_plane(channels, res, &dev),
            make_plane(channels, res, &dev),
        ];
        let tri = Triplane::new(planes).unwrap();
        assert_eq!(tri.channels_per_plane(), channels);
        assert_eq!(tri.resolution(), res);

        // Five points spread across [-1, 1]^3 (corners + center).
        let pts = Tensor::from_vec(
            vec![
                -1.0_f32, -1.0, -1.0, //
                1.0, 1.0, 1.0, //
                0.0, 0.0, 0.0, //
                -0.5, 0.25, 0.75, //
                0.5, -0.25, -0.75,
            ],
            (5, 3),
            &dev,
        )
        .unwrap();

        let out = tri.sample(&pts).unwrap();
        assert_eq!(out.dims(), &[5, channels * 3]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn new_rejects_mismatched_planes() {
        let dev = Device::Cpu;
        let res = 4;
        let planes = [
            make_plane(4, res, &dev),
            make_plane(4, res, &dev),
            make_plane(5, res, &dev), // channel mismatch
        ];
        let err = Triplane::new(planes).unwrap_err();
        assert!(matches!(err, TriplaneError::InvalidShape(_)));
    }
}
