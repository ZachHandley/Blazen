//! Marching-cubes mesh extraction from a 3D scalar field.
//!
//! This is a thin candle-friendly wrapper around the upstream
//! `isosurface` crate's classic marching-cubes implementation. The
//! field is moved to CPU, flattened, then resampled at the `(0, 1)³`
//! unit-cube grid the upstream crate expects; vertices are remapped
//! from `[0, 1]³` to the caller-supplied `bounds` cube on the way out.

use candle_core::Tensor;
use isosurface::marching_cubes::MarchingCubes as IsoMarchingCubes;
use isosurface::source::Source;
use thiserror::Error;

/// Marching-cubes mesh extractor.
///
/// Stateless entry point — see [`MarchingCubes::extract`].
pub struct MarchingCubes;

impl MarchingCubes {
    /// Extract a triangle mesh from a `(resolution, resolution,
    /// resolution)` f32 scalar field.
    ///
    /// `iso_value` separates inside (`field < iso_value`) from outside
    /// (`field > iso_value`). For signed distance fields the natural
    /// choice is `0.0`; for occupancy grids `0.5` is typical.
    ///
    /// `bounds` is `(min, max)` of the destination axis-aligned bounding
    /// box: returned vertex coordinates are remapped from the unit-cube
    /// sampling grid `[0, 1]³` to `[min, max]` component-wise.
    ///
    /// Returns `(vertices, indices)` where each entry in `vertices` is
    /// an `[x, y, z]` triple in world space and `indices` is a flat
    /// triangle list (each consecutive triple is one triangle, CCW
    /// matching the upstream `isosurface` crate's convention).
    pub fn extract(
        field: &Tensor,
        iso_value: f32,
        bounds: ([f32; 3], [f32; 3]),
    ) -> Result<(Vec<[f32; 3]>, Vec<u32>), MarchingCubesError> {
        let dims = field.dims();
        if dims.len() != 3 {
            return Err(MarchingCubesError::InvalidShape(format!(
                "scalar field must be 3-D (D, H, W); got shape {dims:?}"
            )));
        }
        let d = dims[0];
        if dims[1] != d || dims[2] != d {
            return Err(MarchingCubesError::InvalidShape(format!(
                "scalar field must be a cube (D, D, D); got shape {dims:?}"
            )));
        }
        if d < 2 {
            return Err(MarchingCubesError::InvalidShape(format!(
                "scalar field resolution must be >= 2; got {d}"
            )));
        }

        // Move to CPU and flatten into a (D*D*D) row-major buffer.
        let cpu = field.to_device(&candle_core::Device::Cpu)?;
        let flat = cpu.flatten_all()?.to_vec1::<f32>()?;

        let (min, max) = bounds;
        for (axis, name) in [(0, 'x'), (1, 'y'), (2, 'z')] {
            if max[axis] <= min[axis] || !max[axis].is_finite() || !min[axis].is_finite() {
                return Err(MarchingCubesError::InvalidShape(format!(
                    "bounds max[{name}]={} must be finite and > min[{name}]={}",
                    max[axis], min[axis]
                )));
            }
        }

        let source = GridScalarField {
            data: flat,
            size: d,
            iso: iso_value,
        };

        let mut vertices_flat: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut mc = IsoMarchingCubes::new(d);
        mc.extract(&source, &mut vertices_flat, &mut indices);

        if !vertices_flat.len().is_multiple_of(3) {
            return Err(MarchingCubesError::Extraction(format!(
                "isosurface returned vertex buffer of length {} which is not a multiple of 3",
                vertices_flat.len()
            )));
        }

        // Remap from the upstream [0, 1]^3 sampling grid to the
        // caller's world-space bounds.
        let span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let vertices: Vec<[f32; 3]> = vertices_flat
            .chunks_exact(3)
            .map(|v| {
                [
                    min[0] + v[0] * span[0],
                    min[1] + v[1] * span[1],
                    min[2] + v[2] * span[2],
                ]
            })
            .collect();

        Ok((vertices, indices))
    }
}

/// Adapter that lets the `isosurface` crate sample our flattened
/// row-major scalar field at `(0, 1)³` unit-cube coordinates.
struct GridScalarField {
    data: Vec<f32>,
    size: usize,
    iso: f32,
}

impl Source for GridScalarField {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        // Map [0, 1] -> nearest grid index, clamping to the cube. The
        // upstream MarchingCubes only samples at exact grid steps it
        // generates itself, so nearest-neighbour is exact here.
        #[allow(clippy::cast_precision_loss)]
        let res_minus_one = (self.size as f32 - 1.0).max(0.0);
        let to_idx = |c: f32| -> usize {
            let mapped = (c * res_minus_one).round();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let idx = mapped.clamp(0.0, res_minus_one) as usize;
            idx
        };
        let ix = to_idx(x);
        let iy = to_idx(y);
        let iz = to_idx(z);
        // Row-major (D, H, W) — index along axis-0 first.
        let offset = (ix * self.size + iy) * self.size + iz;
        self.data[offset] - self.iso
    }
}

/// Errors raised by [`MarchingCubes::extract`].
#[derive(Debug, Error)]
pub enum MarchingCubesError {
    /// The input tensor was not a 3-D cube, or the supplied bounds were
    /// degenerate.
    #[error("invalid scalar field / bounds: {0}")]
    InvalidShape(String),
    /// A candle tensor operation (device move / flatten / to_vec)
    /// failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// The upstream `isosurface` extractor returned an invalid buffer.
    #[error("marching cubes extraction failed: {0}")]
    Extraction(String),
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn extract_unit_sphere_yields_nonzero_mesh() {
        let dev = Device::Cpu;
        let res = 32;
        // SDF of a sphere of radius 0.6 centered at (0.5, 0.5, 0.5)
        // sampled over the [0, 1]^3 cube.
        let mut data = Vec::with_capacity(res * res * res);
        let res_minus_one = (res as f32 - 1.0).max(1.0);
        for ix in 0..res {
            for iy in 0..res {
                for iz in 0..res {
                    let x = (ix as f32 / res_minus_one) - 0.5;
                    let y = (iy as f32 / res_minus_one) - 0.5;
                    let z = (iz as f32 / res_minus_one) - 0.5;
                    let d = (x * x + y * y + z * z).sqrt() - 0.3;
                    data.push(d);
                }
            }
        }
        let field = Tensor::from_vec(data, (res, res, res), &dev).unwrap();
        let (verts, idx) =
            MarchingCubes::extract(&field, 0.0, ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])).unwrap();
        assert!(
            verts.len() >= 100,
            "expected at least 100 sphere vertices, got {}",
            verts.len()
        );
        assert!(idx.len() % 3 == 0, "indices must be a flat triangle list");
        // Every index must reference a real vertex.
        let max_idx = idx.iter().copied().max().unwrap_or(0) as usize;
        assert!(max_idx < verts.len());
    }

    #[test]
    fn extract_rejects_non_cube_field() {
        let dev = Device::Cpu;
        let data = vec![0.0_f32; 4 * 4 * 8];
        let field = Tensor::from_vec(data, (4, 4, 8), &dev).unwrap();
        let err = MarchingCubes::extract(&field, 0.0, ([0.0; 3], [1.0, 1.0, 1.0])).unwrap_err();
        assert!(matches!(err, MarchingCubesError::InvalidShape(_)));
    }
}
