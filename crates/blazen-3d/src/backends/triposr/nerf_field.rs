//! `NeRF`/SDF field MLP sampled from triplane features (Wave T.2).
//!
//! Mirrors the `NeRFMLP` decoder from `TripoSR`
//! (<https://github.com/VAST-AI-Research/TripoSR/blob/main/tsr/models/network_utils.py>).
//! For each query point `(x, y, z) ∈ [-1, 1]³` we:
//!
//! 1. Sample the [`Triplane`] via `triplane.sample(points)` — concatenating
//!    the three per-plane feature vectors into a single
//!    `(N, channels_per_plane * 3)` feature tensor.
//! 2. Forward through a small MLP: an initial `Linear -> ReLU`, followed
//!    by `num_layers - 2` hidden `Linear -> ReLU` blocks, terminated by a
//!    `Linear(hidden_dim, 4)` head.
//! 3. Split the 4-channel head into 1 density channel + 3 color channels.
//! 4. Apply the configured [`DensityActivation`] to density and `sigmoid`
//!    to color.
//!
//! For mesh extraction the [`TripoSrNerfField::density_grid`] helper
//! bakes the density field on a regular `(resolution, resolution,
//! resolution)` grid covering `[-1, 1]³`. The grid layout matches the
//! `(D, H, W)` row-major convention consumed by
//! [`blazen_3d_core::marching_cubes::MarchingCubes::extract`] — i.e.
//! `offset = (ix * D + iy) * D + iz` with `x` slowest and `z` fastest.

// Wave T.2 lands each TripoSR sub-module ahead of the consuming
// `pipeline.rs` rewrite (Wave T.3). Until pipeline.rs picks these types
// up, the public surface here looks dead to rustc / clippy.
#![allow(dead_code)]
// `VarBuilder` is the candle convention for module construction —
// every existing backend takes it by value and the move is a cheap
// `Arc` clone. Matches `triplane_transformer.rs` and the candle
// reference style.
#![allow(clippy::needless_pass_by_value)]
// `usize -> f32` casts here are bounded by config / grid resolutions
// that never approach `2^23`.
#![allow(clippy::cast_precision_loss)]

use blazen_3d_core::triplane::{Triplane, TriplaneError};
use candle_core::{IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear};
use thiserror::Error;

/// Density activation applied to the 1-channel density head output.
///
/// `TripoSR`'s upstream config defaults to `trunc_exp` (truncated
/// exponential) with `density_bias = -1.0`. We keep the plain
/// `Relu` variant available too because the smaller reference
/// configs in the upstream test suite use `ReLU`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityActivation {
    /// Standard rectified linear unit: `max(0, x)`.
    Relu,
    /// Truncated exponential: `exp(min(x, 15))`. Prevents the
    /// unbounded growth that pure `exp` produces during early
    /// training while keeping the smooth, strictly-positive density
    /// shape that `NeRFs` rely on.
    TruncExp,
}

/// Hyperparameters for [`TripoSrNerfField`].
///
/// The defaults returned by [`NerfFieldConfig::base_default`] match the
/// upstream `TripoSR`-base release.
#[derive(Debug, Clone, Copy)]
pub struct NerfFieldConfig {
    /// Channels stored in **one** triplane (the MLP input width is
    /// `3 * triplane_channels_per_plane`).
    pub triplane_channels_per_plane: usize,
    /// Hidden width of every internal MLP layer.
    pub hidden_dim: usize,
    /// Total number of `Linear` layers, counting the input projection
    /// and the final 4-channel head. Must be `>= 2`.
    pub num_layers: usize,
    /// Activation applied to the 1-channel density output.
    pub density_activation: DensityActivation,
}

impl NerfFieldConfig {
    /// `TripoSR`-base default hyperparameters.
    ///
    /// - `triplane_channels_per_plane = 40`
    /// - `hidden_dim = 64`
    /// - `num_layers = 8`
    /// - `density_activation = TruncExp`
    #[must_use]
    pub const fn base_default() -> Self {
        Self {
            triplane_channels_per_plane: 40,
            hidden_dim: 64,
            num_layers: 8,
            density_activation: DensityActivation::TruncExp,
        }
    }
}

/// Result of [`TripoSrNerfField::query`].
///
/// `density` has shape `(N, 1)` after the configured
/// [`DensityActivation`]; `color` has shape `(N, 3)` after sigmoid.
#[derive(Debug)]
pub struct NerfFieldOutput {
    /// Per-point density / occupancy with shape `(N, 1)`.
    pub density: Tensor,
    /// Per-point RGB color with shape `(N, 3)`, each component in
    /// `[0, 1]` thanks to sigmoid.
    pub color: Tensor,
}

/// NeRF/SDF field MLP that consumes a [`Triplane`] and emits per-point
/// `(density, color)` pairs.
///
/// See module docs for the layer recipe and the `density_grid` layout
/// contract.
#[derive(Debug)]
pub struct TripoSrNerfField {
    layers: Vec<Linear>,
    config: NerfFieldConfig,
}

impl TripoSrNerfField {
    /// Construct the MLP by reading every learned tensor from `vb`.
    ///
    /// Sub-paths used:
    ///
    /// - `layers.0` -> `Linear(3 * channels_per_plane -> hidden_dim)`
    /// - `layers.<i>` (for `1 <= i < num_layers - 1`) ->
    ///   `Linear(hidden_dim -> hidden_dim)`
    /// - `layers.<num_layers - 1>` -> `Linear(hidden_dim -> 4)` (1
    ///   density + 3 color channels)
    ///
    /// # Errors
    ///
    /// Returns [`NerfFieldError::InvalidShape`] when `num_layers < 2`,
    /// or [`NerfFieldError::Candle`] when a required tensor is missing
    /// or has the wrong shape.
    pub fn load_from_var_builder(
        vb: VarBuilder,
        config: NerfFieldConfig,
    ) -> Result<Self, NerfFieldError> {
        if config.num_layers < 2 {
            return Err(NerfFieldError::InvalidShape(format!(
                "num_layers must be >= 2 (input projection + output head), got {}",
                config.num_layers
            )));
        }
        let input_dim = config.triplane_channels_per_plane * 3;
        let mut layers = Vec::with_capacity(config.num_layers);
        // Input projection: (3 * channels_per_plane) -> hidden_dim.
        layers.push(linear(input_dim, config.hidden_dim, vb.pp("layers.0"))?);
        // Hidden layers: (num_layers - 2) of hidden_dim -> hidden_dim.
        for i in 1..(config.num_layers - 1) {
            layers.push(linear(
                config.hidden_dim,
                config.hidden_dim,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        // Final head: hidden_dim -> 4 (density + RGB).
        let last_idx = config.num_layers - 1;
        layers.push(linear(
            config.hidden_dim,
            4,
            vb.pp(format!("layers.{last_idx}")),
        )?);

        Ok(Self { layers, config })
    }

    /// Given a [`Triplane`] and a `(N, 3)` tensor of query points in
    /// `[-1, 1]³`, return per-point `(density, color)`.
    ///
    /// # Errors
    ///
    /// Returns [`NerfFieldError::Core3d`] when [`Triplane::sample`]
    /// fails (shape mismatch on the input planes / points), or
    /// [`NerfFieldError::Candle`] when an MLP forward fails.
    pub fn query(
        &self,
        triplane: &Triplane,
        points: &Tensor,
    ) -> Result<NerfFieldOutput, NerfFieldError> {
        // 1. Sample triplane: (N, 3 * channels_per_plane).
        let feats = triplane.sample(points)?;

        // 2. MLP forward: ReLU between every layer except after the
        // final head.
        let last = self.layers.len() - 1;
        let mut x = feats;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i != last {
                x = x.relu()?;
            }
        }
        // x is now (N, 4).

        // 3. Split density (channel 0) from color (channels 1..4).
        let density_raw = x.i((.., 0..1))?;
        let color_raw = x.i((.., 1..4))?;

        // 4. Activations.
        let density = match self.config.density_activation {
            DensityActivation::Relu => density_raw.relu()?,
            DensityActivation::TruncExp => {
                // exp(min(x, 15)) — clamp the upper tail to avoid
                // overflow during early training. candle exposes
                // element-wise `minimum` via `Tensor::minimum`.
                let cap = Tensor::full(15.0_f32, density_raw.shape(), density_raw.device())?;
                let clamped = density_raw.minimum(&cap)?;
                clamped.exp()?
            }
        };
        let color = candle_nn::ops::sigmoid(&color_raw)?;

        Ok(NerfFieldOutput { density, color })
    }

    /// Bake the post-activation density field on a regular
    /// `(resolution, resolution, resolution)` grid that spans
    /// `[-1, 1]³`.
    ///
    /// Grid layout matches the `(D, H, W)` row-major convention
    /// expected by
    /// [`blazen_3d_core::marching_cubes::MarchingCubes::extract`]:
    /// `offset = (ix * resolution + iy) * resolution + iz`, with `x`
    /// slowest, `y` middle, `z` fastest. Output shape is
    /// `(resolution, resolution, resolution)`.
    ///
    /// Points are processed in batches of at most 262 144 to keep
    /// peak intermediate-tensor allocation bounded on dense grids
    /// (e.g. `resolution = 128` -> 2 097 152 points).
    ///
    /// # Errors
    ///
    /// Returns [`NerfFieldError::InvalidShape`] when `resolution < 2`,
    /// or propagates any [`NerfFieldError`] from [`Self::query`].
    pub fn density_grid(
        &self,
        triplane: &Triplane,
        resolution: usize,
    ) -> Result<Tensor, NerfFieldError> {
        // Batch queries to keep peak intermediates bounded; 256 K
        // points * 120-ch features * 4 bytes ≈ 122 MB worst case at
        // the default `triplane_channels_per_plane = 40`.
        const BATCH_POINTS: usize = 256 * 1024;

        if resolution < 2 {
            return Err(NerfFieldError::InvalidShape(format!(
                "density grid resolution must be >= 2, got {resolution}"
            )));
        }
        // `Triplane::sample` reads its planes via `to_vec2::<f32>()`
        // (a CPU round-trip) regardless of where the planes were
        // stored, and re-anchors its output on `points.device()`.
        // Building the query points on CPU is therefore the cheapest
        // path: the sampler stays on CPU and no host-device copies
        // happen on the density-grid hot loop.
        let device = candle_core::Device::Cpu;

        // Build the full (resolution^3, 3) point cloud in row-major
        // (x, y, z) order to match marching_cubes' offset formula:
        //   offset = (ix * D + iy) * D + iz   (x slowest, z fastest).
        let total = resolution
            .checked_mul(resolution)
            .and_then(|hw| hw.checked_mul(resolution))
            .ok_or_else(|| {
                NerfFieldError::InvalidShape(format!(
                    "density grid resolution {resolution} overflows usize when cubed"
                ))
            })?;
        let res_minus_one = (resolution as f32 - 1.0).max(1.0);
        let mut coords: Vec<f32> = Vec::with_capacity(total * 3);
        for ix in 0..resolution {
            let x = -1.0 + 2.0 * (ix as f32) / res_minus_one;
            for iy in 0..resolution {
                let y = -1.0 + 2.0 * (iy as f32) / res_minus_one;
                for iz in 0..resolution {
                    let z = -1.0 + 2.0 * (iz as f32) / res_minus_one;
                    coords.push(x);
                    coords.push(y);
                    coords.push(z);
                }
            }
        }

        let mut density_chunks: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < total {
            let end = (start + BATCH_POINTS).min(total);
            let n = end - start;
            let slice = &coords[start * 3..end * 3];
            let points = Tensor::from_slice(slice, (n, 3), &device)?;
            let out = self.query(triplane, &points)?;
            // out.density is (n, 1) — squeeze to (n,) for cleaner concat.
            density_chunks.push(out.density.squeeze(1)?);
            start = end;
        }

        let flat = if density_chunks.len() == 1 {
            density_chunks
                .pop()
                .expect("len() == 1 implies pop() succeeds")
        } else {
            Tensor::cat(&density_chunks, 0)?
        };
        let grid = flat.reshape((resolution, resolution, resolution))?;
        Ok(grid)
    }
}

/// Errors raised by [`TripoSrNerfField`] construction and forward
/// passes.
#[derive(Debug, Error)]
pub enum NerfFieldError {
    /// A tensor had the wrong rank / dimension, or the config was
    /// invalid (e.g. `num_layers < 2`, `resolution < 2`).
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    /// A candle tensor / module operation failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// A [`Triplane`] sampling / construction step failed.
    #[error("blazen-3d-core error: {0}")]
    Core3d(String),
}

impl From<TriplaneError> for NerfFieldError {
    fn from(err: TriplaneError) -> Self {
        Self::Core3d(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn zero_triplane(channels: usize, resolution: usize, dev: &Device) -> Triplane {
        let plane = Tensor::zeros((1, channels, resolution, resolution), DType::F32, dev).unwrap();
        Triplane::new([plane.clone(), plane.clone(), plane]).unwrap()
    }

    #[test]
    fn nerf_field_query_shapes() {
        let dev = Device::Cpu;
        let cfg = NerfFieldConfig {
            triplane_channels_per_plane: 8,
            hidden_dim: 16,
            num_layers: 3,
            density_activation: DensityActivation::Relu,
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let field = TripoSrNerfField::load_from_var_builder(vb, cfg)
            .expect("nerf field build from zero-init VarBuilder");

        let triplane = zero_triplane(cfg.triplane_channels_per_plane, 4, &dev);

        // 10 query points spread across [-1, 1]^3.
        let mut coords: Vec<f32> = Vec::with_capacity(10 * 3);
        for i in 0..10 {
            let t = (i as f32) / 9.0;
            let v = -1.0 + 2.0 * t;
            coords.push(v);
            coords.push(-v);
            coords.push(v * 0.5);
        }
        let points = Tensor::from_vec(coords, (10, 3), &dev).unwrap();

        let out = field
            .query(&triplane, &points)
            .expect("query should succeed with matching shapes");

        assert_eq!(out.density.dims(), &[10, 1]);
        assert_eq!(out.color.dims(), &[10, 3]);
        assert_eq!(out.density.dtype(), DType::F32);
        assert_eq!(out.color.dtype(), DType::F32);
    }

    #[test]
    fn density_grid_shape() {
        let dev = Device::Cpu;
        let cfg = NerfFieldConfig {
            triplane_channels_per_plane: 8,
            hidden_dim: 16,
            num_layers: 3,
            density_activation: DensityActivation::Relu,
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let field = TripoSrNerfField::load_from_var_builder(vb, cfg)
            .expect("nerf field build from zero-init VarBuilder");

        let triplane = zero_triplane(cfg.triplane_channels_per_plane, 4, &dev);
        let grid = field
            .density_grid(&triplane, 16)
            .expect("density_grid should succeed at modest resolution");

        assert_eq!(grid.dims(), &[16, 16, 16]);
        assert_eq!(grid.dtype(), DType::F32);
    }

    #[test]
    fn config_base_default_matches_reference() {
        let cfg = NerfFieldConfig::base_default();
        assert_eq!(cfg.triplane_channels_per_plane, 40);
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.density_activation, DensityActivation::TruncExp);
    }

    #[test]
    fn trunc_exp_activation_runs() {
        let dev = Device::Cpu;
        let cfg = NerfFieldConfig {
            triplane_channels_per_plane: 4,
            hidden_dim: 8,
            num_layers: 2,
            density_activation: DensityActivation::TruncExp,
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let field = TripoSrNerfField::load_from_var_builder(vb, cfg).unwrap();
        let triplane = zero_triplane(cfg.triplane_channels_per_plane, 4, &dev);
        let points = Tensor::zeros((3, 3), DType::F32, &dev).unwrap();
        let out = field.query(&triplane, &points).unwrap();
        // With zero weights everywhere the pre-activation density is
        // 0, so trunc_exp(0) = 1. Just check it ran and stayed finite.
        let vals = out.density.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!(v.is_finite(), "trunc_exp output should be finite, got {v}");
            assert!((v - 1.0).abs() < 1e-5, "exp(0) should be 1.0, got {v}");
        }
    }
}
