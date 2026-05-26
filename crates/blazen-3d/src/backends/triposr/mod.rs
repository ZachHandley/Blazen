//! `TripoSR` single-image-to-3D backend.
//!
//! Architecture: `DINOv2` image encoder -> triplane transformer decoder
//! -> `NeRF`/SDF field -> marching cubes mesh. Output is a vertex-colored
//! GLB.
//!
//! Upstream: <https://github.com/VAST-AI-Research/TripoSR> (MIT,
//! VAST-AI-Research + Stability AI).
//!
//! # Cycle resolution: trait impl lives in `blazen-llm`
//!
//! `TripoSR` is conceptually one of the
//! `blazen_llm::compute::traits::ThreeDGeneration` providers (the
//! same trait that the existing fal-proxy
//! [`crate::backends::compat::Compat3dProvider`] flow routes through).
//! Implementing that trait inside `blazen-3d` would, however, require a
//! `blazen-3d -> blazen-llm` dependency edge, and `blazen-llm` already
//! depends on `blazen-3d` via its `threed` feature (see
//! `crates/blazen-llm/Cargo.toml`). That would close a hard dep cycle
//! that cargo refuses to compile.
//!
//! To break the cycle without giving up the trait surface, this module
//! defines [`TripoSrBackend`] **here** as a plain struct wrapping a
//! loaded [`TripoSrPipeline`], and the
//! `impl ThreeDGeneration for TripoSrBackend` block lives inside
//! `blazen-llm` (`crates/blazen-llm/src/backends/triposr.rs`), which is
//! free to write the impl because it already has `blazen-3d` as an
//! optional dep through the `threed` feature.
//!
//! # Feature gating
//!
//! The backend is gated on the `triposr` cargo feature, which is OFF
//! by default. Enabling `triposr` pulls in `blazen-3d-core` (shared
//! 3D primitives such as marching cubes) and `candle-core` (the tensor
//! runtime used by every native Blazen ML backend).

mod image_encoder;
mod nerf_field;
mod pipeline;
mod triplane_transformer;
mod weights;

pub use pipeline::{TripoSrPipeline, TripoSrPipelineError};

use std::path::Path;
use std::sync::Arc;

use candle_core::Device;

/// `TripoSR` single-image-to-3D backend.
///
/// Wraps a loaded [`TripoSrPipeline`] in an [`Arc`] so the backend is
/// cheap to clone and share across the `ThreeDGeneration` trait impl
/// (which lives in `blazen-llm` to avoid a dep cycle -- see module
/// docs).
///
/// Construct via [`TripoSrBackend::load_from_paths`] (synchronous, local
/// weights), [`TripoSrBackend::load_from_hf`] (async, downloads weights
/// from Hugging Face), or [`TripoSrBackend::from_pipeline`] when the
/// caller already holds a [`TripoSrPipeline`].
#[derive(Clone)]
pub struct TripoSrBackend {
    pipeline: Arc<TripoSrPipeline>,
}

impl TripoSrBackend {
    /// Construct from a pre-loaded pipeline.
    ///
    /// Useful when the caller wants to drive the pipeline lifecycle
    /// directly (e.g. tests, custom HF revisions, or hand-built fixtures)
    /// and then wrap the result in the backend surface that the
    /// `ThreeDGeneration` trait impl drives.
    #[must_use]
    pub fn from_pipeline(pipeline: TripoSrPipeline) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
        }
    }

    /// Load synchronously from a local weights file/directory.
    ///
    /// `weights_path` is the directory containing the safetensors
    /// checkpoints for the `DINOv2` encoder, triplane transformer, and
    /// `NeRF` field (see [`TripoSrPipeline::load_from_paths`] for the
    /// exact layout).
    ///
    /// # Errors
    ///
    /// Forwards every [`TripoSrPipelineError`] variant raised by
    /// [`TripoSrPipeline::load_from_paths`].
    pub fn load_from_paths(
        weights_path: &Path,
        device: &Device,
    ) -> Result<Self, TripoSrPipelineError> {
        let pipeline = TripoSrPipeline::load_from_paths(weights_path, device)?;
        Ok(Self::from_pipeline(pipeline))
    }

    /// Download `TripoSR` weights from Hugging Face and load the
    /// backend.
    ///
    /// `hf_repo_id` is the HF repo (e.g. `"stabilityai/TripoSR"`).
    /// `revision` pins a specific git revision / tag / branch on the
    /// repo; `None` defaults to the repo's `main` branch.
    ///
    /// # Errors
    ///
    /// Forwards every [`TripoSrPipelineError`] variant raised by
    /// [`TripoSrPipeline::load_from_hf`].
    pub async fn load_from_hf(
        hf_repo_id: &str,
        revision: Option<&str>,
        device: &Device,
    ) -> Result<Self, TripoSrPipelineError> {
        let pipeline = TripoSrPipeline::load_from_hf(hf_repo_id, revision, device).await?;
        Ok(Self::from_pipeline(pipeline))
    }

    /// Stable identifier for this backend, used by routing /
    /// telemetry layers.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        "triposr"
    }

    /// Direct access to the loaded pipeline for callers that don't go
    /// through the `ThreeDGeneration` trait surface (e.g. low-level
    /// integration tests, custom orchestrators).
    #[must_use]
    pub fn pipeline(&self) -> &TripoSrPipeline {
        &self.pipeline
    }
}

impl std::fmt::Debug for TripoSrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TripoSrBackend")
            .field("id", &"triposr")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end identity check: load real weights and confirm the
    /// backend's stable id. Requires `TripoSR` weights to be available
    /// from Hugging Face, so it's ignored by default and opt-in via
    /// `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore = "requires TripoSR weights from Hugging Face"]
    async fn triposr_id_is_stable_with_real_weights() {
        let device = Device::Cpu;
        let backend = TripoSrBackend::load_from_hf("stabilityai/TripoSR", None, &device)
            .await
            .expect("load TripoSR weights");
        assert_eq!(backend.id(), "triposr");
    }
}
