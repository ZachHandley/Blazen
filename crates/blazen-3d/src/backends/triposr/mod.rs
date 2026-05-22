//! `TripoSR` single-image-to-3D backend.
//!
//! # Status (May 2026)
//!
//! Wave T.1 is **scaffolding only**: this module exposes
//! [`TripoSrBackend`] with a correct constructor and a stub
//! [`TripoSrBackend::generate_3d`] inherent method that surfaces a
//! "not yet implemented" error. Wave T.2 lands the real
//! `DINOv2`-base image encoder, triplane-output transformer decoder,
//! `NeRF`/SDF field MLP sampled from triplane features, marching-cubes
//! mesh extraction, `HF` weight loading + safetensors remap, and
//! end-to-end orchestration (image -> `DINOv2` -> triplane -> `NeRF`
//! -> marching cubes -> GLB).
//!
//! Upstream: <https://github.com/VAST-AI-Research/TripoSR> (MIT,
//! VAST-AI-Research + Stability AI). Architecture: `DINOv2` image
//! encoder -> triplane transformer decoder -> `NeRF`/SDF field ->
//! marching cubes mesh. Output is a vertex-colored GLB.
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
//! To break the cycle without giving up the trait surface, Wave T.1
//! defines [`TripoSrBackend`] **here** as a plain struct with an
//! inherent `generate_3d` stub, and Wave T.2 will ship the
//! `impl ThreeDGeneration for TripoSrBackend` block inside
//! `blazen-llm` (which is free to write the impl because it already
//! has `blazen-3d` as an optional dep through the `threed` feature).
//! No code needs to move at that point -- Wave T.2 just lands the impl
//! on the consumer side.
//!
//! # Feature gating
//!
//! The backend is gated on the `triposr` cargo feature, which is OFF
//! by default. Enabling `triposr` pulls in `blazen-3d-core` (shared
//! 3D primitives such as marching cubes) and `candle-core` (the tensor
//! runtime used by every native Blazen ML backend).

mod image_encoder;
mod marching_cubes;
mod nerf_field;
mod pipeline;
mod triplane_transformer;
mod weights;

/// Long-form message returned by the `TripoSR` `generate_3d` stub
/// during Wave T.1 scaffolding. Wave T.2 replaces this with the real
/// `DINOv2` + triplane transformer + `NeRF` + marching-cubes pipeline.
pub(crate) const TRIPOSR_SCAFFOLDING: &str = "TripoSR Wave T.1 scaffolding -- Wave T.2 lands DINOv2 + triplane transformer + NeRF + marching cubes";

/// `TripoSR` single-image-to-3D backend (Wave T.1 scaffolding -- see
/// module docs).
///
/// The `ThreeDGeneration` trait impl for this struct ships in
/// `blazen-llm` during Wave T.2 to avoid a `blazen-llm` <-> `blazen-3d`
/// dependency cycle. Until then, [`TripoSrBackend::generate_3d`] is an
/// inherent stub method that surfaces a clear "not yet implemented"
/// error.
#[derive(Debug, Clone, Default)]
pub struct TripoSrBackend {
    _private: (),
}

impl TripoSrBackend {
    /// Construct a `TripoSR` backend handle. During Wave T.1 every
    /// `generate_3d` call surfaces a stub error; Wave T.2 wires this
    /// up to the real `DINOv2` + triplane + `NeRF` + marching-cubes
    /// pipeline.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }

    /// Stable identifier for this backend, used by routing /
    /// telemetry layers.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        "triposr"
    }

    /// Inherent stub for the eventual
    /// `blazen_llm::compute::traits::ThreeDGeneration::generate_3d`
    /// impl that Wave T.2 ships from `blazen-llm`.
    ///
    /// Returns a `&'static str` error message during Wave T.1. The
    /// return type is intentionally minimal here -- Wave T.2 swaps
    /// this inherent stub for a full trait impl (in `blazen-llm`)
    /// that takes a `ThreeDRequest` and returns a `ThreeDResult` /
    /// `BlazenError`. Keeping the Wave T.1 signature decoupled from
    /// the `blazen-llm` request/result types avoids a fake dep edge
    /// just for the placeholder.
    ///
    /// # Errors
    ///
    /// Always returns [`TRIPOSR_SCAFFOLDING`] during Wave T.1.
    pub fn generate_3d(&self) -> Result<(), &'static str> {
        Err(TRIPOSR_SCAFFOLDING)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triposr_id_is_stable() {
        let backend = TripoSrBackend::new();
        assert_eq!(backend.id(), "triposr");
    }

    #[test]
    fn triposr_generate_3d_returns_scaffolding_error() {
        let backend = TripoSrBackend::new();
        let err = backend.generate_3d().unwrap_err();
        assert!(err.contains("scaffolding"), "err={err}");
        assert!(err.contains("Wave T.2"), "err={err}");
    }
}
