//! The [`Refiner3dBackend`] trait — capability for mesh-cleanup
//! passes (decimation, hole-filling, UV unwrapping, retopology,
//! smoothing).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::Refiner3dError;

/// Capability trait for engines that refine a 3D mesh: decimate, fill
/// holes (poisson reconstruction), unwrap UVs, retopologize, smooth.
///
/// The exact passes applied to a single call depend on the request
/// flags — a refiner is not required to perform every operation, and
/// backends may short-circuit no-op passes.
///
/// Implementors should be cheap to clone (typically wrap any expensive
/// state in `Arc` internally) and safe to share across tokio tasks.
#[async_trait]
pub trait Refiner3dBackend: Send + Sync {
    /// Refine a 3D mesh: decimate, fill holes (poisson
    /// reconstruction), unwrap UVs, retopologize, smooth. The exact
    /// passes applied depend on the request flags.
    ///
    /// # Input format
    ///
    /// `mesh_glb` is GLB (`model/gltf-binary`) bytes.
    ///
    /// # Output format
    ///
    /// The returned [`RefineResult::refined_glb`] is GLB bytes with
    /// the requested passes applied.
    /// [`RefineResult::stats`] reports the before/after triangle
    /// counts and (when UV unwrapping ran) the produced chart count.
    ///
    /// # Errors
    ///
    /// Returns [`Refiner3dError::EngineNotAvailable`] when the backend
    /// was built without the required engine feature;
    /// [`Refiner3dError::InvalidInput`] when `mesh_glb` is malformed;
    /// [`Refiner3dError::Backend`] on processing-time failures; and
    /// [`Refiner3dError::Unsupported`] when the active backend cannot
    /// perform one of the requested passes (e.g. retopology on a
    /// decimation-only refiner).
    async fn refine(
        &self,
        mesh_glb: &[u8],
        request: RefineRequest,
    ) -> Result<RefineResult, Refiner3dError>;
}

/// Request parameters for [`Refiner3dBackend::refine`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineRequest {
    /// If `Some`, decimate the mesh towards this target triangle
    /// count. Backends may interpret the target as a soft hint rather
    /// than a hard limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimate_target_tris: Option<u32>,
    /// `true` to fill holes in the mesh (typically via screened
    /// poisson reconstruction).
    pub fill_holes: bool,
    /// `true` to compute a new UV unwrap for the mesh, replacing any
    /// existing UV channel.
    pub unwrap_uvs: bool,
    /// `true` to retopologize the mesh (rebuild the polygon layout
    /// for animation-friendly edge flow). Typically slower than the
    /// other passes.
    pub retopologize: bool,
    /// Optional Laplacian / Taubin smoothing iteration count. `None`
    /// or `Some(0)` skips smoothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_iterations: Option<u32>,
}

/// Result of a successful [`Refiner3dBackend::refine`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineResult {
    /// GLB bytes with the requested refinement passes applied.
    pub refined_glb: Vec<u8>,
    /// MIME type of [`Self::refined_glb`]; always
    /// `"model/gltf-binary"` for GLB output.
    pub mime_type: String,
    /// Before/after statistics for the refinement run.
    pub stats: RefineStats,
}

/// Summary statistics emitted by a [`Refiner3dBackend::refine`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineStats {
    /// Triangle count of the input mesh.
    pub input_tri_count: u32,
    /// Triangle count of the output (refined) mesh.
    pub output_tri_count: u32,
    /// When [`RefineRequest::unwrap_uvs`] was set, the number of UV
    /// charts the unwrap produced. `None` when UV unwrapping did not
    /// run for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_chart_count: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refine_request_omits_empty_optionals() {
        let req = RefineRequest {
            decimate_target_tris: None,
            fill_holes: true,
            unwrap_uvs: false,
            retopologize: false,
            smooth_iterations: None,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(!json.contains("decimate_target_tris"));
        assert!(!json.contains("smooth_iterations"));
        assert!(json.contains("\"fill_holes\":true"));
    }

    #[test]
    fn refine_stats_roundtrip() {
        let stats = RefineStats {
            input_tri_count: 100_000,
            output_tri_count: 10_000,
            uv_chart_count: Some(8),
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let back: RefineStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.input_tri_count, 100_000);
        assert_eq!(back.output_tri_count, 10_000);
        assert_eq!(back.uv_chart_count, Some(8));
    }
}
