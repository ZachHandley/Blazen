//! UniFFI bindings for the [`blazen_3d`] crate's HTTP-proxy backend.
//!
//! Exposes the four 3D-pipeline capability traits — texturizer, rigger,
//! refiner, animator — as a single [`Compat3dProvider`] UniFFI object
//! whose four async methods forward to a configurable upstream service
//! over `multipart/form-data`. Mirrors the surface in
//! `crates/blazen-py/src/threed.rs` so the Go / Swift / Kotlin / Ruby
//! bindings get the same 3D-pipeline shape as the Python binding.
//!
//! ## Surface
//!
//! * Request records: [`TexturizeRequest`], [`RigRequest`],
//!   [`RefineRequest`], [`AnimateRequest`].
//! * Result records: [`TexturizeResult`] (with [`PbrMaps`]), [`RigResult`],
//!   [`RefineResult`] (with [`RefineStats`]), [`AnimateResult`].
//! * Provider object: [`Compat3dProvider`] with async methods
//!   `texturize`, `rig`, `refine`, `animate`.
//! * Error: [`ThreeDError`] — a flat error enum that absorbs the four
//!   parallel error enums from `blazen-3d` (`Texturizer3dError`,
//!   `Rigger3dError`, `Refiner3dError`, `Animator3dError`).
//!
//! UniFFI prefers types defined in the crate that exports them, so the
//! request and result types are local mirrors of the foreign
//! `blazen_3d::*` types with `From` conversions at the seam.

use std::sync::Arc;
use std::time::Duration;

use blazen_3d::backends::compat::Compat3dProvider as CoreCompat3dProvider;
use blazen_3d::{
    AnimateRequest as CoreAnimateRequest, AnimateResult as CoreAnimateResult, Animator3dBackend,
    Animator3dError, PbrMaps as CorePbrMaps, RefineRequest as CoreRefineRequest,
    RefineResult as CoreRefineResult, RefineStats as CoreRefineStats, Refiner3dBackend,
    Refiner3dError, RigRequest as CoreRigRequest, RigResult as CoreRigResult, Rigger3dBackend,
    Rigger3dError, TexturizeRequest as CoreTexturizeRequest,
    TexturizeResult as CoreTexturizeResult, Texturizer3dBackend, Texturizer3dError,
};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Canonical error returned by every [`Compat3dProvider`] method.
///
/// Absorbs the four parallel error enums from [`blazen_3d`] —
/// [`Texturizer3dError`], [`Rigger3dError`], [`Refiner3dError`], and
/// [`Animator3dError`] — into one flat shape. Foreign callers switch
/// on the variant rather than on which stage produced the error
/// (the call site already determines that).
#[derive(Debug, Error, uniffi::Error)]
pub enum ThreeDError {
    /// The active backend reported a runtime failure (HTTP error,
    /// inference error, etc.).
    #[error("backend: {message}")]
    Backend { message: String },

    /// The caller-supplied input was malformed — invalid mesh bytes,
    /// unsupported container format, malformed request fields, etc.
    #[error("invalid input: {message}")]
    InvalidInput { message: String },

    /// I/O failure while reading a mesh file, reference image, or
    /// model file.
    #[error("io: {message}")]
    Io { message: String },

    /// The selected backend is not available in this build (e.g. the
    /// `compat-proxy` feature is disabled).
    #[error("engine not available: {message}")]
    EngineNotAvailable { message: String },

    /// The capability requested is not supported by the active
    /// backend (e.g. PBR maps from an albedo-only texturizer, video-
    /// driven motion on a text-only animator).
    #[error("unsupported: {message}")]
    Unsupported { message: String },
}

impl From<Texturizer3dError> for ThreeDError {
    fn from(err: Texturizer3dError) -> Self {
        match err {
            Texturizer3dError::Io(e) => Self::Io {
                message: e.to_string(),
            },
            Texturizer3dError::EngineNotAvailable(m) => Self::EngineNotAvailable { message: m },
            Texturizer3dError::InvalidInput(m) => Self::InvalidInput { message: m },
            Texturizer3dError::Backend(m) => Self::Backend { message: m },
            Texturizer3dError::Unsupported(m) => Self::Unsupported { message: m },
        }
    }
}

impl From<Rigger3dError> for ThreeDError {
    fn from(err: Rigger3dError) -> Self {
        match err {
            Rigger3dError::Io(e) => Self::Io {
                message: e.to_string(),
            },
            Rigger3dError::EngineNotAvailable(m) => Self::EngineNotAvailable { message: m },
            Rigger3dError::InvalidInput(m) => Self::InvalidInput { message: m },
            Rigger3dError::Backend(m) => Self::Backend { message: m },
            Rigger3dError::Unsupported(m) => Self::Unsupported { message: m },
        }
    }
}

impl From<Refiner3dError> for ThreeDError {
    fn from(err: Refiner3dError) -> Self {
        match err {
            Refiner3dError::Io(e) => Self::Io {
                message: e.to_string(),
            },
            Refiner3dError::EngineNotAvailable(m) => Self::EngineNotAvailable { message: m },
            Refiner3dError::InvalidInput(m) => Self::InvalidInput { message: m },
            Refiner3dError::Backend(m) => Self::Backend { message: m },
            Refiner3dError::Unsupported(m) => Self::Unsupported { message: m },
        }
    }
}

impl From<Animator3dError> for ThreeDError {
    fn from(err: Animator3dError) -> Self {
        match err {
            Animator3dError::Io(e) => Self::Io {
                message: e.to_string(),
            },
            Animator3dError::EngineNotAvailable(m) => Self::EngineNotAvailable { message: m },
            Animator3dError::InvalidInput(m) => Self::InvalidInput { message: m },
            Animator3dError::Backend(m) => Self::Backend { message: m },
            Animator3dError::Unsupported(m) => Self::Unsupported { message: m },
        }
    }
}

// ---------------------------------------------------------------------------
// PbrMaps / RefineStats
// ---------------------------------------------------------------------------

/// Bundle of PBR (physically-based rendering) material maps produced by
/// a texturizer backend. `albedo_png` is always populated; the other
/// channels are optional and depend on what the backend produces.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PbrMaps {
    /// Base-color / diffuse texture as PNG bytes. Always present.
    pub albedo_png: Vec<u8>,
    /// Tangent-space normal map as PNG bytes, if produced.
    pub normal_png: Option<Vec<u8>>,
    /// Linear roughness map as PNG bytes, if produced.
    pub roughness_png: Option<Vec<u8>>,
    /// Linear metallic map as PNG bytes, if produced.
    pub metallic_png: Option<Vec<u8>>,
}

impl From<CorePbrMaps> for PbrMaps {
    fn from(m: CorePbrMaps) -> Self {
        Self {
            albedo_png: m.albedo_png,
            normal_png: m.normal_png,
            roughness_png: m.roughness_png,
            metallic_png: m.metallic_png,
        }
    }
}

impl From<PbrMaps> for CorePbrMaps {
    fn from(m: PbrMaps) -> Self {
        Self {
            albedo_png: m.albedo_png,
            normal_png: m.normal_png,
            roughness_png: m.roughness_png,
            metallic_png: m.metallic_png,
        }
    }
}

/// Summary statistics emitted by a [`Compat3dProvider::refine`] call.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RefineStats {
    /// Triangle count of the input mesh.
    pub input_tri_count: u32,
    /// Triangle count of the output (refined) mesh.
    pub output_tri_count: u32,
    /// When UV unwrapping ran, the number of UV charts the unwrap
    /// produced. `None` when UV unwrapping did not run for this call.
    pub uv_chart_count: Option<u32>,
}

impl From<CoreRefineStats> for RefineStats {
    fn from(s: CoreRefineStats) -> Self {
        Self {
            input_tri_count: s.input_tri_count,
            output_tri_count: s.output_tri_count,
            uv_chart_count: s.uv_chart_count,
        }
    }
}

impl From<RefineStats> for CoreRefineStats {
    fn from(s: RefineStats) -> Self {
        Self {
            input_tri_count: s.input_tri_count,
            output_tri_count: s.output_tri_count,
            uv_chart_count: s.uv_chart_count,
        }
    }
}

// ---------------------------------------------------------------------------
// TexturizeRequest / TexturizeResult
// ---------------------------------------------------------------------------

/// Request parameters for [`Compat3dProvider::texturize`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct TexturizeRequest {
    /// Text-guided texture prompt (e.g. `"weathered bronze"`).
    pub prompt: Option<String>,
    /// Image-guided reference (PNG or JPEG bytes) used as a style anchor.
    pub reference_image: Option<Vec<u8>>,
    /// Backend-specific style preset (`"stylized"`, `"realistic"`, ...).
    pub style: Option<String>,
    /// Target square texture resolution in pixels.
    pub resolution: Option<u32>,
    /// `true` to request a full PBR material bundle.
    pub pbr: bool,
}

impl From<TexturizeRequest> for CoreTexturizeRequest {
    fn from(r: TexturizeRequest) -> Self {
        Self {
            prompt: r.prompt,
            reference_image: r.reference_image,
            style: r.style,
            resolution: r.resolution,
            pbr: r.pbr,
        }
    }
}

/// Result of a successful [`Compat3dProvider::texturize`] call.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TexturizeResult {
    /// GLB bytes with the new texture (and PBR maps if any) embedded.
    pub textured_glb: Vec<u8>,
    /// MIME type of `textured_glb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Optional out-of-band PBR map bundle. Duplicates of the maps
    /// embedded in `textured_glb` when present.
    pub pbr_maps: Option<PbrMaps>,
}

impl From<CoreTexturizeResult> for TexturizeResult {
    fn from(r: CoreTexturizeResult) -> Self {
        Self {
            textured_glb: r.textured_glb,
            mime_type: r.mime_type,
            pbr_maps: r.pbr_maps.map(PbrMaps::from),
        }
    }
}

// ---------------------------------------------------------------------------
// RigRequest / RigResult
// ---------------------------------------------------------------------------

/// Request parameters for [`Compat3dProvider::rig`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct RigRequest {
    /// Target rig template (`"humanoid"`, `"quadruped"`, `"auto"`).
    pub template: Option<String>,
    /// `true` to apply skin-weight painting after armature placement.
    pub skin: bool,
    /// Optional pose hint (`"t-pose"`, `"a-pose"`, or backend-specific JSON).
    pub pose_hint: Option<String>,
}

impl From<RigRequest> for CoreRigRequest {
    fn from(r: RigRequest) -> Self {
        Self {
            template: r.template,
            skin: r.skin,
            pose_hint: r.pose_hint,
        }
    }
}

/// Result of a successful [`Compat3dProvider::rig`] call.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RigResult {
    /// GLB bytes with the new armature (and skin weights, if requested) embedded.
    pub rigged_glb: Vec<u8>,
    /// MIME type of `rigged_glb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Names of bones in the produced armature, in depth-first traversal order.
    pub bone_names: Vec<String>,
}

impl From<CoreRigResult> for RigResult {
    fn from(r: CoreRigResult) -> Self {
        Self {
            rigged_glb: r.rigged_glb,
            mime_type: r.mime_type,
            bone_names: r.bone_names,
        }
    }
}

// ---------------------------------------------------------------------------
// RefineRequest / RefineResult
// ---------------------------------------------------------------------------

/// Request parameters for [`Compat3dProvider::refine`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct RefineRequest {
    /// Decimate the mesh towards this triangle count.
    pub decimate_target_tris: Option<u32>,
    /// `true` to fill holes via screened poisson reconstruction.
    pub fill_holes: bool,
    /// `true` to compute a new UV unwrap.
    pub unwrap_uvs: bool,
    /// `true` to retopologize the mesh.
    pub retopologize: bool,
    /// Laplacian / Taubin smoothing iteration count.
    pub smooth_iterations: Option<u32>,
}

impl From<RefineRequest> for CoreRefineRequest {
    fn from(r: RefineRequest) -> Self {
        Self {
            decimate_target_tris: r.decimate_target_tris,
            fill_holes: r.fill_holes,
            unwrap_uvs: r.unwrap_uvs,
            retopologize: r.retopologize,
            smooth_iterations: r.smooth_iterations,
        }
    }
}

/// Result of a successful [`Compat3dProvider::refine`] call.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RefineResult {
    /// GLB bytes with the requested refinement passes applied.
    pub refined_glb: Vec<u8>,
    /// MIME type of `refined_glb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Before/after statistics for the refinement run.
    pub stats: RefineStats,
}

impl From<CoreRefineResult> for RefineResult {
    fn from(r: CoreRefineResult) -> Self {
        Self {
            refined_glb: r.refined_glb,
            mime_type: r.mime_type,
            stats: RefineStats::from(r.stats),
        }
    }
}

// ---------------------------------------------------------------------------
// AnimateRequest / AnimateResult
// ---------------------------------------------------------------------------

/// Request parameters for [`Compat3dProvider::animate`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct AnimateRequest {
    /// Text-guided motion prompt (e.g. `"walks forward and waves"`).
    pub prompt: Option<String>,
    /// Optional MP4 bytes for video-driven motion transfer.
    pub driving_video: Option<Vec<u8>>,
    /// Optional BVH motion-capture clip bytes.
    pub bvh_motion: Option<Vec<u8>>,
    /// Requested animation duration in seconds.
    pub duration_seconds: Option<f32>,
    /// Requested animation framerate.
    pub fps: Option<u32>,
    /// `true` to mark the produced animation as a seamless loop.
    pub loop_animation: bool,
}

impl From<AnimateRequest> for CoreAnimateRequest {
    fn from(r: AnimateRequest) -> Self {
        Self {
            prompt: r.prompt,
            driving_video: r.driving_video,
            bvh_motion: r.bvh_motion,
            duration_seconds: r.duration_seconds,
            fps: r.fps,
            loop_animation: r.loop_animation,
        }
    }
}

/// Result of a successful [`Compat3dProvider::animate`] call.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AnimateResult {
    /// GLB bytes with the animation track(s) embedded.
    pub animated_glb: Vec<u8>,
    /// MIME type of `animated_glb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Actual produced duration in seconds (may differ from the request).
    pub duration_seconds: f32,
    /// Actual produced framerate in frames per second (may differ from the request).
    pub fps: u32,
}

impl From<CoreAnimateResult> for AnimateResult {
    fn from(r: CoreAnimateResult) -> Self {
        Self {
            animated_glb: r.animated_glb,
            mime_type: r.mime_type,
            duration_seconds: r.duration_seconds,
            fps: r.fps,
        }
    }
}

// ---------------------------------------------------------------------------
// Compat3dProvider
// ---------------------------------------------------------------------------

/// HTTP-proxy backend implementing all four 3D-pipeline capability
/// traits against a configurable upstream service.
///
/// For every stage, this provider POSTs a `multipart/form-data` request
/// with the mesh GLB and a JSON request body to
/// `{base_url}/v1/3d/{texturize,rig,refine,animate}`, and decodes a
/// base64-wrapped JSON response into the corresponding result record.
#[derive(uniffi::Object)]
pub struct Compat3dProvider {
    inner: Arc<CoreCompat3dProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl Compat3dProvider {
    /// Construct a new HTTP-proxy provider.
    ///
    /// `base_url` is the upstream root URL (e.g.
    /// `"https://3d.example.com"`). `api_key` is an optional bearer
    /// token attached as `Authorization: Bearer ...`. `timeout_secs`
    /// is an optional per-request timeout in seconds (default 600).
    #[uniffi::constructor]
    #[must_use]
    pub fn new(base_url: String, api_key: Option<String>, timeout_secs: Option<u32>) -> Arc<Self> {
        let mut provider = CoreCompat3dProvider::new(base_url);
        if let Some(key) = api_key {
            provider = provider.with_api_key(key);
        }
        if let Some(secs) = timeout_secs {
            provider = provider.with_timeout(Duration::from_secs(u64::from(secs)));
        }
        Arc::new(Self {
            inner: Arc::new(provider),
        })
    }

    /// Apply or generate a texture/material for an existing 3D mesh.
    pub async fn texturize(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, ThreeDError> {
        let provider = Arc::clone(&self.inner);
        let core_req = CoreTexturizeRequest::from(request);
        let result = provider.texturize(&mesh_glb, core_req).await?;
        Ok(TexturizeResult::from(result))
    }

    /// Auto-rig a 3D mesh, producing a GLB with skeletal armature and
    /// (optionally) skin weights embedded.
    pub async fn rig(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: RigRequest,
    ) -> Result<RigResult, ThreeDError> {
        let provider = Arc::clone(&self.inner);
        let core_req = CoreRigRequest::from(request);
        let result = provider.rig(&mesh_glb, core_req).await?;
        Ok(RigResult::from(result))
    }

    /// Refine a 3D mesh: decimate, fill holes, unwrap UVs, retopologize, smooth.
    pub async fn refine(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: RefineRequest,
    ) -> Result<RefineResult, ThreeDError> {
        let provider = Arc::clone(&self.inner);
        let core_req = CoreRefineRequest::from(request);
        let result = provider.refine(&mesh_glb, core_req).await?;
        Ok(RefineResult::from(result))
    }

    /// Animate a rigged 3D mesh from a text prompt, motion-capture clip,
    /// or driving video.
    pub async fn animate(
        self: Arc<Self>,
        rigged_glb: Vec<u8>,
        request: AnimateRequest,
    ) -> Result<AnimateResult, ThreeDError> {
        let provider = Arc::clone(&self.inner);
        let core_req = CoreAnimateRequest::from(request);
        let result = provider.animate(&rigged_glb, core_req).await?;
        Ok(AnimateResult::from(result))
    }
}
