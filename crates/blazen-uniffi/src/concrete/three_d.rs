//! 3D per-engine `#[uniffi::Object]` providers — wraps the canonical
//! Rust concretes from
//! [`blazen_llm::providers::concrete::three_d`] so each
//! UniFFI-supported foreign language gets a real per-engine class
//! instead of a single capability-erased handle.
//!
//! Method names mirror the central [`crate::compute::ThreeDModel`]
//! surface so consumers moving across the deprecation boundary
//! (P4.3 / P4.4 / P4.5) keep the same call sites.
//!
//! ## Engines
//!
//! - [`TripoSrProvider`] — `triposr`. Native candle image-to-3D.
//!   Generation-only; the post-processing methods (`texturize`, `rig`,
//!   `refine`, `animate`) surface [`crate::errors::BlazenError::Unsupported`].
//! - [`Compat3dProvider`] — `threed-compat-proxy`. HTTP-proxy backend
//!   forwarding multipart requests to a configurable upstream service.
//!   Post-processing-only; `generate_from_image` surfaces
//!   [`crate::errors::BlazenError::Unsupported`] (the upstream wire
//!   contract only covers texturize / rig / refine / animate — base
//!   generation must come from a separate backend like
//!   [`TripoSrProvider`]).
//!
//! ## Request / result DTOs
//!
//! The texturize / rig / refine / animate stages exchange parameter
//! and result records that are local uniffi mirrors of the foreign
//! [`blazen_3d::*`] DTOs. UniFFI prefers types defined in the crate
//! that exports them, so this module re-declares them with
//! `#[derive(uniffi::Record)]` plus `From` conversions at the seam.
//!
//! Available whenever the umbrella `threed` feature is enabled (which
//! both `triposr` and `threed-compat-proxy` imply).

#![allow(dead_code, unused_imports)]

use std::sync::Arc;
use std::time::Duration;

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

#[cfg(feature = "threed")]
use crate::compute::ThreeDGenerateResult;

#[cfg(feature = "threed")]
use blazen_3d::{
    AnimateRequest as CoreAnimateRequest, AnimateResult as CoreAnimateResult,
    PbrMaps as CorePbrMaps, RefineRequest as CoreRefineRequest, RefineResult as CoreRefineResult,
    RefineStats as CoreRefineStats, RigRequest as CoreRigRequest, RigResult as CoreRigResult,
    TexturizeRequest as CoreTexturizeRequest, TexturizeResult as CoreTexturizeResult,
};

// ===========================================================================
// PbrMaps / RefineStats — shared DTO sub-records
// ===========================================================================

/// Bundle of PBR (physically-based rendering) material maps produced by
/// a texturizer backend. `albedo_png` is always populated; the other
/// channels are optional and depend on what the backend produces.
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
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
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
impl From<CoreRefineStats> for RefineStats {
    fn from(s: CoreRefineStats) -> Self {
        Self {
            input_tri_count: s.input_tri_count,
            output_tri_count: s.output_tri_count,
            uv_chart_count: s.uv_chart_count,
        }
    }
}

#[cfg(feature = "threed")]
impl From<RefineStats> for CoreRefineStats {
    fn from(s: RefineStats) -> Self {
        Self {
            input_tri_count: s.input_tri_count,
            output_tri_count: s.output_tri_count,
            uv_chart_count: s.uv_chart_count,
        }
    }
}

// ===========================================================================
// TexturizeRequest / TexturizeResult
// ===========================================================================

/// Request parameters for [`Compat3dProvider::texturize`].
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
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
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
impl From<CoreTexturizeResult> for TexturizeResult {
    fn from(r: CoreTexturizeResult) -> Self {
        Self {
            textured_glb: r.textured_glb,
            mime_type: r.mime_type,
            pbr_maps: r.pbr_maps.map(PbrMaps::from),
        }
    }
}

// ===========================================================================
// RigRequest / RigResult
// ===========================================================================

/// Request parameters for [`Compat3dProvider::rig`].
#[cfg(feature = "threed")]
#[derive(Debug, Clone, uniffi::Record)]
pub struct RigRequest {
    /// Target rig template (`"humanoid"`, `"quadruped"`, `"auto"`).
    pub template: Option<String>,
    /// `true` to apply skin-weight painting after armature placement.
    pub skin: bool,
    /// Optional pose hint (`"t-pose"`, `"a-pose"`, or backend-specific JSON).
    pub pose_hint: Option<String>,
}

#[cfg(feature = "threed")]
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
#[cfg(feature = "threed")]
#[derive(Debug, Clone, uniffi::Record)]
pub struct RigResult {
    /// GLB bytes with the new armature (and skin weights, if requested) embedded.
    pub rigged_glb: Vec<u8>,
    /// MIME type of `rigged_glb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Names of bones in the produced armature, in depth-first traversal order.
    pub bone_names: Vec<String>,
}

#[cfg(feature = "threed")]
impl From<CoreRigResult> for RigResult {
    fn from(r: CoreRigResult) -> Self {
        Self {
            rigged_glb: r.rigged_glb,
            mime_type: r.mime_type,
            bone_names: r.bone_names,
        }
    }
}

// ===========================================================================
// RefineRequest / RefineResult
// ===========================================================================

/// Request parameters for [`Compat3dProvider::refine`].
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
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
#[cfg(feature = "threed")]
#[derive(Debug, Clone, uniffi::Record)]
pub struct RefineResult {
    /// GLB bytes with the requested refinement passes applied.
    pub refined_glb: Vec<u8>,
    /// MIME type of `refined_glb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Before/after statistics for the refinement run.
    pub stats: RefineStats,
}

#[cfg(feature = "threed")]
impl From<CoreRefineResult> for RefineResult {
    fn from(r: CoreRefineResult) -> Self {
        Self {
            refined_glb: r.refined_glb,
            mime_type: r.mime_type,
            stats: RefineStats::from(r.stats),
        }
    }
}

// ===========================================================================
// AnimateRequest / AnimateResult
// ===========================================================================

/// Request parameters for [`Compat3dProvider::animate`].
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
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
#[cfg(feature = "threed")]
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

#[cfg(feature = "threed")]
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

// ===========================================================================
// TripoSrProvider — native candle image-to-3D
// ===========================================================================

/// `TripoSR` single-image-to-3D provider.
///
/// Wraps [`blazen_llm::providers::concrete::three_d::TripoSrProvider`]
/// (the candle-runtime `TripoSrBackend` newtype + `ProviderMetadata`
/// stamp). Construction loads the weights — either from `weights_path`
/// (local directory) or by downloading from a Hugging Face repo
/// (defaulting to `"stabilityai/TripoSR"`).
///
/// Generation-only: the texturize / rig / refine / animate post-proc
/// methods on the polymorphic [`crate::concrete::bases::ThreeDProvider`]
/// trait surface [`crate::errors::BlazenError::Unsupported`] — pipe the
/// generated GLB through [`Compat3dProvider`] (or another post-proc
/// backend) to apply textures / armature / refinement / motion.
#[cfg(feature = "triposr")]
#[derive(uniffi::Object)]
pub struct TripoSrProvider {
    inner: Arc<blazen_llm::providers::concrete::three_d::TripoSrProvider>,
}

#[cfg(feature = "triposr")]
#[uniffi::export(async_runtime = "tokio")]
impl TripoSrProvider {
    /// Construct a `TripoSR` provider.
    ///
    /// - `hf_repo_id`: HuggingFace repo to fetch weights from.
    ///   Defaults to `"stabilityai/TripoSR"` when `None`.
    /// - `revision`: optional branch / tag / commit pin on
    ///   `hf_repo_id`.
    /// - `weights_path`: pre-resolved local weights directory. When
    ///   supplied, the HF download is skipped entirely.
    ///
    /// The underlying `TripoSrProvider::new` is async (it may download
    /// weights from Hugging Face). UniFFI constructors are sync, so
    /// we drive it through the shared tokio [`runtime`] — matching
    /// the `new_triposr_3d_model` factory pattern in
    /// [`crate::compute`].
    #[uniffi::constructor]
    pub fn new(
        hf_repo_id: Option<String>,
        revision: Option<String>,
        weights_path: Option<String>,
    ) -> BlazenResult<Arc<Self>> {
        let inner = runtime()
            .block_on(
                blazen_llm::providers::concrete::three_d::TripoSrProvider::new(
                    hf_repo_id,
                    revision,
                    weights_path,
                ),
            )
            .map_err(BlazenError::from)?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Generate a 3D mesh from a single input image.
    ///
    /// `image_bytes` is encoded PNG or JPEG payload, encoded as a
    /// `data:` URI for the upstream `ThreeDGeneration` impl (which
    /// expects an `image_url`). `mesh_resolution` controls the side
    /// length of the density grid sampled from the triplane during
    /// marching cubes; `256` matches the upstream TripoSR reference.
    pub async fn generate_from_image(
        self: Arc<Self>,
        image_bytes: Vec<u8>,
        mesh_resolution: u32,
    ) -> BlazenResult<ThreeDGenerateResult> {
        triposr_generate_from_image(&self.inner, image_bytes, mesh_resolution).await
    }
}

#[cfg(feature = "triposr")]
#[uniffi::export]
impl TripoSrProvider {
    /// Synchronous variant of
    /// [`generate_from_image`](Self::generate_from_image).
    pub fn generate_from_image_blocking(
        self: Arc<Self>,
        image_bytes: Vec<u8>,
        mesh_resolution: u32,
    ) -> BlazenResult<ThreeDGenerateResult> {
        let this = Arc::clone(&self);
        runtime()
            .block_on(async move { this.generate_from_image(image_bytes, mesh_resolution).await })
    }
}

/// Shared `generate_from_image` body used by both the inherent
/// `Arc<Self>` method and the `ThreeDProvider` trait impl on
/// [`TripoSrProvider`].
#[cfg(feature = "triposr")]
async fn triposr_generate_from_image(
    inner: &Arc<blazen_llm::providers::concrete::three_d::TripoSrProvider>,
    image_bytes: Vec<u8>,
    mesh_resolution: u32,
) -> BlazenResult<ThreeDGenerateResult> {
    use base64::Engine as _;
    use blazen_llm::providers::capabilities::ThreeDProvider as _;

    // TripoSR's ThreeDGeneration impl reads `image_url` (HTTP/HTTPS
    // URL or RFC 2397 `data:` URI). Wrap the raw bytes as a base64
    // `data:` URI so callers can pass binary buffers directly
    // without staging them to disk or a temp HTTP server.
    let b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);
    let data_uri = format!("data:application/octet-stream;base64,{b64}");

    let request = blazen_llm::compute::requests::ThreeDRequest {
        prompt: String::new(),
        image_url: Some(data_uri),
        format: None,
        model: None,
        parameters: serde_json::json!({ "mesh_resolution": u64::from(mesh_resolution) }),
    };

    let result = inner.generate_3d(request).await?;

    // TripoSR returns exactly one GLB-encoded mesh in `models[0]`;
    // surface it as the uniffi `ThreeDGenerateResult` record.
    let model = result
        .models
        .into_iter()
        .next()
        .ok_or_else(|| BlazenError::Provider {
            kind: "TripoSrEmptyResult".into(),
            message: "TripoSR returned no models".into(),
            provider: Some("triposr".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;

    let bytes = if let Some(b64) = model.media.base64 {
        base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .map_err(|e| BlazenError::Provider {
                kind: "TripoSrBase64Decode".into(),
                message: format!("failed to decode GLB base64: {e}"),
                provider: Some("triposr".into()),
                status: None,
                endpoint: None,
                request_id: None,
                detail: None,
                retry_after_ms: None,
            })?
    } else {
        Vec::new()
    };

    Ok(ThreeDGenerateResult {
        model_bytes: bytes,
        mime_type: "model/gltf-binary".to_string(),
    })
}

// ===========================================================================
// Compat3dProvider — HTTP-proxy 3D post-processing
// ===========================================================================

/// HTTP-proxy 3D provider implementing the texturize / rig / refine /
/// animate post-processing stages of the 3D pipeline against a
/// configurable upstream service.
///
/// Wraps [`blazen_llm::providers::concrete::three_d::Compat3dProvider`]
/// which itself wraps
/// [`blazen_3d::backends::compat::Compat3dProvider`]. The upstream
/// contract does NOT include a generation endpoint, so
/// [`generate_from_image`](Self::generate_from_image) returns
/// [`BlazenError::Unsupported`] — use [`TripoSrProvider`] (or another
/// generation backend) to produce the base mesh, then forward the
/// result through this provider's post-proc methods.
///
/// For every stage, this provider POSTs a `multipart/form-data`
/// request with the mesh GLB and a JSON request body to
/// `{base_url}/v1/3d/{texturize,rig,refine,animate}` and decodes a
/// base64-wrapped JSON response into the corresponding result record.
#[cfg(feature = "threed-compat-proxy")]
#[derive(uniffi::Object)]
pub struct Compat3dProvider {
    inner: Arc<blazen_llm::providers::concrete::three_d::Compat3dProvider>,
}

#[cfg(feature = "threed-compat-proxy")]
#[uniffi::export(async_runtime = "tokio")]
impl Compat3dProvider {
    /// Construct a new HTTP-proxy provider.
    ///
    /// `base_url` is the upstream root URL (e.g.
    /// `"https://3d.example.com"`). `api_key` is an optional bearer
    /// token attached as `Authorization: Bearer ...`. `timeout_secs`
    /// is an optional per-request timeout in seconds (default 600).
    ///
    /// `timeout_secs` is plumbed through `from_inner` because the
    /// blazen-llm `Compat3dProvider::new` constructor only accepts
    /// `(base_url, api_key)` — the per-request timeout is a builder
    /// on the inner `blazen_3d` provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(base_url: String, api_key: Option<String>, timeout_secs: Option<u32>) -> Arc<Self> {
        let mut backend = blazen_3d::backends::compat::Compat3dProvider::new(base_url);
        if let Some(key) = api_key {
            backend = backend.with_api_key(key);
        }
        if let Some(secs) = timeout_secs {
            backend = backend.with_timeout(Duration::from_secs(u64::from(secs)));
        }
        let inner = blazen_llm::providers::concrete::three_d::Compat3dProvider::from_inner(
            Arc::new(backend),
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Apply or generate a texture/material for an existing 3D mesh.
    pub async fn texturize(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreTexturizeRequest = request.into();
        let core_res = self.inner.texturize(&mesh_glb, core_req).await?;
        Ok(core_res.into())
    }

    /// Auto-rig a 3D mesh, producing a GLB with skeletal armature and
    /// (optionally) skin weights embedded.
    pub async fn rig(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: RigRequest,
    ) -> Result<RigResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreRigRequest = request.into();
        let core_res = self.inner.rig(&mesh_glb, core_req).await?;
        Ok(core_res.into())
    }

    /// Refine a 3D mesh: decimate, fill holes, unwrap UVs, retopologize, smooth.
    pub async fn refine(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: RefineRequest,
    ) -> Result<RefineResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreRefineRequest = request.into();
        let core_res = self.inner.refine(&mesh_glb, core_req).await?;
        Ok(core_res.into())
    }

    /// Animate a rigged 3D mesh from a text prompt, motion-capture
    /// clip, or driving video.
    pub async fn animate(
        self: Arc<Self>,
        rigged_glb: Vec<u8>,
        request: AnimateRequest,
    ) -> Result<AnimateResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreAnimateRequest = request.into();
        let core_res = self.inner.animate(&rigged_glb, core_req).await?;
        Ok(core_res.into())
    }

    /// Generation isn't part of the compat-proxy wire contract.
    ///
    /// Mirrors the inner `ThreeDProvider::generate_3d` `Unsupported`
    /// path so the per-engine class surface stays uniform across
    /// [`TripoSrProvider`] and [`Compat3dProvider`].
    pub async fn generate_from_image(
        self: Arc<Self>,
        _image_bytes: Vec<u8>,
        _mesh_resolution: u32,
    ) -> Result<ThreeDGenerateResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "Compat3dProvider does not support generate_from_image — the HTTP-proxy \
                      upstream only exposes texturize / rig / refine / animate. Use \
                      TripoSrProvider to produce the base mesh, then forward it through \
                      Compat3dProvider's post-proc methods."
                .to_string(),
        })
    }
}

#[cfg(feature = "threed-compat-proxy")]
#[uniffi::export]
impl Compat3dProvider {
    /// Synchronous variant of [`texturize`](Self::texturize).
    pub fn texturize_blocking(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.texturize(mesh_glb, request).await })
    }

    /// Synchronous variant of [`rig`](Self::rig).
    pub fn rig_blocking(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: RigRequest,
    ) -> Result<RigResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.rig(mesh_glb, request).await })
    }

    /// Synchronous variant of [`refine`](Self::refine).
    pub fn refine_blocking(
        self: Arc<Self>,
        mesh_glb: Vec<u8>,
        request: RefineRequest,
    ) -> Result<RefineResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.refine(mesh_glb, request).await })
    }

    /// Synchronous variant of [`animate`](Self::animate).
    pub fn animate_blocking(
        self: Arc<Self>,
        rigged_glb: Vec<u8>,
        request: AnimateRequest,
    ) -> Result<AnimateResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.animate(rigged_glb, request).await })
    }

    /// Synchronous variant of
    /// [`generate_from_image`](Self::generate_from_image) — surfaces
    /// the same `Unsupported` error as the async path.
    pub fn generate_from_image_blocking(
        self: Arc<Self>,
        image_bytes: Vec<u8>,
        mesh_resolution: u32,
    ) -> Result<ThreeDGenerateResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime()
            .block_on(async move { this.generate_from_image(image_bytes, mesh_resolution).await })
    }
}

// ===========================================================================
// Polymorphic trait conformance — `BaseProvider` + `ThreeDProvider`
// ===========================================================================
//
// Each per-engine class implements both the `BaseProvider` root and the
// `ThreeDProvider` capability trait so foreign callers can hold a
// `ThreeDProvider` reference and dispatch across `TripoSrProvider` /
// `Compat3dProvider` interchangeably. UniFFI 0.31 forbids default
// method bodies on exported traits, so the half of the pipeline a
// given engine does not natively cover surfaces `Unsupported` from its
// trait impl block.

// TripoSrProvider ----------------------------------------------------------

#[cfg(feature = "triposr")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for TripoSrProvider {
    fn provider_id(&self) -> String {
        "triposr".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ThreeD
    }
}

#[cfg(feature = "triposr")]
#[async_trait::async_trait]
impl crate::concrete::bases::ThreeDProvider for TripoSrProvider {
    fn provider_id(&self) -> String {
        "triposr".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ThreeD
    }

    async fn generate_from_image(
        &self,
        image_bytes: Vec<u8>,
        mesh_resolution: u32,
    ) -> Result<ThreeDGenerateResult, BlazenError> {
        triposr_generate_from_image(&self.inner, image_bytes, mesh_resolution).await
    }

    async fn texturize(
        &self,
        _mesh_glb: Vec<u8>,
        _request: TexturizeRequest,
    ) -> Result<TexturizeResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "TripoSrProvider does not support texturize — TripoSR is generation-only. \
                      Use Compat3dProvider (or another post-proc backend) for the texturize \
                      stage."
                .to_string(),
        })
    }

    async fn rig(
        &self,
        _mesh_glb: Vec<u8>,
        _request: RigRequest,
    ) -> Result<RigResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "TripoSrProvider does not support rig — TripoSR is generation-only. \
                      Use Compat3dProvider (or another post-proc backend) for the rig stage."
                .to_string(),
        })
    }

    async fn refine(
        &self,
        _mesh_glb: Vec<u8>,
        _request: RefineRequest,
    ) -> Result<RefineResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "TripoSrProvider does not support refine — TripoSR is generation-only. \
                      Use Compat3dProvider (or another post-proc backend) for the refine stage."
                .to_string(),
        })
    }

    async fn animate(
        &self,
        _rigged_glb: Vec<u8>,
        _request: AnimateRequest,
    ) -> Result<AnimateResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "TripoSrProvider does not support animate — TripoSR is generation-only. \
                      Use Compat3dProvider (or another post-proc backend) for the animate stage."
                .to_string(),
        })
    }
}

// Compat3dProvider ---------------------------------------------------------

#[cfg(feature = "threed-compat-proxy")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for Compat3dProvider {
    fn provider_id(&self) -> String {
        "compat-3d".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ThreeD
    }
}

#[cfg(feature = "threed-compat-proxy")]
#[async_trait::async_trait]
impl crate::concrete::bases::ThreeDProvider for Compat3dProvider {
    fn provider_id(&self) -> String {
        "compat-3d".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ThreeD
    }

    async fn generate_from_image(
        &self,
        _image_bytes: Vec<u8>,
        _mesh_resolution: u32,
    ) -> Result<ThreeDGenerateResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "Compat3dProvider does not support generate_from_image — the HTTP-proxy \
                      upstream only exposes texturize / rig / refine / animate. Use \
                      TripoSrProvider to produce the base mesh, then forward it through \
                      Compat3dProvider's post-proc methods."
                .to_string(),
        })
    }

    async fn texturize(
        &self,
        mesh_glb: Vec<u8>,
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreTexturizeRequest = request.into();
        let core_res = self.inner.texturize(&mesh_glb, core_req).await?;
        Ok(core_res.into())
    }

    async fn rig(&self, mesh_glb: Vec<u8>, request: RigRequest) -> Result<RigResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreRigRequest = request.into();
        let core_res = self.inner.rig(&mesh_glb, core_req).await?;
        Ok(core_res.into())
    }

    async fn refine(
        &self,
        mesh_glb: Vec<u8>,
        request: RefineRequest,
    ) -> Result<RefineResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreRefineRequest = request.into();
        let core_res = self.inner.refine(&mesh_glb, core_req).await?;
        Ok(core_res.into())
    }

    async fn animate(
        &self,
        rigged_glb: Vec<u8>,
        request: AnimateRequest,
    ) -> Result<AnimateResult, BlazenError> {
        use blazen_llm::providers::capabilities::ThreeDProvider as _;
        let core_req: CoreAnimateRequest = request.into();
        let core_res = self.inner.animate(&rigged_glb, core_req).await?;
        Ok(core_res.into())
    }
}
