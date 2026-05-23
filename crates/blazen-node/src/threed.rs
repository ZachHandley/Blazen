//! Node.js bindings for the [`blazen_3d`] crate's HTTP-proxy backend.
//!
//! Exposes the four 3D-pipeline capability traits — texturizer, rigger,
//! refiner, animator — as JavaScript classes wrapping the
//! [`Compat3dProvider`] HTTP-proxy backend that forwards each stage to a
//! configurable upstream service over `multipart/form-data`.
//!
//! # Surface
//!
//! * Request types: [`TexturizeRequest`](JsTexturizeRequest),
//!   [`RigRequest`](JsRigRequest), [`RefineRequest`](JsRefineRequest),
//!   [`AnimateRequest`](JsAnimateRequest).
//! * Result types: [`TexturizeResult`](JsTexturizeResult) (with
//!   [`PbrMaps`](JsPbrMaps)), [`RigResult`](JsRigResult),
//!   [`RefineResult`](JsRefineResult) (with [`RefineStats`](JsRefineStats)),
//!   [`AnimateResult`](JsAnimateResult).
//! * Provider: [`Compat3dProvider`](JsCompat3dProvider) with async methods
//!   `texturize`, `rig`, `refine`, `animate`.

use std::sync::Arc;
use std::time::Duration;

use napi::bindgen_prelude::{Buffer, Result};
use napi_derive::napi;

use blazen_3d::backends::compat::Compat3dProvider;
use blazen_3d::{
    AnimateRequest, AnimateResult, Animator3dBackend, PbrMaps, RefineRequest, RefineResult,
    RefineStats, Refiner3dBackend, RigRequest, RigResult, Rigger3dBackend, TexturizeRequest,
    TexturizeResult, Texturizer3dBackend,
};

// ---------------------------------------------------------------------------
// PbrMaps
// ---------------------------------------------------------------------------

/// Bundle of PBR (physically-based rendering) material maps produced by
/// a texturizer backend.
#[napi(js_name = "PbrMaps")]
pub struct JsPbrMaps {
    inner: PbrMaps,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsPbrMaps {
    /// Base-color / diffuse texture as PNG bytes. Always present.
    #[napi(getter, js_name = "albedoPng")]
    pub fn albedo_png(&self) -> Buffer {
        Buffer::from(self.inner.albedo_png.clone())
    }

    /// Tangent-space normal map as PNG bytes, if produced.
    #[napi(getter, js_name = "normalPng")]
    pub fn normal_png(&self) -> Option<Buffer> {
        self.inner
            .normal_png
            .as_ref()
            .map(|b| Buffer::from(b.clone()))
    }

    /// Linear roughness map as PNG bytes, if produced.
    #[napi(getter, js_name = "roughnessPng")]
    pub fn roughness_png(&self) -> Option<Buffer> {
        self.inner
            .roughness_png
            .as_ref()
            .map(|b| Buffer::from(b.clone()))
    }

    /// Linear metallic map as PNG bytes, if produced.
    #[napi(getter, js_name = "metallicPng")]
    pub fn metallic_png(&self) -> Option<Buffer> {
        self.inner
            .metallic_png
            .as_ref()
            .map(|b| Buffer::from(b.clone()))
    }
}

// ---------------------------------------------------------------------------
// RefineStats
// ---------------------------------------------------------------------------

/// Summary statistics emitted by a [`Compat3dProvider.refine`](JsCompat3dProvider::refine) call.
#[napi(js_name = "RefineStats")]
pub struct JsRefineStats {
    inner: RefineStats,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsRefineStats {
    /// Triangle count of the input mesh.
    #[napi(getter, js_name = "inputTriCount")]
    pub fn input_tri_count(&self) -> u32 {
        self.inner.input_tri_count
    }

    /// Triangle count of the output (refined) mesh.
    #[napi(getter, js_name = "outputTriCount")]
    pub fn output_tri_count(&self) -> u32 {
        self.inner.output_tri_count
    }

    /// When UV unwrapping ran, the number of UV charts the unwrap produced.
    /// `null` when UV unwrapping did not run for this call.
    #[napi(getter, js_name = "uvChartCount")]
    pub fn uv_chart_count(&self) -> Option<u32> {
        self.inner.uv_chart_count
    }
}

// ---------------------------------------------------------------------------
// TexturizeRequest
// ---------------------------------------------------------------------------

/// Options bag for [`Compat3dProvider.texturize`](JsCompat3dProvider::texturize).
#[napi(object, js_name = "TexturizeRequestOptions")]
pub struct JsTexturizeRequestOptions {
    /// Text-guided texture prompt (e.g. `"weathered bronze"`).
    pub prompt: Option<String>,
    /// PNG/JPEG bytes used as a style anchor.
    pub reference_image: Option<Buffer>,
    /// Backend-specific style preset (`"stylized"`, `"realistic"`, ...).
    pub style: Option<String>,
    /// Target square texture resolution in pixels.
    pub resolution: Option<u32>,
    /// `true` to request a full PBR material bundle.
    pub pbr: Option<bool>,
}

/// Request parameters for [`Compat3dProvider.texturize`](JsCompat3dProvider::texturize).
#[napi(js_name = "TexturizeRequest")]
pub struct JsTexturizeRequest {
    inner: TexturizeRequest,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsTexturizeRequest {
    /// Construct a new texturize request.
    #[napi(constructor)]
    pub fn new(options: Option<JsTexturizeRequestOptions>) -> Self {
        let options = options.unwrap_or(JsTexturizeRequestOptions {
            prompt: None,
            reference_image: None,
            style: None,
            resolution: None,
            pbr: None,
        });
        Self {
            inner: TexturizeRequest {
                prompt: options.prompt,
                reference_image: options.reference_image.map(Into::into),
                style: options.style,
                resolution: options.resolution,
                pbr: options.pbr.unwrap_or(false),
            },
        }
    }

    #[napi(getter)]
    pub fn prompt(&self) -> Option<String> {
        self.inner.prompt.clone()
    }

    #[napi(getter, js_name = "referenceImage")]
    pub fn reference_image(&self) -> Option<Buffer> {
        self.inner
            .reference_image
            .as_ref()
            .map(|b| Buffer::from(b.clone()))
    }

    #[napi(getter)]
    pub fn style(&self) -> Option<String> {
        self.inner.style.clone()
    }

    #[napi(getter)]
    pub fn resolution(&self) -> Option<u32> {
        self.inner.resolution
    }

    #[napi(getter)]
    pub fn pbr(&self) -> bool {
        self.inner.pbr
    }
}

// ---------------------------------------------------------------------------
// TexturizeResult
// ---------------------------------------------------------------------------

/// Result of a successful [`Compat3dProvider.texturize`](JsCompat3dProvider::texturize) call.
#[napi(js_name = "TexturizeResult")]
pub struct JsTexturizeResult {
    inner: TexturizeResult,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsTexturizeResult {
    /// GLB bytes with the new texture (and PBR maps if any) embedded.
    #[napi(getter, js_name = "texturedGlb")]
    pub fn textured_glb(&self) -> Buffer {
        Buffer::from(self.inner.textured_glb.clone())
    }

    /// MIME type of `texturedGlb`; always `"model/gltf-binary"`.
    #[napi(getter, js_name = "mimeType")]
    pub fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Optional out-of-band PBR map bundle. Duplicates of the maps
    /// embedded in `texturedGlb` when present.
    #[napi(getter, js_name = "pbrMaps")]
    pub fn pbr_maps(&self) -> Option<JsPbrMaps> {
        self.inner
            .pbr_maps
            .as_ref()
            .map(|m| JsPbrMaps { inner: m.clone() })
    }
}

// ---------------------------------------------------------------------------
// RigRequest
// ---------------------------------------------------------------------------

/// Options bag for [`Compat3dProvider.rig`](JsCompat3dProvider::rig).
#[napi(object, js_name = "RigRequestOptions")]
pub struct JsRigRequestOptions {
    /// Target rig template (`"humanoid"`, `"quadruped"`, `"auto"`).
    pub template: Option<String>,
    /// `true` to apply skin-weight painting after armature placement
    /// (default `true`).
    pub skin: Option<bool>,
    /// Optional pose hint (`"t-pose"`, `"a-pose"`, or backend JSON).
    pub pose_hint: Option<String>,
}

/// Request parameters for [`Compat3dProvider.rig`](JsCompat3dProvider::rig).
#[napi(js_name = "RigRequest")]
pub struct JsRigRequest {
    inner: RigRequest,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsRigRequest {
    /// Construct a new rig request.
    #[napi(constructor)]
    pub fn new(options: Option<JsRigRequestOptions>) -> Self {
        let options = options.unwrap_or(JsRigRequestOptions {
            template: None,
            skin: None,
            pose_hint: None,
        });
        Self {
            inner: RigRequest {
                template: options.template,
                skin: options.skin.unwrap_or(true),
                pose_hint: options.pose_hint,
            },
        }
    }

    #[napi(getter)]
    pub fn template(&self) -> Option<String> {
        self.inner.template.clone()
    }

    #[napi(getter)]
    pub fn skin(&self) -> bool {
        self.inner.skin
    }

    #[napi(getter, js_name = "poseHint")]
    pub fn pose_hint(&self) -> Option<String> {
        self.inner.pose_hint.clone()
    }
}

// ---------------------------------------------------------------------------
// RigResult
// ---------------------------------------------------------------------------

/// Result of a successful [`Compat3dProvider.rig`](JsCompat3dProvider::rig) call.
#[napi(js_name = "RigResult")]
pub struct JsRigResult {
    inner: RigResult,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsRigResult {
    /// GLB bytes with the new armature (and skin weights, if requested) embedded.
    #[napi(getter, js_name = "riggedGlb")]
    pub fn rigged_glb(&self) -> Buffer {
        Buffer::from(self.inner.rigged_glb.clone())
    }

    /// MIME type of `riggedGlb`; always `"model/gltf-binary"`.
    #[napi(getter, js_name = "mimeType")]
    pub fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Names of bones in the produced armature, in depth-first traversal order.
    #[napi(getter, js_name = "boneNames")]
    pub fn bone_names(&self) -> Vec<String> {
        self.inner.bone_names.clone()
    }
}

// ---------------------------------------------------------------------------
// RefineRequest
// ---------------------------------------------------------------------------

/// Options bag for [`Compat3dProvider.refine`](JsCompat3dProvider::refine).
#[napi(object, js_name = "RefineRequestOptions")]
pub struct JsRefineRequestOptions {
    /// Decimate the mesh towards this triangle count.
    pub decimate_target_tris: Option<u32>,
    /// `true` to fill holes via screened poisson reconstruction.
    pub fill_holes: Option<bool>,
    /// `true` to compute a new UV unwrap.
    pub unwrap_uvs: Option<bool>,
    /// `true` to retopologize the mesh.
    pub retopologize: Option<bool>,
    /// Laplacian / Taubin smoothing iteration count.
    pub smooth_iterations: Option<u32>,
}

/// Request parameters for [`Compat3dProvider.refine`](JsCompat3dProvider::refine).
#[napi(js_name = "RefineRequest")]
pub struct JsRefineRequest {
    inner: RefineRequest,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsRefineRequest {
    /// Construct a new refine request.
    #[napi(constructor)]
    pub fn new(options: Option<JsRefineRequestOptions>) -> Self {
        let options = options.unwrap_or(JsRefineRequestOptions {
            decimate_target_tris: None,
            fill_holes: None,
            unwrap_uvs: None,
            retopologize: None,
            smooth_iterations: None,
        });
        Self {
            inner: RefineRequest {
                decimate_target_tris: options.decimate_target_tris,
                fill_holes: options.fill_holes.unwrap_or(false),
                unwrap_uvs: options.unwrap_uvs.unwrap_or(false),
                retopologize: options.retopologize.unwrap_or(false),
                smooth_iterations: options.smooth_iterations,
            },
        }
    }

    #[napi(getter, js_name = "decimateTargetTris")]
    pub fn decimate_target_tris(&self) -> Option<u32> {
        self.inner.decimate_target_tris
    }

    #[napi(getter, js_name = "fillHoles")]
    pub fn fill_holes(&self) -> bool {
        self.inner.fill_holes
    }

    #[napi(getter, js_name = "unwrapUvs")]
    pub fn unwrap_uvs(&self) -> bool {
        self.inner.unwrap_uvs
    }

    #[napi(getter)]
    pub fn retopologize(&self) -> bool {
        self.inner.retopologize
    }

    #[napi(getter, js_name = "smoothIterations")]
    pub fn smooth_iterations(&self) -> Option<u32> {
        self.inner.smooth_iterations
    }
}

// ---------------------------------------------------------------------------
// RefineResult
// ---------------------------------------------------------------------------

/// Result of a successful [`Compat3dProvider.refine`](JsCompat3dProvider::refine) call.
#[napi(js_name = "RefineResult")]
pub struct JsRefineResult {
    inner: RefineResult,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsRefineResult {
    /// GLB bytes with the requested refinement passes applied.
    #[napi(getter, js_name = "refinedGlb")]
    pub fn refined_glb(&self) -> Buffer {
        Buffer::from(self.inner.refined_glb.clone())
    }

    /// MIME type of `refinedGlb`; always `"model/gltf-binary"`.
    #[napi(getter, js_name = "mimeType")]
    pub fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Before/after statistics for the refinement run.
    #[napi(getter)]
    pub fn stats(&self) -> JsRefineStats {
        JsRefineStats {
            inner: self.inner.stats.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// AnimateRequest
// ---------------------------------------------------------------------------

/// Options bag for [`Compat3dProvider.animate`](JsCompat3dProvider::animate).
#[napi(object, js_name = "AnimateRequestOptions")]
pub struct JsAnimateRequestOptions {
    /// Text-guided motion prompt (e.g. `"walks forward and waves"`).
    pub prompt: Option<String>,
    /// MP4 bytes for video-driven motion transfer.
    pub driving_video: Option<Buffer>,
    /// BVH motion-capture clip bytes.
    pub bvh_motion: Option<Buffer>,
    /// Requested animation duration in seconds.
    pub duration_seconds: Option<f64>,
    /// Requested animation framerate.
    pub fps: Option<u32>,
    /// `true` to mark the produced animation as a seamless loop.
    pub loop_animation: Option<bool>,
}

/// Request parameters for [`Compat3dProvider.animate`](JsCompat3dProvider::animate).
#[napi(js_name = "AnimateRequest")]
pub struct JsAnimateRequest {
    inner: AnimateRequest,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsAnimateRequest {
    /// Construct a new animate request.
    #[napi(constructor)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(options: Option<JsAnimateRequestOptions>) -> Self {
        let options = options.unwrap_or(JsAnimateRequestOptions {
            prompt: None,
            driving_video: None,
            bvh_motion: None,
            duration_seconds: None,
            fps: None,
            loop_animation: None,
        });
        Self {
            inner: AnimateRequest {
                prompt: options.prompt,
                driving_video: options.driving_video.map(Into::into),
                bvh_motion: options.bvh_motion.map(Into::into),
                duration_seconds: options.duration_seconds.map(|v| v as f32),
                fps: options.fps,
                loop_animation: options.loop_animation.unwrap_or(false),
            },
        }
    }

    #[napi(getter)]
    pub fn prompt(&self) -> Option<String> {
        self.inner.prompt.clone()
    }

    #[napi(getter, js_name = "drivingVideo")]
    pub fn driving_video(&self) -> Option<Buffer> {
        self.inner
            .driving_video
            .as_ref()
            .map(|b| Buffer::from(b.clone()))
    }

    #[napi(getter, js_name = "bvhMotion")]
    pub fn bvh_motion(&self) -> Option<Buffer> {
        self.inner
            .bvh_motion
            .as_ref()
            .map(|b| Buffer::from(b.clone()))
    }

    #[napi(getter, js_name = "durationSeconds")]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.inner.duration_seconds.map(f64::from)
    }

    #[napi(getter)]
    pub fn fps(&self) -> Option<u32> {
        self.inner.fps
    }

    #[napi(getter, js_name = "loopAnimation")]
    pub fn loop_animation(&self) -> bool {
        self.inner.loop_animation
    }
}

// ---------------------------------------------------------------------------
// AnimateResult
// ---------------------------------------------------------------------------

/// Result of a successful [`Compat3dProvider.animate`](JsCompat3dProvider::animate) call.
#[napi(js_name = "AnimateResult")]
pub struct JsAnimateResult {
    inner: AnimateResult,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsAnimateResult {
    /// GLB bytes with the animation track(s) embedded.
    #[napi(getter, js_name = "animatedGlb")]
    pub fn animated_glb(&self) -> Buffer {
        Buffer::from(self.inner.animated_glb.clone())
    }

    /// MIME type of `animatedGlb`; always `"model/gltf-binary"`.
    #[napi(getter, js_name = "mimeType")]
    pub fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Actual produced duration in seconds (may differ from the request).
    #[napi(getter, js_name = "durationSeconds")]
    pub fn duration_seconds(&self) -> f64 {
        f64::from(self.inner.duration_seconds)
    }

    /// Actual produced framerate in frames per second (may differ from the request).
    #[napi(getter)]
    pub fn fps(&self) -> u32 {
        self.inner.fps
    }
}

// ---------------------------------------------------------------------------
// Compat3dProvider
// ---------------------------------------------------------------------------

/// HTTP-proxy backend that implements all four 3D-pipeline capability
/// traits against a configurable upstream service.
///
/// For every stage, this provider POSTs a `multipart/form-data` request
/// with the mesh GLB and a JSON request body to
/// `{baseUrl}/v1/3d/{texturize,rig,refine,animate}`, and decodes a
/// base64-wrapped JSON response into the corresponding result class.
///
/// ```javascript
/// const provider = new Compat3dProvider("https://my-3d-server.example.com", "secret");
/// const result = await provider.texturize(meshGlb, new TexturizeRequest({ prompt: "bronze", pbr: true }));
/// ```
#[napi(js_name = "Compat3dProvider")]
pub struct JsCompat3dProvider {
    inner: Arc<Compat3dProvider>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompat3dProvider {
    /// Construct a new HTTP-proxy provider.
    ///
    /// `apiKey` is sent as `Authorization: Bearer <key>` when provided.
    /// `timeoutSecs` is the per-request timeout in seconds (default 600).
    #[napi(constructor)]
    pub fn new(base_url: String, api_key: Option<String>, timeout_secs: Option<u32>) -> Self {
        let mut provider = Compat3dProvider::new(base_url);
        if let Some(key) = api_key {
            provider = provider.with_api_key(key);
        }
        if let Some(secs) = timeout_secs {
            provider = provider.with_timeout(Duration::from_secs(u64::from(secs)));
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Apply or generate a texture/material for an existing 3D mesh.
    #[napi]
    pub async fn texturize(
        &self,
        mesh_glb: Buffer,
        request: &JsTexturizeRequest,
    ) -> Result<JsTexturizeResult> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        let mesh_bytes: Vec<u8> = mesh_glb.into();
        let result = provider
            .texturize(&mesh_bytes, req)
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(JsTexturizeResult { inner: result })
    }

    /// Auto-rig a 3D mesh, producing a GLB with skeletal armature and
    /// (optionally) skin weights embedded.
    #[napi]
    pub async fn rig(&self, mesh_glb: Buffer, request: &JsRigRequest) -> Result<JsRigResult> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        let mesh_bytes: Vec<u8> = mesh_glb.into();
        let result = provider
            .rig(&mesh_bytes, req)
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(JsRigResult { inner: result })
    }

    /// Refine a 3D mesh: decimate, fill holes, unwrap UVs, retopologize, smooth.
    #[napi]
    pub async fn refine(
        &self,
        mesh_glb: Buffer,
        request: &JsRefineRequest,
    ) -> Result<JsRefineResult> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        let mesh_bytes: Vec<u8> = mesh_glb.into();
        let result = provider
            .refine(&mesh_bytes, req)
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(JsRefineResult { inner: result })
    }

    /// Animate a rigged 3D mesh from a text prompt, motion-capture clip,
    /// or driving video.
    #[napi]
    pub async fn animate(
        &self,
        rigged_glb: Buffer,
        request: &JsAnimateRequest,
    ) -> Result<JsAnimateResult> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        let mesh_bytes: Vec<u8> = rigged_glb.into();
        let result = provider
            .animate(&mesh_bytes, req)
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(JsAnimateResult { inner: result })
    }
}
