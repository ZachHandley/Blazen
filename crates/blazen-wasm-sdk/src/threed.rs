//! WASM bindings for the [`blazen_3d`] request/result data types.
//!
//! # Why data-types-only
//!
//! `blazen_3d::backends::compat::Compat3dProvider` is an HTTP-proxy
//! backend built on top of [`reqwest`], which does not target
//! `wasm32-unknown-unknown` cleanly (it pulls Tokio's IO driver plus
//! native TLS). Rather than reimplementing the HTTP client in WASM, the
//! SDK exposes the public request / result records here so JS callers
//! can construct them in Rust-land and a TypeScript shim at
//! `js/threed.ts` wraps the native multipart wire format on top of the
//! browser `fetch()` API.
//!
//! # Naming convention
//!
//! Every exported type carries a `Threed` prefix to avoid collision
//! with similarly-named records elsewhere in the SDK (e.g. the
//! `Refine*` family in the pipeline module).
//!
//! # Surface
//!
//! | TypeScript type           | Wraps                                       |
//! |---------------------------|---------------------------------------------|
//! | `ThreedTexturizeRequest`  | [`blazen_3d::TexturizeRequest`]             |
//! | `ThreedTexturizeResult`   | [`blazen_3d::TexturizeResult`]              |
//! | `ThreedPbrMaps`           | [`blazen_3d::PbrMaps`]                      |
//! | `ThreedRigRequest`        | [`blazen_3d::RigRequest`]                   |
//! | `ThreedRigResult`         | [`blazen_3d::RigResult`]                    |
//! | `ThreedRefineRequest`     | [`blazen_3d::RefineRequest`]                |
//! | `ThreedRefineResult`      | [`blazen_3d::RefineResult`]                 |
//! | `ThreedRefineStats`       | [`blazen_3d::RefineStats`]                  |
//! | `ThreedAnimateRequest`    | [`blazen_3d::AnimateRequest`]               |
//! | `ThreedAnimateResult`     | [`blazen_3d::AnimateResult`]                |
//!
//! All field names are serialised in `camelCase` to match the
//! convention used by the rest of the SDK's Tsify mirrors (see
//! `agent_types.rs`). `Vec<u8>` fields render as
//! `Uint8Array | number[]` on the TypeScript side.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

use blazen_3d::{
    AnimateRequest as CoreAnimateRequest, AnimateResult as CoreAnimateResult,
    PbrMaps as CorePbrMaps, RefineRequest as CoreRefineRequest, RefineResult as CoreRefineResult,
    RefineStats as CoreRefineStats, RigRequest as CoreRigRequest, RigResult as CoreRigResult,
    TexturizeRequest as CoreTexturizeRequest, TexturizeResult as CoreTexturizeResult,
};

// ---------------------------------------------------------------------------
// PBR maps
// ---------------------------------------------------------------------------

/// Bundle of PBR (physically-based rendering) material maps produced
/// by a texturizer backend. `albedoPng` is always populated; the
/// other channels are optional and depend on what the backend
/// produces.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedPbrMaps {
    /// Base-color / diffuse texture as PNG bytes. Always present.
    pub albedo_png: Vec<u8>,
    /// Tangent-space normal map as PNG bytes, if produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_png: Option<Vec<u8>>,
    /// Linear roughness map as PNG bytes, if produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness_png: Option<Vec<u8>>,
    /// Linear metallic map as PNG bytes, if produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic_png: Option<Vec<u8>>,
}

impl From<CorePbrMaps> for ThreedPbrMaps {
    fn from(m: CorePbrMaps) -> Self {
        Self {
            albedo_png: m.albedo_png,
            normal_png: m.normal_png,
            roughness_png: m.roughness_png,
            metallic_png: m.metallic_png,
        }
    }
}

impl From<ThreedPbrMaps> for CorePbrMaps {
    fn from(m: ThreedPbrMaps) -> Self {
        Self {
            albedo_png: m.albedo_png,
            normal_png: m.normal_png,
            roughness_png: m.roughness_png,
            metallic_png: m.metallic_png,
        }
    }
}

// ---------------------------------------------------------------------------
// Texturize
// ---------------------------------------------------------------------------

/// Request parameters for the `texturize` stage. Apply or generate a
/// texture/material for an existing 3D mesh.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedTexturizeRequest {
    /// Text-guided texture prompt (e.g. `"weathered bronze"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Image-guided reference (PNG or JPEG bytes) used as a style
    /// anchor for the produced texture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image: Option<Vec<u8>>,
    /// Backend-specific style preset (`"stylized"`, `"realistic"`,
    /// `"anime"`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// Target square texture resolution in pixels (e.g. `1024`,
    /// `2048`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<u32>,
    /// `true` to request a full PBR material bundle (albedo, normal,
    /// roughness, metallic). When `false`, the backend may emit only
    /// a flat albedo / diffuse texture.
    pub pbr: bool,
}

impl From<ThreedTexturizeRequest> for CoreTexturizeRequest {
    fn from(r: ThreedTexturizeRequest) -> Self {
        Self {
            prompt: r.prompt,
            reference_image: r.reference_image,
            style: r.style,
            resolution: r.resolution,
            pbr: r.pbr,
        }
    }
}

impl From<CoreTexturizeRequest> for ThreedTexturizeRequest {
    fn from(r: CoreTexturizeRequest) -> Self {
        Self {
            prompt: r.prompt,
            reference_image: r.reference_image,
            style: r.style,
            resolution: r.resolution,
            pbr: r.pbr,
        }
    }
}

/// Result of a successful `texturize` call.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedTexturizeResult {
    /// GLB bytes with the new texture (and PBR maps if any) embedded.
    pub textured_glb: Vec<u8>,
    /// MIME type of `texturedGlb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Optional out-of-band PBR map bundle. Duplicates of the maps
    /// embedded in `texturedGlb` when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pbr_maps: Option<ThreedPbrMaps>,
}

impl From<CoreTexturizeResult> for ThreedTexturizeResult {
    fn from(r: CoreTexturizeResult) -> Self {
        Self {
            textured_glb: r.textured_glb,
            mime_type: r.mime_type,
            pbr_maps: r.pbr_maps.map(ThreedPbrMaps::from),
        }
    }
}

impl From<ThreedTexturizeResult> for CoreTexturizeResult {
    fn from(r: ThreedTexturizeResult) -> Self {
        Self {
            textured_glb: r.textured_glb,
            mime_type: r.mime_type,
            pbr_maps: r.pbr_maps.map(CorePbrMaps::from),
        }
    }
}

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

/// Request parameters for the `rig` stage. Auto-rig a 3D mesh by
/// placing a skeletal armature and optionally painting skin weights.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedRigRequest {
    /// Target rig template (`"humanoid"`, `"quadruped"`, `"auto"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// `true` to apply skin-weight painting after armature placement.
    pub skin: bool,
    /// Optional pose hint (`"t-pose"`, `"a-pose"`, or backend-
    /// specific JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose_hint: Option<String>,
}

impl From<ThreedRigRequest> for CoreRigRequest {
    fn from(r: ThreedRigRequest) -> Self {
        Self {
            template: r.template,
            skin: r.skin,
            pose_hint: r.pose_hint,
        }
    }
}

impl From<CoreRigRequest> for ThreedRigRequest {
    fn from(r: CoreRigRequest) -> Self {
        Self {
            template: r.template,
            skin: r.skin,
            pose_hint: r.pose_hint,
        }
    }
}

/// Result of a successful `rig` call.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedRigResult {
    /// GLB bytes with the new armature (and skin weights, if
    /// requested) embedded.
    pub rigged_glb: Vec<u8>,
    /// MIME type of `riggedGlb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Names of bones in the produced armature, in depth-first
    /// traversal order.
    pub bone_names: Vec<String>,
}

impl From<CoreRigResult> for ThreedRigResult {
    fn from(r: CoreRigResult) -> Self {
        Self {
            rigged_glb: r.rigged_glb,
            mime_type: r.mime_type,
            bone_names: r.bone_names,
        }
    }
}

impl From<ThreedRigResult> for CoreRigResult {
    fn from(r: ThreedRigResult) -> Self {
        Self {
            rigged_glb: r.rigged_glb,
            mime_type: r.mime_type,
            bone_names: r.bone_names,
        }
    }
}

// ---------------------------------------------------------------------------
// Refine
// ---------------------------------------------------------------------------

/// Request parameters for the `refine` stage. Decimate, fill holes,
/// unwrap UVs, retopologize, smooth.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedRefineRequest {
    /// Decimate the mesh towards this triangle count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimate_target_tris: Option<u32>,
    /// `true` to fill holes via screened poisson reconstruction.
    pub fill_holes: bool,
    /// `true` to compute a new UV unwrap.
    pub unwrap_uvs: bool,
    /// `true` to retopologize the mesh (rebuild polygon layout for
    /// animation-friendly edge flow).
    pub retopologize: bool,
    /// Laplacian / Taubin smoothing iteration count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_iterations: Option<u32>,
}

impl From<ThreedRefineRequest> for CoreRefineRequest {
    fn from(r: ThreedRefineRequest) -> Self {
        Self {
            decimate_target_tris: r.decimate_target_tris,
            fill_holes: r.fill_holes,
            unwrap_uvs: r.unwrap_uvs,
            retopologize: r.retopologize,
            smooth_iterations: r.smooth_iterations,
        }
    }
}

impl From<CoreRefineRequest> for ThreedRefineRequest {
    fn from(r: CoreRefineRequest) -> Self {
        Self {
            decimate_target_tris: r.decimate_target_tris,
            fill_holes: r.fill_holes,
            unwrap_uvs: r.unwrap_uvs,
            retopologize: r.retopologize,
            smooth_iterations: r.smooth_iterations,
        }
    }
}

/// Summary statistics emitted by a `refine` call.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedRefineStats {
    /// Triangle count of the input mesh.
    pub input_tri_count: u32,
    /// Triangle count of the output (refined) mesh.
    pub output_tri_count: u32,
    /// When UV unwrapping ran, the number of UV charts the unwrap
    /// produced. `None` when UV unwrapping did not run for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_chart_count: Option<u32>,
}

impl From<CoreRefineStats> for ThreedRefineStats {
    fn from(s: CoreRefineStats) -> Self {
        Self {
            input_tri_count: s.input_tri_count,
            output_tri_count: s.output_tri_count,
            uv_chart_count: s.uv_chart_count,
        }
    }
}

impl From<ThreedRefineStats> for CoreRefineStats {
    fn from(s: ThreedRefineStats) -> Self {
        Self {
            input_tri_count: s.input_tri_count,
            output_tri_count: s.output_tri_count,
            uv_chart_count: s.uv_chart_count,
        }
    }
}

/// Result of a successful `refine` call.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedRefineResult {
    /// GLB bytes with the requested refinement passes applied.
    pub refined_glb: Vec<u8>,
    /// MIME type of `refinedGlb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Before/after statistics for the refinement run.
    pub stats: ThreedRefineStats,
}

impl From<CoreRefineResult> for ThreedRefineResult {
    fn from(r: CoreRefineResult) -> Self {
        Self {
            refined_glb: r.refined_glb,
            mime_type: r.mime_type,
            stats: ThreedRefineStats::from(r.stats),
        }
    }
}

impl From<ThreedRefineResult> for CoreRefineResult {
    fn from(r: ThreedRefineResult) -> Self {
        Self {
            refined_glb: r.refined_glb,
            mime_type: r.mime_type,
            stats: CoreRefineStats::from(r.stats),
        }
    }
}

// ---------------------------------------------------------------------------
// Animate
// ---------------------------------------------------------------------------

/// Request parameters for the `animate` stage. Drive a rigged mesh
/// from a text prompt, motion-capture clip, or driving video.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedAnimateRequest {
    /// Text-guided motion prompt (e.g. `"walks forward and waves"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Optional MP4 bytes for video-driven motion transfer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driving_video: Option<Vec<u8>>,
    /// Optional BVH motion-capture clip bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bvh_motion: Option<Vec<u8>>,
    /// Requested animation duration in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
    /// Requested animation framerate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// `true` to mark the produced animation as a seamless loop.
    pub loop_animation: bool,
}

impl From<ThreedAnimateRequest> for CoreAnimateRequest {
    fn from(r: ThreedAnimateRequest) -> Self {
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

impl From<CoreAnimateRequest> for ThreedAnimateRequest {
    fn from(r: CoreAnimateRequest) -> Self {
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

/// Result of a successful `animate` call.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThreedAnimateResult {
    /// GLB bytes with the animation track(s) embedded.
    pub animated_glb: Vec<u8>,
    /// MIME type of `animatedGlb`; always `"model/gltf-binary"`.
    pub mime_type: String,
    /// Actual produced duration in seconds (may differ from the
    /// request if the backend clamped).
    pub duration_seconds: f32,
    /// Actual produced framerate in frames per second (may differ
    /// from the request if the backend clamped).
    pub fps: u32,
}

impl From<CoreAnimateResult> for ThreedAnimateResult {
    fn from(r: CoreAnimateResult) -> Self {
        Self {
            animated_glb: r.animated_glb,
            mime_type: r.mime_type,
            duration_seconds: r.duration_seconds,
            fps: r.fps,
        }
    }
}

impl From<ThreedAnimateResult> for CoreAnimateResult {
    fn from(r: ThreedAnimateResult) -> Self {
        Self {
            animated_glb: r.animated_glb,
            mime_type: r.mime_type,
            duration_seconds: r.duration_seconds,
            fps: r.fps,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texturize_request_roundtrip() {
        let req = ThreedTexturizeRequest {
            prompt: Some("weathered bronze".into()),
            reference_image: Some(vec![1, 2, 3]),
            style: Some("realistic".into()),
            resolution: Some(2048),
            pbr: true,
        };
        let core: CoreTexturizeRequest = req.clone().into();
        let back: ThreedTexturizeRequest = core.into();
        assert_eq!(back.prompt.as_deref(), Some("weathered bronze"));
        assert_eq!(back.reference_image, Some(vec![1, 2, 3]));
        assert_eq!(back.resolution, Some(2048));
        assert!(back.pbr);
    }

    #[test]
    fn texturize_request_camel_case_json() {
        let req = ThreedTexturizeRequest {
            prompt: None,
            reference_image: Some(vec![0xAA]),
            style: None,
            resolution: None,
            pbr: false,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(
            json.contains("\"referenceImage\""),
            "expected camelCase referenceImage in {json}"
        );
        assert!(!json.contains("reference_image"));
        assert!(!json.contains("\"prompt\""));
    }

    #[test]
    fn pbr_maps_roundtrip() {
        let maps = ThreedPbrMaps {
            albedo_png: vec![1, 2, 3],
            normal_png: Some(vec![4]),
            roughness_png: None,
            metallic_png: None,
        };
        let core: CorePbrMaps = maps.clone().into();
        let back: ThreedPbrMaps = core.into();
        assert_eq!(back.albedo_png, vec![1, 2, 3]);
        assert_eq!(back.normal_png, Some(vec![4]));
    }

    #[test]
    fn refine_stats_roundtrip() {
        let stats = ThreedRefineStats {
            input_tri_count: 100_000,
            output_tri_count: 10_000,
            uv_chart_count: Some(8),
        };
        let core: CoreRefineStats = stats.clone().into();
        let back: ThreedRefineStats = core.into();
        assert_eq!(back.input_tri_count, 100_000);
        assert_eq!(back.uv_chart_count, Some(8));
    }

    #[test]
    fn animate_request_camel_case_json() {
        let req = ThreedAnimateRequest {
            prompt: Some("wave".into()),
            driving_video: None,
            bvh_motion: None,
            duration_seconds: Some(2.5),
            fps: Some(30),
            loop_animation: true,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("\"durationSeconds\":2.5"));
        assert!(json.contains("\"loopAnimation\":true"));
        assert!(!json.contains("loop_animation"));
    }
}
