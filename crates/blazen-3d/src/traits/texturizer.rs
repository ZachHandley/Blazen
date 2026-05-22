//! The [`Texturizer3dBackend`] trait — capability for applying or
//! generating a texture/material on an existing 3D mesh.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::Texturizer3dError;

/// Capability trait for engines that apply or generate a texture /
/// material on an existing 3D mesh.
///
/// Implementors take a GLB (or OBJ) mesh plus a [`TexturizeRequest`]
/// describing the desired texture (text prompt, optional reference
/// image, optional style preset, target resolution, whether to emit
/// PBR maps) and return a GLB with the new texture embedded.
///
/// Implementors should be cheap to clone (typically wrap any expensive
/// state in `Arc` internally) and safe to share across tokio tasks.
#[async_trait]
pub trait Texturizer3dBackend: Send + Sync {
    /// Apply or generate a texture/material for an existing 3D mesh.
    ///
    /// # Input format
    ///
    /// `mesh_glb` is either GLB (`model/gltf-binary`) bytes or OBJ
    /// (`model/obj`) bytes. Backends that only accept one format
    /// should return [`Texturizer3dError::InvalidInput`] for the
    /// other.
    ///
    /// # Output format
    ///
    /// The returned [`TexturizeResult::textured_glb`] is GLB bytes
    /// with the new texture (and PBR material maps if the backend
    /// produces them) embedded. PBR maps that are also returned
    /// out-of-band via [`TexturizeResult::pbr_maps`] are duplicates of
    /// the maps embedded in the GLB — callers may use whichever form
    /// is convenient.
    ///
    /// # Errors
    ///
    /// Returns [`Texturizer3dError::EngineNotAvailable`] when the
    /// backend was built without the required engine feature;
    /// [`Texturizer3dError::InvalidInput`] when `mesh_glb` is not a
    /// supported format or `request` is malformed;
    /// [`Texturizer3dError::Backend`] on inference-time failures; and
    /// [`Texturizer3dError::Unsupported`] when the request asks for a
    /// capability the active backend cannot provide (e.g. PBR maps
    /// from a backend that only emits a flat albedo).
    async fn texturize(
        &self,
        mesh_glb: &[u8],
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, Texturizer3dError>;
}

/// Request parameters for [`Texturizer3dBackend::texturize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TexturizeRequest {
    /// Text-guided texture prompt (e.g. "weathered bronze, dark
    /// patina"). Backends that only support image-guided texturing
    /// should return [`Texturizer3dError::Unsupported`] when this is
    /// set without a reference image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Image-guided reference (PNG or JPEG bytes). Backends use this
    /// as a style/appearance anchor for the produced texture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image: Option<Vec<u8>>,
    /// Backend-specific style preset (e.g. "stylized", "realistic",
    /// "anime"). Free-form string — backends document their own
    /// vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// Target texture resolution in pixels (square; e.g. `1024`,
    /// `2048`). Backends may clamp or quantize to the nearest
    /// supported size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<u32>,
    /// `true` to request a full PBR material bundle (albedo, normal,
    /// roughness, metallic). When `false`, the backend may emit only
    /// an albedo / diffuse texture.
    pub pbr: bool,
}

/// Result of a successful [`Texturizer3dBackend::texturize`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TexturizeResult {
    /// GLB bytes with the new texture (and PBR material maps if any)
    /// embedded.
    pub textured_glb: Vec<u8>,
    /// MIME type of [`Self::textured_glb`]; always
    /// `"model/gltf-binary"` for GLB output.
    pub mime_type: String,
    /// Optional out-of-band PBR map bundle. When present, these maps
    /// are duplicates of the maps embedded in [`Self::textured_glb`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pbr_maps: Option<PbrMaps>,
}

/// Bundle of PBR (physically-based rendering) material maps that
/// accompany a textured mesh.
///
/// `albedo_png` is always populated; the other channels are
/// optional and depend on what the backend produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbrMaps {
    /// Base-color / diffuse texture as PNG bytes. Always present.
    pub albedo_png: Vec<u8>,
    /// Tangent-space normal map as PNG bytes, if produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_png: Option<Vec<u8>>,
    /// Linear roughness map as PNG bytes (`0` = smooth, `255` =
    /// rough), if produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness_png: Option<Vec<u8>>,
    /// Linear metallic map as PNG bytes (`0` = dielectric, `255` =
    /// metallic), if produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic_png: Option<Vec<u8>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texturize_request_omits_empty_optionals() {
        let req = TexturizeRequest {
            prompt: None,
            reference_image: None,
            style: None,
            resolution: None,
            pbr: true,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(!json.contains("prompt"));
        assert!(!json.contains("reference_image"));
        assert!(!json.contains("style"));
        assert!(!json.contains("resolution"));
        assert!(json.contains("\"pbr\":true"));
    }

    #[test]
    fn pbr_maps_roundtrip() {
        let maps = PbrMaps {
            albedo_png: vec![1, 2, 3],
            normal_png: Some(vec![4, 5]),
            roughness_png: None,
            metallic_png: None,
        };
        let json = serde_json::to_string(&maps).expect("serialize");
        let back: PbrMaps = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.albedo_png, vec![1, 2, 3]);
        assert_eq!(back.normal_png, Some(vec![4, 5]));
        assert!(back.roughness_png.is_none());
        assert!(back.metallic_png.is_none());
    }
}
