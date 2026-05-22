//! The [`Rigger3dBackend`] trait — capability for auto-rigging a 3D
//! mesh (skeletal armature + skin weights).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::Rigger3dError;

/// Capability trait for engines that auto-rig a 3D mesh by placing a
/// skeletal armature and optionally painting skin weights.
///
/// Implementors should be cheap to clone (typically wrap any expensive
/// state in `Arc` internally) and safe to share across tokio tasks.
#[async_trait]
pub trait Rigger3dBackend: Send + Sync {
    /// Auto-rig a 3D mesh, producing a GLB with skeletal armature and
    /// skin weights embedded.
    ///
    /// # Input format
    ///
    /// `mesh_glb` is GLB (`model/gltf-binary`) bytes containing a
    /// single mesh. The mesh should be roughly in its rest pose; some
    /// backends accept a [`RigRequest::pose_hint`] to indicate
    /// whether the mesh is in T-pose, A-pose, or a backend-specific
    /// custom pose.
    ///
    /// # Output format
    ///
    /// The returned [`RigResult::rigged_glb`] is GLB bytes with the
    /// armature attached and (when [`RigRequest::skin`] is `true`)
    /// vertex skin weights embedded.
    ///
    /// # Errors
    ///
    /// Returns [`Rigger3dError::EngineNotAvailable`] when the backend
    /// was built without the required engine feature;
    /// [`Rigger3dError::InvalidInput`] when `mesh_glb` is malformed or
    /// `request.template` is not recognised;
    /// [`Rigger3dError::Backend`] on inference-time failures; and
    /// [`Rigger3dError::Unsupported`] when the active backend cannot
    /// satisfy the requested template (e.g. a humanoid-only rigger
    /// asked to produce a quadruped armature).
    async fn rig(&self, mesh_glb: &[u8], request: RigRequest) -> Result<RigResult, Rigger3dError>;
}

/// Request parameters for [`Rigger3dBackend::rig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigRequest {
    /// Target rig template (e.g. `"humanoid"`, `"quadruped"`,
    /// `"auto"`). Free-form string — backends document their own
    /// vocabulary; `"auto"` typically lets the backend infer the
    /// topology from the mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// `true` to apply skin-weight painting after armature placement.
    /// When `false`, the returned GLB carries the armature only and
    /// callers are expected to paint weights themselves (or with a
    /// separate tool).
    pub skin: bool,
    /// Optional pose hint. Either a well-known string (`"t-pose"`,
    /// `"a-pose"`) or a JSON-encoded blob of custom keypoints — the
    /// exact schema is backend-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose_hint: Option<String>,
}

/// Result of a successful [`Rigger3dBackend::rig`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigResult {
    /// GLB bytes with the new armature (and skin weights, if
    /// requested) embedded.
    pub rigged_glb: Vec<u8>,
    /// MIME type of [`Self::rigged_glb`]; always
    /// `"model/gltf-binary"` for GLB output.
    pub mime_type: String,
    /// Names of bones in the produced armature, in depth-first
    /// traversal order from the root bone.
    pub bone_names: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rig_request_omits_empty_optionals() {
        let req = RigRequest {
            template: None,
            skin: true,
            pose_hint: None,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(!json.contains("template"));
        assert!(!json.contains("pose_hint"));
        assert!(json.contains("\"skin\":true"));
    }

    #[test]
    fn rig_result_roundtrip() {
        let result = RigResult {
            rigged_glb: vec![1, 2, 3],
            mime_type: "model/gltf-binary".into(),
            bone_names: vec!["root".into(), "spine".into(), "head".into()],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let back: RigResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.bone_names.len(), 3);
        assert_eq!(back.bone_names[0], "root");
    }
}
