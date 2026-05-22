//! The [`Animator3dBackend`] trait — capability for animating a
//! rigged 3D mesh from a text prompt, motion-capture clip, or driving
//! video.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::Animator3dError;

/// Capability trait for engines that animate a rigged 3D mesh.
///
/// Implementors take a rigged GLB plus an [`AnimateRequest`]
/// describing the desired motion (text prompt, driving video, or BVH
/// motion-capture clip) and return a GLB containing one or more
/// animation tracks.
///
/// Implementors should be cheap to clone (typically wrap any expensive
/// state in `Arc` internally) and safe to share across tokio tasks.
#[async_trait]
pub trait Animator3dBackend: Send + Sync {
    /// Animate a rigged 3D mesh according to a text prompt, motion-
    /// capture data, or driving video.
    ///
    /// # Input format
    ///
    /// `rigged_glb` is GLB (`model/gltf-binary`) bytes containing a
    /// rigged mesh — i.e. the output of a [`crate::Rigger3dBackend`]
    /// run, or an equivalently-prepared asset.
    ///
    /// # Motion source priority
    ///
    /// When multiple inputs are supplied in a single request, the
    /// backend chooses one source — typically in the priority order
    /// `bvh_motion > driving_video > prompt`. Backends document
    /// their own resolution policy.
    ///
    /// # Output format
    ///
    /// The returned [`AnimateResult::animated_glb`] is GLB bytes with
    /// the animation track(s) embedded.
    /// [`AnimateResult::duration_seconds`] and [`AnimateResult::fps`]
    /// report the actual produced timing — backends may clamp the
    /// requested duration / framerate to their supported range.
    ///
    /// # Errors
    ///
    /// Returns [`Animator3dError::EngineNotAvailable`] when the
    /// backend was built without the required engine feature;
    /// [`Animator3dError::InvalidInput`] when `rigged_glb` lacks an
    /// armature or `request` is malformed;
    /// [`Animator3dError::Backend`] on inference-time failures; and
    /// [`Animator3dError::Unsupported`] when the active backend
    /// cannot consume the supplied motion source (e.g. video-driven
    /// motion on a text-only animator).
    async fn animate(
        &self,
        rigged_glb: &[u8],
        request: AnimateRequest,
    ) -> Result<AnimateResult, Animator3dError>;
}

/// Request parameters for [`Animator3dBackend::animate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimateRequest {
    /// Text-guided motion prompt (e.g. "walks forward and waves").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Driving video as MP4 bytes. Backends use this for video-driven
    /// motion transfer (the produced animation mirrors the motion of
    /// the subject in the video).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driving_video: Option<Vec<u8>>,
    /// BVH motion-capture clip as raw bytes. When present, the
    /// backend retargets the BVH skeleton onto the rigged mesh's
    /// armature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bvh_motion: Option<Vec<u8>>,
    /// Requested animation duration in seconds. Backends may clamp to
    /// their supported range; the actually-produced duration is
    /// reported in [`AnimateResult::duration_seconds`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
    /// Requested animation framerate in frames per second. Backends
    /// may clamp or round; the actually-produced framerate is
    /// reported in [`AnimateResult::fps`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// `true` to mark the produced animation as a seamless loop.
    /// Backends that cannot guarantee a seamless loop should still
    /// honour the flag at the GLB metadata level.
    pub loop_animation: bool,
}

/// Result of a successful [`Animator3dBackend::animate`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimateResult {
    /// GLB bytes with the animation track(s) embedded.
    pub animated_glb: Vec<u8>,
    /// MIME type of [`Self::animated_glb`]; always
    /// `"model/gltf-binary"` for GLB output.
    pub mime_type: String,
    /// Actual produced duration in seconds (may differ from the
    /// requested duration if the backend clamped).
    pub duration_seconds: f32,
    /// Actual produced framerate in frames per second (may differ
    /// from the requested framerate if the backend clamped).
    pub fps: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animate_request_omits_empty_optionals() {
        let req = AnimateRequest {
            prompt: Some("walks".into()),
            driving_video: None,
            bvh_motion: None,
            duration_seconds: None,
            fps: None,
            loop_animation: false,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("prompt"));
        assert!(!json.contains("driving_video"));
        assert!(!json.contains("bvh_motion"));
        assert!(!json.contains("duration_seconds"));
        assert!(!json.contains("fps"));
        assert!(json.contains("\"loop_animation\":false"));
    }

    #[test]
    fn animate_result_roundtrip() {
        let result = AnimateResult {
            animated_glb: vec![9, 9, 9],
            mime_type: "model/gltf-binary".into(),
            duration_seconds: 3.5,
            fps: 30,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let back: AnimateResult = serde_json::from_str(&json).expect("deserialize");
        assert!((back.duration_seconds - 3.5).abs() < f32::EPSILON);
        assert_eq!(back.fps, 30);
    }
}
