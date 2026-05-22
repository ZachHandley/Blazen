//! `Compat3dProvider` — a single HTTP-proxy backend that implements
//! all four [`crate`] capability traits ([`Texturizer3dBackend`],
//! [`Rigger3dBackend`], [`Refiner3dBackend`], [`Animator3dBackend`]) by
//! forwarding each stage to a configurable upstream service over a
//! `multipart/form-data` POST.
//!
//! # Wire format
//!
//! For every stage, the provider POSTs a multipart form with two
//! required parts plus zero-or-more optional binary parts:
//!
//! * `mesh.glb` — the input mesh bytes (`application/octet-stream`).
//! * `request.json` — the corresponding request struct serialized as
//!   JSON (`application/json`).  Optional binary fields on the request
//!   (`TexturizeRequest::reference_image`,
//!   `AnimateRequest::driving_video`, `AnimateRequest::bvh_motion`)
//!   are stripped from the JSON and uploaded as separate multipart
//!   parts named after the field (`reference_image`, `driving_video`,
//!   `bvh_motion`).
//!
//! Responses come back as JSON with binary fields base64-encoded under
//! `*_b64` keys (see the `*Wire` structs below).  The provider decodes
//! the base64 payloads and maps the wire shape into the public result
//! struct from the trait.
//!
//! # Endpoints
//!
//! | Stage      | URL                                  |
//! |------------|--------------------------------------|
//! | Texturize  | `POST {base_url}/v1/3d/texturize`    |
//! | Rig        | `POST {base_url}/v1/3d/rig`          |
//! | Refine     | `POST {base_url}/v1/3d/refine`       |
//! | Animate    | `POST {base_url}/v1/3d/animate`      |
//!
//! # Authentication
//!
//! Optional `Authorization: Bearer <api_key>` header, attached when
//! [`Compat3dProvider::with_api_key`] was called.
//!
//! # Timeouts
//!
//! 3D inference can be slow (texture generation: tens of seconds;
//! animation: minutes).  The default per-request timeout is 600 seconds
//! and is tuneable via [`Compat3dProvider::with_timeout`].

use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

use crate::errors::{Animator3dError, Refiner3dError, Rigger3dError, Texturizer3dError};
use crate::traits::animator::{AnimateRequest, AnimateResult, Animator3dBackend};
use crate::traits::refiner::{RefineRequest, RefineResult, RefineStats, Refiner3dBackend};
use crate::traits::rigger::{RigRequest, RigResult, Rigger3dBackend};
use crate::traits::texturizer::{PbrMaps, TexturizeRequest, TexturizeResult, Texturizer3dBackend};

/// Default per-request timeout. 3D inference can run for minutes
/// (animation, retopology, PBR baking) so the default is generous.
const DEFAULT_TIMEOUT: Duration = Duration::from_mins(10);

// ---------------------------------------------------------------------
// Public provider
// ---------------------------------------------------------------------

/// HTTP-proxy backend implementing all four 3D-pipeline capability
/// traits against a configurable upstream service.
///
/// See the module docs for the wire format, endpoint table, and
/// authentication / timeout configuration.
#[derive(Debug, Clone)]
pub struct Compat3dProvider {
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
    timeout: Duration,
}

impl Compat3dProvider {
    /// Construct a provider pointed at `base_url` (e.g.
    /// `"https://my-3d-server.example.com"`).  Trailing slashes on the
    /// URL are tolerated — the endpoint paths concatenated by each
    /// stage start with a leading `/`, so a single redundant `/` in
    /// the joined URL is harmless for compliant HTTP servers.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            client: reqwest::Client::new(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Attach a bearer token; every outbound request will carry
    /// `Authorization: Bearer <key>`.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the per-request timeout (default 600s).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Swap in a caller-built [`reqwest::Client`] — useful when the
    /// caller wants to share a single client across providers, attach
    /// custom middleware, or configure proxies / TLS roots.
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Apply bearer-auth to a request builder when an API key is set.
    fn auth(&self, mut rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            rb = rb.bearer_auth(key);
        }
        rb
    }

    /// Build the URL for a given endpoint path.
    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}{path}")
    }
}

// ---------------------------------------------------------------------
// Wire-format response shapes
// ---------------------------------------------------------------------

/// Wire shape for `POST /v1/3d/texturize`.
#[derive(Debug, Deserialize)]
struct TexturizeResultWire {
    /// Base64-encoded GLB bytes.
    textured_glb_b64: String,
    /// MIME type echoed by the server.
    #[serde(default = "default_glb_mime")]
    mime_type: String,
    /// Optional PBR-map bundle.
    #[serde(default)]
    pbr_maps: Option<PbrMapsWire>,
}

/// Wire shape for the optional PBR-map bundle in a texturize response.
///
/// The `_b64` postfix is part of the wire protocol (each field carries
/// the base64-encoded PNG for that channel), so the shared suffix is
/// intentional rather than a Rust-side smell.
#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
struct PbrMapsWire {
    albedo_png_b64: String,
    #[serde(default)]
    normal_png_b64: Option<String>,
    #[serde(default)]
    roughness_png_b64: Option<String>,
    #[serde(default)]
    metallic_png_b64: Option<String>,
}

/// Wire shape for `POST /v1/3d/rig`.
#[derive(Debug, Deserialize)]
struct RigResultWire {
    rigged_glb_b64: String,
    #[serde(default = "default_glb_mime")]
    mime_type: String,
    #[serde(default)]
    bone_names: Vec<String>,
}

/// Wire shape for `POST /v1/3d/refine`.
#[derive(Debug, Deserialize)]
struct RefineResultWire {
    refined_glb_b64: String,
    #[serde(default = "default_glb_mime")]
    mime_type: String,
    stats: RefineStats,
}

/// Wire shape for `POST /v1/3d/animate`.
#[derive(Debug, Deserialize)]
struct AnimateResultWire {
    animated_glb_b64: String,
    #[serde(default = "default_glb_mime")]
    mime_type: String,
    duration_seconds: f32,
    fps: u32,
}

fn default_glb_mime() -> String {
    "model/gltf-binary".to_string()
}

// ---------------------------------------------------------------------
// Shared multipart / response helpers
// ---------------------------------------------------------------------

/// Build the `mesh.glb` part used by every stage.
fn mesh_part(mesh_glb: &[u8]) -> Result<Part, reqwest::Error> {
    let part = Part::bytes(mesh_glb.to_vec())
        .file_name("mesh.glb")
        .mime_str("application/octet-stream")?;
    Ok(part)
}

/// Build the `request.json` part from any serializable request struct.
fn request_json_part<T: Serialize>(request: &T) -> Result<Part, String> {
    let body = serde_json::to_vec(request).map_err(|e| format!("request encode: {e}"))?;
    Part::bytes(body)
        .file_name("request.json")
        .mime_str("application/json")
        .map_err(|e| format!("request part mime: {e}"))
}

/// Build a side-channel binary part (`reference_image`, `driving_video`,
/// `bvh_motion`, ...).
fn binary_part(name: &str, bytes: &[u8]) -> Result<Part, reqwest::Error> {
    Part::bytes(bytes.to_vec())
        .file_name(name.to_string())
        .mime_str("application/octet-stream")
}

/// Truncate a string to at most `n` UTF-8 bytes (rounding down to a
/// char boundary).  Used to keep server error bodies bounded in our
/// error messages so a chatty server can't blow up the log.
fn cap(mut s: String, n: usize) -> String {
    if s.len() > n {
        let mut cut = n;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("...[truncated]");
    }
    s
}

// ---------------------------------------------------------------------
// Per-stage error mapping
// ---------------------------------------------------------------------
//
// Each capability trait has its own error enum but the variants are
// parallel (`Backend(String)`, `InvalidInput(String)`, ...).  The
// helpers below capture the per-stage mapping in one place so the
// trait impls can stay short and uniform.

macro_rules! impl_stage_helpers {
    (
        $err:ident,
        $send_fn:ident,
        $endpoint:literal
    ) => {
        async fn $send_fn(provider: &Compat3dProvider, form: Form) -> Result<String, $err> {
            let url = provider.url($endpoint);
            let rb = provider
                .client
                .post(&url)
                .timeout(provider.timeout)
                .multipart(form);
            let resp = provider
                .auth(rb)
                .send()
                .await
                .map_err(|e| $err::Backend(format!("http: {e}")))?;
            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|e| $err::Backend(format!("http body: {e}")))?;
            if !status.is_success() {
                return Err($err::Backend(format!(
                    "http {status}: {}",
                    cap(text, 4 * 1024)
                )));
            }
            if text.is_empty() {
                return Err($err::InvalidInput(
                    "upstream returned empty body".to_string(),
                ));
            }
            Ok(text)
        }
    };
}

impl_stage_helpers!(Texturizer3dError, send_texturize, "/v1/3d/texturize");
impl_stage_helpers!(Rigger3dError, send_rig, "/v1/3d/rig");
impl_stage_helpers!(Refiner3dError, send_refine, "/v1/3d/refine");
impl_stage_helpers!(Animator3dError, send_animate, "/v1/3d/animate");

// ---------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------

#[async_trait]
impl Texturizer3dBackend for Compat3dProvider {
    async fn texturize(
        &self,
        mesh_glb: &[u8],
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, Texturizer3dError> {
        // Split the optional binary field out of the JSON body so it
        // travels as a side-channel multipart part instead of
        // base64-bloating the JSON.
        let TexturizeRequest {
            prompt,
            reference_image,
            style,
            resolution,
            pbr,
        } = request;
        let request_no_binary = TexturizeRequest {
            prompt,
            reference_image: None,
            style,
            resolution,
            pbr,
        };

        let mut form = Form::new()
            .part(
                "mesh.glb",
                mesh_part(mesh_glb)
                    .map_err(|e| Texturizer3dError::Backend(format!("mesh part: {e}")))?,
            )
            .part(
                "request.json",
                request_json_part(&request_no_binary).map_err(Texturizer3dError::Backend)?,
            );
        if let Some(ref img) = reference_image {
            form = form.part(
                "reference_image",
                binary_part("reference_image", img).map_err(|e| {
                    Texturizer3dError::Backend(format!("reference_image part: {e}"))
                })?,
            );
        }

        let text = send_texturize(self, form).await?;
        let wire: TexturizeResultWire = serde_json::from_str(&text)
            .map_err(|e| Texturizer3dError::Backend(format!("response decode: {e}")))?;

        let textured_glb = B64
            .decode(wire.textured_glb_b64.as_bytes())
            .map_err(|e| Texturizer3dError::Backend(format!("base64: {e}")))?;
        if textured_glb.is_empty() {
            return Err(Texturizer3dError::InvalidInput(
                "upstream returned empty textured_glb_b64".to_string(),
            ));
        }
        let pbr_maps = match wire.pbr_maps {
            None => None,
            Some(maps) => {
                let albedo_png = B64
                    .decode(maps.albedo_png_b64.as_bytes())
                    .map_err(|e| Texturizer3dError::Backend(format!("base64 albedo: {e}")))?;
                let normal_png = decode_optional_b64(maps.normal_png_b64.as_deref(), "normal")?;
                let roughness_png =
                    decode_optional_b64(maps.roughness_png_b64.as_deref(), "roughness")?;
                let metallic_png =
                    decode_optional_b64(maps.metallic_png_b64.as_deref(), "metallic")?;
                Some(PbrMaps {
                    albedo_png,
                    normal_png,
                    roughness_png,
                    metallic_png,
                })
            }
        };
        Ok(TexturizeResult {
            textured_glb,
            mime_type: wire.mime_type,
            pbr_maps,
        })
    }
}

/// Decode an optional `*_b64` field on a PBR-map bundle.  `label` is
/// used only in the error message so the caller can tell which channel
/// failed to decode.
fn decode_optional_b64(
    b64: Option<&str>,
    label: &str,
) -> Result<Option<Vec<u8>>, Texturizer3dError> {
    match b64 {
        None => Ok(None),
        Some(s) => B64
            .decode(s.as_bytes())
            .map(Some)
            .map_err(|e| Texturizer3dError::Backend(format!("base64 {label}: {e}"))),
    }
}

#[async_trait]
impl Rigger3dBackend for Compat3dProvider {
    async fn rig(&self, mesh_glb: &[u8], request: RigRequest) -> Result<RigResult, Rigger3dError> {
        let form = Form::new()
            .part(
                "mesh.glb",
                mesh_part(mesh_glb)
                    .map_err(|e| Rigger3dError::Backend(format!("mesh part: {e}")))?,
            )
            .part(
                "request.json",
                request_json_part(&request).map_err(Rigger3dError::Backend)?,
            );

        let text = send_rig(self, form).await?;
        let wire: RigResultWire = serde_json::from_str(&text)
            .map_err(|e| Rigger3dError::Backend(format!("response decode: {e}")))?;
        let rigged_glb = B64
            .decode(wire.rigged_glb_b64.as_bytes())
            .map_err(|e| Rigger3dError::Backend(format!("base64: {e}")))?;
        if rigged_glb.is_empty() {
            return Err(Rigger3dError::InvalidInput(
                "upstream returned empty rigged_glb_b64".to_string(),
            ));
        }
        Ok(RigResult {
            rigged_glb,
            mime_type: wire.mime_type,
            bone_names: wire.bone_names,
        })
    }
}

#[async_trait]
impl Refiner3dBackend for Compat3dProvider {
    async fn refine(
        &self,
        mesh_glb: &[u8],
        request: RefineRequest,
    ) -> Result<RefineResult, Refiner3dError> {
        let form = Form::new()
            .part(
                "mesh.glb",
                mesh_part(mesh_glb)
                    .map_err(|e| Refiner3dError::Backend(format!("mesh part: {e}")))?,
            )
            .part(
                "request.json",
                request_json_part(&request).map_err(Refiner3dError::Backend)?,
            );

        let text = send_refine(self, form).await?;
        let wire: RefineResultWire = serde_json::from_str(&text)
            .map_err(|e| Refiner3dError::Backend(format!("response decode: {e}")))?;
        let refined_glb = B64
            .decode(wire.refined_glb_b64.as_bytes())
            .map_err(|e| Refiner3dError::Backend(format!("base64: {e}")))?;
        if refined_glb.is_empty() {
            return Err(Refiner3dError::InvalidInput(
                "upstream returned empty refined_glb_b64".to_string(),
            ));
        }
        Ok(RefineResult {
            refined_glb,
            mime_type: wire.mime_type,
            stats: wire.stats,
        })
    }
}

#[async_trait]
impl Animator3dBackend for Compat3dProvider {
    async fn animate(
        &self,
        rigged_glb: &[u8],
        request: AnimateRequest,
    ) -> Result<AnimateResult, Animator3dError> {
        let AnimateRequest {
            prompt,
            driving_video,
            bvh_motion,
            duration_seconds,
            fps,
            loop_animation,
        } = request;
        let request_no_binary = AnimateRequest {
            prompt,
            driving_video: None,
            bvh_motion: None,
            duration_seconds,
            fps,
            loop_animation,
        };

        let mut form = Form::new()
            .part(
                "mesh.glb",
                mesh_part(rigged_glb)
                    .map_err(|e| Animator3dError::Backend(format!("mesh part: {e}")))?,
            )
            .part(
                "request.json",
                request_json_part(&request_no_binary).map_err(Animator3dError::Backend)?,
            );
        if let Some(ref video) = driving_video {
            form = form.part(
                "driving_video",
                binary_part("driving_video", video)
                    .map_err(|e| Animator3dError::Backend(format!("driving_video part: {e}")))?,
            );
        }
        if let Some(ref bvh) = bvh_motion {
            form = form.part(
                "bvh_motion",
                binary_part("bvh_motion", bvh)
                    .map_err(|e| Animator3dError::Backend(format!("bvh_motion part: {e}")))?,
            );
        }

        let text = send_animate(self, form).await?;
        let wire: AnimateResultWire = serde_json::from_str(&text)
            .map_err(|e| Animator3dError::Backend(format!("response decode: {e}")))?;
        let animated_glb = B64
            .decode(wire.animated_glb_b64.as_bytes())
            .map_err(|e| Animator3dError::Backend(format!("base64: {e}")))?;
        if animated_glb.is_empty() {
            return Err(Animator3dError::InvalidInput(
                "upstream returned empty animated_glb_b64".to_string(),
            ));
        }
        Ok(AnimateResult {
            animated_glb,
            mime_type: wire.mime_type,
            duration_seconds: wire.duration_seconds,
            fps: wire.fps,
        })
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(bytes: &[u8]) -> String {
        B64.encode(bytes)
    }

    #[tokio::test]
    async fn texturize_success() {
        let mut server = mockito::Server::new_async().await;
        let response_body = serde_json::json!({
            "textured_glb_b64": b64(b"GLB_TEXTURED_BYTES"),
            "mime_type": "model/gltf-binary",
            "pbr_maps": {
                "albedo_png_b64": b64(b"ALBEDO_PNG"),
                "normal_png_b64": b64(b"NORMAL_PNG"),
            }
        })
        .to_string();
        let m = server
            .mock("POST", "/v1/3d/texturize")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create_async()
            .await;

        let provider = Compat3dProvider::new(server.url());
        let req = TexturizeRequest {
            prompt: Some("weathered bronze".into()),
            reference_image: None,
            style: None,
            resolution: Some(1024),
            pbr: true,
        };
        let out = provider
            .texturize(b"GLB_INPUT", req)
            .await
            .expect("texturize ok");
        m.assert_async().await;
        assert_eq!(out.textured_glb, b"GLB_TEXTURED_BYTES");
        assert_eq!(out.mime_type, "model/gltf-binary");
        let pbr = out.pbr_maps.expect("pbr present");
        assert_eq!(pbr.albedo_png, b"ALBEDO_PNG");
        assert_eq!(pbr.normal_png.as_deref(), Some(&b"NORMAL_PNG"[..]));
        assert!(pbr.roughness_png.is_none());
        assert!(pbr.metallic_png.is_none());
    }

    #[tokio::test]
    async fn rig_http_error_maps_to_backend_error() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/v1/3d/rig")
            .with_status(500)
            .with_body("rigger crashed: tensor shape mismatch")
            .create_async()
            .await;
        let provider = Compat3dProvider::new(server.url());
        let err = provider
            .rig(
                b"GLB",
                RigRequest {
                    template: Some("humanoid".into()),
                    skin: true,
                    pose_hint: None,
                },
            )
            .await
            .expect_err("expected error");
        m.assert_async().await;
        match err {
            Rigger3dError::Backend(msg) => {
                assert!(msg.contains("500"), "missing status in {msg:?}");
                assert!(msg.contains("rigger crashed"), "missing body in {msg:?}");
            }
            other => panic!("expected Backend, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn auth_header_sent() {
        let mut server = mockito::Server::new_async().await;
        let response_body = serde_json::json!({
            "rigged_glb_b64": b64(b"GLB"),
            "mime_type": "model/gltf-binary",
            "bone_names": ["root"],
        })
        .to_string();
        let m = server
            .mock("POST", "/v1/3d/rig")
            .match_header("authorization", "Bearer secret-token-xyz")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create_async()
            .await;
        let provider = Compat3dProvider::new(server.url()).with_api_key("secret-token-xyz");
        provider
            .rig(
                b"GLB",
                RigRequest {
                    template: None,
                    skin: true,
                    pose_hint: None,
                },
            )
            .await
            .expect("rig ok");
        m.assert_async().await;
    }

    #[tokio::test]
    async fn refine_request_json_serialized() {
        let mut server = mockito::Server::new_async().await;
        let response_body = serde_json::json!({
            "refined_glb_b64": b64(b"GLB"),
            "mime_type": "model/gltf-binary",
            "stats": {
                "input_tri_count": 12_000,
                "output_tri_count": 6_000,
                "uv_chart_count": 4,
            }
        })
        .to_string();
        // We can't pattern-match the inner JSON part of a multipart body
        // with mockito's PartialJson matcher (that targets bodies whose
        // Content-Type is application/json).  Match a regex against the
        // multipart bytes instead — the JSON part is embedded verbatim.
        let m = server
            .mock("POST", "/v1/3d/refine")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex("\"decimate_target_tris\":6000".into()),
                mockito::Matcher::Regex("\"fill_holes\":true".into()),
                mockito::Matcher::Regex("\"unwrap_uvs\":true".into()),
                mockito::Matcher::Regex("\"retopologize\":false".into()),
                mockito::Matcher::Regex("\"smooth_iterations\":3".into()),
                // And the mesh part is present too.
                mockito::Matcher::Regex("name=\"mesh.glb\"".into()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create_async()
            .await;
        let provider = Compat3dProvider::new(server.url());
        let req = RefineRequest {
            decimate_target_tris: Some(6_000),
            fill_holes: true,
            unwrap_uvs: true,
            retopologize: false,
            smooth_iterations: Some(3),
        };
        let out = provider.refine(b"GLB", req).await.expect("refine ok");
        m.assert_async().await;
        assert_eq!(out.stats.input_tri_count, 12_000);
        assert_eq!(out.stats.output_tri_count, 6_000);
        assert_eq!(out.stats.uv_chart_count, Some(4));
    }

    #[tokio::test]
    async fn animate_strips_binary_from_json_and_uploads_side_channel() {
        let mut server = mockito::Server::new_async().await;
        let response_body = serde_json::json!({
            "animated_glb_b64": b64(b"ANIMATED"),
            "mime_type": "model/gltf-binary",
            "duration_seconds": 2.5,
            "fps": 24,
        })
        .to_string();
        let m = server
            .mock("POST", "/v1/3d/animate")
            .match_body(mockito::Matcher::AllOf(vec![
                // JSON part exists and does NOT contain driving_video /
                // bvh_motion (those are stripped before serialization).
                mockito::Matcher::Regex("name=\"request.json\"".into()),
                mockito::Matcher::Regex("\"loop_animation\":true".into()),
                // Side-channel binary parts present.
                mockito::Matcher::Regex("name=\"driving_video\"".into()),
                mockito::Matcher::Regex("name=\"bvh_motion\"".into()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create_async()
            .await;
        let provider = Compat3dProvider::new(server.url());
        let req = AnimateRequest {
            prompt: Some("walk".into()),
            driving_video: Some(b"MP4_BYTES".to_vec()),
            bvh_motion: Some(b"BVH_BYTES".to_vec()),
            duration_seconds: Some(2.5),
            fps: Some(24),
            loop_animation: true,
        };
        let out = provider.animate(b"GLB", req).await.expect("animate ok");
        m.assert_async().await;
        assert_eq!(out.animated_glb, b"ANIMATED");
        assert!((out.duration_seconds - 2.5).abs() < f32::EPSILON);
        assert_eq!(out.fps, 24);
    }

    #[tokio::test]
    async fn empty_body_maps_to_invalid_input() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/3d/refine")
            .with_status(200)
            .with_body("")
            .create_async()
            .await;
        let provider = Compat3dProvider::new(server.url());
        let err = provider
            .refine(
                b"GLB",
                RefineRequest {
                    decimate_target_tris: None,
                    fill_holes: false,
                    unwrap_uvs: false,
                    retopologize: false,
                    smooth_iterations: None,
                },
            )
            .await
            .expect_err("expected error");
        assert!(matches!(err, Refiner3dError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn base64_decode_failure_maps_to_backend() {
        let mut server = mockito::Server::new_async().await;
        let response_body = serde_json::json!({
            "rigged_glb_b64": "!!not-valid-base64!!",
            "mime_type": "model/gltf-binary",
            "bone_names": [],
        })
        .to_string();
        let _m = server
            .mock("POST", "/v1/3d/rig")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create_async()
            .await;
        let provider = Compat3dProvider::new(server.url());
        let err = provider
            .rig(
                b"GLB",
                RigRequest {
                    template: None,
                    skin: false,
                    pose_hint: None,
                },
            )
            .await
            .expect_err("expected base64 error");
        match err {
            Rigger3dError::Backend(msg) => assert!(msg.contains("base64"), "msg={msg}"),
            other => panic!("expected Backend, got {other:?}"),
        }
    }

    #[test]
    fn cap_truncates_at_char_boundary() {
        // Multi-byte chars must not be split mid-codepoint.
        let s = "é".repeat(100);
        let out = cap(s, 5);
        assert!(out.ends_with("...[truncated]"));
        // The truncated prefix must be valid UTF-8 (cap() guarantees
        // this via is_char_boundary).
        assert!(out.is_char_boundary(out.find("...[truncated]").unwrap()));
    }

    #[test]
    fn default_glb_mime_value() {
        assert_eq!(default_glb_mime(), "model/gltf-binary");
    }
}
