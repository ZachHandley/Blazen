//! 3D per-engine `#[uniffi::Object]` providers — populated by P4.2.three_d.
//!
//! Wraps the canonical concrete providers from
//! [`blazen_llm`] (`TripoSrProvider`, `Compat3dProvider`) so each
//! UniFFI-supported foreign language gets a real per-engine class
//! instead of a single capability-erased handle. Method names mirror
//! the central [`crate::compute::ThreeDModel`] surface so consumers
//! moving across the deprecation boundary (P4.3 / P4.4 / P4.5) keep
//! the same call sites.
//!
//! Both providers ship behind their respective cargo features:
//!
//! - [`TripoSrProvider`] — `triposr`. Native candle image-to-3D.
//! - [`Compat3dProvider`] — `threed-compat-proxy`. HTTP-proxy newtype
//!   whose `generate_from_image` surfaces
//!   [`crate::errors::BlazenError::Unsupported`] (the upstream wire
//!   contract only covers the texturize / rig / refine / animate
//!   post-processing stages — base generation must come from a
//!   different backend like [`TripoSrProvider`]).

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use crate::compute::ThreeDGenerateResult;
use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// TripoSrProvider — native candle image-to-3D
// ---------------------------------------------------------------------------

/// `TripoSR` single-image-to-3D provider.
///
/// Wraps [`blazen_llm::TripoSrProvider`] (the candle-runtime
/// `TripoSrBackend` newtype + `ProviderMetadata` stamp). Construction
/// loads the weights — either from `weights_path` (local directory)
/// or by downloading from a Hugging Face repo (defaulting to
/// `"stabilityai/TripoSR"`).
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

        let result = self.inner.generate_3d(request).await?;

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

// ---------------------------------------------------------------------------
// Compat3dProvider intentionally NOT re-exposed here.
// ---------------------------------------------------------------------------
//
// The existing `crate::threed::Compat3dProvider` already exposes the
// HTTP-proxy surface for texturize / rig / refine / animate stages
// (the actual public API of the compat-proxy). A second
// `Compat3dProvider` that only implements `ThreeDProvider::generate_*`
// (returning Unsupported for everything) would collide at the UniFFI
// metadata layer (`UNIFFI_META_..._CONSTRUCTOR_COMPAT3DPROVIDER_NEW`)
// because UniFFI requires globally unique symbol names per type+constructor.
//
// Once the `crate::threed` module is migrated to the concrete/ layout
// in a follow-up sub-wave, the unified `Compat3dProvider` lives here.
