//! 3D concrete provider classes — populated by P4.1.g.
//!
//! Two engines are exposed here today:
//!
//! - [`TripoSrProvider`] — native single-image-to-3D backend driven by
//!   the candle `TripoSrBackend` in `blazen-3d`. Gated on the `triposr`
//!   cargo feature.
//! - [`Compat3dProvider`] — HTTP-proxy newtype around the
//!   `blazen_3d::backends::compat::Compat3dProvider` multipart-forward
//!   client. The upstream proxy implements the texturize / rig / refine
//!   / animate post-processing stages (not generation) so the
//!   [`crate::providers::capabilities::ThreeDProvider`] impl below
//!   returns [`crate::error::BlazenError::Unsupported`] from
//!   `generate_3d`. Gated on the `threed-compat-proxy` cargo feature.
//!
//! Each provider stamps a [`crate::providers::root::ProviderMetadata`]
//! at construction time and exposes the polymorphic
//! [`crate::providers::root::BaseProvider`] root so callers can store it
//! behind `Arc<dyn BaseProvider>` for capability-erased collections.

#![allow(dead_code, unused_imports)]

// ---------------------------------------------------------------------------
// TripoSrProvider — native candle image-to-3D
// ---------------------------------------------------------------------------

/// `TripoSR` single-image-to-3D provider.
///
/// Wraps the candle-runtime
/// [`blazen_3d::backends::triposr::TripoSrBackend`]. Weights are loaded
/// either from a local directory (when `weights_path` is supplied) or
/// downloaded from a Hugging Face repo (defaulting to
/// `stabilityai/TripoSR`). The native `ThreeDGeneration` impl that
/// drives `pipeline().image_to_glb(...)` lives at
/// [`crate::backends::triposr`] and is reused here verbatim through
/// the inner `Arc<TripoSrBackend>` — see that module for the
/// image-fetch / decode / spawn-blocking pipeline.
#[cfg(feature = "triposr")]
pub struct TripoSrProvider {
    inner: std::sync::Arc<blazen_3d::backends::triposr::TripoSrBackend>,
    metadata: crate::providers::root::ProviderMetadata,
}

#[cfg(feature = "triposr")]
impl TripoSrProvider {
    /// Construct a `TripoSR` provider.
    ///
    /// - `hf_repo_id`: `HuggingFace` repo to fetch weights from. Defaults
    ///   to `"stabilityai/TripoSR"` when `None`.
    /// - `revision`: optional branch / tag / commit pin on `hf_repo_id`.
    /// - `weights_path`: pre-resolved local weights directory. When
    ///   supplied, the HF download is skipped entirely and `hf_repo_id`
    ///   / `revision` are ignored.
    ///
    /// Construction mirrors `new_triposr_3d_model` in
    /// `crates/blazen-uniffi/src/compute.rs` so the binding-side
    /// factories and this consumer-facing class agree on defaults and
    /// error semantics.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::BlazenError::Provider`] with provider
    /// `"triposr"` if weights cannot be loaded (missing files, HF
    /// download failure, candle tensor parse error).
    pub async fn new(
        hf_repo_id: Option<String>,
        revision: Option<String>,
        weights_path: Option<String>,
    ) -> Result<Self, crate::error::BlazenError> {
        use std::path::Path;

        let device = candle_core::Device::Cpu;
        let backend = if let Some(path) = weights_path.as_deref() {
            blazen_3d::backends::triposr::TripoSrBackend::load_from_paths(Path::new(path), &device)
                .map_err(|e| crate::error::BlazenError::provider("triposr", e.to_string()))?
        } else {
            let repo = hf_repo_id.unwrap_or_else(|| "stabilityai/TripoSR".to_owned());
            blazen_3d::backends::triposr::TripoSrBackend::load_from_hf(
                &repo,
                revision.as_deref(),
                &device,
            )
            .await
            .map_err(|e| crate::error::BlazenError::provider("triposr", e.to_string()))?
        };

        let metadata = crate::providers::root::ProviderMetadata::new(
            "triposr",
            crate::providers::root::CapabilityKind::ThreeD,
        );

        Ok(Self {
            inner: std::sync::Arc::new(backend),
            metadata,
        })
    }

    /// Construct from an already-loaded backend handle. Useful when the
    /// caller wants to share one weights load across multiple consumer
    /// surfaces (e.g. capability-erased `Arc<dyn BaseProvider>` and a
    /// direct typed handle).
    #[must_use]
    pub fn from_backend(
        backend: std::sync::Arc<blazen_3d::backends::triposr::TripoSrBackend>,
    ) -> Self {
        Self {
            inner: backend,
            metadata: crate::providers::root::ProviderMetadata::new(
                "triposr",
                crate::providers::root::CapabilityKind::ThreeD,
            ),
        }
    }

    /// Borrow the inner backend — escape hatch for advanced callers
    /// that need to drive the candle pipeline directly (custom
    /// `image_to_glb` invocations, batched inference, etc.).
    #[must_use]
    pub fn backend(&self) -> &std::sync::Arc<blazen_3d::backends::triposr::TripoSrBackend> {
        &self.inner
    }
}

#[cfg(feature = "triposr")]
impl std::fmt::Debug for TripoSrProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TripoSrProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "triposr")]
impl crate::providers::root::BaseProvider for TripoSrProvider {
    fn metadata(&self) -> &crate::providers::root::ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "triposr")]
#[async_trait::async_trait]
impl crate::providers::capabilities::ThreeDProvider for TripoSrProvider {
    async fn generate_3d(
        &self,
        request: crate::compute::requests::ThreeDRequest,
    ) -> Result<crate::compute::results::ThreeDResult, crate::error::BlazenError> {
        // Delegate to the existing `ThreeDGeneration` impl on
        // `TripoSrBackend` (lives at `crate::backends::triposr`). It
        // already handles image fetch + decode + spawn_blocking around
        // the candle pipeline + Validation early-out on text-only
        // requests, so wrapping it here is a one-line dispatch.
        <blazen_3d::backends::triposr::TripoSrBackend as crate::compute::ThreeDGeneration>::generate_3d(
            &*self.inner,
            request,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Compat3dProvider — HTTP-proxy newtype
// ---------------------------------------------------------------------------

/// HTTP-proxy 3D provider.
///
/// Newtype around [`blazen_3d::backends::compat::Compat3dProvider`],
/// which forwards multipart requests to a configurable upstream
/// service. The upstream proxy implements the **post-processing**
/// stages of the 3D pipeline (texturize / rig / refine / animate) and
/// does *not* expose a text-to-3D / image-to-3D generation endpoint —
/// so [`crate::providers::capabilities::ThreeDProvider::generate_3d`]
/// returns [`crate::error::BlazenError::Unsupported`] from this
/// provider.
///
/// To drive the post-processing capabilities, hold the inner
/// `blazen-3d` provider via [`Compat3dProvider::backend`] and call the
/// `Texturizer3dBackend` / `Rigger3dBackend` / `Refiner3dBackend` /
/// `Animator3dBackend` trait methods directly. Those traits live in
/// `blazen-3d`; bringing them into the `blazen-llm` capability surface
/// is tracked separately.
#[cfg(feature = "threed-compat-proxy")]
pub struct Compat3dProvider {
    inner: std::sync::Arc<blazen_3d::backends::compat::Compat3dProvider>,
    metadata: crate::providers::root::ProviderMetadata,
}

#[cfg(feature = "threed-compat-proxy")]
impl Compat3dProvider {
    /// Construct a compat-proxy provider pointed at `base_url`
    /// (e.g. `"https://my-3d-server.example.com"`). When `api_key` is
    /// `Some`, every outbound request carries
    /// `Authorization: Bearer <key>`.
    ///
    /// The default per-request timeout (600s) and `reqwest::Client`
    /// from the inner provider are used; advanced callers that need to
    /// tune those can construct the inner provider themselves and use
    /// [`Compat3dProvider::from_inner`].
    #[must_use]
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let mut inner = blazen_3d::backends::compat::Compat3dProvider::new(base_url);
        if let Some(key) = api_key {
            inner = inner.with_api_key(key);
        }
        Self::from_inner(std::sync::Arc::new(inner))
    }

    /// Wrap an already-built inner provider. Useful when the caller
    /// has tuned the timeout / shared a `reqwest::Client` /
    /// configured TLS roots through the inner builder methods.
    #[must_use]
    pub fn from_inner(
        inner: std::sync::Arc<blazen_3d::backends::compat::Compat3dProvider>,
    ) -> Self {
        let metadata = crate::providers::root::ProviderMetadata::new(
            "compat-3d",
            crate::providers::root::CapabilityKind::ThreeD,
        );
        Self { inner, metadata }
    }

    /// Borrow the inner `blazen-3d` provider — drive the texturize /
    /// rig / refine / animate capability traits directly through this
    /// handle.
    #[must_use]
    pub fn backend(&self) -> &std::sync::Arc<blazen_3d::backends::compat::Compat3dProvider> {
        &self.inner
    }
}

#[cfg(feature = "threed-compat-proxy")]
impl std::fmt::Debug for Compat3dProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compat3dProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "threed-compat-proxy")]
impl crate::providers::root::BaseProvider for Compat3dProvider {
    fn metadata(&self) -> &crate::providers::root::ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "threed-compat-proxy")]
#[async_trait::async_trait]
impl crate::providers::capabilities::ThreeDProvider for Compat3dProvider {
    async fn generate_3d(
        &self,
        _request: crate::compute::requests::ThreeDRequest,
    ) -> Result<crate::compute::results::ThreeDResult, crate::error::BlazenError> {
        // The compat-proxy upstream only exposes the post-processing
        // stages of the 3D pipeline — generation itself isn't part of
        // the wire contract. Surface that cleanly instead of forwarding
        // a request the upstream will reject with an opaque 404.
        Err(crate::error::BlazenError::unsupported(
            "Compat3dProvider is an HTTP-proxy for the texturize / rig / refine / animate \
             post-processing stages — text-to-3D / image-to-3D generation is not part of the \
             compat-proxy wire contract. Use TripoSrProvider (or another generation backend) to \
             produce the base mesh, then forward the result through Compat3dProvider's inner \
             `Texturizer3dBackend` / `Rigger3dBackend` / `Refiner3dBackend` / `Animator3dBackend` \
             trait methods.",
        ))
    }
}
