//! Per-engine image-generation `#[uniffi::Object]` providers.
//!
//! Each `<Engine>Provider` here is a thin UniFFI-exported wrapper around
//! the matching canonical concrete provider in
//! [`blazen_llm::providers::concrete::image`]. Foreign bindgen
//! (Go / Swift / Kotlin / Ruby) emits a real class per engine —
//! `DiffusionProvider` (local stable-diffusion.cpp, gated on the
//! `diffusion` feature) and `FalImageGenProvider` (fal.ai cloud
//! image-gen) — rather than overloading the central [`crate::compute::ImageGenModel`]
//! opaque factory.
//!
//! The polymorphic [`crate::concrete::bases::ImageGenProvider`] trait is
//! itself gated on `feature = "diffusion"` to match the upstream
//! capability concretes. The `FalImageGenProvider` here is therefore
//! `#[cfg(feature = "diffusion")]`-gated as well — without that feature
//! the polymorphic trait isn't in scope. The diffusion-rs engine is the
//! only local image backend today; the fal cloud wrapper shares the same
//! capability surface so it carries the same gate.
//!
//! Generation flows through
//! [`blazen_llm::providers::capabilities::ImageGenProvider::generate_image`]
//! using a builder that lifts the flat UniFFI args (`prompt` + optional
//! `width`/`height`) into the richer
//! [`blazen_llm::compute::ImageRequest`] DTO. The wire-format
//! [`crate::compute::ImageGenResult`] mirrors the existing
//! `ImageGenModel` shape so foreign callers see a consistent result type
//! across both the central factory and these per-engine concretes.

#![cfg(feature = "diffusion")]
#![allow(unused_imports)]

use std::sync::Arc;

use crate::compute::ImageGenResult;
use crate::errors::BlazenError;
use crate::llm::Media;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Lift a flat UniFFI arg triple into the inner
/// [`blazen_llm::compute::ImageRequest`] DTO.
fn build_image_request(
    prompt: String,
    width: Option<u32>,
    height: Option<u32>,
) -> blazen_llm::compute::ImageRequest {
    let mut req = blazen_llm::compute::ImageRequest::new(prompt);
    if let (Some(w), Some(h)) = (width, height) {
        req = req.with_size(w, h);
    }
    req
}

/// Project the inner [`blazen_llm::compute::ImageResult`] onto the
/// uniffi wire-format [`ImageGenResult`].
///
/// Mirrors the existing per-image mapping in
/// [`crate::compute`] so URL-only outputs keep landing in
/// `Media::data_base64` and `mime_type` reflects the upstream
/// [`blazen_llm::MediaType`].
fn project_image_result(result: blazen_llm::compute::ImageResult) -> ImageGenResult {
    let images = result
        .images
        .iter()
        .map(|img| {
            let mime_type = img.media.media_type.mime().to_owned();
            let data_base64 = img
                .media
                .base64
                .clone()
                .or_else(|| img.media.url.clone())
                .or_else(|| img.media.raw_content.clone())
                .unwrap_or_default();
            Media {
                kind: "image".to_owned(),
                mime_type,
                data_base64,
            }
        })
        .collect();
    ImageGenResult { images }
}

// ---------------------------------------------------------------------------
// DiffusionProvider — local stable-diffusion.cpp text-to-image
// ---------------------------------------------------------------------------

/// Concrete image-gen provider wrapping
/// [`blazen_llm::providers::concrete::image::DiffusionProvider`].
///
/// Backed by the local `diffusion-rs` (stable-diffusion.cpp) pipeline.
/// Configuration is passed as a JSON-encoded
/// [`blazen_image_diffusion::DiffusionOptions`] payload — the upstream
/// options struct carries eight fields (model id, device, width,
/// height, inference steps, guidance scale, scheduler, cache dir) and
/// `DiffusionOptions: serde::Deserialize`, so a single JSON string
/// keeps the UniFFI surface flat while still exposing every knob.
#[derive(uniffi::Object)]
pub struct DiffusionProvider {
    inner: Arc<blazen_llm::providers::concrete::image::DiffusionProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl DiffusionProvider {
    /// Construct a new local diffusion-rs provider.
    ///
    /// `options_json` is an optional JSON-encoded
    /// [`blazen_llm::DiffusionOptions`]. When `None`, the upstream
    /// defaults are used (512x512, EulerA scheduler, 20 inference
    /// steps). Invalid JSON or option-validation failures surface as
    /// [`BlazenError::Provider`] with provider id `"diffusion-rs"`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] when the JSON cannot be
    /// deserialized or when the upstream options validator rejects the
    /// payload (zero dimensions, empty model id, etc.).
    #[uniffi::constructor]
    pub fn new(options_json: Option<String>) -> Result<Arc<Self>, BlazenError> {
        let options: blazen_llm::DiffusionOptions = match options_json {
            Some(ref json) if !json.is_empty() => {
                serde_json::from_str(json).map_err(|e| BlazenError::Provider {
                    kind: "DiffusionOptions".to_owned(),
                    message: format!("failed to parse options_json: {e}"),
                    provider: Some("diffusion-rs".to_owned()),
                    status: None,
                    endpoint: None,
                    request_id: None,
                    detail: None,
                    retry_after_ms: None,
                })?
            }
            _ => blazen_llm::DiffusionOptions::default(),
        };
        let inner = blazen_llm::providers::concrete::image::DiffusionProvider::new(options)
            .map_err(|e| BlazenError::Provider {
                kind: "DiffusionInit".to_owned(),
                message: e.to_string(),
                provider: Some("diffusion-rs".to_owned()),
                status: None,
                endpoint: None,
                request_id: None,
                detail: None,
                retry_after_ms: None,
            })?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Generate an image from a text prompt.
    ///
    /// `width` and `height` are paired: pass both or neither. When
    /// either is `None` the inner request omits the size override and
    /// the provider's default dimensions apply.
    pub async fn generate_image(
        self: Arc<Self>,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError> {
        use blazen_llm::providers::capabilities::ImageGenProvider as _;
        let req = build_image_request(prompt, width, height);
        let result = self.inner.generate_image(req).await?;
        Ok(project_image_result(result))
    }
}

#[uniffi::export]
impl DiffusionProvider {
    /// Synchronous variant of [`generate_image`](Self::generate_image).
    pub fn generate_image_blocking(
        self: Arc<Self>,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.generate_image(prompt, width, height).await })
    }
}

// ---------------------------------------------------------------------------
// FalImageGenProvider — fal.ai cloud image-gen
// ---------------------------------------------------------------------------

/// Concrete image-gen provider wrapping
/// [`blazen_llm::providers::concrete::image::FalImageGenProvider`].
///
/// Routes through the shared fal.ai queue executor. Image model
/// selection happens per request via
/// [`blazen_llm::compute::ImageRequest::model`]; the optional
/// `default_model` passed at construction is stamped onto outgoing
/// requests by the upstream concrete when the caller doesn't override.
/// When neither is set, the fal bridge's built-in default model is used.
#[derive(uniffi::Object)]
pub struct FalImageGenProvider {
    inner: Arc<blazen_llm::providers::concrete::image::FalImageGenProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalImageGenProvider {
    /// Construct a new fal.ai image-gen provider.
    ///
    /// `api_key` is the fal.ai API key. `default_model` is an optional
    /// default image model id (e.g. `"fal-ai/flux/schnell"`).
    /// `base_url` overrides the default fal queue URL
    /// (`https://queue.fal.run`) — used for proxies / staging
    /// environments.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(
        api_key: String,
        default_model: Option<String>,
        base_url: Option<String>,
    ) -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::image::FalImageGenProvider::new(
            api_key,
            default_model,
            base_url,
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Generate an image from a text prompt.
    pub async fn generate_image(
        self: Arc<Self>,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError> {
        use blazen_llm::providers::capabilities::ImageGenProvider as _;
        let req = build_image_request(prompt, width, height);
        let result = self.inner.generate_image(req).await?;
        Ok(project_image_result(result))
    }
}

#[uniffi::export]
impl FalImageGenProvider {
    /// Synchronous variant of [`generate_image`](Self::generate_image).
    pub fn generate_image_blocking(
        self: Arc<Self>,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.generate_image(prompt, width, height).await })
    }
}

// ---------------------------------------------------------------------------
// Polymorphic capability-base trait impls
// ---------------------------------------------------------------------------
//
// Each engine implements both [`crate::concrete::bases::BaseProvider`] and
// [`crate::concrete::bases::ImageGenProvider`] so foreign
// (Kotlin/Swift/Go) consumers can hold a polymorphic
// `ImageGenProvider` reference and Rust-side code can collect engines
// into capability-erased `Arc<dyn BaseProvider>` containers. The
// inherent `generate_image` methods on each engine (which use
// `self: Arc<Self>`) continue to take precedence at the call site
// `engine.generate_image(...)`; the trait methods (which use `&self`)
// are reachable via UFCS / `dyn ImageGenProvider` dispatch.

// DiffusionProvider --------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for DiffusionProvider {
    fn provider_id(&self) -> String {
        "diffusion-rs".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ImageGen
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::ImageGenProvider for DiffusionProvider {
    fn provider_id(&self) -> String {
        "diffusion-rs".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ImageGen
    }

    async fn generate_image(
        &self,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError> {
        use blazen_llm::providers::capabilities::ImageGenProvider as _;
        let req = build_image_request(prompt, width, height);
        let result = self.inner.generate_image(req).await?;
        Ok(project_image_result(result))
    }
}

// FalImageGenProvider ------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FalImageGenProvider {
    fn provider_id(&self) -> String {
        "fal-image".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ImageGen
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::ImageGenProvider for FalImageGenProvider {
    fn provider_id(&self) -> String {
        "fal-image".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::ImageGen
    }

    async fn generate_image(
        &self,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError> {
        use blazen_llm::providers::capabilities::ImageGenProvider as _;
        let req = build_image_request(prompt, width, height);
        let result = self.inner.generate_image(req).await?;
        Ok(project_image_result(result))
    }
}
