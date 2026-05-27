//! Concrete per-engine image-generation provider classes.
//!
//! Each `<Engine>Provider` newtype wraps an upstream image-gen backend
//! (local `stable-diffusion.cpp` via `diffusion-rs`, fal.ai cloud
//! image-gen) and implements the polymorphic
//! [`crate::providers::root::BaseProvider`] root plus the
//! [`crate::providers::capabilities::ImageGenProvider`] capability
//! sub-trait.
//!
//! Two engines are wired today:
//!
//! - [`DiffusionProvider`] — local `stable-diffusion.cpp` text-to-image
//!   via the [`blazen_image_diffusion::DiffusionProvider`] runtime,
//!   feature-gated on `diffusion`. Without the `diffusion-engine`
//!   forwarding feature the underlying engine returns
//!   [`crate::error::BlazenError::Provider`] with the
//!   `EngineNotAvailable` distinction preserved (see
//!   [`crate::backends::diffusion`]).
//! - [`FalImageGenProvider`] — fal.ai HTTP cloud image-gen via the
//!   shared [`crate::providers::fal::FalProvider`] queue executor.
//!   Always available — fal is not feature-gated.
//!
//! The existing [`crate::compute::traits::ImageGeneration`] impls on
//! the inner types stay in place; this module strictly wraps them
//! behind the new capability-trait surface so consumers can store the
//! provider behind `Arc<dyn ImageGenProvider>` or
//! `Arc<dyn BaseProvider>` for capability-erased collections.
//!
//! `OpenAI` image generation (GPT-Image / DALL-E) is **not** wrapped here:
//! [`crate::providers::openai::OpenAiProvider`] does not implement
//! [`crate::compute::traits::ImageGeneration`] in the current codebase,
//! so there is nothing to delegate to. When that bridge lands, add a
//! corresponding `OpenAiImageGenProvider` newtype here.

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use async_trait::async_trait;

use crate::compute::requests::{ImageRequest, UpscaleRequest};
use crate::compute::results::ImageResult;
use crate::error::BlazenError;
use crate::providers::capabilities::ImageGenProvider;
use crate::providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// DiffusionProvider — local stable-diffusion.cpp text-to-image
// ---------------------------------------------------------------------------

/// Concrete [`ImageGenProvider`] backed by the local
/// [`blazen_image_diffusion::DiffusionProvider`] runtime.
///
/// Wraps the `diffusion-rs` (stable-diffusion.cpp) pipeline. The
/// existing [`crate::compute::traits::ImageGeneration`] bridge in
/// [`crate::backends::diffusion`] handles request validation,
/// `spawn_blocking` dispatch, and the `EngineNotAvailable` /
/// `InvalidOptions` / `ModelLoad` / `Generation` error mapping; this
/// newtype is a thin delegation layer that adds the
/// [`BaseProvider::metadata`] surface.
///
/// Without the `diffusion-engine` forwarding feature on
/// `blazen-image-diffusion`, [`ImageGenProvider::generate_image`]
/// surfaces a [`BlazenError::Provider`] wrapping
/// `EngineNotAvailable`. Upscaling is not supported through the
/// diffusion-rs bridge — pair this provider with a dedicated upscaler.
#[cfg(feature = "diffusion")]
pub struct DiffusionProvider {
    inner: Arc<blazen_image_diffusion::DiffusionProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "diffusion")]
impl DiffusionProvider {
    /// Construct a `DiffusionProvider` from a typed
    /// [`blazen_image_diffusion::DiffusionOptions`].
    ///
    /// Mirrors [`blazen_image_diffusion::DiffusionProvider::from_options`]
    /// — the upstream constructor validates `device` / `model_id` /
    /// `width` / `height` / `num_inference_steps` / `guidance_scale`
    /// and returns [`blazen_image_diffusion::DiffusionError::InvalidOptions`]
    /// on any zero / empty value. That error is mapped onto
    /// [`BlazenError::Provider`] with provider id `"diffusion-rs"`
    /// (the same id the [`crate::backends::diffusion`] bridge uses) so
    /// callers see a consistent provider label across the static
    /// `ImageGeneration` surface and the new capability surface.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] with provider `"diffusion-rs"`
    /// when the underlying options validation fails.
    pub fn new(options: blazen_image_diffusion::DiffusionOptions) -> Result<Self, BlazenError> {
        let inner = blazen_image_diffusion::DiffusionProvider::from_options(options)
            .map_err(|e| BlazenError::provider("diffusion-rs", e.to_string()))?;
        let metadata = ProviderMetadata::new("diffusion-rs", CapabilityKind::ImageGen);
        Ok(Self {
            inner: Arc::new(inner),
            metadata,
        })
    }

    /// Wrap an already-constructed inner provider. Useful when the
    /// caller wants to share one engine load (the `OnceCell` inside
    /// the inner) across multiple consumer surfaces — e.g. both an
    /// `Arc<dyn BaseProvider>` and a direct typed handle.
    #[must_use]
    pub fn from_inner(inner: Arc<blazen_image_diffusion::DiffusionProvider>) -> Self {
        Self {
            inner,
            metadata: ProviderMetadata::new("diffusion-rs", CapabilityKind::ImageGen),
        }
    }

    /// Borrow the inner backend — escape hatch for callers that need
    /// to drive the engine load lifecycle directly (e.g.
    /// [`crate::traits::LocalModel::load`] / `unload`).
    #[must_use]
    pub fn backend(&self) -> &Arc<blazen_image_diffusion::DiffusionProvider> {
        &self.inner
    }
}

#[cfg(feature = "diffusion")]
impl std::fmt::Debug for DiffusionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffusionProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "diffusion")]
impl BaseProvider for DiffusionProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "diffusion")]
#[async_trait]
impl ImageGenProvider for DiffusionProvider {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        // Delegate to the existing `ImageGeneration` impl on the inner
        // type. That impl already handles prompt / dimension /
        // num_images validation and the `diffusion-engine` feature
        // gate, so wrapping it here is a one-line dispatch.
        <blazen_image_diffusion::DiffusionProvider as crate::compute::traits::ImageGeneration>::generate_image(
            &*self.inner,
            request,
        )
        .await
    }

    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        // The diffusion-rs bridge does not support upscaling; forward
        // to the inner impl so the unsupported-error string stays the
        // single source of truth.
        <blazen_image_diffusion::DiffusionProvider as crate::compute::traits::ImageGeneration>::upscale_image(
            &*self.inner,
            request,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// FalImageGenProvider — fal.ai cloud image-gen
// ---------------------------------------------------------------------------

/// Concrete [`ImageGenProvider`] backed by fal.ai's cloud image-gen
/// HTTP surface.
///
/// Wraps the shared [`crate::providers::fal::FalProvider`] queue
/// executor — the same Rust type that powers fal's LLM / embedding /
/// audio / video bindings — and delegates to its existing
/// [`crate::compute::traits::ImageGeneration`] impl, which handles
/// queue submission, polling, and `image_size` / `num_images` /
/// `negative_prompt` parameter shaping.
///
/// Image model selection happens **per request** through
/// [`ImageRequest::model`]; the optional `default_model` passed at
/// construction is stamped onto every outgoing request that doesn't
/// override it. Falls back to the fal bridge's built-in
/// `DEFAULT_IMAGE_MODEL` when neither is set.
pub struct FalImageGenProvider {
    inner: Arc<crate::providers::fal::FalProvider>,
    default_model: Option<String>,
    metadata: ProviderMetadata,
}

impl FalImageGenProvider {
    /// Construct a new fal image-gen provider.
    ///
    /// - `api_key`: fal.ai API key.
    /// - `default_model`: optional default model id (e.g.
    ///   `"fal-ai/flux/schnell"`). When set, every
    ///   [`ImageGenProvider::generate_image`] call that doesn't pass
    ///   [`ImageRequest::model`] uses this model. When `None`, the
    ///   inner fal bridge's hard-coded `DEFAULT_IMAGE_MODEL` is used.
    /// - `base_url`: optional override for the fal queue base URL.
    ///   Forwarded to [`crate::providers::fal::FalProvider::with_base_url`].
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        default_model: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        let mut inner = crate::providers::fal::FalProvider::new(api_key);
        if let Some(url) = base_url {
            inner = inner.with_base_url(url);
        }
        let metadata = ProviderMetadata::new("fal-image", CapabilityKind::ImageGen);
        Self {
            inner: Arc::new(inner),
            default_model,
            metadata,
        }
    }

    /// Wrap an already-constructed [`crate::providers::fal::FalProvider`]
    /// — useful when sharing one HTTP client / retry config across
    /// fal's LLM and image-gen surfaces.
    #[must_use]
    pub fn from_inner(
        inner: Arc<crate::providers::fal::FalProvider>,
        default_model: Option<String>,
    ) -> Self {
        Self {
            inner,
            default_model,
            metadata: ProviderMetadata::new("fal-image", CapabilityKind::ImageGen),
        }
    }

    /// Borrow the inner fal provider — escape hatch for callers that
    /// want to drive the queue executor directly (custom endpoints,
    /// retry tuning, HTTP client injection).
    #[must_use]
    pub fn backend(&self) -> &Arc<crate::providers::fal::FalProvider> {
        &self.inner
    }

    /// The configured default image model, if any.
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }
}

impl std::fmt::Debug for FalImageGenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalImageGenProvider")
            .field("metadata", &self.metadata)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

impl BaseProvider for FalImageGenProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[async_trait]
impl ImageGenProvider for FalImageGenProvider {
    async fn generate_image(&self, mut request: ImageRequest) -> Result<ImageResult, BlazenError> {
        // Stamp the configured default model when the caller did not
        // pass one through `ImageRequest::model`. The inner fal bridge
        // falls back to its own `DEFAULT_IMAGE_MODEL` when both are
        // absent, so callers always have a working default.
        if request.model.is_none()
            && let Some(ref m) = self.default_model
        {
            request.model = Some(m.clone());
        }
        <crate::providers::fal::FalProvider as crate::compute::traits::ImageGeneration>::generate_image(
            &*self.inner,
            request,
        )
        .await
    }

    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        <crate::providers::fal::FalProvider as crate::compute::traits::ImageGeneration>::upscale_image(
            &*self.inner,
            request,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_trait_is_object_safe_for_image_providers() {
        // Compile-time proof: each Box<dyn ImageGenProvider> is only
        // well-formed when the concrete type satisfies the capability
        // trait's object-safety + Send + Sync bounds.
        fn _erased(_: Box<dyn ImageGenProvider>) {}
    }

    #[cfg(feature = "diffusion")]
    #[test]
    fn diffusion_provider_metadata_is_stamped() {
        let opts = blazen_image_diffusion::DiffusionOptions::default();
        let provider = DiffusionProvider::new(opts).expect("default options should validate");
        assert_eq!(provider.provider_id(), "diffusion-rs");
        assert_eq!(provider.metadata().capability, CapabilityKind::ImageGen);
    }

    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[test]
    fn fal_image_provider_metadata_is_stamped() {
        let provider =
            FalImageGenProvider::new("fal-test-key", Some("fal-ai/flux/schnell".to_owned()), None);
        assert_eq!(provider.provider_id(), "fal-image");
        assert_eq!(provider.metadata().capability, CapabilityKind::ImageGen);
        assert_eq!(provider.default_model(), Some("fal-ai/flux/schnell"));
    }
}
