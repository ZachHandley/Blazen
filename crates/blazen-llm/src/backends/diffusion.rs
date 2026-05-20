//! Bridge between [`blazen_image_diffusion::DiffusionProvider`] and the
//! [`ImageGeneration`](crate::compute::ImageGeneration) trait.
//!
//! `diffusion-rs` (the underlying `stable-diffusion.cpp` wrapper) is a
//! synchronous, single-call engine: one [`ImageGeneration::generate_image`]
//! call drives one `gen_img` invocation and returns one
//! [`ImageResult`](crate::compute::ImageResult). There is no per-step
//! callback exposed through Blazen because upstream's `Progress` type has
//! private fields (see the crate-level docs on `blazen-image-diffusion`).
//!
//! As with the other local backends, the [`ComputeProvider`] job-queue
//! verbs (`submit` / `status` / `result` / `cancel`) return
//! [`BlazenError::Unsupported`] -- callers should invoke
//! [`ImageGeneration::generate_image`] directly.
//!
//! Without the `engine` feature on `blazen-image-diffusion` the underlying
//! provider's inherent engine methods are not compiled in, so every call
//! through this bridge surfaces as a [`BlazenError::Provider`] wrapping
//! `EngineNotAvailable`.

use async_trait::async_trait;
#[cfg(feature = "diffusion-engine")]
use base64::Engine as _;
#[cfg(feature = "diffusion-engine")]
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use blazen_image_diffusion::{DiffusionError, DiffusionProvider};

use crate::compute::{
    ComputeProvider, ComputeRequest, ComputeResult, ImageGeneration, ImageRequest, ImageResult,
    JobHandle, JobStatus, UpscaleRequest,
};
use crate::error::BlazenError;
#[cfg(feature = "diffusion-engine")]
use crate::media::{GeneratedImage, MediaOutput, MediaType};
use crate::traits::LocalModel;
#[cfg(feature = "diffusion-engine")]
use crate::types::RequestTiming;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a [`DiffusionError`] into a [`BlazenError`] preserving the
/// distinction between configuration errors and runtime failures.
fn map_err(e: DiffusionError) -> BlazenError {
    match e {
        DiffusionError::InvalidOptions(msg) => {
            BlazenError::provider("diffusion-rs", format!("invalid options: {msg}"))
        }
        DiffusionError::EngineNotAvailable => BlazenError::provider(
            "diffusion-rs",
            "engine feature not enabled -- rebuild blazen-image-diffusion with `engine`",
        ),
        DiffusionError::ModelLoad(msg) | DiffusionError::Generation(msg) => {
            BlazenError::provider("diffusion-rs", msg)
        }
    }
}

/// Convert a freshly-generated raw image into Blazen's typed
/// [`GeneratedImage`]. Only invoked from the engine code path; gated to
/// the upstream `engine` feature because [`blazen_image_diffusion::GeneratedImage`]
/// only exists with `engine` on.
#[cfg(feature = "diffusion-engine")]
fn to_generated_image(raw: blazen_image_diffusion::GeneratedImage) -> GeneratedImage {
    let blazen_image_diffusion::GeneratedImage {
        bytes,
        width,
        height,
    } = raw;
    let media_type = MediaType::detect(&bytes).unwrap_or(MediaType::Png);
    let size_bytes = u64::try_from(bytes.len()).ok();
    let base64 = BASE64_STANDARD.encode(&bytes);
    let mut media = MediaOutput::from_base64(base64, media_type);
    media.file_size = size_bytes;
    GeneratedImage {
        media,
        width: Some(width),
        height: Some(height),
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for DiffusionProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        "diffusion-rs"
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "diffusion-rs runs locally and does not use the ComputeRequest job API; \
             call `ImageGeneration::generate_image` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "diffusion-rs does not expose a job queue -- generation is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "diffusion-rs does not expose a job queue -- generation is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "diffusion-rs generation is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// ImageGeneration
// ---------------------------------------------------------------------------

#[async_trait]
impl ImageGeneration for DiffusionProvider {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        if request.prompt.trim().is_empty() {
            return Err(BlazenError::provider(
                "diffusion-rs",
                "prompt must not be empty",
            ));
        }
        if let Some(w) = request.width
            && w == 0
        {
            return Err(BlazenError::provider(
                "diffusion-rs",
                "request width must be greater than zero",
            ));
        }
        if let Some(h) = request.height
            && h == 0
        {
            return Err(BlazenError::provider(
                "diffusion-rs",
                "request height must be greater than zero",
            ));
        }
        if let Some(n) = request.num_images
            && n > 1
        {
            return Err(BlazenError::unsupported(
                "diffusion-rs bridge currently runs one image per call; \
                 invoke generate_image multiple times for batches",
            ));
        }

        #[cfg(feature = "diffusion-engine")]
        {
            let start = std::time::Instant::now();
            let raw = DiffusionProvider::generate_image_inherent(
                self,
                request.prompt,
                request.negative_prompt,
                request.width,
                request.height,
            )
            .await
            .map_err(map_err)?;
            #[allow(clippy::cast_possible_truncation)]
            let total_ms = start.elapsed().as_millis() as u64;
            let image = to_generated_image(raw);
            Ok(ImageResult {
                images: vec![image],
                timing: RequestTiming {
                    queue_ms: None,
                    execution_ms: Some(total_ms),
                    total_ms: Some(total_ms),
                },
                cost: None,
                usage: None,
                image_count: 1,
                metadata: serde_json::Value::Null,
            })
        }
        #[cfg(not(feature = "diffusion-engine"))]
        {
            let _ = request;
            Err(map_err(DiffusionError::EngineNotAvailable))
        }
    }

    async fn upscale_image(&self, _request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        Err(BlazenError::unsupported(
            "diffusion-rs does not support image upscaling through this bridge -- \
             configure a dedicated upscaler (ESRGAN) at the provider level or \
             use a remote upscale provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: gives `ModelManager` explicit load/unload control
/// over the underlying diffusion-rs pipeline. The implementation forwards
/// to the inherent methods on [`DiffusionProvider`] and wraps
/// [`DiffusionError`] into [`BlazenError::Provider`] via [`map_err`].
#[async_trait]
impl LocalModel for DiffusionProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        DiffusionProvider::load(self).await.map_err(map_err)
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        DiffusionProvider::unload(self).await.map_err(map_err)
    }

    async fn is_loaded(&self) -> bool {
        DiffusionProvider::is_loaded(self).await
    }

    fn device(&self) -> crate::device::Device {
        crate::device::Device::parse(self.device_str()).unwrap_or(crate::device::Device::Cpu)
    }

    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "diffusion-rs does not support LoRA adapters through this bridge -- \
             attach LoRAs at construction time via diffusion-rs modifiers instead",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_image_diffusion::DiffusionOptions;

    fn provider() -> DiffusionProvider {
        DiffusionProvider::from_options(DiffusionOptions::default())
            .expect("default options should validate")
    }

    #[tokio::test]
    async fn provider_id_is_diffusion_rs() {
        assert_eq!(ComputeProvider::provider_id(&provider()), "diffusion-rs");
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let request = ComputeRequest {
            model: "diffusion-rs".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = provider().submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn cancel_is_unsupported() {
        let handle = JobHandle {
            id: "fake".into(),
            provider: "diffusion-rs".into(),
            model: "diffusion-rs".into(),
            submitted_at: chrono::Utc::now(),
        };
        let err = provider().cancel(&handle).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn upscale_is_unsupported() {
        let req = UpscaleRequest::new("file:///nope.png", 2.0);
        let err = ImageGeneration::upscale_image(&provider(), req)
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn invalid_request_zero_width_rejected() {
        let req = ImageRequest::new("a cat").with_size(0, 256);
        let err = ImageGeneration::generate_image(&provider(), req)
            .await
            .unwrap_err();
        assert!(
            matches!(err, BlazenError::Provider { .. }),
            "expected Provider error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn invalid_request_empty_prompt_rejected() {
        let req = ImageRequest::new("   ");
        let err = ImageGeneration::generate_image(&provider(), req)
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Provider { .. }));
    }

    #[tokio::test]
    async fn batch_requests_are_unsupported() {
        let req = ImageRequest::new("a cat").with_count(3);
        let err = ImageGeneration::generate_image(&provider(), req)
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }
}
