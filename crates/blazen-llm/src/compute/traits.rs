//! Compute platform traits for media generation providers.
//!
//! Implement [`ComputeProvider`] as the base, then add media-specific
//! traits ([`ImageGeneration`], [`VideoGeneration`], etc.) for the
//! capabilities your provider supports.
//!
//! ```rust,ignore
//! use blazen_llm::compute::*;
//! use blazen_llm::BlazenError;
//!
//! struct MyImageProvider { /* ... */ }
//!
//! #[async_trait::async_trait]
//! impl ComputeProvider for MyImageProvider {
//!     fn provider_id(&self) -> &str { "my-provider" }
//!     async fn submit(&self, request: ComputeRequest) -> Result<JobHandle, BlazenError> { todo!() }
//!     async fn status(&self, job: &JobHandle) -> Result<JobStatus, BlazenError> { todo!() }
//!     async fn result(&self, job: JobHandle) -> Result<ComputeResult, BlazenError> { todo!() }
//!     async fn cancel(&self, job: &JobHandle) -> Result<(), BlazenError> { todo!() }
//! }
//!
//! #[async_trait::async_trait]
//! impl ImageGeneration for MyImageProvider {
//!     async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
//!         // Your implementation
//!         todo!()
//!     }
//!     async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
//!         // Your implementation
//!         todo!()
//!     }
//! }
//! ```

use async_trait::async_trait;

use super::job::{ComputeRequest, ComputeResult, JobHandle, JobStatus};
use super::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use super::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use crate::error::BlazenError;

// ---------------------------------------------------------------------------
// Core compute trait
// ---------------------------------------------------------------------------

/// A compute platform that supports async job submission and polling.
///
/// This is the base trait for compute providers like fal.ai, Replicate,
/// `RunPod`, etc. It models the common pattern of: submit a job, poll for
/// status, retrieve the result.
#[async_trait]
pub trait ComputeProvider: Send + Sync {
    /// Provider identifier (e.g., "fal", "replicate", "runpod").
    fn provider_id(&self) -> &str;

    /// Submit a compute job and get a handle to track it.
    async fn submit(&self, request: ComputeRequest) -> Result<JobHandle, BlazenError>;

    /// Poll the current status of a submitted job.
    async fn status(&self, job: &JobHandle) -> Result<JobStatus, BlazenError>;

    /// Wait for a job to complete and return the result.
    ///
    /// The default implementation delegates to the provider. Providers may
    /// override this with long-polling, `WebSockets`, or SSE.
    async fn result(&self, job: JobHandle) -> Result<ComputeResult, BlazenError>;

    /// Cancel a running or queued job.
    async fn cancel(&self, job: &JobHandle) -> Result<(), BlazenError>;

    /// Submit a job and wait for the result (convenience method).
    ///
    /// This is equivalent to calling [`ComputeProvider::submit`] followed
    /// by [`ComputeProvider::result`].
    async fn run(&self, request: ComputeRequest) -> Result<ComputeResult, BlazenError> {
        let job = self.submit(request).await?;
        self.result(job).await
    }
}

// ---------------------------------------------------------------------------
// Media-specific traits
// ---------------------------------------------------------------------------

/// Image generation and upscaling capability.
///
/// Providers that support image generation (fal.ai, Replicate, etc.)
/// implement this trait to provide a typed interface for image operations.
#[async_trait]
pub trait ImageGeneration: ComputeProvider {
    /// Generate images from a text prompt.
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError>;

    /// Upscale an existing image.
    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError>;
}

/// Video generation capability.
///
/// Providers that support video synthesis implement this trait.
#[async_trait]
pub trait VideoGeneration: ComputeProvider {
    /// Generate a video from a text prompt.
    async fn text_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError>;

    /// Generate a video from a source image and prompt.
    async fn image_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError>;
}

/// Audio generation capability (TTS, music, sound effects).
///
/// Providers that support audio synthesis implement this trait.
#[async_trait]
pub trait AudioGeneration: ComputeProvider {
    /// Synthesize speech from text.
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError>;

    /// Generate music from a prompt.
    ///
    /// Returns [`BlazenError::Unsupported`] by default.
    async fn generate_music(&self, _request: MusicRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "music generation not supported by this provider",
        ))
    }

    /// Generate sound effects from a prompt.
    ///
    /// Returns [`BlazenError::Unsupported`] by default.
    async fn generate_sfx(&self, _request: MusicRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "sound effect generation not supported by this provider",
        ))
    }
}

/// Audio transcription capability (speech-to-text).
///
/// Providers that support transcription implement this trait.
#[async_trait]
pub trait Transcription: ComputeProvider {
    /// Transcribe audio to text with optional diarization.
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError>;
}

/// 3D model generation capability.
///
/// Providers that support 3D generation implement this trait.
#[async_trait]
pub trait ThreeDGeneration: ComputeProvider {
    /// Generate a 3D model from a text prompt or source image.
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError>;
}

/// A compute provider that supports background removal on existing images.
#[async_trait]
pub trait BackgroundRemoval: ComputeProvider {
    /// Remove the background from an image and return the result.
    async fn remove_background(
        &self,
        request: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// Voice cloning
// ---------------------------------------------------------------------------

/// Voice cloning capability.
///
/// Distinct from `AudioGeneration::text_to_speech` because cloning creates
/// a persisted voice that can be reused across later TTS calls. No
/// Blazen-shipped provider implements this trait -- it exists so users
/// building their own providers (via `CustomProvider`) can wire up
/// services like `ElevenLabs` or `zvoice` into Blazen's capability system.
#[async_trait]
pub trait VoiceCloning: ComputeProvider {
    /// Clone a voice from one or more reference audio clips and return
    /// a persistent handle that can be passed as `SpeechRequest.voice`
    /// on subsequent TTS calls.
    async fn clone_voice(&self, request: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError>;

    /// List all voices known to this provider (presets + cloned).
    ///
    /// Returns `BlazenError::Unsupported` by default.
    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        Err(BlazenError::unsupported(
            "list_voices not supported by this provider",
        ))
    }

    /// Delete a previously cloned voice.
    ///
    /// Returns `BlazenError::Unsupported` by default.
    async fn delete_voice(&self, voice: &VoiceHandle) -> Result<(), BlazenError> {
        let _ = voice;
        Err(BlazenError::unsupported(
            "delete_voice not supported by this provider",
        ))
    }
}

// Backwards-compatible type alias for the old trait name.
/// Alias for [`ImageGeneration`] -- the old name before the multi-modal
/// expansion.
pub trait ImageModel: ImageGeneration {}
impl<T: ImageGeneration> ImageModel for T {}
