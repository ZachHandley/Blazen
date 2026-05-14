//! Universal `CustomProvider` -- the extension point for any provider.
//!
//! [`CustomProvider`] is a **foreign-implementable trait** with sixteen typed
//! async methods (completion + streaming + embeddings + thirteen compute/media
//! capabilities). Every method has a default implementation that returns
//! [`BlazenError::Unsupported`]; users override only the methods their provider
//! actually supports.
//!
//! [`CustomProviderHandle`] is the concrete wrapper bindings hand back to
//! callers. It owns an `Arc<dyn CustomProvider>` plus per-instance defaults
//! (system prompt, tools, response format, role-specific compute defaults,
//! retry config), applies those defaults before dispatching to the inner
//! provider, and implements every relevant Blazen trait (`CompletionModel`,
//! `ComputeProvider`, `AudioGeneration`, `ImageGeneration`, `VideoGeneration`,
//! `Transcription`, `ThreeDGeneration`, `VoiceCloning`, `BackgroundRemoval`,
//! and `CustomProvider` itself for clean composition).
//!
//! ## Convenience constructors
//!
//! - [`ollama`] / [`lm_studio`] -- one-liner for the two most common
//!   local-model servers. Build an `OpenAi`-protocol `CustomProviderHandle`
//!   with the right base URL.
//! - [`openai_compat`] -- arbitrary `OpenAI`-compatible server.
//!
//! Adding a new capability method to a trait like [`AudioGeneration`] in
//! `crate::compute::traits` automatically works for custom providers -- just
//! add a method to [`CustomProvider`] with a default `Unsupported` impl and
//! forward through [`CustomProviderHandle`].

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
use super::openai_compat::AuthMethod;
use super::openai_compat::OpenAiCompatConfig;
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
use super::openai_compat::OpenAiCompatProvider;
use crate::compute::job::{ComputeRequest, ComputeResult, JobHandle, JobStatus};
use crate::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use crate::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use crate::compute::traits::{
    AudioGeneration, BackgroundRemoval, ComputeProvider, ImageGeneration, ThreeDGeneration,
    Transcription, VideoGeneration, VoiceCloning,
};
use crate::error::BlazenError;
use crate::http::HttpClient;
use crate::providers::base::BaseProvider;
use crate::providers::defaults::{
    AudioMusicProviderDefaults, AudioSpeechProviderDefaults, BackgroundRemovalProviderDefaults,
    BaseProviderDefaults, BeforeCompletionRequestHook, BeforeRequestHook,
    CompletionProviderDefaults, ImageGenerationProviderDefaults, ImageUpscaleProviderDefaults,
    ThreeDProviderDefaults, TranscriptionProviderDefaults, VideoProviderDefaults,
    VoiceCloningProviderDefaults,
};
use crate::retry::RetryConfig;
use crate::traits::CompletionModel;
use crate::types::{
    CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk, ToolDefinition,
};

// ---------------------------------------------------------------------------
// ApiProtocol
// ---------------------------------------------------------------------------

/// Selects how a [`CustomProviderHandle`] talks to its backend for completion calls.
#[derive(Debug, Clone)]
pub enum ApiProtocol {
    /// `OpenAI` Chat Completions wire format. Framework handles request body,
    /// SSE parsing, tool-call serialization, and retries. The wrapped config
    /// supplies the base URL, model, optional API key, and headers.
    OpenAi(OpenAiCompatConfig),

    /// User-defined protocol. The handle dispatches every method to a typed
    /// `CustomProvider` implementation.
    Custom,
    // Future: Anthropic(AnthropicConfig)
}

// ---------------------------------------------------------------------------
// CustomProvider trait
// ---------------------------------------------------------------------------

/// User-extensible provider trait. Foreign bindings expose this as a class,
/// protocol, or interface that users implement; Rust users can implement it
/// directly on their own type.
///
/// All sixteen methods have default implementations that return
/// [`BlazenError::Unsupported`]. Users override only the ones their provider
/// actually supports.
///
/// Wrap an `Arc<dyn CustomProvider>` in [`CustomProviderHandle`] to apply
/// per-instance defaults (system prompt, tools, role-specific compute
/// defaults, retry config) and to plug into Blazen's trait surface.
#[async_trait]
pub trait CustomProvider: Send + Sync + 'static {
    /// Stable provider identifier for logs and metrics. Required.
    fn provider_id(&self) -> &str;

    /// Model identifier. Defaults to [`Self::provider_id`]; override when the
    /// provider serves a single distinct model name.
    fn model_id(&self) -> &str {
        self.provider_id()
    }

    /// Optional provider-level default retry configuration.
    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        None
    }

    /// Optional escape-hatch HTTP client the provider exposes to callers.
    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        None
    }

    // ---- Completion ----

    /// Perform a non-streaming chat completion.
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::complete not implemented",
        ))
    }

    /// Perform a streaming chat completion.
    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::unsupported(
            "CustomProvider::stream not implemented",
        ))
    }

    // ---- Embedding ----

    /// Embed one or more texts. Default returns `Unsupported`.
    async fn embed(&self, _texts: Vec<String>) -> Result<EmbeddingResponse, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::embed not implemented",
        ))
    }

    // ---- Compute / media ----

    /// Synthesize speech from text. Default returns `Unsupported`.
    async fn text_to_speech(&self, _req: SpeechRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::text_to_speech not implemented",
        ))
    }

    /// Generate music from a prompt. Default returns `Unsupported`.
    async fn generate_music(&self, _req: MusicRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::generate_music not implemented",
        ))
    }

    /// Generate sound effects from a prompt. Default returns `Unsupported`.
    async fn generate_sfx(&self, _req: MusicRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::generate_sfx not implemented",
        ))
    }

    /// Clone a voice from reference audio. Default returns `Unsupported`.
    async fn clone_voice(&self, _req: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::clone_voice not implemented",
        ))
    }

    /// List voices known to the provider. Default returns `Unsupported`.
    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::list_voices not implemented",
        ))
    }

    /// Delete a previously-cloned voice. Default returns `Unsupported`.
    async fn delete_voice(&self, _voice: VoiceHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::delete_voice not implemented",
        ))
    }

    /// Generate images from a prompt. Default returns `Unsupported`.
    async fn generate_image(&self, _req: ImageRequest) -> Result<ImageResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::generate_image not implemented",
        ))
    }

    /// Upscale an existing image. Default returns `Unsupported`.
    async fn upscale_image(&self, _req: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::upscale_image not implemented",
        ))
    }

    /// Generate a video from a text prompt. Default returns `Unsupported`.
    async fn text_to_video(&self, _req: VideoRequest) -> Result<VideoResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::text_to_video not implemented",
        ))
    }

    /// Generate a video from a source image + prompt. Default returns `Unsupported`.
    async fn image_to_video(&self, _req: VideoRequest) -> Result<VideoResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::image_to_video not implemented",
        ))
    }

    /// Transcribe audio to text. Default returns `Unsupported`.
    async fn transcribe(
        &self,
        _req: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::transcribe not implemented",
        ))
    }

    /// Generate a 3D model. Default returns `Unsupported`.
    async fn generate_3d(&self, _req: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::generate_3d not implemented",
        ))
    }

    /// Remove the background from an image. Default returns `Unsupported`.
    async fn remove_background(
        &self,
        _req: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider::remove_background not implemented",
        ))
    }
}

// ---------------------------------------------------------------------------
// CustomProviderAsCompletionModel — adapter (private)
// ---------------------------------------------------------------------------

/// Internal adapter that turns an `Arc<dyn CustomProvider>` into an
/// `Arc<dyn CompletionModel>`, so [`BaseProvider::new`] can wrap it for
/// completion-defaults application.
pub(crate) struct CustomProviderAsCompletionModel(pub Arc<dyn CustomProvider>);

impl std::fmt::Debug for CustomProviderAsCompletionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomProviderAsCompletionModel")
            .field("provider_id", &self.0.provider_id())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl CompletionModel for CustomProviderAsCompletionModel {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.0.retry_config()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.0.http_client()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        self.0.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.0.stream(request).await
    }
}

// ---------------------------------------------------------------------------
// CustomProviderHandle — concrete wrapper with defaults
// ---------------------------------------------------------------------------

/// Concrete wrapper around an `Arc<dyn CustomProvider>` that applies
/// per-instance defaults (completion + nine role-specific compute defaults)
/// before delegating each method to the inner provider.
///
/// Bindings hold this. It implements every relevant Blazen trait
/// (`CompletionModel`, `ComputeProvider`, all media-capability traits, and
/// `CustomProvider` itself for clean nesting).
pub struct CustomProviderHandle {
    inner: Arc<dyn CustomProvider>,
    base: BaseProvider,
    retry_config: Option<Arc<RetryConfig>>,
    audio_speech_defaults: Option<AudioSpeechProviderDefaults>,
    audio_music_defaults: Option<AudioMusicProviderDefaults>,
    voice_cloning_defaults: Option<VoiceCloningProviderDefaults>,
    image_generation_defaults: Option<ImageGenerationProviderDefaults>,
    image_upscale_defaults: Option<ImageUpscaleProviderDefaults>,
    video_defaults: Option<VideoProviderDefaults>,
    transcription_defaults: Option<TranscriptionProviderDefaults>,
    three_d_defaults: Option<ThreeDProviderDefaults>,
    background_removal_defaults: Option<BackgroundRemovalProviderDefaults>,
    protocol: ApiProtocol,
}

impl std::fmt::Debug for CustomProviderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomProviderHandle")
            .field("provider_id", &self.inner.provider_id())
            .field("protocol", &self.protocol)
            .field("base", &self.base)
            .finish_non_exhaustive()
    }
}

impl Clone for CustomProviderHandle {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            base: self.base.clone(),
            retry_config: self.retry_config.clone(),
            audio_speech_defaults: self.audio_speech_defaults.clone(),
            audio_music_defaults: self.audio_music_defaults.clone(),
            voice_cloning_defaults: self.voice_cloning_defaults.clone(),
            image_generation_defaults: self.image_generation_defaults.clone(),
            image_upscale_defaults: self.image_upscale_defaults.clone(),
            video_defaults: self.video_defaults.clone(),
            transcription_defaults: self.transcription_defaults.clone(),
            three_d_defaults: self.three_d_defaults.clone(),
            background_removal_defaults: self.background_removal_defaults.clone(),
            protocol: self.protocol.clone(),
        }
    }
}

impl CustomProviderHandle {
    /// Wrap any `Arc<dyn CustomProvider>` with default-application semantics.
    /// Pair with the `with_*` builders to configure completion + role-specific
    /// compute defaults.
    #[must_use]
    pub fn new(inner: Arc<dyn CustomProvider>) -> Self {
        let completion_model: Arc<dyn CompletionModel> =
            Arc::new(CustomProviderAsCompletionModel(Arc::clone(&inner)));
        Self {
            inner,
            base: BaseProvider::new(completion_model),
            retry_config: None,
            audio_speech_defaults: None,
            audio_music_defaults: None,
            voice_cloning_defaults: None,
            image_generation_defaults: None,
            image_upscale_defaults: None,
            video_defaults: None,
            transcription_defaults: None,
            three_d_defaults: None,
            background_removal_defaults: None,
            protocol: ApiProtocol::Custom,
        }
    }

    /// Set the provider-level default retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Set the system prompt prepended to every completion request (when no
    /// system message is already present).
    #[must_use]
    pub fn with_system_prompt(mut self, s: impl Into<String>) -> Self {
        self.base = self.base.with_system_prompt(s);
        self
    }

    /// Replace the default tool list. Per-request tools are appended on top of
    /// this default; collisions favor the per-request tool.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.base = self.base.with_tools(tools);
        self
    }

    /// Set the default `response_format`. Per-request `response_format`
    /// overrides this default.
    #[must_use]
    pub fn with_response_format(mut self, fmt: serde_json::Value) -> Self {
        self.base = self.base.with_response_format(fmt);
        self
    }

    /// Set the universal before-request hook. Fires for EVERY provider request
    /// (completion, audio, image, video, etc.) as a JSON-level mutation.
    #[must_use]
    pub fn with_before_request(mut self, hook: BeforeRequestHook) -> Self {
        self.base = self.base.with_before_request(hook);
        self
    }

    /// Set the typed before-completion hook. Fires AFTER the universal
    /// `before_request` hook, with a typed view of the `CompletionRequest`.
    #[must_use]
    pub fn with_before_completion(mut self, hook: BeforeCompletionRequestHook) -> Self {
        self.base = self.base.with_before_completion(hook);
        self
    }

    /// Replace the entire `CompletionProviderDefaults` for the underlying
    /// completion path.
    #[must_use]
    pub fn with_completion_defaults(mut self, defaults: CompletionProviderDefaults) -> Self {
        self.base = self.base.set_defaults(defaults);
        self
    }

    /// Replace just the universal `BaseProviderDefaults` portion — preserves
    /// role-specific fields (system prompt, tools, response format,
    /// before-completion hook).
    #[must_use]
    pub fn with_base_defaults(mut self, base: BaseProviderDefaults) -> Self {
        self.base = self.base.set_base_defaults(base);
        self
    }

    /// Read-only access to the configured completion defaults.
    #[must_use]
    pub fn completion_defaults(&self) -> &CompletionProviderDefaults {
        self.base.defaults()
    }

    /// Attach defaults applied to every `text_to_speech` call.
    #[must_use]
    pub fn with_audio_speech_defaults(mut self, defaults: AudioSpeechProviderDefaults) -> Self {
        self.audio_speech_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `generate_music` / `generate_sfx` call.
    #[must_use]
    pub fn with_audio_music_defaults(mut self, defaults: AudioMusicProviderDefaults) -> Self {
        self.audio_music_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `clone_voice` call.
    #[must_use]
    pub fn with_voice_cloning_defaults(mut self, defaults: VoiceCloningProviderDefaults) -> Self {
        self.voice_cloning_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `generate_image` call.
    #[must_use]
    pub fn with_image_generation_defaults(
        mut self,
        defaults: ImageGenerationProviderDefaults,
    ) -> Self {
        self.image_generation_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `upscale_image` call.
    #[must_use]
    pub fn with_image_upscale_defaults(mut self, defaults: ImageUpscaleProviderDefaults) -> Self {
        self.image_upscale_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `text_to_video` / `image_to_video` call.
    #[must_use]
    pub fn with_video_defaults(mut self, defaults: VideoProviderDefaults) -> Self {
        self.video_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `transcribe` call.
    #[must_use]
    pub fn with_transcription_defaults(mut self, defaults: TranscriptionProviderDefaults) -> Self {
        self.transcription_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `generate_3d` call.
    #[must_use]
    pub fn with_three_d_defaults(mut self, defaults: ThreeDProviderDefaults) -> Self {
        self.three_d_defaults = Some(defaults);
        self
    }

    /// Attach defaults applied to every `remove_background` call.
    #[must_use]
    pub fn with_background_removal_defaults(
        mut self,
        defaults: BackgroundRemovalProviderDefaults,
    ) -> Self {
        self.background_removal_defaults = Some(defaults);
        self
    }

    /// Replace the API protocol marker. Built by the factories ([`ollama`],
    /// [`lm_studio`], [`openai_compat`]) and is informational only — the
    /// inner provider determines actual wire behavior.
    #[must_use]
    pub fn with_protocol(mut self, protocol: ApiProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Inherent accessor: the configured provider id of the inner provider.
    #[must_use]
    pub fn provider_id_str(&self) -> &str {
        self.inner.provider_id()
    }

    /// The protocol this handle reports.
    #[must_use]
    pub fn protocol(&self) -> &ApiProtocol {
        &self.protocol
    }

    /// Escape hatch returning the inner provider's HTTP client, if any.
    #[must_use]
    pub fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.inner.http_client()
    }

    /// Reference to the wrapped trait object.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn CustomProvider> {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// CustomProvider impl for CustomProviderHandle (nesting + default application)
// ---------------------------------------------------------------------------

#[async_trait]
impl CustomProvider for CustomProviderHandle {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config
            .as_ref()
            .or_else(|| self.inner.retry_config())
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.inner.http_client()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        // `BaseProvider` applies completion defaults then calls the inner
        // CompletionModel adapter, which forwards to `self.inner.complete`.
        self.base.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.base.stream(request).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, BlazenError> {
        self.inner.embed(texts).await
    }

    async fn text_to_speech(&self, mut req: SpeechRequest) -> Result<AudioResult, BlazenError> {
        if let Some(d) = &self.audio_speech_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.text_to_speech(req).await
    }

    async fn generate_music(&self, mut req: MusicRequest) -> Result<AudioResult, BlazenError> {
        if let Some(d) = &self.audio_music_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.generate_music(req).await
    }

    async fn generate_sfx(&self, mut req: MusicRequest) -> Result<AudioResult, BlazenError> {
        if let Some(d) = &self.audio_music_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.generate_sfx(req).await
    }

    async fn clone_voice(&self, mut req: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        if let Some(d) = &self.voice_cloning_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.clone_voice(req).await
    }

    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        self.inner.list_voices().await
    }

    async fn delete_voice(&self, voice: VoiceHandle) -> Result<(), BlazenError> {
        self.inner.delete_voice(voice).await
    }

    async fn generate_image(&self, mut req: ImageRequest) -> Result<ImageResult, BlazenError> {
        if let Some(d) = &self.image_generation_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.generate_image(req).await
    }

    async fn upscale_image(&self, mut req: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        if let Some(d) = &self.image_upscale_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.upscale_image(req).await
    }

    async fn text_to_video(&self, mut req: VideoRequest) -> Result<VideoResult, BlazenError> {
        if let Some(d) = &self.video_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.text_to_video(req).await
    }

    async fn image_to_video(&self, mut req: VideoRequest) -> Result<VideoResult, BlazenError> {
        if let Some(d) = &self.video_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.image_to_video(req).await
    }

    async fn transcribe(
        &self,
        mut req: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        if let Some(d) = &self.transcription_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.transcribe(req).await
    }

    async fn generate_3d(&self, mut req: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        if let Some(d) = &self.three_d_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.generate_3d(req).await
    }

    async fn remove_background(
        &self,
        mut req: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        if let Some(d) = &self.background_removal_defaults {
            d.apply(&mut req).await?;
        }
        self.inner.remove_background(req).await
    }
}

// ---------------------------------------------------------------------------
// CompletionModel impl
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for CustomProviderHandle {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config
            .as_ref()
            .or_else(|| self.base.retry_config())
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.inner.http_client()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        self.base.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.base.stream(request).await
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider + media-capability impls
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for CustomProviderHandle {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider does not expose raw compute job submission",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider does not expose raw compute job status",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider does not expose raw compute job result",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "CustomProvider does not expose raw compute job cancel",
        ))
    }
}

#[async_trait]
impl AudioGeneration for CustomProviderHandle {
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        <Self as CustomProvider>::text_to_speech(self, request).await
    }

    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        <Self as CustomProvider>::generate_music(self, request).await
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        <Self as CustomProvider>::generate_sfx(self, request).await
    }
}

#[async_trait]
impl VoiceCloning for CustomProviderHandle {
    async fn clone_voice(&self, request: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        <Self as CustomProvider>::clone_voice(self, request).await
    }

    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        <Self as CustomProvider>::list_voices(self).await
    }

    async fn delete_voice(&self, voice: &VoiceHandle) -> Result<(), BlazenError> {
        <Self as CustomProvider>::delete_voice(self, voice.clone()).await
    }
}

#[async_trait]
impl ImageGeneration for CustomProviderHandle {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        <Self as CustomProvider>::generate_image(self, request).await
    }

    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        <Self as CustomProvider>::upscale_image(self, request).await
    }
}

#[async_trait]
impl VideoGeneration for CustomProviderHandle {
    async fn text_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        <Self as CustomProvider>::text_to_video(self, request).await
    }

    async fn image_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        <Self as CustomProvider>::image_to_video(self, request).await
    }
}

#[async_trait]
impl Transcription for CustomProviderHandle {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        <Self as CustomProvider>::transcribe(self, request).await
    }
}

#[async_trait]
impl ThreeDGeneration for CustomProviderHandle {
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        <Self as CustomProvider>::generate_3d(self, request).await
    }
}

#[async_trait]
impl BackgroundRemoval for CustomProviderHandle {
    async fn remove_background(
        &self,
        request: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        <Self as CustomProvider>::remove_background(self, request).await
    }
}

// ---------------------------------------------------------------------------
// OpenAi-backed CustomProvider (internal, used by factories)
// ---------------------------------------------------------------------------

/// Internal: an `OpenAiCompatProvider`-backed `CustomProvider` impl. Routes
/// `complete`/`stream` through the OpenAi-compat wire client; every other
/// method inherits the trait's default `Unsupported` return.
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
struct OpenAiBackedCustomProvider {
    provider_id: String,
    inner: OpenAiCompatProvider,
}

#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
#[async_trait]
impl CustomProvider for OpenAiBackedCustomProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        <OpenAiCompatProvider as CompletionModel>::model_id(&self.inner)
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        <OpenAiCompatProvider as CompletionModel>::retry_config(&self.inner)
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        Some(OpenAiCompatProvider::http_client(&self.inner))
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.inner.stream(request).await
    }
}

// ---------------------------------------------------------------------------
// Free-function factories
// ---------------------------------------------------------------------------

/// Build a [`CustomProviderHandle`] that speaks the `OpenAI` Chat Completions
/// protocol. The supplied [`OpenAiCompatConfig`] determines base URL, model,
/// auth, and headers.
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
#[must_use]
pub fn openai_compat(
    provider_id: impl Into<String>,
    config: OpenAiCompatConfig,
) -> CustomProviderHandle {
    let id: String = provider_id.into();
    let inner: Arc<dyn CustomProvider> = Arc::new(OpenAiBackedCustomProvider {
        provider_id: id,
        inner: OpenAiCompatProvider::new(config.clone()),
    });
    let mut handle = CustomProviderHandle::new(inner);
    handle.protocol = ApiProtocol::OpenAi(config);
    handle
}

/// Convenience constructor for an Ollama server (defaults to no API key,
/// `http://{host}:{port}/v1` base URL).
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
#[must_use]
pub fn ollama(host: impl AsRef<str>, port: u16, model: impl Into<String>) -> CustomProviderHandle {
    let config = OpenAiCompatConfig {
        provider_name: "ollama".into(),
        base_url: format!("http://{}:{port}/v1", host.as_ref()),
        api_key: String::new(),
        default_model: model.into(),
        auth_method: AuthMethod::Bearer,
        extra_headers: Vec::new(),
        query_params: Vec::new(),
        supports_model_listing: true,
    };
    openai_compat("ollama", config)
}

/// Convenience constructor for an LM Studio server (defaults to no API key,
/// `http://{host}:{port}/v1` base URL; LM Studio's default port is `1234`).
#[cfg(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    feature = "reqwest",
    target_os = "wasi"
))]
#[must_use]
pub fn lm_studio(
    host: impl AsRef<str>,
    port: u16,
    model: impl Into<String>,
) -> CustomProviderHandle {
    let config = OpenAiCompatConfig {
        provider_name: "lm_studio".into(),
        base_url: format!("http://{}:{port}/v1", host.as_ref()),
        api_key: String::new(),
        default_model: model.into(),
        auth_method: AuthMethod::Bearer,
        extra_headers: Vec::new(),
        query_params: Vec::new(),
        supports_model_listing: true,
    };
    openai_compat("lm_studio", config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::GeneratedAudio;
    use crate::types::{ChatMessage, RequestTiming};

    fn empty_timing() -> RequestTiming {
        RequestTiming {
            queue_ms: None,
            execution_ms: None,
            total_ms: None,
        }
    }

    /// Stub provider that only implements `text_to_speech`. Used to exercise
    /// the typed-trait dispatch and the default `Unsupported` for everything
    /// else.
    struct StubTtsProvider;

    #[async_trait]
    impl CustomProvider for StubTtsProvider {
        fn provider_id(&self) -> &'static str {
            "stub-tts"
        }

        async fn text_to_speech(&self, _req: SpeechRequest) -> Result<AudioResult, BlazenError> {
            Ok(AudioResult {
                audio: Vec::<GeneratedAudio>::new(),
                timing: empty_timing(),
                cost: None,
                usage: None,
                audio_seconds: 0.0,
                metadata: serde_json::Value::Null,
            })
        }
    }

    /// Stub provider that implements `complete` only. Used to exercise the
    /// completion path (including `BaseProvider`-applied defaults).
    struct StubCompleteProvider;

    #[async_trait]
    impl CustomProvider for StubCompleteProvider {
        fn provider_id(&self) -> &'static str {
            "stub-complete"
        }

        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, BlazenError> {
            Ok(CompletionResponse {
                content: Some("hello from stub".to_string()),
                tool_calls: Vec::new(),
                reasoning: None,
                citations: Vec::new(),
                artifacts: Vec::new(),
                usage: None,
                model: "stub-complete".to_string(),
                finish_reason: None,
                cost: None,
                timing: None,
                images: Vec::new(),
                audio: Vec::new(),
                videos: Vec::new(),
                metadata: serde_json::Value::Null,
            })
        }
    }

    #[tokio::test]
    async fn stub_provider_dispatches_typed_method() {
        let handle = CustomProviderHandle::new(Arc::new(StubTtsProvider));
        let r = <CustomProviderHandle as CustomProvider>::text_to_speech(
            &handle,
            SpeechRequest::new("hi"),
        )
        .await
        .unwrap();
        assert_eq!(r.audio.len(), 0);
    }

    #[tokio::test]
    async fn unimplemented_method_returns_unsupported() {
        let handle = CustomProviderHandle::new(Arc::new(StubTtsProvider));
        let err = <CustomProviderHandle as CustomProvider>::generate_image(
            &handle,
            ImageRequest::new("..."),
        )
        .await
        .expect_err("expected unsupported");
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn complete_dispatches_to_inner() {
        let handle = CustomProviderHandle::new(Arc::new(StubCompleteProvider));
        let req = CompletionRequest::new(vec![ChatMessage::user("hi")]);
        let resp = <CustomProviderHandle as CompletionModel>::complete(&handle, req)
            .await
            .unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello from stub"));
    }

    #[tokio::test]
    async fn provider_id_reflects_inner() {
        let handle = CustomProviderHandle::new(Arc::new(StubTtsProvider));
        assert_eq!(handle.provider_id_str(), "stub-tts");
        assert_eq!(
            <CustomProviderHandle as ComputeProvider>::provider_id(&handle),
            "stub-tts"
        );
    }

    #[tokio::test]
    async fn completion_defaults_propagate_through_handle() {
        let handle = CustomProviderHandle::new(Arc::new(StubCompleteProvider))
            .with_system_prompt("be terse");

        let req = CompletionRequest::new(vec![ChatMessage::user("hi")]);
        let resp = <CustomProviderHandle as CompletionModel>::complete(&handle, req)
            .await
            .unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello from stub"));

        // Sanity: read back the defaults we just set.
        assert_eq!(
            handle.completion_defaults().system_prompt.as_deref(),
            Some("be terse")
        );
    }

    #[tokio::test]
    async fn audio_speech_defaults_applied_before_dispatch() {
        use crate::providers::defaults::{AudioSpeechProviderDefaults, BeforeSpeechRequestHook};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let hook: BeforeSpeechRequestHook = Arc::new(move |_req| {
            let c = Arc::clone(&counter_clone);
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });
        let handle = CustomProviderHandle::new(Arc::new(StubTtsProvider))
            .with_audio_speech_defaults(AudioSpeechProviderDefaults::new().with_before(hook));

        let _ = <CustomProviderHandle as CustomProvider>::text_to_speech(
            &handle,
            SpeechRequest::new("hello world"),
        )
        .await
        .unwrap();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "audio speech hook should fire once"
        );
    }

    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[test]
    fn ollama_factory_builds_openai_protocol() {
        let h = ollama("192.168.1.50", 11434, "llama3.1");
        assert_eq!(h.provider_id_str(), "ollama");
        assert_eq!(
            <CustomProviderHandle as CompletionModel>::model_id(&h),
            "llama3.1"
        );
        match h.protocol() {
            ApiProtocol::OpenAi(cfg) => {
                assert_eq!(cfg.base_url, "http://192.168.1.50:11434/v1");
                assert_eq!(cfg.provider_name, "ollama");
            }
            ApiProtocol::Custom => panic!("expected OpenAi protocol"),
        }
    }

    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[test]
    fn lm_studio_factory_builds_openai_protocol() {
        let h = lm_studio("localhost", 1234, "qwen2.5-coder");
        assert_eq!(h.provider_id_str(), "lm_studio");
        match h.protocol() {
            ApiProtocol::OpenAi(cfg) => {
                assert_eq!(cfg.base_url, "http://localhost:1234/v1");
                assert_eq!(cfg.provider_name, "lm_studio");
            }
            ApiProtocol::Custom => panic!("expected OpenAi protocol"),
        }
    }
}
