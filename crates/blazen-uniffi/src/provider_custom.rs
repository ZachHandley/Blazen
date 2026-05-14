//! UniFFI [`CustomProvider`] — foreign-implementable trait that mirrors the
//! upstream [`blazen_llm::providers::custom::CustomProvider`] across the
//! Go / Swift / Kotlin / Ruby FFI surface.
//!
//! ## Surface
//!
//! Two top-level types are exposed to foreign code:
//!
//! - [`CustomProvider`] (Rust ident is also `CustomProvider`) — a
//!   `#[uniffi::export(with_foreign)]` trait the foreign user implements
//!   directly. Swift sees `protocol CustomProvider`, Kotlin sees
//!   `interface CustomProvider`, Go sees `interface CustomProvider`. It has 16
//!   typed async methods + 1 sync `provider_id` accessor (the 16 are
//!   `complete`, `stream`, `embed`, and the 13 compute / media methods that
//!   match the upstream trait). Every method returns `Unsupported` in the
//!   adapter when the foreign side has not overridden it — except foreign
//!   languages require all methods to be implemented at the language level,
//!   so language-specific wrappers provide a "base class" with throwing
//!   defaults to recover ergonomics.
//!
//! - [`CustomProviderHandle`] — a `#[derive(uniffi::Object)]` opaque handle
//!   that wraps the upstream [`blazen_llm::CustomProviderHandle`] (which in
//!   turn owns an `Arc<dyn blazen_llm::CustomProvider>`). Bindings construct
//!   this via one of the four factory free functions ([`ollama`],
//!   [`lm_studio`], [`openai_compat`], [`custom_provider_from_foreign`]).
//!   Then they call any of the 16 typed methods on it, or chain
//!   `.as_base()` to reach the [`BaseProvider`] surface for system-prompt /
//!   tool / response-format defaults.
//!
//! ## Adapter
//!
//! [`UniffiToCoreCustomProviderAdapter`] is the internal bridge: it wraps an
//! `Arc<dyn CustomProvider>` (UniFFI's foreign-implemented trait object) and
//! implements [`blazen_llm::CustomProvider`] (the upstream Rust trait). On
//! each method call it converts the UniFFI record (e.g.
//! [`crate::compute_types::SpeechRequest`]) to the upstream type (e.g.
//! [`blazen_llm::compute::SpeechRequest`]), forwards to the foreign
//! implementation, and converts the response back.
//!
//! For the `stream` method, UniFFI cannot transport a
//! `Pin<Box<dyn Stream>>` across the FFI, so the foreign-facing API takes a
//! [`crate::streaming::CompletionStreamSink`] and pushes chunks into it. The
//! adapter on the Rust side bridges the sink-callback shape back into a
//! stream by allocating an internal channel.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_llm::CompletionModel as CoreCompletionModel;
use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ImageGeneration, ThreeDGeneration, Transcription,
    VideoGeneration, VoiceCloning,
};
use blazen_llm::error::BlazenError as CoreBlazenError;
use blazen_llm::providers::base::BaseProvider as CoreBaseProvider;
use blazen_llm::providers::custom::CustomProvider as CoreCustomProvider;
use blazen_llm::providers::custom::CustomProviderHandle as CoreCustomProviderHandle;
use blazen_llm::providers::openai_compat::{
    AuthMethod as CoreAuthMethod, OpenAiCompatConfig as CoreOpenAiCompatConfig,
};
use blazen_llm::types::{
    CompletionRequest as CoreCompletionRequest, EmbeddingResponse as CoreEmbeddingResponse,
    StreamChunk as CoreStreamChunk, ToolCall as CoreToolCall,
};
use futures_util::Stream;
use tokio::sync::mpsc;

use crate::compute_types::{
    AudioResult, BackgroundRemovalRequest, ImageRequest, ImageResult, MusicRequest, SpeechRequest,
    ThreeDRequest, ThreeDResult, TranscriptionRequest, TranscriptionResult, UpscaleRequest,
    VideoRequest, VideoResult, VoiceCloneRequest, VoiceHandle,
};
use crate::errors::{BlazenError, BlazenResult};
use crate::llm::{
    ChatMessage, CompletionRequest, CompletionResponse, EmbeddingResponse, TokenUsage, ToolCall,
};
use crate::provider_api_protocol::OpenAiCompatConfig;
use crate::provider_base::BaseProvider;
use crate::streaming::{CompletionStreamSink, StreamChunk as UniffiStreamChunk};

// ---------------------------------------------------------------------------
// Foreign-implementable `CustomProvider` trait
// ---------------------------------------------------------------------------

/// User-extensible provider trait the foreign side implements directly.
///
/// Mirrors [`blazen_llm::CustomProvider`] across the UniFFI boundary. Has 16
/// typed async methods (completion, streaming-via-sink, embeddings, plus 13
/// compute / media methods) and one sync `provider_id` accessor.
///
/// ## How foreign users use it
///
/// Foreign users implement this trait on their own type and pass an instance
/// to [`custom_provider_from_foreign`] to obtain a [`CustomProviderHandle`]
/// usable wherever Blazen expects a provider.
///
/// UniFFI's `with_foreign` traits require every method to be implemented at
/// the foreign language level — there is no cross-FFI Rust "default impl"
/// fallback. Each language binding ships a base class / extension that
/// supplies throwing `Unsupported` defaults so users only need to override
/// the capabilities their provider actually supports.
///
/// ## Wire-format shape
///
/// All argument and return types are UniFFI Records ([`SpeechRequest`],
/// [`AudioResult`], ...) defined in [`crate::compute_types`]. The
/// [`UniffiToCoreCustomProviderAdapter`] converts these to the upstream
/// [`blazen_llm::compute`] types on each call.
///
/// ## Async story
///
/// Every method except [`provider_id`](Self::provider_id) is `async` on the
/// Rust side. UniFFI exposes the methods as:
/// - Go: blocking functions, safe from goroutines (compose with channels)
/// - Swift: `async throws` methods
/// - Kotlin: `suspend fun` methods
/// - Ruby: blocking methods (wrap in `Async { ... }` block for fiber concurrency)
#[uniffi::export(with_foreign)]
#[async_trait]
pub trait CustomProvider: Send + Sync {
    /// Stable provider identifier for logs and metrics.
    fn provider_id(&self) -> String;

    /// Perform a non-streaming chat completion.
    async fn complete(&self, request: CompletionRequest) -> BlazenResult<CompletionResponse>;

    /// Perform a streaming chat completion, pushing chunks into the supplied
    /// sink. The implementation must call `sink.on_done` exactly once on
    /// success or `sink.on_error` exactly once on failure.
    async fn stream(
        &self,
        request: CompletionRequest,
        sink: Arc<dyn CompletionStreamSink>,
    ) -> BlazenResult<()>;

    /// Embed one or more texts.
    async fn embed(&self, texts: Vec<String>) -> BlazenResult<EmbeddingResponse>;

    /// Synthesize speech from text.
    async fn text_to_speech(&self, request: SpeechRequest) -> BlazenResult<AudioResult>;

    /// Generate music from a prompt.
    async fn generate_music(&self, request: MusicRequest) -> BlazenResult<AudioResult>;

    /// Generate sound effects from a prompt.
    async fn generate_sfx(&self, request: MusicRequest) -> BlazenResult<AudioResult>;

    /// Clone a voice from reference audio.
    async fn clone_voice(&self, request: VoiceCloneRequest) -> BlazenResult<VoiceHandle>;

    /// List voices known to the provider.
    async fn list_voices(&self) -> BlazenResult<Vec<VoiceHandle>>;

    /// Delete a previously-cloned voice.
    async fn delete_voice(&self, voice: VoiceHandle) -> BlazenResult<()>;

    /// Generate images from a prompt.
    async fn generate_image(&self, request: ImageRequest) -> BlazenResult<ImageResult>;

    /// Upscale an existing image.
    async fn upscale_image(&self, request: UpscaleRequest) -> BlazenResult<ImageResult>;

    /// Generate a video from a text prompt.
    async fn text_to_video(&self, request: VideoRequest) -> BlazenResult<VideoResult>;

    /// Generate a video from a source image + prompt.
    async fn image_to_video(&self, request: VideoRequest) -> BlazenResult<VideoResult>;

    /// Transcribe audio to text.
    async fn transcribe(&self, request: TranscriptionRequest) -> BlazenResult<TranscriptionResult>;

    /// Generate a 3D model.
    async fn generate_3d(&self, request: ThreeDRequest) -> BlazenResult<ThreeDResult>;

    /// Remove the background from an image.
    async fn remove_background(
        &self,
        request: BackgroundRemovalRequest,
    ) -> BlazenResult<ImageResult>;
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert an upstream `serde_json::Value` "tool calls" snapshot into UniFFI
/// [`ToolCall`]s, mirroring `From<CoreToolCall> for ToolCall` in [`crate::llm`].
fn tool_calls_to_uniffi(calls: Vec<CoreToolCall>) -> Vec<ToolCall> {
    calls.into_iter().map(ToolCall::from).collect()
}

/// Convert an upstream [`CoreStreamChunk`] into the UniFFI [`UniffiStreamChunk`]
/// record. Mirrors the flattening done by the streaming free function in
/// [`crate::streaming`].
fn stream_chunk_to_uniffi(chunk: &CoreStreamChunk) -> UniffiStreamChunk {
    UniffiStreamChunk {
        content_delta: chunk.delta.clone().unwrap_or_default(),
        tool_calls: tool_calls_to_uniffi(chunk.tool_calls.clone()),
        is_final: chunk.finish_reason.is_some(),
    }
}

/// Translate a UniFFI-side [`BlazenError`] back into the upstream
/// [`CoreBlazenError`] so the framework's typed error matching keeps working
/// across the dispatch boundary. Mirrors the lossy direction taken by the
/// old `host_dispatch.rs` conversion — preserve the variant family and the
/// message; treat unknown variants as a provider error.
fn core_error_from_ffi(err: BlazenError) -> CoreBlazenError {
    match err {
        BlazenError::Auth { message } => CoreBlazenError::Auth { message },
        BlazenError::Unsupported { message } => CoreBlazenError::Unsupported { message },
        BlazenError::Validation { message } => CoreBlazenError::Validation {
            field: None,
            message,
        },
        BlazenError::ContentPolicy { message } => CoreBlazenError::ContentPolicy { message },
        BlazenError::RateLimit {
            retry_after_ms,
            message: _,
        } => CoreBlazenError::RateLimit { retry_after_ms },
        BlazenError::Timeout {
            elapsed_ms,
            message: _,
        } => CoreBlazenError::Timeout { elapsed_ms },
        BlazenError::Tool { message } => CoreBlazenError::Tool {
            name: None,
            message,
        },
        BlazenError::Provider {
            kind: _,
            message,
            provider,
            status,
            ..
        } => CoreBlazenError::Provider {
            provider: provider.unwrap_or_else(|| "custom-provider".to_owned()),
            message,
            status_code: status.and_then(|s| u16::try_from(s).ok()),
        },
        other => CoreBlazenError::provider("custom-provider", other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Adapter: UniFFI `CustomProvider` -> blazen-llm `CustomProvider`
// ---------------------------------------------------------------------------

/// Bridges a foreign-implemented `Arc<dyn CustomProvider>` (the UniFFI
/// trait) into the upstream [`CoreCustomProvider`] trait that
/// [`blazen_llm::CustomProviderHandle`] depends on.
///
/// On each method call:
/// 1. Converts the upstream request type to its UniFFI record counterpart
///    (e.g. [`blazen_llm::compute::SpeechRequest`] ->
///    [`crate::compute_types::SpeechRequest`]).
/// 2. Forwards through the foreign impl.
/// 3. Converts the returned UniFFI record back to the upstream type.
///
/// Foreign callers never see this type — it is instantiated internally by
/// [`custom_provider_from_foreign`].
pub(crate) struct UniffiToCoreCustomProviderAdapter {
    inner: Arc<dyn CustomProvider>,
    /// Cached `provider_id` so the upstream trait's `&str` return shape can
    /// borrow without reaching across the FFI on every access.
    cached_provider_id: String,
}

impl UniffiToCoreCustomProviderAdapter {
    /// Wrap a foreign-implemented [`CustomProvider`] as an
    /// `Arc<dyn CoreCustomProvider>` ready for
    /// [`blazen_llm::CustomProviderHandle::new`].
    pub(crate) fn new(inner: Arc<dyn CustomProvider>) -> Arc<dyn CoreCustomProvider> {
        let cached_provider_id = inner.provider_id();
        Arc::new(Self {
            inner,
            cached_provider_id,
        })
    }

    /// Helper: dispatch a UniFFI `CompletionRequest` into the upstream
    /// `CoreCompletionRequest` shape and forward to the inner foreign
    /// `complete`.
    async fn complete_inner(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<blazen_llm::types::CompletionResponse, CoreBlazenError> {
        let ffi_request = core_completion_request_to_ffi(request)?;
        let resp = self
            .inner
            .complete(ffi_request)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(ffi_completion_response_to_core(resp))
    }
}

/// Convert an upstream [`CoreCompletionRequest`] into the UniFFI
/// [`CompletionRequest`] record. Loses the multimodal `modalities` /
/// `image_config` / `audio_config` slots since the UniFFI record doesn't
/// surface them (matching the asymmetry already present in `crate::llm`).
fn core_completion_request_to_ffi(
    req: CoreCompletionRequest,
) -> Result<CompletionRequest, CoreBlazenError> {
    let messages = req.messages.into_iter().map(ChatMessage::from).collect();
    let tools: Vec<crate::llm::Tool> = req.tools.into_iter().map(Into::into).collect();
    let response_format_json = req
        .response_format
        .map(|v| serde_json::to_string(&v))
        .transpose()
        .map_err(|e| CoreBlazenError::Serialization(e.to_string()))?;
    Ok(CompletionRequest {
        messages,
        tools,
        temperature: req.temperature.map(f64::from),
        max_tokens: req.max_tokens,
        top_p: req.top_p.map(f64::from),
        model: req.model,
        response_format_json,
        system: None,
    })
}

/// Convert a UniFFI [`CompletionResponse`] into the upstream
/// `blazen_llm::CompletionResponse`. The UniFFI shape collapses several
/// upstream optional fields (`reasoning`, `citations`, `artifacts`, multimodal
/// outputs) to empty / `None`; the foreign-side provider doesn't surface
/// them.
fn ffi_completion_response_to_core(
    resp: CompletionResponse,
) -> blazen_llm::types::CompletionResponse {
    let usage = if resp.usage.total_tokens == 0
        && resp.usage.prompt_tokens == 0
        && resp.usage.completion_tokens == 0
    {
        None
    } else {
        Some(token_usage_ffi_to_core(resp.usage))
    };
    let tool_calls = resp
        .tool_calls
        .into_iter()
        .filter_map(|tc| CoreToolCall::try_from(tc).ok())
        .collect();
    let finish_reason = if resp.finish_reason.is_empty() {
        None
    } else {
        Some(resp.finish_reason)
    };
    blazen_llm::types::CompletionResponse {
        content: if resp.content.is_empty() {
            None
        } else {
            Some(resp.content)
        },
        tool_calls,
        reasoning: None,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage,
        model: resp.model,
        finish_reason,
        cost: None,
        timing: None,
        images: Vec::new(),
        audio: Vec::new(),
        videos: Vec::new(),
        metadata: serde_json::Value::Null,
    }
}

/// Map the UniFFI [`TokenUsage`] (`u64`s) back to the upstream
/// [`blazen_llm::types::TokenUsage`] (`u32`s).
#[allow(clippy::cast_possible_truncation)]
fn token_usage_ffi_to_core(u: TokenUsage) -> blazen_llm::types::TokenUsage {
    blazen_llm::types::TokenUsage {
        prompt_tokens: u.prompt_tokens.min(u64::from(u32::MAX)) as u32,
        completion_tokens: u.completion_tokens.min(u64::from(u32::MAX)) as u32,
        total_tokens: u.total_tokens.min(u64::from(u32::MAX)) as u32,
        cached_input_tokens: u.cached_input_tokens.min(u64::from(u32::MAX)) as u32,
        reasoning_tokens: u.reasoning_tokens.min(u64::from(u32::MAX)) as u32,
        audio_input_tokens: 0,
        audio_output_tokens: 0,
    }
}

/// Map an upstream [`CoreEmbeddingResponse`] into the UniFFI shape. Foreign
/// users return the UniFFI shape from their `embed` impl; the inverse is what
/// we need here.
fn ffi_embedding_response_to_core(resp: EmbeddingResponse) -> CoreEmbeddingResponse {
    let embeddings = resp
        .embeddings
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
        .collect();
    let usage = if resp.usage.total_tokens == 0
        && resp.usage.prompt_tokens == 0
        && resp.usage.completion_tokens == 0
    {
        None
    } else {
        Some(token_usage_ffi_to_core(resp.usage))
    };
    CoreEmbeddingResponse {
        embeddings,
        model: resp.model,
        usage,
        cost: None,
        timing: None,
        metadata: serde_json::Value::Null,
    }
}

/// Internal stream sink that pushes chunks into a channel for the upstream
/// `stream()` adapter. Mirrors the channel pattern the foreign-side
/// streaming sinks use, just inverted: the foreign impl is the producer here,
/// and the channel receiver becomes the Rust-side `Stream`.
struct ChannelSink {
    tx: mpsc::UnboundedSender<Result<CoreStreamChunk, CoreBlazenError>>,
}

#[async_trait]
impl CompletionStreamSink for ChannelSink {
    async fn on_chunk(&self, chunk: UniffiStreamChunk) -> BlazenResult<()> {
        let core_chunk = CoreStreamChunk {
            delta: if chunk.content_delta.is_empty() {
                None
            } else {
                Some(chunk.content_delta)
            },
            tool_calls: chunk
                .tool_calls
                .into_iter()
                .filter_map(|tc| CoreToolCall::try_from(tc).ok())
                .collect(),
            finish_reason: None,
            reasoning_delta: None,
            citations: Vec::new(),
            artifacts: Vec::new(),
        };
        // Channel-closed errors are silently dropped — the consumer has gone
        // away (e.g. dropped the stream early). The foreign impl will see the
        // next `on_chunk` succeed-or-error normally; we don't need to fail
        // here.
        let _ = self.tx.send(Ok(core_chunk));
        Ok(())
    }

    async fn on_done(&self, finish_reason: String, _usage: TokenUsage) -> BlazenResult<()> {
        let chunk = CoreStreamChunk {
            delta: None,
            tool_calls: Vec::new(),
            finish_reason: if finish_reason.is_empty() {
                Some("stop".to_owned())
            } else {
                Some(finish_reason)
            },
            reasoning_delta: None,
            citations: Vec::new(),
            artifacts: Vec::new(),
        };
        let _ = self.tx.send(Ok(chunk));
        Ok(())
    }

    async fn on_error(&self, err: BlazenError) -> BlazenResult<()> {
        let _ = self.tx.send(Err(core_error_from_ffi(err)));
        Ok(())
    }
}

#[async_trait]
impl CoreCustomProvider for UniffiToCoreCustomProviderAdapter {
    fn provider_id(&self) -> &str {
        &self.cached_provider_id
    }

    async fn complete(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<blazen_llm::types::CompletionResponse, CoreBlazenError> {
        self.complete_inner(request).await
    }

    async fn stream(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CoreStreamChunk, CoreBlazenError>> + Send>>,
        CoreBlazenError,
    > {
        let ffi_request = core_completion_request_to_ffi(request)?;
        let (tx, rx) = mpsc::unbounded_channel();
        let sink: Arc<dyn CompletionStreamSink> = Arc::new(ChannelSink { tx });
        let inner = Arc::clone(&self.inner);
        // Drive the foreign `stream` impl in a background task. The task's
        // outcome flows back through the channel (success terminates via
        // `on_done`, failure via `on_error`); the spawn result itself only
        // matters if the task panicked, in which case we synthesize an
        // internal error chunk.
        tokio::spawn(async move {
            if let Err(e) = inner.stream(ffi_request, Arc::clone(&sink)).await {
                // Convert the error into a synthetic on_error invocation so
                // the consumer observes a single terminal failure.
                let _ = sink.on_error(e).await;
            }
        });
        // Adapt the unbounded channel receiver into the Stream the upstream
        // trait expects. `unfold` carries a `(receiver, terminated)` tuple so
        // we can short-circuit once we see a finish-reason chunk or an error
        // — the foreign sink contract says nothing else flows after those.
        let stream = futures_util::stream::unfold((rx, false), |(mut rx, terminated)| async move {
            if terminated {
                return None;
            }
            let item = rx.recv().await?;
            let is_terminal =
                matches!(&item, Ok(c) if c.finish_reason.is_some()) || matches!(&item, Err(_));
            Some((item, (rx, is_terminal)))
        });
        Ok(Box::pin(stream))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<CoreEmbeddingResponse, CoreBlazenError> {
        let resp = self.inner.embed(texts).await.map_err(core_error_from_ffi)?;
        Ok(ffi_embedding_response_to_core(resp))
    }

    async fn text_to_speech(
        &self,
        req: blazen_llm::compute::SpeechRequest,
    ) -> Result<blazen_llm::compute::AudioResult, CoreBlazenError> {
        let ffi: SpeechRequest = req.into();
        let r = self
            .inner
            .text_to_speech(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn generate_music(
        &self,
        req: blazen_llm::compute::MusicRequest,
    ) -> Result<blazen_llm::compute::AudioResult, CoreBlazenError> {
        let ffi: MusicRequest = req.into();
        let r = self
            .inner
            .generate_music(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn generate_sfx(
        &self,
        req: blazen_llm::compute::MusicRequest,
    ) -> Result<blazen_llm::compute::AudioResult, CoreBlazenError> {
        let ffi: MusicRequest = req.into();
        let r = self
            .inner
            .generate_sfx(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn clone_voice(
        &self,
        req: blazen_llm::compute::VoiceCloneRequest,
    ) -> Result<blazen_llm::compute::VoiceHandle, CoreBlazenError> {
        let ffi: VoiceCloneRequest = req.into();
        let r = self
            .inner
            .clone_voice(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn list_voices(&self) -> Result<Vec<blazen_llm::compute::VoiceHandle>, CoreBlazenError> {
        let r = self
            .inner
            .list_voices()
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into_iter().map(Into::into).collect())
    }

    async fn delete_voice(
        &self,
        voice: blazen_llm::compute::VoiceHandle,
    ) -> Result<(), CoreBlazenError> {
        let ffi: VoiceHandle = voice.into();
        self.inner
            .delete_voice(ffi)
            .await
            .map_err(core_error_from_ffi)
    }

    async fn generate_image(
        &self,
        req: blazen_llm::compute::ImageRequest,
    ) -> Result<blazen_llm::compute::ImageResult, CoreBlazenError> {
        let ffi: ImageRequest = req.into();
        let r = self
            .inner
            .generate_image(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn upscale_image(
        &self,
        req: blazen_llm::compute::UpscaleRequest,
    ) -> Result<blazen_llm::compute::ImageResult, CoreBlazenError> {
        let ffi: UpscaleRequest = req.into();
        let r = self
            .inner
            .upscale_image(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn text_to_video(
        &self,
        req: blazen_llm::compute::VideoRequest,
    ) -> Result<blazen_llm::compute::VideoResult, CoreBlazenError> {
        let ffi: VideoRequest = req.into();
        let r = self
            .inner
            .text_to_video(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn image_to_video(
        &self,
        req: blazen_llm::compute::VideoRequest,
    ) -> Result<blazen_llm::compute::VideoResult, CoreBlazenError> {
        let ffi: VideoRequest = req.into();
        let r = self
            .inner
            .image_to_video(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn transcribe(
        &self,
        req: blazen_llm::compute::TranscriptionRequest,
    ) -> Result<blazen_llm::compute::TranscriptionResult, CoreBlazenError> {
        let ffi: TranscriptionRequest = req.into();
        let r = self
            .inner
            .transcribe(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn generate_3d(
        &self,
        req: blazen_llm::compute::ThreeDRequest,
    ) -> Result<blazen_llm::compute::ThreeDResult, CoreBlazenError> {
        let ffi: ThreeDRequest = req.into();
        let r = self
            .inner
            .generate_3d(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }

    async fn remove_background(
        &self,
        req: blazen_llm::compute::BackgroundRemovalRequest,
    ) -> Result<blazen_llm::compute::ImageResult, CoreBlazenError> {
        let ffi: BackgroundRemovalRequest = req.into();
        let r = self
            .inner
            .remove_background(ffi)
            .await
            .map_err(core_error_from_ffi)?;
        Ok(r.into())
    }
}

// ---------------------------------------------------------------------------
// `CustomProviderHandle` UniFFI Object
// ---------------------------------------------------------------------------

/// Opaque UniFFI handle that wraps the upstream
/// [`blazen_llm::CustomProviderHandle`].
///
/// Construct via one of the four free factory functions ([`ollama`],
/// [`lm_studio`], [`openai_compat`], [`custom_provider_from_foreign`]). All
/// 16 typed compute / completion methods dispatch through the inner handle,
/// which applies any per-instance defaults attached via the builders before
/// forwarding to the underlying [`CustomProvider`].
///
/// The paired [`BaseProvider`] handle returned by [`as_base`](Self::as_base)
/// exposes builder-style completion-defaults customisation
/// (`with_system_prompt`, `with_tools_json`, ...).
#[derive(uniffi::Object)]
pub struct CustomProviderHandle {
    /// Upstream handle. Holds the `Arc<dyn CoreCustomProvider>` plus all
    /// per-instance defaults the builders configure.
    inner: parking_lot::RwLock<CoreCustomProviderHandle>,
    /// Paired [`BaseProvider`] handle so foreign callers can chain
    /// `.with_system_prompt(...)` etc. via [`as_base`](Self::as_base) and
    /// hand the result to APIs taking a generic `CompletionModel`.
    base: Arc<BaseProvider>,
}

impl CustomProviderHandle {
    /// Internal: build a [`CustomProviderHandle`] from an upstream
    /// [`CoreCustomProviderHandle`].
    fn from_core(core: CoreCustomProviderHandle) -> Arc<Self> {
        let completion_model: Arc<dyn CoreCompletionModel> = Arc::new(core.clone());
        let base = BaseProvider::from_core(CoreBaseProvider::new(completion_model));
        Arc::new(Self {
            inner: parking_lot::RwLock::new(core),
            base,
        })
    }

    /// Internal: snapshot the upstream handle. Used by builder methods.
    fn snapshot(&self) -> CoreCustomProviderHandle {
        self.inner.read().clone()
    }
}

#[uniffi::export]
impl CustomProviderHandle {
    /// Return the paired [`BaseProvider`] handle for builder-style chaining.
    ///
    /// Use for `.with_system_prompt(...)`, `.with_tools_json(...)`,
    /// `.with_response_format_json(...)`, or to hand the provider to an API
    /// expecting an opaque `CompletionModel`-shaped handle.
    #[must_use]
    pub fn as_base(self: Arc<Self>) -> Arc<BaseProvider> {
        Arc::clone(&self.base)
    }

    /// The provider id of the wrapped inner provider.
    #[must_use]
    pub fn provider_id(self: Arc<Self>) -> String {
        self.inner.read().provider_id_str().to_owned()
    }

    /// Attach a default system prompt applied to every completion request
    /// that doesn't already include a system message.
    ///
    /// Returns a fresh handle (clone-with-mutation) so the call composes
    /// fluently with other `with_*` builders.
    #[must_use]
    pub fn with_system_prompt(self: Arc<Self>, prompt: String) -> Arc<Self> {
        let next = self.snapshot().with_system_prompt(prompt);
        Self::from_core(next)
    }

    /// Replace the default tool list. JSON-encoded `Vec<ToolDefinition>`.
    ///
    /// Malformed JSON or an empty string yields an empty tool list — matching
    /// the permissive shape of the other `*_json` helpers on this type.
    #[must_use]
    pub fn with_tools_json(self: Arc<Self>, tools_json: String) -> Arc<Self> {
        let tools = if tools_json.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&tools_json).unwrap_or_default()
        };
        let next = self.snapshot().with_tools(tools);
        Self::from_core(next)
    }

    /// Set the default `response_format`. JSON-encoded `serde_json::Value`.
    ///
    /// Malformed JSON or an empty string is treated as JSON null (no default
    /// response format).
    #[must_use]
    pub fn with_response_format_json(self: Arc<Self>, fmt_json: String) -> Arc<Self> {
        let value: serde_json::Value = if fmt_json.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&fmt_json).unwrap_or(serde_json::Value::Null)
        };
        let next = self.snapshot().with_response_format(value);
        Self::from_core(next)
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl CustomProviderHandle {
    /// Perform a non-streaming chat completion. Applies any configured
    /// completion defaults (system prompt, tools, response format) before
    /// dispatching to the inner provider.
    pub async fn complete(
        self: Arc<Self>,
        request: CompletionRequest,
    ) -> BlazenResult<CompletionResponse> {
        let core_request = CoreCompletionRequest::try_from(request)?;
        let handle = self.snapshot();
        let response =
            <CoreCustomProviderHandle as CoreCompletionModel>::complete(&handle, core_request)
                .await
                .map_err(BlazenError::from)?;
        Ok(CompletionResponse::from(response))
    }

    /// Drive a streaming chat completion, dispatching each chunk to the sink.
    ///
    /// Symmetric with [`crate::streaming::complete_streaming`]: success and
    /// failure are both delivered via the sink; the function itself only
    /// returns `Err` if the initial request conversion fails.
    pub async fn stream(
        self: Arc<Self>,
        request: CompletionRequest,
        sink: Arc<dyn CompletionStreamSink>,
    ) -> BlazenResult<()> {
        use futures_util::StreamExt;
        let core_request = CoreCompletionRequest::try_from(request)?;
        let handle = self.snapshot();
        let mut stream =
            match <CoreCustomProviderHandle as CoreCompletionModel>::stream(&handle, core_request)
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    let _ = sink.on_error(BlazenError::from(e)).await;
                    return Ok(());
                }
            };
        let mut last_finish_reason = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    if let Some(reason) = &chunk.finish_reason {
                        last_finish_reason = reason.clone();
                    }
                    let ffi_chunk = stream_chunk_to_uniffi(&chunk);
                    if let Err(e) = sink.on_chunk(ffi_chunk).await {
                        // Sink-side error short-circuits the stream — surface
                        // it back through `on_error` to keep the contract.
                        let _ = sink.on_error(e).await;
                        return Ok(());
                    }
                }
                Err(e) => {
                    let _ = sink.on_error(BlazenError::from(e)).await;
                    return Ok(());
                }
            }
        }
        let _ = sink
            .on_done(last_finish_reason, TokenUsage::default())
            .await;
        Ok(())
    }

    /// Embed one or more texts via the inner provider.
    pub async fn embed(self: Arc<Self>, texts: Vec<String>) -> BlazenResult<EmbeddingResponse> {
        let handle = self.snapshot();
        let response = <CoreCustomProviderHandle as CoreCustomProvider>::embed(&handle, texts)
            .await
            .map_err(BlazenError::from)?;
        Ok(EmbeddingResponse::from(response))
    }

    /// Synthesize speech from text. Applies the configured speech defaults
    /// (if any) before dispatching to the inner provider.
    pub async fn text_to_speech(
        self: Arc<Self>,
        request: SpeechRequest,
    ) -> BlazenResult<AudioResult> {
        let handle = self.snapshot();
        let r = AudioGeneration::text_to_speech(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Generate music from a prompt.
    pub async fn generate_music(
        self: Arc<Self>,
        request: MusicRequest,
    ) -> BlazenResult<AudioResult> {
        let handle = self.snapshot();
        let r = AudioGeneration::generate_music(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Generate sound effects from a prompt.
    pub async fn generate_sfx(self: Arc<Self>, request: MusicRequest) -> BlazenResult<AudioResult> {
        let handle = self.snapshot();
        let r = AudioGeneration::generate_sfx(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Clone a voice from reference audio.
    pub async fn clone_voice(
        self: Arc<Self>,
        request: VoiceCloneRequest,
    ) -> BlazenResult<VoiceHandle> {
        let handle = self.snapshot();
        let r = VoiceCloning::clone_voice(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// List voices known to the provider.
    pub async fn list_voices(self: Arc<Self>) -> BlazenResult<Vec<VoiceHandle>> {
        let handle = self.snapshot();
        let voices = VoiceCloning::list_voices(&handle)
            .await
            .map_err(BlazenError::from)?;
        Ok(voices.into_iter().map(VoiceHandle::from).collect())
    }

    /// Delete a previously-cloned voice. Takes the voice id as a string so
    /// foreign callers can pass `voice_handle.id` directly without
    /// reconstructing the full record.
    pub async fn delete_voice(self: Arc<Self>, voice_id: String) -> BlazenResult<bool> {
        let handle = self.snapshot();
        let core_voice = blazen_llm::compute::VoiceHandle {
            id: voice_id,
            name: String::new(),
            provider: handle.provider_id_str().to_owned(),
            language: None,
            description: None,
            metadata: serde_json::Value::Null,
        };
        VoiceCloning::delete_voice(&handle, &core_voice)
            .await
            .map_err(BlazenError::from)?;
        Ok(true)
    }

    /// Generate an image from a text prompt.
    pub async fn generate_image(
        self: Arc<Self>,
        request: ImageRequest,
    ) -> BlazenResult<ImageResult> {
        let handle = self.snapshot();
        let r = ImageGeneration::generate_image(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Upscale an existing image.
    pub async fn upscale_image(
        self: Arc<Self>,
        request: UpscaleRequest,
    ) -> BlazenResult<ImageResult> {
        let handle = self.snapshot();
        let r = ImageGeneration::upscale_image(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Generate a video from a text prompt.
    pub async fn text_to_video(
        self: Arc<Self>,
        request: VideoRequest,
    ) -> BlazenResult<VideoResult> {
        let handle = self.snapshot();
        let r = VideoGeneration::text_to_video(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Generate a video from a reference image.
    pub async fn image_to_video(
        self: Arc<Self>,
        request: VideoRequest,
    ) -> BlazenResult<VideoResult> {
        let handle = self.snapshot();
        let r = VideoGeneration::image_to_video(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Transcribe audio to text.
    pub async fn transcribe(
        self: Arc<Self>,
        request: TranscriptionRequest,
    ) -> BlazenResult<TranscriptionResult> {
        let handle = self.snapshot();
        let r = Transcription::transcribe(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Generate a 3D model.
    pub async fn generate_3d(
        self: Arc<Self>,
        request: ThreeDRequest,
    ) -> BlazenResult<ThreeDResult> {
        let handle = self.snapshot();
        let r = ThreeDGeneration::generate_3d(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }

    /// Remove the background from an existing image.
    pub async fn remove_background(
        self: Arc<Self>,
        request: BackgroundRemovalRequest,
    ) -> BlazenResult<ImageResult> {
        let handle = self.snapshot();
        let r = BackgroundRemoval::remove_background(&handle, request.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(r.into())
    }
}

// ---------------------------------------------------------------------------
// Free-function factories
// ---------------------------------------------------------------------------

/// Build a [`CustomProviderHandle`] for an arbitrary OpenAI-compatible
/// backend.
///
/// Use for vLLM, llama.cpp's server, TGI, hosted OpenAI-compat services —
/// anything that speaks the official OpenAI chat-completions wire format. The
/// supplied [`OpenAiCompatConfig`] selects base URL, model, auth method,
/// headers, and query parameters.
#[uniffi::export]
#[must_use]
pub fn openai_compat(provider_id: String, config: OpenAiCompatConfig) -> Arc<CustomProviderHandle> {
    let core_config: CoreOpenAiCompatConfig = config.into();
    let core = blazen_llm::openai_compat(provider_id, core_config);
    CustomProviderHandle::from_core(core)
}

/// Convenience: build a [`CustomProviderHandle`] for a local Ollama server.
///
/// Equivalent to [`openai_compat`] with `base_url = http://{host}:{port}/v1`
/// and no API key. Defaults: `host = "localhost"`, `port = 11434`.
#[uniffi::export]
#[must_use]
pub fn ollama(model: String, host: Option<String>, port: Option<u16>) -> Arc<CustomProviderHandle> {
    let host = host.unwrap_or_else(|| "localhost".to_owned());
    let port = port.unwrap_or(11434);
    let core = blazen_llm::ollama(host, port, model);
    CustomProviderHandle::from_core(core)
}

/// Convenience: build a [`CustomProviderHandle`] for an LM Studio server.
///
/// Equivalent to [`openai_compat`] with `base_url = http://{host}:{port}/v1`
/// and no API key. Defaults: `host = "localhost"`, `port = 1234`.
#[uniffi::export]
#[must_use]
pub fn lm_studio(
    model: String,
    host: Option<String>,
    port: Option<u16>,
) -> Arc<CustomProviderHandle> {
    let host = host.unwrap_or_else(|| "localhost".to_owned());
    let port = port.unwrap_or(1234);
    let core = blazen_llm::lm_studio(host, port, model);
    CustomProviderHandle::from_core(core)
}

/// Build a [`CustomProviderHandle`] from a foreign-implemented
/// [`CustomProvider`].
///
/// This is the factory foreign users invoke after implementing the
/// `CustomProvider` protocol/interface/trait on their own type:
///
/// ```kotlin
/// class MyProvider : CustomProvider { /* ... 16 methods ... */ }
/// val handle = customProviderFromForeign(MyProvider())
/// val resp = handle.complete(request)
/// ```
///
/// The handle holds an internal adapter that converts UniFFI records to
/// upstream `blazen_llm::compute` types on each call.
#[uniffi::export]
#[must_use]
pub fn custom_provider_from_foreign(
    provider: Arc<dyn CustomProvider>,
) -> Arc<CustomProviderHandle> {
    let adapter = UniffiToCoreCustomProviderAdapter::new(provider);
    let core = CoreCustomProviderHandle::new(adapter);
    CustomProviderHandle::from_core(core)
}

// ---------------------------------------------------------------------------
// Convenience: build a fully-specified `OpenAiCompatConfig`.
// ---------------------------------------------------------------------------

/// Build a fully-specified [`OpenAiCompatConfig`] from positional arguments.
///
/// Convenience for foreign callers that don't want to construct the
/// [`OpenAiCompatConfig`] record by hand. Mirrors the
/// [`openai_compat`] factory's shape.
#[uniffi::export]
#[must_use]
pub fn new_openai_compat_config(
    provider_name: String,
    base_url: String,
    api_key: String,
    default_model: String,
    auth_method: crate::provider_api_protocol::AuthMethod,
    supports_model_listing: bool,
) -> OpenAiCompatConfig {
    let core = CoreOpenAiCompatConfig {
        provider_name,
        base_url,
        api_key,
        default_model,
        auth_method: CoreAuthMethod::from(auth_method),
        extra_headers: Vec::new(),
        query_params: Vec::new(),
        supports_model_listing,
    };
    (&core).into()
}
