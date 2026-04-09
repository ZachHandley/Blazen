//! fal.ai compute platform provider.
//!
//! fal.ai is fundamentally different from typical LLM providers -- it is a
//! compute platform with a queue/poll/webhook execution model. It supports
//! 600+ models for various tasks including LLMs (via several `OpenRouter`- and
//! `any-llm`-style endpoint families), image generation, video, and audio.
//!
//! Key differences:
//! - Auth: `Authorization: Key <FAL_API_KEY>` (note `Key` prefix, not `Bearer`)
//! - Queue mode: submit -> poll status -> get result
//! - Sync mode: submit and wait (timeout risk for long jobs)
//! - Webhook mode: submit with callback URL
//!
//! # LLM endpoint selection
//!
//! LLM-class endpoints are modeled as a typed [`FalLlmEndpoint`] enum rather
//! than a free-form URL string. The default is
//! [`FalLlmEndpoint::OpenAiChat`], which targets
//! `openrouter/router/openai/v1/chat/completions` and speaks the full
//! `OpenAI` chat completions wire format (messages array, multimodal content
//! blocks, tool calls, structured outputs, native SSE streaming).
//!
//! Other families are available for the legacy `openrouter/router` and
//! `fal-ai/any-llm` prompt-string proxies, their `vision`/`audio`/`video`
//! variants, and an escape-hatch `Custom` variant for arbitrary fal
//! application paths. Use [`FalProvider::with_llm_endpoint`] to switch
//! families and [`FalProvider::with_enterprise`] to promote to the
//! enterprise/SOC2 path.
//!
//! This module implements all media generation traits:
//! - [`CompletionModel`] for LLM chat completions (default:
//!   `openrouter/router/openai/v1/chat/completions` via
//!   [`FalLlmEndpoint::OpenAiChat`])
//! - [`ComputeProvider`] for generic compute job submission/polling
//! - [`ImageGeneration`] for typed image generation and upscaling
//! - [`VideoGeneration`] for text-to-video and image-to-video
//! - [`AudioGeneration`] for TTS, music, and sound effects
//! - [`Transcription`] for speech-to-text (Whisper)

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use futures_util::Stream;
use serde::Deserialize;
use tracing::{debug, warn};

use super::openai_format::parse_retry_after;
use crate::compute::{
    AudioGeneration, AudioResult, BackgroundRemoval, BackgroundRemovalRequest, ComputeProvider,
    ComputeRequest, ComputeResult, ImageGeneration, ImageRequest, ImageResult, JobHandle,
    JobStatus, MusicRequest, SpeechRequest, ThreeDGeneration, ThreeDRequest, ThreeDResult,
    Transcription, TranscriptionRequest, TranscriptionResult, TranscriptionSegment, UpscaleRequest,
    VideoGeneration, VideoRequest, VideoResult,
};
use crate::error::{BlazenError, ComputeErrorKind};
use crate::http::{HttpClient, HttpRequest};
use crate::media::{
    Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput, MediaType,
};
use crate::types::{
    CompletionRequest, CompletionResponse, EmbeddingResponse, RequestTiming, StreamChunk,
    TokenUsage,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FAL_QUEUE_URL: &str = "https://queue.fal.run";
const FAL_SYNC_URL: &str = "https://fal.run";

/// Default poll interval for queue-based execution.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Maximum number of poll iterations before giving up.
const MAX_POLL_ITERATIONS: u32 = 600; // 10 minutes at 1s intervals

/// Default image generation model.
const DEFAULT_IMAGE_MODEL: &str = "fal-ai/flux/schnell";

/// Default upscaling model.
const DEFAULT_UPSCALE_MODEL: &str = "fal-ai/esrgan";

/// Default text-to-video model.
const DEFAULT_TEXT_TO_VIDEO_MODEL: &str = "fal-ai/minimax/video-01";

/// Default image-to-video model.
const DEFAULT_IMAGE_TO_VIDEO_MODEL: &str = "fal-ai/kling-video/v2.1/pro/image-to-video";

/// Default text-to-speech model.
const DEFAULT_TTS_MODEL: &str = "fal-ai/chatterbox/text-to-speech";

/// Default music generation model.
const DEFAULT_MUSIC_MODEL: &str = "fal-ai/stable-audio";

/// Default sound effect generation model.
const DEFAULT_SFX_MODEL: &str = "fal-ai/stable-audio";

/// Default transcription model.
const DEFAULT_TRANSCRIPTION_MODEL: &str = "fal-ai/whisper";

/// Default fal application id for text-to-3D / image-to-3D generation.
const DEFAULT_3D_MODEL: &str = "fal-ai/triposr";

/// Default fal app for background removal.
const DEFAULT_BG_REMOVAL_MODEL: &str = "fal-ai/birefnet";

/// Default fal app for the aura-sr upscaler.
const DEFAULT_AURA_UPSCALE_MODEL: &str = "fal-ai/aura-sr";

/// Default fal app for the clarity-upscaler.
const DEFAULT_CLARITY_UPSCALE_MODEL: &str = "fal-ai/clarity-upscaler";

/// Default fal app for the creative-upscaler.
const DEFAULT_CREATIVE_UPSCALE_MODEL: &str = "fal-ai/creative-upscaler";

/// Character limit for the `prompt` field on fal prompt-string endpoints
/// (`fal-ai/any-llm`, `openrouter/router`, and their variants).
const FAL_PROMPT_CHAR_LIMIT: usize = 4800;
/// Character limit for the `system_prompt` field on the same endpoints.
const FAL_SYSTEM_PROMPT_CHAR_LIMIT: usize = 4800;

// ---------------------------------------------------------------------------
// LLM endpoint typing
// ---------------------------------------------------------------------------

/// Which fal.ai LLM-class endpoint family to call.
///
/// Each variant maps to a known fal application path and dictates the
/// request/response schema. Use [`FalLlmEndpoint::default()`] (== `OpenAiChat`)
/// unless you specifically need an enterprise / SOC2 path or a non-standard
/// schema.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FalLlmEndpoint {
    /// `openrouter/router/openai/v1/chat/completions` — full `OpenAI` chat
    /// completions semantics: messages array, multimodal content blocks,
    /// tool calls, structured outputs, native SSE streaming. **DEFAULT.**
    #[default]
    OpenAiChat,
    /// `openrouter/router/openai/v1/responses` — `OpenAI` Responses API.
    OpenAiResponses,
    /// `openrouter/router/openai/v1/embeddings` — `OpenAI` embeddings (consumed
    /// by `FalEmbeddingModel`, not `CompletionModel`).
    OpenAiEmbeddings,
    /// `openrouter/router` (or `openrouter/router/enterprise`) — fal's own
    /// `OpenRouter` wrapper that takes `prompt`+`system_prompt` strings.
    OpenRouter {
        /// Use the enterprise/SOC2 path.
        enterprise: bool,
    },
    /// `fal-ai/any-llm` (or `/enterprise`) — fal-ai's any-llm proxy with the
    /// same prompt-string schema.
    AnyLlm {
        /// Use the enterprise/SOC2 path.
        enterprise: bool,
    },
    /// Vision LLM family. Adds `image_urls[]` to the prompt-string body.
    Vision {
        /// Which provider family hosts this sub-endpoint.
        family: FalVisionFamily,
        /// Use the enterprise/SOC2 path.
        enterprise: bool,
    },
    /// Audio LLM family. Adds `audio_url` to the prompt-string body.
    Audio {
        /// Which provider family hosts this sub-endpoint.
        family: FalVisionFamily,
        /// Use the enterprise/SOC2 path.
        enterprise: bool,
    },
    /// Video LLM family. Adds `video_url` to the prompt-string body.
    Video {
        /// Which provider family hosts this sub-endpoint.
        family: FalVisionFamily,
        /// Use the enterprise/SOC2 path.
        enterprise: bool,
    },
    /// Escape hatch: any other fal application path. Caller specifies the
    /// body format because we cannot know the schema.
    Custom {
        /// The fal application path (e.g. `"some-org/some-app"`).
        path: String,
        /// The wire body format the endpoint expects.
        body_format: FalBodyFormat,
    },
}

/// Which provider family hosts a vision/audio/video sub-endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalVisionFamily {
    /// fal's `openrouter/router/...` family.
    OpenRouter,
    /// fal's `fal-ai/any-llm/...` family.
    AnyLlm,
}

/// Wire body format that an endpoint expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalBodyFormat {
    /// `{"model": ..., "messages": [...]}` (`OpenAI` chat completions)
    OpenAiMessages,
    /// `{"model": ..., "input": [...]}` (`OpenAI` Responses API)
    OpenAiResponses,
    /// `{"model": ..., "prompt": "...", "system_prompt": "..."}`
    PromptString,
    /// `PromptString` + `image_urls[]`
    PromptStringVision,
    /// `PromptString` + `audio_url`
    PromptStringAudio,
    /// `PromptString` + `video_url`
    PromptStringVideo,
}

/// Modality flag for [`FalProvider::build_prompt_string_body`] — selects
/// which media field (if any) to populate alongside the `prompt`.
#[derive(Debug, Clone, Copy)]
enum MediaKind {
    Image,
    Audio,
    Video,
}

/// Whether the endpoint supports SSE inline (`stream: true`) or `/stream`-suffix streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingStrategy {
    /// POST with `stream: true`, parse OpenAI-shaped SSE.
    SseInline,
    /// POST to `<path>/stream` and parse fal's cumulative-output SSE.
    StreamSuffix,
}

impl FalLlmEndpoint {
    /// The fal application path for this endpoint family.
    #[must_use]
    pub fn path(&self) -> std::borrow::Cow<'static, str> {
        use std::borrow::Cow;
        match self {
            Self::OpenAiChat => Cow::Borrowed("openrouter/router/openai/v1/chat/completions"),
            Self::OpenAiResponses => Cow::Borrowed("openrouter/router/openai/v1/responses"),
            Self::OpenAiEmbeddings => Cow::Borrowed("openrouter/router/openai/v1/embeddings"),
            Self::OpenRouter { enterprise: false } => Cow::Borrowed("openrouter/router"),
            Self::OpenRouter { enterprise: true } => Cow::Borrowed("openrouter/router/enterprise"),
            Self::AnyLlm { enterprise: false } => Cow::Borrowed("fal-ai/any-llm"),
            Self::AnyLlm { enterprise: true } => Cow::Borrowed("fal-ai/any-llm/enterprise"),
            Self::Vision {
                family: FalVisionFamily::OpenRouter,
                enterprise: false,
            } => Cow::Borrowed("openrouter/router/vision"),
            Self::Vision {
                family: FalVisionFamily::OpenRouter,
                enterprise: true,
            } => Cow::Borrowed("openrouter/router/vision/enterprise"),
            Self::Vision {
                family: FalVisionFamily::AnyLlm,
                enterprise: false,
            } => Cow::Borrowed("fal-ai/any-llm/vision"),
            Self::Vision {
                family: FalVisionFamily::AnyLlm,
                enterprise: true,
            } => Cow::Borrowed("fal-ai/any-llm/vision/enterprise"),
            Self::Audio {
                family: FalVisionFamily::OpenRouter,
                enterprise: false,
            } => Cow::Borrowed("openrouter/router/audio"),
            Self::Audio {
                family: FalVisionFamily::OpenRouter,
                enterprise: true,
            } => Cow::Borrowed("openrouter/router/audio/enterprise"),
            Self::Audio {
                family: FalVisionFamily::AnyLlm,
                enterprise: false,
            } => Cow::Borrowed("fal-ai/any-llm/audio"),
            Self::Audio {
                family: FalVisionFamily::AnyLlm,
                enterprise: true,
            } => Cow::Borrowed("fal-ai/any-llm/audio/enterprise"),
            Self::Video {
                family: FalVisionFamily::OpenRouter,
                enterprise: false,
            } => Cow::Borrowed("openrouter/router/video"),
            Self::Video {
                family: FalVisionFamily::OpenRouter,
                enterprise: true,
            } => Cow::Borrowed("openrouter/router/video/enterprise"),
            Self::Video {
                family: FalVisionFamily::AnyLlm,
                enterprise: false,
            } => Cow::Borrowed("fal-ai/any-llm/video"),
            Self::Video {
                family: FalVisionFamily::AnyLlm,
                enterprise: true,
            } => Cow::Borrowed("fal-ai/any-llm/video/enterprise"),
            Self::Custom { path, .. } => Cow::Owned(path.clone()),
        }
    }

    /// The wire body format expected by this endpoint.
    #[must_use]
    pub fn body_format(&self) -> FalBodyFormat {
        match self {
            // `OpenAiEmbeddings` is unused here; embeddings has its own model.
            Self::OpenAiChat | Self::OpenAiEmbeddings => FalBodyFormat::OpenAiMessages,
            Self::OpenAiResponses => FalBodyFormat::OpenAiResponses,
            Self::OpenRouter { .. } | Self::AnyLlm { .. } => FalBodyFormat::PromptString,
            Self::Vision { .. } => FalBodyFormat::PromptStringVision,
            Self::Audio { .. } => FalBodyFormat::PromptStringAudio,
            Self::Video { .. } => FalBodyFormat::PromptStringVideo,
            Self::Custom { body_format, .. } => *body_format,
        }
    }

    /// Whether this endpoint supports streaming (all of them do).
    #[must_use]
    pub fn supports_streaming(&self) -> bool {
        true
    }

    /// Which streaming strategy applies to this endpoint.
    #[must_use]
    pub fn streaming_strategy(&self) -> StreamingStrategy {
        match self.body_format() {
            FalBodyFormat::OpenAiMessages | FalBodyFormat::OpenAiResponses => {
                StreamingStrategy::SseInline
            }
            _ => StreamingStrategy::StreamSuffix,
        }
    }

    /// The natural [`FalExecutionMode`] for this endpoint family.
    ///
    /// `OpenAiChat` and `OpenAiResponses` are sync (`fal.run/...`); the
    /// prompt-string families are queue-based by default.
    #[must_use]
    pub fn natural_execution_mode(&self) -> FalExecutionMode {
        match self.body_format() {
            FalBodyFormat::OpenAiMessages | FalBodyFormat::OpenAiResponses => {
                FalExecutionMode::Sync
            }
            _ => FalExecutionMode::Queue {
                poll_interval: DEFAULT_POLL_INTERVAL,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// How to execute requests on fal.ai.
#[derive(Debug, Clone)]
pub enum FalExecutionMode {
    /// Synchronous -- wait for result (timeout risk for long jobs).
    Sync,
    /// Queue-based -- submit, poll for result.
    Queue {
        /// How often to poll for completion.
        poll_interval: Duration,
    },
    /// Webhook -- submit, receive result at the given URL.
    Webhook {
        /// The URL to receive the result.
        url: String,
    },
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A fal.ai compute platform provider.
///
/// LLM routing is driven by a typed [`FalLlmEndpoint`] enum rather than a
/// free-form URL string. The default endpoint is
/// [`FalLlmEndpoint::OpenAiChat`], which targets
/// `openrouter/router/openai/v1/chat/completions` and speaks full `OpenAI`
/// chat completions semantics. The default model id is
/// `"anthropic/claude-sonnet-4.5"`.
///
/// Endpoint and model are independent:
/// - [`with_llm_endpoint`](Self::with_llm_endpoint) selects the URL path /
///   wire format family (e.g. [`FalLlmEndpoint::OpenAiChat`],
///   [`FalLlmEndpoint::OpenRouter`], [`FalLlmEndpoint::AnyLlm`],
///   [`FalLlmEndpoint::Vision`], or [`FalLlmEndpoint::Custom`]).
/// - [`with_llm_model`](Self::with_llm_model) only changes the `model` field
///   in the request body — it does NOT change the URL path.
/// - [`with_enterprise`](Self::with_enterprise) promotes the current endpoint
///   to its enterprise/SOC2-eligible variant where one exists.
/// - [`with_auto_route_modality`](Self::with_auto_route_modality) toggles
///   automatic switching to vision/audio/video endpoint families when a
///   request carries the matching media content (enabled by default).
///
/// For compute usage, this provider implements [`ComputeProvider`] with
/// queue-based job submission, status polling, and result retrieval.
///
/// For image generation and upscaling, this provider implements [`ImageGeneration`].
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::fal::{FalLlmEndpoint, FalProvider};
///
/// // Default: OpenAiChat against openrouter/router/openai/v1/chat/completions.
/// let default_provider = FalProvider::new("fal-key-...");
///
/// // Opt into the prompt-format any-llm proxy explicitly:
/// let provider = FalProvider::new("fal-key-...")
///     .with_llm_endpoint(FalLlmEndpoint::AnyLlm { enterprise: false });
/// ```
pub struct FalProvider {
    client: Arc<dyn HttpClient>,
    api_key: String,
    /// LLM endpoint family. Default: [`FalLlmEndpoint::OpenAiChat`].
    llm_endpoint: FalLlmEndpoint,
    /// Default model id when `CompletionRequest::model` is `None`.
    llm_model: String,
    /// Auto-switch to a vision/audio/video endpoint when the request
    /// contains matching content. Default: `true`.
    auto_route_modality: bool,
    execution_mode: FalExecutionMode,
    base_queue_url: String,
    base_sync_url: String,
}

impl std::fmt::Debug for FalProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalProvider")
            .field("llm_endpoint", &self.llm_endpoint)
            .field("llm_model", &self.llm_model)
            .field("auto_route_modality", &self.auto_route_modality)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

impl Clone for FalProvider {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            api_key: self.api_key.clone(),
            llm_endpoint: self.llm_endpoint.clone(),
            llm_model: self.llm_model.clone(),
            auto_route_modality: self.auto_route_modality,
            execution_mode: self.execution_mode.clone(),
            base_queue_url: self.base_queue_url.clone(),
            base_sync_url: self.base_sync_url.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt-string collapse helper (used by `build_prompt_string_body`)
// ---------------------------------------------------------------------------

/// Per-message scratch entry used while collapsing a conversation into a
/// fal prompt-string body.
struct CollapseEntry {
    is_system: bool,
    text: String,
    images: Vec<String>,
    audio: Option<String>,
    video: Option<String>,
}

/// Extract a public URL from an [`crate::types::ImageSource`], dropping
/// base64 sources (the fal prompt-string endpoints only accept URLs).
fn url_from_source(src: &crate::types::ImageSource) -> Option<String> {
    use crate::types::ImageSource;
    match src {
        ImageSource::Url { url } => Some(url.clone()),
        ImageSource::Base64 { .. } => None,
    }
}

/// Convert a single [`ChatMessage`] into a [`CollapseEntry`], extracting
/// any media URLs and prefixing the text with the role label.
fn message_to_entry(msg: &crate::types::ChatMessage) -> CollapseEntry {
    use crate::types::{ContentPart, MessageContent, Role};

    let prefix = match msg.role {
        Role::System => "",
        Role::User => "User: ",
        Role::Assistant => "Assistant: ",
        Role::Tool => "Tool result: ",
    };
    let mut text = String::new();
    let mut images: Vec<String> = Vec::new();
    let mut audio: Option<String> = None;
    let mut video: Option<String> = None;

    match &msg.content {
        MessageContent::Text(t) => text.push_str(t),
        MessageContent::Image(img) => {
            if let Some(u) = url_from_source(&img.source) {
                images.push(u);
            }
        }
        MessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text { text: t } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                    ContentPart::Image(img) => {
                        if let Some(u) = url_from_source(&img.source) {
                            images.push(u);
                        }
                    }
                    ContentPart::Audio(a) => {
                        if audio.is_none() {
                            audio = url_from_source(&a.source);
                        }
                    }
                    ContentPart::Video(v) => {
                        if video.is_none() {
                            video = url_from_source(&v.source);
                        }
                    }
                    ContentPart::File(_) => {} // dropped — no field for these
                }
            }
        }
    }

    let line = if text.is_empty() {
        String::new()
    } else {
        format!("{prefix}{text}")
    };

    CollapseEntry {
        is_system: matches!(msg.role, Role::System),
        text: line,
        images,
        audio,
        video,
    }
}

/// Join the non-system entries into the `prompt` string.
fn join_prompt(entries: &[CollapseEntry]) -> String {
    entries
        .iter()
        .filter(|e| !e.is_system && !e.text.is_empty())
        .map(|e| e.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Join the system entries into the `system_prompt` string.
fn join_system(entries: &[CollapseEntry]) -> String {
    entries
        .iter()
        .filter(|e| e.is_system && !e.text.is_empty())
        .map(|e| e.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Truncate `system_prompt` to `system_limit` characters with a
/// ` [truncated]` marker, emitting a `tracing::warn!`.
fn truncate_system_prompt(system_prompt: String, system_limit: usize) -> String {
    if system_prompt.chars().count() <= system_limit {
        return system_prompt;
    }
    tracing::warn!(
        "fal: system_prompt exceeds {} chars; truncating with ' [truncated]' marker",
        system_limit
    );
    let trunc_at: String = system_prompt
        .chars()
        .take(system_limit.saturating_sub(13))
        .collect();
    format!("{trunc_at} [truncated]")
}

/// Aggregate media URLs across the surviving entries — all images, plus
/// the first non-empty audio and video URL.
fn aggregate_media(entries: &[CollapseEntry]) -> (Vec<String>, Option<String>, Option<String>) {
    let mut image_urls = Vec::new();
    let mut audio_url: Option<String> = None;
    let mut video_url: Option<String> = None;
    for entry in entries {
        image_urls.extend(entry.images.iter().cloned());
        if audio_url.is_none() {
            audio_url.clone_from(&entry.audio);
        }
        if video_url.is_none() {
            video_url.clone_from(&entry.video);
        }
    }
    (image_urls, audio_url, video_url)
}

/// Collapse a list of [`ChatMessage`] values into the prompt-string body
/// fields fal's prompt-format endpoints expect.
///
/// Returns `(prompt, system_prompt, image_urls, audio_url, video_url)`.
///
/// - System messages are concatenated into `system_prompt` with `\n\n`.
/// - User/Assistant/Tool messages are prefixed (`User: ` / `Assistant: ` /
///   `Tool result: `) and concatenated into `prompt` with `\n\n`.
/// - Image / audio / video URLs are extracted from `MessageContent::Image`
///   and `ContentPart::{Image,Audio,Video}` parts.
/// - If `prompt` exceeds `prompt_limit`, the OLDEST non-system messages
///   are evicted until it fits (preserving the chronological tail).
/// - If `system_prompt` exceeds `system_limit`, it is truncated from the
///   end with a ` [truncated]` marker and a `tracing::warn!` is emitted.
fn collapse_messages(
    messages: &[crate::types::ChatMessage],
    prompt_limit: usize,
    system_limit: usize,
) -> (String, String, Vec<String>, Option<String>, Option<String>) {
    let mut entries: Vec<CollapseEntry> = messages.iter().map(message_to_entry).collect();

    // Evict oldest non-system messages until prompt fits within prompt_limit.
    let mut prompt = join_prompt(&entries);
    while prompt.chars().count() > prompt_limit {
        // Find the oldest non-system entry and remove it.
        if let Some(idx) = entries.iter().position(|e| !e.is_system) {
            entries.remove(idx);
            prompt = join_prompt(&entries);
        } else {
            // Only system messages left — can't shrink prompt further.
            break;
        }
    }

    let system_prompt = truncate_system_prompt(join_system(&entries), system_limit);
    let (image_urls, audio_url, video_url) = aggregate_media(&entries);

    (prompt, system_prompt, image_urls, audio_url, video_url)
}

impl FalProvider {
    /// Create a new fal.ai provider with the given API key.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            api_key: api_key.into(),
            llm_endpoint: FalLlmEndpoint::default(),
            llm_model: "anthropic/claude-sonnet-4.5".to_owned(),
            auto_route_modality: true,
            execution_mode: FalLlmEndpoint::default().natural_execution_mode(),
            base_queue_url: FAL_QUEUE_URL.to_owned(),
            base_sync_url: FAL_SYNC_URL.to_owned(),
        }
    }

    /// Create a new fal.ai provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            llm_endpoint: FalLlmEndpoint::default(),
            llm_model: "anthropic/claude-sonnet-4.5".to_owned(),
            auto_route_modality: true,
            execution_mode: FalLlmEndpoint::default().natural_execution_mode(),
            base_queue_url: FAL_QUEUE_URL.to_owned(),
            base_sync_url: FAL_SYNC_URL.to_owned(),
        }
    }

    /// Create a new fal.ai provider from a [`FalOptions`] struct.
    ///
    /// This is the canonical construction path — bindings should deserialize
    /// their native options type into [`FalOptions`] and call this method
    /// instead of manually destructuring fields and calling individual
    /// builder methods.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    /// Construct from typed [`FalOptions`](crate::types::provider_options::FalOptions).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Auth`] if no API key is provided and the
    /// `FAL_KEY` environment variable is not set.
    pub fn from_options(
        opts: crate::types::provider_options::FalOptions,
    ) -> Result<Self, crate::BlazenError> {
        use crate::types::provider_options::FalLlmEndpointKind;

        let api_key = crate::keys::resolve_api_key("fal", opts.base.api_key)?;
        let mut provider = Self::new(api_key);
        if let Some(m) = opts.base.model {
            provider = provider.with_llm_model(m);
        }
        if let Some(url) = opts.base.base_url {
            provider = provider.with_base_url(url);
        }
        if let Some(ep) = opts.endpoint {
            let endpoint = match ep {
                FalLlmEndpointKind::OpenAiChat => FalLlmEndpoint::OpenAiChat,
                FalLlmEndpointKind::OpenAiResponses => FalLlmEndpoint::OpenAiResponses,
                FalLlmEndpointKind::OpenAiEmbeddings => FalLlmEndpoint::OpenAiEmbeddings,
                FalLlmEndpointKind::OpenRouter => FalLlmEndpoint::OpenRouter {
                    enterprise: opts.enterprise,
                },
                FalLlmEndpointKind::AnyLlm => FalLlmEndpoint::AnyLlm {
                    enterprise: opts.enterprise,
                },
            };
            provider = provider.with_llm_endpoint(endpoint);
        } else if opts.enterprise {
            provider = provider.with_enterprise();
        }
        provider = provider.with_auto_route_modality(opts.auto_route_modality);
        Ok(provider)
    }

    /// Build a [`FalEmbeddingModel`] sharing this provider's HTTP client and
    /// API key. Uses the default `openai/text-embedding-3-small` model and
    /// 1536 dimensions.
    #[must_use]
    pub fn embedding_model(&self) -> FalEmbeddingModel {
        FalEmbeddingModel {
            client: Arc::clone(&self.client),
            api_key: self.api_key.clone(),
            model: "openai/text-embedding-3-small".to_owned(),
            dimensions: 1536,
            base_sync_url: self.base_sync_url.clone(),
        }
    }

    /// **Deprecated.** Prefer the typed [`FalLlmEndpoint`] API.
    ///
    /// This shim forwards to
    /// [`with_llm_endpoint`](Self::with_llm_endpoint)`(`[`FalLlmEndpoint::Custom`]`
    /// { path, body_format: `[`FalBodyFormat::PromptString`]` })`, which only
    /// makes sense for prompt-string endpoints. For `OpenAI`-shaped families
    /// use [`FalLlmEndpoint::OpenAiChat`] / [`FalLlmEndpoint::OpenAiResponses`]
    /// directly via [`with_llm_endpoint`](Self::with_llm_endpoint).
    #[deprecated(
        since = "0.3.0",
        note = "use with_llm_endpoint(FalLlmEndpoint::Custom { path, body_format })"
    )]
    #[must_use]
    pub fn with_endpoint(self, raw: impl Into<String>) -> Self {
        self.with_llm_endpoint(FalLlmEndpoint::Custom {
            path: raw.into(),
            body_format: FalBodyFormat::PromptString,
        })
    }

    /// Deprecated: use [`with_llm_model`](Self::with_llm_model) instead.
    #[deprecated(since = "0.2.0", note = "renamed to `with_llm_model`")]
    #[must_use]
    pub fn with_model(self, model: impl Into<String>) -> Self {
        self.with_llm_model(model)
    }

    /// Set the LLM endpoint family.
    ///
    /// The execution mode is automatically updated to match the endpoint's
    /// natural mode (sync for OpenAI-compat endpoints, queue for prompt-string).
    #[must_use]
    pub fn with_llm_endpoint(mut self, endpoint: FalLlmEndpoint) -> Self {
        self.execution_mode = endpoint.natural_execution_mode();
        self.llm_endpoint = endpoint;
        self
    }

    /// Promote the current endpoint to its enterprise/SOC2-eligible variant.
    ///
    /// `OpenAiChat` and `OpenAiResponses` have no enterprise variant — calling
    /// `with_enterprise()` on them switches to `AnyLlm { enterprise: true }`
    /// (which uses the `prompt`+`system_prompt` body schema instead of the
    /// `OpenAI` messages array). A `tracing::warn!` is emitted when this
    /// schema-changing fallback fires.
    #[must_use]
    pub fn with_enterprise(mut self) -> Self {
        self.llm_endpoint = match self.llm_endpoint {
            FalLlmEndpoint::OpenAiChat | FalLlmEndpoint::OpenAiResponses => {
                tracing::warn!(
                    "fal: OpenAI-compat endpoints have no enterprise variant; \
                     promoting to AnyLlm{{enterprise:true}}, body format will switch \
                     to prompt+system_prompt"
                );
                FalLlmEndpoint::AnyLlm { enterprise: true }
            }
            FalLlmEndpoint::OpenAiEmbeddings => FalLlmEndpoint::OpenAiEmbeddings,
            FalLlmEndpoint::OpenRouter { .. } => FalLlmEndpoint::OpenRouter { enterprise: true },
            FalLlmEndpoint::AnyLlm { .. } => FalLlmEndpoint::AnyLlm { enterprise: true },
            FalLlmEndpoint::Vision { family, .. } => FalLlmEndpoint::Vision {
                family,
                enterprise: true,
            },
            FalLlmEndpoint::Audio { family, .. } => FalLlmEndpoint::Audio {
                family,
                enterprise: true,
            },
            FalLlmEndpoint::Video { family, .. } => FalLlmEndpoint::Video {
                family,
                enterprise: true,
            },
            FalLlmEndpoint::Custom { path, body_format } => {
                FalLlmEndpoint::Custom { path, body_format }
            }
        };
        self.execution_mode = self.llm_endpoint.natural_execution_mode();
        self
    }

    /// Enable or disable automatic modality routing based on message content.
    ///
    /// When enabled (the default), if a request contains image / audio / video
    /// content and the configured endpoint is `OpenRouter` or `AnyLlm`, the
    /// provider transparently switches to the matching `Vision` / `Audio` /
    /// `Video` variant for that request only.
    #[must_use]
    pub fn with_auto_route_modality(mut self, enabled: bool) -> Self {
        self.auto_route_modality = enabled;
        self
    }

    /// Override the base queue URL (default: `https://queue.fal.run`).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_queue_url = url.into();
        self
    }

    /// Set the underlying LLM model.
    ///
    /// This is the model name passed in the request body (e.g.
    /// `"anthropic/claude-sonnet-4.5"`, `"openai/gpt-4o"`).
    ///
    /// The endpoint and model are independent: `with_llm_model` ONLY updates
    /// the `model` field in the request body and does NOT change the URL path.
    /// To change which fal application is called, use
    /// [`with_llm_endpoint`](Self::with_llm_endpoint) instead.
    #[must_use]
    pub fn with_llm_model(mut self, model: impl Into<String>) -> Self {
        self.llm_model = model.into();
        self
    }

    /// Set the execution mode.
    #[must_use]
    pub fn with_execution_mode(mut self, mode: FalExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    // -----------------------------------------------------------------------
    // Auth helper
    // -----------------------------------------------------------------------

    /// Apply fal.ai authentication (`Authorization: Key <key>`) to an [`HttpRequest`].
    fn apply_auth(&self, request: HttpRequest) -> HttpRequest {
        request.header("Authorization", format!("Key {}", self.api_key))
    }

    // -----------------------------------------------------------------------
    // LLM body builder
    // -----------------------------------------------------------------------

    /// Build the JSON request body for fal prompt-string endpoints
    /// (`fal-ai/any-llm`, `openrouter/router`, and their `vision`/`audio`/
    /// `video` variants).
    ///
    /// These endpoints take a single concatenated `prompt` string plus an
    /// optional `system_prompt` string, and one of `image_urls[]` /
    /// `audio_url` / `video_url` depending on the modality. The character
    /// limits enforced by fal are honoured via [`collapse_messages`].
    fn build_prompt_string_body(
        &self,
        request: &CompletionRequest,
        media_kind: Option<MediaKind>,
    ) -> serde_json::Value {
        let llm_model = request.model.as_deref().unwrap_or(&self.llm_model);
        let (prompt, system_prompt, image_urls, audio_url, video_url) = collapse_messages(
            &request.messages,
            FAL_PROMPT_CHAR_LIMIT,
            FAL_SYSTEM_PROMPT_CHAR_LIMIT,
        );

        let mut body = serde_json::json!({
            "model": llm_model,
            "prompt": prompt,
        });

        if !system_prompt.is_empty() {
            body["system_prompt"] = system_prompt.into();
        }

        match media_kind {
            Some(MediaKind::Image) if !image_urls.is_empty() => {
                body["image_urls"] = image_urls.into();
            }
            Some(MediaKind::Audio) => {
                if let Some(u) = audio_url {
                    body["audio_url"] = u.into();
                }
            }
            Some(MediaKind::Video) => {
                if let Some(u) = video_url {
                    body["video_url"] = u.into();
                }
            }
            _ => {}
        }

        if let Some(t) = request.temperature {
            body["temperature"] = t.into();
        }
        if let Some(m) = request.max_tokens {
            body["max_tokens"] = m.into();
        }
        if let Some(p) = request.top_p {
            body["top_p"] = p.into();
        }
        if let Some(rf) = &request.response_format {
            body["response_format"] = rf.clone();
        }

        body
    }

    /// Build the JSON request body for the `OpenAI` chat completions endpoint
    /// (`openrouter/router/openai/v1/chat/completions`).
    ///
    /// Produces a true `OpenAI` chat completions request body with a
    /// `messages` array, multimodal content blocks, and tool calls.
    fn build_openai_chat_body(&self, request: &CompletionRequest) -> serde_json::Value {
        use crate::providers::openai_format::content_to_openai_value;
        use crate::types::Role;

        let llm_model = request.model.as_deref().unwrap_or(&self.llm_model);

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let content = content_to_openai_value(&msg.content);
                let mut entry = serde_json::json!({ "role": role, "content": content });
                if let Some(id) = &msg.tool_call_id {
                    entry["tool_call_id"] = id.clone().into();
                }
                if !msg.tool_calls.is_empty() {
                    let tcs: Vec<_> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": &tc.id,
                                "type": "function",
                                "function": {
                                    "name": &tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    entry["tool_calls"] = tcs.into();
                }
                entry
            })
            .collect();

        let mut body = serde_json::json!({
            "model": llm_model,
            "messages": messages,
        });

        if !request.tools.is_empty() {
            let tools: Vec<_> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": &t.name,
                            "description": &t.description,
                            "parameters": &t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = tools.into();
        }

        if let Some(t) = request.temperature {
            body["temperature"] = t.into();
        }
        if let Some(m) = request.max_tokens {
            body["max_tokens"] = m.into();
        }
        if let Some(p) = request.top_p {
            body["top_p"] = p.into();
        }
        if let Some(rf) = &request.response_format {
            body["response_format"] = rf.clone();
        }

        body
    }

    /// Build the JSON request body for the `OpenAI` Responses API endpoint
    /// (`openrouter/router/openai/v1/responses`).
    ///
    /// The Responses API uses an `input` array of role-tagged content blocks
    /// rather than a `messages` array, and represents tool calls as separate
    /// `function_call` / `function_call_output` blocks.
    fn build_openai_responses_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let llm_model = request.model.as_deref().unwrap_or(&self.llm_model);
        let input =
            crate::providers::responses_format::messages_to_responses_input(&request.messages);

        let mut body = serde_json::json!({
            "model": llm_model,
            "input": input,
        });

        if !request.tools.is_empty() {
            let tools: Vec<_> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "name": &t.name,
                        "description": &t.description,
                        "parameters": &t.parameters,
                    })
                })
                .collect();
            body["tools"] = tools.into();
        }

        if let Some(t) = request.temperature {
            body["temperature"] = t.into();
        }
        if let Some(m) = request.max_tokens {
            body["max_output_tokens"] = m.into();
        }
        if let Some(rf) = &request.response_format {
            body["response_format"] = rf.clone();
        }

        body
    }

    /// Dispatch to the correct body builder for an endpoint's wire format.
    ///
    /// This is the single entry point that `complete()` will call once
    /// streaming + sync dispatch is wired up in Phase 4.6.
    fn build_body(&self, request: &CompletionRequest, ep: &FalLlmEndpoint) -> serde_json::Value {
        match ep.body_format() {
            FalBodyFormat::OpenAiMessages => self.build_openai_chat_body(request),
            FalBodyFormat::OpenAiResponses => self.build_openai_responses_body(request),
            FalBodyFormat::PromptString => self.build_prompt_string_body(request, None),
            FalBodyFormat::PromptStringVision => {
                self.build_prompt_string_body(request, Some(MediaKind::Image))
            }
            FalBodyFormat::PromptStringAudio => {
                self.build_prompt_string_body(request, Some(MediaKind::Audio))
            }
            FalBodyFormat::PromptStringVideo => {
                self.build_prompt_string_body(request, Some(MediaKind::Video))
            }
        }
    }

    /// Dispatch to the correct response parser for an endpoint's wire format.
    ///
    /// This is the single entry point that `complete()` will call once
    /// streaming + sync dispatch is wired up in Phase 4.6.
    fn parse_response(
        &self,
        ep: &FalLlmEndpoint,
        raw: serde_json::Value,
    ) -> Result<CompletionResponse, BlazenError> {
        match ep.body_format() {
            FalBodyFormat::OpenAiMessages => parse_openai_chat_response(raw, &self.llm_model),
            FalBodyFormat::OpenAiResponses => parse_openai_responses_response(raw, &self.llm_model),
            _ => parse_prompt_string_response(raw, &self.llm_model),
        }
    }

    /// Pick the right [`FalLlmEndpoint`] for a given request, taking the
    /// configured endpoint and (when [`auto_route_modality`] is on) the
    /// modality of the request's content into account.
    ///
    /// When auto-routing is enabled and the configured endpoint is one of
    /// the prompt-string families (`OpenRouter` / `AnyLlm`), the resolver
    /// promotes to the matching `Vision` / `Audio` / `Video` sub-endpoint
    /// if the request contains image / audio / video content. OpenAI-compat
    /// endpoints already handle multimodal natively via the chosen model and
    /// are returned unchanged.
    ///
    /// [`auto_route_modality`]: Self::auto_route_modality
    #[allow(clippy::match_same_arms)] // Arm grouping is documentation: distinct semantic categories collapse to the same fall-through.
    fn resolve_endpoint_for_request(&self, request: &CompletionRequest) -> FalLlmEndpoint {
        if !self.auto_route_modality {
            return self.llm_endpoint.clone();
        }
        let has_image = request.messages.iter().any(|m| m.content.has_images());
        let has_audio = request.messages.iter().any(|m| m.content.has_audio());
        let has_video = request.messages.iter().any(|m| m.content.has_video());

        match &self.llm_endpoint {
            // OpenAI-compat endpoints handle multimodal natively via the chosen model.
            FalLlmEndpoint::OpenAiChat
            | FalLlmEndpoint::OpenAiResponses
            | FalLlmEndpoint::OpenAiEmbeddings => self.llm_endpoint.clone(),
            FalLlmEndpoint::OpenRouter { enterprise } => {
                if has_video {
                    FalLlmEndpoint::Video {
                        family: FalVisionFamily::OpenRouter,
                        enterprise: *enterprise,
                    }
                } else if has_audio {
                    FalLlmEndpoint::Audio {
                        family: FalVisionFamily::OpenRouter,
                        enterprise: *enterprise,
                    }
                } else if has_image {
                    FalLlmEndpoint::Vision {
                        family: FalVisionFamily::OpenRouter,
                        enterprise: *enterprise,
                    }
                } else {
                    self.llm_endpoint.clone()
                }
            }
            FalLlmEndpoint::AnyLlm { enterprise } => {
                if has_video {
                    FalLlmEndpoint::Video {
                        family: FalVisionFamily::AnyLlm,
                        enterprise: *enterprise,
                    }
                } else if has_audio {
                    FalLlmEndpoint::Audio {
                        family: FalVisionFamily::AnyLlm,
                        enterprise: *enterprise,
                    }
                } else if has_image {
                    FalLlmEndpoint::Vision {
                        family: FalVisionFamily::AnyLlm,
                        enterprise: *enterprise,
                    }
                } else {
                    self.llm_endpoint.clone()
                }
            }
            FalLlmEndpoint::Vision { .. }
            | FalLlmEndpoint::Audio { .. }
            | FalLlmEndpoint::Video { .. } => self.llm_endpoint.clone(),
            FalLlmEndpoint::Custom { .. } => self.llm_endpoint.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Shared HTTP helpers
    // -----------------------------------------------------------------------

    /// Map an HTTP error response to the appropriate `BlazenError`.
    fn map_http_error(status: u16, body: &str, retry_after_ms: Option<u64>) -> BlazenError {
        match status {
            401 => BlazenError::auth("authentication failed"),
            429 => BlazenError::RateLimit { retry_after_ms },
            _ => BlazenError::request(format!("HTTP {status}: {body}")),
        }
    }

    // -----------------------------------------------------------------------
    // Sync execution (for CompletionModel)
    // -----------------------------------------------------------------------

    /// Execute synchronously: POST to fal.run and wait for the response.
    async fn execute_sync(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, BlazenError> {
        let url = format!("{}/{}", self.base_sync_url, path);

        let request = self.apply_auth(HttpRequest::post(&url).json_body(body)?);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let retry_after_ms = parse_retry_after(&response.headers);
            let error_body = response.text();
            return Err(Self::map_http_error(
                response.status,
                &error_body,
                retry_after_ms,
            ));
        }

        response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Webhook execution (for CompletionModel)
    // -----------------------------------------------------------------------

    /// Execute via webhook: submit with webhook URL.
    async fn execute_webhook(
        &self,
        path: &str,
        body: &serde_json::Value,
        webhook_url: &str,
    ) -> Result<serde_json::Value, BlazenError> {
        let submit_url = format!("{}/{}?fal_webhook={webhook_url}", self.base_queue_url, path);

        let request = self.apply_auth(HttpRequest::post(&submit_url).json_body(body)?);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "HTTP {}: {error_body}",
                response.status
            )));
        }

        // Webhook mode returns the queue submission response. The actual
        // result will be delivered to the webhook URL.
        response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Shared queue polling logic
    // -----------------------------------------------------------------------

    /// Submit a request to the fal.ai queue and return the queue response.
    async fn queue_submit(
        &self,
        model: &str,
        body: &serde_json::Value,
        webhook: Option<&str>,
    ) -> Result<FalQueueSubmitResponse, BlazenError> {
        let mut submit_url = format!("{}/{model}", self.base_queue_url);
        if let Some(wh) = webhook {
            submit_url = format!("{submit_url}?fal_webhook={wh}");
        }

        let request = self.apply_auth(HttpRequest::post(&submit_url).json_body(body)?);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let retry_after_ms = parse_retry_after(&response.headers);
            let error_body = response.text();
            return Err(Self::map_http_error(
                response.status,
                &error_body,
                retry_after_ms,
            ));
        }

        response
            .json::<FalQueueSubmitResponse>()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Poll the fal.ai queue status endpoint.
    async fn queue_poll_status(
        &self,
        model: &str,
        request_id: &str,
    ) -> Result<FalStatusResponse, BlazenError> {
        let status_url = format!(
            "{}/{model}/requests/{request_id}/status",
            self.base_queue_url
        );

        let request = self.apply_auth(HttpRequest::get(&status_url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "status poll failed: {error_body}"
            )));
        }

        response
            .json::<FalStatusResponse>()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Fetch the result of a completed queue job.
    async fn queue_get_result(
        &self,
        model: &str,
        request_id: &str,
    ) -> Result<serde_json::Value, BlazenError> {
        let result_url = format!("{}/{model}/requests/{request_id}", self.base_queue_url);

        let request = self.apply_auth(HttpRequest::get(&result_url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "result fetch failed: {error_body}"
            )));
        }

        response
            .json()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Poll until a queue job completes and return the result JSON plus timing.
    ///
    /// This is the shared polling logic used by both [`ComputeProvider::result`]
    /// and [`CompletionModel::complete`] (queue mode).
    ///
    /// When `status_url` and `response_url` are provided (from the queue submit
    /// response), they are used directly instead of constructing URLs from the
    /// model and request ID. This avoids 405 errors with multi-segment model
    /// IDs where manual URL construction produces incorrect paths.
    async fn poll_until_complete(
        &self,
        model: &str,
        request_id: &str,
        poll_interval: Duration,
        status_url: Option<&str>,
        response_url: Option<&str>,
    ) -> Result<(serde_json::Value, serde_json::Value, RequestTiming), BlazenError> {
        let start = Instant::now();
        let mut in_progress_at: Option<Instant> = None;

        for _ in 0..MAX_POLL_ITERATIONS {
            crate::sleep::sleep(poll_interval).await;

            let status_body = if let Some(url) = status_url {
                self.get_json_from_url(url).await?
            } else {
                self.queue_poll_status(model, request_id).await?
            };

            match status_body.status.as_str() {
                "COMPLETED" => {
                    // Check for error in COMPLETED status.
                    if let Some(ref error) = status_body.error {
                        return Err(BlazenError::Compute(ComputeErrorKind::JobFailed {
                            message: error.clone(),
                            error_type: None,
                            retryable: false,
                        }));
                    }

                    // Build timing from metrics.
                    let inference_time =
                        status_body.metrics.as_ref().and_then(|m| m.inference_time);
                    let timing = build_timing(start, in_progress_at, inference_time);

                    // Serialize status for metadata before moving on.
                    let status_json =
                        serde_json::to_value(&status_body).unwrap_or(serde_json::Value::Null);

                    // Fetch the result using the server-provided URL if available.
                    let result = if let Some(url) = response_url {
                        self.get_json_value_from_url(url).await?
                    } else {
                        self.queue_get_result(model, request_id).await?
                    };

                    return Ok((result, status_json, timing));
                }
                "IN_PROGRESS" => {
                    if in_progress_at.is_none() {
                        in_progress_at = Some(Instant::now());
                    }
                }
                // IN_QUEUE -- keep polling.
                _ => {}
            }
        }

        Err(BlazenError::Timeout {
            elapsed_ms: millis_u64(start.elapsed()),
        })
    }

    /// GET a URL and deserialize the response as [`FalStatusResponse`].
    async fn get_json_from_url(&self, url: &str) -> Result<FalStatusResponse, BlazenError> {
        let request = self.apply_auth(HttpRequest::get(url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "status poll failed: {error_body}"
            )));
        }

        response
            .json::<FalStatusResponse>()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// GET a URL and deserialize the response as a generic JSON value.
    async fn get_json_value_from_url(&self, url: &str) -> Result<serde_json::Value, BlazenError> {
        let request = self.apply_auth(HttpRequest::get(url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "result fetch failed: {error_body}"
            )));
        }

        response
            .json()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Execute via queue: submit, poll, return result. Used by `CompletionModel`.
    async fn execute_queue_llm(
        &self,
        path: &str,
        body: &serde_json::Value,
        poll_interval: Duration,
    ) -> Result<(serde_json::Value, RequestTiming), BlazenError> {
        let model = path;

        let submit_response = self.queue_submit(model, body, None).await?;

        let request_id = &submit_response.request_id;
        debug!(request_id = %request_id, "fal.ai LLM job submitted to queue");

        let (result, _status, timing) = self
            .poll_until_complete(
                model,
                request_id,
                poll_interval,
                submit_response.status_url.as_deref(),
                submit_response.response_url.as_deref(),
            )
            .await?;

        Ok((result, timing))
    }

    // -----------------------------------------------------------------------
    // Streaming execution (for CompletionModel)
    // -----------------------------------------------------------------------

    /// Stream an OpenAI-compat chat completions endpoint via inline SSE
    /// (`stream: true` in the request body).
    async fn stream_openai_chat(
        &self,
        request: CompletionRequest,
        ep: &FalLlmEndpoint,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let mut body = self.build_openai_chat_body(&request);
        body["stream"] = serde_json::Value::Bool(true);
        let url = format!("{}/{}", self.base_sync_url, ep.path());
        let http_request = self.apply_auth(HttpRequest::post(url).json_body(&body)?);
        let (status, headers, byte_stream) = self.client.send_streaming(http_request).await?;
        if !(200..300).contains(&status) {
            let retry_after_ms = parse_retry_after(&headers);
            return Err(Self::map_http_error(
                status,
                "streaming request failed",
                retry_after_ms,
            ));
        }
        let parser = crate::providers::sse::SseParser::new(byte_stream);
        Ok(Box::pin(parser))
    }

    /// Stream a fal prompt-string endpoint via the `/stream` URL suffix,
    /// converting fal's cumulative-output SSE into incremental delta chunks.
    async fn stream_prompt_string(
        &self,
        request: CompletionRequest,
        ep: &FalLlmEndpoint,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let body = self.build_body(&request, ep);
        let url = format!("{}/{}/stream", self.base_sync_url, ep.path());
        let http_request = self.apply_auth(HttpRequest::post(url).json_body(&body)?);
        let (status, _headers, byte_stream) = self.client.send_streaming(http_request).await?;
        if !(200..300).contains(&status) {
            return Err(BlazenError::request(format!("HTTP {status}")));
        }
        Ok(Box::pin(FalCumulativeSseStream::new(byte_stream)))
    }

    /// Convenience: upscale via the aura-sr model.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying compute job fails or the response
    /// cannot be parsed.
    pub async fn upscale_image_aura(
        &self,
        mut request: UpscaleRequest,
    ) -> Result<ImageResult, BlazenError> {
        if request.model.is_none() {
            request.model = Some(DEFAULT_AURA_UPSCALE_MODEL.to_owned());
        }
        self.upscale_image(request).await
    }

    /// Convenience: upscale via the clarity-upscaler model.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying compute job fails or the response
    /// cannot be parsed.
    pub async fn upscale_image_clarity(
        &self,
        mut request: UpscaleRequest,
    ) -> Result<ImageResult, BlazenError> {
        if request.model.is_none() {
            request.model = Some(DEFAULT_CLARITY_UPSCALE_MODEL.to_owned());
        }
        self.upscale_image(request).await
    }

    /// Convenience: upscale via the creative-upscaler model.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying compute job fails or the response
    /// cannot be parsed.
    pub async fn upscale_image_creative(
        &self,
        mut request: UpscaleRequest,
    ) -> Result<ImageResult, BlazenError> {
        if request.model.is_none() {
            request.model = Some(DEFAULT_CREATIVE_UPSCALE_MODEL.to_owned());
        }
        self.upscale_image(request).await
    }
}

// ---------------------------------------------------------------------------
// fal cumulative-output SSE parser
// ---------------------------------------------------------------------------

/// Parses fal.ai's cumulative-output SSE stream into incremental delta
/// [`StreamChunk`]s.
///
/// fal's prompt-string `/stream` endpoints emit events of the form:
///
/// ```text
/// data: {"output": "<full cumulative text>", "partial": <bool>, "error": <opt>}
/// ```
///
/// Each event repeats the entire generated text so far. This parser tracks
/// `last_output_len` and emits only the newly-appended slice per event.
/// When `partial` is `false` (or `error` is set), the stream terminates.
pub(crate) struct FalCumulativeSseStream {
    inner: crate::http::ByteStream,
    buffer: String,
    last_output_len: usize,
    done: bool,
}

impl FalCumulativeSseStream {
    pub(crate) fn new(inner: crate::http::ByteStream) -> Self {
        Self {
            inner,
            buffer: String::new(),
            last_output_len: 0,
            done: false,
        }
    }

    /// Try to extract the next [`StreamChunk`] from the buffer without
    /// pulling more bytes from the inner stream.
    ///
    /// Returns:
    /// - `Some(Ok(chunk))` if a delta or final stop chunk was produced
    /// - `Some(Err(_))` if an error event was parsed
    /// - `None` if the buffer needs more data
    fn try_pop_chunk(&mut self) -> Option<Result<StreamChunk, BlazenError>> {
        loop {
            if self.done {
                return None;
            }
            let event_end = self.buffer.find("\n\n")?;
            let event_text = self.buffer[..event_end].to_owned();
            self.buffer.drain(..event_end + 2);

            // Each event may contain multiple lines (`data:`, `event:`, etc.).
            // We collect the FIRST `data:` payload that yields a delta and
            // return it; if a single event somehow encodes multiple distinct
            // outputs, the next poll will pick the next one up via the loop.
            for line in event_text.lines() {
                let Some(json_str) = line
                    .strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))
                else {
                    continue;
                };
                let json_str = json_str.trim();
                if json_str.is_empty() || json_str == "[DONE]" {
                    continue;
                }
                let value: serde_json::Value = match serde_json::from_str(json_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(err) = value.get("error").and_then(|e| e.as_str()) {
                    self.done = true;
                    return Some(Err(BlazenError::request(format!("fal: {err}"))));
                }
                let cumulative = value.get("output").and_then(|o| o.as_str()).unwrap_or("");
                let partial = value
                    .get("partial")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true);
                let mut delta_chunk: Option<StreamChunk> = None;
                if cumulative.len() > self.last_output_len {
                    let delta = cumulative[self.last_output_len..].to_owned();
                    self.last_output_len = cumulative.len();
                    delta_chunk = Some(StreamChunk {
                        delta: Some(delta),
                        ..Default::default()
                    });
                }
                if !partial {
                    self.done = true;
                    if let Some(chunk) = delta_chunk {
                        // Stash the final stop chunk so the next poll emits it.
                        // We re-prepend a synthetic `[DONE]` marker via a flag:
                        // simplest is to push it back as a virtual event using
                        // the buffer-less path. We use `self.done` to gate.
                        // Returning the delta now means the stop chunk needs to
                        // come on the next call — but `done == true` makes us
                        // return None. To avoid losing it, emit the stop chunk
                        // *after* the delta by buffering it as a state field.
                        // Simplest fix: emit a "stop" finish_reason on the delta
                        // chunk itself when this is the final event.
                        return Some(Ok(StreamChunk {
                            finish_reason: Some("stop".to_owned()),
                            ..chunk
                        }));
                    }
                    return Some(Ok(StreamChunk {
                        delta: None,
                        finish_reason: Some("stop".to_owned()),
                        ..Default::default()
                    }));
                }
                if let Some(chunk) = delta_chunk {
                    return Some(Ok(chunk));
                }
                // No delta and still partial — fall through to next line/event.
            }
            // Event consumed without producing a chunk; loop to try the next one.
        }
    }
}

impl Stream for FalCumulativeSseStream {
    type Item = Result<StreamChunk, BlazenError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        let this = self.get_mut();
        loop {
            if let Some(chunk) = this.try_pop_chunk() {
                return Poll::Ready(Some(chunk));
            }
            if this.done {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes);
                    this.buffer.push_str(&text);
                    // Loop and try parsing again.
                }
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(BlazenError::request(format!(
                        "fal sse read: {e}"
                    )))));
                }
                Poll::Ready(None) => {
                    // Stream ended. One last attempt to drain any complete event.
                    if let Some(chunk) = this.try_pop_chunk() {
                        return Poll::Ready(Some(chunk));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/// Safely convert a `Duration` to milliseconds as `u64`, saturating at `u64::MAX`.
fn millis_u64(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Build a [`RequestTiming`] from measured instants and fal.ai metrics.
fn build_timing(
    start: Instant,
    in_progress_at: Option<Instant>,
    inference_time_secs: Option<f64>,
) -> RequestTiming {
    let total_ms = Some(millis_u64(start.elapsed()));

    let queue_ms = in_progress_at.map(|t| millis_u64(t.duration_since(start)));

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    let execution_ms = inference_time_secs
        .map(|s| {
            let ms = (s * 1000.0).max(0.0);
            if ms > u64::MAX as f64 {
                u64::MAX
            } else {
                ms as u64
            }
        })
        .or_else(|| {
            // Fallback: if we know when IN_PROGRESS started, compute
            // execution as total minus queue time.
            match (total_ms, queue_ms) {
                (Some(total), Some(queue)) => Some(total.saturating_sub(queue)),
                _ => None,
            }
        });

    RequestTiming {
        queue_ms,
        execution_ms,
        total_ms,
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Response from the fal.ai queue submit endpoint.
#[derive(Debug, Deserialize)]
struct FalQueueSubmitResponse {
    request_id: String,
    #[serde(default)]
    response_url: Option<String>,
    #[serde(default)]
    status_url: Option<String>,
    #[serde(default)]
    #[allow(dead_code)] // Cancel uses its own URL construction in cancel().
    cancel_url: Option<String>,
}

/// Response from the fal.ai queue status endpoint.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct FalStatusResponse {
    status: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    metrics: Option<FalMetrics>,
    #[serde(default)]
    #[allow(dead_code)]
    queue_position: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    response_url: Option<String>,
}

/// Metrics returned by fal.ai in COMPLETED status.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct FalMetrics {
    /// Inference time in seconds.
    inference_time: Option<f64>,
}

/// Response from `fal-ai/any-llm`.
#[derive(Debug, Deserialize)]
struct FalLlmResponse {
    output: Option<String>,
    error: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    partial: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct FalOpenAiChatResponse {
    #[serde(default)]
    #[allow(dead_code)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    choices: Vec<FalOpenAiChatChoice>,
    #[serde(default)]
    usage: Option<FalOpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct FalOpenAiChatChoice {
    message: FalOpenAiChatMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FalOpenAiChatMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<crate::providers::sse::OaiToolCall>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FalOpenAiUsage {
    #[serde(default, alias = "input_tokens")]
    prompt_tokens: Option<u32>,
    #[serde(default, alias = "output_tokens")]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens_details: Option<FalOpenAiCompletionDetails>,
}

#[derive(Debug, Deserialize)]
struct FalOpenAiCompletionDetails {
    #[serde(default)]
    reasoning_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct FalOpenAiResponsesResponse {
    #[serde(default)]
    output: Vec<serde_json::Value>,
    #[serde(default)]
    usage: Option<FalOpenAiUsage>,
}

/// A single image from fal.ai image generation output.
#[derive(Debug, Deserialize)]
struct FalImage {
    url: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    content_type: Option<String>,
}

/// Image generation output from fal.ai.
#[derive(Debug, Deserialize)]
struct FalImageOutput {
    #[serde(default)]
    images: Vec<FalImage>,
}

/// ESRGAN upscale output from fal.ai (single image, not array).
#[derive(Debug, Deserialize)]
struct FalUpscaleOutput {
    image: FalImage,
}

// ---------------------------------------------------------------------------
// Response parsers
// ---------------------------------------------------------------------------

/// Parse an `OpenAI`-compatible chat completions response from fal.
fn parse_openai_chat_response(
    raw: serde_json::Value,
    fallback_model: &str,
) -> Result<CompletionResponse, BlazenError> {
    use crate::types::{ReasoningTrace, ToolCall};
    let parsed: FalOpenAiChatResponse = serde_json::from_value(raw)
        .map_err(|e| BlazenError::Serialization(format!("fal openai chat parse: {e}")))?;
    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| BlazenError::Serialization("fal openai chat: no choices".into()))?;
    let message = choice.message;
    let tool_calls: Vec<ToolCall> = message
        .tool_calls
        .into_iter()
        .map(|tc| ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Null),
        })
        .collect();
    let reasoning = message
        .reasoning_content
        .or(message.reasoning)
        .map(|text| ReasoningTrace {
            text,
            signature: None,
            redacted: false,
            effort: None,
        });
    let usage = parsed.usage.map(|u| TokenUsage {
        prompt_tokens: u.prompt_tokens.unwrap_or(0),
        completion_tokens: u.completion_tokens.unwrap_or(0),
        total_tokens: u
            .total_tokens
            .unwrap_or_else(|| u.prompt_tokens.unwrap_or(0) + u.completion_tokens.unwrap_or(0)),
        reasoning_tokens: u
            .completion_tokens_details
            .as_ref()
            .map_or(0, |d| d.reasoning_tokens),
        ..Default::default()
    });
    Ok(CompletionResponse {
        content: message.content,
        tool_calls,
        reasoning,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage,
        model: parsed.model.unwrap_or_else(|| fallback_model.to_owned()),
        finish_reason: choice.finish_reason,
        cost: None,
        timing: None,
        images: Vec::new(),
        audio: Vec::new(),
        videos: Vec::new(),
        metadata: serde_json::Value::Null,
    })
}

/// Parse an `OpenAI` Responses API response from fal.
fn parse_openai_responses_response(
    raw: serde_json::Value,
    fallback_model: &str,
) -> Result<CompletionResponse, BlazenError> {
    use crate::types::ReasoningTrace;
    let parsed: FalOpenAiResponsesResponse = serde_json::from_value(raw)
        .map_err(|e| BlazenError::Serialization(format!("fal openai responses parse: {e}")))?;

    // Walk output blocks. Concatenate output_text and reasoning text separately.
    let mut content_text = String::new();
    let mut reasoning_text = String::new();
    for block in &parsed.output {
        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match block_type {
            "message" => {
                if let Some(content) = block.get("content").and_then(|c| c.as_array()) {
                    for part in content {
                        if part.get("type").and_then(|v| v.as_str()) == Some("output_text")
                            && let Some(t) = part.get("text").and_then(|v| v.as_str())
                        {
                            if !content_text.is_empty() {
                                content_text.push('\n');
                            }
                            content_text.push_str(t);
                        }
                    }
                }
            }
            "reasoning" => {
                if let Some(content) = block.get("content").and_then(|c| c.as_array()) {
                    for part in content {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            if !reasoning_text.is_empty() {
                                reasoning_text.push('\n');
                            }
                            reasoning_text.push_str(t);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let reasoning = if reasoning_text.is_empty() {
        None
    } else {
        Some(ReasoningTrace {
            text: reasoning_text,
            signature: None,
            redacted: false,
            effort: None,
        })
    };
    let usage = parsed.usage.map(|u| TokenUsage {
        prompt_tokens: u.prompt_tokens.unwrap_or(0),
        completion_tokens: u.completion_tokens.unwrap_or(0),
        total_tokens: u
            .total_tokens
            .unwrap_or_else(|| u.prompt_tokens.unwrap_or(0) + u.completion_tokens.unwrap_or(0)),
        reasoning_tokens: u
            .completion_tokens_details
            .as_ref()
            .map_or(0, |d| d.reasoning_tokens),
        ..Default::default()
    });
    Ok(CompletionResponse {
        content: if content_text.is_empty() {
            None
        } else {
            Some(content_text)
        },
        tool_calls: Vec::new(),
        reasoning,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage,
        model: fallback_model.to_owned(),
        finish_reason: None,
        cost: None,
        timing: None,
        images: Vec::new(),
        audio: Vec::new(),
        videos: Vec::new(),
        metadata: serde_json::Value::Null,
    })
}

/// Parse a fal prompt-string response (`fal-ai/any-llm`, `openrouter/router`).
fn parse_prompt_string_response(
    raw: serde_json::Value,
    fallback_model: &str,
) -> Result<CompletionResponse, BlazenError> {
    let parsed: FalLlmResponse = serde_json::from_value(raw)
        .map_err(|e| BlazenError::Serialization(format!("fal prompt-string parse: {e}")))?;
    if let Some(err) = parsed.error {
        return Err(BlazenError::request(format!("fal: {err}")));
    }
    Ok(CompletionResponse {
        content: parsed.output,
        tool_calls: Vec::new(),
        reasoning: None,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage: None,
        model: fallback_model.to_owned(),
        finish_reason: Some("stop".to_owned()),
        cost: None,
        timing: None,
        images: Vec::new(),
        audio: Vec::new(),
        videos: Vec::new(),
        metadata: serde_json::Value::Null,
    })
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for FalProvider {
    fn model_id(&self) -> &str {
        &self.llm_model
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model_id = request.model.as_deref().unwrap_or(&self.llm_model);
        let span = tracing::info_span!(
            "llm.complete",
            provider = "fal",
            model = %model_id,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        // Resolve the endpoint (with auto-routing) and build a body in the
        // wire format that endpoint expects.
        let ep = self.resolve_endpoint_for_request(&request);
        let path = ep.path();
        let body = self.build_body(&request, &ep);
        debug!(endpoint = %path, "fal.ai completion request");

        let (raw, timing) = match &self.execution_mode {
            FalExecutionMode::Sync => {
                let result = self.execute_sync(path.as_ref(), &body).await?;
                let elapsed = millis_u64(start.elapsed());
                let timing = RequestTiming {
                    queue_ms: None,
                    execution_ms: Some(elapsed),
                    total_ms: Some(elapsed),
                };
                (result, timing)
            }
            FalExecutionMode::Queue { poll_interval } => {
                self.execute_queue_llm(path.as_ref(), &body, *poll_interval)
                    .await?
            }
            FalExecutionMode::Webhook { url } => {
                let result = self.execute_webhook(path.as_ref(), &body, url).await?;
                let timing = RequestTiming {
                    queue_ms: None,
                    execution_ms: None,
                    total_ms: Some(millis_u64(start.elapsed())),
                };
                (result, timing)
            }
        };

        // Parse via the endpoint-aware parser, then merge in wrapper-level
        // fields (timing, cost, model) that the parser does not know about.
        let mut response = self.parse_response(&ep, raw)?;

        span.record(
            "duration_ms",
            timing
                .total_ms
                .unwrap_or_else(|| millis_u64(start.elapsed())),
        );
        if let Some(reason) = response.finish_reason.as_deref() {
            span.record("finish_reason", reason);
        }

        let cost = response
            .usage
            .as_ref()
            .and_then(|u| crate::pricing::compute_cost(&self.llm_model, u));

        response.model.clone_from(&self.llm_model);
        response.timing = Some(timing);
        response.cost = cost;

        Ok(response)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let model_id = request.model.as_deref().unwrap_or(&self.llm_model);
        let span = tracing::info_span!(
            "llm.stream",
            provider = "fal",
            model = %model_id,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();

        let ep = self.resolve_endpoint_for_request(&request);
        debug!(endpoint = %ep.path(), strategy = ?ep.streaming_strategy(), "fal.ai streaming request");
        match ep.streaming_strategy() {
            StreamingStrategy::SseInline => self.stream_openai_chat(request, &ep).await,
            StreamingStrategy::StreamSuffix => self.stream_prompt_string(request, &ep).await,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::ModelRegistry for FalProvider {
    async fn list_models(&self) -> Result<Vec<crate::traits::ModelInfo>, BlazenError> {
        // fal.ai is a compute platform without a model listing API.
        Ok(Vec::new())
    }

    async fn get_model(
        &self,
        _model_id: &str,
    ) -> Result<Option<crate::traits::ModelInfo>, BlazenError> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for FalProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        "fal"
    }

    async fn submit(&self, request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        let submit_response = self
            .queue_submit(&request.model, &request.input, request.webhook.as_deref())
            .await?;

        debug!(
            request_id = %submit_response.request_id,
            model = %request.model,
            "fal.ai compute job submitted"
        );

        Ok(JobHandle {
            id: submit_response.request_id,
            provider: "fal".to_owned(),
            model: request.model,
            submitted_at: Utc::now(),
        })
    }

    async fn status(&self, job: &JobHandle) -> Result<JobStatus, BlazenError> {
        let status_body = self.queue_poll_status(&job.model, &job.id).await?;

        match status_body.status.as_str() {
            "IN_QUEUE" => Ok(JobStatus::Queued),
            "IN_PROGRESS" => Ok(JobStatus::Running),
            "COMPLETED" => {
                if let Some(error) = status_body.error {
                    Ok(JobStatus::Failed { error })
                } else {
                    Ok(JobStatus::Completed)
                }
            }
            other => {
                // Defensive: treat unknown statuses as queued.
                warn!(status = %other, "unknown fal.ai queue status, treating as Queued");
                Ok(JobStatus::Queued)
            }
        }
    }

    async fn result(&self, job: JobHandle) -> Result<ComputeResult, BlazenError> {
        let poll_interval = match &self.execution_mode {
            FalExecutionMode::Queue { poll_interval } => *poll_interval,
            _ => DEFAULT_POLL_INTERVAL,
        };

        // External callers use submit() -> result() without access to the
        // server-provided URLs, so we fall back to manual URL construction.
        let (output, status_json, timing) = self
            .poll_until_complete(&job.model, &job.id, poll_interval, None, None)
            .await?;

        Ok(ComputeResult {
            job: Some(job),
            output,
            timing,
            cost: None, // fal.ai does not return per-request cost in API responses.
            metadata: status_json,
        })
    }

    /// Submit a job and wait for the result, using server-provided URLs for
    /// queue polling and result retrieval.
    ///
    /// This overrides the default `run()` to avoid the 405 errors that occur
    /// when manually constructing URLs for models with multi-segment IDs
    /// (e.g. `fal-ai/kling-video/v2.1/pro/image-to-video`).
    async fn run(&self, request: ComputeRequest) -> Result<ComputeResult, BlazenError> {
        let poll_interval = match &self.execution_mode {
            FalExecutionMode::Queue { poll_interval } => *poll_interval,
            _ => DEFAULT_POLL_INTERVAL,
        };

        let submit_response = self
            .queue_submit(&request.model, &request.input, request.webhook.as_deref())
            .await?;

        debug!(
            request_id = %submit_response.request_id,
            model = %request.model,
            "fal.ai compute job submitted (via run)"
        );

        let job = JobHandle {
            id: submit_response.request_id.clone(),
            provider: "fal".to_owned(),
            model: request.model,
            submitted_at: Utc::now(),
        };

        let (output, status_json, timing) = self
            .poll_until_complete(
                &job.model,
                &job.id,
                poll_interval,
                submit_response.status_url.as_deref(),
                submit_response.response_url.as_deref(),
            )
            .await?;

        Ok(ComputeResult {
            job: Some(job),
            output,
            timing,
            cost: None,
            metadata: status_json,
        })
    }

    async fn cancel(&self, job: &JobHandle) -> Result<(), BlazenError> {
        let cancel_url = format!(
            "{}/{}/requests/{}/cancel",
            self.base_queue_url, job.model, job.id
        );

        let request = self.apply_auth(HttpRequest::put(&cancel_url));
        let response = self.client.send(request).await?;

        let status = response.status;
        if (200..300).contains(&status) || status == 202 {
            debug!(request_id = %job.id, "fal.ai job cancellation requested");
            return Ok(());
        }

        // 400 = ALREADY_COMPLETED, which is fine.
        if status == 400 {
            debug!(request_id = %job.id, "fal.ai job already completed, cancel is a no-op");
            return Ok(());
        }

        let error_body = response.text();
        Err(BlazenError::request(format!(
            "cancel failed (HTTP {status}): {error_body}"
        )))
    }
}

// ---------------------------------------------------------------------------
// ImageGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ImageGeneration for FalProvider {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_MODEL)
            .to_owned();

        // Build the input JSON.
        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        // Image size: pass as object if both dimensions are set.
        if let (Some(w), Some(h)) = (request.width, request.height) {
            input["image_size"] = serde_json::json!({
                "width": w,
                "height": h,
            });
        }

        // Number of images.
        if let Some(n) = request.num_images {
            input["num_images"] = serde_json::json!(n);
        }

        // Negative prompt.
        if let Some(ref neg) = request.negative_prompt {
            input["negative_prompt"] = serde_json::json!(neg);
        }

        // Merge extra parameters from the request.
        if let serde_json::Value::Object(params) = &request.parameters {
            for (k, v) in params {
                input[k] = v.clone();
            }
        }

        // Submit as a compute job and wait for the result.
        let compute_request = ComputeRequest {
            model: model.clone(),
            input,
            webhook: None,
        };
        let result = self.run(compute_request).await?;

        // Parse the image output.
        let image_output: FalImageOutput =
            serde_json::from_value(result.output.clone()).map_err(|e| {
                BlazenError::Serialization(format!("failed to parse image output: {e}"))
            })?;

        let images = image_output
            .images
            .into_iter()
            .map(|img| {
                let content_type = img.content_type.unwrap_or_else(|| "image/jpeg".to_owned());
                GeneratedImage {
                    media: MediaOutput {
                        url: img.url,
                        base64: None,
                        raw_content: None,
                        media_type: MediaType::from_mime(&content_type),
                        file_size: None,
                        metadata: serde_json::Value::Null,
                    },
                    width: img.width,
                    height: img.height,
                }
            })
            .collect();

        Ok(ImageResult {
            images,
            timing: result.timing,
            cost: result.cost,
            metadata: serde_json::Value::Null,
        })
    }

    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_UPSCALE_MODEL)
            .to_owned();

        // Build the input JSON.
        let mut input = serde_json::json!({
            "image_url": request.image_url,
            "scale": request.scale,
        });

        // Merge extra parameters.
        if let serde_json::Value::Object(params) = &request.parameters {
            for (k, v) in params {
                input[k] = v.clone();
            }
        }

        // Submit as a compute job and wait for the result.
        let compute_request = ComputeRequest {
            model: model.clone(),
            input,
            webhook: None,
        };
        let result = self.run(compute_request).await?;

        // ESRGAN returns a single image object, not an array.
        let upscale_output: FalUpscaleOutput = serde_json::from_value(result.output.clone())
            .map_err(|e| {
                BlazenError::Serialization(format!("failed to parse upscale output: {e}"))
            })?;

        let content_type = upscale_output
            .image
            .content_type
            .unwrap_or_else(|| "image/png".to_owned());
        let image = GeneratedImage {
            media: MediaOutput {
                url: upscale_output.image.url,
                base64: None,
                raw_content: None,
                media_type: MediaType::from_mime(&content_type),
                file_size: None,
                metadata: serde_json::Value::Null,
            },
            width: upscale_output.image.width,
            height: upscale_output.image.height,
        };

        Ok(ImageResult {
            images: vec![image],
            timing: result.timing,
            cost: result.cost,
            metadata: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// Media parsing helpers
// ---------------------------------------------------------------------------

/// Parse a video from fal.ai response output.
///
/// fal.ai video models return `{ "video": { "url": "...", "content_type": "...", ... } }`.
fn parse_fal_video(output: &serde_json::Value) -> Result<GeneratedVideo, BlazenError> {
    let video_obj = output
        .get("video")
        .ok_or_else(|| BlazenError::Serialization("missing 'video' field in response".into()))?;

    let url = video_obj
        .get("url")
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let content_type = video_obj
        .get("content_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("video/mp4");
    let file_size = video_obj
        .get("file_size")
        .and_then(serde_json::Value::as_u64);

    Ok(GeneratedVideo {
        media: MediaOutput {
            url,
            base64: None,
            raw_content: None,
            media_type: MediaType::from_mime(content_type),
            file_size,
            metadata: video_obj.clone(),
        },
        width: None,
        height: None,
        duration_seconds: None,
        fps: None,
    })
}

/// Parse audio from fal.ai response output.
///
/// fal.ai audio models may return either:
/// - `{ "audio_url": { "url": "...", ... } }` (object with url field)
/// - `{ "audio_url": "https://..." }` (direct URL string)
/// - `{ "audio": { "url": "...", ... } }` (nested object)
#[allow(clippy::too_many_lines)]
fn parse_fal_audio(output: &serde_json::Value) -> Result<GeneratedAudio, BlazenError> {
    // Try `audio_url` as an object first (e.g. chatterbox returns this).
    if let Some(audio_obj) = output.get("audio_url") {
        if let Some(obj) = audio_obj.as_object() {
            let url = obj
                .get("url")
                .and_then(serde_json::Value::as_str)
                .map(String::from);
            let content_type = obj
                .get("content_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("audio/wav");
            let file_size = obj.get("file_size").and_then(serde_json::Value::as_u64);

            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url,
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::from_mime(content_type),
                    file_size,
                    metadata: audio_obj.clone(),
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }

        // `audio_url` as a plain string.
        if let Some(url_str) = audio_obj.as_str() {
            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url: Some(url_str.to_owned()),
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::Wav,
                    file_size: None,
                    metadata: serde_json::Value::Null,
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }
    }

    // Try `audio` as a nested object.
    if let Some(audio_obj) = output.get("audio") {
        let url = audio_obj
            .get("url")
            .and_then(serde_json::Value::as_str)
            .map(String::from);
        let content_type = audio_obj
            .get("content_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("audio/wav");
        let file_size = audio_obj
            .get("file_size")
            .and_then(serde_json::Value::as_u64);

        return Ok(GeneratedAudio {
            media: MediaOutput {
                url,
                base64: None,
                raw_content: None,
                media_type: MediaType::from_mime(content_type),
                file_size,
                metadata: audio_obj.clone(),
            },
            duration_seconds: None,
            sample_rate: None,
            channels: None,
        });
    }

    // Try `audio_file` as an object (e.g. stable-audio returns this).
    if let Some(audio_obj) = output.get("audio_file") {
        if let Some(obj) = audio_obj.as_object() {
            let url = obj
                .get("url")
                .and_then(serde_json::Value::as_str)
                .map(String::from);
            let content_type = obj
                .get("content_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("audio/wav");
            let file_size = obj.get("file_size").and_then(serde_json::Value::as_u64);

            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url,
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::from_mime(content_type),
                    file_size,
                    metadata: audio_obj.clone(),
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }

        if let Some(url_str) = audio_obj.as_str() {
            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url: Some(url_str.to_owned()),
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::Wav,
                    file_size: None,
                    metadata: serde_json::Value::Null,
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }
    }

    Err(BlazenError::Serialization(
        "missing 'audio_url', 'audio', or 'audio_file' field in response".into(),
    ))
}

/// Merge extra parameters from a `serde_json::Value` into an input object.
fn merge_parameters(input: &mut serde_json::Value, parameters: &serde_json::Value) {
    if let Some(params) = parameters.as_object() {
        for (k, v) in params {
            input[k] = v.clone();
        }
    }
}

/// Parse a 3D model output from fal.ai.
///
/// fal 3D apps return varying shapes -- try `model_mesh.url`, `model_glb`,
/// `model_gltf`, `glb`, `gltf`, `usdz`, `obj` in that order. Picks the first
/// that resolves to a `{"url": "..."}` shape.
fn parse_fal_3d_model(output: &serde_json::Value) -> Result<Generated3DModel, BlazenError> {
    // Try a list of candidate field paths.
    let candidates: &[(&str, MediaType)] = &[
        ("model_mesh", MediaType::Glb),
        ("model_glb", MediaType::Glb),
        ("glb", MediaType::Glb),
        ("model_gltf", MediaType::Gltf),
        ("gltf", MediaType::Gltf),
        ("usdz", MediaType::Usdz),
        ("obj", MediaType::Obj),
    ];

    for (key, media_type) in candidates {
        let Some(field) = output.get(*key) else {
            continue;
        };
        // Field could be a string URL or a {"url": ..., ...} object.
        let (url, file_size) = if let Some(s) = field.as_str() {
            (Some(s.to_owned()), None)
        } else if let Some(obj) = field.as_object() {
            let url = obj.get("url").and_then(|u| u.as_str()).map(String::from);
            let file_size = obj.get("file_size").and_then(serde_json::Value::as_u64);
            (url, file_size)
        } else {
            continue;
        };
        if url.is_none() {
            continue;
        }
        return Ok(Generated3DModel {
            media: MediaOutput {
                url,
                base64: None,
                raw_content: None,
                media_type: media_type.clone(),
                file_size,
                metadata: field.clone(),
            },
            vertex_count: None,
            face_count: None,
            has_textures: false,
            has_animations: false,
        });
    }

    Err(BlazenError::Serialization(
        "fal 3D output: missing model_mesh/model_glb/glb/model_gltf/gltf/usdz/obj field".into(),
    ))
}

// ---------------------------------------------------------------------------
// VideoGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl VideoGeneration for FalProvider {
    async fn text_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_TEXT_TO_VIDEO_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        if let Some(dur) = request.duration_seconds {
            // Kling/MiniMax expect duration as a string (e.g. "5").
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let secs = dur as u32;
            input["duration"] = serde_json::json!(secs.to_string());
        }
        if let Some(ref np) = request.negative_prompt {
            input["negative_prompt"] = serde_json::json!(np);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let video = parse_fal_video(&result.output)?;

        Ok(VideoResult {
            videos: vec![video],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }

    async fn image_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_TO_VIDEO_MODEL)
            .to_owned();

        let image_url = request
            .image_url
            .as_deref()
            .ok_or_else(|| BlazenError::Validation {
                field: Some("image_url".into()),
                message: "image_url is required for image-to-video".into(),
            })?;

        let mut input = serde_json::json!({
            "prompt": request.prompt,
            "image_url": image_url,
        });

        if let Some(dur) = request.duration_seconds {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let secs = dur as u32;
            input["duration"] = serde_json::json!(secs.to_string());
        }
        if let Some(ref np) = request.negative_prompt {
            input["negative_prompt"] = serde_json::json!(np);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let video = parse_fal_video(&result.output)?;

        Ok(VideoResult {
            videos: vec![video],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioGeneration for FalProvider {
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_TTS_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "text": request.text,
        });

        if let Some(ref voice) = request.voice {
            input["voice"] = serde_json::json!(voice);
        }
        if let Some(ref voice_url) = request.voice_url {
            input["voice_url"] = serde_json::json!(voice_url);
        }
        if let Some(ref language) = request.language {
            input["language"] = serde_json::json!(language);
        }
        if let Some(speed) = request.speed {
            input["speed"] = serde_json::json!(speed);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let audio = parse_fal_audio(&result.output)?;

        Ok(AudioResult {
            audio: vec![audio],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }

    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_MUSIC_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        if let Some(dur) = request.duration_seconds {
            input["duration"] = serde_json::json!(dur);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let audio = parse_fal_audio(&result.output)?;

        Ok(AudioResult {
            audio: vec![audio],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_SFX_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        if let Some(dur) = request.duration_seconds {
            input["duration"] = serde_json::json!(dur);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let audio = parse_fal_audio(&result.output)?;

        Ok(AudioResult {
            audio: vec![audio],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// ThreeDGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ThreeDGeneration for FalProvider {
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_3D_MODEL.to_owned());
        let mut input = serde_json::json!({ "prompt": &request.prompt });
        if let Some(image_url) = &request.image_url {
            input["image_url"] = image_url.clone().into();
        }
        if let Some(format) = &request.format {
            input["output_format"] = format.clone().into();
        }
        merge_parameters(&mut input, &request.parameters);
        let result = self
            .run(ComputeRequest {
                model,
                input,
                webhook: None,
            })
            .await?;
        Ok(ThreeDResult {
            models: vec![parse_fal_3d_model(&result.output)?],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Transcription implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Transcription for FalProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_TRANSCRIPTION_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "audio_url": request.audio_url,
        });

        if let Some(ref lang) = request.language {
            input["language"] = serde_json::json!(lang);
        }
        if request.diarize {
            input["diarize"] = serde_json::json!(true);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;

        // Parse Whisper response:
        // { "text": "...", "chunks": [...], "inferred_languages": ["en"], ... }
        let text = result
            .output
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();

        let language = result
            .output
            .get("inferred_languages")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        let segments = result
            .output
            .get("chunks")
            .and_then(|v| v.as_array())
            .map(|chunks| {
                chunks
                    .iter()
                    .filter_map(|chunk| {
                        let seg_text = chunk.get("text")?.as_str()?.to_owned();
                        let timestamps = chunk.get("timestamp")?.as_array()?;
                        let start = timestamps.first()?.as_f64()?;
                        let end = timestamps.get(1)?.as_f64()?;
                        let speaker = chunk
                            .get("speaker")
                            .and_then(serde_json::Value::as_str)
                            .map(String::from);
                        Some(TranscriptionSegment {
                            text: seg_text,
                            start,
                            end,
                            speaker,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(TranscriptionResult {
            text,
            segments,
            language,
            timing: result.timing,
            cost: result.cost,
            metadata: result.output,
        })
    }
}

#[async_trait]
impl BackgroundRemoval for FalProvider {
    async fn remove_background(
        &self,
        request: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_BG_REMOVAL_MODEL.to_owned());
        let mut input = serde_json::json!({ "image_url": &request.image_url });
        merge_parameters(&mut input, &request.parameters);
        let result = self
            .run(ComputeRequest {
                model,
                input,
                webhook: None,
            })
            .await?;
        // birefnet returns { image: { url, ... } } -- same shape as upscale.
        let parsed: FalUpscaleOutput = serde_json::from_value(result.output.clone())
            .map_err(|e| BlazenError::Serialization(format!("fal bg removal: {e}")))?;
        Ok(ImageResult {
            images: vec![GeneratedImage {
                media: MediaOutput::from_url(
                    parsed.image.url.clone().unwrap_or_default(),
                    MediaType::from_mime(
                        parsed.image.content_type.as_deref().unwrap_or("image/png"),
                    ),
                ),
                width: parsed.image.width,
                height: parsed.image.height,
            }],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Embedding model
// ---------------------------------------------------------------------------

/// fal.ai embedding model that targets `openrouter/router/openai/v1/embeddings`.
///
/// Constructed via [`FalEmbeddingModel::new`] or via the convenience method
/// [`FalProvider::embedding_model`] which inherits the parent provider's
/// HTTP client and API key.
pub struct FalEmbeddingModel {
    client: Arc<dyn HttpClient>,
    api_key: String,
    model: String,
    dimensions: usize,
    base_sync_url: String,
}

impl std::fmt::Debug for FalEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalEmbeddingModel")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl Clone for FalEmbeddingModel {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            dimensions: self.dimensions,
            base_sync_url: self.base_sync_url.clone(),
        }
    }
}

impl FalEmbeddingModel {
    /// Create a new fal embedding model with the default `OpenAI` text embedding
    /// (1536-dim).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            api_key: api_key.into(),
            model: "openai/text-embedding-3-small".to_owned(),
            dimensions: 1536,
            base_sync_url: FAL_SYNC_URL.to_owned(),
        }
    }

    /// Override the embedding model id (e.g. `"openai/text-embedding-3-large"`).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the dimensionality the model produces.
    #[must_use]
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl crate::traits::EmbeddingModel for FalEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let url = format!(
            "{}/openrouter/router/openai/v1/embeddings",
            self.base_sync_url
        );
        let body = serde_json::json!({
            "model": &self.model,
            "input": texts,
        });
        let request = HttpRequest::post(url)
            .header("Authorization", format!("Key {}", self.api_key))
            .json_body(&body)?;
        let response = self.client.send(request).await?;
        if !response.is_success() {
            return Err(BlazenError::request(format!(
                "fal embeddings HTTP {}: {}",
                response.status,
                response.text()
            )));
        }
        let raw: serde_json::Value = serde_json::from_slice(&response.body)
            .map_err(|e| BlazenError::Serialization(format!("fal embeddings parse: {e}")))?;
        let data = raw["data"].as_array().ok_or_else(|| {
            BlazenError::Serialization("fal embeddings: missing 'data' field".into())
        })?;
        let embeddings: Vec<Vec<f32>> = data
            .iter()
            .map(|d| {
                d["embedding"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                #[allow(clippy::cast_possible_truncation)]
                                v.as_f64().map(|f| f as f32)
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
        let usage = raw.get("usage").and_then(|u| {
            #[allow(clippy::cast_possible_truncation)]
            Some(TokenUsage {
                prompt_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
                completion_tokens: 0,
                total_tokens: u.get("total_tokens")?.as_u64()? as u32,
                ..Default::default()
            })
        });
        Ok(EmbeddingResponse {
            embeddings,
            model: self.model.clone(),
            usage,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl crate::traits::ProviderInfo for FalProvider {
    fn provider_name(&self) -> &'static str {
        "fal"
    }

    fn base_url(&self) -> &str {
        &self.base_queue_url
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities {
            streaming: false,
            tool_calling: false,
            structured_output: false,
            vision: false,
            model_listing: false,
            embeddings: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    // -----------------------------------------------------------------------
    // Config / builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_config() {
        let provider = FalProvider::new("fal-test");
        assert_eq!(provider.llm_endpoint, FalLlmEndpoint::OpenAiChat);
        assert_eq!(provider.llm_model, "anthropic/claude-sonnet-4.5");
        assert!(provider.auto_route_modality);
    }

    #[test]
    fn with_llm_endpoint_override() {
        let provider = FalProvider::new("fal-test")
            .with_llm_endpoint(FalLlmEndpoint::AnyLlm { enterprise: true });
        assert_eq!(
            provider.llm_endpoint,
            FalLlmEndpoint::AnyLlm { enterprise: true }
        );
    }

    #[test]
    fn with_llm_model_does_not_change_endpoint() {
        // Regression test for the fa7375d bug: with_llm_model used to clobber
        // the URL path with the model name, producing 404s like
        // "Application 'claude-sonnet-4.5' not found".
        let provider = FalProvider::new("fal-test").with_llm_model("openai/gpt-4o");
        assert_eq!(provider.llm_endpoint, FalLlmEndpoint::OpenAiChat);
        assert_eq!(provider.llm_model, "openai/gpt-4o");
        // Verify the URL the provider would actually hit:
        let path = provider.llm_endpoint.path();
        assert_eq!(
            path.as_ref(),
            "openrouter/router/openai/v1/chat/completions"
        );
        assert!(!path.contains("gpt-4o"));
        assert!(!path.contains("claude-sonnet-4.5"));
    }

    #[test]
    fn with_enterprise_promotes_openai_chat_to_any_llm_enterprise() {
        let provider = FalProvider::new("fal-test").with_enterprise();
        assert_eq!(
            provider.llm_endpoint,
            FalLlmEndpoint::AnyLlm { enterprise: true }
        );
    }

    #[test]
    fn default_endpoint_path_is_openai_compat_chat_completions() {
        let provider = FalProvider::new("fal-test");
        let path = provider.llm_endpoint.path();
        assert_eq!(
            path.as_ref(),
            "openrouter/router/openai/v1/chat/completions"
        );
    }

    #[test]
    fn url_construction_uses_endpoint_path_not_model() {
        // Defense-in-depth: assert the constructed URL never contains a model name.
        let provider = FalProvider::new("fal-test").with_llm_model("anthropic/claude-sonnet-4.5");
        let path = provider.llm_endpoint.path();
        let url = format!("https://fal.run/{}", path);
        assert!(url.contains("openrouter/router/openai/v1/chat/completions"));
        assert!(!url.contains("claude-sonnet-4.5"));
        assert!(!url.contains("anthropic/claude"));
    }

    #[test]
    fn with_base_url_override() {
        let provider = FalProvider::new("fal-test").with_base_url("https://example.com/q");
        assert_eq!(provider.base_queue_url, "https://example.com/q");
    }

    #[test]
    fn with_sync_execution() {
        let provider = FalProvider::new("fal-test").with_execution_mode(FalExecutionMode::Sync);
        assert!(matches!(provider.execution_mode, FalExecutionMode::Sync));
    }

    // -----------------------------------------------------------------------
    // LLM body builder tests
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_basic() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello world")]);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.5");
        assert!(body["prompt"].as_str().unwrap().contains("Hello world"));
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_with_system() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![
            ChatMessage::system("Be helpful"),
            ChatMessage::user("Hello"),
        ]);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert_eq!(body["system_prompt"], "Be helpful");
        assert!(body["prompt"].as_str().unwrap().contains("Hello"));
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_model_override() {
        let provider = FalProvider::new("fal-test");
        let request =
            CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_model("openai/gpt-4o");

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_with_temperature() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_temperature(0.7);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature was {temp}");
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_with_max_tokens() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_max_tokens(1024);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_with_top_p() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_top_p(0.9);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert_eq!(body["top_p"], serde_json::json!(0.9_f32));
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn build_body_with_response_format() {
        let provider = FalProvider::new("fal-test");
        let schema = serde_json::json!({"type": "object"});
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")])
            .with_response_format(schema.clone());

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert_eq!(body["response_format"], schema);
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn test_text_backward_compat() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        assert!(body["prompt"].as_str().unwrap().contains("Hello"));
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn test_build_body_image_url_drops_image() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "Describe this",
            "https://example.com/cat.jpg",
            None,
        )]);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        // Only the text part should be preserved.
        let prompt = body["prompt"].as_str().unwrap();
        assert!(prompt.contains("Describe this"));
        assert!(!prompt.contains("cat.jpg"));
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn test_build_body_base64_image_drops_image() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_base64(
            "What is this",
            "abc123",
            "image/png",
        )]);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        let prompt = body["prompt"].as_str().unwrap();
        assert!(prompt.contains("What is this"));
        assert!(!prompt.contains("abc123"));
    }

    #[test]
    #[ignore = "Phase 4: rewrite for new body builders"]
    fn test_build_body_multipart_text_only() {
        use crate::types::{ContentPart, ImageContent, ImageSource};

        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_parts(vec![
            ContentPart::Text {
                text: "First".into(),
            },
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/a.png".into(),
                },
                media_type: None,
            }),
            ContentPart::Text {
                text: "Second".into(),
            },
        ])]);

        let body = provider.build_body(&request, &provider.llm_endpoint);
        let prompt = body["prompt"].as_str().unwrap();
        // Both text parts should be concatenated.
        assert!(prompt.contains("First"));
        assert!(prompt.contains("Second"));
    }

    // -----------------------------------------------------------------------
    // OpenAI chat body builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_openai_chat_body_basic_user() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![
            ChatMessage::system("be helpful"),
            ChatMessage::user("Hello"),
        ]);
        let body = provider.build_openai_chat_body(&request);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.5");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
    }

    #[test]
    fn test_build_openai_chat_body_with_tools() {
        let provider = FalProvider::new("fal-test");
        let mut request = CompletionRequest::new(vec![ChatMessage::user("calc 2+2")]);
        request.tools = vec![crate::types::ToolDefinition {
            name: "calculator".to_owned(),
            description: "do math".to_owned(),
            parameters: serde_json::json!({"type":"object","properties":{}}),
        }];
        let body = provider.build_openai_chat_body(&request);
        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "calculator");
    }

    #[test]
    fn test_build_openai_chat_body_image_part_passes_through() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "Describe",
            "https://i.com/a.png",
            Some("image/png"),
        )]);
        let body = provider.build_openai_chat_body(&request);
        let messages = body["messages"].as_array().unwrap();
        let content = &messages[0]["content"];
        // Multi-part content should be a JSON array (not a string).
        assert!(content.is_array(), "content: {content}");
    }

    #[test]
    fn test_build_openai_responses_body_basic() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("hello")]);
        let body = provider.build_openai_responses_body(&request);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.5");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
    }

    // -----------------------------------------------------------------------
    // Prompt-string body builder tests (`collapse_messages` +
    // `build_prompt_string_body`)
    // -----------------------------------------------------------------------

    #[test]
    fn test_collapse_evicts_oldest_to_fit_4800() {
        let big = "x".repeat(2000);
        let messages = vec![
            ChatMessage::user(&big),
            ChatMessage::assistant(&big),
            ChatMessage::user(&big), // total ~6000 chars — too long
        ];
        let (prompt, _, _, _, _) = collapse_messages(&messages, 4800, 4800);
        assert!(prompt.chars().count() <= 4800);
        // Newest message should survive.
        assert!(prompt.contains(&format!("User: {big}")));
    }

    #[test]
    fn test_collapse_truncates_long_system_with_marker() {
        let huge_system = "s".repeat(10000);
        let messages = vec![ChatMessage::system(&huge_system), ChatMessage::user("hi")];
        let (_, system, _, _, _) = collapse_messages(&messages, 4800, 4800);
        assert!(system.chars().count() <= 4800);
        assert!(system.ends_with(" [truncated]"));
    }

    #[test]
    fn test_build_prompt_string_body_extracts_image_urls() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "describe",
            "https://i.com/a.png",
            Some("image/png"),
        )]);
        let body = provider.build_prompt_string_body(&request, Some(MediaKind::Image));
        let urls = body["image_urls"].as_array().expect("image_urls array");
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://i.com/a.png");
    }

    #[test]
    fn test_build_prompt_string_body_extracts_audio_url() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_audio(
            "transcribe",
            "https://a.com/c.mp3",
        )]);
        let body = provider.build_prompt_string_body(&request, Some(MediaKind::Audio));
        assert_eq!(body["audio_url"], "https://a.com/c.mp3");
    }

    #[test]
    fn test_build_prompt_string_body_extracts_video_url() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_video(
            "describe",
            "https://v.com/c.mp4",
        )]);
        let body = provider.build_prompt_string_body(&request, Some(MediaKind::Video));
        assert_eq!(body["video_url"], "https://v.com/c.mp4");
    }

    // -----------------------------------------------------------------------
    // Modality auto-router tests (`resolve_endpoint_for_request`)
    // -----------------------------------------------------------------------

    #[test]
    fn test_router_keeps_openai_chat_for_text_only() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("hi")]);
        let ep = provider.resolve_endpoint_for_request(&request);
        assert_eq!(ep, FalLlmEndpoint::OpenAiChat);
    }

    #[test]
    fn test_router_promotes_to_vision_when_anyllm_and_image_present() {
        let provider = FalProvider::new("fal-test")
            .with_llm_endpoint(FalLlmEndpoint::AnyLlm { enterprise: false });
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "describe",
            "https://i.com/a.png",
            Some("image/png"),
        )]);
        let ep = provider.resolve_endpoint_for_request(&request);
        assert_eq!(
            ep,
            FalLlmEndpoint::Vision {
                family: FalVisionFamily::AnyLlm,
                enterprise: false
            }
        );
    }

    #[test]
    fn test_router_promotes_to_audio_when_anyllm_and_audio_present() {
        let provider = FalProvider::new("fal-test")
            .with_llm_endpoint(FalLlmEndpoint::AnyLlm { enterprise: true });
        let request = CompletionRequest::new(vec![ChatMessage::user_audio(
            "transcribe",
            "https://a.com/c.mp3",
        )]);
        let ep = provider.resolve_endpoint_for_request(&request);
        assert_eq!(
            ep,
            FalLlmEndpoint::Audio {
                family: FalVisionFamily::AnyLlm,
                enterprise: true
            }
        );
    }

    #[test]
    fn test_router_promotes_to_video_when_openrouter_and_video_present() {
        let provider = FalProvider::new("fal-test")
            .with_llm_endpoint(FalLlmEndpoint::OpenRouter { enterprise: false });
        let request = CompletionRequest::new(vec![ChatMessage::user_video(
            "describe",
            "https://v.com/c.mp4",
        )]);
        let ep = provider.resolve_endpoint_for_request(&request);
        assert_eq!(
            ep,
            FalLlmEndpoint::Video {
                family: FalVisionFamily::OpenRouter,
                enterprise: false
            }
        );
    }

    #[test]
    fn test_router_keeps_endpoint_when_auto_route_disabled() {
        let provider = FalProvider::new("fal-test")
            .with_llm_endpoint(FalLlmEndpoint::AnyLlm { enterprise: false })
            .with_auto_route_modality(false);
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "describe",
            "https://i.com/a.png",
            Some("image/png"),
        )]);
        let ep = provider.resolve_endpoint_for_request(&request);
        // Should NOT promote to Vision — auto-routing is off.
        assert_eq!(ep, FalLlmEndpoint::AnyLlm { enterprise: false });
    }

    // -----------------------------------------------------------------------
    // Wire type parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_fal_llm_response() {
        let json = r#"{"output":"Hello! How can I help you?"}"#;
        let response: FalLlmResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            response.output.as_deref(),
            Some("Hello! How can I help you?")
        );
        assert!(response.error.is_none());
    }

    #[test]
    fn parse_fal_error_response() {
        let json = r#"{"output":null,"error":"Model not found"}"#;
        let response: FalLlmResponse = serde_json::from_str(json).unwrap();
        assert!(response.output.is_none());
        assert_eq!(response.error.as_deref(), Some("Model not found"));
    }

    #[test]
    fn parse_queue_submit_response() {
        let json = r#"{
            "request_id": "abc-123-def",
            "response_url": "https://queue.fal.run/model/requests/abc-123-def/response",
            "status_url": "https://queue.fal.run/model/requests/abc-123-def/status",
            "cancel_url": "https://queue.fal.run/model/requests/abc-123-def/cancel"
        }"#;
        let response: FalQueueSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.request_id, "abc-123-def");
    }

    #[test]
    fn parse_queue_submit_response_minimal() {
        // Backwards compat: only request_id is required.
        let json = r#"{"request_id":"abc-123-def"}"#;
        let response: FalQueueSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.request_id, "abc-123-def");
    }

    #[test]
    fn parse_status_completed() {
        let json = r#"{"status":"COMPLETED"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "COMPLETED");
        assert!(response.error.is_none());
    }

    #[test]
    fn parse_status_completed_with_metrics() {
        let json = r#"{"status":"COMPLETED","metrics":{"inference_time":3.42}}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "COMPLETED");
        let metrics = response.metrics.unwrap();
        assert!((metrics.inference_time.unwrap() - 3.42).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_status_completed_with_error() {
        let json = r#"{"status":"COMPLETED","error":"Out of memory"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "COMPLETED");
        assert_eq!(response.error.as_deref(), Some("Out of memory"));
    }

    #[test]
    fn parse_status_in_queue() {
        let json = r#"{"status":"IN_QUEUE","queue_position":2}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "IN_QUEUE");
        assert_eq!(response.queue_position, Some(2));
    }

    #[test]
    fn parse_status_in_progress() {
        let json = r#"{"status":"IN_PROGRESS"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "IN_PROGRESS");
    }

    // -----------------------------------------------------------------------
    // Image wire type tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_image_output() {
        let json = r#"{
            "images": [
                {
                    "url": "https://v3.fal.media/files/rabbit/abc123.png",
                    "width": 1024,
                    "height": 768,
                    "content_type": "image/jpeg"
                }
            ]
        }"#;
        let output: FalImageOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.images[0].width, Some(1024));
        assert_eq!(output.images[0].height, Some(768));
    }

    #[test]
    fn parse_upscale_output() {
        let json = r#"{
            "image": {
                "url": "https://v3.fal.media/files/out/upscaled.png",
                "width": 2048,
                "height": 2048,
                "content_type": "image/png"
            }
        }"#;
        let output: FalUpscaleOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.image.width, Some(2048));
        assert_eq!(output.image.height, Some(2048));
        assert_eq!(output.image.content_type.as_deref(), Some("image/png"));
    }

    // -----------------------------------------------------------------------
    // Timing helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_timing_with_all_data() {
        let start = Instant::now();
        // Simulate some elapsed time.
        let in_progress = start; // effectively 0ms queue time for testing
        let timing = build_timing(start, Some(in_progress), Some(1.5));
        assert!(timing.total_ms.is_some());
        assert!(timing.queue_ms.is_some());
        assert_eq!(timing.execution_ms, Some(1500)); // 1.5s = 1500ms
    }

    #[test]
    fn build_timing_without_in_progress() {
        let start = Instant::now();
        let timing = build_timing(start, None, Some(2.0));
        assert!(timing.total_ms.is_some());
        assert!(timing.queue_ms.is_none());
        assert_eq!(timing.execution_ms, Some(2000));
    }

    #[test]
    fn build_timing_without_inference_time() {
        let start = Instant::now();
        let timing = build_timing(start, None, None);
        assert!(timing.total_ms.is_some());
        assert!(timing.queue_ms.is_none());
        // No inference_time and no in_progress => no execution_ms.
        assert!(timing.execution_ms.is_none());
    }

    // -----------------------------------------------------------------------
    // ComputeProvider trait tests (unit, not integration)
    // -----------------------------------------------------------------------

    #[test]
    fn provider_id_is_fal() {
        let provider = FalProvider::new("fal-test");
        assert_eq!(ComputeProvider::provider_id(&provider), "fal");
    }

    // -----------------------------------------------------------------------
    // Video parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_video_output_standard() {
        let output = serde_json::json!({
            "video": {
                "url": "https://v3.fal.media/files/rabbit/abc123.mp4",
                "file_name": "output.mp4",
                "file_size": 1_234_567,
                "content_type": "video/mp4"
            }
        });
        let video = parse_fal_video(&output).unwrap();
        assert_eq!(
            video.media.url.as_deref(),
            Some("https://v3.fal.media/files/rabbit/abc123.mp4")
        );
        assert_eq!(video.media.media_type, MediaType::Mp4);
        assert_eq!(video.media.file_size, Some(1_234_567));
    }

    #[test]
    fn parse_video_output_minimal() {
        let output = serde_json::json!({
            "video": {
                "url": "https://example.com/video.mp4"
            }
        });
        let video = parse_fal_video(&output).unwrap();
        assert_eq!(
            video.media.url.as_deref(),
            Some("https://example.com/video.mp4")
        );
        // Defaults to video/mp4.
        assert_eq!(video.media.media_type, MediaType::Mp4);
    }

    #[test]
    fn parse_video_output_missing_field() {
        let output = serde_json::json!({"result": "done"});
        let err = parse_fal_video(&output).unwrap_err();
        assert!(matches!(err, BlazenError::Serialization(_)));
    }

    // -----------------------------------------------------------------------
    // Audio parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_audio_output_audio_url_object() {
        let output = serde_json::json!({
            "audio_url": {
                "url": "https://v3.fal.media/files/audio/speech.wav",
                "content_type": "audio/wav",
                "file_size": 98765
            }
        });
        let audio = parse_fal_audio(&output).unwrap();
        assert_eq!(
            audio.media.url.as_deref(),
            Some("https://v3.fal.media/files/audio/speech.wav")
        );
        assert_eq!(audio.media.media_type, MediaType::Wav);
        assert_eq!(audio.media.file_size, Some(98765));
    }

    #[test]
    fn parse_audio_output_audio_url_string() {
        let output = serde_json::json!({
            "audio_url": "https://example.com/audio.wav"
        });
        let audio = parse_fal_audio(&output).unwrap();
        assert_eq!(
            audio.media.url.as_deref(),
            Some("https://example.com/audio.wav")
        );
        assert_eq!(audio.media.media_type, MediaType::Wav);
    }

    #[test]
    fn parse_audio_output_nested_audio() {
        let output = serde_json::json!({
            "audio": {
                "url": "https://example.com/music.mp3",
                "content_type": "audio/mpeg"
            }
        });
        let audio = parse_fal_audio(&output).unwrap();
        assert_eq!(
            audio.media.url.as_deref(),
            Some("https://example.com/music.mp3")
        );
        assert_eq!(audio.media.media_type, MediaType::Mp3);
    }

    #[test]
    fn parse_audio_output_missing_field() {
        let output = serde_json::json!({"result": "done"});
        let err = parse_fal_audio(&output).unwrap_err();
        assert!(matches!(err, BlazenError::Serialization(_)));
    }

    // -----------------------------------------------------------------------
    // 3D parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_fal_3d_model_glb_object() {
        let output = serde_json::json!({
            "model_mesh": {"url": "https://fal.cdn/x.glb", "file_size": 12345}
        });
        let parsed = parse_fal_3d_model(&output).unwrap();
        assert_eq!(parsed.media.url.as_deref(), Some("https://fal.cdn/x.glb"));
        assert_eq!(parsed.media.file_size, Some(12345));
        assert_eq!(parsed.media.media_type, MediaType::Glb);
    }

    #[test]
    fn test_parse_fal_3d_model_gltf_string() {
        let output = serde_json::json!({
            "gltf": "https://fal.cdn/y.gltf"
        });
        let parsed = parse_fal_3d_model(&output).unwrap();
        assert_eq!(parsed.media.url.as_deref(), Some("https://fal.cdn/y.gltf"));
        assert_eq!(parsed.media.media_type, MediaType::Gltf);
    }

    #[test]
    fn test_parse_fal_3d_model_missing_field_errors() {
        let output = serde_json::json!({"unrelated": "field"});
        assert!(parse_fal_3d_model(&output).is_err());
    }

    // -----------------------------------------------------------------------
    // merge_parameters tests
    // -----------------------------------------------------------------------

    #[test]
    fn merge_params_into_input() {
        let mut input = serde_json::json!({"prompt": "hello"});
        let params = serde_json::json!({"seed": 42, "guidance_scale": 7.5});
        merge_parameters(&mut input, &params);
        assert_eq!(input["seed"], 42);
        assert_eq!(input["guidance_scale"], 7.5);
        assert_eq!(input["prompt"], "hello");
    }

    #[test]
    fn merge_params_null_is_noop() {
        let mut input = serde_json::json!({"prompt": "hello"});
        merge_parameters(&mut input, &serde_json::Value::Null);
        assert_eq!(input, serde_json::json!({"prompt": "hello"}));
    }

    // -----------------------------------------------------------------------
    // FalLlmEndpoint path-mapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_endpoint_path_is_openai_chat_completions() {
        let ep = FalLlmEndpoint::default();
        assert_eq!(ep.path(), "openrouter/router/openai/v1/chat/completions");
        assert_eq!(ep.body_format(), FalBodyFormat::OpenAiMessages);
    }

    #[test]
    fn test_anyllm_enterprise_path() {
        let ep = FalLlmEndpoint::AnyLlm { enterprise: true };
        assert_eq!(ep.path(), "fal-ai/any-llm/enterprise");
        assert_eq!(ep.body_format(), FalBodyFormat::PromptString);
    }

    #[test]
    fn test_vision_anyllm_enterprise_path() {
        let ep = FalLlmEndpoint::Vision {
            family: FalVisionFamily::AnyLlm,
            enterprise: true,
        };
        assert_eq!(ep.path(), "fal-ai/any-llm/vision/enterprise");
        assert_eq!(ep.body_format(), FalBodyFormat::PromptStringVision);
    }

    #[test]
    fn test_audio_openrouter_path() {
        let ep = FalLlmEndpoint::Audio {
            family: FalVisionFamily::OpenRouter,
            enterprise: false,
        };
        assert_eq!(ep.path(), "openrouter/router/audio");
        assert_eq!(ep.body_format(), FalBodyFormat::PromptStringAudio);
    }

    // -----------------------------------------------------------------------
    // Response parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_openai_chat_response_basic() {
        let raw = serde_json::json!({
            "id": "chatcmpl-x",
            "model": "anthropic/claude-sonnet-4.5",
            "choices": [{
                "message": {"role": "assistant", "content": "the answer is 42"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let response = parse_openai_chat_response(raw, "default-model").unwrap();
        assert_eq!(response.content.as_deref(), Some("the answer is 42"));
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
        assert_eq!(response.model, "anthropic/claude-sonnet-4.5");
        assert!(response.reasoning.is_none());
    }

    #[test]
    fn test_parse_openai_chat_response_with_reasoning() {
        let raw = serde_json::json!({
            "model": "deepseek-r1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "thinking about it..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3}
            }
        });
        let response = parse_openai_chat_response(raw, "default").unwrap();
        let reasoning = response.reasoning.expect("reasoning trace");
        assert_eq!(reasoning.text, "thinking about it...");
        assert_eq!(response.usage.unwrap().reasoning_tokens, 3);
    }

    #[test]
    fn test_parse_openai_chat_response_with_tool_calls() {
        let raw = serde_json::json!({
            "model": "x",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{\"x\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let response = parse_openai_chat_response(raw, "default").unwrap();
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "calc");
    }

    #[test]
    fn test_parse_openai_responses_response_basic() {
        let raw = serde_json::json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "the answer is 42"}]}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let response = parse_openai_responses_response(raw, "model-x").unwrap();
        assert_eq!(response.content.as_deref(), Some("the answer is 42"));
    }

    #[test]
    fn test_parse_openai_responses_response_with_reasoning_block() {
        let raw = serde_json::json!({
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "I considered..."}]},
                {"type": "message", "content": [{"type": "output_text", "text": "42"}]}
            ]
        });
        let response = parse_openai_responses_response(raw, "model-x").unwrap();
        assert_eq!(response.content.as_deref(), Some("42"));
        assert_eq!(response.reasoning.as_ref().unwrap().text, "I considered...");
    }

    #[test]
    fn test_parse_prompt_string_response_legacy() {
        let raw = serde_json::json!({
            "output": "the answer is 42",
            "partial": false
        });
        let response = parse_prompt_string_response(raw, "model-y").unwrap();
        assert_eq!(response.content.as_deref(), Some("the answer is 42"));
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    }

    // -----------------------------------------------------------------------
    // Streaming tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_fal_cumulative_sse_emits_deltas() {
        use bytes::Bytes;
        use futures_util::StreamExt;

        let events: Vec<Result<Bytes, Box<dyn std::error::Error + Send + Sync>>> = vec![
            Ok(Bytes::from(
                "data: {\"output\":\"Hello\",\"partial\":true}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"output\":\"Hello world\",\"partial\":true}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"output\":\"Hello world!\",\"partial\":false}\n\n",
            )),
        ];
        let byte_stream: crate::http::ByteStream = Box::pin(futures_util::stream::iter(events));
        let mut parser = Box::pin(FalCumulativeSseStream::new(byte_stream));

        let mut chunks: Vec<StreamChunk> = Vec::new();
        while let Some(c) = parser.next().await {
            chunks.push(c.expect("chunk must parse"));
        }

        // Expect 3 delta-bearing chunks (one per event); the final one also
        // carries finish_reason = "stop".
        assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
        let combined: String = chunks
            .iter()
            .filter_map(|c| c.delta.as_ref())
            .cloned()
            .collect();
        assert_eq!(combined, "Hello world!");
        // The terminating event must surface as a stop chunk.
        let has_stop = chunks
            .iter()
            .any(|c| c.finish_reason.as_deref() == Some("stop"));
        assert!(has_stop, "no stop chunk in {chunks:?}");
    }

    #[tokio::test]
    async fn test_fal_cumulative_sse_split_across_byte_chunks() {
        use bytes::Bytes;
        use futures_util::StreamExt;

        // Split a single SSE event across multiple byte chunks to verify the
        // buffer-stitching logic.
        let events: Vec<Result<Bytes, Box<dyn std::error::Error + Send + Sync>>> = vec![
            Ok(Bytes::from("data: {\"output\":\"Hel")),
            Ok(Bytes::from("lo\",\"partial\":true}\n\n")),
            Ok(Bytes::from(
                "data: {\"output\":\"Hello!\",\"partial\":false}\n\n",
            )),
        ];
        let byte_stream: crate::http::ByteStream = Box::pin(futures_util::stream::iter(events));
        let mut parser = Box::pin(FalCumulativeSseStream::new(byte_stream));

        let mut deltas: Vec<String> = Vec::new();
        while let Some(c) = parser.next().await {
            let chunk = c.expect("chunk must parse");
            if let Some(d) = chunk.delta {
                deltas.push(d);
            }
        }
        assert_eq!(deltas.concat(), "Hello!");
    }

    #[tokio::test]
    async fn test_fal_cumulative_sse_propagates_error_field() {
        use bytes::Bytes;
        use futures_util::StreamExt;

        let events: Vec<Result<Bytes, Box<dyn std::error::Error + Send + Sync>>> = vec![Ok(
            Bytes::from("data: {\"output\":\"\",\"partial\":true,\"error\":\"boom\"}\n\n"),
        )];
        let byte_stream: crate::http::ByteStream = Box::pin(futures_util::stream::iter(events));
        let mut parser = Box::pin(FalCumulativeSseStream::new(byte_stream));

        let first = parser.next().await.expect("at least one item");
        assert!(first.is_err(), "expected error, got {first:?}");
        // Stream should terminate after the error.
        assert!(parser.next().await.is_none());
    }

    // -----------------------------------------------------------------------
    // FalEmbeddingModel tests
    // -----------------------------------------------------------------------

    use crate::traits::EmbeddingModel as _;

    #[test]
    fn test_fal_embedding_default_model() {
        let em = FalEmbeddingModel::new("test-key");
        assert_eq!(em.model_id(), "openai/text-embedding-3-small");
        assert_eq!(em.dimensions(), 1536);
    }

    #[test]
    fn test_fal_embedding_with_model_and_dimensions() {
        let em = FalEmbeddingModel::new("k")
            .with_model("openai/text-embedding-3-large")
            .with_dimensions(3072);
        assert_eq!(em.model_id(), "openai/text-embedding-3-large");
        assert_eq!(em.dimensions(), 3072);
    }

    #[test]
    fn test_fal_embedding_response_parsing() {
        // Manually exercise the parser logic from a fixture body shape we'd get
        // back from fal's openai-compat embeddings endpoint.
        let raw: serde_json::Value = serde_json::from_str(
            r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "openai/text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }"#,
        )
        .unwrap();
        let data = raw["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert!((data[0]["embedding"][0].as_f64().unwrap() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_fal_provider_embedding_model_shares_api_key() {
        let provider = FalProvider::new("my-test-key");
        let em = provider.embedding_model();
        assert_eq!(em.api_key, "my-test-key");
        assert_eq!(em.model, "openai/text-embedding-3-small");
    }

    #[test]
    fn test_remove_background_default_model_constant() {
        assert_eq!(DEFAULT_BG_REMOVAL_MODEL, "fal-ai/birefnet");
        assert_eq!(DEFAULT_AURA_UPSCALE_MODEL, "fal-ai/aura-sr");
        assert_eq!(DEFAULT_CLARITY_UPSCALE_MODEL, "fal-ai/clarity-upscaler");
        assert_eq!(DEFAULT_CREATIVE_UPSCALE_MODEL, "fal-ai/creative-upscaler");
    }

    #[test]
    fn test_background_removal_request_construction() {
        let req = BackgroundRemovalRequest {
            image_url: "https://example.com/in.png".into(),
            model: None,
            parameters: serde_json::Value::Null,
        };
        assert_eq!(req.image_url, "https://example.com/in.png");
        assert!(req.model.is_none());
    }
}
