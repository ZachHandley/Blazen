//! Compute request/response types for the UniFFI bindings.
//!
//! Mirrors the typed media-generation surface exposed by
//! [`blazen_llm::compute::requests`] and [`blazen_llm::compute::results`] —
//! the 9 request types (image / upscale / video / speech / voice-clone / music
//! / transcription / 3D / background-removal) and 7 result types (image /
//! video / audio / 3D / transcription / segment / voice handle) that drive
//! the `BaseProvider` + `CustomProvider` foreign-side extension surface.
//!
//! Distinct from [`crate::compute`], which exposes a parallel "job model"
//! abstraction (`TtsModel` / `SttModel` / `ImageGenModel`) with flattened
//! request/response shapes for direct dispatch. This module exposes the
//! richer typed surface that user-defined `CustomProvider` subclasses will
//! return / consume.
//!
//! ## Wire-format shape
//!
//! UniFFI doesn't expose `serde_json::Value` or `PathBuf` directly, so:
//!
//! - All `serde_json::Value` fields (`parameters`, `metadata`) cross the FFI
//!   boundary as JSON-encoded `String`. The empty string and `"null"` both
//!   round-trip to [`serde_json::Value::Null`].
//! - [`MediaSource`](blazen_llm::types::MediaSource) (in
//!   [`TranscriptionRequest::audio_source`]) crosses as an
//!   `Option<String>` JSON encoding of the full tagged enum. This preserves
//!   the 5 upstream variants (`url`, `base64`, `file`, `provider_file`,
//!   `handle`) losslessly.
//! - [`MediaType`](blazen_llm::MediaType) crosses as a `String` MIME type
//!   (`"image/png"`, `"video/mp4"`, ...). Unknown MIMEs round-trip via
//!   [`MediaType::Other`].
//! - `PathBuf` is exposed as `String` (UTF-8 lossy on the reverse direction).
//!
//! Other primitives (`u32` / `f32` / `f64` / `bool` / `Option<T>` / `Vec<T>`)
//! map 1:1 to their UniFFI equivalents.

use blazen_llm::compute::{
    AudioResult as CoreAudioResult, ImageResult as CoreImageResult,
    ThreeDResult as CoreThreeDResult, TranscriptionResult as CoreTranscriptionResult,
    TranscriptionSegment as CoreTranscriptionSegment, VideoResult as CoreVideoResult,
    VoiceHandle as CoreVoiceHandle,
};
use blazen_llm::compute::{
    BackgroundRemovalRequest as CoreBackgroundRemovalRequest, ImageRequest as CoreImageRequest,
    MusicRequest as CoreMusicRequest, SpeechRequest as CoreSpeechRequest,
    ThreeDRequest as CoreThreeDRequest, TranscriptionRequest as CoreTranscriptionRequest,
    UpscaleRequest as CoreUpscaleRequest, VideoRequest as CoreVideoRequest,
    VoiceCloneRequest as CoreVoiceCloneRequest,
};
use blazen_llm::media::{
    Generated3DModel as CoreGenerated3DModel, GeneratedAudio as CoreGeneratedAudio,
    GeneratedImage as CoreGeneratedImage, GeneratedVideo as CoreGeneratedVideo,
    MediaOutput as CoreMediaOutput, MediaType as CoreMediaType,
};
use blazen_llm::types::{RequestTiming as CoreRequestTiming, TokenUsage as CoreTokenUsage};

use crate::llm::TokenUsage;

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// Encode a `serde_json::Value` as a string for the FFI boundary.
///
/// Returns the empty string on serialization failure (defensive — the input
/// is itself a `serde_json::Value`, so serialization can only fail in
/// pathological cases like deeply-recursive externally-owned types).
fn json_to_string(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

/// Decode a JSON string from the FFI boundary back into a `serde_json::Value`.
///
/// Empty / malformed input round-trips to [`serde_json::Value::Null`], matching
/// the upstream `#[serde(default)]` behavior on the source fields.
fn string_to_json(s: &str) -> serde_json::Value {
    if s.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
}

/// Encode the inner [`CoreTokenUsage`] from the wire-format [`TokenUsage`].
///
/// The wire format only carries 5 of the 7 upstream fields (`audio_input_tokens`
/// and `audio_output_tokens` are not exposed via UniFFI — see `llm.rs`'s
/// `TokenUsage` definition for context). Round-tripping a wire `TokenUsage`
/// through this helper therefore zeroes those two audio counters.
#[allow(clippy::cast_possible_truncation)]
fn token_usage_to_core(u: TokenUsage) -> CoreTokenUsage {
    CoreTokenUsage {
        prompt_tokens: u.prompt_tokens.min(u64::from(u32::MAX)) as u32,
        completion_tokens: u.completion_tokens.min(u64::from(u32::MAX)) as u32,
        total_tokens: u.total_tokens.min(u64::from(u32::MAX)) as u32,
        cached_input_tokens: u.cached_input_tokens.min(u64::from(u32::MAX)) as u32,
        reasoning_tokens: u.reasoning_tokens.min(u64::from(u32::MAX)) as u32,
        audio_input_tokens: 0,
        audio_output_tokens: 0,
    }
}

// ---------------------------------------------------------------------------
// RequestTiming
// ---------------------------------------------------------------------------

/// Timing metadata for a compute request.
///
/// All three counters are optional; a `None` value means "the provider did
/// not report this timing breakdown" rather than zero.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RequestTiming {
    pub queue_ms: Option<u64>,
    pub execution_ms: Option<u64>,
    pub total_ms: Option<u64>,
}

impl From<CoreRequestTiming> for RequestTiming {
    fn from(t: CoreRequestTiming) -> Self {
        Self {
            queue_ms: t.queue_ms,
            execution_ms: t.execution_ms,
            total_ms: t.total_ms,
        }
    }
}

impl From<RequestTiming> for CoreRequestTiming {
    fn from(t: RequestTiming) -> Self {
        Self {
            queue_ms: t.queue_ms,
            execution_ms: t.execution_ms,
            total_ms: t.total_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// MediaOutput
// ---------------------------------------------------------------------------

/// A single piece of generated media (image, video, audio, 3D, ...).
///
/// At least one of `url`, `base64`, or `raw_content` is populated.
/// `media_type` is the canonical MIME string (`"image/png"`, `"video/mp4"`,
/// `"model/gltf-binary"`, ...). Unknown MIMEs map back to
/// [`CoreMediaType::Other`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct MediaOutput {
    pub url: Option<String>,
    pub base64: Option<String>,
    pub raw_content: Option<String>,
    /// MIME type string (e.g. `"image/png"`).
    pub media_type: String,
    pub file_size: Option<u64>,
    /// JSON-encoded arbitrary metadata. Empty string round-trips to `null`.
    pub metadata: String,
}

impl From<CoreMediaOutput> for MediaOutput {
    fn from(m: CoreMediaOutput) -> Self {
        Self {
            url: m.url,
            base64: m.base64,
            raw_content: m.raw_content,
            media_type: m.media_type.mime().to_string(),
            file_size: m.file_size,
            metadata: json_to_string(&m.metadata),
        }
    }
}

impl From<MediaOutput> for CoreMediaOutput {
    fn from(m: MediaOutput) -> Self {
        Self {
            url: m.url,
            base64: m.base64,
            raw_content: m.raw_content,
            media_type: media_type_from_mime(&m.media_type),
            file_size: m.file_size,
            metadata: string_to_json(&m.metadata),
        }
    }
}

/// Reverse-lookup a [`CoreMediaType`] from its MIME string.
///
/// Falls back to [`CoreMediaType::Other`] for unrecognised MIMEs.
fn media_type_from_mime(mime: &str) -> CoreMediaType {
    match mime {
        "image/png" => CoreMediaType::Png,
        "image/jpeg" => CoreMediaType::Jpeg,
        "image/webp" => CoreMediaType::WebP,
        "image/gif" => CoreMediaType::Gif,
        "image/svg+xml" => CoreMediaType::Svg,
        "image/bmp" => CoreMediaType::Bmp,
        "image/tiff" => CoreMediaType::Tiff,
        "image/avif" => CoreMediaType::Avif,
        "image/x-icon" => CoreMediaType::Ico,
        "video/mp4" => CoreMediaType::Mp4,
        "video/webm" => CoreMediaType::WebM,
        "video/quicktime" => CoreMediaType::Mov,
        "video/x-msvideo" => CoreMediaType::Avi,
        "video/x-matroska" => CoreMediaType::Mkv,
        "audio/mpeg" => CoreMediaType::Mp3,
        "audio/wav" => CoreMediaType::Wav,
        "audio/ogg" => CoreMediaType::Ogg,
        "audio/flac" => CoreMediaType::Flac,
        "audio/aac" => CoreMediaType::Aac,
        "audio/mp4" => CoreMediaType::M4a,
        "audio/webm" => CoreMediaType::WebmAudio,
        "model/gltf-binary" => CoreMediaType::Glb,
        "model/gltf+json" => CoreMediaType::Gltf,
        "model/obj" => CoreMediaType::Obj,
        "application/octet-stream-fbx" => CoreMediaType::Fbx,
        "model/vnd.usdz+zip" => CoreMediaType::Usdz,
        "model/stl" => CoreMediaType::Stl,
        "model/ply" => CoreMediaType::Ply,
        "application/pdf" => CoreMediaType::Pdf,
        other => CoreMediaType::Other {
            mime: other.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Generated media wrappers
// ---------------------------------------------------------------------------

/// A single generated image with optional dimension metadata.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GeneratedImage {
    pub media: MediaOutput,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl From<CoreGeneratedImage> for GeneratedImage {
    fn from(g: CoreGeneratedImage) -> Self {
        Self {
            media: g.media.into(),
            width: g.width,
            height: g.height,
        }
    }
}

impl From<GeneratedImage> for CoreGeneratedImage {
    fn from(g: GeneratedImage) -> Self {
        Self {
            media: g.media.into(),
            width: g.width,
            height: g.height,
        }
    }
}

/// A single generated video with optional metadata.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GeneratedVideo {
    pub media: MediaOutput,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_seconds: Option<f32>,
    pub fps: Option<f32>,
}

impl From<CoreGeneratedVideo> for GeneratedVideo {
    fn from(g: CoreGeneratedVideo) -> Self {
        Self {
            media: g.media.into(),
            width: g.width,
            height: g.height,
            duration_seconds: g.duration_seconds,
            fps: g.fps,
        }
    }
}

impl From<GeneratedVideo> for CoreGeneratedVideo {
    fn from(g: GeneratedVideo) -> Self {
        Self {
            media: g.media.into(),
            width: g.width,
            height: g.height,
            duration_seconds: g.duration_seconds,
            fps: g.fps,
        }
    }
}

/// A single generated audio clip with optional metadata.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GeneratedAudio {
    pub media: MediaOutput,
    pub duration_seconds: Option<f32>,
    pub sample_rate: Option<u32>,
    /// Number of channels, if known. UniFFI doesn't have a `u8` distinct from
    /// `u32`, so the upstream `Option<u8>` widens to `Option<u32>`.
    pub channels: Option<u32>,
}

impl From<CoreGeneratedAudio> for GeneratedAudio {
    fn from(g: CoreGeneratedAudio) -> Self {
        Self {
            media: g.media.into(),
            duration_seconds: g.duration_seconds,
            sample_rate: g.sample_rate,
            channels: g.channels.map(u32::from),
        }
    }
}

impl From<GeneratedAudio> for CoreGeneratedAudio {
    fn from(g: GeneratedAudio) -> Self {
        Self {
            media: g.media.into(),
            duration_seconds: g.duration_seconds,
            sample_rate: g.sample_rate,
            // Saturating cast — channels > 255 is nonsensical for audio.
            channels: g.channels.map(|c| u8::try_from(c).unwrap_or(u8::MAX)),
        }
    }
}

/// A single generated 3D model with optional mesh metadata.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Generated3DModel {
    pub media: MediaOutput,
    pub vertex_count: Option<u64>,
    pub face_count: Option<u64>,
    pub has_textures: bool,
    pub has_animations: bool,
}

impl From<CoreGenerated3DModel> for Generated3DModel {
    fn from(g: CoreGenerated3DModel) -> Self {
        Self {
            media: g.media.into(),
            vertex_count: g.vertex_count,
            face_count: g.face_count,
            has_textures: g.has_textures,
            has_animations: g.has_animations,
        }
    }
}

impl From<Generated3DModel> for CoreGenerated3DModel {
    fn from(g: Generated3DModel) -> Self {
        Self {
            media: g.media.into(),
            vertex_count: g.vertex_count,
            face_count: g.face_count,
            has_textures: g.has_textures,
            has_animations: g.has_animations,
        }
    }
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Request to generate images from a text prompt.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_images: Option<u32>,
    pub model: Option<String>,
    /// JSON-encoded provider-specific parameters. Empty string is `null`.
    pub parameters: String,
}

impl From<CoreImageRequest> for ImageRequest {
    fn from(r: CoreImageRequest) -> Self {
        Self {
            prompt: r.prompt,
            negative_prompt: r.negative_prompt,
            width: r.width,
            height: r.height,
            num_images: r.num_images,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<ImageRequest> for CoreImageRequest {
    fn from(r: ImageRequest) -> Self {
        Self {
            prompt: r.prompt,
            negative_prompt: r.negative_prompt,
            width: r.width,
            height: r.height,
            num_images: r.num_images,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to upscale an image.
#[derive(Debug, Clone, uniffi::Record)]
pub struct UpscaleRequest {
    pub image_url: String,
    pub scale: f32,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreUpscaleRequest> for UpscaleRequest {
    fn from(r: CoreUpscaleRequest) -> Self {
        Self {
            image_url: r.image_url,
            scale: r.scale,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<UpscaleRequest> for CoreUpscaleRequest {
    fn from(r: UpscaleRequest) -> Self {
        Self {
            image_url: r.image_url,
            scale: r.scale,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to generate a video.
#[derive(Debug, Clone, uniffi::Record)]
pub struct VideoRequest {
    pub prompt: String,
    pub image_url: Option<String>,
    pub duration_seconds: Option<f32>,
    pub negative_prompt: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreVideoRequest> for VideoRequest {
    fn from(r: CoreVideoRequest) -> Self {
        Self {
            prompt: r.prompt,
            image_url: r.image_url,
            duration_seconds: r.duration_seconds,
            negative_prompt: r.negative_prompt,
            width: r.width,
            height: r.height,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<VideoRequest> for CoreVideoRequest {
    fn from(r: VideoRequest) -> Self {
        Self {
            prompt: r.prompt,
            image_url: r.image_url,
            duration_seconds: r.duration_seconds,
            negative_prompt: r.negative_prompt,
            width: r.width,
            height: r.height,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to generate speech from text (TTS).
#[derive(Debug, Clone, uniffi::Record)]
pub struct SpeechRequest {
    pub text: String,
    pub voice: Option<String>,
    pub voice_url: Option<String>,
    pub language: Option<String>,
    pub speed: Option<f32>,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreSpeechRequest> for SpeechRequest {
    fn from(r: CoreSpeechRequest) -> Self {
        Self {
            text: r.text,
            voice: r.voice,
            voice_url: r.voice_url,
            language: r.language,
            speed: r.speed,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<SpeechRequest> for CoreSpeechRequest {
    fn from(r: SpeechRequest) -> Self {
        Self {
            text: r.text,
            voice: r.voice,
            voice_url: r.voice_url,
            language: r.language,
            speed: r.speed,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to clone a voice from one or more reference audio clips.
#[derive(Debug, Clone, uniffi::Record)]
pub struct VoiceCloneRequest {
    pub name: String,
    pub reference_urls: Vec<String>,
    pub language: Option<String>,
    pub description: Option<String>,
    pub parameters: String,
}

impl From<CoreVoiceCloneRequest> for VoiceCloneRequest {
    fn from(r: CoreVoiceCloneRequest) -> Self {
        Self {
            name: r.name,
            reference_urls: r.reference_urls,
            language: r.language,
            description: r.description,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<VoiceCloneRequest> for CoreVoiceCloneRequest {
    fn from(r: VoiceCloneRequest) -> Self {
        Self {
            name: r.name,
            reference_urls: r.reference_urls,
            language: r.language,
            description: r.description,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to generate music or sound effects.
#[derive(Debug, Clone, uniffi::Record)]
pub struct MusicRequest {
    pub prompt: String,
    pub duration_seconds: Option<f32>,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreMusicRequest> for MusicRequest {
    fn from(r: CoreMusicRequest) -> Self {
        Self {
            prompt: r.prompt,
            duration_seconds: r.duration_seconds,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<MusicRequest> for CoreMusicRequest {
    fn from(r: MusicRequest) -> Self {
        Self {
            prompt: r.prompt,
            duration_seconds: r.duration_seconds,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to transcribe audio to text.
///
/// `audio_source_json` carries the full upstream
/// [`MediaSource`](blazen_llm::types::MediaSource) as a JSON-encoded string
/// (one of `{"type":"url","url":"..."}`, `{"type":"base64","data":"..."}`,
/// `{"type":"file","path":"..."}`, `{"type":"provider_file",...}`,
/// `{"type":"handle",...}`). `None` falls back to using `audio_url`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TranscriptionRequest {
    pub audio_url: String,
    /// JSON-encoded `MediaSource`. When `Some`, takes precedence over
    /// `audio_url`. See module docs for the wire shape.
    pub audio_source_json: Option<String>,
    pub language: Option<String>,
    pub diarize: bool,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreTranscriptionRequest> for TranscriptionRequest {
    fn from(r: CoreTranscriptionRequest) -> Self {
        Self {
            audio_url: r.audio_url,
            audio_source_json: r
                .audio_source
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap_or_default()),
            language: r.language,
            diarize: r.diarize,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<TranscriptionRequest> for CoreTranscriptionRequest {
    fn from(r: TranscriptionRequest) -> Self {
        let audio_source = r
            .audio_source_json
            .as_deref()
            .filter(|s| !s.is_empty())
            .and_then(|s| serde_json::from_str(s).ok());
        Self {
            audio_url: r.audio_url,
            audio_source,
            language: r.language,
            diarize: r.diarize,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request to generate a 3D model.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ThreeDRequest {
    pub prompt: String,
    pub image_url: Option<String>,
    pub format: Option<String>,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreThreeDRequest> for ThreeDRequest {
    fn from(r: CoreThreeDRequest) -> Self {
        Self {
            prompt: r.prompt,
            image_url: r.image_url,
            format: r.format,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<ThreeDRequest> for CoreThreeDRequest {
    fn from(r: ThreeDRequest) -> Self {
        Self {
            prompt: r.prompt,
            image_url: r.image_url,
            format: r.format,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

/// Request for background removal on an existing image.
#[derive(Debug, Clone, uniffi::Record)]
pub struct BackgroundRemovalRequest {
    pub image_url: String,
    pub model: Option<String>,
    pub parameters: String,
}

impl From<CoreBackgroundRemovalRequest> for BackgroundRemovalRequest {
    fn from(r: CoreBackgroundRemovalRequest) -> Self {
        Self {
            image_url: r.image_url,
            model: r.model,
            parameters: json_to_string(&r.parameters),
        }
    }
}

impl From<BackgroundRemovalRequest> for CoreBackgroundRemovalRequest {
    fn from(r: BackgroundRemovalRequest) -> Self {
        Self {
            image_url: r.image_url,
            model: r.model,
            parameters: string_to_json(&r.parameters),
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of an image generation or upscale operation.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ImageResult {
    pub images: Vec<GeneratedImage>,
    pub timing: RequestTiming,
    pub cost: Option<f64>,
    pub usage: Option<TokenUsage>,
    pub image_count: u32,
    pub metadata: String,
}

impl From<CoreImageResult> for ImageResult {
    fn from(r: CoreImageResult) -> Self {
        Self {
            images: r.images.into_iter().map(GeneratedImage::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(TokenUsage::from),
            image_count: r.image_count,
            metadata: json_to_string(&r.metadata),
        }
    }
}

impl From<ImageResult> for CoreImageResult {
    fn from(r: ImageResult) -> Self {
        Self {
            images: r.images.into_iter().map(CoreGeneratedImage::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(token_usage_to_core),
            image_count: r.image_count,
            metadata: string_to_json(&r.metadata),
        }
    }
}

/// Result of a video generation operation.
#[derive(Debug, Clone, uniffi::Record)]
pub struct VideoResult {
    pub videos: Vec<GeneratedVideo>,
    pub timing: RequestTiming,
    pub cost: Option<f64>,
    pub usage: Option<TokenUsage>,
    pub video_seconds: f64,
    pub metadata: String,
}

impl From<CoreVideoResult> for VideoResult {
    fn from(r: CoreVideoResult) -> Self {
        Self {
            videos: r.videos.into_iter().map(GeneratedVideo::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(TokenUsage::from),
            video_seconds: r.video_seconds,
            metadata: json_to_string(&r.metadata),
        }
    }
}

impl From<VideoResult> for CoreVideoResult {
    fn from(r: VideoResult) -> Self {
        Self {
            videos: r.videos.into_iter().map(CoreGeneratedVideo::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(token_usage_to_core),
            video_seconds: r.video_seconds,
            metadata: string_to_json(&r.metadata),
        }
    }
}

/// Result of an audio generation or TTS operation.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AudioResult {
    pub audio: Vec<GeneratedAudio>,
    pub timing: RequestTiming,
    pub cost: Option<f64>,
    pub usage: Option<TokenUsage>,
    pub audio_seconds: f64,
    pub metadata: String,
}

impl From<CoreAudioResult> for AudioResult {
    fn from(r: CoreAudioResult) -> Self {
        Self {
            audio: r.audio.into_iter().map(GeneratedAudio::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(TokenUsage::from),
            audio_seconds: r.audio_seconds,
            metadata: json_to_string(&r.metadata),
        }
    }
}

impl From<AudioResult> for CoreAudioResult {
    fn from(r: AudioResult) -> Self {
        Self {
            audio: r.audio.into_iter().map(CoreGeneratedAudio::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(token_usage_to_core),
            audio_seconds: r.audio_seconds,
            metadata: string_to_json(&r.metadata),
        }
    }
}

/// Result of a 3D model generation operation.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ThreeDResult {
    pub models: Vec<Generated3DModel>,
    pub timing: RequestTiming,
    pub cost: Option<f64>,
    pub usage: Option<TokenUsage>,
    pub metadata: String,
}

impl From<CoreThreeDResult> for ThreeDResult {
    fn from(r: CoreThreeDResult) -> Self {
        Self {
            models: r.models.into_iter().map(Generated3DModel::from).collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(TokenUsage::from),
            metadata: json_to_string(&r.metadata),
        }
    }
}

impl From<ThreeDResult> for CoreThreeDResult {
    fn from(r: ThreeDResult) -> Self {
        Self {
            models: r
                .models
                .into_iter()
                .map(CoreGenerated3DModel::from)
                .collect(),
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(token_usage_to_core),
            metadata: string_to_json(&r.metadata),
        }
    }
}

/// A single segment within a transcription.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub speaker: Option<String>,
}

impl From<CoreTranscriptionSegment> for TranscriptionSegment {
    fn from(s: CoreTranscriptionSegment) -> Self {
        Self {
            text: s.text,
            start: s.start,
            end: s.end,
            speaker: s.speaker,
        }
    }
}

impl From<TranscriptionSegment> for CoreTranscriptionSegment {
    fn from(s: TranscriptionSegment) -> Self {
        Self {
            text: s.text,
            start: s.start,
            end: s.end,
            speaker: s.speaker,
        }
    }
}

/// Result of a transcription operation.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub timing: RequestTiming,
    pub cost: Option<f64>,
    pub usage: Option<TokenUsage>,
    pub audio_seconds: f64,
    pub metadata: String,
}

impl From<CoreTranscriptionResult> for TranscriptionResult {
    fn from(r: CoreTranscriptionResult) -> Self {
        Self {
            text: r.text,
            segments: r
                .segments
                .into_iter()
                .map(TranscriptionSegment::from)
                .collect(),
            language: r.language,
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(TokenUsage::from),
            audio_seconds: r.audio_seconds,
            metadata: json_to_string(&r.metadata),
        }
    }
}

impl From<TranscriptionResult> for CoreTranscriptionResult {
    fn from(r: TranscriptionResult) -> Self {
        Self {
            text: r.text,
            segments: r
                .segments
                .into_iter()
                .map(CoreTranscriptionSegment::from)
                .collect(),
            language: r.language,
            timing: r.timing.into(),
            cost: r.cost,
            usage: r.usage.map(token_usage_to_core),
            audio_seconds: r.audio_seconds,
            metadata: string_to_json(&r.metadata),
        }
    }
}

/// A persisted voice identifier returned by a `VoiceCloning` provider.
#[derive(Debug, Clone, uniffi::Record)]
pub struct VoiceHandle {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub language: Option<String>,
    pub description: Option<String>,
    pub metadata: String,
}

impl From<CoreVoiceHandle> for VoiceHandle {
    fn from(h: CoreVoiceHandle) -> Self {
        Self {
            id: h.id,
            name: h.name,
            provider: h.provider,
            language: h.language,
            description: h.description,
            metadata: json_to_string(&h.metadata),
        }
    }
}

impl From<VoiceHandle> for CoreVoiceHandle {
    fn from(h: VoiceHandle) -> Self {
        Self {
            id: h.id,
            name: h.name,
            provider: h.provider,
            language: h.language,
            description: h.description,
            metadata: string_to_json(&h.metadata),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_request_roundtrip() {
        let core = CoreImageRequest::new("a cat")
            .with_size(512, 768)
            .with_count(2)
            .with_negative_prompt("blurry")
            .with_model("flux-dev");
        let wire: ImageRequest = core.clone().into();
        let back: CoreImageRequest = wire.into();
        assert_eq!(back.prompt, core.prompt);
        assert_eq!(back.width, core.width);
        assert_eq!(back.height, core.height);
        assert_eq!(back.num_images, core.num_images);
        assert_eq!(back.negative_prompt, core.negative_prompt);
        assert_eq!(back.model, core.model);
    }

    #[test]
    fn transcription_request_audio_source_roundtrip_url() {
        let core = CoreTranscriptionRequest::new("https://example.com/clip.mp3")
            .with_language("en")
            .with_diarize(true);
        let wire: TranscriptionRequest = core.clone().into();
        // No audio_source on this constructor, so audio_source_json is None.
        assert!(wire.audio_source_json.is_none());
        let back: CoreTranscriptionRequest = wire.into();
        assert_eq!(back.audio_url, core.audio_url);
        assert_eq!(back.language, core.language);
        assert_eq!(back.diarize, core.diarize);
        assert!(back.audio_source.is_none());
    }

    #[test]
    fn transcription_request_audio_source_roundtrip_file() {
        let core = CoreTranscriptionRequest::from_file("/tmp/audio.wav");
        let wire: TranscriptionRequest = core.clone().into();
        // File variant should serialize to a tagged JSON string.
        let json = wire.audio_source_json.as_deref().unwrap();
        assert!(json.contains("\"file\""));
        assert!(json.contains("/tmp/audio.wav"));
        let back: CoreTranscriptionRequest = wire.into();
        assert!(back.audio_source.is_some());
    }

    #[test]
    fn media_output_roundtrip_known_mime() {
        let core = CoreMediaOutput::from_url("https://example.com/cat.png", CoreMediaType::Png);
        let wire: MediaOutput = core.clone().into();
        assert_eq!(wire.media_type, "image/png");
        let back: CoreMediaOutput = wire.into();
        assert_eq!(back.media_type, CoreMediaType::Png);
        assert_eq!(back.url, core.url);
    }

    #[test]
    fn media_output_roundtrip_unknown_mime() {
        let core = CoreMediaOutput {
            url: Some("https://example.com/x".to_string()),
            base64: None,
            raw_content: None,
            media_type: CoreMediaType::Other {
                mime: "application/x-custom".to_string(),
            },
            file_size: None,
            metadata: serde_json::Value::Null,
        };
        let wire: MediaOutput = core.into();
        assert_eq!(wire.media_type, "application/x-custom");
        let back: CoreMediaOutput = wire.into();
        match back.media_type {
            CoreMediaType::Other { mime } => assert_eq!(mime, "application/x-custom"),
            other => panic!("expected Other variant, got {other:?}"),
        }
    }

    #[test]
    fn request_timing_roundtrip() {
        let core = CoreRequestTiming {
            queue_ms: Some(10),
            execution_ms: Some(200),
            total_ms: Some(210),
        };
        let wire: RequestTiming = core.clone().into();
        let back: CoreRequestTiming = wire.into();
        assert_eq!(back.queue_ms, core.queue_ms);
        assert_eq!(back.execution_ms, core.execution_ms);
        assert_eq!(back.total_ms, core.total_ms);
    }

    #[test]
    fn parameters_empty_string_round_trips_to_null() {
        let req = ImageRequest {
            prompt: "x".to_string(),
            negative_prompt: None,
            width: None,
            height: None,
            num_images: None,
            model: None,
            parameters: String::new(),
        };
        let core: CoreImageRequest = req.into();
        assert!(core.parameters.is_null());
    }

    #[test]
    fn voice_handle_roundtrip() {
        let core = CoreVoiceHandle {
            id: "v1".to_string(),
            name: "Alice".to_string(),
            provider: "elevenlabs".to_string(),
            language: Some("en".to_string()),
            description: None,
            metadata: serde_json::json!({"stability": 0.5}),
        };
        let wire: VoiceHandle = core.clone().into();
        let back: CoreVoiceHandle = wire.into();
        assert_eq!(back.id, core.id);
        assert_eq!(back.name, core.name);
        assert_eq!(back.provider, core.provider);
        assert_eq!(back.language, core.language);
        // Metadata round-trip through JSON preserves the object.
        assert_eq!(back.metadata, core.metadata);
    }
}
