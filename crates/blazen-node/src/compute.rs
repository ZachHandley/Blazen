//! JavaScript bindings for compute request/result types and media types.
//!
//! Exposes all typed request and result structs from [`blazen_llm::compute`]
//! and media types from [`blazen_llm::media`] as `#[napi(object)]` interfaces
//! for TypeScript consumption.

use napi_derive::napi;

// ---------------------------------------------------------------------------
// Compute request types
// ---------------------------------------------------------------------------

/// Request to generate images from a text prompt.
#[napi(object)]
pub struct JsImageRequest {
    /// The text prompt describing the desired image.
    pub prompt: String,
    /// Negative prompt (things to avoid in the image).
    #[napi(js_name = "negativePrompt")]
    pub negative_prompt: Option<String>,
    /// Desired image width in pixels.
    pub width: Option<u32>,
    /// Desired image height in pixels.
    pub height: Option<u32>,
    /// Number of images to generate.
    #[napi(js_name = "numImages")]
    pub num_images: Option<u32>,
    /// Model override (provider-specific model identifier).
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Request to upscale an image.
#[napi(object)]
pub struct JsUpscaleRequest {
    /// URL of the image to upscale.
    #[napi(js_name = "imageUrl")]
    pub image_url: String,
    /// Scale factor (e.g., 2.0 for 2x, 4.0 for 4x).
    pub scale: f64,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Request to generate a video.
#[napi(object)]
pub struct JsVideoRequest {
    /// Text prompt describing the desired video.
    pub prompt: String,
    /// Source image URL for image-to-video generation.
    #[napi(js_name = "imageUrl")]
    pub image_url: Option<String>,
    /// Desired duration in seconds.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
    /// Negative prompt (things to avoid).
    #[napi(js_name = "negativePrompt")]
    pub negative_prompt: Option<String>,
    /// Desired video width in pixels.
    pub width: Option<u32>,
    /// Desired video height in pixels.
    pub height: Option<u32>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Request to generate speech from text (TTS).
#[napi(object)]
pub struct JsSpeechRequest {
    /// The text to synthesize into speech.
    pub text: String,
    /// Voice identifier (provider-specific).
    pub voice: Option<String>,
    /// URL to a reference voice sample for voice cloning.
    #[napi(js_name = "voiceUrl")]
    pub voice_url: Option<String>,
    /// Language code (e.g. "en", "fr", "ja").
    pub language: Option<String>,
    /// Speech speed multiplier (1.0 = normal).
    pub speed: Option<f64>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Request to generate music or sound effects.
#[napi(object)]
pub struct JsMusicRequest {
    /// Text prompt describing the desired audio.
    pub prompt: String,
    /// Desired duration in seconds.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Request to transcribe audio to text.
#[napi(object)]
pub struct JsTranscriptionRequest {
    /// URL of the audio file to transcribe.
    #[napi(js_name = "audioUrl")]
    pub audio_url: String,
    /// Language hint (e.g. "en", "fr").
    pub language: Option<String>,
    /// Whether to perform speaker diarization.
    pub diarize: Option<bool>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Request to generate a 3D model.
#[napi(object)]
pub struct JsThreeDRequest {
    /// Text prompt describing the desired 3D model.
    pub prompt: Option<String>,
    /// Source image URL for image-to-3D generation.
    #[napi(js_name = "imageUrl")]
    pub image_url: Option<String>,
    /// Desired output format (e.g. "glb", "obj", "usdz").
    pub format: Option<String>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Low-level compute job types
// ---------------------------------------------------------------------------

/// Input for a generic compute job.
#[napi(object)]
pub struct JsComputeRequest {
    /// The model/endpoint to run (e.g., "fal-ai/flux/dev").
    pub model: String,
    /// Input parameters as JSON (model-specific).
    pub input: serde_json::Value,
    /// Optional webhook URL for async completion notification.
    pub webhook: Option<String>,
}

/// A handle to a submitted compute job.
#[napi(object)]
pub struct JsJobHandle {
    /// Provider-assigned job/request identifier.
    pub id: String,
    /// Provider name (e.g., "fal", "replicate", "runpod").
    pub provider: String,
    /// The model/endpoint that was invoked.
    pub model: String,
    /// When the job was submitted (ISO 8601).
    #[napi(js_name = "submittedAt")]
    pub submitted_at: String,
}

/// Status of a compute job.
#[napi(string_enum)]
pub enum JsJobStatus {
    /// Job is waiting in the provider's queue.
    #[napi(value = "queued")]
    Queued,
    /// Job is currently executing.
    #[napi(value = "running")]
    Running,
    /// Job completed successfully.
    #[napi(value = "completed")]
    Completed,
    /// Job failed with an error.
    #[napi(value = "failed")]
    Failed,
    /// Job was cancelled.
    #[napi(value = "cancelled")]
    Cancelled,
}

/// Result of a completed compute job.
#[napi(object)]
pub struct JsComputeResult {
    /// The job handle that produced this result, if available.
    pub job: Option<JsJobHandle>,
    /// Output data (model-specific JSON).
    pub output: serde_json::Value,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Raw provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// Timing breakdown for a compute request.
#[napi(object)]
pub struct JsComputeTiming {
    /// Time spent waiting in queue, in milliseconds.
    #[napi(js_name = "queueMs")]
    pub queue_ms: Option<i64>,
    /// Time spent executing, in milliseconds.
    #[napi(js_name = "executionMs")]
    pub execution_ms: Option<i64>,
    /// Total wall-clock time, in milliseconds.
    #[napi(js_name = "totalMs")]
    pub total_ms: Option<i64>,
}

// ---------------------------------------------------------------------------
// Media output types
// ---------------------------------------------------------------------------

/// A single piece of generated media content.
#[napi(object)]
pub struct JsMediaOutput {
    /// URL where the media can be downloaded.
    pub url: Option<String>,
    /// Base64-encoded media data.
    pub base64: Option<String>,
    /// Raw text content for text-based formats (SVG, OBJ, GLTF JSON).
    #[napi(js_name = "rawContent")]
    pub raw_content: Option<String>,
    /// The MIME type of the media (e.g. "image/png", "video/mp4").
    #[napi(js_name = "mediaType")]
    pub media_type: String,
    /// File size in bytes, if known.
    #[napi(js_name = "fileSize")]
    pub file_size: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// A single generated image with optional dimension metadata.
#[napi(object)]
pub struct JsGeneratedImage {
    /// The image media output.
    pub media: JsMediaOutput,
    /// Image width in pixels, if known.
    pub width: Option<u32>,
    /// Image height in pixels, if known.
    pub height: Option<u32>,
}

/// A single generated video with optional metadata.
#[napi(object)]
pub struct JsGeneratedVideo {
    /// The video media output.
    pub media: JsMediaOutput,
    /// Video width in pixels, if known.
    pub width: Option<u32>,
    /// Video height in pixels, if known.
    pub height: Option<u32>,
    /// Duration in seconds, if known.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
    /// Frames per second, if known.
    pub fps: Option<f64>,
}

/// A single generated audio clip with optional metadata.
#[napi(object)]
pub struct JsGeneratedAudio {
    /// The audio media output.
    pub media: JsMediaOutput,
    /// Duration in seconds, if known.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
    /// Sample rate in Hz, if known.
    #[napi(js_name = "sampleRate")]
    pub sample_rate: Option<u32>,
    /// Number of audio channels, if known.
    pub channels: Option<u32>,
}

/// A single generated 3D model with optional mesh metadata.
#[napi(object)]
pub struct JsGenerated3DModel {
    /// The 3D model media output.
    pub media: JsMediaOutput,
    /// Total vertex count, if known.
    #[napi(js_name = "vertexCount")]
    pub vertex_count: Option<f64>,
    /// Total face/triangle count, if known.
    #[napi(js_name = "faceCount")]
    pub face_count: Option<f64>,
    /// Whether the model includes texture data.
    #[napi(js_name = "hasTextures")]
    pub has_textures: bool,
    /// Whether the model includes animation data.
    #[napi(js_name = "hasAnimations")]
    pub has_animations: bool,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of an image generation or upscale operation.
#[napi(object)]
pub struct JsImageResult {
    /// The generated or upscaled images.
    pub images: Vec<JsGeneratedImage>,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// Result of a video generation operation.
#[napi(object)]
pub struct JsVideoResult {
    /// The generated videos.
    pub videos: Vec<JsGeneratedVideo>,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// Result of an audio generation or TTS operation.
#[napi(object)]
pub struct JsAudioResult {
    /// The generated audio clips.
    pub audio: Vec<JsGeneratedAudio>,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// A single segment within a transcription.
#[napi(object)]
pub struct JsTranscriptionSegment {
    /// The transcribed text for this segment.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Speaker label, if diarization was enabled.
    pub speaker: Option<String>,
}

/// Result of a transcription operation.
#[napi(object)]
pub struct JsTranscriptionResult {
    /// The full transcribed text.
    pub text: String,
    /// Time-aligned segments, if available.
    pub segments: Vec<JsTranscriptionSegment>,
    /// Detected or specified language code (e.g. "en", "fr").
    pub language: Option<String>,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// Result of a 3D model generation operation.
#[napi(object)]
pub struct JsThreeDResult {
    /// The generated 3D models.
    pub models: Vec<JsGenerated3DModel>,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Media type constants
// ---------------------------------------------------------------------------

/// Returns an object mapping friendly names to MIME type strings.
///
/// ```typescript
/// import { mediaTypes } from 'blazen';
///
/// const types = mediaTypes();
/// console.log(types.png);  // "image/png"
/// console.log(types.mp4);  // "video/mp4"
/// ```
#[napi(js_name = "mediaTypes")]
#[must_use]
pub fn media_types() -> JsMediaTypeMap {
    JsMediaTypeMap {
        // Images
        png: "image/png".to_owned(),
        jpeg: "image/jpeg".to_owned(),
        webp: "image/webp".to_owned(),
        gif: "image/gif".to_owned(),
        svg: "image/svg+xml".to_owned(),
        bmp: "image/bmp".to_owned(),
        tiff: "image/tiff".to_owned(),
        avif: "image/avif".to_owned(),
        ico: "image/x-icon".to_owned(),
        // Video
        mp4: "video/mp4".to_owned(),
        webm: "video/webm".to_owned(),
        mov: "video/quicktime".to_owned(),
        avi: "video/x-msvideo".to_owned(),
        mkv: "video/x-matroska".to_owned(),
        // Audio
        mp3: "audio/mpeg".to_owned(),
        wav: "audio/wav".to_owned(),
        ogg: "audio/ogg".to_owned(),
        flac: "audio/flac".to_owned(),
        aac: "audio/aac".to_owned(),
        m4a: "audio/mp4".to_owned(),
        // 3D Models
        glb: "model/gltf-binary".to_owned(),
        gltf: "model/gltf+json".to_owned(),
        obj: "model/obj".to_owned(),
        fbx: "application/octet-stream".to_owned(),
        usdz: "model/vnd.usdz+zip".to_owned(),
        stl: "model/stl".to_owned(),
        ply: "application/x-ply".to_owned(),
        // Documents
        pdf: "application/pdf".to_owned(),
    }
}

/// Map of friendly format names to their MIME type strings.
#[napi(object)]
pub struct JsMediaTypeMap {
    // Images
    pub png: String,
    pub jpeg: String,
    pub webp: String,
    pub gif: String,
    pub svg: String,
    pub bmp: String,
    pub tiff: String,
    pub avif: String,
    pub ico: String,
    // Video
    pub mp4: String,
    pub webm: String,
    pub mov: String,
    pub avi: String,
    pub mkv: String,
    // Audio
    pub mp3: String,
    pub wav: String,
    pub ogg: String,
    pub flac: String,
    pub aac: String,
    pub m4a: String,
    // 3D Models
    pub glb: String,
    pub gltf: String,
    pub obj: String,
    pub fbx: String,
    pub usdz: String,
    pub stl: String,
    pub ply: String,
    // Documents
    pub pdf: String,
}
