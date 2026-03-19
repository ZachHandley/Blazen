//! Compute request types for image, video, audio, transcription, and 3D generation.

use napi_derive::napi;

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
