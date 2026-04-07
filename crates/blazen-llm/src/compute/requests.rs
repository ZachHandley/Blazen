//! Typed request types for media generation operations.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Image
// ---------------------------------------------------------------------------

/// Request to generate images from a text prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRequest {
    /// The text prompt describing the desired image.
    pub prompt: String,
    /// Negative prompt (things to avoid in the image).
    pub negative_prompt: Option<String>,
    /// Desired image width in pixels.
    pub width: Option<u32>,
    /// Desired image height in pixels.
    pub height: Option<u32>,
    /// Number of images to generate.
    pub num_images: Option<u32>,
    /// Model override (provider-specific model identifier).
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl ImageRequest {
    /// Create a new image request with the given prompt.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            width: None,
            height: None,
            num_images: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the image dimensions.
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set the number of images to generate.
    #[must_use]
    pub fn with_count(mut self, count: u32) -> Self {
        self.num_images = Some(count);
        self
    }

    /// Set a negative prompt.
    #[must_use]
    pub fn with_negative_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(prompt.into());
        self
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Upscale
// ---------------------------------------------------------------------------

/// Request to upscale an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleRequest {
    /// URL of the image to upscale.
    pub image_url: String,
    /// Scale factor (e.g., 2.0 for 2x, 4.0 for 4x).
    pub scale: f32,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl UpscaleRequest {
    /// Create a new upscale request.
    #[must_use]
    pub fn new(image_url: impl Into<String>, scale: f32) -> Self {
        Self {
            image_url: image_url.into(),
            scale,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

/// Request to generate a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoRequest {
    /// Text prompt describing the desired video.
    pub prompt: String,
    /// Source image URL for image-to-video generation.
    pub image_url: Option<String>,
    /// Desired duration in seconds.
    pub duration_seconds: Option<f32>,
    /// Negative prompt (things to avoid).
    pub negative_prompt: Option<String>,
    /// Desired video width in pixels.
    pub width: Option<u32>,
    /// Desired video height in pixels.
    pub height: Option<u32>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl VideoRequest {
    /// Create a new text-to-video request.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            image_url: None,
            duration_seconds: None,
            negative_prompt: None,
            width: None,
            height: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Create an image-to-video request.
    #[must_use]
    pub fn for_image(image_url: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            image_url: Some(image_url.into()),
            duration_seconds: None,
            negative_prompt: None,
            width: None,
            height: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the desired duration.
    #[must_use]
    pub fn with_duration(mut self, seconds: f32) -> Self {
        self.duration_seconds = Some(seconds);
        self
    }

    /// Set the video dimensions.
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Speech (TTS)
// ---------------------------------------------------------------------------

/// Request to generate speech from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// The text to synthesize into speech.
    pub text: String,
    /// Voice identifier (provider-specific).
    pub voice: Option<String>,
    /// URL to a reference voice sample for voice cloning.
    pub voice_url: Option<String>,
    /// Language code (e.g. "en", "fr", "ja").
    pub language: Option<String>,
    /// Speech speed multiplier (1.0 = normal).
    pub speed: Option<f32>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl SpeechRequest {
    /// Create a new TTS request.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: None,
            voice_url: None,
            language: None,
            speed: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the voice identifier.
    #[must_use]
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Set a reference voice URL for voice cloning.
    #[must_use]
    pub fn with_voice_url(mut self, url: impl Into<String>) -> Self {
        self.voice_url = Some(url.into());
        self
    }

    /// Set the language code.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the speech speed multiplier.
    #[must_use]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Music / SFX
// ---------------------------------------------------------------------------

/// Request to generate music or sound effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicRequest {
    /// Text prompt describing the desired audio.
    pub prompt: String,
    /// Desired duration in seconds.
    pub duration_seconds: Option<f32>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl MusicRequest {
    /// Create a new music/SFX generation request.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            duration_seconds: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the desired duration.
    #[must_use]
    pub fn with_duration(mut self, seconds: f32) -> Self {
        self.duration_seconds = Some(seconds);
        self
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

/// Request to transcribe audio to text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    /// URL of the audio file to transcribe.
    pub audio_url: String,
    /// Language hint (e.g. "en", "fr").
    pub language: Option<String>,
    /// Whether to perform speaker diarization.
    pub diarize: bool,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl TranscriptionRequest {
    /// Create a new transcription request.
    #[must_use]
    pub fn new(audio_url: impl Into<String>) -> Self {
        Self {
            audio_url: audio_url.into(),
            language: None,
            diarize: false,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the language hint.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Enable or disable speaker diarization.
    #[must_use]
    pub fn with_diarize(mut self, diarize: bool) -> Self {
        self.diarize = diarize;
        self
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// 3D
// ---------------------------------------------------------------------------

/// Request to generate a 3D model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDRequest {
    /// Text prompt describing the desired 3D model.
    pub prompt: String,
    /// Source image URL for image-to-3D generation.
    pub image_url: Option<String>,
    /// Desired output format (e.g. "glb", "obj", "usdz").
    pub format: Option<String>,
    /// Model override.
    pub model: Option<String>,
    /// Additional provider-specific parameters.
    pub parameters: serde_json::Value,
}

impl ThreeDRequest {
    /// Create a new text-to-3D request.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            image_url: None,
            format: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Create an image-to-3D request.
    #[must_use]
    pub fn from_image(url: impl Into<String>) -> Self {
        Self {
            prompt: String::new(),
            image_url: Some(url.into()),
            format: None,
            model: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the desired output format.
    #[must_use]
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Override the model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Background removal
// ---------------------------------------------------------------------------

/// Request for background removal on an existing image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundRemovalRequest {
    /// URL of the source image.
    pub image_url: String,
    /// Optional model id override.
    pub model: Option<String>,
    /// Provider-specific parameters merged into the request body.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_request_builder() {
        let req = ImageRequest::new("a cat")
            .with_size(512, 512)
            .with_count(4)
            .with_negative_prompt("blurry")
            .with_model("flux-dev");
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.width, Some(512));
        assert_eq!(req.height, Some(512));
        assert_eq!(req.num_images, Some(4));
        assert_eq!(req.negative_prompt.as_deref(), Some("blurry"));
        assert_eq!(req.model.as_deref(), Some("flux-dev"));
    }

    #[test]
    fn upscale_request_builder() {
        let req = UpscaleRequest::new("https://example.com/img.png", 4.0).with_model("esrgan");
        assert_eq!(req.image_url, "https://example.com/img.png");
        assert!((req.scale - 4.0).abs() < f32::EPSILON);
        assert_eq!(req.model.as_deref(), Some("esrgan"));
    }

    #[test]
    fn video_request_builder() {
        let req = VideoRequest::new("a sunset timelapse")
            .with_duration(5.0)
            .with_size(1920, 1080)
            .with_model("kling");
        assert_eq!(req.prompt, "a sunset timelapse");
        assert_eq!(req.duration_seconds, Some(5.0));
        assert_eq!(req.width, Some(1920));
        assert_eq!(req.height, Some(1080));
        assert_eq!(req.model.as_deref(), Some("kling"));
        assert!(req.image_url.is_none());
    }

    #[test]
    fn video_request_for_image() {
        let req = VideoRequest::for_image("https://example.com/img.png", "animate this");
        assert_eq!(req.prompt, "animate this");
        assert_eq!(
            req.image_url.as_deref(),
            Some("https://example.com/img.png")
        );
    }

    #[test]
    fn speech_request_builder() {
        let req = SpeechRequest::new("Hello world")
            .with_voice("alloy")
            .with_voice_url("https://example.com/voice.wav")
            .with_language("en")
            .with_speed(1.5)
            .with_model("tts-1");
        assert_eq!(req.text, "Hello world");
        assert_eq!(req.voice.as_deref(), Some("alloy"));
        assert_eq!(
            req.voice_url.as_deref(),
            Some("https://example.com/voice.wav")
        );
        assert_eq!(req.language.as_deref(), Some("en"));
        assert_eq!(req.speed, Some(1.5));
        assert_eq!(req.model.as_deref(), Some("tts-1"));
    }

    #[test]
    fn music_request_builder() {
        let req = MusicRequest::new("upbeat jazz")
            .with_duration(30.0)
            .with_model("musicgen");
        assert_eq!(req.prompt, "upbeat jazz");
        assert_eq!(req.duration_seconds, Some(30.0));
        assert_eq!(req.model.as_deref(), Some("musicgen"));
    }

    #[test]
    fn transcription_request_builder() {
        let req = TranscriptionRequest::new("https://example.com/audio.mp3")
            .with_language("fr")
            .with_diarize(true)
            .with_model("whisper-v3");
        assert_eq!(req.audio_url, "https://example.com/audio.mp3");
        assert_eq!(req.language.as_deref(), Some("fr"));
        assert!(req.diarize);
        assert_eq!(req.model.as_deref(), Some("whisper-v3"));
    }

    #[test]
    fn three_d_request_builder() {
        let req = ThreeDRequest::new("a 3D cat")
            .with_format("glb")
            .with_model("triposr");
        assert_eq!(req.prompt, "a 3D cat");
        assert_eq!(req.format.as_deref(), Some("glb"));
        assert_eq!(req.model.as_deref(), Some("triposr"));
        assert!(req.image_url.is_none());
    }

    #[test]
    fn three_d_request_from_image() {
        let req = ThreeDRequest::from_image("https://example.com/img.png").with_format("obj");
        assert_eq!(
            req.image_url.as_deref(),
            Some("https://example.com/img.png")
        );
        assert_eq!(req.format.as_deref(), Some("obj"));
    }
}
