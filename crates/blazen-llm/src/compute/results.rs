//! Typed result types for media generation operations.

use serde::{Deserialize, Serialize};

use crate::media::{Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo};
use crate::types::{RequestTiming, TokenUsage};

// ---------------------------------------------------------------------------
// Image result
// ---------------------------------------------------------------------------

/// Result of an image generation or upscale operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ImageResult {
    /// The generated or upscaled images.
    pub images: Vec<GeneratedImage>,
    /// Request timing breakdown.
    pub timing: RequestTiming,
    /// Cost in USD, if reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Token usage statistics, if reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Number of images returned by this call.
    #[serde(default)]
    pub image_count: u32,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Video result
// ---------------------------------------------------------------------------

/// Result of a video generation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct VideoResult {
    /// The generated videos.
    pub videos: Vec<GeneratedVideo>,
    /// Request timing breakdown.
    pub timing: RequestTiming,
    /// Cost in USD, if reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Token usage statistics, if reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Total duration in seconds across all returned videos.
    #[serde(default)]
    pub video_seconds: f64,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Audio result
// ---------------------------------------------------------------------------

/// Result of an audio generation or TTS operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct AudioResult {
    /// The generated audio clips.
    pub audio: Vec<GeneratedAudio>,
    /// Request timing breakdown.
    pub timing: RequestTiming,
    /// Cost in USD, if reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Token usage statistics, if reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Total duration in seconds across all returned audio clips.
    #[serde(default)]
    pub audio_seconds: f64,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// 3D result
// ---------------------------------------------------------------------------

/// Result of a 3D model generation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ThreeDResult {
    /// The generated 3D models.
    pub models: Vec<Generated3DModel>,
    /// Request timing breakdown.
    pub timing: RequestTiming,
    /// Cost in USD, if reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Token usage statistics, if reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Transcription result
// ---------------------------------------------------------------------------

/// A single segment within a transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct TranscriptionSegment {
    /// The transcribed text for this segment.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Speaker label, if diarization was enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// Result of a transcription operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct TranscriptionResult {
    /// The full transcribed text.
    pub text: String,
    /// Time-aligned segments, if available.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected or specified language code (e.g. "en", "fr").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Request timing breakdown.
    pub timing: RequestTiming,
    /// Cost in USD, if reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Token usage statistics, if reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Duration in seconds of the input audio that was transcribed.
    #[serde(default)]
    pub audio_seconds: f64,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Voice handle (voice cloning result)
// ---------------------------------------------------------------------------

/// A persisted voice identifier returned by a `VoiceCloning` provider.
///
/// Can be passed as `SpeechRequest.voice` on subsequent TTS calls to use
/// the cloned voice for synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct VoiceHandle {
    /// Provider-specific voice identifier (e.g. `ElevenLabs` `voice_id`).
    pub id: String,
    /// Human-readable name for the voice.
    pub name: String,
    /// Which provider owns this voice (`"elevenlabs"`, `"zvoice"`, ...).
    pub provider: String,
    /// Optional language code if the voice is language-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_timing() -> RequestTiming {
        RequestTiming {
            queue_ms: None,
            execution_ms: None,
            total_ms: None,
        }
    }

    #[test]
    fn image_result_usage_roundtrip() {
        let result = ImageResult {
            images: Vec::new(),
            timing: empty_timing(),
            cost: None,
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                ..Default::default()
            }),
            image_count: 3,
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&result).expect("serialize ImageResult");
        let decoded: ImageResult = serde_json::from_str(&json).expect("deserialize ImageResult");
        assert_eq!(decoded.image_count, 3);
        let usage = decoded.usage.expect("usage preserved");
        assert_eq!(usage.prompt_tokens, 10);
    }

    #[test]
    fn audio_result_audio_seconds_roundtrip() {
        let result = AudioResult {
            audio: Vec::new(),
            timing: empty_timing(),
            cost: None,
            usage: Some(TokenUsage {
                prompt_tokens: 5,
                ..Default::default()
            }),
            audio_seconds: 12.5,
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&result).expect("serialize AudioResult");
        let decoded: AudioResult = serde_json::from_str(&json).expect("deserialize AudioResult");
        assert!((decoded.audio_seconds - 12.5).abs() < f64::EPSILON);
        let usage = decoded.usage.expect("usage preserved");
        assert_eq!(usage.prompt_tokens, 5);
    }

    #[test]
    fn video_result_video_seconds_roundtrip() {
        let result = VideoResult {
            videos: Vec::new(),
            timing: empty_timing(),
            cost: None,
            usage: Some(TokenUsage {
                prompt_tokens: 7,
                ..Default::default()
            }),
            video_seconds: 30.0,
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&result).expect("serialize VideoResult");
        let decoded: VideoResult = serde_json::from_str(&json).expect("deserialize VideoResult");
        assert!((decoded.video_seconds - 30.0).abs() < f64::EPSILON);
        let usage = decoded.usage.expect("usage preserved");
        assert_eq!(usage.prompt_tokens, 7);
    }

    #[test]
    fn threed_result_usage_roundtrip() {
        let result = ThreeDResult {
            models: Vec::new(),
            timing: empty_timing(),
            cost: None,
            usage: Some(TokenUsage {
                prompt_tokens: 42,
                ..Default::default()
            }),
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&result).expect("serialize ThreeDResult");
        let decoded: ThreeDResult = serde_json::from_str(&json).expect("deserialize ThreeDResult");
        let usage = decoded.usage.expect("usage preserved");
        assert_eq!(usage.prompt_tokens, 42);
    }

    #[test]
    fn transcription_result_audio_seconds_roundtrip() {
        let result = TranscriptionResult {
            text: String::new(),
            segments: Vec::new(),
            language: None,
            timing: empty_timing(),
            cost: None,
            usage: Some(TokenUsage {
                prompt_tokens: 99,
                ..Default::default()
            }),
            audio_seconds: 8.25,
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&result).expect("serialize TranscriptionResult");
        let decoded: TranscriptionResult =
            serde_json::from_str(&json).expect("deserialize TranscriptionResult");
        assert!((decoded.audio_seconds - 8.25).abs() < f64::EPSILON);
        let usage = decoded.usage.expect("usage preserved");
        assert_eq!(usage.prompt_tokens, 99);
    }

    #[test]
    fn usage_omitted_when_none() {
        let result = ImageResult {
            images: Vec::new(),
            timing: empty_timing(),
            cost: None,
            usage: None,
            image_count: 0,
            metadata: serde_json::Value::Null,
        };
        let value: serde_json::Value =
            serde_json::to_value(&result).expect("serialize ImageResult");
        let object = value.as_object().expect("object");
        assert!(
            !object.contains_key("usage"),
            "usage key should be omitted when None"
        );
    }
}
