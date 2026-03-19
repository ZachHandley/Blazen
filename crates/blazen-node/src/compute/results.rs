//! Compute result types for image, video, audio, transcription, and 3D operations.

use napi_derive::napi;

use crate::types::{JsGenerated3DModel, JsGeneratedAudio, JsGeneratedImage, JsGeneratedVideo};

use super::job::JsComputeTiming;

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
