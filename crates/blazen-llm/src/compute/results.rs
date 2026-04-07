//! Typed result types for media generation operations.

use serde::{Deserialize, Serialize};

use crate::media::{Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo};
use crate::types::RequestTiming;

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
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}
