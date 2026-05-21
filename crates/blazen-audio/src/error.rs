//! Capability-agnostic error type for audio backends.

use thiserror::Error;

/// Errors that can be returned by any [`AudioBackend`](crate::AudioBackend).
///
/// Each variant is intentionally generic so per-engine crates can map their
/// engine-specific failures (e.g. whisper.cpp `WhisperError`, candle
/// `candle_core::Error`, openai `reqwest::Error`) into one of these without
/// leaking engine types into the shared surface.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The backend was used before [`AudioBackend::load`](crate::AudioBackend::load)
    /// completed successfully.
    #[error("backend not loaded: {0}")]
    NotLoaded(String),

    /// The caller-supplied request was structurally invalid (e.g. empty
    /// text for a TTS request, zero-length audio for a transcription).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The capability requested (e.g. voice cloning, music generation,
    /// streaming transcription) is not supported by this backend.
    #[error("capability not supported: {0}")]
    Unsupported(String),

    /// A backend-internal failure (engine-specific). The wrapped string
    /// should already be human-readable.
    #[error("backend error: {0}")]
    Backend(String),

    /// An I/O failure while reading or writing audio bytes / files.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// A `serde_json` failure while encoding or decoding request /
    /// response payloads (the `parameters` field is a `serde_json::Value`).
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}
