//! Error type for the [`SttBackend`](crate::SttBackend) trait surface and
//! backend implementations.

use blazen_audio::AudioError;
use thiserror::Error;

/// Errors returned by [`SttBackend`](crate::SttBackend) implementations and
/// the [`SttProvider`](crate::SttProvider) / [`DynSttProvider`](crate::DynSttProvider)
/// wrappers.
///
/// Each variant mirrors a class of failure that any speech-to-text backend
/// can produce. Backend-specific error types (e.g. `whisper_rs` errors) are
/// flattened into one of these variants with their own `Display` text
/// preserved, so the public surface does not leak engine types.
#[derive(Debug, Error)]
pub enum SttError {
    /// The selected backend is not available in this build (e.g. the
    /// `whispercpp` feature is disabled but the caller requested the
    /// whisper.cpp backend).
    #[error("STT engine not available: {0}")]
    EngineNotAvailable(String),

    /// A required option was missing or has an invalid value.
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// Loading model weights (download, file open, format parse) failed.
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Decoding / inference / segment retrieval failed at runtime.
    #[error("transcription failed: {0}")]
    Transcription(String),

    /// The capability requested (streaming, diarization, etc.) is not
    /// supported by the active backend.
    #[error("capability not supported: {0}")]
    Unsupported(String),

    /// An I/O failure while reading an audio file or model file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<SttError> for AudioError {
    fn from(err: SttError) -> Self {
        match err {
            SttError::EngineNotAvailable(msg) => Self::Backend(format!("stt engine: {msg}")),
            SttError::InvalidOptions(msg) => Self::InvalidInput(msg),
            SttError::ModelLoad(msg) => Self::Backend(format!("stt model load: {msg}")),
            SttError::Transcription(msg) => Self::Backend(format!("stt transcribe: {msg}")),
            SttError::Unsupported(msg) => Self::Unsupported(msg),
            SttError::Io(e) => Self::Io(e),
        }
    }
}
