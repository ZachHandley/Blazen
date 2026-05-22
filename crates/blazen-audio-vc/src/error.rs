//! Error type for the [`VoiceConversionBackend`](crate::VoiceConversionBackend)
//! trait surface and backend implementations.

use blazen_audio::AudioError;
use thiserror::Error;

/// Errors returned by
/// [`VoiceConversionBackend`](crate::VoiceConversionBackend) implementations.
///
/// Each variant mirrors a class of failure that any voice-conversion
/// backend can produce. Backend-specific error types (e.g. ONNX-runtime
/// or candle errors) are flattened into one of these variants with their
/// own `Display` text preserved, so the public surface does not leak
/// engine types.
#[derive(Debug, Error)]
pub enum VcError {
    /// The selected backend is not available in this build (e.g. the
    /// `rvc` feature is disabled but the caller requested the RVC
    /// backend).
    #[error("voice-conversion engine not available: {0}")]
    EngineNotAvailable(String),

    /// Loading model weights (download, file open, format parse) failed.
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Inference / decoding failed at runtime.
    #[error("voice conversion failed: {0}")]
    Conversion(String),

    /// The caller asked to convert to a target voice that is not
    /// registered with this backend.
    #[error("target voice not found: {0}")]
    VoiceNotFound(String),

    /// The capability requested (streaming conversion, voice
    /// registration, voice listing, etc.) is not supported by the
    /// active backend.
    #[error("capability not supported: {0}")]
    Unsupported(String),

    /// An I/O failure while reading an audio file or model file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<VcError> for AudioError {
    fn from(err: VcError) -> Self {
        match err {
            VcError::EngineNotAvailable(msg) => Self::Backend(format!("vc engine: {msg}")),
            VcError::ModelLoad(msg) => Self::Backend(format!("vc model load: {msg}")),
            VcError::Conversion(msg) => Self::Backend(format!("vc convert: {msg}")),
            VcError::VoiceNotFound(msg) => Self::InvalidInput(format!("target voice: {msg}")),
            VcError::Unsupported(msg) => Self::Unsupported(msg),
            VcError::Io(e) => Self::Io(e),
        }
    }
}
