//! Capability-specific error type for codec backends.
//!
//! [`CodecError`] sits one level above [`blazen_audio::AudioError`]: it
//! converts cleanly to and from `AudioError` so the bridge layer in
//! `blazen-llm` can forward codec failures through the shared
//! audio surface, while keeping codec-specific variants
//! ([`Self::NotYetImplemented`], [`Self::EngineNotAvailable`], ...)
//! visible to in-process Rust callers.

use std::io;

use blazen_audio::AudioError;
use thiserror::Error;

/// Errors that can be returned by any [`crate::CodecBackend`].
#[derive(Debug, Error)]
pub enum CodecError {
    /// The crate was compiled without the matching backend feature
    /// (`encodec`, `dac`, `snac`, ...), so the inference stack is not
    /// linked in. Every codec call short-circuits with this error.
    #[error(
        "blazen-audio-codec was built without the matching backend feature -- \
         rebuild with `--features <backend>` (e.g. `encodec`) to enable on-device codec inference"
    )]
    EngineNotAvailable,

    /// The backend is a known slot (e.g. DAC, SNAC) but its real
    /// implementation has not landed yet. Carries a detailed message
    /// pointing at the tracking wave.
    #[error("{0}")]
    NotYetImplemented(String),

    /// Hugging Face Hub fetch failed while pulling model weights
    /// (offline, 404, auth, network, ...).
    #[error("hf-hub fetch failed for {repo}: {source}")]
    HfHub {
        /// Repo identifier (e.g. `facebook/encodec_24khz`).
        repo: String,
        /// Underlying error message from `hf-hub`.
        #[source]
        source: io::Error,
    },

    /// Local filesystem error (cache directory, weight file, ...).
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// Underlying candle inference / tensor error.
    #[error("candle error: {0}")]
    Candle(String),

    /// The caller-supplied PCM or token vector does not satisfy the
    /// codec's requirements (wrong sample rate, empty buffer,
    /// non-aligned token count, ...).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Catch-all for unexpected runtime conditions.
    #[error("{0}")]
    Other(String),
}

impl CodecError {
    /// Convenience constructor for [`Self::NotYetImplemented`].
    #[must_use]
    pub fn not_yet_implemented(message: impl Into<String>) -> Self {
        Self::NotYetImplemented(message.into())
    }

    /// Convenience constructor for [`Self::InvalidInput`].
    #[must_use]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Convenience constructor for [`Self::Other`].
    #[must_use]
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}

#[cfg(feature = "candle-backbone")]
impl From<candle_core::Error> for CodecError {
    fn from(err: candle_core::Error) -> Self {
        Self::Candle(err.to_string())
    }
}

/// Lossy projection of [`CodecError`] into the capability-agnostic
/// [`blazen_audio::AudioError`] surface used at the bridge boundary.
impl From<CodecError> for AudioError {
    fn from(err: CodecError) -> Self {
        match err {
            CodecError::EngineNotAvailable => Self::Unsupported(
                "codec backend not built (rebuild with the matching feature flag)".to_string(),
            ),
            CodecError::NotYetImplemented(msg) => Self::Unsupported(msg),
            CodecError::InvalidInput(msg) => Self::InvalidInput(msg),
            CodecError::Io(io) => Self::Io(io),
            CodecError::HfHub { repo, source } => {
                Self::Backend(format!("hf-hub fetch failed for {repo}: {source}"))
            }
            CodecError::Candle(msg) => Self::Backend(format!("candle error: {msg}")),
            CodecError::Other(msg) => Self::Backend(msg),
        }
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, CodecError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_round_trip_messages() {
        let nyi = CodecError::not_yet_implemented("foo");
        assert!(matches!(nyi, CodecError::NotYetImplemented(ref m) if m == "foo"));

        let invalid = CodecError::invalid_input("bar");
        assert!(matches!(invalid, CodecError::InvalidInput(ref m) if m == "bar"));

        let other = CodecError::other("baz");
        assert!(matches!(other, CodecError::Other(ref m) if m == "baz"));
    }

    #[test]
    fn engine_not_available_projects_to_unsupported() {
        let err: AudioError = CodecError::EngineNotAvailable.into();
        assert!(matches!(err, AudioError::Unsupported(_)));
    }

    #[test]
    fn invalid_input_projects_to_invalid_input() {
        let err: AudioError = CodecError::invalid_input("empty").into();
        match err {
            AudioError::InvalidInput(msg) => assert_eq!(msg, "empty"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}
