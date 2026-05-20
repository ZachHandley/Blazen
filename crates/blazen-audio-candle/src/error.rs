//! Error types for the candle-audio bridge.
//!
//! These errors stay inside `blazen-audio-candle`; the
//! [`backends::candle_audio`](crate) bridge layer in `blazen-llm` converts
//! them into `BlazenError` at the trait boundary so downstream consumers
//! only ever see `BlazenError` variants.

use std::io;

use thiserror::Error;

/// Errors raised by the candle-audio crate.
#[derive(Debug, Error)]
pub enum CandleAudioError {
    /// The crate was compiled without the `engine` feature, so the underlying
    /// candle inference stack is not linked in. Every model entry point
    /// short-circuits with this error.
    #[error(
        "blazen-audio-candle was built without the `engine` feature -- \
         rebuild with `--features engine` (and optionally `cuda` or `metal`) \
         to enable on-device audio generation"
    )]
    EngineNotAvailable,

    /// The model the caller requested is a known scaffold (e.g. MusicGen)
    /// but is not yet implemented in candle-transformers 0.10.2. Carries a
    /// detailed message including upstream tracking links.
    #[error("{0}")]
    NotYetImplemented(String),

    /// The Hugging Face Hub fetch failed (offline, 404, auth, network, etc.).
    #[error("hf-hub fetch failed for {repo}: {source}")]
    HfHub {
        /// Repo identifier (e.g. `facebook/encodec_24khz`).
        repo: String,
        /// Underlying error message from `hf-hub`.
        #[source]
        source: io::Error,
    },

    /// Local filesystem error (cache directory, weight file, etc.).
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// Candle returned an inference / tensor error.
    #[error("candle error: {0}")]
    Candle(String),

    /// The input PCM the caller passed does not satisfy the model's
    /// requirements (wrong sample rate, empty buffer, NaN/Inf samples, ...).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Catch-all for unexpected runtime conditions.
    #[error("{0}")]
    Other(String),
}

impl CandleAudioError {
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

#[cfg(feature = "engine")]
impl From<candle_core::Error> for CandleAudioError {
    fn from(err: candle_core::Error) -> Self {
        Self::Candle(err.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, CandleAudioError>;
