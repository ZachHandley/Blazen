//! Error type for the music + SFX generation surface.
//!
//! Music backends (MusicGen, AudioGen, Stable Audio, …) live in this crate
//! and surface their failures through [`MusicError`]. Bridge layers in
//! `blazen-llm` are responsible for converting these into the broader
//! `BlazenError` type at the trait boundary.

use std::io;

use thiserror::Error;

/// Errors raised by music + SFX generation backends.
#[derive(Debug, Error)]
pub enum MusicError {
    /// The crate was compiled without the engine feature for this backend
    /// (e.g. `musicgen`), so the underlying candle inference stack is not
    /// linked in.
    #[error(
        "blazen-audio-music was built without the engine feature for this backend -- \
         rebuild with the appropriate feature flag (e.g. `--features musicgen`) \
         to enable on-device music or SFX generation"
    )]
    EngineNotAvailable,

    /// The model the caller requested is a known scaffold (e.g. MusicGen,
    /// AudioGen, Stable Audio) but is not yet implemented in this crate's
    /// supported candle-transformers release. Carries a detailed message
    /// including upstream tracking links.
    #[error("{0}")]
    NotYetImplemented(String),

    /// The Hugging Face Hub fetch failed (offline, 404, auth, network, etc.).
    #[error("hf-hub fetch failed for {repo}: {source}")]
    HfHub {
        /// Repo identifier (e.g. `facebook/musicgen-small`).
        repo: String,
        /// Underlying error from `hf-hub`.
        #[source]
        source: io::Error,
    },

    /// Local filesystem error (cache directory, weight file, etc.).
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// Candle returned an inference / tensor error.
    #[error("candle error: {0}")]
    Candle(String),

    /// The caller-supplied request was structurally invalid (e.g. empty
    /// prompt, zero duration, NaN/Inf in conditioning tensors, …).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Catch-all for unexpected runtime conditions.
    #[error("{0}")]
    Other(String),
}

impl MusicError {
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

#[cfg(feature = "musicgen")]
impl From<candle_core::Error> for MusicError {
    fn from(err: candle_core::Error) -> Self {
        Self::Candle(err.to_string())
    }
}
