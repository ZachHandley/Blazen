//! Capability-specific error type for TTS backends.
//!
//! Wraps [`blazen_audio::AudioError`] with extra TTS-only variants (e.g.
//! `EngineNotAvailable` for the local `any-tts` backend when the feature
//! is off, and the rich `OpenAi*` HTTP failure modes). The plain
//! `From<AudioError>` impl lets backends bubble up shared errors without
//! ceremony.

use blazen_audio::AudioError;
use thiserror::Error;

/// Error type for any TTS backend in `blazen-audio-tts`.
#[derive(Debug, Error)]
pub enum TtsError {
    /// A required option was missing or invalid.
    #[error("tts invalid options: {0}")]
    InvalidOptions(String),

    /// The voice model could not be downloaded or loaded.
    #[error("tts model load failed: {0}")]
    ModelLoad(String),

    /// A synthesis operation failed.
    #[error("tts synthesis failed: {0}")]
    Synthesis(String),

    /// A required engine feature is not compiled in (e.g. `anytts`,
    /// `openai`). The wrapped string is a human-readable hint about which
    /// feature flag to enable.
    #[error("tts engine not available: {0}")]
    EngineNotAvailable(String),

    /// A capability the caller requested is not supported by this
    /// backend (e.g. voice cloning on a backend that only synthesizes
    /// presets). Mirrors [`AudioError::Unsupported`].
    #[error("tts capability not supported: {0}")]
    Unsupported(String),

    /// Authentication against a remote TTS endpoint failed.
    #[error("tts auth failed: {0}")]
    Auth(String),

    /// A remote TTS endpoint rate-limited the request.
    #[error("tts rate limited (retry_after_seconds={retry_after_seconds:?}): {message}")]
    RateLimit {
        /// `Retry-After` header in seconds, if any.
        retry_after_seconds: Option<u64>,
        /// Server-reported message.
        message: String,
    },

    /// A non-2xx response from a remote TTS endpoint that wasn't 401 or 429.
    #[error("tts server returned {status} for {url}: {message}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Endpoint URL that failed.
        url: String,
        /// Human-readable error message.
        message: String,
    },

    /// HTTP transport failure (DNS, TCP, TLS, body decode, …).
    #[error("tts http error: {0}")]
    Http(String),

    /// A successful response body could not be decoded as the expected
    /// JSON shape.
    #[error("tts failed to decode response body: {0}")]
    Decode(String),

    /// A shared `blazen-audio` error bubbled up from a backend.
    #[error(transparent)]
    Audio(#[from] AudioError),
}

impl From<TtsError> for AudioError {
    fn from(value: TtsError) -> Self {
        match value {
            TtsError::Audio(inner) => inner,
            TtsError::InvalidOptions(msg) => AudioError::InvalidInput(msg),
            TtsError::Unsupported(msg) | TtsError::EngineNotAvailable(msg) => {
                AudioError::Unsupported(msg)
            }
            other => AudioError::Backend(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_error_round_trips_through_tts_error() {
        let tts: TtsError = AudioError::InvalidInput("empty text".into()).into();
        let back: AudioError = tts.into();
        assert!(matches!(back, AudioError::InvalidInput(_)));
    }

    #[test]
    fn engine_not_available_carries_hint() {
        let err = TtsError::EngineNotAvailable("enable `anytts`".into());
        let msg = err.to_string();
        assert!(msg.contains("anytts"));
    }
}
