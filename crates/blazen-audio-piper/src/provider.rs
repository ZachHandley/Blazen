//! The [`PiperProvider`] type -- stub for Phase 9.
//!
//! The actual `AudioGeneration` trait implementation will be added once the
//! ONNX Runtime engine is wired up for Piper voice models.

use std::fmt;

use crate::PiperOptions;

/// Error type for Piper TTS operations.
#[derive(Debug)]
pub enum PiperError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The voice model could not be downloaded or found.
    ModelLoad(String),
    /// A synthesis operation failed.
    Synthesis(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for PiperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "piper invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "piper model load failed: {msg}"),
            Self::Synthesis(msg) => write!(f, "piper synthesis failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "piper engine not available: compile with the `engine` feature"
            ),
        }
    }
}

impl std::error::Error for PiperError {}

/// A local TTS provider backed by [`Piper`](https://github.com/rhasspy/piper).
///
/// Constructed via [`PiperProvider::from_options`]. The `AudioGeneration`
/// trait implementation will be added in Phase 9.
pub struct PiperProvider {
    /// The voice model identifier that was requested.
    model_id: Option<String>,
    /// Full options preserved for deferred engine initialisation.
    #[allow(dead_code)]
    options: PiperOptions,
    // engine: ... -- will hold the ONNX Runtime session once wired (Phase 9)
}

impl PiperProvider {
    /// Create a new provider from the given options.
    ///
    /// This currently validates the options and stores them. The actual
    /// Piper engine will be initialised in Phase 9.
    ///
    /// # Errors
    ///
    /// Returns [`PiperError::InvalidOptions`] if a specified string field
    /// is present but empty.
    pub fn from_options(opts: PiperOptions) -> Result<Self, PiperError> {
        if let Some(ref model_id) = opts.model_id
            && model_id.is_empty()
        {
            return Err(PiperError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        Ok(Self {
            model_id: opts.model_id.clone(),
            options: opts,
        })
    }

    /// The model identifier that was passed at construction time.
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Whether the engine feature is compiled in.
    #[must_use]
    pub fn engine_available(&self) -> bool {
        cfg!(feature = "engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PiperOptions;

    #[test]
    fn from_options_with_defaults() {
        let opts = PiperOptions::default();
        let provider = PiperProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_id().is_none());
    }

    #[test]
    fn from_options_with_model_id() {
        let opts = PiperOptions {
            model_id: Some("en_US-amy-medium".into()),
            ..PiperOptions::default()
        };
        let provider = PiperProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.model_id(), Some("en_US-amy-medium"));
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let opts = PiperOptions {
            model_id: Some(String::new()),
            ..PiperOptions::default()
        };
        let result = PiperProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_accepts_speaker_id() {
        let opts = PiperOptions {
            model_id: Some("en_US-amy-medium".into()),
            speaker_id: Some(3),
            ..PiperOptions::default()
        };
        let provider = PiperProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.model_id(), Some("en_US-amy-medium"));
    }

    #[test]
    fn engine_not_available_display() {
        let err = PiperError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[test]
    fn engine_available_reflects_feature() {
        let provider = PiperProvider::from_options(PiperOptions::default()).unwrap();
        if cfg!(feature = "engine") {
            assert!(provider.engine_available());
        } else {
            assert!(!provider.engine_available());
        }
    }
}
