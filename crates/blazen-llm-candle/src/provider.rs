//! The [`CandleLlmProvider`] type -- stub for Phase 7.
//!
//! The actual `CompletionModel` trait implementation will be added once the
//! candle engine dependencies are wired up.

use std::fmt;

use crate::CandleLlmOptions;

/// Error type for candle LLM operations.
#[derive(Debug)]
pub enum CandleLlmError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// An inference operation failed.
    Inference(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for CandleLlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "candle LLM invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "candle LLM model load failed: {msg}"),
            Self::Inference(msg) => write!(f, "candle LLM inference failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "candle LLM engine not available: compile with the `engine` feature"
            ),
        }
    }
}

impl std::error::Error for CandleLlmError {}

/// A local LLM provider backed by [`candle`](https://github.com/huggingface/candle).
///
/// Constructed via [`CandleLlmProvider::from_options`]. The `CompletionModel`
/// trait implementation will be added in Phase 7.
pub struct CandleLlmProvider {
    /// The `HuggingFace` model ID that was requested.
    model_id: Option<String>,
    /// Full options preserved for deferred engine initialisation.
    #[allow(dead_code)]
    options: CandleLlmOptions,
    // engine: ... -- will hold the candle model context once wired (Phase 7)
}

impl CandleLlmProvider {
    /// Create a new provider from the given options.
    ///
    /// This currently validates the options and stores them. The actual
    /// candle engine will be initialised in Phase 7.
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::InvalidOptions`] if a specified string field
    /// is present but empty.
    pub fn from_options(opts: CandleLlmOptions) -> Result<Self, CandleLlmError> {
        if let Some(ref model_id) = opts.model_id
            && model_id.is_empty()
        {
            return Err(CandleLlmError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(CandleLlmError::InvalidOptions(
                "device must not be empty when specified".into(),
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
    use crate::CandleLlmOptions;

    #[test]
    fn from_options_with_defaults() {
        let opts = CandleLlmOptions::default();
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_id().is_none());
    }

    #[test]
    fn from_options_with_model_id() {
        let opts = CandleLlmOptions {
            model_id: Some("meta-llama/Llama-3.2-1B".into()),
            ..CandleLlmOptions::default()
        };
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.model_id(), Some("meta-llama/Llama-3.2-1B"));
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let opts = CandleLlmOptions {
            model_id: Some(String::new()),
            ..CandleLlmOptions::default()
        };
        let result = CandleLlmProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_rejects_empty_device() {
        let opts = CandleLlmOptions {
            device: Some(String::new()),
            ..CandleLlmOptions::default()
        };
        let result = CandleLlmProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_accepts_valid_device() {
        let opts = CandleLlmOptions {
            device: Some("cuda:0".into()),
            ..CandleLlmOptions::default()
        };
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_id().is_none());
    }

    #[test]
    fn engine_not_available_display() {
        let err = CandleLlmError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[test]
    fn engine_available_reflects_feature() {
        let provider = CandleLlmProvider::from_options(CandleLlmOptions::default()).unwrap();
        if cfg!(feature = "engine") {
            assert!(provider.engine_available());
        } else {
            assert!(!provider.engine_available());
        }
    }
}
