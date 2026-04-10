//! The [`LlamaCppProvider`] type -- stub for Phase 8.
//!
//! The actual `CompletionModel` trait implementation will be added once the
//! llama.cpp engine bindings are wired up.

use std::fmt;

use crate::LlamaCppOptions;

/// Error type for llama.cpp operations.
#[derive(Debug)]
pub enum LlamaCppError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be loaded.
    ModelLoad(String),
    /// An inference operation failed.
    Inference(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for LlamaCppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "llama.cpp invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "llama.cpp model load failed: {msg}"),
            Self::Inference(msg) => write!(f, "llama.cpp inference failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "llama.cpp engine not available: compile with the `engine` feature"
            ),
        }
    }
}

impl std::error::Error for LlamaCppError {}

/// A local LLM provider backed by [`llama.cpp`](https://github.com/ggerganov/llama.cpp).
///
/// Constructed via [`LlamaCppProvider::from_options`]. The `CompletionModel`
/// trait implementation will be added in Phase 8.
pub struct LlamaCppProvider {
    /// The model path that was requested.
    model_path: Option<String>,
    /// Full options preserved for deferred engine initialisation.
    #[allow(dead_code)]
    options: LlamaCppOptions,
    // ctx: ... -- will hold the llama.cpp context once wired (Phase 8)
}

impl LlamaCppProvider {
    /// Create a new provider from the given options.
    ///
    /// This currently validates the options and stores them. The actual
    /// llama.cpp context will be initialised in Phase 8.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::InvalidOptions`] if a specified string field
    /// is present but empty.
    pub fn from_options(opts: LlamaCppOptions) -> Result<Self, LlamaCppError> {
        if let Some(ref model_path) = opts.model_path
            && model_path.is_empty()
        {
            return Err(LlamaCppError::InvalidOptions(
                "model_path must not be empty when specified".into(),
            ));
        }

        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(LlamaCppError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        Ok(Self {
            model_path: opts.model_path.clone(),
            options: opts,
        })
    }

    /// The model path that was passed at construction time.
    #[must_use]
    pub fn model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
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
    use crate::LlamaCppOptions;

    #[test]
    fn from_options_with_defaults() {
        let opts = LlamaCppOptions::default();
        let provider = LlamaCppProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_path().is_none());
    }

    #[test]
    fn from_options_with_model_path() {
        let opts = LlamaCppOptions {
            model_path: Some("/models/llama.gguf".into()),
            ..LlamaCppOptions::default()
        };
        let provider = LlamaCppProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.model_path(), Some("/models/llama.gguf"));
    }

    #[test]
    fn from_options_rejects_empty_model_path() {
        let opts = LlamaCppOptions {
            model_path: Some(String::new()),
            ..LlamaCppOptions::default()
        };
        let result = LlamaCppProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_rejects_empty_device() {
        let opts = LlamaCppOptions {
            device: Some(String::new()),
            ..LlamaCppOptions::default()
        };
        let result = LlamaCppProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_accepts_valid_device() {
        let opts = LlamaCppOptions {
            device: Some("cuda:0".into()),
            ..LlamaCppOptions::default()
        };
        let provider = LlamaCppProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_path().is_none());
    }

    #[test]
    fn engine_not_available_display() {
        let err = LlamaCppError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[test]
    fn engine_available_reflects_feature() {
        let provider = LlamaCppProvider::from_options(LlamaCppOptions::default()).unwrap();
        if cfg!(feature = "engine") {
            assert!(provider.engine_available());
        } else {
            assert!(!provider.engine_available());
        }
    }
}
