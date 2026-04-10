//! The [`CandleEmbedModel`] type -- stub for Phase 6.1.
//!
//! The actual `EmbeddingModel` trait implementation will be added in Phase 6.2
//! once the candle engine API is wired up.

use std::fmt;

use crate::CandleEmbedOptions;

/// Error type for candle embedding operations.
#[derive(Debug)]
pub enum CandleEmbedError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// An embedding operation failed.
    Embedding(String),
}

impl fmt::Display for CandleEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "candle embed invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "candle embed model load failed: {msg}"),
            Self::Embedding(msg) => write!(f, "candle embed operation failed: {msg}"),
        }
    }
}

impl std::error::Error for CandleEmbedError {}

/// A local embedding model backed by [`candle`](https://github.com/huggingface/candle).
///
/// Constructed via [`CandleEmbedModel::from_options`]. The `EmbeddingModel`
/// trait implementation will be added in Phase 6.2.
pub struct CandleEmbedModel {
    /// The resolved model ID that was requested.
    model_id: String,
    /// Full options preserved for deferred engine initialisation.
    #[allow(dead_code)]
    options: CandleEmbedOptions,
    // model: ... -- will hold the candle model once wired (Phase 6.2)
    // tokenizer: ... -- will hold the tokenizer once wired (Phase 6.2)
}

impl CandleEmbedModel {
    /// Create a new embedding model from the given options.
    ///
    /// This currently validates the options and stores them. The actual
    /// candle model will be initialised in Phase 6.2.
    ///
    /// # Errors
    ///
    /// Returns [`CandleEmbedError::InvalidOptions`] if the device string is
    /// present but empty, or if the model ID is present but empty.
    pub fn from_options(opts: CandleEmbedOptions) -> Result<Self, CandleEmbedError> {
        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(CandleEmbedError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        if let Some(ref model_id) = opts.model_id
            && model_id.is_empty()
        {
            return Err(CandleEmbedError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(ref revision) = opts.revision
            && revision.is_empty()
        {
            return Err(CandleEmbedError::InvalidOptions(
                "revision must not be empty when specified".into(),
            ));
        }

        let model_id = opts.effective_model_id().to_owned();

        Ok(Self {
            model_id,
            options: opts,
        })
    }

    /// The `HuggingFace` model ID that was configured at construction time.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CandleEmbedOptions;

    #[test]
    fn from_options_with_defaults() {
        let opts = CandleEmbedOptions::default();
        let model = CandleEmbedModel::from_options(opts).expect("should succeed");
        assert_eq!(model.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn from_options_with_custom_model() {
        let opts = CandleEmbedOptions {
            model_id: Some("BAAI/bge-small-en-v1.5".into()),
            ..CandleEmbedOptions::default()
        };
        let model = CandleEmbedModel::from_options(opts).expect("should succeed");
        assert_eq!(model.model_id(), "BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn from_options_rejects_empty_device() {
        let opts = CandleEmbedOptions {
            device: Some(String::new()),
            ..CandleEmbedOptions::default()
        };
        let result = CandleEmbedModel::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let opts = CandleEmbedOptions {
            model_id: Some(String::new()),
            ..CandleEmbedOptions::default()
        };
        let result = CandleEmbedModel::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_rejects_empty_revision() {
        let opts = CandleEmbedOptions {
            revision: Some(String::new()),
            ..CandleEmbedOptions::default()
        };
        let result = CandleEmbedModel::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_accepts_valid_device() {
        let opts = CandleEmbedOptions {
            device: Some("cuda:0".into()),
            ..CandleEmbedOptions::default()
        };
        let model = CandleEmbedModel::from_options(opts).expect("should succeed");
        assert_eq!(model.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn from_options_accepts_valid_revision() {
        let opts = CandleEmbedOptions {
            revision: Some("main".into()),
            ..CandleEmbedOptions::default()
        };
        let model = CandleEmbedModel::from_options(opts).expect("should succeed");
        assert_eq!(model.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
    }
}
