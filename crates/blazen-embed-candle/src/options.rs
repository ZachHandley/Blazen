//! Configuration options for the candle local embedding backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`CandleEmbedModel`](crate::CandleEmbedModel).
///
/// All fields are optional and have sensible defaults. The model defaults to
/// `sentence-transformers/all-MiniLM-L6-v2`.
///
/// # Examples
///
/// ```
/// use blazen_embed_candle::CandleEmbedOptions;
///
/// // Use defaults (all-MiniLM-L6-v2, CPU)
/// let opts = CandleEmbedOptions::default();
/// assert!(opts.model_id.is_none());
///
/// // Override specific fields
/// let opts = CandleEmbedOptions {
///     model_id: Some("BAAI/bge-small-en-v1.5".into()),
///     device: Some("cuda:0".into()),
///     ..CandleEmbedOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CandleEmbedOptions {
    /// `HuggingFace` model repository ID (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
    ///
    /// When `None`, defaults to `"sentence-transformers/all-MiniLM-L6-v2"`.
    pub model_id: Option<String>,

    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Accepts the same format strings as `blazen_llm::Device::parse`:
    /// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`.
    ///
    /// When `None`, defaults to `"cpu"`.
    pub device: Option<String>,

    /// Model revision / git ref on `HuggingFace` (e.g. `"main"`, a commit hash).
    ///
    /// When `None`, uses the default branch.
    pub revision: Option<String>,

    /// Path to cache downloaded models.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

impl CandleEmbedOptions {
    /// The default `HuggingFace` model ID used when [`model_id`](Self::model_id) is `None`.
    pub const DEFAULT_MODEL_ID: &'static str = "sentence-transformers/all-MiniLM-L6-v2";

    /// Returns the effective model ID, falling back to the default.
    #[must_use]
    pub fn effective_model_id(&self) -> &str {
        self.model_id.as_deref().unwrap_or(Self::DEFAULT_MODEL_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options() {
        let opts = CandleEmbedOptions::default();
        assert!(opts.model_id.is_none());
        assert!(opts.device.is_none());
        assert!(opts.revision.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn effective_model_id_default() {
        let opts = CandleEmbedOptions::default();
        assert_eq!(
            opts.effective_model_id(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
    }

    #[test]
    fn effective_model_id_override() {
        let opts = CandleEmbedOptions {
            model_id: Some("BAAI/bge-small-en-v1.5".into()),
            ..CandleEmbedOptions::default()
        };
        assert_eq!(opts.effective_model_id(), "BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = CandleEmbedOptions {
            model_id: Some("custom/model".into()),
            device: Some("cuda:0".into()),
            ..CandleEmbedOptions::default()
        };
        assert_eq!(opts.model_id.as_deref(), Some("custom/model"));
        assert_eq!(opts.device.as_deref(), Some("cuda:0"));
        assert!(opts.revision.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn serde_roundtrip_options() {
        let opts = CandleEmbedOptions {
            model_id: Some("BAAI/bge-small-en-v1.5".into()),
            device: Some("cuda:0".into()),
            revision: Some("main".into()),
            cache_dir: Some(PathBuf::from("/tmp/candle-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: CandleEmbedOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_id.as_deref(), Some("BAAI/bge-small-en-v1.5"));
        assert_eq!(parsed.device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.revision.as_deref(), Some("main"));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/candle-cache"))
        );
    }
}
