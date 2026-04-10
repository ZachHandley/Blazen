//! Configuration options for the candle local LLM backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`CandleLlmProvider`](crate::CandleLlmProvider).
///
/// All fields are optional and have sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_llm_candle::CandleLlmOptions;
///
/// // Use defaults
/// let opts = CandleLlmOptions::default();
/// assert!(opts.model_id.is_none());
///
/// // Override specific fields
/// let opts = CandleLlmOptions {
///     model_id: Some("meta-llama/Llama-3.2-1B".into()),
///     device: Some("cuda:0".into()),
///     ..CandleLlmOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CandleLlmOptions {
    /// `HuggingFace` model ID (e.g. `"meta-llama/Llama-3.2-1B"`)
    /// or local path to model weights.
    ///
    /// When `None`, must be set before inference can run.
    pub model_id: Option<String>,

    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Accepts the same format strings as `blazen_llm::Device::parse`:
    /// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`.
    ///
    /// When `None`, defaults to `"cpu"`.
    pub device: Option<String>,

    /// Quantization format string (e.g. `"q4_k_m"` for GGUF).
    ///
    /// Accepts the same format strings as `blazen_llm::Quantization::parse`:
    /// `"f32"`, `"f16"`, `"bf16"`, `"q8_0"`, `"q6_k"`, `"q5_k_m"`,
    /// `"q4_k_m"`, `"q4_k_s"`, `"q3_k_m"`, `"q2_k"`.
    ///
    /// When `None`, the backend will use the model's native precision.
    pub quantization: Option<String>,

    /// Revision / branch on `HuggingFace` to fetch (e.g. `"main"`, `"refs/pr/42"`).
    ///
    /// When `None`, defaults to the repository's default branch.
    pub revision: Option<String>,

    /// Maximum context length in tokens.
    ///
    /// When `None`, the model's built-in context length is used.
    pub context_length: Option<usize>,

    /// Path to cache downloaded models.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_all_none() {
        let opts = CandleLlmOptions::default();
        assert!(opts.model_id.is_none());
        assert!(opts.device.is_none());
        assert!(opts.quantization.is_none());
        assert!(opts.revision.is_none());
        assert!(opts.context_length.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = CandleLlmOptions {
            model_id: Some("meta-llama/Llama-3.2-1B".into()),
            device: Some("cuda:0".into()),
            ..CandleLlmOptions::default()
        };
        assert_eq!(opts.model_id.as_deref(), Some("meta-llama/Llama-3.2-1B"));
        assert_eq!(opts.device.as_deref(), Some("cuda:0"));
        assert!(opts.quantization.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = CandleLlmOptions {
            model_id: Some("test/model".into()),
            device: Some("cpu".into()),
            quantization: Some("q4_k_m".into()),
            revision: Some("main".into()),
            context_length: Some(4096),
            cache_dir: Some(PathBuf::from("/tmp/candle-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: CandleLlmOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_id.as_deref(), Some("test/model"));
        assert_eq!(parsed.device.as_deref(), Some("cpu"));
        assert_eq!(parsed.quantization.as_deref(), Some("q4_k_m"));
        assert_eq!(parsed.revision.as_deref(), Some("main"));
        assert_eq!(parsed.context_length, Some(4096));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/candle-cache"))
        );
    }
}
