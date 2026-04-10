//! Configuration options for the mistral.rs local LLM backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`MistralRsProvider`](crate::MistralRsProvider).
///
/// `model_id` is required -- all other fields have sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_llm_mistralrs::MistralRsOptions;
///
/// let opts = MistralRsOptions {
///     model_id: "mistralai/Mistral-7B-Instruct-v0.3".into(),
///     ..MistralRsOptions::required("mistralai/Mistral-7B-Instruct-v0.3")
/// };
/// assert_eq!(opts.model_id, "mistralai/Mistral-7B-Instruct-v0.3");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsOptions {
    /// `HuggingFace` model ID (e.g. `"mistralai/Mistral-7B-Instruct-v0.3"`)
    /// or local path to a GGUF file.
    pub model_id: String,

    /// Quantization format string (e.g. `"q4_k_m"` for GGUF).
    ///
    /// Accepts the same format strings as `blazen_llm::Quantization::parse`:
    /// `"f32"`, `"f16"`, `"bf16"`, `"q8_0"`, `"q6_k"`, `"q5_k_m"`,
    /// `"q4_k_m"`, `"q4_k_s"`, `"q3_k_m"`, `"q2_k"`, `"gptq-4bit"`,
    /// `"awq-4bit"`.
    ///
    /// When `None`, the backend will use the model's native precision or
    /// auto-detect from the file format.
    pub quantization: Option<String>,

    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Accepts the same format strings as `blazen_llm::Device::parse`:
    /// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`, `"vulkan:N"`, `"rocm:N"`.
    ///
    /// When `None`, defaults to `"cpu"`.
    pub device: Option<String>,

    /// Maximum context length in tokens.
    ///
    /// When `None`, the model's built-in context length is used.
    pub context_length: Option<usize>,

    /// Maximum batch size for concurrent requests.
    ///
    /// When `None`, the engine's default batch size is used.
    pub max_batch_size: Option<usize>,

    /// Chat template override (Jinja2 format string).
    ///
    /// When `None`, the model's built-in chat template is used.
    pub chat_template: Option<String>,

    /// Path to cache downloaded models.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,

    /// Whether this model should be loaded as a vision/multimodal model.
    ///
    /// When `true`, the provider builds the model via mistral.rs's
    /// `MultimodalModelBuilder` instead of `TextModelBuilder`, allowing
    /// chat requests to include image content parts. Vision models like
    /// `LLaVA`, `Phi-3.5-vision`, `Qwen2-VL`, and `Idefics` require this
    /// flag.
    ///
    /// Text-only models will fail to load with this flag set, and text-only
    /// builders will reject image inputs at inference time. Set this to
    /// `true` exactly when the model you are pointing `model_id` at is a
    /// vision-capable model.
    #[serde(default)]
    pub vision: bool,
}

impl MistralRsOptions {
    /// Create options with only the required `model_id` field set.
    ///
    /// All optional fields are `None`. Use struct update syntax to override
    /// individual fields:
    ///
    /// ```
    /// use blazen_llm_mistralrs::MistralRsOptions;
    ///
    /// let opts = MistralRsOptions {
    ///     device: Some("metal".into()),
    ///     context_length: Some(8192),
    ///     ..MistralRsOptions::required("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    /// };
    /// ```
    #[must_use]
    pub fn required(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            quantization: None,
            device: None,
            context_length: None,
            max_batch_size: None,
            chat_template: None,
            cache_dir: None,
            vision: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_sets_model_id() {
        let opts = MistralRsOptions::required("my-org/my-model");
        assert_eq!(opts.model_id, "my-org/my-model");
        assert!(opts.quantization.is_none());
        assert!(opts.device.is_none());
        assert!(opts.context_length.is_none());
        assert!(opts.max_batch_size.is_none());
        assert!(opts.chat_template.is_none());
        assert!(opts.cache_dir.is_none());
        assert!(!opts.vision);
    }

    #[test]
    fn vision_flag_toggles() {
        let opts = MistralRsOptions {
            vision: true,
            ..MistralRsOptions::required("some/vision-model")
        };
        assert!(opts.vision);
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = MistralRsOptions {
            context_length: Some(4096),
            ..MistralRsOptions::required("test/model")
        };
        assert_eq!(opts.model_id, "test/model");
        assert_eq!(opts.context_length, Some(4096));
        assert!(opts.quantization.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = MistralRsOptions {
            model_id: "test/model".into(),
            quantization: Some("q4_k_m".into()),
            device: Some("cpu".into()),
            context_length: Some(8192),
            max_batch_size: Some(4),
            chat_template: Some("{{ bos_token }}".into()),
            cache_dir: Some(PathBuf::from("/tmp/cache")),
            vision: true,
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: MistralRsOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_id, opts.model_id);
        assert_eq!(parsed.context_length, opts.context_length);
        assert_eq!(parsed.quantization, Some("q4_k_m".into()));
        assert_eq!(parsed.device, Some("cpu".into()));
        assert!(parsed.vision);
    }
}
