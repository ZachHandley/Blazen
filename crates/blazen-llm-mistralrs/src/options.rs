//! Configuration options for the mistral.rs local LLM backend.
//!
//! `MistralRsOptions` embeds [`blazen_local_llm::LocalLlmOptions`] (the
//! cross-backend base) in its `base` field for every option shared across
//! candle / mistralrs / llama.cpp, and keeps mistralrs-specific knobs
//! ([`Self::max_batch_size`], [`Self::chat_template`], [`Self::vision`]) at
//! the outer level. mistral.rs's `TextModelBuilder` fetches the tokenizer
//! from `model_id` internally — there's no override hook in the upstream
//! API today, so [`blazen_local_llm::LocalLlmOptions::tokenizer_repo`] is
//! accepted on the wire (for parity with the other backends) but ignored
//! by the engine; setting it logs a warning at construction time.

use blazen_local_llm::LocalLlmOptions;
use serde::{Deserialize, Serialize};

/// Options for constructing a [`MistralRsProvider`](crate::MistralRsProvider).
///
/// `base.model_id` is required at engine-load time -- all other fields have
/// sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_llm_mistralrs::MistralRsOptions;
///
/// let opts = MistralRsOptions::required("mistralai/Mistral-7B-Instruct-v0.3");
/// assert_eq!(
///     opts.base.model_id.as_deref(),
///     Some("mistralai/Mistral-7B-Instruct-v0.3"),
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MistralRsOptions {
    /// Shared local-LLM options (`model_id`, `device`, `quantization`,
    /// `revision`, `context_length`, `cache_dir`, `initial_adapters`,
    /// `tokenizer_repo`). Flattened into the JSON shape so existing wire
    /// formats keep working.
    #[serde(flatten)]
    pub base: LocalLlmOptions,

    /// Maximum batch size for concurrent requests.
    ///
    /// When `None`, the engine's default batch size is used.
    pub max_batch_size: Option<usize>,

    /// Chat template override (Jinja2 format string).
    ///
    /// When `None`, the model's built-in chat template is used.
    pub chat_template: Option<String>,

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
    /// `true` exactly when the model you are pointing `base.model_id` at is
    /// a vision-capable model.
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
    /// use blazen_local_llm::LocalLlmOptions;
    ///
    /// let opts = MistralRsOptions {
    ///     base: LocalLlmOptions::new()
    ///         .with_model_id("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    ///         .with_device("metal")
    ///         .with_context_length(8192),
    ///     ..MistralRsOptions::default()
    /// };
    /// ```
    #[must_use]
    pub fn required(model_id: impl Into<String>) -> Self {
        Self {
            base: LocalLlmOptions::new().with_model_id(model_id),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn required_sets_model_id() {
        let opts = MistralRsOptions::required("my-org/my-model");
        assert_eq!(opts.base.model_id.as_deref(), Some("my-org/my-model"));
        assert!(opts.base.quantization.is_none());
        assert!(opts.base.device.is_none());
        assert!(opts.base.context_length.is_none());
        assert!(opts.max_batch_size.is_none());
        assert!(opts.chat_template.is_none());
        assert!(opts.base.cache_dir.is_none());
        assert!(!opts.vision);
        assert!(opts.base.initial_adapters.is_empty());
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
            base: LocalLlmOptions::new()
                .with_model_id("test/model")
                .with_context_length(4096),
            ..MistralRsOptions::default()
        };
        assert_eq!(opts.base.model_id.as_deref(), Some("test/model"));
        assert_eq!(opts.base.context_length, Some(4096));
        assert!(opts.base.quantization.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = MistralRsOptions {
            base: LocalLlmOptions {
                model_id: Some("test/model".into()),
                quantization: Some("q4_k_m".into()),
                device: Some("cpu".into()),
                context_length: Some(8192),
                cache_dir: Some(PathBuf::from("/tmp/cache")),
                ..LocalLlmOptions::default()
            },
            max_batch_size: Some(4),
            chat_template: Some("{{ bos_token }}".into()),
            vision: true,
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: MistralRsOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.base.model_id.as_deref(), Some("test/model"));
        assert_eq!(parsed.base.context_length, Some(8192));
        assert_eq!(parsed.base.quantization.as_deref(), Some("q4_k_m"));
        assert_eq!(parsed.base.device.as_deref(), Some("cpu"));
        assert!(parsed.vision);
    }

    #[test]
    fn initial_adapters_serde_roundtrip() {
        let opts = MistralRsOptions {
            base: LocalLlmOptions {
                initial_adapters: vec![
                    PathBuf::from("/cache/lora-a"),
                    PathBuf::from("/cache/lora-b"),
                ],
                ..LocalLlmOptions::new().with_model_id("test/model")
            },
            ..MistralRsOptions::default()
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: MistralRsOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.base.initial_adapters.len(), 2);
        assert_eq!(
            parsed.base.initial_adapters[0],
            PathBuf::from("/cache/lora-a")
        );
    }

    #[test]
    fn initial_adapters_defaults_empty_when_missing_from_json() {
        let json = r#"{"model_id":"test/model"}"#;
        let parsed: MistralRsOptions = serde_json::from_str(json).expect("deserialize");
        assert!(parsed.base.initial_adapters.is_empty());
    }
}
