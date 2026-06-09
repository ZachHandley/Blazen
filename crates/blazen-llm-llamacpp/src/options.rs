//! Configuration options for the llama.cpp local LLM backend.
//!
//! `LlamaCppOptions` embeds [`blazen_local_llm::LocalLlmOptions`] (the
//! cross-backend base) in its `base` field for every option shared with
//! candle / mistralrs, and keeps the llama.cpp-specific
//! [`Self::n_gpu_layers`] knob at the outer level.
//!
//! [`blazen_local_llm::LocalLlmOptions::model_id`] is the unified field for
//! "where do I find the weights". For llama.cpp it accepts either a local
//! filesystem path to a GGUF file or a Hugging Face `repo/filename.gguf`
//! string (auto-detected at load time). llama.cpp reads the tokenizer
//! straight out of the GGUF file itself, so
//! [`blazen_local_llm::LocalLlmOptions::tokenizer_repo`] is accepted on the
//! wire for cross-backend parity but has no effect — setting it logs a
//! warning at construction time.

use blazen_local_llm::LocalLlmOptions;
use serde::{Deserialize, Serialize};

/// Options for constructing a [`LlamaCppProvider`](crate::LlamaCppProvider).
///
/// All fields are optional and have sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_llm_llamacpp::LlamaCppOptions;
/// use blazen_local_llm::LocalLlmOptions;
///
/// // Use defaults
/// let opts = LlamaCppOptions::default();
/// assert!(opts.base.model_id.is_none());
///
/// // Override specific fields
/// let opts = LlamaCppOptions {
///     base: LocalLlmOptions::new().with_model_id("/models/llama-3.2-1b-q4_k_m.gguf"),
///     n_gpu_layers: Some(32),
///     ..LlamaCppOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppOptions {
    /// Shared local-LLM options (`model_id`, `device`, `quantization`,
    /// `revision`, `context_length`, `cache_dir`, `initial_adapters`,
    /// `tokenizer_repo`). `model_id` accepts either a local path or an HF
    /// `repo/filename.gguf` string. Flattened into the JSON shape so
    /// existing wire formats keep working.
    #[serde(flatten)]
    pub base: LocalLlmOptions,

    /// Number of layers to offload to GPU.
    ///
    /// When `None`, all layers stay on CPU (unless the device is set to a
    /// GPU device, in which case the engine default applies).
    pub n_gpu_layers: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn default_options_all_none() {
        let opts = LlamaCppOptions::default();
        assert!(opts.base.model_id.is_none());
        assert!(opts.base.device.is_none());
        assert!(opts.base.quantization.is_none());
        assert!(opts.base.context_length.is_none());
        assert!(opts.n_gpu_layers.is_none());
        assert!(opts.base.cache_dir.is_none());
        assert!(opts.base.initial_adapters.is_empty());
    }

    #[test]
    fn initial_adapters_serde_roundtrip() {
        let opts = LlamaCppOptions {
            base: LocalLlmOptions {
                initial_adapters: vec![PathBuf::from("/cache/lora-a.gguf")],
                ..LocalLlmOptions::default()
            },
            ..LlamaCppOptions::default()
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: LlamaCppOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.base.initial_adapters.len(), 1);
        assert_eq!(
            parsed.base.initial_adapters[0],
            PathBuf::from("/cache/lora-a.gguf")
        );
    }

    #[test]
    fn initial_adapters_defaults_empty_when_missing_from_json() {
        let json = r#"{"model_id":"/models/llama.gguf"}"#;
        let parsed: LlamaCppOptions = serde_json::from_str(json).expect("deserialize");
        assert!(parsed.base.initial_adapters.is_empty());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = LlamaCppOptions {
            base: LocalLlmOptions::new().with_model_id("/models/llama.gguf"),
            n_gpu_layers: Some(32),
        };
        assert_eq!(opts.base.model_id.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(opts.n_gpu_layers, Some(32));
        assert!(opts.base.device.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = LlamaCppOptions {
            base: LocalLlmOptions {
                model_id: Some("/models/llama.gguf".into()),
                device: Some("cuda:0".into()),
                quantization: Some("q4_k_m".into()),
                context_length: Some(8192),
                cache_dir: Some(PathBuf::from("/tmp/llamacpp-cache")),
                ..LocalLlmOptions::default()
            },
            n_gpu_layers: Some(32),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: LlamaCppOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.base.model_id.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(parsed.base.device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.base.quantization.as_deref(), Some("q4_k_m"));
        assert_eq!(parsed.base.context_length, Some(8192));
        assert_eq!(parsed.n_gpu_layers, Some(32));
        assert_eq!(
            parsed.base.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/llamacpp-cache"))
        );
    }
}
