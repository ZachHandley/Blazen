//! Configuration options for the candle local LLM backend.
//!
//! `CandleLlmOptions` is a thin shell over [`blazen_local_llm::LocalLlmOptions`]
//! (the cross-backend base) plus candle-backend-specific knobs (currently just
//! [`Self::force_safetensors`]). The base struct holds every option that is
//! meaningful regardless of engine — `model_id`, `tokenizer_repo`, `device`,
//! `quantization`, `revision`, `context_length`, `cache_dir`,
//! `initial_adapters` — so call sites read them as `opts.base.<field>`.

use blazen_local_llm::LocalLlmOptions;
use serde::{Deserialize, Serialize};

/// Options for constructing a [`CandleLlmProvider`](crate::CandleLlmProvider).
///
/// All fields are optional and have sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_llm_candle::CandleLlmOptions;
/// use blazen_local_llm::LocalLlmOptions;
///
/// // Use defaults
/// let opts = CandleLlmOptions::default();
/// assert!(opts.base.model_id.is_none());
///
/// // Override specific fields via the shared base struct
/// let opts = CandleLlmOptions {
///     base: LocalLlmOptions::new()
///         .with_model_id("meta-llama/Llama-3.2-1B")
///         .with_device("cuda:0"),
///     ..CandleLlmOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CandleLlmOptions {
    /// Shared local-LLM options (`model_id`, `tokenizer_repo`, `device`,
    /// `quantization`, `revision`, `context_length`, `cache_dir`,
    /// `initial_adapters`). Flattened into the JSON shape so existing wire
    /// formats keep working.
    #[serde(flatten)]
    pub base: LocalLlmOptions,

    /// Force the safetensors (non-quantized) loader path even when a
    /// repo contains both GGUF and safetensors weights.
    ///
    /// Default `false`. Auto-detection at load time normally prefers
    /// GGUF when both formats are present (smaller, faster); set this
    /// to `true` to opt into the dequantized safetensors path — required
    /// to ground the per-Linear hooks future `LoRA` merging needs.
    ///
    /// Has no effect when only one format is present in the repo: the
    /// loader uses whichever is available.
    #[serde(default)]
    pub force_safetensors: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn default_options_all_none() {
        let opts = CandleLlmOptions::default();
        assert!(opts.base.model_id.is_none());
        assert!(opts.base.tokenizer_repo.is_none());
        assert!(opts.base.device.is_none());
        assert!(opts.base.quantization.is_none());
        assert!(opts.base.revision.is_none());
        assert!(opts.base.context_length.is_none());
        assert!(opts.base.cache_dir.is_none());
        assert!(opts.base.initial_adapters.is_empty());
        assert!(!opts.force_safetensors);
    }

    #[test]
    fn force_safetensors_round_trip() {
        let opts = CandleLlmOptions {
            force_safetensors: true,
            ..CandleLlmOptions::default()
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: CandleLlmOptions = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.force_safetensors);
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions::new()
                .with_model_id("meta-llama/Llama-3.2-1B")
                .with_device("cuda:0"),
            ..CandleLlmOptions::default()
        };
        assert_eq!(
            opts.base.model_id.as_deref(),
            Some("meta-llama/Llama-3.2-1B")
        );
        assert_eq!(opts.base.device.as_deref(), Some("cuda:0"));
        assert!(opts.base.quantization.is_none());
    }

    #[test]
    fn tokenizer_repo_round_trip() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions::new()
                .with_model_id("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                .with_tokenizer_repo("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            ..CandleLlmOptions::default()
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: CandleLlmOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.base.tokenizer_repo.as_deref(),
            Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        );
    }

    #[test]
    fn serde_roundtrip() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions {
                model_id: Some("test/model".into()),
                tokenizer_repo: Some("test/tokenizer".into()),
                device: Some("cpu".into()),
                quantization: Some("q4_k_m".into()),
                revision: Some("main".into()),
                context_length: Some(4096),
                cache_dir: Some(PathBuf::from("/tmp/candle-cache")),
                initial_adapters: vec![PathBuf::from("/tmp/adapter-a")],
            },
            force_safetensors: false,
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: CandleLlmOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.base.model_id.as_deref(), Some("test/model"));
        assert_eq!(parsed.base.device.as_deref(), Some("cpu"));
        assert_eq!(parsed.base.quantization.as_deref(), Some("q4_k_m"));
        assert_eq!(parsed.base.revision.as_deref(), Some("main"));
        assert_eq!(parsed.base.context_length, Some(4096));
        assert_eq!(
            parsed.base.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/candle-cache"))
        );
        assert_eq!(
            parsed.base.tokenizer_repo.as_deref(),
            Some("test/tokenizer")
        );
    }
}
