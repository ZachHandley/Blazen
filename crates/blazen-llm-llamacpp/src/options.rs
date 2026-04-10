//! Configuration options for the llama.cpp local LLM backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`LlamaCppProvider`](crate::LlamaCppProvider).
///
/// All fields are optional and have sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_llm_llamacpp::LlamaCppOptions;
///
/// // Use defaults
/// let opts = LlamaCppOptions::default();
/// assert!(opts.model_path.is_none());
///
/// // Override specific fields
/// let opts = LlamaCppOptions {
///     model_path: Some("/models/llama-3.2-1b-q4_k_m.gguf".into()),
///     n_gpu_layers: Some(32),
///     ..LlamaCppOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppOptions {
    /// Path to the GGUF model file, or a `HuggingFace` model ID.
    ///
    /// When `None`, must be set before inference can run.
    pub model_path: Option<String>,

    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Accepts the same format strings as `blazen_llm::Device::parse`:
    /// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`, `"vulkan:N"`, `"rocm:N"`.
    ///
    /// When `None`, defaults to `"cpu"`.
    pub device: Option<String>,

    /// Quantization format string (e.g. `"q4_k_m"`).
    ///
    /// When `None`, the model's native quantization is used (auto-detected
    /// from the GGUF header).
    pub quantization: Option<String>,

    /// Maximum context length in tokens.
    ///
    /// When `None`, the model's built-in context length is used.
    pub context_length: Option<usize>,

    /// Number of layers to offload to GPU.
    ///
    /// When `None`, all layers stay on CPU (unless the device is set to a
    /// GPU device, in which case the engine default applies).
    pub n_gpu_layers: Option<u32>,

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
        let opts = LlamaCppOptions::default();
        assert!(opts.model_path.is_none());
        assert!(opts.device.is_none());
        assert!(opts.quantization.is_none());
        assert!(opts.context_length.is_none());
        assert!(opts.n_gpu_layers.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = LlamaCppOptions {
            model_path: Some("/models/llama.gguf".into()),
            n_gpu_layers: Some(32),
            ..LlamaCppOptions::default()
        };
        assert_eq!(opts.model_path.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(opts.n_gpu_layers, Some(32));
        assert!(opts.device.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = LlamaCppOptions {
            model_path: Some("/models/llama.gguf".into()),
            device: Some("cuda:0".into()),
            quantization: Some("q4_k_m".into()),
            context_length: Some(8192),
            n_gpu_layers: Some(32),
            cache_dir: Some(PathBuf::from("/tmp/llamacpp-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: LlamaCppOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_path.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(parsed.device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.quantization.as_deref(), Some("q4_k_m"));
        assert_eq!(parsed.context_length, Some(8192));
        assert_eq!(parsed.n_gpu_layers, Some(32));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/llamacpp-cache"))
        );
    }
}
