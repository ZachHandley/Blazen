//! Shared option + request types for Blazen's local-LLM backends.
//!
//! Sits below `blazen-llm-candle`, `blazen-llm-mistralrs`, `blazen-llm-llamacpp`,
//! `blazen-llm` (orchestrator), and `blazen-controlplane` so that all five can
//! agree on a single strongly-typed shape for "options that apply to any local
//! LLM" (the `LocalLlmOptions` base) and "a typed request to materialise a
//! local model" (the `LocalModelRequest` carried through the factory seam in
//! `blazen-llm::providers::factory`).
//!
//! Backend-specific knobs (e.g. candle's `force_safetensors`, llama.cpp's
//! `n_gpu_layers`, mistral.rs's `vision`) stay on each backend's own Options
//! struct, which embeds [`LocalLlmOptions`] via a `pub base: LocalLlmOptions`
//! field. Use the [`LocalLlmOptionsBuilder`] convenience type for ergonomic
//! construction.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options shared by every local-LLM backend (`candle`, `mistralrs`,
/// `llama.cpp`). Embedded as the `base` field on each backend's own
/// `*Options` struct.
///
/// All fields default to `None` / empty. Backend-specific knobs live on the
/// outer struct; this struct holds only the configuration that is meaningful
/// regardless of which engine is selected.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LocalLlmOptions {
    /// Hugging Face model repo ID (e.g. `"meta-llama/Llama-3.2-1B-Instruct"`)
    /// or a local filesystem path to the weights. Backends that require an
    /// HF repo (mistralrs `TextModelBuilder`) error at load time when this is
    /// a non-repo path; backends that accept either (candle, llama.cpp) treat
    /// the string as a path when it points at an existing file.
    pub model_id: Option<String>,

    /// Optional **separate** Hugging Face repo to fetch `tokenizer.json` from.
    /// Use this when [`Self::model_id`] points at a quantization-only repo
    /// (the common pattern in `TheBloke/*-GGUF`, `bartowski/*-GGUF`, etc.)
    /// whose owners don't redistribute the tokenizer. When `None`, the
    /// tokenizer is fetched from [`Self::model_id`].
    ///
    /// Example: `model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"` +
    /// `tokenizer_repo = Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0")`.
    pub tokenizer_repo: Option<String>,

    /// Revision / branch / tag / commit on the Hugging Face repo
    /// (e.g. `"main"`, `"refs/pr/42"`, a commit SHA). Applies to both
    /// [`Self::model_id`] and [`Self::tokenizer_repo`].
    pub revision: Option<String>,

    /// Hardware device specifier — `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`.
    /// Each backend parses this with its own device resolver. `None` ⇒ each
    /// backend's CPU default.
    pub device: Option<String>,

    /// Quantization format string — `"f32"`, `"f16"`, `"bf16"`, `"q8_0"`,
    /// `"q6_k"`, `"q5_k_m"`, `"q4_k_m"`, `"q4_k_s"`, `"q3_k_m"`, `"q2_k"`.
    /// Backend treats `None` as "use native precision".
    pub quantization: Option<String>,

    /// Maximum context length in tokens. `None` ⇒ the model's built-in cap.
    pub context_length: Option<usize>,

    /// Per-call override for the model-file cache directory.
    /// `None` ⇒ `blazen-model-cache`'s default
    /// (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,

    /// PEFT/LoRA adapter directories to mount immediately after the base
    /// model loads. Each directory must contain `adapter_config.json` and
    /// `adapter_model.safetensors`. The adapter id defaults to the
    /// directory's last path component; callers needing custom ids should
    /// mount via the backend's `load_adapter` method after construction.
    #[serde(default)]
    pub initial_adapters: Vec<PathBuf>,
}

impl LocalLlmOptions {
    /// Construct an empty options struct (all fields `None` / empty).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder shorthand — sugar over field assignment.
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Builder shorthand for [`Self::tokenizer_repo`].
    #[must_use]
    pub fn with_tokenizer_repo(mut self, repo: impl Into<String>) -> Self {
        self.tokenizer_repo = Some(repo.into());
        self
    }

    /// Builder shorthand for [`Self::revision`].
    #[must_use]
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Builder shorthand for [`Self::device`].
    #[must_use]
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Builder shorthand for [`Self::quantization`].
    #[must_use]
    pub fn with_quantization(mut self, quant: impl Into<String>) -> Self {
        self.quantization = Some(quant.into());
        self
    }

    /// Builder shorthand for [`Self::context_length`].
    #[must_use]
    pub fn with_context_length(mut self, n: usize) -> Self {
        self.context_length = Some(n);
        self
    }

    /// Builder shorthand for [`Self::cache_dir`].
    #[must_use]
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Returns the effective tokenizer repo: [`Self::tokenizer_repo`] when
    /// set, otherwise [`Self::model_id`]. Returns `None` when neither is set.
    #[must_use]
    pub fn effective_tokenizer_repo(&self) -> Option<&str> {
        self.tokenizer_repo.as_deref().or(self.model_id.as_deref())
    }
}

/// A typed request to materialise a local model, carried through the
/// `LocalModelFactory` seam in `blazen-llm::providers::factory`. Replaces the
/// previous (`provider: &str`, `model: &str`) two-string signature so that
/// downstream code (controlplane `ManagerHandle`, the per-binding facades)
/// can plumb a full `LocalLlmOptions` payload through end-to-end.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LocalModelRequest {
    /// Provider identifier (e.g. `"candle"`, `"mistralrs"`, `"llamacpp"`).
    /// Routes the request to the matching backend.
    pub provider: String,

    /// Logical model name. Takes precedence over `options.model_id` when
    /// non-empty; the implementation should overwrite
    /// `options.model_id = Some(model)` before delegating to the backend so
    /// the request is self-consistent.
    pub model: String,

    /// Full options payload. Includes the shared base fields plus any
    /// per-backend-specific fields serialised via `serde_json::Value` in
    /// [`Self::backend_extras`] — see the per-binding wrappers for typed
    /// access on each backend.
    pub options: LocalLlmOptions,

    /// Opaque per-backend extra fields (e.g. candle's `force_safetensors`,
    /// mistral.rs's `vision`, llama.cpp's `n_gpu_layers`). Encoded as
    /// `serde_json::Value` so this struct stays cheap to transport across
    /// the factory seam without a hard dependency on every backend's typed
    /// option struct. Each backend's `LocalModelFactory` implementation
    /// deserialises this into its own backend-specific options struct.
    #[serde(default)]
    pub backend_extras: serde_json::Value,
}

impl LocalModelRequest {
    /// Construct a request from the four canonical fields.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        options: LocalLlmOptions,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            options,
            backend_extras: serde_json::Value::Null,
        }
    }

    /// Attach typed per-backend extras by serialising `extras` into the
    /// [`Self::backend_extras`] field.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json` error if `extras` cannot be serialised. In
    /// practice every backend's options struct derives `Serialize` so this
    /// is infallible at the type level — the `Result` is a future-proof
    /// safety net.
    pub fn with_backend_extras<T: serde::Serialize>(
        mut self,
        extras: &T,
    ) -> Result<Self, serde_json::Error> {
        self.backend_extras = serde_json::to_value(extras)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_are_all_none() {
        let opts = LocalLlmOptions::default();
        assert!(opts.model_id.is_none());
        assert!(opts.tokenizer_repo.is_none());
        assert!(opts.revision.is_none());
        assert!(opts.device.is_none());
        assert!(opts.quantization.is_none());
        assert!(opts.context_length.is_none());
        assert!(opts.cache_dir.is_none());
        assert!(opts.initial_adapters.is_empty());
    }

    #[test]
    fn builders_compose() {
        let opts = LocalLlmOptions::new()
            .with_model_id("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
            .with_tokenizer_repo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            .with_revision("main")
            .with_device("cuda:0")
            .with_quantization("q4_k_m")
            .with_context_length(2048)
            .with_cache_dir("/var/blazen/models");
        assert_eq!(
            opts.model_id.as_deref(),
            Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        );
        assert_eq!(
            opts.effective_tokenizer_repo(),
            Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        );
        assert_eq!(opts.context_length, Some(2048));
    }

    #[test]
    fn effective_tokenizer_repo_falls_back_to_model_id() {
        let opts = LocalLlmOptions::new().with_model_id("unsloth/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(
            opts.effective_tokenizer_repo(),
            Some("unsloth/Qwen2.5-0.5B-Instruct-GGUF")
        );
    }

    #[test]
    fn serde_roundtrip() {
        let opts = LocalLlmOptions {
            model_id: Some("a/b".into()),
            tokenizer_repo: Some("c/d".into()),
            revision: Some("main".into()),
            device: Some("cpu".into()),
            quantization: Some("q4_k_m".into()),
            context_length: Some(4096),
            cache_dir: Some(PathBuf::from("/var/cache")),
            initial_adapters: vec![PathBuf::from("/var/adapter")],
        };
        let json = serde_json::to_string(&opts).expect("ser");
        let parsed: LocalLlmOptions = serde_json::from_str(&json).expect("de");
        assert_eq!(opts, parsed);
    }

    #[derive(Serialize, serde::Deserialize, Debug, PartialEq)]
    struct CandleExtras {
        force_safetensors: bool,
    }

    #[test]
    fn local_model_request_carries_options_and_extras() {
        let opts = LocalLlmOptions::new().with_model_id("a/b");
        let req = LocalModelRequest::new("candle", "tinyllama", opts)
            .with_backend_extras(&CandleExtras {
                force_safetensors: true,
            })
            .expect("ser extras");
        assert_eq!(req.provider, "candle");
        assert_eq!(req.model, "tinyllama");
        let parsed: CandleExtras = serde_json::from_value(req.backend_extras).expect("de");
        assert_eq!(
            parsed,
            CandleExtras {
                force_safetensors: true
            }
        );
    }
}
