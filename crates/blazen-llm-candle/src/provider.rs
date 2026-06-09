//! The [`CandleLlmProvider`] type -- local LLM inference via candle.
//!
//! When the `engine` feature is enabled, this module provides a fully
//! functional local LLM backend using GGUF quantized models loaded through
//! `candle-transformers`. Without the feature, the provider compiles but
//! returns [`CandleLlmError::EngineNotAvailable`] for all inference calls.

use std::fmt;
use std::path::PathBuf;

use crate::CandleLlmOptions;

#[cfg(feature = "engine")]
use std::path::Path;

/// Error type for candle LLM operations.
#[derive(Debug)]
pub enum CandleLlmError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// An inference operation failed.
    Inference(String),
    /// The detected model/format is recognised but the engine has not
    /// wired support for it yet (e.g. safetensors path detects a
    /// `model_type` other than `llama`).
    Unsupported(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for CandleLlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "candle LLM invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "candle LLM model load failed: {msg}"),
            Self::Inference(msg) => write!(f, "candle LLM inference failed: {msg}"),
            Self::Unsupported(msg) => write!(f, "candle LLM unsupported: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "candle LLM engine not available: compile with the `engine` feature"
            ),
        }
    }
}

impl std::error::Error for CandleLlmError {}

// ---------------------------------------------------------------------------
// Engine implementation (behind feature gate)
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
mod engine {
    use std::path::PathBuf;

    use candle_core::quantized::gguf_file;
    use candle_core::{Device, Tensor};
    use candle_transformers::generation::LogitsProcessor;
    use candle_transformers::models::quantized_llama::ModelWeights;
    use tokenizers::Tokenizer;

    use super::CandleLlmError;
    use crate::CandleLlmOptions;

    /// Resolve the candle [`Device`] from the user's device string.
    fn resolve_device(device_str: Option<&str>) -> Result<Device, CandleLlmError> {
        match device_str.unwrap_or("cpu") {
            "cpu" => Ok(Device::Cpu),
            s if s.starts_with("cuda") => {
                let ordinal = s
                    .strip_prefix("cuda:")
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(0);
                Device::new_cuda(ordinal)
                    .map_err(|e| CandleLlmError::ModelLoad(format!("CUDA device error: {e}")))
            }
            "metal" => Device::new_metal(0)
                .map_err(|e| CandleLlmError::ModelLoad(format!("Metal device error: {e}"))),
            other => Err(CandleLlmError::InvalidOptions(format!(
                "unsupported device: {other}"
            ))),
        }
    }

    /// Find GGUF files for a model on `HuggingFace` Hub.
    ///
    /// GGUF repos typically name their quantized files like
    /// `model-q4_k_m.gguf` or just `model.gguf`. We look for files matching
    /// the requested quantization, falling back to any `.gguf` file.
    async fn download_gguf(
        repo_id: &str,
        revision: Option<&str>,
        quantization: Option<&str>,
    ) -> Result<PathBuf, CandleLlmError> {
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(|e| CandleLlmError::ModelLoad(format!("HF API error: {e}")))?;

        let repo = if let Some(rev) = revision {
            api.repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            ))
        } else {
            api.model(repo_id.to_string())
        };

        // Try to find the right GGUF file. Common naming patterns:
        //   - <name>-<quant>.gguf  (e.g. model-q4_k_m.gguf)
        //   - <name>.gguf
        //   - unsloth.Q4_K_M.gguf (Unsloth convention)
        // We try a quantization-specific name first, then fall back.
        let quant_suffix = quantization.unwrap_or("Q4_K_M");
        let candidates = [
            format!("unsloth.{quant_suffix}.gguf"),
            format!("model-{}.gguf", quant_suffix.to_lowercase()),
            format!("{quant_suffix}.gguf"),
            "model.gguf".to_string(),
        ];

        for candidate in &candidates {
            if let Ok(path) = repo.get(candidate).await {
                tracing::info!(file = %candidate, repo = repo_id, "downloaded GGUF model");
                return Ok(path);
            }
        }

        Err(CandleLlmError::ModelLoad(format!(
            "no GGUF file found in repo '{repo_id}'; \
             tried: {}",
            candidates.join(", ")
        )))
    }

    /// Download the tokenizer for a model repo.
    async fn download_tokenizer(
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<Tokenizer, CandleLlmError> {
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(false)
            .build()
            .map_err(|e| CandleLlmError::ModelLoad(format!("HF API error: {e}")))?;

        let repo = if let Some(rev) = revision {
            api.repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            ))
        } else {
            api.model(repo_id.to_string())
        };

        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| CandleLlmError::ModelLoad(format!("tokenizer download failed: {e}")))?;

        Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CandleLlmError::ModelLoad(format!("failed to load tokenizer: {e}")))
    }

    /// Loaded candle engine state.
    pub(crate) struct CandleEngine {
        pub(crate) model: ModelWeights,
        pub(crate) tokenizer: Tokenizer,
        pub(crate) device: Device,
        pub(crate) context_length: usize,
    }

    impl CandleEngine {
        /// Load a quantized GGUF model from `HuggingFace` Hub.
        pub(crate) async fn load(opts: &CandleLlmOptions) -> Result<Self, CandleLlmError> {
            let repo_id = opts.base.model_id.as_deref().ok_or_else(|| {
                CandleLlmError::InvalidOptions("model_id is required for engine init".into())
            })?;

            let device = resolve_device(opts.base.device.as_deref())?;
            let revision = opts.base.revision.as_deref();
            let quantization = opts.base.quantization.as_deref();
            // tokenizer_repo: when set, fetch tokenizer.json from this repo
            // instead of the GGUF (model_id) repo — the common pattern for
            // TheBloke / bartowski GGUF-only repos whose owners don't
            // redistribute the tokenizer.
            let tokenizer_repo = opts.base.tokenizer_repo.as_deref().unwrap_or(repo_id);

            // Download model and tokenizer in parallel.
            let (gguf_path, tokenizer) = tokio::try_join!(
                download_gguf(repo_id, revision, quantization),
                download_tokenizer(tokenizer_repo, revision),
            )?;

            // Load the GGUF model (CPU-bound -- run on blocking thread).
            let device_clone = device.clone();
            let model = tokio::task::spawn_blocking(move || {
                let mut file = std::fs::File::open(&gguf_path).map_err(|e| {
                    CandleLlmError::ModelLoad(format!("cannot open {}: {e}", gguf_path.display()))
                })?;

                let content = gguf_file::Content::read(&mut file)
                    .map_err(|e| CandleLlmError::ModelLoad(format!("GGUF parse error: {e}")))?;

                ModelWeights::from_gguf(content, &mut file, &device_clone)
                    .map_err(|e| CandleLlmError::ModelLoad(format!("model weight load error: {e}")))
            })
            .await
            .map_err(|e| CandleLlmError::ModelLoad(format!("blocking task failed: {e}")))??;

            let context_length = opts.base.context_length.unwrap_or(4096);

            tracing::info!(
                repo = repo_id,
                context_length,
                device = ?opts.base.device,
                "candle LLM engine loaded"
            );

            if !opts.base.initial_adapters.is_empty() {
                tracing::warn!(
                    count = opts.base.initial_adapters.len(),
                    "initial_adapters configured but candle backend's quantized_llama \
                     engine has no per-Linear hook; adapters will fail to mount until \
                     a per-architecture wrapper is implemented"
                );
            }

            Ok(Self {
                model,
                tokenizer,
                device,
                context_length,
            })
        }
    }

    /// Format chat messages into a prompt string.
    ///
    /// Uses the tokenizer's chat template if available, otherwise falls
    /// back to a simple `<|system|>`, `<|user|>`, `<|assistant|>` format
    /// that works with most instruction-tuned models.
    pub(crate) fn format_prompt(messages: &[(String, String)]) -> String {
        // We use a simple tag-based chat template that works with most
        // instruction-tuned models. A future enhancement could extract
        // the chat_template from the tokenizer's config and use Jinja
        // rendering, but this format covers Phi, ChatML, and similar
        // models well enough.

        let mut prompt = String::new();
        for (role, content) in messages {
            match role.as_str() {
                "system" => {
                    prompt.push_str("<|system|>\n");
                    prompt.push_str(content);
                    prompt.push_str("\n<|end|>\n");
                }
                "user" => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(content);
                    prompt.push_str("\n<|end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|assistant|>\n");
                    prompt.push_str(content);
                    prompt.push_str("\n<|end|>\n");
                }
                _ => {
                    // Tool or unknown role -- treat as user.
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(content);
                    prompt.push_str("\n<|end|>\n");
                }
            }
        }
        // Signal the model to generate an assistant response.
        prompt.push_str("<|assistant|>\n");

        prompt
    }

    /// Run autoregressive token generation on the loaded model.
    ///
    /// Returns the generated text and the number of tokens produced.
    pub(crate) fn generate(
        engine: &mut CandleEngine,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<(String, usize), CandleLlmError> {
        let encoding = engine
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| CandleLlmError::Inference(format!("tokenization failed: {e}")))?;

        let prompt_tokens = encoding.get_ids().to_vec();

        // Truncate if the prompt exceeds the context window.
        let max_prompt = engine.context_length.saturating_sub(max_tokens);
        let prompt_tokens = if prompt_tokens.len() > max_prompt {
            tracing::warn!(
                prompt_len = prompt_tokens.len(),
                max_prompt,
                "truncating prompt to fit context window"
            );
            prompt_tokens[prompt_tokens.len() - max_prompt..].to_vec()
        } else {
            prompt_tokens
        };

        let eos_token = engine
            .tokenizer
            .token_to_id("<|end|>")
            .or_else(|| engine.tokenizer.token_to_id("</s>"))
            .or_else(|| engine.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| engine.tokenizer.token_to_id("<|im_end|>"));

        let temp = temperature.unwrap_or(0.7);
        let mut logits_processor = LogitsProcessor::new(/* seed */ 42, Some(temp), top_p);

        // Feed the prompt through the model.
        let input = Tensor::new(prompt_tokens.as_slice(), &engine.device)
            .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;

        let logits = engine
            .model
            .forward(&input, 0)
            .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;

        let logits = logits
            .squeeze(0)
            .map_err(|e| CandleLlmError::Inference(format!("squeeze failed: {e}")))?;

        let last_logits = logits
            .get(logits.dim(0).unwrap_or(1) - 1)
            .map_err(|e| CandleLlmError::Inference(format!("get last logits failed: {e}")))?;

        let mut next_token = logits_processor
            .sample(&last_logits)
            .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);

        // Autoregressive loop.
        for i in 0..max_tokens {
            // Check for EOS.
            if eos_token.is_some_and(|eos| next_token == eos) {
                break;
            }

            generated_tokens.push(next_token);

            // Feed the new token.
            let input = Tensor::new(&[next_token], &engine.device)
                .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;

            let seq_pos = prompt_tokens.len() + i;
            let logits = engine
                .model
                .forward(&input, seq_pos)
                .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;

            let logits = logits
                .squeeze(0)
                .map_err(|e| CandleLlmError::Inference(format!("squeeze failed: {e}")))?;

            let logits = logits
                .get(logits.dim(0).unwrap_or(1) - 1)
                .map_err(|e| CandleLlmError::Inference(format!("get logits failed: {e}")))?;

            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;
        }

        let text = engine
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| CandleLlmError::Inference(format!("detokenization failed: {e}")))?;

        let token_count = generated_tokens.len();
        Ok((text, token_count))
    }

    /// Run autoregressive generation yielding one token at a time.
    ///
    /// Returns an iterator-like structure via a callback for each token.
    /// This is used by the streaming bridge.
    pub(crate) fn generate_streaming<F>(
        engine: &mut CandleEngine,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f64>,
        top_p: Option<f64>,
        mut on_token: F,
    ) -> Result<usize, CandleLlmError>
    where
        F: FnMut(String),
    {
        let encoding = engine
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| CandleLlmError::Inference(format!("tokenization failed: {e}")))?;

        let prompt_tokens = encoding.get_ids().to_vec();

        let max_prompt = engine.context_length.saturating_sub(max_tokens);
        let prompt_tokens = if prompt_tokens.len() > max_prompt {
            prompt_tokens[prompt_tokens.len() - max_prompt..].to_vec()
        } else {
            prompt_tokens
        };

        let eos_token = engine
            .tokenizer
            .token_to_id("<|end|>")
            .or_else(|| engine.tokenizer.token_to_id("</s>"))
            .or_else(|| engine.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| engine.tokenizer.token_to_id("<|im_end|>"));

        let temp = temperature.unwrap_or(0.7);
        let mut logits_processor = LogitsProcessor::new(42, Some(temp), top_p);

        let input = Tensor::new(prompt_tokens.as_slice(), &engine.device)
            .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;

        let logits = engine
            .model
            .forward(&input, 0)
            .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;

        let logits = logits
            .squeeze(0)
            .map_err(|e| CandleLlmError::Inference(format!("squeeze failed: {e}")))?;

        let last_logits = logits
            .get(logits.dim(0).unwrap_or(1) - 1)
            .map_err(|e| CandleLlmError::Inference(format!("get last logits failed: {e}")))?;

        let mut next_token = logits_processor
            .sample(&last_logits)
            .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;

        let mut token_count = 0usize;

        for i in 0..max_tokens {
            if eos_token.is_some_and(|eos| next_token == eos) {
                break;
            }

            // Decode this single token.
            let text = engine
                .tokenizer
                .decode(&[next_token], true)
                .map_err(|e| CandleLlmError::Inference(format!("detokenization failed: {e}")))?;

            if !text.is_empty() {
                on_token(text);
            }

            token_count += 1;

            let input = Tensor::new(&[next_token], &engine.device)
                .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;

            let seq_pos = prompt_tokens.len() + i;
            let logits = engine
                .model
                .forward(&input, seq_pos)
                .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;

            let logits = logits
                .squeeze(0)
                .map_err(|e| CandleLlmError::Inference(format!("squeeze failed: {e}")))?;

            let logits = logits
                .get(logits.dim(0).unwrap_or(1) - 1)
                .map_err(|e| CandleLlmError::Inference(format!("get logits failed: {e}")))?;

            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;
        }

        Ok(token_count)
    }
}

// ---------------------------------------------------------------------------
// Format-dispatch wrapper for the two engine paths.
// ---------------------------------------------------------------------------

/// Loaded engine, either GGUF-quantized or non-quantized safetensors.
///
/// The variant is chosen at load time by [`load_with_autodetect`]
/// based on a HF Hub metadata probe and the user's
/// [`CandleLlmOptions::force_safetensors`] override.
#[cfg(feature = "engine")]
pub(crate) enum LoadedEngine {
    /// Quantized GGUF path via `candle_transformers::quantized_llama`.
    Gguf(engine::CandleEngine),
    /// Non-quantized safetensors path via
    /// `candle_transformers::models::llama::Llama`. Foundation for
    /// the Wave C LoRA-merge work.
    Safetensors(crate::safetensors_engine::SafetensorsEngine),
}

/// Outcome of an HF Hub format probe, before any download.
#[cfg(feature = "engine")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatChoice {
    Gguf,
    Safetensors,
}

/// Decide which engine path to take given the repo's discovered file
/// layout and the user's preference.
#[cfg(feature = "engine")]
fn choose_format(
    layout: &crate::safetensors_engine::RepoLayout,
    force_safetensors: bool,
    repo_id: &str,
) -> Result<FormatChoice, CandleLlmError> {
    match (layout.has_gguf(), layout.has_safetensors()) {
        (true, true) => {
            if force_safetensors {
                Ok(FormatChoice::Safetensors)
            } else {
                Ok(FormatChoice::Gguf)
            }
        }
        (true, false) => Ok(FormatChoice::Gguf),
        (false, true) => Ok(FormatChoice::Safetensors),
        (false, false) => Err(CandleLlmError::InvalidOptions(format!(
            "repo '{repo_id}' contains neither GGUF nor safetensors weights"
        ))),
    }
}

/// Load whichever engine matches the repo, honouring
/// `force_safetensors` for tiebreaks.
///
/// `pre_registered_adapters` is consulted only on the safetensors path
/// (GGUF cannot consume merged deltas); when non-empty its adapters are
/// re-parsed against the engine device and folded into the initial
/// `VarBuilder` via
/// [`crate::safetensors_engine::SafetensorsEngine::load_with_adapters`].
#[cfg(feature = "engine")]
async fn load_with_autodetect(
    opts: &CandleLlmOptions,
    pre_registered_adapters: &[MountedAdapterRecord],
) -> Result<LoadedEngine, CandleLlmError> {
    let repo_id = opts.base.model_id.as_deref().ok_or_else(|| {
        CandleLlmError::InvalidOptions("model_id is required for engine init".into())
    })?;
    let revision = opts.base.revision.as_deref();

    let layout = crate::safetensors_engine::probe_repo_layout(repo_id, revision).await?;
    let choice = choose_format(&layout, opts.force_safetensors, repo_id)?;

    tracing::info!(
        repo = repo_id,
        gguf_count = layout.gguf.len(),
        safetensors_count = layout.safetensors.len(),
        ?choice,
        pre_registered_adapters = pre_registered_adapters.len(),
        "candle engine format auto-detect"
    );

    match choice {
        FormatChoice::Gguf => {
            if !pre_registered_adapters.is_empty() {
                return Err(CandleLlmError::Unsupported(
                    "candle GGUF + LoRA requires dequantize-merge-requantize; \
                     use safetensors format or the mistralrs backend"
                        .into(),
                ));
            }
            engine::CandleEngine::load(opts)
                .await
                .map(LoadedEngine::Gguf)
        }
        FormatChoice::Safetensors => {
            // Why: deltas must be precomputed on the engine's
            // ultimate device. Resolve once here and pass it through.
            let device = resolve_device_for_initial_load(opts.base.device.as_deref())?;
            let parsed = parse_adapters_on_device(pre_registered_adapters, &device)?;
            crate::safetensors_engine::SafetensorsEngine::load_with_adapters(opts, &parsed)
                .await
                .map(LoadedEngine::Safetensors)
        }
    }
}

/// Resolve the candle [`candle_core::Device`] from the caller's device
/// string for the purposes of pre-load adapter parsing. Kept in this
/// module to avoid widening [`crate::safetensors_engine`]'s private
/// surface for what is essentially the same lookup as
/// [`crate::safetensors_engine::resolve_device`].
#[cfg(feature = "engine")]
fn resolve_device_for_initial_load(
    device_str: Option<&str>,
) -> Result<candle_core::Device, CandleLlmError> {
    match device_str.unwrap_or("cpu") {
        "cpu" => Ok(candle_core::Device::Cpu),
        s if s.starts_with("cuda") => {
            let ordinal = s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(0);
            candle_core::Device::new_cuda(ordinal)
                .map_err(|e| CandleLlmError::ModelLoad(format!("CUDA device error: {e}")))
        }
        "metal" => candle_core::Device::new_metal(0)
            .map_err(|e| CandleLlmError::ModelLoad(format!("Metal device error: {e}"))),
        other => Err(CandleLlmError::InvalidOptions(format!(
            "unsupported device: {other}"
        ))),
    }
}

/// Re-parse every recorded adapter directory against `device` so the
/// `(B@A) * scale` deltas land on the same device the merging backend
/// will consume them from.
#[cfg(feature = "engine")]
fn parse_adapters_on_device(
    records: &[MountedAdapterRecord],
    device: &candle_core::Device,
) -> Result<Vec<crate::lora::LoadedAdapter>, CandleLlmError> {
    records
        .iter()
        .map(|r| {
            crate::lora::LoadedAdapter::from_dir(
                &r.adapter_dir,
                r.adapter_id.clone(),
                r.scale,
                device,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public provider type
// ---------------------------------------------------------------------------

/// A local LLM provider backed by [`candle`](https://github.com/huggingface/candle).
///
/// Constructed via [`CandleLlmProvider::from_options`]. When the `engine`
/// feature is enabled, the provider downloads GGUF quantized models from
/// `HuggingFace` Hub and runs autoregressive text generation locally.
///
/// # Architecture support
///
/// The provider uses `candle-transformers`' quantized Llama weights loader
/// (`ModelWeights`), which supports GGUF models for Llama, Mistral, Phi,
/// Gemma, and other architectures that share the Llama-family KV-cache
/// structure.
pub struct CandleLlmProvider {
    /// The `HuggingFace` model ID that was requested.
    model_id: Option<String>,
    /// Full options preserved for deferred engine initialisation.
    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    options: CandleLlmOptions,
    /// Lazily loaded candle engine, wrapped in
    /// `Arc<Mutex<Option<...>>>` so we can both (a) auto-load on first
    /// use and (b) explicitly unload later to free GPU memory.
    ///
    /// A `tokio::sync::Mutex` is used instead of `RwLock` because the
    /// engine's `ModelWeights` forward pass requires `&mut self` and
    /// inference must hold an exclusive guard for the entire generation
    /// loop. Holding the guard across an `await` -- required because the
    /// load path downloads weights from `HuggingFace` Hub -- is safe with
    /// `tokio::sync::Mutex`, and passing an `OwnedMutexGuard` into
    /// `spawn_blocking` keeps the lock alive across the thread hop so
    /// that an `unload` call cannot yank the engine out from under an
    /// in-flight inference.
    #[cfg(feature = "engine")]
    engine: std::sync::Arc<tokio::sync::Mutex<Option<LoadedEngine>>>,
    /// Mounted `LoRA` adapters, keyed by caller-supplied adapter id.
    ///
    /// Adapters are parsed + validated against the PEFT-canonical
    /// layout but the candle `quantized_llama::ModelWeights` forward
    /// pass exposes no per-`Linear` hook, so deltas cannot be applied
    /// in-place. [`Self::load_adapter`] therefore validates the
    /// adapter directory and returns an `Unsupported`-shaped
    /// [`CandleLlmError::Inference`] instead of recording the mount;
    /// this map exists for diagnostics + future per-architecture
    /// wrappers that will be able to consume the parsed layers.
    #[cfg(feature = "engine")]
    loaded_adapters: std::sync::Arc<
        tokio::sync::RwLock<std::collections::HashMap<String, MountedAdapterRecord>>,
    >,
}

/// Bookkeeping record for a successfully-parsed adapter mount. Kept
/// separate from [`crate::lora::LoadedAdapter`] so the public
/// [`CandleLlmProvider::list_adapters`] return shape stays cheap-to-clone.
#[cfg(feature = "engine")]
#[derive(Debug, Clone)]
pub struct MountedAdapterRecord {
    /// Caller-chosen identifier (echoes `AdapterOptions::adapter_id`).
    pub adapter_id: String,
    /// Directory the adapter was mounted from.
    pub adapter_dir: PathBuf,
    /// Caller-supplied runtime scale multiplier.
    pub scale: f32,
}

impl CandleLlmProvider {
    /// Create a new provider from the given options.
    ///
    /// This validates the options and stores them. The actual candle engine
    /// is not loaded until [`CandleLlmProvider::load`] is called (or
    /// lazily on the first inference call).
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::InvalidOptions`] if a specified string field
    /// is present but empty.
    pub fn from_options(opts: CandleLlmOptions) -> Result<Self, CandleLlmError> {
        if let Some(ref model_id) = opts.base.model_id
            && model_id.is_empty()
        {
            return Err(CandleLlmError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(ref device) = opts.base.device
            && device.is_empty()
        {
            return Err(CandleLlmError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        Ok(Self {
            model_id: opts.base.model_id.clone(),
            options: opts,
            #[cfg(feature = "engine")]
            engine: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            #[cfg(feature = "engine")]
            loaded_adapters: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
        })
    }

    /// The model identifier that was passed at construction time.
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// The configured device string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`), or
    /// `None` if no device was specified (which defaults to CPU at inference time).
    #[must_use]
    pub fn device_str(&self) -> Option<&str> {
        self.options.base.device.as_deref()
    }

    /// Whether the engine feature is compiled in.
    #[must_use]
    pub fn engine_available(&self) -> bool {
        cfg!(feature = "engine")
    }

    // -----------------------------------------------------------------------
    // Explicit load / unload (always in the public API)
    //
    // These mirror the `infer` / `infer_stream` cfg dual-stub pattern so
    // that the public surface is identical with and without the `engine`
    // feature, and so the `blazen_llm::LocalModel` trait bridge in
    // `blazen-llm/src/backends/candle_llm.rs` can call them unconditionally.
    // -----------------------------------------------------------------------

    /// Eagerly load the candle engine. Idempotent -- if the engine is
    /// already loaded, this is a no-op that returns `Ok(())`.
    ///
    /// Downloads the model weights and tokenizer from `HuggingFace` Hub
    /// (if not already cached) and initialises the inference engine.
    ///
    /// Inference methods ([`Self::infer`], [`Self::infer_stream`]) will
    /// still auto-load on first call if [`Self::load`] was never invoked,
    /// so explicit loading is only needed when the caller wants to pay
    /// the initialization cost up-front (e.g. to avoid latency spikes
    /// during a time-sensitive workflow step).
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::EngineNotAvailable`] if the `engine`
    /// feature is not compiled in.
    /// Returns [`CandleLlmError::ModelLoad`] if the download or load fails.
    #[cfg(feature = "engine")]
    pub async fn load(&self) -> Result<(), CandleLlmError> {
        // Reuse the guarded loader. Drop the returned guard immediately;
        // we only wanted the side effect of populating the Option.
        let _ = self.get_or_load_engine_guard().await?;
        Ok(())
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`CandleLlmError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn load(&self) -> Result<(), CandleLlmError> {
        Err(CandleLlmError::EngineNotAvailable)
    }

    /// Drop the loaded engine and free its VRAM / memory. Idempotent --
    /// if the engine is already unloaded, this is a no-op that returns
    /// `Ok(())`.
    ///
    /// Note: if an inference task is still holding an `OwnedMutexGuard`
    /// (which is passed into `spawn_blocking` for the whole generation
    /// loop), `unload` will wait for the mutex and only then set the
    /// inner `Option` to `None`. This serialises unload vs in-flight
    /// inference correctly; the engine is dropped the moment no other
    /// guard is alive.
    ///
    /// # Errors
    ///
    /// This method currently never returns an error; the `Result` return
    /// type is preserved to match [`crate::CandleLlmError`] conventions
    /// and the [`blazen_llm::traits::LocalModel`] trait contract.
    #[cfg(feature = "engine")]
    pub async fn unload(&self) -> Result<(), CandleLlmError> {
        let mut guard = self.engine.lock().await;
        // Drop the CandleEngine. VRAM is freed by the Drop impls on the
        // inner `ModelWeights` / `Device` handles.
        *guard = None;
        Ok(())
    }

    /// Stub: engine not available. Always succeeds as a no-op, matching
    /// the idempotent-unload contract even when there is no engine to
    /// unload in the first place.
    ///
    /// # Errors
    ///
    /// This method never returns an error.
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn unload(&self) -> Result<(), CandleLlmError> {
        Ok(())
    }

    /// Whether the engine is currently loaded in memory / VRAM.
    #[cfg(feature = "engine")]
    pub async fn is_loaded(&self) -> bool {
        self.engine.lock().await.is_some()
    }

    /// Stub: without the engine feature there is never a loaded model,
    /// so this always returns `false`.
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn is_loaded(&self) -> bool {
        false
    }

    /// Acquire an [`OwnedMutexGuard`] over the engine slot, loading the
    /// engine lazily if the slot is empty.
    ///
    /// The returned guard holds the `Mutex<Option<CandleEngine>>` locked
    /// for as long as the caller keeps it alive. Callers pass it into
    /// `tokio::task::spawn_blocking` so that the lock stays held across
    /// the async-to-blocking thread hop; this guarantees that an
    /// `unload` call in another task cannot yank the engine out from
    /// under an in-flight inference.
    ///
    /// Holding the guard across the `CandleEngine::load(...)` await is
    /// intentional -- it serialises concurrent loaders onto a single
    /// `HuggingFace` download.
    ///
    /// [`OwnedMutexGuard`]: tokio::sync::OwnedMutexGuard
    #[cfg(feature = "engine")]
    async fn get_or_load_engine_guard(
        &self,
    ) -> Result<tokio::sync::OwnedMutexGuard<Option<LoadedEngine>>, CandleLlmError> {
        let mut guard = std::sync::Arc::clone(&self.engine).lock_owned().await;
        if guard.is_none() {
            let pre_registered: Vec<MountedAdapterRecord> = {
                let adapters_guard = self.loaded_adapters.read().await;
                adapters_guard.values().cloned().collect()
            };
            let eng = load_with_autodetect(&self.options, &pre_registered).await?;
            *guard = Some(eng);
        }
        Ok(guard)
    }

    /// Run a non-streaming inference on the given messages.
    ///
    /// Each message is a `(role, content)` pair where role is one of
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::EngineNotAvailable`] if the `engine`
    /// feature is not compiled in.
    /// Returns [`CandleLlmError::Inference`] if generation fails.
    #[allow(clippy::unused_async)] // async is required when `engine` feature is on
    pub async fn infer(
        &self,
        messages: Vec<(String, String)>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<CandleInferenceResult, CandleLlmError> {
        #[cfg(feature = "engine")]
        {
            // Acquire an owned guard in async-land: this loads the
            // engine if needed and keeps the Mutex locked across the
            // transition into `spawn_blocking`, so `unload` cannot
            // yank the engine out mid-inference.
            let mut owned_guard = self.get_or_load_engine_guard().await?;
            let max_tokens = max_tokens.unwrap_or(512);
            let start = std::time::Instant::now();

            let result = tokio::task::spawn_blocking(move || {
                // `get_or_load_engine_guard` populated the Option
                // before returning, and we hold the exclusive lock, so
                // no other task could have reset it. Guard against the
                // impossible `None` case with a structured error
                // instead of a panic so the public API surface stays
                // panic-free.
                let loaded = owned_guard.as_mut().ok_or_else(|| {
                    CandleLlmError::Inference(
                        "engine slot empty under the mutex after successful load".into(),
                    )
                })?;

                match loaded {
                    LoadedEngine::Gguf(eng) => {
                        let prompt = engine::format_prompt(&messages);
                        let prompt_token_count = eng
                            .tokenizer
                            .encode(prompt.as_str(), true)
                            .map_or(0, |enc| enc.get_ids().len());
                        let (text, completion_tokens) =
                            engine::generate(eng, &prompt, max_tokens, temperature, top_p)?;
                        Ok::<_, CandleLlmError>((text, prompt_token_count, completion_tokens))
                    }
                    LoadedEngine::Safetensors(eng) => {
                        let prompt = crate::safetensors_engine::format_prompt(eng, &messages);
                        let prompt_token_count = eng
                            .tokenizer
                            .encode(prompt.as_str(), true)
                            .map_or(0, |enc| enc.get_ids().len());
                        let (text, completion_tokens) = crate::safetensors_engine::infer_engine(
                            eng,
                            &prompt,
                            max_tokens,
                            temperature,
                            top_p,
                        )?;
                        Ok::<_, CandleLlmError>((text, prompt_token_count, completion_tokens))
                    }
                }
                // `owned_guard` is dropped here, releasing the mutex.
            })
            .await
            .map_err(|e| CandleLlmError::Inference(format!("blocking task failed: {e}")))??;

            let elapsed = start.elapsed();
            let (text, prompt_tokens, completion_tokens) = result;

            Ok(CandleInferenceResult {
                content: text,
                prompt_tokens,
                completion_tokens,
                total_time_secs: elapsed.as_secs_f64(),
            })
        }
        #[cfg(not(feature = "engine"))]
        {
            let _ = (messages, max_tokens, temperature, top_p);
            Err(CandleLlmError::EngineNotAvailable)
        }
    }

    /// Run a streaming inference, returning each generated token via a
    /// channel.
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::EngineNotAvailable`] if the `engine`
    /// feature is not compiled in.
    #[allow(clippy::unused_async)] // async is required when `engine` feature is on
    pub async fn infer_stream(
        &self,
        messages: Vec<(String, String)>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, CandleLlmError>>, CandleLlmError> {
        #[cfg(feature = "engine")]
        {
            // Acquire the owned guard up front: this loads the engine
            // if needed and keeps the mutex locked for the whole
            // streaming task so that `unload` cannot yank it out.
            let mut owned_guard = self.get_or_load_engine_guard().await?;
            let max_tokens = max_tokens.unwrap_or(512);
            let (tx, rx) = tokio::sync::mpsc::channel(64);

            tokio::task::spawn_blocking(move || {
                // `get_or_load_engine_guard` populated the Option
                // before returning, and we hold the exclusive lock, so
                // no other task could have reset it. Guard against the
                // impossible `None` case by emitting a structured
                // error on the channel instead of panicking, so the
                // public API surface stays panic-free.
                let Some(loaded) = owned_guard.as_mut() else {
                    let _ = tx.blocking_send(Err(CandleLlmError::Inference(
                        "engine slot empty under the mutex after successful load".into(),
                    )));
                    return;
                };

                let result = match loaded {
                    LoadedEngine::Gguf(eng) => {
                        let prompt = engine::format_prompt(&messages);
                        engine::generate_streaming(
                            eng,
                            &prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            |token_text| {
                                let _ = tx.blocking_send(Ok(token_text));
                            },
                        )
                        .map(|_| ())
                    }
                    LoadedEngine::Safetensors(eng) => {
                        let prompt = crate::safetensors_engine::format_prompt(eng, &messages);
                        crate::safetensors_engine::infer_stream_engine(
                            eng,
                            &prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            |token_text| {
                                let _ = tx.blocking_send(Ok(token_text));
                            },
                        )
                        .map(|_| ())
                    }
                };

                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e));
                }
                // `owned_guard` and `tx` are both dropped here: the
                // mutex is released and the channel closes.
            });

            Ok(rx)
        }
        #[cfg(not(feature = "engine"))]
        {
            let _ = (messages, max_tokens, temperature, top_p);
            Err(CandleLlmError::EngineNotAvailable)
        }
    }

    // -----------------------------------------------------------------------
    // LoRA adapter mount / unmount / list
    //
    // The current candle engine runs `quantized_llama::ModelWeights`,
    // a monolithic GGUF model whose forward pass exposes no per-`Linear`
    // hook. The PEFT adapter format is fully parsed + validated here (so
    // bad inputs are caught early) but `load_adapter` returns a clear
    // `Unsupported` error rather than silently mounting a no-op delta.
    // The parsed [`crate::lora::LoadedAdapter`] machinery is the
    // building block future per-architecture wrappers (e.g. a hand-built
    // Qwen2 / Llama2 forward path that exposes per-layer `Linear`s) will
    // consume to apply the deltas.
    // -----------------------------------------------------------------------

    /// Mount a PEFT-format `LoRA` adapter onto the loaded base model.
    ///
    /// `adapter_dir` must contain `adapter_config.json` and
    /// `adapter_model.safetensors` (the PEFT canonical layout). The
    /// adapter is parsed + validated immediately, then registered in
    /// the provider's adapter table. The action taken depends on which
    /// engine variant is currently loaded:
    ///
    /// * [`LoadedEngine::Safetensors`] — the engine is rebuilt with the
    ///   merged set of adapters; the returned record reports a
    ///   `Rebuilt` mount strategy at the caller-trait layer.
    /// * [`LoadedEngine::Gguf`] — `LoRA` on the quantized GGUF path
    ///   requires dequantize-merge-requantize work that is not
    ///   implemented here, so a clear [`CandleLlmError::Unsupported`]
    ///   is returned.
    ///
    /// If the engine is not yet loaded, the adapter is recorded against
    /// the empty engine slot and the next [`Self::load`] / inference
    /// call will pull the adapter set into the initial build via
    /// [`crate::safetensors_engine::SafetensorsEngine::load_with_adapters`]
    /// (subject to repo format: GGUF auto-detects still error with
    /// `Unsupported`).
    ///
    /// # Errors
    ///
    /// - [`CandleLlmError::EngineNotAvailable`] without the `engine` feature.
    /// - [`CandleLlmError::InvalidOptions`] if the directory is missing
    ///   either canonical file, or `adapter_config.json` fails parsing.
    /// - [`CandleLlmError::ModelLoad`] if the safetensors file fails to
    ///   load or its keys violate the PEFT pairing convention, or if
    ///   the safetensors-path rebuild fails.
    /// - [`CandleLlmError::Unsupported`] when the active engine is the
    ///   GGUF / quantized path.
    #[cfg(feature = "engine")]
    pub async fn load_adapter(
        &self,
        adapter_dir: &Path,
        adapter_id: String,
        scale: f32,
    ) -> Result<MountedAdapterRecord, CandleLlmError> {
        if !adapter_dir.is_dir() {
            return Err(CandleLlmError::InvalidOptions(format!(
                "adapter_dir does not exist or is not a directory: {}",
                adapter_dir.display()
            )));
        }
        let cfg_path = adapter_dir.join("adapter_config.json");
        let weights_path = adapter_dir.join("adapter_model.safetensors");
        if !cfg_path.is_file() {
            return Err(CandleLlmError::InvalidOptions(format!(
                "adapter_config.json missing at {}",
                cfg_path.display()
            )));
        }
        if !weights_path.is_file() {
            return Err(CandleLlmError::InvalidOptions(format!(
                "adapter_model.safetensors missing at {}",
                weights_path.display()
            )));
        }

        {
            let guard = self.loaded_adapters.read().await;
            if guard.contains_key(&adapter_id) {
                return Err(CandleLlmError::InvalidOptions(format!(
                    "adapter '{adapter_id}' is already mounted"
                )));
            }
        }

        // Why: validate immediately against the active engine's device
        // when one is loaded, so the `(B@A)` precompute lands on the
        // same device used by inference. Falls back to CPU if no
        // engine is loaded — the rebuild path re-parses on the engine
        // device anyway.
        let device = {
            let guard = self.engine.lock().await;
            match guard.as_ref() {
                Some(LoadedEngine::Safetensors(eng)) => eng.device.clone(),
                Some(LoadedEngine::Gguf(eng)) => eng.device.clone(),
                None => candle_core::Device::Cpu,
            }
        };
        crate::lora::LoadedAdapter::from_dir(adapter_dir, adapter_id.clone(), scale, &device)?;

        let record = MountedAdapterRecord {
            adapter_id: adapter_id.clone(),
            adapter_dir: adapter_dir.to_path_buf(),
            scale,
        };

        let mut engine_guard = self.engine.lock().await;
        match engine_guard.as_ref() {
            Some(LoadedEngine::Gguf(_)) => {
                return Err(CandleLlmError::Unsupported(format!(
                    "candle GGUF + LoRA requires dequantize-merge-requantize; \
                     use safetensors format or the mistralrs backend \
                     (adapter '{adapter_id}' parsed cleanly but cannot be attached)"
                )));
            }
            Some(LoadedEngine::Safetensors(_)) => {
                let next_records: Vec<MountedAdapterRecord> = {
                    let adapters_guard = self.loaded_adapters.read().await;
                    let mut list: Vec<MountedAdapterRecord> =
                        adapters_guard.values().cloned().collect();
                    list.push(record.clone());
                    list
                };
                let parsed = parse_adapters_on_device(&next_records, &device)?;
                let new_engine = crate::safetensors_engine::SafetensorsEngine::load_with_adapters(
                    &self.options,
                    &parsed,
                )
                .await?;
                *engine_guard = Some(LoadedEngine::Safetensors(new_engine));
            }
            None => {
                // Why: no engine loaded yet — just register; the next
                // load call will pull `loaded_adapters` into the
                // initial build.
            }
        }
        drop(engine_guard);

        let mut adapters_guard = self.loaded_adapters.write().await;
        adapters_guard.insert(adapter_id, record.clone());
        Ok(record)
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`CandleLlmError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        adapter_id: String,
        scale: f32,
    ) -> Result<MountedAdapterRecord, CandleLlmError> {
        let _ = (adapter_dir, adapter_id, scale);
        Err(CandleLlmError::EngineNotAvailable)
    }

    /// Remove a previously-mounted adapter by id. Idempotent — removing
    /// a non-existent id returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Currently never returns an error; the `Result` is preserved for
    /// API consistency with the rest of the [`CandleLlmProvider`]
    /// surface (and to forward-compat against a future implementation
    /// that does perform fallible work).
    #[cfg(feature = "engine")]
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), CandleLlmError> {
        let removed_existed = {
            let mut guard = self.loaded_adapters.write().await;
            guard.remove(adapter_id).is_some()
        };
        if !removed_existed {
            return Ok(());
        }

        let mut engine_guard = self.engine.lock().await;
        match engine_guard.as_ref() {
            Some(LoadedEngine::Safetensors(eng)) => {
                let device = eng.device.clone();
                let remaining: Vec<MountedAdapterRecord> = {
                    let adapters_guard = self.loaded_adapters.read().await;
                    adapters_guard.values().cloned().collect()
                };
                let parsed = parse_adapters_on_device(&remaining, &device)?;
                let new_engine = crate::safetensors_engine::SafetensorsEngine::load_with_adapters(
                    &self.options,
                    &parsed,
                )
                .await?;
                *engine_guard = Some(LoadedEngine::Safetensors(new_engine));
            }
            // Why: on GGUF the load_adapter path would have errored
            // before recording the adapter, so we should never reach
            // here with a Gguf engine + non-empty removal. Defensive
            // no-op keeps the rebuild scoped to safetensors.
            Some(LoadedEngine::Gguf(_)) | None => {}
        }
        Ok(())
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`CandleLlmError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), CandleLlmError> {
        let _ = adapter_id;
        Err(CandleLlmError::EngineNotAvailable)
    }

    /// Snapshot of the currently-mounted adapters.
    #[cfg(feature = "engine")]
    pub async fn list_adapters(&self) -> Vec<MountedAdapterRecord> {
        self.loaded_adapters
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Stub: without the engine feature there are never any adapters.
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn list_adapters(&self) -> Vec<MountedAdapterRecord> {
        Vec::new()
    }
}

/// Public empty-shaped stub of [`MountedAdapterRecord`] for builds
/// without the `engine` feature so the [`CandleLlmProvider::list_adapters`]
/// signature stays identical across feature configurations.
#[cfg(not(feature = "engine"))]
#[derive(Debug, Clone)]
pub struct MountedAdapterRecord {
    pub adapter_id: String,
    pub adapter_dir: PathBuf,
    pub scale: f32,
}

// ---------------------------------------------------------------------------
// Inference result
// ---------------------------------------------------------------------------

/// Result from a non-streaming candle inference call.
#[derive(Debug, Clone)]
pub struct CandleInferenceResult {
    /// The generated text content.
    pub content: String,
    /// Number of prompt tokens consumed.
    pub prompt_tokens: usize,
    /// Number of completion tokens generated.
    pub completion_tokens: usize,
    /// Wall-clock time for the inference in seconds.
    pub total_time_secs: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CandleLlmOptions;
    use blazen_local_llm::LocalLlmOptions;

    #[test]
    fn from_options_with_defaults() {
        let opts = CandleLlmOptions::default();
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_id().is_none());
    }

    #[test]
    fn from_options_with_model_id() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions::new().with_model_id("meta-llama/Llama-3.2-1B"),
            ..CandleLlmOptions::default()
        };
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.model_id(), Some("meta-llama/Llama-3.2-1B"));
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions {
                model_id: Some(String::new()),
                ..LocalLlmOptions::default()
            },
            ..CandleLlmOptions::default()
        };
        let result = CandleLlmProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_rejects_empty_device() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions {
                device: Some(String::new()),
                ..LocalLlmOptions::default()
            },
            ..CandleLlmOptions::default()
        };
        let result = CandleLlmProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_accepts_valid_device() {
        let opts = CandleLlmOptions {
            base: LocalLlmOptions::new().with_device("cuda:0"),
            ..CandleLlmOptions::default()
        };
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_id().is_none());
    }

    #[test]
    fn engine_not_available_display() {
        let err = CandleLlmError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[test]
    fn engine_available_reflects_feature() {
        let provider = CandleLlmProvider::from_options(CandleLlmOptions::default()).unwrap();
        if cfg!(feature = "engine") {
            assert!(provider.engine_available());
        } else {
            assert!(!provider.engine_available());
        }
    }

    #[cfg(feature = "engine")]
    #[test]
    fn format_prompt_produces_expected_output() {
        let messages = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello!".to_string()),
        ];
        let prompt = engine::format_prompt(&messages);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    /// Integration test that downloads a model and runs inference.
    /// Ignored by default because it requires network access and downloads
    /// a large model file.
    #[tokio::test]
    #[ignore = "downloads a large model + needs network"]
    async fn test_candle_llm_inference() {
        // TheBloke's *-GGUF repos ship the GGUF weights but NOT
        // tokenizer.json. Point tokenizer_repo at the upstream
        // TinyLlama repo for the tokenizer.
        let opts = CandleLlmOptions {
            base: LocalLlmOptions::new()
                .with_model_id("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                .with_tokenizer_repo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                .with_quantization("Q4_K_M")
                .with_context_length(2048),
            ..CandleLlmOptions::default()
        };

        let provider = CandleLlmProvider::from_options(opts).expect("options valid");
        provider.load().await.expect("model should load");

        let messages = vec![
            (
                "system".to_string(),
                "You are a helpful assistant.".to_string(),
            ),
            ("user".to_string(), "What is 2 + 2?".to_string()),
        ];

        let result = provider
            .infer(messages, Some(64), Some(0.1), None)
            .await
            .expect("inference should succeed");

        assert!(!result.content.is_empty(), "should produce text");
        assert!(result.completion_tokens > 0, "should produce tokens");
    }
}
