//! The [`CandleLlmProvider`] type -- local LLM inference via candle.
//!
//! When the `engine` feature is enabled, this module provides a fully
//! functional local LLM backend using GGUF quantized models loaded through
//! `candle-transformers`. Without the feature, the provider compiles but
//! returns [`CandleLlmError::EngineNotAvailable`] for all inference calls.

use std::fmt;

use crate::CandleLlmOptions;

/// Error type for candle LLM operations.
#[derive(Debug)]
pub enum CandleLlmError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// An inference operation failed.
    Inference(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for CandleLlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "candle LLM invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "candle LLM model load failed: {msg}"),
            Self::Inference(msg) => write!(f, "candle LLM inference failed: {msg}"),
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
            #[cfg(feature = "cuda")]
            s if s.starts_with("cuda") => {
                let ordinal = s
                    .strip_prefix("cuda:")
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(0);
                Device::new_cuda(ordinal)
                    .map_err(|e| CandleLlmError::ModelLoad(format!("CUDA device error: {e}")))
            }
            #[cfg(feature = "metal")]
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
            let repo_id = opts.model_id.as_deref().ok_or_else(|| {
                CandleLlmError::InvalidOptions("model_id is required for engine init".into())
            })?;

            let device = resolve_device(opts.device.as_deref())?;
            let revision = opts.revision.as_deref();
            let quantization = opts.quantization.as_deref();

            // Download model and tokenizer in parallel.
            let (gguf_path, tokenizer) = tokio::try_join!(
                download_gguf(repo_id, revision, quantization),
                download_tokenizer(repo_id, revision),
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

            let context_length = opts.context_length.unwrap_or(4096);

            tracing::info!(
                repo = repo_id,
                context_length,
                device = ?opts.device,
                "candle LLM engine loaded"
            );

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
    /// The loaded engine state (only present when `engine` feature is on
    /// and [`CandleLlmProvider::load`] has been called).
    #[cfg(feature = "engine")]
    engine: Option<std::sync::Arc<tokio::sync::Mutex<engine::CandleEngine>>>,
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
        if let Some(ref model_id) = opts.model_id
            && model_id.is_empty()
        {
            return Err(CandleLlmError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(CandleLlmError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        Ok(Self {
            model_id: opts.model_id.clone(),
            options: opts,
            #[cfg(feature = "engine")]
            engine: None,
        })
    }

    /// The model identifier that was passed at construction time.
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Whether the engine feature is compiled in.
    #[must_use]
    pub fn engine_available(&self) -> bool {
        cfg!(feature = "engine")
    }

    /// Eagerly load the candle engine.
    ///
    /// Downloads the model weights and tokenizer from `HuggingFace` Hub
    /// (if not already cached) and initialises the inference engine.
    ///
    /// This is optional -- the engine is also loaded lazily on the first
    /// inference call. Call this method when you want to control when the
    /// (potentially slow) download and load happens.
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::EngineNotAvailable`] if the `engine`
    /// feature is not compiled in.
    /// Returns [`CandleLlmError::ModelLoad`] if the download or load fails.
    #[allow(clippy::unused_async)] // async is required when `engine` feature is on
    pub async fn load(&mut self) -> Result<(), CandleLlmError> {
        #[cfg(feature = "engine")]
        {
            let eng = engine::CandleEngine::load(&self.options).await?;
            self.engine = Some(std::sync::Arc::new(tokio::sync::Mutex::new(eng)));
            Ok(())
        }
        #[cfg(not(feature = "engine"))]
        {
            Err(CandleLlmError::EngineNotAvailable)
        }
    }

    /// Ensure the engine is loaded, loading it lazily if needed.
    #[cfg(feature = "engine")]
    async fn ensure_engine(
        &mut self,
    ) -> Result<std::sync::Arc<tokio::sync::Mutex<engine::CandleEngine>>, CandleLlmError> {
        if self.engine.is_none() {
            self.load().await?;
        }
        self.engine.clone().ok_or(CandleLlmError::Inference(
            "engine failed to initialise".into(),
        ))
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
        &mut self,
        messages: Vec<(String, String)>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<CandleInferenceResult, CandleLlmError> {
        #[cfg(feature = "engine")]
        {
            let engine_arc = self.ensure_engine().await?;
            let max_tokens = max_tokens.unwrap_or(512);
            let start = std::time::Instant::now();

            let result = tokio::task::spawn_blocking(move || {
                let mut engine = engine_arc.blocking_lock();
                let prompt = engine::format_prompt(&messages);

                let prompt_token_count = engine
                    .tokenizer
                    .encode(prompt.as_str(), true)
                    .map(|enc| enc.get_ids().len())
                    .unwrap_or(0);

                let (text, completion_tokens) =
                    engine::generate(&mut engine, &prompt, max_tokens, temperature, top_p)?;

                Ok::<_, CandleLlmError>((text, prompt_token_count, completion_tokens))
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
        &mut self,
        messages: Vec<(String, String)>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, CandleLlmError>>, CandleLlmError> {
        #[cfg(feature = "engine")]
        {
            let engine_arc = self.ensure_engine().await?;
            let max_tokens = max_tokens.unwrap_or(512);
            let (tx, rx) = tokio::sync::mpsc::channel(64);

            tokio::task::spawn_blocking(move || {
                let mut engine = engine_arc.blocking_lock();
                let prompt = engine::format_prompt(&messages);

                let result = engine::generate_streaming(
                    &mut engine,
                    &prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    |token_text| {
                        let _ = tx.blocking_send(Ok(token_text));
                    },
                );

                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e));
                }
                // tx is dropped here, closing the channel.
            });

            Ok(rx)
        }
        #[cfg(not(feature = "engine"))]
        {
            let _ = (messages, max_tokens, temperature, top_p);
            Err(CandleLlmError::EngineNotAvailable)
        }
    }
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

    #[test]
    fn from_options_with_defaults() {
        let opts = CandleLlmOptions::default();
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert!(provider.model_id().is_none());
    }

    #[test]
    fn from_options_with_model_id() {
        let opts = CandleLlmOptions {
            model_id: Some("meta-llama/Llama-3.2-1B".into()),
            ..CandleLlmOptions::default()
        };
        let provider = CandleLlmProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.model_id(), Some("meta-llama/Llama-3.2-1B"));
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let opts = CandleLlmOptions {
            model_id: Some(String::new()),
            ..CandleLlmOptions::default()
        };
        let result = CandleLlmProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_rejects_empty_device() {
        let opts = CandleLlmOptions {
            device: Some(String::new()),
            ..CandleLlmOptions::default()
        };
        let result = CandleLlmProvider::from_options(opts);
        assert!(result.is_err());
    }

    #[test]
    fn from_options_accepts_valid_device() {
        let opts = CandleLlmOptions {
            device: Some("cuda:0".into()),
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
    #[ignore]
    async fn test_candle_llm_inference() {
        let opts = CandleLlmOptions {
            model_id: Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into()),
            quantization: Some("Q4_K_M".into()),
            context_length: Some(2048),
            ..CandleLlmOptions::default()
        };

        let mut provider = CandleLlmProvider::from_options(opts).expect("options valid");
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
