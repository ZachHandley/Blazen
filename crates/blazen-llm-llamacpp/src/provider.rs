//! The [`LlamaCppProvider`] type -- local LLM inference via llama.cpp.
//!
//! When the `engine` feature is enabled, constructing a provider will load the
//! GGUF model file via the `llama-cpp-2` bindings and prepare a llama.cpp
//! context for inference. Without the feature, the provider validates options
//! but returns [`LlamaCppError::EngineNotAvailable`] from inference methods.

use std::fmt;
use std::pin::Pin;

use crate::LlamaCppOptions;

/// Type alias for the boxed inference chunk stream returned by streaming methods.
pub type InferenceChunkStream =
    Pin<Box<dyn futures_util::Stream<Item = Result<InferenceChunk, LlamaCppError>> + Send>>;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error type for llama.cpp operations.
#[derive(Debug)]
pub enum LlamaCppError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be loaded.
    ModelLoad(String),
    /// An inference operation failed.
    Inference(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for LlamaCppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "llama.cpp invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "llama.cpp model load failed: {msg}"),
            Self::Inference(msg) => write!(f, "llama.cpp inference failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "llama.cpp engine not available: compile with the `engine` feature"
            ),
        }
    }
}

impl std::error::Error for LlamaCppError {}

// ---------------------------------------------------------------------------
// Inference result types (always available)
// ---------------------------------------------------------------------------

/// Result of a single non-streaming inference call.
#[derive(Debug)]
pub struct InferenceResult {
    /// The generated text content, if any.
    pub content: Option<String>,
    /// Why the model stopped generating.
    pub finish_reason: String,
    /// The model identifier that produced this result.
    pub model: String,
    /// Token usage statistics.
    pub usage: InferenceUsage,
}

/// Token usage from the engine.
#[derive(Debug, Clone, Default)]
pub struct InferenceUsage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
    /// Total wall-clock inference time in seconds.
    pub total_time_sec: f32,
}

/// A single chunk from streaming inference.
#[derive(Debug, Clone)]
pub struct InferenceChunk {
    /// Incremental text content.
    pub delta: Option<String>,
    /// Present in the final chunk when generation stops.
    pub finish_reason: Option<String>,
}

/// Simplified chat role for the llama.cpp bridge layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single chat message for input to the provider.
#[derive(Debug, Clone)]
pub struct ChatMessageInput {
    /// Who produced this message.
    pub role: ChatRole,
    /// The textual content of the message.
    pub text: String,
}

impl ChatMessageInput {
    /// Build a chat message.
    #[must_use]
    pub fn new(role: ChatRole, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A local LLM provider backed by [`llama.cpp`](https://github.com/ggerganov/llama.cpp).
///
/// Constructed via [`LlamaCppProvider::from_options`]. When the `engine`
/// feature is active, the model is loaded lazily on the first inference
/// call. Without the feature, only validation is performed and inference
/// methods return [`LlamaCppError::EngineNotAvailable`].
pub struct LlamaCppProvider {
    /// The model path that was requested.
    model_path: Option<String>,
    /// Full options preserved for deferred engine initialisation.
    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    options: LlamaCppOptions,
    /// Lazily loaded llama.cpp engine handle.
    ///
    /// Wrapped in `Arc<OnceCell>` so that:
    /// - The model is loaded once on first use (not at construction time).
    /// - Streaming can spawn a `'static` task that borrows the engine
    ///   independently of `&self`.
    #[cfg(feature = "engine")]
    engine: std::sync::Arc<tokio::sync::OnceCell<Engine>>,
}

/// Inner engine state. Only compiled with the `engine` feature.
#[cfg(feature = "engine")]
struct Engine {
    backend: llama_cpp_2::llama_backend::LlamaBackend,
    model: llama_cpp_2::model::LlamaModel,
    context_length: u32,
    model_id: String,
}

impl LlamaCppProvider {
    /// Create a new provider from the given options.
    ///
    /// This is always synchronous -- the model is loaded lazily on first
    /// inference when the `engine` feature is active. Without the feature,
    /// only validation is performed and inference methods return
    /// [`LlamaCppError::EngineNotAvailable`].
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::InvalidOptions`] if a specified string field
    /// is present but empty.
    #[allow(clippy::unused_async)]
    pub async fn from_options(opts: LlamaCppOptions) -> Result<Self, LlamaCppError> {
        Self::validate_options(&opts)?;
        let model_path = opts.model_path.clone();
        Ok(Self {
            model_path,
            #[cfg(feature = "engine")]
            engine: std::sync::Arc::new(tokio::sync::OnceCell::new()),
            options: opts,
        })
    }

    /// Validate the options without loading the engine.
    fn validate_options(opts: &LlamaCppOptions) -> Result<(), LlamaCppError> {
        if let Some(ref model_path) = opts.model_path
            && model_path.is_empty()
        {
            return Err(LlamaCppError::InvalidOptions(
                "model_path must not be empty when specified".into(),
            ));
        }

        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(LlamaCppError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        Ok(())
    }

    /// The model path that was passed at construction time.
    #[must_use]
    pub fn model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
    }

    /// Whether the engine feature is compiled in.
    #[must_use]
    pub fn engine_available(&self) -> bool {
        cfg!(feature = "engine")
    }

    /// The model identifier (the model path or repo ID).
    #[must_use]
    pub fn model_id(&self) -> &str {
        self.model_path.as_deref().unwrap_or("llamacpp")
    }

    // -----------------------------------------------------------------------
    // Inference methods (always available in the public API)
    // -----------------------------------------------------------------------

    /// Run non-streaming inference and return the result.
    ///
    /// When the `engine` feature is not enabled, returns
    /// [`LlamaCppError::EngineNotAvailable`].
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::Inference`] on engine failure, or
    /// [`LlamaCppError::EngineNotAvailable`] if the engine is not compiled in.
    #[cfg(feature = "engine")]
    pub async fn infer(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceResult, LlamaCppError> {
        self.infer_engine(messages).await
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`LlamaCppError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn infer(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceResult, LlamaCppError> {
        let _ = messages;
        Err(LlamaCppError::EngineNotAvailable)
    }

    /// Run streaming inference and return a `'static` stream of chunks.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::Inference`] if the stream cannot be started,
    /// or [`LlamaCppError::EngineNotAvailable`] if the engine is not compiled in.
    #[cfg(feature = "engine")]
    pub async fn infer_stream(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceChunkStream, LlamaCppError> {
        // Ensure the engine is loaded before spawning the blocking stream task.
        let _ = self.get_engine().await?;
        Ok(self.infer_stream_engine(messages))
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`LlamaCppError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn infer_stream(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceChunkStream, LlamaCppError> {
        let _ = messages;
        Err(LlamaCppError::EngineNotAvailable)
    }
}

// ---------------------------------------------------------------------------
// Engine construction (only with `engine` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
async fn resolve_model_path(opts: &LlamaCppOptions) -> Result<std::path::PathBuf, LlamaCppError> {
    let model_path = opts.model_path.as_deref().ok_or_else(|| {
        LlamaCppError::InvalidOptions("model_path is required for llama.cpp inference".into())
    })?;

    let path = std::path::Path::new(model_path);

    // If it's already a local file, use it directly.
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // If it looks like a HuggingFace repo ID (contains a '/'), attempt download.
    if model_path.contains('/') {
        // Expect format: "org/repo/filename.gguf" or "org/repo" with
        // auto-detection. We split on the last '/' to get repo and filename.
        let (repo_id, filename) = if model_path.matches('/').count() >= 2 {
            // "org/repo/filename.gguf" -> repo="org/repo", file="filename.gguf"
            let last_slash = model_path.rfind('/').unwrap();
            (&model_path[..last_slash], &model_path[last_slash + 1..])
        } else {
            return Err(LlamaCppError::InvalidOptions(format!(
                "model_path \"{model_path}\" looks like a HuggingFace repo ID \
                 but no filename was specified. Use format: \
                 \"org/repo/model-file.gguf\""
            )));
        };

        tracing::info!(
            repo_id = repo_id,
            filename = filename,
            "downloading model from HuggingFace Hub"
        );

        let cache = if let Some(ref cache_dir) = opts.cache_dir {
            blazen_model_cache::ModelCache::with_dir(cache_dir.clone())
        } else {
            blazen_model_cache::ModelCache::new()
                .map_err(|e| LlamaCppError::ModelLoad(format!("failed to init model cache: {e}")))?
        };

        let local_path: std::path::PathBuf = cache
            .download(repo_id, filename, None)
            .await
            .map_err(|e| LlamaCppError::ModelLoad(format!("failed to download model: {e}")))?;

        return Ok(local_path);
    }

    Err(LlamaCppError::ModelLoad(format!(
        "model file not found: {model_path}"
    )))
}

#[cfg(feature = "engine")]
async fn build_engine(opts: &LlamaCppOptions) -> Result<Engine, LlamaCppError> {
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    let model_path = resolve_model_path(opts).await?;
    let model_id = model_path.file_name().map_or_else(
        || "llamacpp".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );

    tracing::info!(
        model_path = %model_path.display(),
        "loading llama.cpp model"
    );

    // Build engine on a blocking thread since model loading is CPU-heavy.
    let context_length = opts.context_length;
    let n_gpu_layers = opts.n_gpu_layers;

    let engine = tokio::task::spawn_blocking(move || -> Result<Engine, LlamaCppError> {
        let mut backend = LlamaBackend::init()
            .map_err(|e| LlamaCppError::ModelLoad(format!("failed to init llama backend: {e}")))?;

        // Suppress llama.cpp's own stderr logging in favour of tracing.
        backend.void_logs();

        let mut model_params = LlamaModelParams::default();
        if let Some(n_gpu) = n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(n_gpu);
        }

        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| LlamaCppError::ModelLoad(format!("{e}")))?;

        // Use the requested context length, or fall back to the model's training length.
        let ctx_len = context_length.map_or_else(
            || model.n_ctx_train(),
            #[allow(clippy::cast_possible_truncation)]
            |c| c as u32,
        );

        Ok(Engine {
            backend,
            model,
            context_length: ctx_len,
            model_id,
        })
    })
    .await
    .map_err(|e| LlamaCppError::ModelLoad(format!("spawn_blocking panicked: {e}")))??;

    Ok(engine)
}

// ---------------------------------------------------------------------------
// Chat template formatting (only with `engine` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
fn format_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    messages: &[ChatMessageInput],
) -> Result<String, LlamaCppError> {
    use llama_cpp_2::model::LlamaChatMessage;

    // Try to use the model's built-in chat template first.
    let chat_messages: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };
            LlamaChatMessage::new(role.to_string(), msg.text.clone()).map_err(|e| {
                LlamaCppError::Inference(format!("failed to create chat message: {e}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Retrieve the model's chat template, falling back to ChatML if none is found.
    let template = model.chat_template(None).unwrap_or_else(|_| {
        llama_cpp_2::model::LlamaChatTemplate::new("chatml")
            .expect("chatml template should always be valid")
    });

    model
        .apply_chat_template(&template, &chat_messages, true)
        .map_err(|e| LlamaCppError::Inference(format!("failed to apply chat template: {e}")))
}

/// Simple fallback prompt formatting when the model's template is not available.
#[cfg(feature = "engine")]
fn format_prompt_chatml(messages: &[ChatMessageInput]) -> String {
    use std::fmt::Write;

    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        let _ = write!(prompt, "<|im_start|>{role}\n{}<|im_end|>\n", msg.text);
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ---------------------------------------------------------------------------
// Engine-backed inference (only with `engine` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
impl LlamaCppProvider {
    /// Lazily initialise the engine on first use.
    async fn get_engine(&self) -> Result<&Engine, LlamaCppError> {
        let opts = self.options.clone();
        self.engine
            .get_or_try_init(|| async { build_engine(&opts).await })
            .await
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    async fn infer_engine(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceResult, LlamaCppError> {
        // Ensure the engine is loaded, then clone the Arc for the blocking task.
        let _ = self.get_engine().await?;
        let engine = std::sync::Arc::clone(&self.engine);

        tokio::task::spawn_blocking(move || {
            use llama_cpp_2::context::params::LlamaContextParams;
            use llama_cpp_2::llama_batch::LlamaBatch;
            use llama_cpp_2::model::AddBos;
            use llama_cpp_2::sampling::LlamaSampler;
            use std::num::NonZeroU32;

            // Engine is guaranteed initialized by the `get_engine()` call above.
            let eng = engine.get().expect("engine should be initialized");

            let prompt = format_prompt(&eng.model, &messages)
                .or_else(|_| Ok::<String, LlamaCppError>(format_prompt_chatml(&messages)))?;

            // Tokenize the prompt.
            let tokens = eng
                .model
                .str_to_token(&prompt, AddBos::Always)
                .map_err(|e| LlamaCppError::Inference(format!("tokenization failed: {e}")))?;

            let prompt_token_count = tokens.len() as u32;

            // Create a context for this inference call.
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(eng.context_length))
                .with_n_batch(eng.context_length);

            let mut ctx = eng
                .model
                .new_context(&eng.backend, ctx_params)
                .map_err(|e| LlamaCppError::Inference(format!("failed to create context: {e}")))?;

            // Encode the prompt into the context.
            let mut batch = LlamaBatch::new(eng.context_length as usize, 1);
            let last_index = tokens.len() as i32 - 1;
            for (i, token) in (0_i32..).zip(tokens.into_iter()) {
                let is_last = i == last_index;
                batch.add(token, i, &[0], is_last).map_err(|e| {
                    LlamaCppError::Inference(format!("failed to add token to batch: {e}"))
                })?;
            }

            ctx.decode(&mut batch)
                .map_err(|e| LlamaCppError::Inference(format!("prompt decode failed: {e}")))?;

            // Generation loop.
            let max_gen_tokens = eng.context_length.saturating_sub(prompt_token_count);
            let mut n_cur = batch.n_tokens();
            let mut generated_text = String::new();
            let mut n_generated: u32 = 0;
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(0.8),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::dist(1234),
            ]);

            let start = std::time::Instant::now();

            let mut finish_reason = "length".to_string();

            while n_generated < max_gen_tokens {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                // Check for end of generation.
                if eng.model.is_eog_token(token) {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Decode token to text.
                if let Ok(piece) = eng.model.token_to_piece(token, &mut decoder, true, None) {
                    generated_text.push_str(&piece);
                }

                n_generated += 1;

                // Prepare next batch.
                batch.clear();
                batch.add(token, n_cur, &[0], true).map_err(|e| {
                    LlamaCppError::Inference(format!("failed to add generated token: {e}"))
                })?;
                n_cur += 1;

                ctx.decode(&mut batch).map_err(|e| {
                    LlamaCppError::Inference(format!("decode failed during generation: {e}"))
                })?;
            }

            let elapsed = start.elapsed();
            let total_time_sec = elapsed.as_secs_f32();

            let content = if generated_text.is_empty() {
                None
            } else {
                Some(generated_text)
            };

            Ok(InferenceResult {
                content,
                finish_reason,
                model: eng.model_id.clone(),
                usage: InferenceUsage {
                    prompt_tokens: prompt_token_count,
                    completion_tokens: n_generated,
                    total_tokens: prompt_token_count + n_generated,
                    total_time_sec,
                },
            })
        })
        .await
        .map_err(|e| LlamaCppError::Inference(format!("spawn_blocking panicked: {e}")))?
    }

    /// Returns a `'static` stream by spawning a blocking task that generates
    /// tokens one at a time and sends them through a channel.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn infer_stream_engine(&self, messages: Vec<ChatMessageInput>) -> InferenceChunkStream {
        use tokio_stream::wrappers::ReceiverStream;

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let engine = std::sync::Arc::clone(&self.engine);

        tokio::task::spawn_blocking(move || {
            use llama_cpp_2::context::params::LlamaContextParams;
            use llama_cpp_2::llama_batch::LlamaBatch;
            use llama_cpp_2::model::AddBos;
            use llama_cpp_2::sampling::LlamaSampler;
            use std::num::NonZeroU32;

            let send_err = |e: LlamaCppError| {
                let _ = tx.blocking_send(Err(e));
            };

            // Engine is guaranteed initialized by the caller.
            let eng = engine.get().expect("engine should be initialized");

            let prompt = match format_prompt(&eng.model, &messages) {
                Ok(p) => p,
                Err(_) => format_prompt_chatml(&messages),
            };

            let tokens = match eng.model.str_to_token(&prompt, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    send_err(LlamaCppError::Inference(format!(
                        "tokenization failed: {e}"
                    )));
                    return;
                }
            };

            let prompt_token_count = tokens.len() as u32;

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(eng.context_length))
                .with_n_batch(eng.context_length);

            let mut ctx = match eng.model.new_context(&eng.backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    send_err(LlamaCppError::Inference(format!(
                        "failed to create context: {e}"
                    )));
                    return;
                }
            };

            let mut batch = LlamaBatch::new(eng.context_length as usize, 1);
            let last_index = tokens.len() as i32 - 1;
            for (i, token) in (0_i32..).zip(tokens.into_iter()) {
                let is_last = i == last_index;
                if let Err(e) = batch.add(token, i, &[0], is_last) {
                    send_err(LlamaCppError::Inference(format!(
                        "failed to add token to batch: {e}"
                    )));
                    return;
                }
            }

            if let Err(e) = ctx.decode(&mut batch) {
                send_err(LlamaCppError::Inference(format!(
                    "prompt decode failed: {e}"
                )));
                return;
            }

            let max_gen_tokens = eng.context_length.saturating_sub(prompt_token_count);
            let mut n_cur = batch.n_tokens();
            let mut n_generated: u32 = 0;
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(0.8),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::dist(1234),
            ]);

            while n_generated < max_gen_tokens {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                if eng.model.is_eog_token(token) {
                    // Send final chunk with finish_reason.
                    let _ = tx.blocking_send(Ok(InferenceChunk {
                        delta: None,
                        finish_reason: Some("stop".to_string()),
                    }));
                    return;
                }

                if let Ok(piece) = eng.model.token_to_piece(token, &mut decoder, true, None)
                    && tx
                        .blocking_send(Ok(InferenceChunk {
                            delta: Some(piece),
                            finish_reason: None,
                        }))
                        .is_err()
                {
                    return; // Receiver dropped.
                }

                n_generated += 1;

                batch.clear();
                if let Err(e) = batch.add(token, n_cur, &[0], true) {
                    send_err(LlamaCppError::Inference(format!(
                        "failed to add generated token: {e}"
                    )));
                    return;
                }
                n_cur += 1;

                if let Err(e) = ctx.decode(&mut batch) {
                    send_err(LlamaCppError::Inference(format!(
                        "decode failed during generation: {e}"
                    )));
                    return;
                }
            }

            // Max tokens reached.
            let _ = tx.blocking_send(Ok(InferenceChunk {
                delta: None,
                finish_reason: Some("length".to_string()),
            }));
        });

        Box::pin(ReceiverStream::new(rx))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LlamaCppOptions;

    #[tokio::test]
    async fn from_options_with_defaults() {
        let opts = LlamaCppOptions::default();
        let provider = LlamaCppProvider::from_options(opts)
            .await
            .expect("should succeed");
        assert!(provider.model_path().is_none());
    }

    #[tokio::test]
    async fn from_options_rejects_empty_model_path() {
        let opts = LlamaCppOptions {
            model_path: Some(String::new()),
            ..LlamaCppOptions::default()
        };
        let result = LlamaCppProvider::from_options(opts).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn from_options_rejects_empty_device() {
        let opts = LlamaCppOptions {
            device: Some(String::new()),
            ..LlamaCppOptions::default()
        };
        let result = LlamaCppProvider::from_options(opts).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn from_options_accepts_valid_device() {
        let opts = LlamaCppOptions {
            device: Some("cuda:0".into()),
            ..LlamaCppOptions::default()
        };
        let provider = LlamaCppProvider::from_options(opts)
            .await
            .expect("should succeed");
        assert!(provider.model_path().is_none());
    }

    #[test]
    fn engine_not_available_display() {
        let err = LlamaCppError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[tokio::test]
    async fn engine_available_reflects_feature() {
        let provider = LlamaCppProvider::from_options(LlamaCppOptions::default())
            .await
            .unwrap();
        if cfg!(feature = "engine") {
            assert!(provider.engine_available());
        } else {
            assert!(!provider.engine_available());
        }
    }

    #[tokio::test]
    async fn infer_without_engine_returns_error() {
        #[cfg(not(feature = "engine"))]
        {
            let opts = LlamaCppOptions::default();
            let provider = LlamaCppProvider::from_options(opts).await.unwrap();
            let result = provider.infer(vec![]).await;
            assert!(
                matches!(result, Err(LlamaCppError::EngineNotAvailable)),
                "expected EngineNotAvailable, got {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn infer_stream_without_engine_returns_error() {
        #[cfg(not(feature = "engine"))]
        {
            let opts = LlamaCppOptions::default();
            let provider = LlamaCppProvider::from_options(opts).await.unwrap();
            let result = provider.infer_stream(vec![]).await;
            assert!(
                matches!(result, Err(LlamaCppError::EngineNotAvailable)),
                "expected EngineNotAvailable"
            );
        }
    }

    #[test]
    fn chat_message_input_constructor() {
        let msg = ChatMessageInput::new(ChatRole::User, "hello");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.text, "hello");
    }

    #[test]
    fn chat_roles_are_distinct() {
        assert_ne!(ChatRole::System, ChatRole::User);
        assert_ne!(ChatRole::User, ChatRole::Assistant);
        assert_ne!(ChatRole::Assistant, ChatRole::Tool);
    }

    /// Verify the ChatML fallback prompt formatter produces the expected format.
    #[cfg(feature = "engine")]
    #[test]
    fn format_prompt_chatml_produces_expected_output() {
        let messages = vec![
            ChatMessageInput::new(ChatRole::System, "You are helpful."),
            ChatMessageInput::new(ChatRole::User, "Hello!"),
        ];
        let prompt = format_prompt_chatml(&messages);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
