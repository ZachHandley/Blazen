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
    /// Mounting (or activating) a `LoRA` adapter failed.
    AdapterFailed(String),
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
            Self::AdapterFailed(msg) => write!(f, "llama.cpp adapter mount failed: {msg}"),
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

/// One `LoRA` adapter currently mounted on a [`LlamaCppProvider`].
///
/// Tracked in `LlamaCppProvider::loaded_adapters` so subsequent
/// [`LlamaCppProvider::load_adapter`] / [`LlamaCppProvider::unload_adapter`]
/// calls can re-apply the correct set on each freshly-created inference
/// context.
#[derive(Debug, Clone)]
pub struct MountedAdapter {
    pub adapter_id: String,
    pub source_path: std::path::PathBuf,
    pub scale: f32,
    /// Handle to the loaded adapter on the engine side. Only present when the
    /// `engine` feature is active and the adapter has actually been
    /// initialised on the loaded model. Wrapped in
    /// `Arc<AdapterHandleCell>` so it can be cheaply cloned out of the
    /// `RwLock` and shared between the provider's adapter map and the inner
    /// activation helper.
    #[cfg(feature = "engine")]
    pub(crate) handle: std::sync::Arc<AdapterHandleCell>,
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
    /// Lazily loaded llama.cpp engine handle, wrapped in
    /// `Arc<RwLock<Option<Arc<...>>>>` so we can both (a) auto-load on first
    /// use and (b) explicitly unload later to free VRAM / memory. The inner
    /// `Arc<Engine>` is cloned out of the lock for each inference call so
    /// that the spawned blocking task can own the handle without holding
    /// the read guard across a `spawn_blocking` boundary.
    ///
    /// [`Engine`] is **not** `Clone` (it wraps the `llama-cpp-2` backend
    /// and model handles, which are not cloneable), so wrapping it in an
    /// inner `Arc` gives cheap clones out of the lock.
    #[cfg(feature = "engine")]
    engine: std::sync::Arc<tokio::sync::RwLock<Option<std::sync::Arc<Engine>>>>,
    /// `LoRA` adapters currently mounted on this provider, keyed by
    /// `adapter_id`. The map is shared between the public adapter API
    /// and the inference helpers so each freshly-built inference context
    /// can re-activate the current set.
    ///
    /// Multiple adapters can be active concurrently: every entry in the
    /// map is mounted in a single batched call to the raw
    /// `llama_set_adapters_lora` FFI on each freshly-created inference
    /// context (the `llama-cpp-2` safe wrapper only exposes a
    /// single-adapter "replace-all-with-one" form, which is bypassed
    /// here). [`LlamaCppProvider::list_adapters`] /
    /// [`LlamaCppProvider::unload_adapter`] target individual handles
    /// by id.
    #[cfg(feature = "engine")]
    loaded_adapters:
        std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, MountedAdapter>>>,
}

impl std::fmt::Debug for LlamaCppProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppProvider")
            .field("model_path", &self.model_path)
            .finish_non_exhaustive()
    }
}

/// Interior-mutability cell for a `LlamaLoraAdapter`. The upstream wrapper
/// holds a raw `NonNull` pointer with no Send/Sync impls; we move it into
/// inference threads via `spawn_blocking`, which requires Send. The
/// underlying `llama_adapter_lora` C handle is safe to share across threads
/// as long as `llama_set_adapters_lora` / `llama_adapter_lora_free` are not
/// called concurrently on it — Blazen serialises both via the provider's
/// `RwLock<HashMap<...>>`, so the unsafe impl is sound.
#[cfg(feature = "engine")]
pub(crate) struct AdapterHandleCell {
    pub(crate) adapter: std::sync::Mutex<llama_cpp_2::model::LlamaLoraAdapter>,
}

#[cfg(feature = "engine")]
#[allow(unsafe_code)]
unsafe impl Send for AdapterHandleCell {}
#[cfg(feature = "engine")]
#[allow(unsafe_code)]
unsafe impl Sync for AdapterHandleCell {}

#[cfg(feature = "engine")]
impl std::fmt::Debug for AdapterHandleCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterHandleCell").finish_non_exhaustive()
    }
}

/// Send shim for moving a freshly-loaded `LlamaLoraAdapter` back across a
/// `spawn_blocking` boundary. The underlying C handle is safe to transfer
/// because the only ops we perform on it (`llama_set_adapters_lora`,
/// `llama_adapter_lora_free` at Drop) are serialised through the
/// provider's `RwLock<HashMap<...>>`.
#[cfg(feature = "engine")]
struct SendAdapter(llama_cpp_2::model::LlamaLoraAdapter);

#[cfg(feature = "engine")]
#[allow(unsafe_code)]
unsafe impl Send for SendAdapter {}

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
        let model_path = opts.base.model_id.clone();
        if opts.base.tokenizer_repo.is_some() {
            // llama.cpp reads the tokenizer straight out of the GGUF
            // file, so a separate-tokenizer-repo override is meaningless
            // here. Accept the field for cross-backend parity but warn so
            // users aren't silently confused.
            tracing::warn!(
                tokenizer_repo = ?opts.base.tokenizer_repo,
                "LlamaCppOptions.base.tokenizer_repo is ignored — llama.cpp \
                 embeds the tokenizer in the GGUF file"
            );
        }
        Ok(Self {
            model_path,
            #[cfg(feature = "engine")]
            engine: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
            #[cfg(feature = "engine")]
            loaded_adapters: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            options: opts,
        })
    }

    /// Validate the options without loading the engine.
    fn validate_options(opts: &LlamaCppOptions) -> Result<(), LlamaCppError> {
        if let Some(ref model_id) = opts.base.model_id
            && model_id.is_empty()
        {
            return Err(LlamaCppError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(ref device) = opts.base.device
            && device.is_empty()
        {
            return Err(LlamaCppError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        Ok(())
    }

    /// The model path (or HF `repo/file.gguf` reference) passed at
    /// construction time. Kept under the legacy name `model_path` for
    /// caller compatibility; mirrors
    /// [`blazen_local_llm::LocalLlmOptions::model_id`] internally.
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

    /// The configured device string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`), or
    /// `None` if no device was specified.
    #[must_use]
    pub fn device_str(&self) -> Option<&str> {
        self.options.base.device.as_deref()
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
    // `async` is required to match the non-engine stub and keep the public
    // signature stable across feature configurations. The body hands off
    // work to a background task and does not need to await anything here;
    // engine loading happens inside the spawned task so load errors surface
    // as the first item on the returned stream.
    #[allow(clippy::unused_async)]
    pub async fn infer_stream(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceChunkStream, LlamaCppError> {
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

    // -----------------------------------------------------------------------
    // Explicit load / unload (always in the public API)
    //
    // These mirror the `infer` / `infer_stream` cfg dual-stub pattern so
    // that the public surface is identical with and without the `engine`
    // feature, and so the `blazen_llm::LocalModel` trait bridge in
    // `blazen-llm/src/backends/llamacpp.rs` can call them unconditionally.
    // -----------------------------------------------------------------------

    /// Load the model explicitly. Idempotent -- if already loaded, this
    /// is a no-op that returns `Ok(())`.
    ///
    /// Inference methods ([`Self::infer`], [`Self::infer_stream`]) will
    /// still auto-load on first call if [`Self::load`] was never invoked,
    /// so explicit loading is only needed when the caller wants to
    /// pay the initialization cost up-front (e.g. to avoid latency
    /// spikes during a time-sensitive workflow step).
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::ModelLoad`] if the underlying model fails
    /// to load (e.g. missing GGUF file, corrupt weights, or an
    /// incompatible accelerator backend).
    #[cfg(feature = "engine")]
    pub async fn load(&self) -> Result<(), LlamaCppError> {
        // Reuse the existing loader logic.
        let _ = self.get_or_load_engine().await?;
        Ok(())
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`LlamaCppError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn load(&self) -> Result<(), LlamaCppError> {
        Err(LlamaCppError::EngineNotAvailable)
    }

    /// Drop the loaded model and free its VRAM / memory. Idempotent --
    /// if the model is already unloaded, this is a no-op that returns
    /// `Ok(())`.
    ///
    /// Note: if a blocking inference task is still holding an
    /// `Arc<Engine>` clone, the underlying `Engine` will only be dropped
    /// (and its VRAM freed) once that task finishes. `unload` always
    /// releases the provider's own reference immediately.
    ///
    /// # Errors
    ///
    /// This method currently never returns an error; the `Result` return
    /// type is preserved to match [`crate::LlamaCppError`] conventions
    /// and the [`blazen_llm::traits::LocalModel`] trait contract.
    #[cfg(feature = "engine")]
    pub async fn unload(&self) -> Result<(), LlamaCppError> {
        let mut guard = self.engine.write().await;
        // Drop the Arc<Engine>. When no other clones remain (e.g. from an
        // in-flight blocking inference task), VRAM is freed by the Drop
        // impl on the underlying llama.cpp model + backend.
        *guard = None;
        // Why: adapter handles are tied to the LlamaModel that just got
        // dropped — keeping them would dangle the underlying C pointers.
        let mut adapters_guard = self.loaded_adapters.write().await;
        adapters_guard.clear();
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
    pub async fn unload(&self) -> Result<(), LlamaCppError> {
        Ok(())
    }

    /// Whether the model is currently loaded in memory / VRAM.
    #[cfg(feature = "engine")]
    pub async fn is_loaded(&self) -> bool {
        self.engine.read().await.is_some()
    }

    /// Stub: without the engine feature there is never a loaded model,
    /// so this always returns `false`.
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn is_loaded(&self) -> bool {
        false
    }

    /// Mount a `LoRA` adapter on the loaded base model via llama.cpp's
    /// hot-attach API (`llama_adapter_lora_init` +
    /// `llama_set_adapters_lora`).
    ///
    /// The adapter is loaded into the live model handle and re-applied to
    /// every inference context the provider builds going forward — no
    /// engine rebuild is performed. The base model must already be loaded
    /// (or be loadable lazily); this method triggers a load if needed.
    ///
    /// Multiple adapters can be active simultaneously: this provider
    /// calls the underlying `llama_set_adapters_lora` directly via raw
    /// FFI to batch every loaded adapter into a single context-mount
    /// call (the `llama-cpp-2` safe wrapper only exposes the
    /// "replace-all-with-one" form, which is why we bypass it here).
    ///
    /// `adapter_path` must be a llama.cpp-compatible adapter file (GGUF
    /// `LoRA` format). PEFT-canonical `safetensors` adapters must be
    /// converted to GGUF first via llama.cpp's
    /// `convert_lora_to_gguf.py`; this provider does not perform that
    /// conversion.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::AdapterFailed`] when the file cannot be
    /// loaded as a `LoRA` adapter by llama.cpp, or when a duplicate
    /// `adapter_id` is requested.
    #[cfg(feature = "engine")]
    pub async fn load_adapter(
        &self,
        adapter_path: &std::path::Path,
        adapter_id: impl Into<String>,
        scale: f32,
    ) -> Result<MountedAdapter, LlamaCppError> {
        let adapter_id = adapter_id.into();

        {
            let guard = self.loaded_adapters.read().await;
            if guard.contains_key(&adapter_id) {
                return Err(LlamaCppError::AdapterFailed(format!(
                    "adapter '{adapter_id}' is already mounted",
                )));
            }
        }

        let engine = self.get_or_load_engine().await?;

        if !adapter_path.exists() {
            return Err(LlamaCppError::AdapterFailed(format!(
                "adapter path does not exist: {}",
                adapter_path.display()
            )));
        }

        let adapter_path_owned = adapter_path.to_path_buf();
        let adapter = tokio::task::spawn_blocking(move || {
            engine
                .model
                .lora_adapter_init(&adapter_path_owned)
                .map(SendAdapter)
        })
        .await
        .map_err(|e| LlamaCppError::AdapterFailed(format!("spawn_blocking panicked: {e}")))?
        .map_err(|e| {
            LlamaCppError::AdapterFailed(format!(
                "lora_adapter_init failed for '{}': {e}",
                adapter_path.display()
            ))
        })?
        .0;

        let mounted = MountedAdapter {
            adapter_id: adapter_id.clone(),
            source_path: adapter_path.to_path_buf(),
            scale,
            handle: std::sync::Arc::new(AdapterHandleCell {
                adapter: std::sync::Mutex::new(adapter),
            }),
        };

        {
            let mut guard = self.loaded_adapters.write().await;
            guard.insert(adapter_id, mounted.clone());
        }

        Ok(mounted)
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`LlamaCppError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn load_adapter(
        &self,
        _adapter_path: &std::path::Path,
        _adapter_id: impl Into<String>,
        _scale: f32,
    ) -> Result<MountedAdapter, LlamaCppError> {
        Err(LlamaCppError::EngineNotAvailable)
    }

    /// Remove a previously-mounted `LoRA` adapter. The adapter is dropped
    /// from the provider's registry; subsequent contexts will no longer
    /// have it applied.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::AdapterFailed`] if no adapter with the
    /// given id is currently mounted.
    #[cfg(feature = "engine")]
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), LlamaCppError> {
        let mut guard = self.loaded_adapters.write().await;
        if guard.remove(adapter_id).is_none() {
            return Err(LlamaCppError::AdapterFailed(format!(
                "adapter '{adapter_id}' is not mounted",
            )));
        }
        Ok(())
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`LlamaCppError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn unload_adapter(&self, _adapter_id: &str) -> Result<(), LlamaCppError> {
        Err(LlamaCppError::EngineNotAvailable)
    }

    /// Snapshot the list of `LoRA` adapters currently mounted on this
    /// provider.
    #[cfg(feature = "engine")]
    pub async fn list_adapters(&self) -> Vec<MountedAdapter> {
        self.loaded_adapters
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Stub: engine not available; no adapters can be mounted.
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn list_adapters(&self) -> Vec<MountedAdapter> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Engine construction (only with `engine` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
async fn resolve_model_path(opts: &LlamaCppOptions) -> Result<std::path::PathBuf, LlamaCppError> {
    let model_path = opts.base.model_id.as_deref().ok_or_else(|| {
        LlamaCppError::InvalidOptions("model_id is required for llama.cpp inference".into())
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
                "model_id \"{model_path}\" looks like a HuggingFace repo ID \
                 but no filename was specified. Use format: \
                 \"org/repo/model-file.gguf\""
            )));
        };

        tracing::info!(
            repo_id = repo_id,
            filename = filename,
            "downloading model from HuggingFace Hub"
        );

        let cache = if let Some(ref cache_dir) = opts.base.cache_dir {
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
    let context_length = opts.base.context_length;
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

/// Mount every `LlamaCppOptions::initial_adapters` entry on the
/// freshly-loaded engine into the shared adapter map. Free-function form
/// so the streaming task (which only has clones of `options` and the
/// `Arc<RwLock<...>>` maps) can call it without `&self`.
#[cfg(feature = "engine")]
async fn bootstrap_initial_adapters_for(
    options: &LlamaCppOptions,
    engine: &std::sync::Arc<Engine>,
    loaded_adapters: &std::sync::Arc<
        tokio::sync::RwLock<std::collections::HashMap<String, MountedAdapter>>,
    >,
) -> Result<(), LlamaCppError> {
    if options.base.initial_adapters.is_empty() {
        return Ok(());
    }
    for path in &options.base.initial_adapters {
        if !path.exists() {
            return Err(LlamaCppError::AdapterFailed(format!(
                "initial adapter path does not exist: {}",
                path.display()
            )));
        }
        let adapter_id = path.display().to_string();
        {
            let guard = loaded_adapters.read().await;
            if guard.contains_key(&adapter_id) {
                continue;
            }
        }
        let engine_for_blocking = std::sync::Arc::clone(engine);
        let path_for_blocking = path.clone();
        let adapter = tokio::task::spawn_blocking(move || {
            engine_for_blocking
                .model
                .lora_adapter_init(&path_for_blocking)
                .map(SendAdapter)
        })
        .await
        .map_err(|e| LlamaCppError::AdapterFailed(format!("spawn_blocking panicked: {e}")))?
        .map_err(|e| {
            LlamaCppError::AdapterFailed(format!(
                "lora_adapter_init failed for initial adapter '{}': {e}",
                path.display()
            ))
        })?
        .0;
        let mounted = MountedAdapter {
            adapter_id: adapter_id.clone(),
            source_path: path.clone(),
            scale: 1.0,
            handle: std::sync::Arc::new(AdapterHandleCell {
                adapter: std::sync::Mutex::new(adapter),
            }),
        };
        let mut guard = loaded_adapters.write().await;
        guard.insert(adapter_id, mounted);
    }
    Ok(())
}

/// Layout mirror of `llama_cpp_2::context::LlamaContext<'a>` used to
/// extract the underlying `*mut llama_context` C pointer. The upstream
/// type holds the pointer in a `pub(crate)` field and exposes no public
/// accessor, so the only way to call the raw multi-adapter FFI
/// (`llama_set_adapters_lora` with `n_adapters > 1`) is to mirror the
/// struct definition byte-for-byte here and transmute a reference.
///
/// Field declarations must stay in lockstep with
/// `llama-cpp-2`'s `LlamaContext`. The `lora_context_layout_matches`
/// const assert below catches size/alignment drift at compile time.
#[cfg(feature = "engine")]
#[allow(dead_code)]
struct LlamaContextLayoutMirror<'a> {
    context: std::ptr::NonNull<llama_cpp_sys_2::llama_context>,
    model: &'a llama_cpp_2::model::LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
}

#[cfg(feature = "engine")]
const _: () = {
    // Why: catch upstream struct-layout changes that would silently
    // misinterpret the transmuted pointer as something other than the
    // raw C `llama_context*`.
    assert!(
        std::mem::size_of::<LlamaContextLayoutMirror<'static>>()
            == std::mem::size_of::<llama_cpp_2::context::LlamaContext<'static>>(),
        "LlamaContext layout mirror size mismatch — llama-cpp-2 changed its struct"
    );
    assert!(
        std::mem::align_of::<LlamaContextLayoutMirror<'static>>()
            == std::mem::align_of::<llama_cpp_2::context::LlamaContext<'static>>(),
        "LlamaContext layout mirror alignment mismatch — llama-cpp-2 changed its struct"
    );
};

/// Extract the raw `*mut llama_context` from a safe-wrapper context.
///
/// # Safety
///
/// The caller must guarantee that `LlamaContextLayoutMirror` mirrors the
/// real `LlamaContext` field declarations exactly. The const-asserts
/// above check size and alignment, which together with identical field
/// types in the same order is sufficient for rustc to assign identical
/// layouts to both types in the same compilation.
#[cfg(feature = "engine")]
#[allow(unsafe_code)]
unsafe fn raw_context_ptr(
    ctx: &llama_cpp_2::context::LlamaContext<'_>,
) -> *mut llama_cpp_sys_2::llama_context {
    // SAFETY: LlamaContextLayoutMirror has identical field declarations,
    // size, and alignment to LlamaContext (validated by const-asserts).
    let mirror: &LlamaContextLayoutMirror<'_> = unsafe {
        &*std::ptr::from_ref::<llama_cpp_2::context::LlamaContext<'_>>(ctx)
            .cast::<LlamaContextLayoutMirror<'_>>()
    };
    mirror.context.as_ptr()
}

/// Extract the raw `*mut llama_adapter_lora` from a safe-wrapper adapter.
///
/// `LlamaLoraAdapter` is `#[repr(transparent)]` over
/// `NonNull<llama_adapter_lora>`, so this transmute is sound.
///
/// # Safety
///
/// Caller must keep `adapter` alive for as long as the returned pointer
/// is in use.
#[cfg(feature = "engine")]
#[allow(unsafe_code)]
unsafe fn raw_adapter_ptr(
    adapter: &llama_cpp_2::model::LlamaLoraAdapter,
) -> *mut llama_cpp_sys_2::llama_adapter_lora {
    // SAFETY: LlamaLoraAdapter is repr(transparent) over
    // NonNull<llama_adapter_lora>.
    let nn: &std::ptr::NonNull<llama_cpp_sys_2::llama_adapter_lora> = unsafe {
        &*std::ptr::from_ref::<llama_cpp_2::model::LlamaLoraAdapter>(adapter)
            .cast::<std::ptr::NonNull<llama_cpp_sys_2::llama_adapter_lora>>()
    };
    nn.as_ptr()
}

/// Apply every mounted `LoRA` adapter to a freshly-built inference
/// context in a single batched call.
///
/// Bypasses `llama_cpp_2::context::LlamaContext::lora_adapter_set`,
/// which hard-codes `n_adapters = 1` and therefore replaces all
/// previously-set adapters every call — making it impossible to mount
/// more than one adapter at a time through the safe API. We instead
/// build one `(adapters, scales)` array and invoke
/// `llama_set_adapters_lora` directly so all adapters take effect
/// together.
#[cfg(feature = "engine")]
#[allow(unsafe_code)]
fn apply_adapters_to_context(
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    adapters: &[MountedAdapter],
) -> Result<(), LlamaCppError> {
    if adapters.is_empty() {
        return Ok(());
    }

    let mut guards = Vec::with_capacity(adapters.len());
    for mounted in adapters {
        let guard = mounted.handle.adapter.lock().map_err(|e| {
            LlamaCppError::AdapterFailed(format!(
                "adapter '{}' mutex poisoned: {e}",
                mounted.adapter_id
            ))
        })?;
        guards.push((mounted, guard));
    }

    let mut adapter_ptrs: Vec<*mut llama_cpp_sys_2::llama_adapter_lora> = guards
        .iter()
        // SAFETY: each guard keeps the LlamaLoraAdapter alive for the
        // duration of the FFI call below.
        .map(|(_, g)| unsafe { raw_adapter_ptr(g) })
        .collect();
    let mut scales: Vec<f32> = guards.iter().map(|(m, _)| m.scale).collect();

    // SAFETY: ctx is held by &mut for the duration of the call; adapter
    // pointers and the scales buffer outlive the call; n_adapters
    // matches the array lengths.
    let rc = unsafe {
        llama_cpp_sys_2::llama_set_adapters_lora(
            raw_context_ptr(ctx),
            adapter_ptrs.as_mut_ptr(),
            adapter_ptrs.len(),
            scales.as_mut_ptr(),
        )
    };

    drop(guards);

    if rc != 0 {
        return Err(LlamaCppError::AdapterFailed(format!(
            "llama_set_adapters_lora returned {rc} when activating {} adapter(s)",
            adapters.len()
        )));
    }
    Ok(())
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
    /// Return the loaded engine, loading it on first use.
    ///
    /// Uses a double-checked `RwLock` pattern: the fast path takes a read
    /// lock and clones out the existing `Arc<Engine>`; if the engine is
    /// not yet loaded, it drops the read lock, acquires a write lock,
    /// re-checks (another task may have loaded it concurrently), and
    /// finally builds the engine.
    ///
    /// Returns an owned `Arc<Engine>` so the caller can hold it across
    /// await points and pass it to spawned blocking tasks without holding
    /// the read lock.
    async fn get_or_load_engine(&self) -> Result<std::sync::Arc<Engine>, LlamaCppError> {
        // Fast path: acquire read lock, check if already loaded, clone Arc.
        {
            let guard = self.engine.read().await;
            if let Some(engine) = guard.as_ref() {
                return Ok(std::sync::Arc::clone(engine));
            }
        }
        // Slow path: acquire write lock, double-check, build, clone Arc.
        let mut guard = self.engine.write().await;
        let just_loaded = guard.is_none();
        if just_loaded {
            let engine = build_engine(&self.options).await?;
            *guard = Some(std::sync::Arc::new(engine));
        }
        // SAFETY: we just set Some above (or found Some from a concurrent loader).
        let engine = std::sync::Arc::clone(guard.as_ref().expect("engine loaded above"));
        drop(guard);
        if just_loaded {
            self.bootstrap_initial_adapters(&engine).await?;
        }
        Ok(engine)
    }

    /// Mount every `LlamaCppOptions::initial_adapters` entry on the
    /// freshly-loaded engine. Called from the slow path of
    /// [`Self::get_or_load_engine`].
    async fn bootstrap_initial_adapters(
        &self,
        engine: &std::sync::Arc<Engine>,
    ) -> Result<(), LlamaCppError> {
        bootstrap_initial_adapters_for(&self.options, engine, &self.loaded_adapters).await
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
        // Load the engine (or grab the already-loaded handle) and move an
        // owned `Arc<Engine>` into the blocking task so it stays alive for
        // the duration of inference, independent of any concurrent `unload`.
        let engine = self.get_or_load_engine().await?;
        // Snapshot the active adapters before crossing the spawn_blocking
        // boundary so we never hold the RwLock across an await/move.
        let adapters: Vec<MountedAdapter> = self
            .loaded_adapters
            .read()
            .await
            .values()
            .cloned()
            .collect();

        tokio::task::spawn_blocking(move || {
            use llama_cpp_2::context::params::LlamaContextParams;
            use llama_cpp_2::llama_batch::LlamaBatch;
            use llama_cpp_2::model::AddBos;
            use llama_cpp_2::sampling::LlamaSampler;
            use std::num::NonZeroU32;

            let eng: &Engine = &engine;

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

            apply_adapters_to_context(&mut ctx, &adapters)?;

            // Encode the prompt into the context.
            let mut batch = LlamaBatch::new(eng.context_length as usize, 1);
            let last_index = tokens.len() as i32 - 1;
            for (i, token) in (0_i32..).zip(tokens) {
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

    /// Returns a `'static` stream by spawning an async task that owns a
    /// clone of the engine `Arc<RwLock<...>>` and the options, initialises
    /// the engine lazily, and then delegates the actual token-generation
    /// work to a `spawn_blocking` task which forwards mapped chunks
    /// through a channel.
    ///
    /// Engine errors (including failure to load or start the stream) are
    /// delivered as error items on the returned stream. The blocking task
    /// holds an owned `Arc<Engine>` for the entire duration of streaming,
    /// so an explicit `unload` from another task cannot yank the engine
    /// out from under an in-flight stream -- the `Engine` stays alive
    /// until this task drops its `Arc`.
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

        // Clone the Arc + options so the spawned task can outlive `&self`.
        let engine_lock = std::sync::Arc::clone(&self.engine);
        let adapters_lock = std::sync::Arc::clone(&self.loaded_adapters);
        let options = self.options.clone();

        tokio::spawn(async move {
            // Lazily initialise the engine inside the spawned task. We
            // inline the double-checked `RwLock` load pattern here instead
            // of calling `get_or_load_engine` so that we can preserve the
            // existing error-delivery contract (load errors appear as the
            // first stream item, not as a synchronous return error).
            let (engine, just_loaded): (std::sync::Arc<Engine>, bool) = {
                // Fast path: acquire read lock, check if already loaded.
                {
                    let guard = engine_lock.read().await;
                    if let Some(engine) = guard.as_ref() {
                        (std::sync::Arc::clone(engine), false)
                    } else {
                        drop(guard);
                        // Slow path: acquire write lock, double-check, build.
                        let mut guard = engine_lock.write().await;
                        let did_load = guard.is_none();
                        if did_load {
                            match build_engine(&options).await {
                                Ok(eng) => {
                                    *guard = Some(std::sync::Arc::new(eng));
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                    return;
                                }
                            }
                        }
                        // SAFETY: we just set Some above (or found Some from a concurrent loader).
                        (
                            std::sync::Arc::clone(guard.as_ref().expect("engine loaded above")),
                            did_load,
                        )
                    }
                }
            };

            if just_loaded
                && let Err(e) =
                    bootstrap_initial_adapters_for(&options, &engine, &adapters_lock).await
            {
                let _ = tx.send(Err(e)).await;
                return;
            }

            // Snapshot active adapters before crossing into the blocking
            // thread so we never hold the RwLock across an FFI boundary.
            let adapters: Vec<MountedAdapter> =
                adapters_lock.read().await.values().cloned().collect();

            // Hand the actual (blocking) token-generation loop off to a
            // dedicated blocking thread. The `Arc<Engine>` we hold keeps
            // the model alive for the duration of the task.
            let _ = tokio::task::spawn_blocking(move || {
                use llama_cpp_2::context::params::LlamaContextParams;
                use llama_cpp_2::llama_batch::LlamaBatch;
                use llama_cpp_2::model::AddBos;
                use llama_cpp_2::sampling::LlamaSampler;
                use std::num::NonZeroU32;

                let send_err = |e: LlamaCppError| {
                    let _ = tx.blocking_send(Err(e));
                };

                let eng: &Engine = &engine;

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

                if let Err(e) = apply_adapters_to_context(&mut ctx, &adapters) {
                    send_err(e);
                    return;
                }

                let mut batch = LlamaBatch::new(eng.context_length as usize, 1);
                let last_index = tokens.len() as i32 - 1;
                for (i, token) in (0_i32..).zip(tokens) {
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
            })
            .await;
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
    async fn from_options_rejects_empty_model_id() {
        let opts = LlamaCppOptions {
            base: blazen_local_llm::LocalLlmOptions {
                model_id: Some(String::new()),
                ..blazen_local_llm::LocalLlmOptions::default()
            },
            ..LlamaCppOptions::default()
        };
        let result = LlamaCppProvider::from_options(opts).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn from_options_rejects_empty_device() {
        let opts = LlamaCppOptions {
            base: blazen_local_llm::LocalLlmOptions {
                device: Some(String::new()),
                ..blazen_local_llm::LocalLlmOptions::default()
            },
            ..LlamaCppOptions::default()
        };
        let result = LlamaCppProvider::from_options(opts).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn from_options_accepts_valid_device() {
        let opts = LlamaCppOptions {
            base: blazen_local_llm::LocalLlmOptions::new().with_device("cuda:0"),
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

    /// Verify the `ChatML` fallback prompt formatter produces the expected format.
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
