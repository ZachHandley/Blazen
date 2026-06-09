//! The [`MistralRsProvider`] type -- local LLM inference via mistral.rs.
//!
//! When the `engine` feature is enabled, constructing a provider will load the
//! model into memory using the mistral.rs runtime. Without the feature, the
//! provider validates options but returns [`MistralRsError::EngineNotAvailable`]
//! from inference methods.

use std::fmt;
use std::path::PathBuf;
use std::pin::Pin;

use futures_util::Stream;

use crate::MistralRsOptions;

/// Type alias for the boxed inference chunk stream returned by streaming methods.
pub type InferenceChunkStream =
    Pin<Box<dyn Stream<Item = Result<InferenceChunk, MistralRsError>> + Send>>;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error type for mistral.rs operations.
#[derive(Debug)]
pub enum MistralRsError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The engine failed to initialise.
    Init(String),
    /// An inference operation failed.
    Inference(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
    /// Mounting (or remounting) a `LoRA` adapter failed at engine rebuild time.
    AdapterFailed(String),
}

impl fmt::Display for MistralRsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "mistral.rs invalid options: {msg}"),
            Self::Init(msg) => write!(f, "mistral.rs init failed: {msg}"),
            Self::Inference(msg) => write!(f, "mistral.rs inference failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "mistral.rs engine not available: compile with the `engine` feature"
            ),
            Self::AdapterFailed(msg) => write!(f, "mistral.rs adapter mount failed: {msg}"),
        }
    }
}

impl std::error::Error for MistralRsError {}

/// One `LoRA` adapter currently mounted on a [`MistralRsProvider`].
///
/// Tracked in `MistralRsProvider::mounted_adapters` so subsequent
/// [`MistralRsProvider::load_adapter`] / [`MistralRsProvider::unload_adapter`]
/// calls can rebuild the engine with the correct set of adapters.
#[derive(Debug, Clone)]
pub struct MountedAdapter {
    pub adapter_id: String,
    pub adapter_dir: std::path::PathBuf,
    pub scale: f32,
}

// ---------------------------------------------------------------------------
// Inference result types (always available)
// ---------------------------------------------------------------------------

/// Result of a single non-streaming inference call.
#[derive(Debug)]
pub struct InferenceResult {
    /// The generated text content, if any.
    pub content: Option<String>,
    /// Reasoning / chain-of-thought content, if the model exposes it.
    pub reasoning_content: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<InferenceToolCall>,
    /// Why the model stopped generating.
    pub finish_reason: String,
    /// The model identifier that produced this result.
    pub model: String,
    /// Token usage statistics.
    pub usage: InferenceUsage,
}

/// A tool call returned by the engine.
#[derive(Debug, Clone)]
pub struct InferenceToolCall {
    /// Provider-assigned call identifier.
    pub id: String,
    /// The function name.
    pub name: String,
    /// The arguments as a JSON string.
    pub arguments: String,
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
    /// Incremental reasoning content.
    pub reasoning_delta: Option<String>,
    /// Tool calls completed in this chunk.
    pub tool_calls: Vec<InferenceToolCall>,
    /// Present in the final chunk when generation stops.
    pub finish_reason: Option<String>,
}

/// Simplified chat role for the mistral.rs bridge layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Source of an image payload for a multimodal chat message.
///
/// This type is intentionally decoupled from the `image` crate so that
/// downstream code can construct vision requests without depending on the
/// `engine` feature. When the `engine` feature is off, the provider accepts
/// these values but returns [`MistralRsError::EngineNotAvailable`] from
/// inference methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceImageSource {
    /// Raw encoded image bytes (PNG, JPEG, WebP, etc.). The format is
    /// auto-detected by the `image` crate at engine time.
    Bytes(Vec<u8>),
    /// Path to a local image file. Read and decoded at engine time.
    Path(PathBuf),
}

/// An image payload attached to a [`ChatMessageInput`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceImage {
    /// Where the image data lives.
    pub source: InferenceImageSource,
}

impl InferenceImage {
    /// Build an image from raw encoded bytes.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            source: InferenceImageSource::Bytes(bytes),
        }
    }

    /// Build an image from a local file path.
    #[must_use]
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            source: InferenceImageSource::Path(path.into()),
        }
    }
}

/// A single chat message, optionally carrying image attachments.
///
/// Passed to [`MistralRsProvider::infer`] and
/// [`MistralRsProvider::infer_stream`]. Text-only messages leave `images`
/// empty; vision messages append one or more [`InferenceImage`] entries.
///
/// Images are only honoured when the provider was constructed with
/// [`MistralRsOptions::vision`] set to `true` -- otherwise the provider will
/// refuse the request with [`MistralRsError::InvalidOptions`].
#[derive(Debug, Clone)]
pub struct ChatMessageInput {
    /// Who produced this message.
    pub role: ChatRole,
    /// The textual content of the message. For image-only inputs, set this
    /// to a short prompt such as `"Describe this image."`; mistral.rs always
    /// expects an accompanying text segment.
    pub text: String,
    /// Zero or more images attached to this message.
    pub images: Vec<InferenceImage>,
}

impl ChatMessageInput {
    /// Build a text-only chat message.
    #[must_use]
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
            images: Vec::new(),
        }
    }

    /// Build a chat message with images attached.
    #[must_use]
    pub fn with_images(
        role: ChatRole,
        text: impl Into<String>,
        images: Vec<InferenceImage>,
    ) -> Self {
        Self {
            role,
            text: text.into(),
            images,
        }
    }

    /// Whether this message has any image attachments.
    #[must_use]
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A local LLM provider backed by [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs).
///
/// Constructed via [`MistralRsProvider::from_options`]. When the `engine`
/// feature is active, the model is loaded lazily on the first inference
/// call. Without the feature, only validation is performed and inference
/// methods return [`MistralRsError::EngineNotAvailable`].
pub struct MistralRsProvider {
    /// The `HuggingFace` model ID or local path that was requested.
    model_id: String,
    /// Full options preserved for deferred engine initialisation.
    options: MistralRsOptions,
    /// Lazily loaded mistral.rs model handle, wrapped in
    /// `Arc<RwLock<Option<Arc<...>>>>` so we can both (a) auto-load on first
    /// use and (b) explicitly unload later to free GPU memory. The inner
    /// `Arc<Model>` is cloned out of the lock for each inference call so
    /// streaming tasks can own the handle without holding the read guard
    /// across await points.
    ///
    /// `mistralrs::Model` is **not** `Clone`, so wrapping it in an inner
    /// `Arc` gives cheap clones out of the lock.
    #[cfg(feature = "engine")]
    engine: std::sync::Arc<tokio::sync::RwLock<Option<std::sync::Arc<mistralrs::Model>>>>,
    /// `LoRA` adapters currently mounted on this provider, ordered by mount time.
    /// Each `load_adapter` call appends here; each `unload_adapter` call removes.
    /// On engine (re)load, the dirs are passed to `LoraModelBuilder`.
    #[cfg(feature = "engine")]
    mounted_adapters: std::sync::Arc<tokio::sync::RwLock<Vec<MountedAdapter>>>,
}

impl std::fmt::Debug for MistralRsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralRsProvider")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl MistralRsProvider {
    /// Create a new provider from the given options.
    ///
    /// This is always synchronous -- the model is loaded lazily on first
    /// inference when the `engine` feature is active. Without the feature,
    /// only validation is performed and inference methods return
    /// [`MistralRsError::EngineNotAvailable`].
    ///
    /// # Errors
    ///
    /// Returns [`MistralRsError::InvalidOptions`] if `model_id` is empty.
    pub fn from_options(opts: MistralRsOptions) -> Result<Self, MistralRsError> {
        let model_id = opts.base.model_id.as_deref().unwrap_or("");
        if model_id.is_empty() {
            return Err(MistralRsError::InvalidOptions(
                "model_id must not be empty".into(),
            ));
        }
        if opts.base.tokenizer_repo.is_some() {
            // mistral.rs's TextModelBuilder fetches the tokenizer from
            // model_id internally; there's no override hook in the
            // upstream API. Accept the field for cross-backend parity
            // but warn so users aren't silently confused.
            tracing::warn!(
                tokenizer_repo = ?opts.base.tokenizer_repo,
                "MistralRsOptions.tokenizer_repo is ignored — mistral.rs has no \
                 separate-tokenizer-repo hook; switch to the candle backend if \
                 you need it"
            );
        }

        #[cfg(feature = "engine")]
        let initial_mounted: Vec<MountedAdapter> = opts
            .base
            .initial_adapters
            .iter()
            .map(|dir| MountedAdapter {
                adapter_id: dir.display().to_string(),
                adapter_dir: dir.clone(),
                scale: 1.0,
            })
            .collect();

        Ok(Self {
            model_id: model_id.to_owned(),
            #[cfg(feature = "engine")]
            engine: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
            #[cfg(feature = "engine")]
            mounted_adapters: std::sync::Arc::new(tokio::sync::RwLock::new(initial_mounted)),
            options: opts,
        })
    }

    /// The model identifier that was passed at construction time.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Access the stored options.
    #[must_use]
    pub fn options(&self) -> &MistralRsOptions {
        &self.options
    }

    /// Whether the engine feature is compiled in and the model is loaded.
    #[must_use]
    pub fn engine_available(&self) -> bool {
        cfg!(feature = "engine")
    }

    // -----------------------------------------------------------------------
    // Inference methods (always available in the public API)
    // -----------------------------------------------------------------------

    /// Run non-streaming inference and return the result.
    ///
    /// Accepts both text-only and multimodal messages. Messages with image
    /// attachments are only honoured when the provider was built with
    /// [`MistralRsOptions::vision`](crate::MistralRsOptions::vision) set to
    /// `true`; otherwise the call fails with
    /// [`MistralRsError::InvalidOptions`].
    ///
    /// When the `engine` feature is not enabled, returns
    /// [`MistralRsError::EngineNotAvailable`].
    ///
    /// # Errors
    ///
    /// Returns [`MistralRsError::Inference`] on engine failure, or
    /// [`MistralRsError::EngineNotAvailable`] if the engine is not compiled in.
    #[cfg(feature = "engine")]
    pub async fn infer(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceResult, MistralRsError> {
        self.infer_engine(messages).await
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`MistralRsError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn infer(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceResult, MistralRsError> {
        let _ = messages;
        Err(MistralRsError::EngineNotAvailable)
    }

    /// Run streaming inference and return a `'static` stream of chunks.
    ///
    /// Accepts both text-only and multimodal messages. See [`Self::infer`]
    /// for the vision-mode contract.
    ///
    /// The returned stream is decoupled from `&self` so that it satisfies
    /// the `'static` bound required by the `Model` trait.
    ///
    /// # Errors
    ///
    /// Returns [`MistralRsError::Inference`] if the stream cannot be started,
    /// or [`MistralRsError::InvalidOptions`] if image attachments are
    /// supplied without enabling vision mode.
    #[cfg(feature = "engine")]
    // `async` is required to match the non-engine stub and keep the public
    // signature stable across feature configurations. The body hands off
    // work to a background task and does not need to await anything here.
    #[allow(clippy::unused_async)]
    pub async fn infer_stream(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceChunkStream, MistralRsError> {
        if !self.options.vision && messages.iter().any(ChatMessageInput::has_images) {
            return Err(MistralRsError::InvalidOptions(
                "image content supplied but vision mode is disabled -- \
                 set MistralRsOptions.vision = true and load a vision-capable model"
                    .into(),
            ));
        }
        Ok(self.infer_stream_engine(messages))
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`MistralRsError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn infer_stream(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceChunkStream, MistralRsError> {
        let _ = messages;
        Err(MistralRsError::EngineNotAvailable)
    }

    // -----------------------------------------------------------------------
    // Explicit load / unload (always in the public API)
    //
    // These mirror the `infer` / `infer_stream` cfg dual-stub pattern so
    // that the public surface is identical with and without the `engine`
    // feature, and so the `blazen_llm::LocalModel` trait bridge in
    // `blazen-llm/src/backends/mistralrs.rs` can call them unconditionally.
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
    /// Returns [`MistralRsError::Init`] if the underlying model fails to
    /// build (e.g. missing weights, unsupported architecture, or an
    /// incompatible accelerator backend).
    #[cfg(feature = "engine")]
    pub async fn load(&self) -> Result<(), MistralRsError> {
        // Reuse the existing loader logic.
        let _ = self.get_or_load_engine().await?;
        Ok(())
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`MistralRsError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn load(&self) -> Result<(), MistralRsError> {
        Err(MistralRsError::EngineNotAvailable)
    }

    /// Drop the loaded model and free its VRAM / memory. Idempotent --
    /// if the model is already unloaded, this is a no-op that returns
    /// `Ok(())`.
    ///
    /// Note: if a streaming inference task is still holding an
    /// `Arc<Model>` clone, the underlying `Model` will only be dropped
    /// (and its VRAM freed) once that task finishes consuming its
    /// stream. `unload` always releases the provider's own reference
    /// immediately.
    ///
    /// # Errors
    ///
    /// This method currently never returns an error; the `Result` return
    /// type is preserved to match [`crate::MistralRsError`] conventions
    /// and the [`blazen_llm::traits::LocalModel`] trait contract.
    #[cfg(feature = "engine")]
    pub async fn unload(&self) -> Result<(), MistralRsError> {
        let mut guard = self.engine.write().await;
        // Drop the Arc<Model>. When no other clones remain (e.g. from an
        // in-flight streaming task), VRAM is freed by the Drop impl on
        // the underlying mistral.rs runner.
        *guard = None;
        let mut adapters_guard = self.mounted_adapters.write().await;
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
    pub async fn unload(&self) -> Result<(), MistralRsError> {
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

    /// Mount a `LoRA` adapter at runtime by rebuilding the engine with the
    /// adapter directory included in [`mistralrs::LoraModelBuilder`].
    ///
    /// # Errors
    ///
    /// Returns [`MistralRsError::InvalidOptions`] if an adapter with the same
    /// `adapter_id` is already mounted, or [`MistralRsError::AdapterFailed`]
    /// if the engine rebuild fails.
    #[cfg(feature = "engine")]
    pub async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        adapter_id: impl Into<String>,
        scale: f32,
    ) -> Result<MountedAdapter, MistralRsError> {
        let adapter_id = adapter_id.into();

        {
            let guard = self.mounted_adapters.read().await;
            if guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(MistralRsError::InvalidOptions(format!(
                    "adapter '{adapter_id}' is already mounted",
                )));
            }
        }

        let new_entry = MountedAdapter {
            adapter_id: adapter_id.clone(),
            adapter_dir: adapter_dir.to_path_buf(),
            scale,
        };

        let new_list: Vec<MountedAdapter> = {
            let guard = self.mounted_adapters.read().await;
            let mut list = guard.clone();
            list.push(new_entry.clone());
            list
        };

        let new_engine = build_engine(&self.options, &new_list).await?;

        {
            let mut engine_guard = self.engine.write().await;
            *engine_guard = Some(std::sync::Arc::new(new_engine));
        }
        {
            let mut adapters_guard = self.mounted_adapters.write().await;
            *adapters_guard = new_list;
        }
        Ok(new_entry)
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`MistralRsError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _adapter_id: impl Into<String>,
        _scale: f32,
    ) -> Result<MountedAdapter, MistralRsError> {
        Err(MistralRsError::EngineNotAvailable)
    }

    /// Unmount a previously mounted `LoRA` adapter by rebuilding the engine
    /// with the remaining adapter set.
    ///
    /// # Errors
    ///
    /// Returns [`MistralRsError::InvalidOptions`] if no adapter with the
    /// given `adapter_id` is currently mounted, or
    /// [`MistralRsError::AdapterFailed`] if the engine rebuild fails.
    #[cfg(feature = "engine")]
    pub async fn unload_adapter(&self, adapter_id: &str) -> Result<(), MistralRsError> {
        let new_list: Vec<MountedAdapter> = {
            let guard = self.mounted_adapters.read().await;
            if !guard.iter().any(|a| a.adapter_id == adapter_id) {
                return Err(MistralRsError::InvalidOptions(format!(
                    "adapter '{adapter_id}' is not mounted",
                )));
            }
            guard
                .iter()
                .filter(|a| a.adapter_id != adapter_id)
                .cloned()
                .collect()
        };

        let new_engine = build_engine(&self.options, &new_list).await?;

        {
            let mut engine_guard = self.engine.write().await;
            *engine_guard = Some(std::sync::Arc::new(new_engine));
        }
        {
            let mut adapters_guard = self.mounted_adapters.write().await;
            *adapters_guard = new_list;
        }
        Ok(())
    }

    /// Stub: engine not available.
    ///
    /// # Errors
    ///
    /// Always returns [`MistralRsError::EngineNotAvailable`].
    #[cfg(not(feature = "engine"))]
    #[allow(clippy::unused_async)]
    pub async fn unload_adapter(&self, _adapter_id: &str) -> Result<(), MistralRsError> {
        Err(MistralRsError::EngineNotAvailable)
    }

    /// Snapshot the list of `LoRA` adapters currently mounted on this provider.
    #[cfg(feature = "engine")]
    pub async fn list_adapters(&self) -> Vec<MountedAdapter> {
        self.mounted_adapters.read().await.clone()
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
async fn build_engine(
    opts: &MistralRsOptions,
    adapters: &[MountedAdapter],
) -> Result<mistralrs::Model, MistralRsError> {
    use mistralrs::{LoraModelBuilder, MultimodalModelBuilder, TextModelBuilder};

    let model_id = opts.base.model_id.as_deref().ok_or_else(|| {
        MistralRsError::InvalidOptions("model_id is required for engine init".into())
    })?;
    tracing::info!(
        model_id = %model_id,
        vision = opts.vision,
        adapter_count = adapters.len(),
        "loading mistral.rs model"
    );

    if opts.vision && !adapters.is_empty() {
        return Err(MistralRsError::InvalidOptions(
            "vision mode is incompatible with LoRA adapters -- \
             mistralrs 0.8.1 LoraModelBuilder wraps TextModelBuilder only"
                .into(),
        ));
    }

    if opts.vision {
        let mut builder = MultimodalModelBuilder::new(model_id);
        if let Some(ref tmpl) = opts.chat_template {
            builder = builder.with_chat_template(tmpl.clone());
        }
        builder
            .build()
            .await
            .map_err(|e| MistralRsError::Init(e.to_string()))
    } else {
        let mut builder = TextModelBuilder::new(model_id);
        if let Some(ref tmpl) = opts.chat_template {
            builder = builder.with_chat_template(tmpl.clone());
        }
        if adapters.is_empty() {
            builder
                .build()
                .await
                .map_err(|e| MistralRsError::Init(e.to_string()))
        } else {
            let adapter_paths = adapters
                .iter()
                .map(|a| a.adapter_dir.display().to_string())
                .collect::<Vec<_>>();
            LoraModelBuilder::from_text_model_builder(builder, adapter_paths)
                .build()
                .await
                .map_err(|e| MistralRsError::AdapterFailed(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Image decoding helpers (only with `engine` feature)
// ---------------------------------------------------------------------------

/// Decode a [`InferenceImage`] into an `image::DynamicImage`.
#[cfg(feature = "engine")]
fn decode_image(img: &InferenceImage) -> Result<image::DynamicImage, MistralRsError> {
    match &img.source {
        InferenceImageSource::Bytes(bytes) => image::load_from_memory(bytes).map_err(|e| {
            MistralRsError::InvalidOptions(format!("failed to decode image bytes: {e}"))
        }),
        InferenceImageSource::Path(path) => image::open(path).map_err(|e| {
            MistralRsError::InvalidOptions(format!(
                "failed to open image file {}: {e}",
                path.display()
            ))
        }),
    }
}

/// Convert a list of Blazen `ChatMessageInput`s into mistral.rs
/// `MultimodalMessages`, decoding every attached image via the `image`
/// crate.
///
/// If no message has image attachments, the resulting `MultimodalMessages`
/// contains only text — this is equivalent to `TextMessages` for the
/// mistral.rs runtime and is safe to send to any multimodal pipeline.
#[cfg(feature = "engine")]
fn build_multimodal_messages(
    messages: &[ChatMessageInput],
) -> Result<mistralrs::MultimodalMessages, MistralRsError> {
    use mistralrs::{MultimodalMessages, TextMessageRole};

    let mut out = MultimodalMessages::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => TextMessageRole::System,
            ChatRole::User => TextMessageRole::User,
            ChatRole::Assistant => TextMessageRole::Assistant,
            ChatRole::Tool => TextMessageRole::Tool,
        };
        if msg.images.is_empty() {
            out = out.add_message(role, &msg.text);
        } else {
            let mut decoded = Vec::with_capacity(msg.images.len());
            for img in &msg.images {
                decoded.push(decode_image(img)?);
            }
            out = out.add_image_message(role, &msg.text, decoded);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Engine-backed inference (only with `engine` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
impl MistralRsProvider {
    /// Return the loaded engine, loading it on first use.
    ///
    /// Uses a double-checked `RwLock` pattern: the fast path takes a read
    /// lock and clones out the existing `Arc<Model>`; if the engine is not
    /// yet loaded, it drops the read lock, acquires a write lock,
    /// re-checks (another task may have loaded it concurrently), and
    /// finally builds the engine.
    ///
    /// Returns an owned `Arc<Model>` so the caller can hold it across
    /// await points and pass it to spawned tasks without holding the
    /// read lock.
    async fn get_or_load_engine(&self) -> Result<std::sync::Arc<mistralrs::Model>, MistralRsError> {
        // Fast path: acquire read lock, check if already loaded, clone Arc.
        {
            let guard = self.engine.read().await;
            if let Some(model) = guard.as_ref() {
                return Ok(std::sync::Arc::clone(model));
            }
        }
        // Snapshot the mounted adapter list before taking the write lock,
        // so we never hold a lock across the engine build await.
        let adapters = self.mounted_adapters.read().await.clone();
        // Slow path: acquire write lock, double-check, build, clone Arc.
        let mut guard = self.engine.write().await;
        if guard.is_none() {
            let model = build_engine(&self.options, &adapters).await?;
            *guard = Some(std::sync::Arc::new(model));
        }
        // SAFETY: we just set Some above (or found Some from a concurrent loader).
        let model = guard.as_ref().expect("engine loaded above");
        Ok(std::sync::Arc::clone(model))
    }

    async fn infer_engine(
        &self,
        messages: Vec<ChatMessageInput>,
    ) -> Result<InferenceResult, MistralRsError> {
        use mistralrs::{TextMessageRole, TextMessages};

        // Refuse images when vision mode is disabled so the user gets a
        // clear diagnostic rather than a cryptic pipeline error.
        if !self.options.vision && messages.iter().any(ChatMessageInput::has_images) {
            return Err(MistralRsError::InvalidOptions(
                "image content supplied but vision mode is disabled -- \
                 set MistralRsOptions.vision = true and load a vision-capable model"
                    .into(),
            ));
        }

        let engine = self.get_or_load_engine().await?;

        let response = if self.options.vision {
            let msgs = build_multimodal_messages(&messages)?;
            engine
                .send_chat_request(msgs)
                .await
                .map_err(|e| MistralRsError::Inference(e.to_string()))?
        } else {
            let mut msgs = TextMessages::new();
            for msg in &messages {
                let mr_role = match msg.role {
                    ChatRole::System => TextMessageRole::System,
                    ChatRole::User => TextMessageRole::User,
                    ChatRole::Assistant => TextMessageRole::Assistant,
                    ChatRole::Tool => TextMessageRole::Tool,
                };
                msgs = msgs.add_message(mr_role, &msg.text);
            }
            engine
                .send_chat_request(msgs)
                .await
                .map_err(|e| MistralRsError::Inference(e.to_string()))?
        };

        let choice = response
            .choices
            .first()
            .ok_or_else(|| MistralRsError::Inference("no choices in response".into()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| InferenceToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        #[allow(clippy::cast_possible_truncation)]
        let usage = InferenceUsage {
            prompt_tokens: response.usage.prompt_tokens as u32,
            completion_tokens: response.usage.completion_tokens as u32,
            total_tokens: response.usage.total_tokens as u32,
            total_time_sec: response.usage.total_time_sec,
        };

        Ok(InferenceResult {
            content: choice.message.content.clone(),
            reasoning_content: choice.message.reasoning_content.clone(),
            tool_calls,
            finish_reason: choice.finish_reason.clone(),
            model: response.model.clone(),
            usage,
        })
    }

    /// Returns a `'static` stream by spawning a task that owns a clone of
    /// the engine `Arc<RwLock<...>>` and the options, initialises the engine
    /// lazily, and forwards mapped chunks through a channel.
    ///
    /// Engine errors (including failure to load or start the stream) are
    /// delivered as error items on the returned stream. The spawned task
    /// holds an owned `Arc<Model>` for the entire duration of streaming,
    /// so an explicit `unload` from another task cannot yank the model
    /// out from under an in-flight stream — the `Model` stays alive
    /// until this task drops its `Arc`.
    fn infer_stream_engine(&self, messages: Vec<ChatMessageInput>) -> InferenceChunkStream {
        use tokio_stream::wrappers::ReceiverStream;

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // Clone the Arc + options so the spawned task can outlive `&self`.
        let engine_lock = std::sync::Arc::clone(&self.engine);
        let adapters_lock = std::sync::Arc::clone(&self.mounted_adapters);
        let options = self.options.clone();

        tokio::spawn(async move {
            use mistralrs::{Response, TextMessageRole, TextMessages};

            // Lazily initialise the engine inside the spawned task. We
            // inline the double-checked `RwLock` load pattern here instead
            // of calling `get_or_load_engine` so that we can preserve the
            // existing error-delivery contract (load errors appear as the
            // first stream item, not as a synchronous return error).
            let engine: std::sync::Arc<mistralrs::Model> = {
                // Fast path: acquire read lock, check if already loaded.
                {
                    let guard = engine_lock.read().await;
                    if let Some(model) = guard.as_ref() {
                        std::sync::Arc::clone(model)
                    } else {
                        drop(guard);
                        // Snapshot adapters before taking the write lock.
                        let adapters = adapters_lock.read().await.clone();
                        // Slow path: acquire write lock, double-check, build.
                        let mut guard = engine_lock.write().await;
                        if guard.is_none() {
                            match build_engine(&options, &adapters).await {
                                Ok(model) => {
                                    *guard = Some(std::sync::Arc::new(model));
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                    return;
                                }
                            }
                        }
                        // SAFETY: we just set Some above (or found Some from a concurrent loader).
                        std::sync::Arc::clone(guard.as_ref().expect("engine loaded above"))
                    }
                }
            };

            // Build the appropriate request type for the pipeline that was
            // loaded. Vision mode uses `MultimodalMessages` (which also
            // handles text-only inputs); text mode uses `TextMessages`.
            //
            // Boxing into a `dyn` trait object is not possible here because
            // `stream_chat_request` is generic over `RequestLike` — so we
            // branch at the call site instead.
            let stream_result = if options.vision {
                let msgs = match build_multimodal_messages(&messages) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };
                engine.stream_chat_request(msgs).await
            } else {
                let mut msgs = TextMessages::new();
                for msg in &messages {
                    let mr_role = match msg.role {
                        ChatRole::System => TextMessageRole::System,
                        ChatRole::User => TextMessageRole::User,
                        ChatRole::Assistant => TextMessageRole::Assistant,
                        ChatRole::Tool => TextMessageRole::Tool,
                    };
                    msgs = msgs.add_message(mr_role, &msg.text);
                }
                engine.stream_chat_request(msgs).await
            };

            let mut stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(MistralRsError::Inference(e.to_string()))).await;
                    return;
                }
            };

            while let Some(resp) = stream.next().await {
                let mapped = match resp {
                    Response::Chunk(chunk) => {
                        let Some(choice) = chunk.choices.first() else {
                            continue;
                        };
                        let tool_calls = choice
                            .delta
                            .tool_calls
                            .as_ref()
                            .map(|tcs| {
                                tcs.iter()
                                    .map(|tc| InferenceToolCall {
                                        id: tc.id.clone(),
                                        name: tc.function.name.clone(),
                                        arguments: tc.function.arguments.clone(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();

                        Ok(InferenceChunk {
                            delta: choice.delta.content.clone(),
                            reasoning_delta: choice.delta.reasoning_content.clone(),
                            tool_calls,
                            finish_reason: choice.finish_reason.clone(),
                        })
                    }
                    Response::ModelError(msg, _) => Err(MistralRsError::Inference(msg)),
                    Response::InternalError(e) | Response::ValidationError(e) => {
                        Err(MistralRsError::Inference(e.to_string()))
                    }
                    _ => continue,
                };

                if tx.send(mapped).await.is_err() {
                    break; // Receiver dropped.
                }
            }
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

    #[test]
    fn from_options_stores_model_id() {
        let provider =
            MistralRsProvider::from_options(MistralRsOptions::required("test-org/test-model"))
                .expect("should succeed");
        assert_eq!(provider.model_id(), "test-org/test-model");
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let result = MistralRsProvider::from_options(MistralRsOptions::required(""));
        assert!(result.is_err());
    }

    /// GPU/inference smoke: download a small chat model and generate text.
    /// Gated on `engine` (the real inference path) and `#[ignore]`'d so the
    /// beastpc-e2e `--run-ignored only` step runs it. Runs on CPU unless the
    /// binary additionally enables `mistralrs/cuda` (the crate-level `cuda`
    /// feature is a marker — see Cargo.toml).
    #[cfg(feature = "engine")]
    #[tokio::test]
    #[ignore = "downloads ~1 GB Qwen2.5-0.5B-Instruct + runs inference"]
    async fn smoke_generate_small_model() {
        // A small FULL HF model (safetensors + tokenizer.json) — the provider
        // builds via `TextModelBuilder`, which needs a tokenizer.json that
        // GGUF-only repos don't ship.
        let opts = MistralRsOptions {
            base: blazen_local_llm::LocalLlmOptions::new()
                .with_model_id("Qwen/Qwen2.5-0.5B-Instruct")
                .with_context_length(2048),
            ..MistralRsOptions::default()
        };
        let provider = MistralRsProvider::from_options(opts).expect("options valid");
        provider.load().await.expect("model should load");

        let messages = vec![
            ChatMessageInput::text(ChatRole::System, "You are a helpful assistant."),
            ChatMessageInput::text(ChatRole::User, "What is 2 + 2? Answer in one word."),
        ];
        let result = provider
            .infer(messages)
            .await
            .expect("inference should succeed");
        assert!(
            result.content.as_deref().is_some_and(|c| !c.is_empty()),
            "should produce non-empty text"
        );
        assert!(
            result.usage.completion_tokens > 0,
            "should produce completion tokens"
        );
    }

    #[test]
    fn engine_not_available_display() {
        let err = MistralRsError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[test]
    fn engine_available_reflects_feature() {
        let provider =
            MistralRsProvider::from_options(MistralRsOptions::required("test/model")).unwrap();
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
            let opts = MistralRsOptions::required("test/model");
            let provider = MistralRsProvider::from_options(opts).unwrap();
            let result = provider.infer(vec![]).await;
            assert!(
                matches!(result, Err(MistralRsError::EngineNotAvailable)),
                "expected EngineNotAvailable, got {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn infer_stream_without_engine_returns_error() {
        #[cfg(not(feature = "engine"))]
        {
            let opts = MistralRsOptions::required("test/model");
            let provider = MistralRsProvider::from_options(opts).unwrap();
            let result = provider.infer_stream(vec![]).await;
            assert!(
                matches!(result, Err(MistralRsError::EngineNotAvailable)),
                "expected EngineNotAvailable"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ChatMessageInput / InferenceImage constructors (always available)
    // -----------------------------------------------------------------------

    #[test]
    fn chat_message_input_text_only() {
        let msg = ChatMessageInput::text(ChatRole::User, "hello");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.text, "hello");
        assert!(msg.images.is_empty());
        assert!(!msg.has_images());
    }

    #[test]
    fn chat_message_input_with_images() {
        let msg = ChatMessageInput::with_images(
            ChatRole::User,
            "describe this",
            vec![
                InferenceImage::from_bytes(vec![1, 2, 3]),
                InferenceImage::from_path("/tmp/a.png"),
            ],
        );
        assert!(msg.has_images());
        assert_eq!(msg.images.len(), 2);
        assert!(matches!(
            &msg.images[0].source,
            InferenceImageSource::Bytes(b) if b == &vec![1, 2, 3]
        ));
        assert!(matches!(
            &msg.images[1].source,
            InferenceImageSource::Path(p) if p == std::path::Path::new("/tmp/a.png")
        ));
    }

    // -----------------------------------------------------------------------
    // Vision mode enforcement
    // -----------------------------------------------------------------------

    /// Without vision mode enabled, supplying image content must be
    /// rejected with `InvalidOptions` before the engine is ever touched.
    /// `InferenceChunkStream` is not `Debug`, so we only `matches!` here.
    #[cfg(feature = "engine")]
    #[tokio::test]
    async fn infer_stream_rejects_images_when_vision_disabled() {
        let opts = MistralRsOptions::required("test/model");
        let provider = MistralRsProvider::from_options(opts).unwrap();
        let messages = vec![ChatMessageInput::with_images(
            ChatRole::User,
            "describe",
            vec![InferenceImage::from_bytes(vec![0; 8])],
        )];
        let result = provider.infer_stream(messages).await;
        assert!(
            matches!(result, Err(MistralRsError::InvalidOptions(_))),
            "expected InvalidOptions"
        );
    }

    // -----------------------------------------------------------------------
    // build_multimodal_messages: structural verification without inference
    // -----------------------------------------------------------------------

    /// Verify that `build_multimodal_messages` accepts a well-formed PNG
    /// byte buffer and produces a populated `MultimodalMessages` without
    /// loading any model. The actual multimodal request type from
    /// mistral.rs does not expose its internal state for direct inspection,
    /// so we rely on the fact that successful construction means both the
    /// image decoder and the builder walked through every message without
    /// error.
    #[cfg(feature = "engine")]
    #[test]
    fn build_multimodal_messages_decodes_png_bytes() {
        // Minimal 1x1 red PNG encoded with the `image` crate -- avoids
        // depending on a disk fixture for a pure library-side test.
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
        let mut png_bytes: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .expect("encode 1x1 PNG");

        let messages = vec![
            ChatMessageInput::text(ChatRole::System, "be concise"),
            ChatMessageInput::with_images(
                ChatRole::User,
                "describe",
                vec![InferenceImage::from_bytes(png_bytes)],
            ),
        ];

        let result = build_multimodal_messages(&messages);
        assert!(
            result.is_ok(),
            "expected build_multimodal_messages to succeed, got {result:?}"
        );
    }

    /// Bad image bytes produce a clean `InvalidOptions` error instead of a
    /// panic.
    #[cfg(feature = "engine")]
    #[test]
    fn build_multimodal_messages_rejects_invalid_bytes() {
        let messages = vec![ChatMessageInput::with_images(
            ChatRole::User,
            "x",
            vec![InferenceImage::from_bytes(vec![0xff; 16])],
        )];
        let result = build_multimodal_messages(&messages);
        assert!(
            matches!(result, Err(MistralRsError::InvalidOptions(_))),
            "expected InvalidOptions, got {result:?}"
        );
    }

    /// Text-only messages round-trip through the multimodal builder
    /// without any image decoding.
    #[cfg(feature = "engine")]
    #[test]
    fn build_multimodal_messages_text_only() {
        let messages = vec![
            ChatMessageInput::text(ChatRole::System, "system"),
            ChatMessageInput::text(ChatRole::User, "user"),
            ChatMessageInput::text(ChatRole::Assistant, "assistant"),
        ];
        let result = build_multimodal_messages(&messages);
        assert!(result.is_ok());
    }
}
