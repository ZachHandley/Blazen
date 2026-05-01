//! The [`FastEmbedModel`] type providing local embeddings via fastembed.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::FastEmbedOptions;

/// Error type for fastembed operations.
#[derive(Debug)]
pub enum FastEmbedError {
    /// The model name was not recognised by fastembed.
    UnknownModel(String),
    /// The fastembed model failed to initialise.
    Init(String),
    /// An embedding operation failed.
    Embed(String),
    /// The internal mutex was poisoned.
    MutexPoisoned(String),
    /// A blocking task panicked.
    TaskPanicked(String),
}

impl fmt::Display for FastEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownModel(msg) => write!(f, "unknown fastembed model: {msg}"),
            Self::Init(msg) => write!(f, "fastembed init failed: {msg}"),
            Self::Embed(msg) => write!(f, "fastembed embed failed: {msg}"),
            Self::MutexPoisoned(msg) => write!(f, "fastembed mutex poisoned: {msg}"),
            Self::TaskPanicked(msg) => write!(f, "fastembed blocking task panicked: {msg}"),
        }
    }
}

impl std::error::Error for FastEmbedError {}

/// Response from a fastembed embedding operation.
#[derive(Debug, Clone)]
pub struct FastEmbedResponse {
    /// The embedding vectors -- one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// The model identifier that produced these embeddings.
    pub model: String,
}

/// A local embedding model backed by [`fastembed`] (ONNX Runtime).
///
/// Constructed via [`FastEmbedModel::from_options`]. The underlying
/// `fastembed::TextEmbedding` is synchronous, so all calls to
/// [`FastEmbedModel::embed`] are dispatched onto Tokio's blocking thread pool
/// via [`tokio::task::spawn_blocking`].
pub struct FastEmbedModel {
    /// The fastembed model handle. Wrapped in `Arc<Mutex<...>>` because
    /// `TextEmbedding::embed` takes `&mut self` and we need to move the
    /// handle into `spawn_blocking` closures.
    model: Arc<Mutex<fastembed::TextEmbedding>>,
    /// The model identifier string returned by `model_id()`.
    model_id: String,
    /// Embedding dimensionality for this model.
    dims: usize,
    /// Optional batch size override.
    batch_size: Option<usize>,
}

// `fastembed::TextEmbedding` is `Send` (it contains `ort::Session` which is
// `Send`). `Arc<Mutex<T: Send>>` is `Send + Sync`, so `FastEmbedModel`
// auto-derives both traits.

impl FastEmbedModel {
    /// Construct a new [`FastEmbedModel`] from the given options.
    ///
    /// This is a blocking operation that may download model weights from
    /// `HuggingFace` on first use. Call from a context where blocking is
    /// acceptable (e.g. application startup), or wrap in
    /// [`tokio::task::spawn_blocking`].
    ///
    /// # Errors
    ///
    /// Returns [`FastEmbedError`] if the fastembed model fails to initialise
    /// (e.g. unknown model name, network error during download).
    pub fn from_options(opts: FastEmbedOptions) -> Result<Self, FastEmbedError> {
        // Resolve the fastembed EmbeddingModel enum variant.
        let fe_model = if let Some(ref name) = opts.model_name {
            name.parse::<fastembed::EmbeddingModel>()
                .map_err(|e| FastEmbedError::UnknownModel(format!("\"{name}\": {e}")))?
        } else {
            fastembed::EmbeddingModel::default()
        };

        // Look up the model info to get dimensions.
        let model_info =
            <fastembed::EmbeddingModel as fastembed::ModelTrait>::get_model_info(&fe_model)
                .ok_or_else(|| {
                    FastEmbedError::Init(format!("no model info found for {fe_model:?}"))
                })?;
        let dims = model_info.dim;
        let model_code = model_info.model_code.clone();

        // Build init options.
        let mut init_opts = fastembed::TextInitOptions::new(fe_model);
        if let Some(cache_dir) = opts.cache_dir {
            init_opts = init_opts.with_cache_dir(cache_dir);
        }
        if let Some(show) = opts.show_download_progress {
            init_opts = init_opts.with_show_download_progress(show);
        }

        let te = fastembed::TextEmbedding::try_new(init_opts)
            .map_err(|e| FastEmbedError::Init(e.to_string()))?;

        Ok(Self {
            model: Arc::new(Mutex::new(te)),
            model_id: model_code,
            dims,
            batch_size: opts.max_batch_size,
        })
    }

    /// The model identifier (e.g. `"Xenova/bge-small-en-v1.5"`).
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Embedding vector dimensionality for this model.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dims
    }

    /// Embed one or more texts, returning one vector per input text.
    ///
    /// The fastembed crate's embed method is synchronous -- this function
    /// dispatches the work onto Tokio's blocking thread pool via
    /// [`tokio::task::spawn_blocking`] to avoid starving the async runtime.
    ///
    /// # Errors
    ///
    /// Returns [`FastEmbedError`] if the underlying fastembed call fails or
    /// the blocking task panics.
    pub async fn embed(&self, texts: &[String]) -> Result<FastEmbedResponse, FastEmbedError> {
        if texts.is_empty() {
            return Ok(FastEmbedResponse {
                embeddings: vec![],
                model: self.model_id.clone(),
            });
        }

        // Clone inputs and the Arc handle so we can move them into the
        // blocking closure.
        let texts_owned: Vec<String> = texts.to_vec();
        let batch_size = self.batch_size;
        let model_id = self.model_id.clone();
        let model_handle = Arc::clone(&self.model);

        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model = model_handle
                .lock()
                .map_err(|e| FastEmbedError::MutexPoisoned(e.to_string()))?;
            let result: Vec<Vec<f32>> = model
                .embed(&texts_owned, batch_size)
                .map_err(|e| FastEmbedError::Embed(e.to_string()))?;
            Ok::<Vec<Vec<f32>>, FastEmbedError>(result)
        })
        .await
        .map_err(|e| FastEmbedError::TaskPanicked(e.to_string()))??;

        Ok(FastEmbedResponse {
            embeddings,
            model: model_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires model download from HuggingFace"]
    fn from_options_default_loads_model() {
        let model = FastEmbedModel::from_options(FastEmbedOptions::default())
            .expect("should create model with default options");
        assert!(model.dimensions() > 0);
        assert!(!model.model_id().is_empty());
    }

    #[tokio::test]
    #[ignore = "requires model download from HuggingFace"]
    async fn embed_returns_correct_count() {
        let model = FastEmbedModel::from_options(FastEmbedOptions::default())
            .expect("should create model with default options");
        let response = model
            .embed(&["hello".into(), "world".into()])
            .await
            .expect("embedding should succeed");
        assert_eq!(response.embeddings.len(), 2);
        assert!(!response.embeddings[0].is_empty());
        assert_eq!(response.embeddings[0].len(), model.dimensions());
    }

    #[tokio::test]
    async fn embed_empty_input_returns_empty() {
        // This test does NOT require model download because we short-circuit
        // on empty input. But we still need a model instance, so we skip if
        // the model is not cached locally.
        let Ok(model) = FastEmbedModel::from_options(FastEmbedOptions::default()) else {
            eprintln!("skipping embed_empty_input_returns_empty: model not available");
            return;
        };
        let response = model.embed(&[]).await.expect("empty embed should succeed");
        assert!(response.embeddings.is_empty());
    }
}
