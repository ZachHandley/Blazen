//! The [`CandleEmbedModel`] type -- local text embeddings via candle.
//!
//! When the `engine` feature is enabled, [`CandleEmbedModel::from_options`]
//! downloads and loads a BERT-family sentence-transformer from `HuggingFace` Hub,
//! and [`CandleEmbedModel::embed`] runs forward passes through the model to
//! produce embedding vectors.
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but [`CandleEmbedModel::embed`] returns
//! [`CandleEmbedError::EngineNotAvailable`].

use std::fmt;

use crate::CandleEmbedOptions;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error type for candle embedding operations.
#[derive(Debug)]
pub enum CandleEmbedError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// An embedding operation failed.
    Embedding(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
    /// A blocking task panicked.
    TaskPanicked(String),
}

impl fmt::Display for CandleEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "candle embed invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "candle embed model load failed: {msg}"),
            Self::Embedding(msg) => write!(f, "candle embed operation failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "candle embed engine not available: compile with the `engine` feature"
            ),
            Self::TaskPanicked(msg) => {
                write!(f, "candle embed blocking task panicked: {msg}")
            }
        }
    }
}

impl std::error::Error for CandleEmbedError {}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Response from a candle embedding operation.
#[derive(Debug, Clone)]
pub struct CandleEmbedResponse {
    /// The embedding vectors -- one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// The model identifier that produced these embeddings.
    pub model: String,
}

// ---------------------------------------------------------------------------
// Engine-backed implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
mod engine {
    use std::sync::{Arc, Mutex};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use hf_hub::api::tokio::Api;
    use tokenizers::Tokenizer;

    use super::{CandleEmbedError, CandleEmbedOptions, CandleEmbedResponse};

    /// Inner state that holds the loaded model and tokenizer.
    struct CandleEngine {
        model: BertModel,
        tokenizer: Tokenizer,
        device: Device,
        dims: usize,
    }

    // SAFETY: `BertModel` is backed by `candle_core::Tensor` which stores data
    // on a `Device`. On CPU the backing storage is a heap-allocated `Vec` which
    // is `Send`. The `Tokenizer` from the `tokenizers` crate is also `Send`.
    // We wrap them in `Mutex` to synchronise access, making the whole thing
    // `Send + Sync`.
    #[allow(unsafe_code)]
    unsafe impl Send for CandleEngine {}

    /// A local embedding model backed by candle.
    pub struct CandleEmbedModel {
        /// The resolved model ID.
        model_id: String,
        /// Full options preserved for reference.
        #[allow(dead_code)]
        options: CandleEmbedOptions,
        /// The loaded engine. Wrapped in `Arc<Mutex<...>>` because the model
        /// is not `Sync` and we dispatch onto `spawn_blocking`.
        engine: Arc<Mutex<CandleEngine>>,
        /// Embedding dimensionality.
        dims: usize,
    }

    impl CandleEmbedModel {
        /// Create a new embedding model from the given options.
        ///
        /// Downloads model weights, tokenizer, and config from `HuggingFace` Hub
        /// on first use. This is an async operation that should be called at
        /// startup or wrapped in a context where blocking is acceptable.
        ///
        /// # Errors
        ///
        /// Returns [`CandleEmbedError`] if option validation fails, the model
        /// cannot be downloaded, or the engine fails to initialise.
        pub async fn from_options(opts: CandleEmbedOptions) -> Result<Self, CandleEmbedError> {
            validate_options(&opts)?;

            let model_id = opts.effective_model_id().to_owned();
            let revision = opts.revision.clone().unwrap_or_else(|| "main".to_owned());

            tracing::info!(model = %model_id, revision = %revision, "loading candle embedding model");

            // Resolve the candle device.
            let device = resolve_device(&opts)?;

            // Download model artifacts from `HuggingFace` Hub.
            let api = Api::new().map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to create HF API client: {e}"))
            })?;
            let repo = api.repo(hf_hub::Repo::with_revision(
                model_id.clone(),
                hf_hub::RepoType::Model,
                revision,
            ));

            let config_path = repo.get("config.json").await.map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to download config.json: {e}"))
            })?;
            let tokenizer_path = repo.get("tokenizer.json").await.map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to download tokenizer.json: {e}"))
            })?;
            let weights_path = repo.get("model.safetensors").await.map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to download model.safetensors: {e}"))
            })?;

            // Load config.
            let config_bytes = std::fs::read(&config_path).map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to read config.json: {e}"))
            })?;
            let config: BertConfig = serde_json::from_slice(&config_bytes).map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to parse config.json: {e}"))
            })?;

            let dims = config.hidden_size;

            // Load tokenizer.
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to load tokenizer: {e}"))
            })?;

            // Load weights and build the model.
            //
            // SAFETY: Memory-mapping the safetensors file is safe as long as
            // the file is not modified while mapped. This is the standard
            // pattern used by candle for loading model weights.
            #[allow(unsafe_code)]
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).map_err(
                    |e| CandleEmbedError::ModelLoad(format!("failed to load model weights: {e}")),
                )?
            };

            let model = BertModel::load(vb, &config).map_err(|e| {
                CandleEmbedError::ModelLoad(format!("failed to build BERT model: {e}"))
            })?;

            tracing::info!(model = %model_id, dims = dims, "candle embedding model loaded");

            let engine = CandleEngine {
                model,
                tokenizer,
                device,
                dims,
            };

            Ok(Self {
                model_id,
                options: opts,
                engine: Arc::new(Mutex::new(engine)),
                dims,
            })
        }

        /// The `HuggingFace` model ID that was configured at construction time.
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
        /// The candle forward pass is CPU-bound -- this function dispatches the
        /// work onto Tokio's blocking thread pool via [`tokio::task::spawn_blocking`].
        ///
        /// # Errors
        ///
        /// Returns [`CandleEmbedError`] if tokenization or inference fails.
        pub async fn embed(
            &self,
            texts: &[String],
        ) -> Result<CandleEmbedResponse, CandleEmbedError> {
            if texts.is_empty() {
                return Ok(CandleEmbedResponse {
                    embeddings: vec![],
                    model: self.model_id.clone(),
                });
            }

            let texts_owned: Vec<String> = texts.to_vec();
            let model_id = self.model_id.clone();
            let engine_handle = Arc::clone(&self.engine);

            let embeddings = tokio::task::spawn_blocking(move || {
                let engine = engine_handle
                    .lock()
                    .map_err(|e| CandleEmbedError::Embedding(format!("mutex poisoned: {e}")))?;

                run_inference(&engine, &texts_owned)
            })
            .await
            .map_err(|e| CandleEmbedError::TaskPanicked(e.to_string()))??;

            Ok(CandleEmbedResponse {
                embeddings,
                model: model_id,
            })
        }
    }

    /// Run the BERT forward pass and mean-pool + L2-normalize the output.
    fn run_inference(
        engine: &CandleEngine,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, CandleEmbedError> {
        let tokenizer = &engine.tokenizer;
        let model = &engine.model;
        let device = &engine.device;

        // Tokenize all inputs.
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| CandleEmbedError::Embedding(format!("tokenization failed: {e}")))?;

        let n = encodings.len();

        // Build padded input_ids and attention_mask tensors.
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut all_ids: Vec<u32> = Vec::with_capacity(n * max_len);
        let mut all_type_ids: Vec<u32> = Vec::with_capacity(n * max_len);
        let mut all_mask: Vec<f32> = Vec::with_capacity(n * max_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let type_ids = enc.get_type_ids();
            let len = ids.len();

            all_ids.extend_from_slice(ids);
            all_type_ids.extend_from_slice(type_ids);
            all_mask.extend(std::iter::repeat_n(1.0_f32, len));

            // Pad to max_len.
            let pad = max_len - len;
            if pad > 0 {
                all_ids.extend(std::iter::repeat_n(0_u32, pad));
                all_type_ids.extend(std::iter::repeat_n(0_u32, pad));
                all_mask.extend(std::iter::repeat_n(0.0_f32, pad));
            }
        }

        let input_ids = Tensor::from_vec(all_ids, (n, max_len), device).map_err(|e| {
            CandleEmbedError::Embedding(format!("failed to create input_ids tensor: {e}"))
        })?;
        let token_type_ids = Tensor::from_vec(all_type_ids, (n, max_len), device).map_err(|e| {
            CandleEmbedError::Embedding(format!("failed to create token_type_ids tensor: {e}"))
        })?;
        let attention_mask =
            Tensor::from_vec(all_mask.clone(), (n, max_len), device).map_err(|e| {
                CandleEmbedError::Embedding(format!("failed to create attention_mask tensor: {e}"))
            })?;

        // Forward pass: shape (batch, seq_len, hidden_size).
        let output = model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| CandleEmbedError::Embedding(format!("model forward pass failed: {e}")))?;

        // Mean pooling: mask out padding tokens, sum, divide by count.
        // attention_mask shape: (batch, seq_len) -> unsqueeze to (batch, seq_len, 1).
        let mask_3d = Tensor::from_vec(all_mask, (n, max_len), device)
            .and_then(|t| t.unsqueeze(2))
            .map_err(|e| {
                CandleEmbedError::Embedding(format!("failed to reshape attention mask: {e}"))
            })?;

        // Broadcast multiply to zero out padding.
        let masked = output
            .broadcast_mul(&mask_3d)
            .map_err(|e| CandleEmbedError::Embedding(format!("mask multiply failed: {e}")))?;

        // Sum over seq_len dimension.
        let summed = masked
            .sum(1)
            .map_err(|e| CandleEmbedError::Embedding(format!("sum over sequence failed: {e}")))?;

        // Count of non-padding tokens per sample.
        let counts = mask_3d
            .sum(1)
            .map_err(|e| CandleEmbedError::Embedding(format!("mask count failed: {e}")))?;

        // Avoid division by zero (clamp counts to minimum 1e-9).
        let counts_clamped = counts
            .clamp(1e-9_f64, f64::MAX)
            .map_err(|e| CandleEmbedError::Embedding(format!("clamp failed: {e}")))?;

        // Mean pool: (batch, hidden_size).
        let mean_pooled = summed.broadcast_div(&counts_clamped).map_err(|e| {
            CandleEmbedError::Embedding(format!("mean pooling division failed: {e}"))
        })?;

        // L2 normalize: divide each vector by its L2 norm.
        let l2_norms = mean_pooled
            .sqr()
            .and_then(|t| t.sum(1))
            .and_then(|t| t.sqrt())
            .and_then(|t| t.unsqueeze(1))
            .and_then(|t| t.clamp(1e-12_f64, f64::MAX))
            .map_err(|e| CandleEmbedError::Embedding(format!("L2 norm computation failed: {e}")))?;

        let normalized = mean_pooled
            .broadcast_div(&l2_norms)
            .map_err(|e| CandleEmbedError::Embedding(format!("L2 normalization failed: {e}")))?;

        // Convert to Vec<Vec<f32>>.
        let flat: Vec<f32> = normalized
            .to_vec2()
            .map_err(|e| {
                CandleEmbedError::Embedding(format!("tensor to vec conversion failed: {e}"))
            })?
            .into_iter()
            .flatten()
            .collect();

        let hidden = engine.dims;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * hidden;
            let end = start + hidden;
            result.push(flat[start..end].to_vec());
        }

        Ok(result)
    }

    /// Validate options (shared between engine and non-engine paths).
    fn validate_options(opts: &CandleEmbedOptions) -> Result<(), CandleEmbedError> {
        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(CandleEmbedError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        if let Some(ref model_id) = opts.model_id
            && model_id.is_empty()
        {
            return Err(CandleEmbedError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(ref revision) = opts.revision
            && revision.is_empty()
        {
            return Err(CandleEmbedError::InvalidOptions(
                "revision must not be empty when specified".into(),
            ));
        }

        Ok(())
    }

    /// Map the device string from options to a candle [`Device`].
    fn resolve_device(opts: &CandleEmbedOptions) -> Result<Device, CandleEmbedError> {
        let device_str = opts.device.as_deref().unwrap_or("cpu");
        match device_str {
            "cpu" => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            s if s.starts_with("cuda") => {
                let ordinal = if s == "cuda" {
                    0
                } else if let Some(suffix) = s.strip_prefix("cuda:") {
                    suffix.parse::<usize>().map_err(|e| {
                        CandleEmbedError::InvalidOptions(format!(
                            "invalid CUDA device ordinal: {e}"
                        ))
                    })?
                } else {
                    return Err(CandleEmbedError::InvalidOptions(format!(
                        "unrecognised device specifier: {s}"
                    )));
                };
                Device::new_cuda(ordinal).map_err(|e| {
                    CandleEmbedError::InvalidOptions(format!(
                        "failed to initialise CUDA device {ordinal}: {e}"
                    ))
                })
            }
            #[cfg(not(feature = "cuda"))]
            s if s.starts_with("cuda") => Err(CandleEmbedError::InvalidOptions(
                "CUDA requested but the `cuda` feature is not enabled".into(),
            )),
            #[cfg(feature = "metal")]
            "metal" => Device::new_metal(0).map_err(|e| {
                CandleEmbedError::InvalidOptions(format!("failed to initialise Metal device: {e}"))
            }),
            #[cfg(not(feature = "metal"))]
            "metal" => Err(CandleEmbedError::InvalidOptions(
                "Metal requested but the `metal` feature is not enabled".into(),
            )),
            other => Err(CandleEmbedError::InvalidOptions(format!(
                "unrecognised device specifier: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Non-engine stub implementation
// ---------------------------------------------------------------------------

#[cfg(not(feature = "engine"))]
mod stub {
    use super::{CandleEmbedError, CandleEmbedOptions, CandleEmbedResponse};

    /// A local embedding model backed by candle (stub -- engine not available).
    pub struct CandleEmbedModel {
        /// The resolved model ID.
        model_id: String,
        /// Full options preserved for reference.
        #[allow(dead_code)]
        options: CandleEmbedOptions,
    }

    impl CandleEmbedModel {
        /// Create a new embedding model from the given options.
        ///
        /// Without the `engine` feature, only option validation is performed.
        /// The [`embed`](Self::embed) method will return
        /// [`CandleEmbedError::EngineNotAvailable`].
        ///
        /// # Errors
        ///
        /// Returns [`CandleEmbedError::InvalidOptions`] if the device string is
        /// present but empty, or if the model ID is present but empty.
        pub fn from_options(opts: CandleEmbedOptions) -> Result<Self, CandleEmbedError> {
            if let Some(ref device) = opts.device
                && device.is_empty()
            {
                return Err(CandleEmbedError::InvalidOptions(
                    "device must not be empty when specified".into(),
                ));
            }

            if let Some(ref model_id) = opts.model_id
                && model_id.is_empty()
            {
                return Err(CandleEmbedError::InvalidOptions(
                    "model_id must not be empty when specified".into(),
                ));
            }

            if let Some(ref revision) = opts.revision
                && revision.is_empty()
            {
                return Err(CandleEmbedError::InvalidOptions(
                    "revision must not be empty when specified".into(),
                ));
            }

            let model_id = opts.effective_model_id().to_owned();

            Ok(Self {
                model_id,
                options: opts,
            })
        }

        /// The `HuggingFace` model ID that was configured at construction time.
        #[must_use]
        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        /// Embedding vector dimensionality for this model.
        ///
        /// Returns 0 when the engine feature is not enabled, because the actual
        /// model config has not been loaded.
        #[must_use]
        pub fn dimensions(&self) -> usize {
            0
        }

        /// Embed texts (stub).
        ///
        /// # Errors
        ///
        /// Always returns [`CandleEmbedError::EngineNotAvailable`].
        #[allow(clippy::unused_async)]
        pub async fn embed(
            &self,
            _texts: &[String],
        ) -> Result<CandleEmbedResponse, CandleEmbedError> {
            Err(CandleEmbedError::EngineNotAvailable)
        }
    }
}

// ---------------------------------------------------------------------------
// Re-export the correct implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "engine")]
pub use engine::CandleEmbedModel;
#[cfg(not(feature = "engine"))]
pub use stub::CandleEmbedModel;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CandleEmbedOptions;

    #[test]
    fn engine_not_available_display() {
        let err = CandleEmbedError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(
            msg.contains("engine"),
            "message should mention engine: {msg}"
        );
    }

    #[test]
    fn invalid_options_display() {
        let err = CandleEmbedError::InvalidOptions("test error".into());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn model_load_display() {
        let err = CandleEmbedError::ModelLoad("not found".into());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn embedding_display() {
        let err = CandleEmbedError::Embedding("inference failed".into());
        assert!(err.to_string().contains("inference failed"));
    }

    #[test]
    fn task_panicked_display() {
        let err = CandleEmbedError::TaskPanicked("panic".into());
        assert!(err.to_string().contains("panic"));
    }

    // Non-engine tests (always run).
    #[cfg(not(feature = "engine"))]
    mod stub_tests {
        use super::*;

        #[test]
        fn from_options_with_defaults() {
            let opts = CandleEmbedOptions::default();
            let model = CandleEmbedModel::from_options(opts).expect("should succeed");
            assert_eq!(model.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
        }

        #[test]
        fn from_options_with_custom_model() {
            let opts = CandleEmbedOptions {
                model_id: Some("BAAI/bge-small-en-v1.5".into()),
                ..CandleEmbedOptions::default()
            };
            let model = CandleEmbedModel::from_options(opts).expect("should succeed");
            assert_eq!(model.model_id(), "BAAI/bge-small-en-v1.5");
        }

        #[test]
        fn from_options_rejects_empty_device() {
            let opts = CandleEmbedOptions {
                device: Some(String::new()),
                ..CandleEmbedOptions::default()
            };
            let result = CandleEmbedModel::from_options(opts);
            assert!(result.is_err());
        }

        #[test]
        fn from_options_rejects_empty_model_id() {
            let opts = CandleEmbedOptions {
                model_id: Some(String::new()),
                ..CandleEmbedOptions::default()
            };
            let result = CandleEmbedModel::from_options(opts);
            assert!(result.is_err());
        }

        #[test]
        fn from_options_rejects_empty_revision() {
            let opts = CandleEmbedOptions {
                revision: Some(String::new()),
                ..CandleEmbedOptions::default()
            };
            let result = CandleEmbedModel::from_options(opts);
            assert!(result.is_err());
        }

        #[test]
        fn from_options_accepts_valid_device() {
            let opts = CandleEmbedOptions {
                device: Some("cuda:0".into()),
                ..CandleEmbedOptions::default()
            };
            let model = CandleEmbedModel::from_options(opts).expect("should succeed");
            assert_eq!(model.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
        }

        #[test]
        fn from_options_accepts_valid_revision() {
            let opts = CandleEmbedOptions {
                revision: Some("main".into()),
                ..CandleEmbedOptions::default()
            };
            let model = CandleEmbedModel::from_options(opts).expect("should succeed");
            assert_eq!(model.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
        }

        #[tokio::test]
        async fn embed_returns_engine_not_available() {
            let opts = CandleEmbedOptions::default();
            let model = CandleEmbedModel::from_options(opts).expect("should succeed");
            let result = model.embed(&["hello".into()]).await;
            assert!(
                matches!(result, Err(CandleEmbedError::EngineNotAvailable)),
                "expected EngineNotAvailable, got {result:?}"
            );
        }

        #[test]
        fn stub_dimensions_returns_zero() {
            let opts = CandleEmbedOptions::default();
            let model = CandleEmbedModel::from_options(opts).expect("should succeed");
            assert_eq!(model.dimensions(), 0);
        }
    }

    // Engine-only tests (require model download, marked #[ignore]).
    #[cfg(feature = "engine")]
    mod engine_tests {
        use super::*;

        #[tokio::test]
        #[ignore = "requires model download from HuggingFace"]
        async fn from_options_loads_model() {
            let model = CandleEmbedModel::from_options(CandleEmbedOptions::default())
                .await
                .expect("should create model with default options");
            assert!(model.dimensions() > 0);
            assert!(!model.model_id().is_empty());
        }

        #[tokio::test]
        #[ignore = "requires model download from HuggingFace"]
        async fn embed_returns_correct_count() {
            let model = CandleEmbedModel::from_options(CandleEmbedOptions::default())
                .await
                .expect("should create model with default options");
            let response = model
                .embed(&["hello world".into(), "goodbye world".into()])
                .await
                .expect("embedding should succeed");
            assert_eq!(response.embeddings.len(), 2);
            assert!(!response.embeddings[0].is_empty());
            assert_eq!(response.embeddings[0].len(), model.dimensions());
        }

        #[tokio::test]
        #[ignore = "requires model download from HuggingFace"]
        async fn embed_empty_input_returns_empty() {
            let model = CandleEmbedModel::from_options(CandleEmbedOptions::default())
                .await
                .expect("should create model with default options");
            let response = model.embed(&[]).await.expect("empty embed should succeed");
            assert!(response.embeddings.is_empty());
        }

        #[tokio::test]
        #[ignore = "requires model download from HuggingFace"]
        async fn embeddings_are_l2_normalized() {
            let model = CandleEmbedModel::from_options(CandleEmbedOptions::default())
                .await
                .expect("should create model");
            let response = model
                .embed(&["test sentence".into()])
                .await
                .expect("embed should succeed");
            let vec = &response.embeddings[0];
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "embedding should be L2-normalized, got norm={norm}"
            );
        }
    }
}
