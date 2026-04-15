//! The [`TractEmbedModel`] type providing local embeddings via `tract-onnx`.
//!
//! This is the pure-Rust counterpart to `blazen-embed-fastembed`. It loads the
//! same model catalog via `tract-onnx` instead of `onnxruntime`, so it builds
//! on targets where ONNX Runtime's prebuilt binaries are unavailable (musl,
//! WASM, etc.). The public API mirrors `FastEmbedModel` so callers can swap
//! backends without touching their own code.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tract_onnx::prelude::*;

use crate::options::{ModelInfo, Pooling, TractOptions, lookup};

/// Error type for tract embedding operations. Variants mirror
/// `blazen_embed_fastembed::FastEmbedError` so consumers can swap backends
/// without touching error-handling code.
#[derive(Debug, thiserror::Error)]
pub enum TractError {
    /// The caller requested a model name that is not in our registry.
    #[error("unknown tract embed model: {0}")]
    UnknownModel(String),

    /// Loading the tokenizer, downloading weights, or parsing the ONNX graph
    /// failed during [`TractEmbedModel::from_options`].
    #[error("tract model init failed: {0}")]
    Init(String),

    /// Running the ONNX graph or pooling the output failed during an embed
    /// call.
    #[error("tract embed failed: {0}")]
    Embed(String),

    /// The internal mutex guarding the tract model handle was poisoned by a
    /// previous panic.
    #[error("mutex poisoned: {0}")]
    MutexPoisoned(String),

    /// The blocking task that ran the tract pipeline panicked.
    #[error("blocking task panicked: {0}")]
    TaskPanicked(String),
}

/// Response from a tract embedding operation. Same shape as
/// `blazen_embed_fastembed::FastEmbedResponse` for drop-in compatibility.
#[derive(Debug, Clone)]
pub struct TractResponse {
    /// The embedding vectors — one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// The model identifier that produced these embeddings (typically the
    /// Hugging Face repo id).
    pub model: String,
}

/// Type alias for the runnable tract graph we hold onto. Bare `SimplePlan`
/// satisfies every tract version we've tested — the concrete generics match
/// what `into_runnable()` returns after `into_optimized()`.
type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A local embedding model backed by [`tract_onnx`] (pure Rust ONNX inference).
///
/// Wraps a loaded ONNX graph plus its tokenizer. All inference is dispatched
/// onto [`tokio::task::spawn_blocking`] because tract runs synchronously and
/// is CPU-bound.
pub struct TractEmbedModel {
    /// The runnable tract plan. Wrapped in `Arc<Mutex<...>>` because `run()`
    /// takes `&self` but we need interior mutability across `spawn_blocking`
    /// boundaries; the underlying plan itself is already internally re-entrant
    /// but we serialize to keep scratch-buffer reuse predictable.
    model: Arc<Mutex<TractModel>>,
    /// The tokenizer. `tokenizers::Tokenizer` is `Send + Sync` so a bare `Arc`
    /// is sufficient — no mutex needed.
    tokenizer: Arc<tokenizers::Tokenizer>,
    /// Hugging Face repo id used to load the weights. Exposed via
    /// [`Self::model_id`].
    model_id: String,
    /// Output embedding dimensionality. Exposed via [`Self::dimensions`].
    dims: usize,
    /// Pooling strategy applied to the final hidden states tensor.
    pooling: Pooling,
    /// Maximum sequence length passed to the tokenizer. Tract graphs in
    /// fastembed's registry are all trained with the BERT-family default of
    /// 512; we hardcode that here to match.
    max_length: usize,
    /// How many input texts to batch into a single forward pass. `None` means
    /// "run the whole input vector in one pass".
    batch_size: Option<usize>,
    /// Number of model inputs. BERT-family graphs expect 3 (`input_ids`,
    /// `attention_mask`, `token_type_ids`); distilled or 2-input variants
    /// (common for sentence-transformers) expect 2. We inspect this at load
    /// time and feed the matching tensors at inference time.
    input_count: usize,
}

impl TractEmbedModel {
    /// Build a [`TractEmbedModel`] from the given options.
    ///
    /// This is a synchronous function even though model download is async:
    /// when called from inside a tokio runtime (the common case) we
    /// `block_on` the download using the current runtime's handle; when
    /// called outside any runtime we spin up a small current-thread runtime
    /// just for the downloads. This matches `FastEmbedModel::from_options`'s
    /// sync contract so the two backends are drop-in swappable.
    ///
    /// # Errors
    ///
    /// Returns [`TractError::UnknownModel`] if the name does not map to a
    /// registry entry, or [`TractError::Init`] for any failure during
    /// download, tokenizer load, or ONNX parse.
    pub fn from_options(opts: TractOptions) -> Result<Self, TractError> {
        let TractOptions {
            model_name,
            cache_dir,
            max_batch_size,
            show_download_progress: _,
        } = opts;

        let name = model_name.as_deref();
        let info = lookup(name)
            .ok_or_else(|| TractError::UnknownModel(name.unwrap_or("<none>").to_string()))?;

        // Build the model cache, honoring an optional override.
        let cache = if let Some(dir) = cache_dir {
            blazen_model_cache::ModelCache::with_dir(dir)
        } else {
            blazen_model_cache::ModelCache::new()
                .map_err(|e| TractError::Init(format!("cache init failed: {e}")))?
        };

        // Download all required files. Runs inside a current-thread runtime
        // if we're not already inside one.
        let (onnx_path, tokenizer_path) = block_on_downloads(&cache, info)?;

        // Load tokenizer from disk.
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TractError::Init(format!("tokenizer load failed: {e}")))?;

        // Load and optimize the ONNX graph.
        let model = tract_onnx::onnx()
            .model_for_path(&onnx_path)
            .map_err(|e| TractError::Init(format!("onnx parse failed: {e}")))?
            .into_optimized()
            .map_err(|e| TractError::Init(format!("onnx optimize failed: {e}")))?
            .into_runnable()
            .map_err(|e| TractError::Init(format!("onnx runnable failed: {e}")))?;

        // Count the model's inputs so `embed` knows whether to build
        // `token_type_ids` (3-input BERT) or not (2-input sentence-transformer).
        let input_count = model.model().inputs.len();

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            model_id: info.model_code.to_string(),
            dims: info.dim,
            pooling: info.pooling,
            max_length: 512,
            batch_size: max_batch_size,
            input_count,
        })
    }

    /// The Hugging Face model id this instance was loaded from
    /// (e.g. `"Xenova/bge-small-en-v1.5"`).
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Output embedding dimensionality (the size of each row in the
    /// [`TractResponse::embeddings`] vector).
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dims
    }

    /// Embed one or more texts, returning one L2-normalized vector per input.
    ///
    /// Dispatches onto [`tokio::task::spawn_blocking`] because tract is
    /// synchronous CPU-bound work and must not run on the async runtime.
    ///
    /// # Errors
    ///
    /// Returns [`TractError::Embed`] for tokenization or inference failures,
    /// [`TractError::MutexPoisoned`] if a prior panic poisoned the model lock,
    /// or [`TractError::TaskPanicked`] if the blocking task itself panics.
    pub async fn embed(&self, texts: &[String]) -> Result<TractResponse, TractError> {
        if texts.is_empty() {
            return Ok(TractResponse {
                embeddings: Vec::new(),
                model: self.model_id.clone(),
            });
        }

        let texts_owned: Vec<String> = texts.to_vec();
        let model_handle = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let dims = self.dims;
        let pooling = self.pooling;
        let max_length = self.max_length;
        let batch_size = self.batch_size;
        let input_count = self.input_count;
        let model_id = self.model_id.clone();

        let embeddings = tokio::task::spawn_blocking(move || {
            embed_blocking(
                &model_handle,
                &tokenizer,
                &texts_owned,
                dims,
                pooling,
                max_length,
                batch_size,
                input_count,
            )
        })
        .await
        .map_err(|e| TractError::TaskPanicked(e.to_string()))??;

        Ok(TractResponse {
            embeddings,
            model: model_id,
        })
    }
}

/// Run the async `ModelCache` downloads from a sync context, returning the
/// paths to the ONNX file and the tokenizer file. All `additional_files` are
/// fetched too (and kept co-located on disk next to the main model file so
/// tract can find them via relative path lookups), but their paths are not
/// returned because tract doesn't need to know them explicitly.
fn block_on_downloads(
    cache: &blazen_model_cache::ModelCache,
    info: &ModelInfo,
) -> Result<(PathBuf, PathBuf), TractError> {
    let do_downloads = async {
        let onnx = cache
            .download(info.model_code, info.model_file, None)
            .await
            .map_err(|e| {
                TractError::Init(format!("failed to download {}: {}", info.model_file, e))
            })?;

        let tokenizer = cache
            .download(info.model_code, "tokenizer.json", None)
            .await
            .map_err(|e| TractError::Init(format!("failed to download tokenizer.json: {e}")))?;

        for extra in info.additional_files {
            cache
                .download(info.model_code, extra, None)
                .await
                .map_err(|e| TractError::Init(format!("failed to download {extra}: {e}")))?;
        }

        Ok::<_, TractError>((onnx, tokenizer))
    };

    // Prefer the current tokio runtime so we don't double-schedule inside an
    // already-running reactor. `block_in_place` lets us block the current
    // worker thread without starving the runtime (it moves other tasks off
    // this thread). Outside any runtime, build a single-thread runtime ad hoc.
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(do_downloads))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| TractError::Init(format!("runtime build failed: {e}")))?;
        rt.block_on(do_downloads)
    }
}

/// The synchronous hot path. Invoked from [`tokio::task::spawn_blocking`] —
/// must not touch the async runtime.
#[allow(clippy::too_many_arguments)] // internal helper, arity is load-bearing
fn embed_blocking(
    model_handle: &Mutex<TractModel>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    dims: usize,
    pooling: Pooling,
    max_length: usize,
    batch_size: Option<usize>,
    input_count: usize,
) -> Result<Vec<Vec<f32>>, TractError> {
    // Chunk inputs by batch_size so memory stays bounded on large calls.
    let chunk_size = batch_size.unwrap_or(texts.len()).max(1);
    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(chunk_size) {
        let chunk_vec: Vec<String> = chunk.to_vec();

        // Tokenize the chunk. `encode_batch` returns one Encoding per input.
        let encodings = tokenizer
            .encode_batch(chunk_vec, true)
            .map_err(|e| TractError::Embed(format!("tokenize failed: {e}")))?;

        let batch = encodings.len();

        // Determine padded sequence length for this chunk: min of longest
        // encoding and the model's max_length.
        let seq_len = encodings
            .iter()
            .map(tokenizers::Encoding::len)
            .max()
            .unwrap_or(0)
            .min(max_length);

        if seq_len == 0 {
            // All inputs were empty post-tokenization — emit zero vectors so
            // the caller still gets one output per input.
            for _ in 0..batch {
                all_embeddings.push(vec![0.0; dims]);
            }
            continue;
        }

        // Build flat [batch * seq_len] i64 buffers for input_ids,
        // attention_mask, and (if the model expects it) token_type_ids.
        let mut input_ids = vec![0_i64; batch * seq_len];
        let mut attention_mask = vec![0_i64; batch * seq_len];
        let mut token_type_ids = vec![0_i64; batch * seq_len];

        for (row, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let take = ids.len().min(seq_len);
            let base = row * seq_len;
            for i in 0..take {
                input_ids[base + i] = i64::from(ids[i]);
                attention_mask[base + i] = i64::from(mask[i]);
                token_type_ids[base + i] = i64::from(types[i]);
            }
        }

        // Wrap into tract tensors via tract_ndarray (tract re-exports
        // ndarray 0.16 as `tract_ndarray` in its prelude).
        let ids_arr = tract_ndarray::Array2::from_shape_vec((batch, seq_len), input_ids)
            .map_err(|e| TractError::Embed(format!("ids reshape failed: {e}")))?;
        let mask_arr = tract_ndarray::Array2::from_shape_vec((batch, seq_len), attention_mask)
            .map_err(|e| TractError::Embed(format!("mask reshape failed: {e}")))?;
        let types_arr = tract_ndarray::Array2::from_shape_vec((batch, seq_len), token_type_ids)
            .map_err(|e| TractError::Embed(format!("types reshape failed: {e}")))?;

        let ids_tensor: Tensor = ids_arr.clone().into();
        let mask_tensor: Tensor = mask_arr.clone().into();
        let types_tensor: Tensor = types_arr.into();

        // Assemble inputs matching the graph's arity. Convention for
        // BERT-family ONNX: [input_ids, attention_mask, token_type_ids].
        let inputs: TVec<TValue> = if input_count >= 3 {
            tvec!(ids_tensor.into(), mask_tensor.into(), types_tensor.into())
        } else {
            tvec!(ids_tensor.into(), mask_tensor.into())
        };

        // Run the graph under the mutex.
        let outputs = {
            let locked = model_handle
                .lock()
                .map_err(|e| TractError::MutexPoisoned(e.to_string()))?;
            locked
                .run(inputs)
                .map_err(|e| TractError::Embed(format!("tract run failed: {e}")))?
        };

        let hidden = outputs
            .first()
            .ok_or_else(|| TractError::Embed("no outputs from tract graph".to_string()))?;

        // Expect shape [batch, seq_len, hidden]. Some graphs emit the pooled
        // vector directly as [batch, hidden] — handle that too.
        let view = hidden
            .to_array_view::<f32>()
            .map_err(|e| TractError::Embed(format!("output view failed: {e}")))?;

        let pooled: Vec<Vec<f32>> = match view.ndim() {
            3 => {
                // [batch, seq_len, hidden] — apply the requested pooling.
                let array = view
                    .view()
                    .into_dimensionality::<tract_ndarray::Ix3>()
                    .map_err(|e| TractError::Embed(format!("output ndim coerce failed: {e}")))?;
                pool_hidden_states(array, &mask_arr, pooling, dims)?
            }
            2 => {
                // [batch, hidden] — already pooled, pass through.
                let array = view
                    .view()
                    .into_dimensionality::<tract_ndarray::Ix2>()
                    .map_err(|e| TractError::Embed(format!("output ndim coerce failed: {e}")))?;
                array
                    .outer_iter()
                    .map(|row| row.iter().copied().collect::<Vec<f32>>())
                    .collect()
            }
            other => {
                return Err(TractError::Embed(format!(
                    "unexpected output rank {other}, expected 2 or 3"
                )));
            }
        };

        // L2-normalize each row so cosine similarity == dot product downstream.
        for mut row in pooled {
            l2_normalize(&mut row);
            all_embeddings.push(row);
        }
    }

    Ok(all_embeddings)
}

/// Collapse `[batch, seq_len, hidden]` into `[batch, hidden]` using the given
/// pooling strategy and the attention mask.
fn pool_hidden_states(
    hidden: tract_ndarray::ArrayView3<f32>,
    mask: &tract_ndarray::Array2<i64>,
    pooling: Pooling,
    dims: usize,
) -> Result<Vec<Vec<f32>>, TractError> {
    let (batch, seq_len, hidden_dim) = hidden.dim();
    if hidden_dim != dims {
        return Err(TractError::Embed(format!(
            "model output hidden size {hidden_dim} != expected dim {dims}"
        )));
    }

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);

    match pooling {
        Pooling::Cls => {
            // First token per row.
            for b in 0..batch {
                let slice = hidden.slice(tract_ndarray::s![b, 0_usize, ..]);
                out.push(slice.iter().copied().collect());
            }
        }
        Pooling::Mean => {
            // Sum hidden[b, t, :] * mask[b, t] over t, divide by sum(mask[b, :]).
            for b in 0..batch {
                let mut acc = vec![0.0_f32; hidden_dim];
                let mut weight_sum: f32 = 0.0;
                for t in 0..seq_len {
                    #[allow(clippy::cast_precision_loss)]
                    let w = mask[[b, t]] as f32;
                    if w == 0.0 {
                        continue;
                    }
                    weight_sum += w;
                    for h in 0..hidden_dim {
                        acc[h] += hidden[[b, t, h]] * w;
                    }
                }
                let denom = weight_sum.max(1e-12);
                for v in &mut acc {
                    *v /= denom;
                }
                out.push(acc);
            }
        }
    }

    Ok(out)
}

/// L2-normalize `v` in place. Uses a small epsilon to avoid dividing by zero
/// on all-zero rows (which would otherwise produce NaNs).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_empty_input_returns_empty() {
        // We can construct a fake model without downloading by hand-building
        // the struct — but that's fragile. Instead, skip if the default model
        // isn't already cached locally; otherwise, exercise the empty path.
        let model = match TractEmbedModel::from_options(TractOptions::default()) {
            Ok(m) => m,
            Err(_) => {
                eprintln!("skipping embed_empty_input_returns_empty: model not available");
                return;
            }
        };
        let response = model.embed(&[]).await.expect("empty embed should succeed");
        assert!(response.embeddings.is_empty());
        assert_eq!(response.model, model.model_id());
    }

    #[test]
    fn unknown_model_name_is_rejected() {
        let opts = TractOptions {
            model_name: Some("NotARealModel".to_string()),
            ..TractOptions::default()
        };
        let err = TractEmbedModel::from_options(opts).unwrap_err();
        assert!(matches!(err, TractError::UnknownModel(_)));
    }

    #[tokio::test]
    #[ignore = "requires model download from HuggingFace"]
    async fn embed_returns_correct_count_and_dims() {
        let model = TractEmbedModel::from_options(TractOptions::default())
            .expect("should create model with default options");
        let response = model
            .embed(&["hello".into(), "world".into()])
            .await
            .expect("embedding should succeed");
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0].len(), model.dimensions());
        // L2-normalized vectors have a norm of ~1.0.
        let norm: f32 = response.embeddings[0]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected ~1.0 norm, got {norm}");
    }
}
