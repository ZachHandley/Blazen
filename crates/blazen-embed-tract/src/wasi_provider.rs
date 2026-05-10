//! Wasi-side counterpart to [`crate::provider::TractEmbedModel`].
//!
//! Mirrors the public shape of [`crate::wasm_provider::WasmTractEmbedModel`]
//! but pulls model weights and the tokenizer over `reqwest::get` instead of
//! `web_sys::fetch`. This is the path used by the napi-rs wasi build for
//! Cloudflare Workers / Deno, where `@napi-rs/wasm-runtime` polyfills
//! networking via the host runtime so plain `reqwest` works.

use std::cell::RefCell;
use std::rc::Rc;

use tract_onnx::prelude::*;

use crate::options::{Pooling, TractOptions, lookup};

/// Error type for the wasi tract embedding provider. Distinct from the native
/// [`TractError`](crate::provider::TractError) and the
/// [`WasmTractError`](crate::wasm_provider::WasmTractError) because the failure
/// modes differ: there is no `tokio::spawn_blocking` join (single-threaded
/// runtime), and the download path is `reqwest::get` rather than `hf-hub` or
/// `web_sys::fetch`.
#[derive(Debug, thiserror::Error)]
pub enum WasiTractError {
    /// The caller requested a model name that is not in our registry.
    #[error("unknown tract embed model: {0}")]
    UnknownModel(String),

    /// `reqwest::get` failed, the response status was non-2xx, or the body
    /// could not be read into a byte buffer.
    #[error("fetch failed for {url}: {message}")]
    Fetch {
        /// URL that failed to fetch.
        url: String,
        /// Human-readable reason from the underlying HTTP client.
        message: String,
    },

    /// Loading the tokenizer or parsing the ONNX graph failed during
    /// [`WasiTractEmbedModel::create`].
    #[error("tract model init failed: {0}")]
    Init(String),

    /// Running the ONNX graph or pooling the output failed during an
    /// [`WasiTractEmbedModel::embed`] call.
    #[error("tract embed failed: {0}")]
    Embed(String),
}

/// Response from a wasi tract embedding operation. Same shape as
/// [`TractResponse`](crate::provider::TractResponse) on the native side.
#[derive(Debug, Clone)]
pub struct WasiTractResponse {
    /// The embedding vectors — one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// The model identifier that produced these embeddings (typically the
    /// Hugging Face repo id, but for the wasi path it is whatever URL the
    /// caller passed for the ONNX file).
    pub model: String,
}

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A local embedding model backed by [`tract_onnx`] for the napi-rs wasi
/// runtime (Cloudflare Workers / Deno).
///
/// Inference runs synchronously on the wasi instance; there is no equivalent
/// of `tokio::task::spawn_blocking` here because the napi-rs wasi runtime is
/// single-threaded from Rust's perspective. Callers should be mindful that
/// `embed()` will block the host's event loop for the duration of the forward
/// pass.
pub struct WasiTractEmbedModel {
    // `Rc<RefCell<...>>` rather than `Arc<Mutex<...>>` because the napi-rs
    // wasi runtime is single-threaded — we do not need cross-thread
    // synchronization, but we do need interior mutability since `tract`'s
    // `run()` takes `&mut self` through the plan in some configurations.
    model: Rc<RefCell<TractModel>>,
    tokenizer: Rc<tokenizers::Tokenizer>,
    model_id: String,
    dims: usize,
    pooling: Pooling,
    max_length: usize,
    batch_size: Option<usize>,
    input_count: usize,
}

// SAFETY: the napi-rs wasi runtime is single-threaded from Rust's perspective.
// `Send` and `Sync` are vacuously satisfied because there is no second thread
// to race with. We need these impls so the model can be stored behind
// `Arc<dyn EmbeddingModel>` in downstream code that is generic over native +
// wasm targets.
#[allow(unsafe_code)]
unsafe impl Send for WasiTractEmbedModel {}
#[allow(unsafe_code)]
unsafe impl Sync for WasiTractEmbedModel {}

impl std::fmt::Debug for WasiTractEmbedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasiTractEmbedModel")
            .field("model_id", &self.model_id)
            .field("dims", &self.dims)
            .field("pooling", &self.pooling)
            .field("max_length", &self.max_length)
            .field("batch_size", &self.batch_size)
            .field("input_count", &self.input_count)
            .finish_non_exhaustive()
    }
}

impl WasiTractEmbedModel {
    /// Build a [`WasiTractEmbedModel`] by fetching ONNX weights and a
    /// tokenizer from the given URLs.
    ///
    /// The URLs must point to a raw ONNX protobuf and a HuggingFace-format
    /// `tokenizer.json` respectively. Typical setups serve the files from
    /// `huggingface.co/<repo>/resolve/main/...` or a CDN.
    ///
    /// `options.model_name` is used purely for resolving registry metadata
    /// (dims, pooling strategy, max sequence length). When `model_name` is
    /// `Some`, the registry entry's `dim` and `pooling` are used. When it is
    /// `None`, defaults are derived from the registry default
    /// (`BGESmallENV15`).
    ///
    /// `options.cache_dir` is ignored on wasi — there is no persistent
    /// filesystem in Workers / Deno wasi sandboxes.
    /// `options.show_download_progress` is also ignored.
    ///
    /// # Errors
    ///
    /// Returns [`WasiTractError::UnknownModel`] if `options.model_name` does
    /// not match any registry entry, [`WasiTractError::Fetch`] if either URL
    /// fails to download, or [`WasiTractError::Init`] for tokenizer parse or
    /// ONNX parse failures.
    pub async fn create(
        model_url: &str,
        tokenizer_url: &str,
        options: TractOptions,
    ) -> Result<Self, WasiTractError> {
        let info = lookup(options.model_name.as_deref()).ok_or_else(|| {
            WasiTractError::UnknownModel(
                options
                    .model_name
                    .clone()
                    .unwrap_or_else(|| "<none>".to_string()),
            )
        })?;

        let model_bytes = fetch_bytes(model_url).await?;
        let tokenizer_bytes = fetch_bytes(tokenizer_url).await?;

        let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| WasiTractError::Init(format!("tokenizer parse failed: {e}")))?;

        // `model_for_read` accepts any `impl Read`; a `Cursor` over the
        // fetched byte buffer satisfies that without touching the filesystem.
        let mut cursor = std::io::Cursor::new(model_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| WasiTractError::Init(format!("onnx parse failed: {e}")))?
            .into_optimized()
            .map_err(|e| WasiTractError::Init(format!("onnx optimize failed: {e}")))?
            .into_runnable()
            .map_err(|e| WasiTractError::Init(format!("onnx runnable failed: {e}")))?;

        let input_count = model.model().inputs.len();

        Ok(Self {
            model: Rc::new(RefCell::new(model)),
            tokenizer: Rc::new(tokenizer),
            model_id: info.model_code.to_string(),
            dims: info.dim,
            pooling: info.pooling,
            max_length: 512,
            batch_size: options.max_batch_size,
            input_count,
        })
    }

    /// The Hugging Face model id this instance was loaded from
    /// (e.g. `"Xenova/bge-small-en-v1.5"`). Resolved from the registry entry
    /// matching `options.model_name`.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Output embedding dimensionality (the size of each row in the
    /// [`WasiTractResponse::embeddings`] vector).
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dims
    }

    /// Embed one or more texts, returning one L2-normalized vector per input.
    ///
    /// Inference runs synchronously on the wasi instance; the `async`
    /// signature is for API parity with the native [`TractEmbedModel`] and to
    /// leave room for future yields between batches.
    ///
    /// # Errors
    ///
    /// Returns [`WasiTractError::Embed`] for tokenization or inference
    /// failures.
    // Kept `async` to mirror the native `TractEmbedModel::embed` signature so
    // both providers fit the same `EmbeddingModel` trait without a separate
    // wasi shim. The body itself is currently fully synchronous.
    #[allow(clippy::unused_async)]
    pub async fn embed(&self, texts: &[String]) -> Result<WasiTractResponse, WasiTractError> {
        if texts.is_empty() {
            return Ok(WasiTractResponse {
                embeddings: Vec::new(),
                model: self.model_id.clone(),
            });
        }

        let embeddings = embed_sync(
            &self.model,
            &self.tokenizer,
            texts,
            self.dims,
            self.pooling,
            self.max_length,
            self.batch_size,
            self.input_count,
        )?;

        Ok(WasiTractResponse {
            embeddings,
            model: self.model_id.clone(),
        })
    }
}

/// Fetch a URL and return its body as a byte vector.
///
/// Uses `reqwest::get` — `@napi-rs/wasm-runtime` polyfills the underlying
/// network primitives via the host runtime (Node / Workers / Deno), so the
/// same `reqwest` build that works on native targets works here too.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, WasiTractError> {
    let resp = reqwest::get(url).await.map_err(|e| WasiTractError::Fetch {
        url: url.to_string(),
        message: e.to_string(),
    })?;

    if !resp.status().is_success() {
        return Err(WasiTractError::Fetch {
            url: url.to_string(),
            message: format!("HTTP {}", resp.status()),
        });
    }

    let bytes = resp.bytes().await.map_err(|e| WasiTractError::Fetch {
        url: url.to_string(),
        message: e.to_string(),
    })?;

    Ok(bytes.to_vec())
}

/// Synchronous embedding pipeline shared by [`WasiTractEmbedModel::embed`].
///
/// Mirrors the body of `embed_blocking` in the native provider — same
/// tokenization, same batching, same pooling, same L2 normalization. The only
/// difference is that the model is held behind `Rc<RefCell<...>>` instead of
/// `Arc<Mutex<...>>` because the wasi runtime is single-threaded.
#[allow(clippy::too_many_arguments)]
fn embed_sync(
    model_handle: &RefCell<TractModel>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    dims: usize,
    pooling: Pooling,
    max_length: usize,
    batch_size: Option<usize>,
    input_count: usize,
) -> Result<Vec<Vec<f32>>, WasiTractError> {
    let chunk_size = batch_size.unwrap_or(texts.len()).max(1);
    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(chunk_size) {
        let chunk_vec: Vec<String> = chunk.to_vec();

        let encodings = tokenizer
            .encode_batch(chunk_vec, true)
            .map_err(|e| WasiTractError::Embed(format!("tokenize failed: {e}")))?;

        let batch = encodings.len();

        let seq_len = encodings
            .iter()
            .map(tokenizers::Encoding::len)
            .max()
            .unwrap_or(0)
            .min(max_length);

        if seq_len == 0 {
            for _ in 0..batch {
                all_embeddings.push(vec![0.0; dims]);
            }
            continue;
        }

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

        let ids_arr = tract_ndarray::Array2::from_shape_vec((batch, seq_len), input_ids)
            .map_err(|e| WasiTractError::Embed(format!("ids reshape failed: {e}")))?;
        let mask_arr = tract_ndarray::Array2::from_shape_vec((batch, seq_len), attention_mask)
            .map_err(|e| WasiTractError::Embed(format!("mask reshape failed: {e}")))?;
        let types_arr = tract_ndarray::Array2::from_shape_vec((batch, seq_len), token_type_ids)
            .map_err(|e| WasiTractError::Embed(format!("types reshape failed: {e}")))?;

        let ids_tensor: Tensor = ids_arr.clone().into();
        let mask_tensor: Tensor = mask_arr.clone().into();
        let types_tensor: Tensor = types_arr.into();

        let inputs: TVec<TValue> = if input_count >= 3 {
            tvec!(ids_tensor.into(), mask_tensor.into(), types_tensor.into())
        } else {
            tvec!(ids_tensor.into(), mask_tensor.into())
        };

        let outputs = {
            let locked = model_handle.borrow();
            locked
                .run(inputs)
                .map_err(|e| WasiTractError::Embed(format!("tract run failed: {e}")))?
        };

        let hidden = outputs
            .first()
            .ok_or_else(|| WasiTractError::Embed("no outputs from tract graph".to_string()))?;

        let view = hidden
            .to_array_view::<f32>()
            .map_err(|e| WasiTractError::Embed(format!("output view failed: {e}")))?;

        let pooled: Vec<Vec<f32>> = match view.ndim() {
            3 => {
                let array = view
                    .view()
                    .into_dimensionality::<tract_ndarray::Ix3>()
                    .map_err(|e| {
                        WasiTractError::Embed(format!("output ndim coerce failed: {e}"))
                    })?;
                pool_hidden_states(array, &mask_arr, pooling, dims)?
            }
            2 => {
                let array = view
                    .view()
                    .into_dimensionality::<tract_ndarray::Ix2>()
                    .map_err(|e| {
                        WasiTractError::Embed(format!("output ndim coerce failed: {e}"))
                    })?;
                array
                    .outer_iter()
                    .map(|row| row.iter().copied().collect::<Vec<f32>>())
                    .collect()
            }
            other => {
                return Err(WasiTractError::Embed(format!(
                    "unexpected output rank {other}, expected 2 or 3"
                )));
            }
        };

        for mut row in pooled {
            l2_normalize(&mut row);
            all_embeddings.push(row);
        }
    }

    Ok(all_embeddings)
}

fn pool_hidden_states(
    hidden: tract_ndarray::ArrayView3<f32>,
    mask: &tract_ndarray::Array2<i64>,
    pooling: Pooling,
    dims: usize,
) -> Result<Vec<Vec<f32>>, WasiTractError> {
    let (batch, seq_len, hidden_dim) = hidden.dim();
    if hidden_dim != dims {
        return Err(WasiTractError::Embed(format!(
            "model output hidden size {hidden_dim} != expected dim {dims}"
        )));
    }

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);

    match pooling {
        Pooling::Cls => {
            for b in 0..batch {
                let slice = hidden.slice(tract_ndarray::s![b, 0_usize, ..]);
                out.push(slice.iter().copied().collect());
            }
        }
        Pooling::Mean => {
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

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}
