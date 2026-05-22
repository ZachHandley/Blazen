//! Bring-Your-Own (BYO) backend hooks for the WASM SDK.
//!
//! The WASM SDK only ships the candle backend out of the box (llama.cpp and
//! mistral.rs are too large to deliver over the wire in a browser tab). For
//! everything else — WebGPU-accelerated inference via
//! [`@mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm),
//! [`@huggingface/transformers`](https://github.com/huggingface/transformers.js),
//! ONNX Runtime Web, a remote vLLM proxy, etc. — callers supply a JS object
//! that implements the [`BlazenJsBackend`] interface and register it with
//! [`crate::manager::WasmModelManager`] via
//! [`crate::manager::WasmModelManager::register_byo_backend`].
//!
//! # JS / TypeScript contract
//!
//! The backend is a plain JS object satisfying this TypeScript surface:
//!
//! ```ts
//! interface BlazenJsBackend {
//!   /** Model identifier used for logging, routing, and pricing lookup. */
//!   modelId: string;
//!
//!   /** Approximate memory footprint of the loaded model in bytes. */
//!   memoryBytes?: number | (() => number | Promise<number>);
//!
//!   /** Optional embedding dimensionality (required if `embed` is provided). */
//!   embeddingDimensions?: number;
//!
//!   /** Optional device hint: `"cpu"`, `"gpu"`, `"gpu:0"`, etc. */
//!   device?: string | (() => string);
//!
//!   /** Lazy load the model into memory. Idempotent. */
//!   load?: () => Promise<void> | void;
//!
//!   /** Unload the model and free its memory. Idempotent. */
//!   unload?: () => Promise<void> | void;
//!
//!   /** Whether the model is currently loaded. */
//!   isLoaded?: () => boolean | Promise<boolean>;
//!
//!   /** Required: non-streaming completion. */
//!   complete(request: ModelRequest): Promise<ModelResponse>;
//!
//!   /**
//!    * Optional: streaming completion. If absent, `stream()` falls back to
//!    * `complete()` and emits a single chunk. The `onChunk` callback expects
//!    * `StreamChunk`-shaped objects.
//!    */
//!   streamComplete?(
//!     request: ModelRequest,
//!     onChunk: (chunk: StreamChunk) => void,
//!   ): Promise<void>;
//!
//!   /**
//!    * Optional: produce vector embeddings. Returns one Float32Array per
//!    * input text. If absent, the backend does NOT implement
//!    * `EmbeddingModel` and embedding calls reject with `unsupported`.
//!    */
//!   embed?(texts: string[]): Promise<Float32Array[] | number[][]>;
//!
//!   /**
//!    * Optional: load a PEFT/LoRA adapter. Receives the adapter bytes and a
//!    * caller-supplied options object. Resolves to the assigned adapter id.
//!    */
//!   loadAdapter?(
//!     adapterBytes: Uint8Array,
//!     options: { adapterId: string; scale?: number },
//!   ): Promise<string>;
//!
//!   /** Optional: unmount a previously-loaded adapter. */
//!   unloadAdapter?(adapterId: string): Promise<void>;
//! }
//! ```
//!
//! # Dispatch flow
//!
//! [`JsBackendShim`] holds a clone of the JS object plus extracted method
//! handles. Inference methods on [`Model`] / [`EmbeddingModel`]
//! serialize the Rust request via [`serde_wasm_bindgen`], invoke the JS
//! method through [`js_sys::Function::call*`], await the returned promise
//! through [`wasm_bindgen_futures::JsFuture`], and deserialize the result.
//!
//! Streaming forwards chunks through a Rust-side `Closure` that pushes into
//! a shared `Vec<StreamChunk>` (same pattern as
//! [`crate::js_model::JsModelHandler`]).
//!
//! # Safety
//!
//! All `JsValue` / `Function` handles are wrapped in newtypes that
//! `unsafe impl Send + Sync`. WASM is single-threaded so the marker traits
//! are vacuously satisfied.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use js_sys::{Array, Float32Array, Function, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use blazen_llm::traits::{
    AdapterHandle, AdapterMountStrategy, AdapterOptions, AdapterStatus, Model,
    EmbeddingModel, LocalModel,
};
use blazen_llm::types::{ModelRequest, ModelResponse, EmbeddingResponse, StreamChunk};
use blazen_llm::{BlazenError, Device};

// ---------------------------------------------------------------------------
// SendFuture — same pattern as agent.rs / js_completion.rs / js_embedding.rs.
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// Newtype wrappers asserting Send + Sync.
// ---------------------------------------------------------------------------

/// `Send + Sync` wrapper around a JS [`Function`].
struct JsFn(Function);

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsFn {}
unsafe impl Sync for JsFn {}

/// `Send + Sync` wrapper around an arbitrary [`JsValue`] (the JS host object).
struct JsHost(JsValue);

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsHost {}
unsafe impl Sync for JsHost {}

// ---------------------------------------------------------------------------
// Optional-fn extraction helpers.
// ---------------------------------------------------------------------------

/// Read an optional function-valued key off a JS object.
///
/// - Missing / `undefined` / `null` -> `Ok(None)`.
/// - Present and a function -> `Ok(Some(fn))`.
/// - Present but not a function -> `Err(...)`.
fn opt_fn(obj: &JsValue, key: &str) -> Result<Option<Function>, JsValue> {
    let val = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|e| JsValue::from_str(&format!("backend.{key} lookup failed: {e:?}")))?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let func: Function = val
        .dyn_into()
        .map_err(|_| JsValue::from_str(&format!("backend.{key} must be a function")))?;
    Ok(Some(func))
}

/// Read an optionally-callable field. Returns `Some(fn)` if the value is a
/// function, `None` if the field is absent OR present as a non-function
/// (e.g. a number / string / object). Useful for overloaded fields like
/// `memoryBytes` that may be either a static value or a callback.
fn opt_callable(obj: &JsValue, key: &str) -> Result<Option<Function>, JsValue> {
    let val = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|e| JsValue::from_str(&format!("backend.{key} lookup failed: {e:?}")))?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    Ok(val.dyn_into::<Function>().ok())
}

/// Read a required function-valued key off a JS object.
fn req_fn(obj: &JsValue, key: &str) -> Result<Function, JsValue> {
    let val = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|e| JsValue::from_str(&format!("backend.{key} lookup failed: {e:?}")))?;
    if val.is_undefined() || val.is_null() {
        return Err(JsValue::from_str(&format!(
            "backend.{key} is required but missing"
        )));
    }
    val.dyn_into()
        .map_err(|_| JsValue::from_str(&format!("backend.{key} must be a function")))
}

/// Read a string-typed field, returning `None` if absent.
fn opt_string(obj: &JsValue, key: &str) -> Option<String> {
    Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}

/// Read a number-typed field, returning `None` if absent.
fn opt_f64(obj: &JsValue, key: &str) -> Option<f64> {
    Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
}

/// Await `val` if it is a promise; otherwise return it as-is.
async fn await_if_promise(val: JsValue) -> Result<JsValue, JsValue> {
    if val.has_type::<Promise>() {
        let promise: Promise = val.unchecked_into();
        JsFuture::from(promise).await
    } else {
        Ok(val)
    }
}

// ---------------------------------------------------------------------------
// JsBackendShim — the unified BYO adapter.
// ---------------------------------------------------------------------------

/// Adapter that bridges a JS-side [`BlazenJsBackend`]-shaped object to the
/// Rust trait surface ([`LocalModel`], [`Model`], and optionally
/// [`EmbeddingModel`]).
///
/// Built by [`JsBackendShim::from_js`] which validates the contract once at
/// construction. Subsequent calls dispatch through cached
/// [`Function`] handles without re-walking the JS prototype chain.
pub struct JsBackendShim {
    model_id: String,
    embedding_dimensions: usize,
    memory_estimate_bytes: u64,
    device: Device,

    /// Original JS host object — kept alive so the captured methods don't
    /// dangle. Passed as `this` when invoking the methods.
    host: JsHost,

    // Lifecycle (all optional; defaults are no-ops / sensible defaults).
    load_fn: Option<JsFn>,
    unload_fn: Option<JsFn>,
    is_loaded_fn: Option<JsFn>,
    memory_bytes_fn: Option<JsFn>,

    // Completion surface.
    complete_fn: JsFn,
    stream_complete_fn: Option<JsFn>,

    // Embedding surface (optional).
    embed_fn: Option<JsFn>,

    // Adapter surface (optional).
    load_adapter_fn: Option<JsFn>,
    unload_adapter_fn: Option<JsFn>,
}

// SAFETY: WASM is single-threaded — all interior Js handles are vacuously
// Send + Sync.
unsafe impl Send for JsBackendShim {}
unsafe impl Sync for JsBackendShim {}

impl std::fmt::Debug for JsBackendShim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsBackendShim")
            .field("model_id", &self.model_id)
            .field("embedding_dimensions", &self.embedding_dimensions)
            .field("memory_estimate_bytes", &self.memory_estimate_bytes)
            .field("device", &self.device)
            .field("has_load", &self.load_fn.is_some())
            .field("has_unload", &self.unload_fn.is_some())
            .field("has_is_loaded", &self.is_loaded_fn.is_some())
            .field("has_memory_bytes", &self.memory_bytes_fn.is_some())
            .field("has_stream", &self.stream_complete_fn.is_some())
            .field("has_embed", &self.embed_fn.is_some())
            .field("has_load_adapter", &self.load_adapter_fn.is_some())
            .field("has_unload_adapter", &self.unload_adapter_fn.is_some())
            .finish_non_exhaustive()
    }
}

impl JsBackendShim {
    /// Validate the JS contract and capture the method handles.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JS object is not an object, the
    /// required `complete` method is missing or not a function, or any
    /// optional method key is present but not a function.
    pub fn from_js(host: JsValue) -> Result<Self, JsValue> {
        if !host.is_object() {
            return Err(JsValue::from_str("BYO backend must be a JS object"));
        }

        let model_id = opt_string(&host, "modelId").unwrap_or_else(|| "byo-backend".to_owned());

        // Embedding dimensions: optional, required only if embed() is provided.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let embedding_dimensions = opt_f64(&host, "embeddingDimensions")
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map_or(0usize, |n| n as usize);

        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let memory_estimate_bytes = opt_f64(&host, "memoryBytes")
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map_or(0u64, |n| n as u64);

        let device = opt_string(&host, "device")
            .as_deref()
            .and_then(|s| Device::parse(s).ok())
            .unwrap_or(Device::Cpu);

        let complete_fn = req_fn(&host, "complete")?;
        let stream_complete_fn = opt_fn(&host, "streamComplete")?;
        let embed_fn = opt_fn(&host, "embed")?;
        let load_fn = opt_fn(&host, "load")?;
        let unload_fn = opt_fn(&host, "unload")?;
        let is_loaded_fn = opt_fn(&host, "isLoaded")?;
        // `memoryBytes` is overloaded: it may be a number (static estimate),
        // a function (dynamic callback), or absent. Only treat it as a
        // function if it actually IS one — the numeric form is already
        // captured in `memory_estimate_bytes` above.
        let memory_bytes_fn = opt_callable(&host, "memoryBytes")?;
        let load_adapter_fn = opt_fn(&host, "loadAdapter")?;
        let unload_adapter_fn = opt_fn(&host, "unloadAdapter")?;

        if embed_fn.is_some() && embedding_dimensions == 0 {
            return Err(JsValue::from_str(
                "backend.embed is provided but backend.embeddingDimensions \
                 is missing or zero — required so EmbeddingModel can report \
                 its vector dimensionality",
            ));
        }

        Ok(Self {
            model_id,
            embedding_dimensions,
            memory_estimate_bytes,
            device,
            host: JsHost(host),
            load_fn: load_fn.map(JsFn),
            unload_fn: unload_fn.map(JsFn),
            is_loaded_fn: is_loaded_fn.map(JsFn),
            memory_bytes_fn: memory_bytes_fn.map(JsFn),
            complete_fn: JsFn(complete_fn),
            stream_complete_fn: stream_complete_fn.map(JsFn),
            embed_fn: embed_fn.map(JsFn),
            load_adapter_fn: load_adapter_fn.map(JsFn),
            unload_adapter_fn: unload_adapter_fn.map(JsFn),
        })
    }

    /// Whether this backend exposes an `embed` method.
    #[must_use]
    pub fn has_embed(&self) -> bool {
        self.embed_fn.is_some()
    }

    /// The embedding dimensionality, if any.
    #[must_use]
    pub fn embedding_dimensions(&self) -> usize {
        self.embedding_dimensions
    }

    /// The captured model id.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.model_id
    }

    /// Memory estimate (bytes) the manager should charge against the pool
    /// when this backend is loaded. Sourced from the JS object's
    /// `memoryBytes` field (numeric form) at construction time.
    #[must_use]
    pub fn memory_bytes_estimate(&self) -> u64 {
        self.memory_estimate_bytes
    }

    // -----------------------------------------------------------------------
    // Internal dispatch helpers (non-Send futures, wrapped by SendFuture at
    // the trait boundary).
    // -----------------------------------------------------------------------

    async fn complete_impl(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let js_req = serde_wasm_bindgen::to_value(&request)
            .map_err(|e| BlazenError::provider("byo_backend", e.to_string()))?;

        let result = self
            .complete_fn
            .0
            .call1(&self.host.0, &js_req)
            .map_err(|e| {
                BlazenError::provider("byo_backend", format!("complete() threw: {e:?}"))
            })?;

        let resolved = await_if_promise(result).await.map_err(|e| {
            BlazenError::provider("byo_backend", format!("complete() rejected: {e:?}"))
        })?;

        serde_wasm_bindgen::from_value::<ModelResponse>(resolved).map_err(|e| {
            BlazenError::invalid_response(format!(
                "BYO backend.complete() returned a value that did not deserialize \
                 to ModelResponse: {e}"
            ))
        })
    }

    async fn stream_with_handler_impl(
        &self,
        request: ModelRequest,
        handler: &Function,
    ) -> Result<Vec<StreamChunk>, BlazenError> {
        use std::cell::RefCell;
        use std::rc::Rc;

        let js_req = serde_wasm_bindgen::to_value(&request)
            .map_err(|e| BlazenError::provider("byo_backend", e.to_string()))?;

        let chunks: Rc<RefCell<Vec<StreamChunk>>> = Rc::new(RefCell::new(Vec::new()));
        let chunks_ref = Rc::clone(&chunks);

        let on_chunk = Closure::wrap(Box::new(move |js_chunk: JsValue| {
            if let Ok(chunk) = serde_wasm_bindgen::from_value::<StreamChunk>(js_chunk) {
                chunks_ref.borrow_mut().push(chunk);
            }
        }) as Box<dyn FnMut(JsValue)>);

        let result = handler
            .call2(&self.host.0, &js_req, on_chunk.as_ref().unchecked_ref())
            .map_err(|e| {
                BlazenError::provider("byo_backend", format!("streamComplete() threw: {e:?}"))
            })?;

        // Always await — handles both Promise and value returns.
        await_if_promise(result).await.map_err(|e| {
            BlazenError::stream_error(format!("BYO backend.streamComplete() rejected: {e:?}"))
        })?;

        drop(on_chunk);
        Ok(chunks.borrow().clone())
    }

    async fn embed_impl(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let Some(embed_fn) = self.embed_fn.as_ref() else {
            return Err(BlazenError::unsupported(
                "this BYO backend does not implement embed()",
            ));
        };

        // Build a JS Array of strings from `texts`.
        #[allow(clippy::cast_possible_truncation)]
        let arr = Array::new_with_length(texts.len() as u32);
        for (i, t) in texts.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            arr.set(i as u32, JsValue::from_str(t));
        }

        let result = embed_fn
            .0
            .call1(&self.host.0, &arr)
            .map_err(|e| BlazenError::provider("byo_backend", format!("embed() threw: {e:?}")))?;

        let resolved = await_if_promise(result).await.map_err(|e| {
            BlazenError::provider("byo_backend", format!("embed() rejected: {e:?}"))
        })?;

        let outer: Array = resolved.dyn_into().map_err(|_| {
            BlazenError::provider(
                "byo_backend",
                "embed() must return an Array of Float32Array or number[][]",
            )
        })?;

        let len = outer.length();
        let mut embeddings = Vec::with_capacity(len as usize);
        for i in 0..len {
            let item = outer.get(i);
            // Accept both Float32Array and plain number[] for ergonomics.
            if let Ok(typed) = item.clone().dyn_into::<Float32Array>() {
                embeddings.push(typed.to_vec());
            } else if let Ok(arr) = item.dyn_into::<Array>() {
                let mut row = Vec::with_capacity(arr.length() as usize);
                for j in 0..arr.length() {
                    #[allow(clippy::cast_possible_truncation)]
                    let v = arr.get(j).as_f64().ok_or_else(|| {
                        BlazenError::provider(
                            "byo_backend",
                            format!("embed()[{i}][{j}] is not a number"),
                        )
                    })? as f32;
                    row.push(v);
                }
                embeddings.push(row);
            } else {
                return Err(BlazenError::provider(
                    "byo_backend",
                    format!("embed()[{i}] is not a Float32Array or number[]"),
                ));
            }
        }

        Ok(EmbeddingResponse {
            embeddings,
            model: self.model_id.clone(),
            usage: None,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }

    async fn invoke_void(&self, func: &Function, label: &str) -> Result<(), BlazenError> {
        let result = func
            .call0(&self.host.0)
            .map_err(|e| BlazenError::provider("byo_backend", format!("{label}() threw: {e:?}")))?;
        await_if_promise(result).await.map_err(|e| {
            BlazenError::provider("byo_backend", format!("{label}() rejected: {e:?}"))
        })?;
        Ok(())
    }

    async fn load_impl(&self) -> Result<(), BlazenError> {
        let Some(f) = self.load_fn.as_ref() else {
            return Ok(());
        };
        self.invoke_void(&f.0, "load").await
    }

    async fn unload_impl(&self) -> Result<(), BlazenError> {
        let Some(f) = self.unload_fn.as_ref() else {
            return Ok(());
        };
        self.invoke_void(&f.0, "unload").await
    }

    async fn is_loaded_impl(&self) -> bool {
        let Some(f) = self.is_loaded_fn.as_ref() else {
            return false;
        };
        let Ok(raw) = f.0.call0(&self.host.0) else {
            return false;
        };
        match await_if_promise(raw).await {
            Ok(v) => v.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    }

    async fn memory_bytes_impl(&self) -> Option<u64> {
        if let Some(f) = self.memory_bytes_fn.as_ref()
            && let Ok(raw) = f.0.call0(&self.host.0)
            && let Ok(v) = await_if_promise(raw).await
        {
            if v.is_null() || v.is_undefined() {
                return None;
            }
            if let Some(n) = v.as_f64()
                && n.is_finite()
                && n >= 0.0
            {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                return Some(n as u64);
            }
        }
        if self.memory_estimate_bytes == 0 {
            None
        } else {
            Some(self.memory_estimate_bytes)
        }
    }

    async fn load_adapter_impl(
        &self,
        adapter_dir: &std::path::Path,
        options: AdapterOptions,
    ) -> Result<AdapterHandle, BlazenError> {
        let Some(f) = self.load_adapter_fn.as_ref() else {
            return Err(BlazenError::unsupported(
                "this BYO backend does not implement loadAdapter()",
            ));
        };

        // In a browser sandbox there's no filesystem — most callers will pass
        // a path-like sentinel (or an empty path) and have already pre-loaded
        // the adapter bytes elsewhere. We still forward the path string as
        // the first argument for JS-side use, plus the options object.
        let path_str = adapter_dir.to_string_lossy().into_owned();
        let opts_obj = js_sys::Object::new();
        let _ = Reflect::set(
            &opts_obj,
            &JsValue::from_str("adapterId"),
            &JsValue::from_str(&options.adapter_id),
        );
        #[allow(clippy::cast_lossless)]
        let _ = Reflect::set(
            &opts_obj,
            &JsValue::from_str("scale"),
            &JsValue::from_f64(f64::from(options.scale)),
        );
        let _ = Reflect::set(
            &opts_obj,
            &JsValue::from_str("sourceDir"),
            &JsValue::from_str(&path_str),
        );

        let result =
            f.0.call2(
                &self.host.0,
                &Uint8Array::new_with_length(0).into(),
                &opts_obj.into(),
            )
            .map_err(|e| {
                BlazenError::provider("byo_backend", format!("loadAdapter() threw: {e:?}"))
            })?;
        let resolved = await_if_promise(result).await.map_err(|e| {
            BlazenError::provider("byo_backend", format!("loadAdapter() rejected: {e:?}"))
        })?;

        let assigned_id = resolved
            .as_string()
            .unwrap_or_else(|| options.adapter_id.clone());
        // adapter_dir is part of the public trait signature but isn't echoed
        // back through AdapterHandle (which only carries memory_bytes +
        // strategy); we accept it for parity with the native LocalModel API.
        let _ = adapter_dir;

        Ok(AdapterHandle {
            adapter_id: assigned_id,
            memory_bytes: 0,
            mount_strategy: AdapterMountStrategy::Attached,
        })
    }

    async fn unload_adapter_impl(&self, handle: &AdapterHandle) -> Result<(), BlazenError> {
        let Some(f) = self.unload_adapter_fn.as_ref() else {
            return Err(BlazenError::unsupported(
                "this BYO backend does not implement unloadAdapter()",
            ));
        };
        let result =
            f.0.call1(&self.host.0, &JsValue::from_str(&handle.adapter_id))
                .map_err(|e| {
                    BlazenError::provider("byo_backend", format!("unloadAdapter() threw: {e:?}"))
                })?;
        await_if_promise(result).await.map_err(|e| {
            BlazenError::provider("byo_backend", format!("unloadAdapter() rejected: {e:?}"))
        })?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Trait impls.
// ---------------------------------------------------------------------------

#[async_trait]
impl LocalModel for JsBackendShim {
    async fn load(&self) -> Result<(), BlazenError> {
        SendFuture(self.load_impl()).await
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        SendFuture(self.unload_impl()).await
    }

    async fn is_loaded(&self) -> bool {
        SendFuture(self.is_loaded_impl()).await
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    async fn memory_bytes(&self) -> Option<u64> {
        SendFuture(self.memory_bytes_impl()).await
    }

    async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        options: AdapterOptions,
    ) -> Result<AdapterHandle, BlazenError> {
        SendFuture(self.load_adapter_impl(adapter_dir, options)).await
    }

    async fn unload_adapter(&self, handle: &AdapterHandle) -> Result<(), BlazenError> {
        SendFuture(self.unload_adapter_impl(handle)).await
    }

    async fn list_adapters(&self) -> Vec<AdapterStatus> {
        Vec::new()
    }
}

#[async_trait]
impl Model for JsBackendShim {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        SendFuture(self.complete_impl(request)).await
    }

    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        if let Some(handler) = self.stream_complete_fn.as_ref() {
            let chunks = SendFuture(self.stream_with_handler_impl(request, &handler.0)).await?;
            Ok(Box::pin(futures_util::stream::iter(
                chunks.into_iter().map(Ok),
            )))
        } else {
            // Fallback: call complete() and yield one chunk with the full response.
            let response = self.complete(request).await?;
            let chunk = StreamChunk {
                delta: response.content,
                tool_calls: response.tool_calls,
                finish_reason: response.finish_reason,
                reasoning_delta: response.reasoning.map(|r| r.text),
                citations: response.citations,
                artifacts: response.artifacts,
            };
            Ok(Box::pin(futures_util::stream::once(async { Ok(chunk) })))
        }
    }
}

#[async_trait]
impl EmbeddingModel for JsBackendShim {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        self.embedding_dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        SendFuture(self.embed_impl(texts)).await
    }
}

// ---------------------------------------------------------------------------
// TypeScript surface-section declaration (re-exported in `.d.ts`).
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_BLAZEN_JS_BACKEND: &str = r"
/**
 * Bring-Your-Own (BYO) inference backend contract.
 *
 * Implement this interface in JavaScript / TypeScript to plug a custom
 * inference backend (WebGPU via @mlc-ai/web-llm, transformers.js, ONNX
 * Runtime Web, a remote vLLM/TGI proxy, etc.) into the WASM SDK without
 * bloating the WASM binary with a Rust-side engine. Register the
 * implementation with `ModelManager.registerByoBackend(id, backend)`.
 */
export interface BlazenJsBackend {
  modelId: string;
  memoryBytes?: number | (() => number | Promise<number>);
  embeddingDimensions?: number;
  device?: string | (() => string);

  load?: () => Promise<void> | void;
  unload?: () => Promise<void> | void;
  isLoaded?: () => boolean | Promise<boolean>;

  complete(request: ModelRequest): Promise<ModelResponse>;
  streamComplete?(
    request: ModelRequest,
    onChunk: (chunk: StreamChunk) => void,
  ): Promise<void>;
  embed?(texts: string[]): Promise<Float32Array[] | number[][]>;

  loadAdapter?(
    adapterBytes: Uint8Array,
    options: { adapterId: string; scale?: number; sourceDir?: string },
  ): Promise<string>;
  unloadAdapter?(adapterId: string): Promise<void>;
}
";

// ---------------------------------------------------------------------------
// Public WASM constructor — wrap a JS object into a Model handle.
// ---------------------------------------------------------------------------

/// A `wasm-bindgen`-visible factory that wraps a `BlazenJsBackend` object
/// and returns a [`crate::model::WasmModel`] backed by
/// the BYO shim. Useful when the caller wants to use BYO outside the
/// `ModelManager` (e.g. for ad-hoc one-shot completions).
///
/// # Errors
///
/// Returns the underlying [`JsBackendShim::from_js`] error if `backend` is
/// not a JS object, the required `complete` method is missing or not a
/// function, or any optional method key is present but not a function.
#[wasm_bindgen(js_name = "byoBackendAsModel")]
pub fn byo_backend_as_model(
    backend: JsValue,
) -> Result<crate::model::WasmModel, JsValue> {
    let shim: Arc<dyn Model> = Arc::new(JsBackendShim::from_js(backend)?);
    Ok(crate::model::WasmModel::from_arc(shim))
}

/// Construct a [`JsBackendShim`] from a JS object. Intended for use by
/// [`crate::manager::WasmModelManager::register_byo_backend`].
pub(crate) fn shim_from_js(backend: JsValue) -> Result<Arc<JsBackendShim>, JsValue> {
    Ok(Arc::new(JsBackendShim::from_js(backend)?))
}
