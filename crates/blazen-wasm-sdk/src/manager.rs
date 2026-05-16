//! WASM bindings for `blazen_manager::ModelManager`.
//!
//! This wraps the upstream [`blazen_manager::ModelManager`] directly via a
//! JS-callback `LocalModel` adapter, so the WASM SDK shares the same
//! memory-budget tracking and LRU eviction logic used by the native engine.
//!
//! ```js
//! const manager = new ModelManager(8); // 8 GB CPU memory budget
//!
//! const model = CompletionModel.webLlm('Llama-3.1-8B-Instruct-q4f32_1-MLC');
//! let loaded = false;
//! await manager.register('llama-8b', model, 4_000_000_000, {
//!   load: async () => { loaded = true; /* load model */ },
//!   unload: async () => { loaded = false; /* unload model */ },
//!   isLoaded: () => loaded,                       // optional
//!   memoryBytes: async () => 4_000_000_000,       // optional
//!   device: () => 'cpu',                          // optional, defaults to 'cpu'
//! });
//!
//! // The `model` argument is also optional -- pass `null` if you don't have one:
//! await manager.register('m2', null, 1_000_000_000, {
//!   load: async () => {},
//!   unload: async () => {},
//! });
//!
//! await manager.load('llama-8b');
//! const status = await manager.status();
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};

use blazen_llm::{BlazenError, Device, LocalModel, Pool};
use blazen_manager::ModelManager;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as agent.rs / js_embedding.rs)
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
// JsClosure -- Send + Sync wrapper around a JS Function
// ---------------------------------------------------------------------------

/// Newtype wrapping a JS [`Function`] that asserts `Send + Sync`.
///
/// SAFETY: WASM is single-threaded; `Send`/`Sync` are vacuously satisfied.
struct JsClosure(Function);

unsafe impl Send for JsClosure {}
unsafe impl Sync for JsClosure {}

// ---------------------------------------------------------------------------
// Pool label parsing
// ---------------------------------------------------------------------------

/// Parse a pool label like `"cpu"`, `"gpu"`, or `"gpu:N"` into a [`Pool`].
fn parse_pool_label(label: &str) -> Result<Pool, JsValue> {
    match label {
        "cpu" => Ok(Pool::Cpu),
        "gpu" => Ok(Pool::Gpu(0)),
        s if s.starts_with("gpu:") => s
            .strip_prefix("gpu:")
            .and_then(|n| n.parse::<usize>().ok())
            .map(Pool::Gpu)
            .ok_or_else(|| {
                JsValue::from_str(&format!(
                    "invalid pool label '{label}': expected 'cpu', 'gpu', or 'gpu:N' \
                     where N is a non-negative integer"
                ))
            }),
        _ => Err(JsValue::from_str(&format!(
            "invalid pool label '{label}': expected 'cpu', 'gpu', or 'gpu:N' \
             where N is a non-negative integer"
        ))),
    }
}

// ---------------------------------------------------------------------------
// JsLocalModelAdapter -- bridges JS lifecycle callbacks to LocalModel
// ---------------------------------------------------------------------------

/// Adapter that implements [`LocalModel`] by invoking JS callbacks.
///
/// `load` and `unload` call the corresponding JS functions and await the
/// returned `Promise`. `is_loaded` and `memory_bytes` invoke their optional
/// JS callbacks if provided; otherwise they fall back to conservative
/// defaults (`false` and `Some(memory_estimate_bytes)` respectively).
struct JsLocalModelAdapter {
    id: String,
    load_fn: Arc<JsClosure>,
    unload_fn: Arc<JsClosure>,
    is_loaded_fn: Option<Arc<JsClosure>>,
    memory_bytes_fn: Option<Arc<JsClosure>>,
    memory_estimate_bytes: u64,
    device: Device,
}

impl JsLocalModelAdapter {
    /// Invoke a JS callback and await the resulting promise (if any).
    async fn invoke(func: &Function, label: &str, model_id: &str) -> Result<(), BlazenError> {
        let result = func.call0(&JsValue::NULL).map_err(|e| {
            BlazenError::provider(
                "wasm_model_manager",
                format!("model '{model_id}' lifecycle.{label}() threw: {e:?}"),
            )
        })?;

        if result.has_type::<Promise>() {
            let promise: Promise = result.unchecked_into();
            JsFuture::from(promise).await.map_err(|e| {
                BlazenError::provider(
                    "wasm_model_manager",
                    format!("model '{model_id}' lifecycle.{label}() rejected: {e:?}"),
                )
            })?;
        }

        Ok(())
    }

    /// Invoke a JS callback that returns a value (or a `Promise` resolving to a
    /// value) and return the resolved [`JsValue`].
    async fn invoke_with_result(
        func: &Function,
        label: &str,
        model_id: &str,
    ) -> Result<JsValue, BlazenError> {
        let result = func.call0(&JsValue::NULL).map_err(|e| {
            BlazenError::provider(
                "wasm_model_manager",
                format!("model '{model_id}' lifecycle.{label}() threw: {e:?}"),
            )
        })?;

        if result.has_type::<Promise>() {
            let promise: Promise = result.unchecked_into();
            let val = JsFuture::from(promise).await.map_err(|e| {
                BlazenError::provider(
                    "wasm_model_manager",
                    format!("model '{model_id}' lifecycle.{label}() rejected: {e:?}"),
                )
            })?;
            Ok(val)
        } else {
            Ok(result)
        }
    }

    async fn load_impl(&self) -> Result<(), BlazenError> {
        Self::invoke(&self.load_fn.0, "load", &self.id).await
    }

    async fn unload_impl(&self) -> Result<(), BlazenError> {
        Self::invoke(&self.unload_fn.0, "unload", &self.id).await
    }

    async fn is_loaded_impl(&self) -> bool {
        let Some(cb) = self.is_loaded_fn.as_ref() else {
            // No callback provided -- the upstream `ModelManager` tracks the
            // loaded flag itself, so we conservatively report `false`.
            return false;
        };
        match Self::invoke_with_result(&cb.0, "isLoaded", &self.id).await {
            Ok(val) => val.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    }

    async fn memory_bytes_impl(&self) -> Option<u64> {
        let Some(cb) = self.memory_bytes_fn.as_ref() else {
            return Some(self.memory_estimate_bytes);
        };
        match Self::invoke_with_result(&cb.0, "memoryBytes", &self.id).await {
            Ok(val) => {
                if val.is_null() || val.is_undefined() {
                    return None;
                }
                match val.as_f64() {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss
                    )]
                    Some(n) if n.is_finite() && n >= 0.0 => Some(n as u64),
                    _ => Some(self.memory_estimate_bytes),
                }
            }
            Err(_) => Some(self.memory_estimate_bytes),
        }
    }
}

#[async_trait]
impl LocalModel for JsLocalModelAdapter {
    async fn load(&self) -> Result<(), BlazenError> {
        // SAFETY: WASM is single-threaded; the future is vacuously Send.
        SendFuture(self.load_impl()).await
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        // SAFETY: WASM is single-threaded; the future is vacuously Send.
        SendFuture(self.unload_impl()).await
    }

    async fn is_loaded(&self) -> bool {
        // SAFETY: WASM is single-threaded; the future is vacuously Send.
        SendFuture(self.is_loaded_impl()).await
    }

    fn device(&self) -> Device {
        self.device
    }

    async fn memory_bytes(&self) -> Option<u64> {
        // SAFETY: WASM is single-threaded; the future is vacuously Send.
        SendFuture(self.memory_bytes_impl()).await
    }
}

// ---------------------------------------------------------------------------
// WasmModelManager
// ---------------------------------------------------------------------------

/// Memory-budget-aware model manager exposed to JS / TS.
///
/// Wraps [`blazen_manager::ModelManager`] and bridges JS lifecycle callbacks
/// (`load`, `unload`) to Rust via [`JsLocalModelAdapter`]. All async methods
/// return a `Promise<void>` (or the appropriate resolved value).
///
/// The WASM SDK's single-argument constructor sizes the CPU memory pool —
/// WebGPU/WASM workloads almost always run on the host. To configure a GPU
/// pool too, supply both `cpuRamGb` and `gpuVramGb` to the constructor
/// (advanced).
///
/// ```js
/// const manager = new ModelManager(8); // 8 GB CPU pool
/// await manager.register('my-model', model, 5_000_000_000, {
///   load: async () => console.log('loading...'),
///   unload: async () => console.log('unloading...'),
/// });
/// await manager.load('my-model');
/// ```
#[wasm_bindgen(js_name = "ModelManager")]
pub struct WasmModelManager {
    inner: Arc<ModelManager>,
    /// Cached CPU pool budget in bytes for synchronous `budgetBytes` access.
    cpu_budget_cache: f64,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmModelManager {}
unsafe impl Sync for WasmModelManager {}

/// Extract the lifecycle callbacks from a `lifecycle` JS object.
///
/// Returns a tuple of `(load, unload, isLoaded, memoryBytes, device)`:
/// - `load` and `unload` are required and must be functions.
/// - `isLoaded`, `memoryBytes`, and `device` are optional. If absent (or
///   `undefined`/`null`) they are returned as `None`. If present but not a
///   function, an error is returned.
fn extract_lifecycle(
    lifecycle: &JsValue,
) -> Result<
    (
        Function,
        Function,
        Option<Function>,
        Option<Function>,
        Option<Function>,
    ),
    JsValue,
> {
    if !lifecycle.is_object() {
        return Err(JsValue::from_str("lifecycle must be an object"));
    }

    let load_val = Reflect::get(lifecycle, &JsValue::from_str("load"))
        .map_err(|e| JsValue::from_str(&format!("lifecycle.load lookup failed: {e:?}")))?;
    let unload_val = Reflect::get(lifecycle, &JsValue::from_str("unload"))
        .map_err(|e| JsValue::from_str(&format!("lifecycle.unload lookup failed: {e:?}")))?;

    let load: Function = load_val
        .dyn_into()
        .map_err(|_| JsValue::from_str("lifecycle.load must be a function"))?;
    let unload: Function = unload_val
        .dyn_into()
        .map_err(|_| JsValue::from_str("lifecycle.unload must be a function"))?;

    let is_loaded = extract_optional_fn(lifecycle, "isLoaded")?;
    let memory_bytes = extract_optional_fn(lifecycle, "memoryBytes")?;
    let device = extract_optional_fn(lifecycle, "device")?;

    Ok((load, unload, is_loaded, memory_bytes, device))
}

/// Read an optional function-valued key from a JS object.
///
/// - Missing / `undefined` / `null` -> `Ok(None)`.
/// - Present and a function -> `Ok(Some(fn))`.
/// - Present but not a function -> `Err(...)`.
fn extract_optional_fn(obj: &JsValue, key: &str) -> Result<Option<Function>, JsValue> {
    let val = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|e| JsValue::from_str(&format!("lifecycle.{key} lookup failed: {e:?}")))?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let func: Function = val
        .dyn_into()
        .map_err(|_| JsValue::from_str(&format!("lifecycle.{key} must be a function")))?;
    Ok(Some(func))
}

/// Invoke the optional `device` callback to resolve the model's target device.
/// Returns [`Device::Cpu`] if no callback was supplied or the call fails.
fn resolve_device_from_callback(device_fn: Option<&Function>) -> Device {
    let Some(func) = device_fn else {
        return Device::Cpu;
    };
    match func.call0(&JsValue::NULL) {
        Ok(val) => val
            .as_string()
            .as_deref()
            .and_then(|s| Device::parse(s).ok())
            .unwrap_or(Device::Cpu),
        Err(_) => Device::Cpu,
    }
}

#[wasm_bindgen(js_class = "ModelManager")]
impl WasmModelManager {
    /// Create a new model manager with the given per-pool memory budgets in
    /// gigabytes.
    ///
    /// @param cpuRamGb - Host RAM budget in gigabytes for `Pool::Cpu`.
    /// @param gpuVramGb - Optional GPU VRAM budget in gigabytes for
    ///                    `Pool::Gpu(0)`. Defaults to `0` (no GPU pool).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(cpu_ram_gb: f64, gpu_vram_gb: Option<f64>) -> Self {
        let gpu_gb = gpu_vram_gb.unwrap_or(0.0);
        Self {
            inner: Arc::new(ModelManager::with_budgets_gb(cpu_ram_gb, gpu_gb)),
            cpu_budget_cache: cpu_ram_gb * 1_073_741_824.0,
        }
    }

    /// Register a model with its estimated memory footprint.
    ///
    /// The model starts in the unloaded state. The `lifecycle` object must
    /// implement `load()` and `unload()` async methods that are called when
    /// the manager needs to load or unload the model. It may also provide
    /// optional `isLoaded()`, `memoryBytes()`, and `device()` callbacks
    /// (sync or async) which are forwarded to the underlying [`LocalModel`]
    /// trait methods.
    ///
    /// Returns a `Promise<void>` that resolves once registration completes.
    ///
    /// @param id                  - Unique identifier for this model.
    /// @param model               - The model value (`CompletionModel`, etc.) or `null`.
    /// @param memoryEstimateBytes - Estimated memory footprint in bytes.
    /// @param lifecycle           - Object with `load()` and `unload()` async methods,
    ///                              plus optional `isLoaded()`, `memoryBytes()`, and
    ///                              `device()` callbacks.
    ///
    /// ```js
    /// // Minimal: only required callbacks (defaults to Pool::Cpu).
    /// await manager.register('m1', null, 5e9, {
    ///   load: async () => {},
    ///   unload: async () => {},
    /// });
    ///
    /// // Full: opt into runtime isLoaded / memoryBytes / device queries.
    /// let loaded = false;
    /// await manager.register('m2', null, 5e9, {
    ///   load: async () => { loaded = true; },
    ///   unload: async () => { loaded = false; },
    ///   isLoaded: () => loaded,
    ///   memoryBytes: async () => 5e9,
    ///   device: () => 'cuda:0',
    /// });
    /// ```
    ///
    /// # Errors
    ///
    /// Rejects if `lifecycle` is not an object, its `load`/`unload` keys are
    /// not functions, or its optional callback keys are present but not
    /// functions.
    #[wasm_bindgen]
    #[allow(
        clippy::needless_pass_by_value,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn register(
        &self,
        id: String,
        _model: Option<JsValue>,
        memory_estimate_bytes: f64,
        lifecycle: Object,
    ) -> Result<Promise, JsValue> {
        let (load_fn, unload_fn, is_loaded_fn, memory_bytes_fn, device_fn) =
            extract_lifecycle(lifecycle.as_ref())?;

        let device = resolve_device_from_callback(device_fn.as_ref());

        let adapter: Arc<dyn LocalModel> = Arc::new(JsLocalModelAdapter {
            id: id.clone(),
            load_fn: Arc::new(JsClosure(load_fn)),
            unload_fn: Arc::new(JsClosure(unload_fn)),
            is_loaded_fn: is_loaded_fn.map(|f| Arc::new(JsClosure(f))),
            memory_bytes_fn: memory_bytes_fn.map(|f| Arc::new(JsClosure(f))),
            memory_estimate_bytes: memory_estimate_bytes as u64,
            device,
        });

        let inner = Arc::clone(&self.inner);
        Ok(future_to_promise(SendFuture(async move {
            inner
                .register(&id, adapter, memory_estimate_bytes as u64)
                .await;
            Ok(JsValue::UNDEFINED)
        })))
    }

    /// Unregister a model, removing it from the manager.
    ///
    /// If the model is currently loaded, it is **not** unloaded first --
    /// call `unload()` before `unregister()` if cleanup is needed.
    ///
    /// `blazen_manager::ModelManager` does not currently expose an
    /// unregister entrypoint; this method is a no-op preserved for API
    /// compatibility.
    #[wasm_bindgen]
    pub fn unregister(&self, _id: &str) {
        // No-op: upstream `ModelManager` has no `unregister` method.
        // Callers should `unload` before discarding the manager.
    }

    /// Load a model, evicting LRU models in the same pool if needed.
    ///
    /// Returns a `Promise<void>` that resolves when the model is loaded.
    #[wasm_bindgen]
    pub fn load(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            inner
                .load(&id)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e}")))?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Unload a model, freeing its memory budget.
    ///
    /// Returns a `Promise<void>` that resolves when the model is unloaded.
    /// Idempotent -- unloading an already-unloaded model is a no-op.
    #[wasm_bindgen]
    pub fn unload(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            inner
                .unload(&id)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e}")))?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Check whether a model is currently loaded.
    #[wasm_bindgen(js_name = "isLoaded")]
    pub fn is_loaded(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            let loaded = inner.is_loaded(&id).await;
            Ok(JsValue::from_bool(loaded))
        }))
    }

    /// Total memory currently used by loaded models in the given pool.
    ///
    /// @param pool - Pool label like `"cpu"` or `"gpu:0"`. Defaults to `"cpu"`.
    #[wasm_bindgen(js_name = "usedBytes")]
    pub fn used_bytes(&self, pool: Option<String>) -> Result<Promise, JsValue> {
        let pool = parse_pool_label(pool.as_deref().unwrap_or("cpu"))?;
        let inner = Arc::clone(&self.inner);
        Ok(future_to_promise(SendFuture(async move {
            let used = inner.used_bytes(pool).await;
            #[allow(clippy::cast_precision_loss)]
            Ok(JsValue::from_f64(used as f64))
        })))
    }

    /// Available memory within the given pool's budget.
    ///
    /// @param pool - Pool label like `"cpu"` or `"gpu:0"`. Defaults to `"cpu"`.
    #[wasm_bindgen(js_name = "availableBytes")]
    pub fn available_bytes(&self, pool: Option<String>) -> Result<Promise, JsValue> {
        let pool = parse_pool_label(pool.as_deref().unwrap_or("cpu"))?;
        let inner = Arc::clone(&self.inner);
        Ok(future_to_promise(SendFuture(async move {
            let available = inner.available_bytes(pool).await;
            #[allow(clippy::cast_precision_loss)]
            Ok(JsValue::from_f64(available as f64))
        })))
    }

    /// The total CPU pool budget in bytes.
    #[wasm_bindgen(getter, js_name = "budgetBytes")]
    #[must_use]
    pub fn budget_bytes(&self) -> f64 {
        self.cpu_budget_cache
    }

    /// All configured pools and their budgets in bytes.
    ///
    /// Returns `Array<{ pool: string, budgetBytes: number }>`.
    #[wasm_bindgen]
    pub fn pools(&self) -> Result<JsValue, JsValue> {
        let pools = self.inner.pools();
        // Sort for deterministic ordering across calls (HashMap iteration is not stable).
        let mut entries: Vec<(Pool, u64)> = pools;
        entries.sort_by_key(|(p, _)| match p {
            Pool::Cpu => (0, 0),
            Pool::Gpu(n) => (1, *n),
        });

        let arr = js_sys::Array::new();
        for (p, b) in entries {
            let obj = Object::new();
            let _ = Reflect::set(&obj, &"pool".into(), &JsValue::from_str(&format!("{p}")));
            #[allow(clippy::cast_precision_loss)]
            let budget = JsValue::from_f64(b as f64);
            let _ = Reflect::set(&obj, &"budgetBytes".into(), &budget);
            arr.push(&obj);
        }
        Ok(arr.into())
    }

    /// Status of all registered models.
    ///
    /// Returns a `Promise<Array<{ id: string, loaded: boolean, memoryEstimateBytes: number, pool: string }>>`.
    #[wasm_bindgen]
    pub fn status(&self) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            let statuses = inner.status().await;
            let arr = js_sys::Array::new();
            for s in statuses {
                let obj = Object::new();
                let _ = Reflect::set(&obj, &"id".into(), &JsValue::from_str(&s.id));
                let _ = Reflect::set(&obj, &"loaded".into(), &JsValue::from_bool(s.loaded));
                #[allow(clippy::cast_precision_loss)]
                let mem = JsValue::from_f64(s.memory_estimate_bytes as f64);
                let _ = Reflect::set(&obj, &"memoryEstimateBytes".into(), &mem);
                let _ = Reflect::set(
                    &obj,
                    &"pool".into(),
                    &JsValue::from_str(&format!("{}", s.pool)),
                );
                arr.push(&obj);
            }
            Ok(arr.into())
        }))
    }
}
