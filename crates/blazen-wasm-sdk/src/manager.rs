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

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

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
    /// Captured for parity with the native [`LocalModel`] adapter surface.
    /// In WASM, adapter dispatch goes through `WasmModelManager`'s own
    /// callback map (because the Rust-side `ModelManager` requires a
    /// filesystem path we don't have in browsers), so these handles are
    /// retained on the adapter purely so the same lifecycle object can
    /// service both routes if a future Rust-side WASM `load_adapter`
    /// path materializes.
    #[allow(dead_code)]
    load_adapter_fn: Option<Arc<JsClosure>>,
    #[allow(dead_code)]
    unload_adapter_fn: Option<Arc<JsClosure>>,
    #[allow(dead_code)]
    list_adapters_fn: Option<Arc<JsClosure>>,
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
        self.device.clone()
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
///
/// # Adapter (PEFT/LoRA) semantics in WASM
///
/// Unlike the native bindings, the WASM SDK's [`Self::load_adapter`],
/// [`Self::unload_adapter`], and [`Self::list_adapters`] methods BYPASS
/// the Rust-side [`ModelManager`] entirely. There is no filesystem path
/// to give it — the browser sandbox has no `std::fs`. Instead, the bytes
/// are forwarded directly to the JS lifecycle callbacks
/// (`loadAdapter`/`unloadAdapter`/`listAdapters`) supplied at register
/// time. The JS lifecycle IS the adapter manager in browsers; the SDK
/// just routes calls to it.
///
/// Expected JS callback signatures:
/// - `loadAdapter(adapterBytes: Uint8Array, options: object) -> Promise<string>`
///   resolves to the assigned adapter id.
/// - `unloadAdapter(adapterId: string) -> Promise<void>`.
/// - `listAdapters() -> Array<{ adapterId, scale, sourceDir, memoryBytes }>`
///   (sync or async). If absent, `listAdapters` returns an empty array.
///
/// If `loadAdapter` or `unloadAdapter` were not provided at register time,
/// the returned promise rejects with a diagnostic message — the SDK has
/// no built-in browser backend that can mount adapters on its own.
#[wasm_bindgen(js_name = "ModelManager")]
pub struct WasmModelManager {
    inner: Arc<ModelManager>,
    /// Cached CPU pool budget in bytes for synchronous `budgetBytes` access.
    cpu_budget_cache: f64,
    /// Per-model adapter callbacks captured at register time. Indexed by
    /// model id. Used by `load_adapter` / `unload_adapter` / `list_adapters`
    /// to forward bytes directly to JS without going through the Rust-side
    /// `ModelManager` (which requires a filesystem path).
    adapter_callbacks: Arc<Mutex<HashMap<String, AdapterCallbacks>>>,
}

/// JS callbacks for browser-side adapter management, captured at register
/// time and dispatched by `WasmModelManager::{load,unload,list}_adapter`.
#[derive(Clone, Default)]
struct AdapterCallbacks {
    load: Option<Arc<JsClosure>>,
    unload: Option<Arc<JsClosure>>,
    list: Option<Arc<JsClosure>>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmModelManager {}
unsafe impl Sync for WasmModelManager {}

/// Lifecycle callbacks extracted from a `lifecycle` JS object.
///
/// `load` and `unload` are required; everything else is optional.
struct LifecycleCallbacks {
    load: Function,
    unload: Function,
    is_loaded: Option<Function>,
    memory_bytes: Option<Function>,
    device: Option<Function>,
    load_adapter: Option<Function>,
    unload_adapter: Option<Function>,
    list_adapters: Option<Function>,
}

/// Extract the lifecycle callbacks from a `lifecycle` JS object.
///
/// - `load` and `unload` are required and must be functions.
/// - `isLoaded`, `memoryBytes`, `device`, `loadAdapter`, `unloadAdapter`,
///   and `listAdapters` are optional. If absent (or `undefined`/`null`)
///   they are returned as `None`. If present but not a function, an error
///   is returned.
fn extract_lifecycle(lifecycle: &JsValue) -> Result<LifecycleCallbacks, JsValue> {
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
    let load_adapter = extract_optional_fn(lifecycle, "loadAdapter")?;
    let unload_adapter = extract_optional_fn(lifecycle, "unloadAdapter")?;
    let list_adapters = extract_optional_fn(lifecycle, "listAdapters")?;

    Ok(LifecycleCallbacks {
        load,
        unload,
        is_loaded,
        memory_bytes,
        device,
        load_adapter,
        unload_adapter,
        list_adapters,
    })
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
            adapter_callbacks: Arc::new(Mutex::new(HashMap::new())),
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
    ///
    /// # Panics
    ///
    /// Panics if the internal adapter-callbacks mutex is poisoned (should
    /// not happen in single-threaded WASM).
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
        let cbs = extract_lifecycle(lifecycle.as_ref())?;

        let device = resolve_device_from_callback(cbs.device.as_ref());

        let load_adapter_fn = cbs.load_adapter.map(|f| Arc::new(JsClosure(f)));
        let unload_adapter_fn = cbs.unload_adapter.map(|f| Arc::new(JsClosure(f)));
        let list_adapters_fn = cbs.list_adapters.map(|f| Arc::new(JsClosure(f)));

        // Stash the adapter callbacks for the bypass-mode `load_adapter` /
        // `unload_adapter` / `list_adapters` methods. These don't flow
        // through the Rust-side `ModelManager` (no `Path` in browsers).
        {
            let mut map = self
                .adapter_callbacks
                .lock()
                .expect("adapter_callbacks mutex poisoned");
            map.insert(
                id.clone(),
                AdapterCallbacks {
                    load: load_adapter_fn.clone(),
                    unload: unload_adapter_fn.clone(),
                    list: list_adapters_fn.clone(),
                },
            );
        }

        let adapter: Arc<dyn LocalModel> = Arc::new(JsLocalModelAdapter {
            id: id.clone(),
            load_fn: Arc::new(JsClosure(cbs.load)),
            unload_fn: Arc::new(JsClosure(cbs.unload)),
            is_loaded_fn: cbs.is_loaded.map(|f| Arc::new(JsClosure(f))),
            memory_bytes_fn: cbs.memory_bytes.map(|f| Arc::new(JsClosure(f))),
            load_adapter_fn,
            unload_adapter_fn,
            list_adapters_fn,
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

    /// Mount a `PEFT`-format `LoRA` adapter via the registered model's JS lifecycle.
    ///
    /// Unlike the native bindings, the WASM SDK does NOT call the Rust-side
    /// [`ModelManager::load_adapter`] — there's no filesystem path to give it
    /// inside a browser sandbox. Instead, the `adapter_bytes` (an inflated
    /// `PEFT` adapter archive — typically a `Uint8Array` of a `.safetensors` /
    /// tarball / zip the caller has already prepared) and the `options`
    /// object are forwarded verbatim to the `loadAdapter` callback that was
    /// supplied to [`Self::register`]. The callback is expected to resolve
    /// to a string adapter id, which becomes the resolved value of the
    /// returned promise.
    ///
    /// `options` should follow the shape `{ adapterId: string, scale?: number }`;
    /// the Rust side does not introspect it.
    ///
    /// # Errors
    ///
    /// Rejects with `JsValue::from_str` if no `loadAdapter` callback was
    /// registered for `id` (the SDK has no built-in browser backend that
    /// can mount adapters on its own), if the model id is unknown, or if
    /// the underlying JS callback throws / rejects.
    ///
    /// # Panics
    ///
    /// Panics if the internal adapter-callbacks mutex is poisoned (a panic
    /// while another caller was holding the lock — should not happen in
    /// single-threaded WASM).
    #[wasm_bindgen(js_name = "loadAdapter")]
    pub fn load_adapter(
        &self,
        id: String,
        adapter_bytes: js_sys::Uint8Array,
        options: Object,
    ) -> Promise {
        let callbacks = self.adapter_callbacks.clone();
        let bytes_val: JsValue = adapter_bytes.into();
        let opts_val: JsValue = options.into();
        future_to_promise(SendFuture(async move {
            let cb = {
                let map = callbacks.lock().expect("adapter_callbacks mutex poisoned");
                map.get(&id).and_then(|c| c.load.clone())
            };
            let Some(cb) = cb else {
                return Err(JsValue::from_str(&format!(
                    "model '{id}' lifecycle does not implement loadAdapter \
                     (in-browser adapter mounting not supported by SDK built-in backends)"
                )));
            };
            let result = cb.0.call2(&JsValue::NULL, &bytes_val, &opts_val).map_err(|e| {
                JsValue::from_str(&format!(
                    "model '{id}' lifecycle.loadAdapter() threw: {e:?}"
                ))
            })?;
            if result.has_type::<Promise>() {
                let promise: Promise = result.unchecked_into();
                let val = JsFuture::from(promise).await.map_err(|e| {
                    JsValue::from_str(&format!(
                        "model '{id}' lifecycle.loadAdapter() rejected: {e:?}"
                    ))
                })?;
                Ok(val)
            } else {
                Ok(result)
            }
        }))
    }

    /// Unmount a previously-loaded adapter via the registered model's JS
    /// lifecycle.
    ///
    /// Bypasses the Rust-side [`ModelManager`] for the same reason as
    /// [`Self::load_adapter`]. Forwards `adapter_id` to the `unloadAdapter`
    /// callback registered at [`Self::register`] time.
    ///
    /// # Errors
    ///
    /// Rejects if no `unloadAdapter` callback was registered for `id`,
    /// the model id is unknown, or the underlying JS callback throws /
    /// rejects.
    ///
    /// # Panics
    ///
    /// Panics if the internal adapter-callbacks mutex is poisoned (should
    /// not happen in single-threaded WASM).
    #[wasm_bindgen(js_name = "unloadAdapter")]
    pub fn unload_adapter(&self, id: String, adapter_id: String) -> Promise {
        let callbacks = self.adapter_callbacks.clone();
        future_to_promise(SendFuture(async move {
            let cb = {
                let map = callbacks.lock().expect("adapter_callbacks mutex poisoned");
                map.get(&id).and_then(|c| c.unload.clone())
            };
            let Some(cb) = cb else {
                return Err(JsValue::from_str(&format!(
                    "model '{id}' lifecycle does not implement unloadAdapter \
                     (in-browser adapter mounting not supported by SDK built-in backends)"
                )));
            };
            let result = cb
                .0
                .call1(&JsValue::NULL, &JsValue::from_str(&adapter_id))
                .map_err(|e| {
                    JsValue::from_str(&format!(
                        "model '{id}' lifecycle.unloadAdapter() threw: {e:?}"
                    ))
                })?;
            if result.has_type::<Promise>() {
                let promise: Promise = result.unchecked_into();
                JsFuture::from(promise).await.map_err(|e| {
                    JsValue::from_str(&format!(
                        "model '{id}' lifecycle.unloadAdapter() rejected: {e:?}"
                    ))
                })?;
            }
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// List adapters currently mounted on the given model.
    ///
    /// Bypasses the Rust-side [`ModelManager`] for the same reason as
    /// [`Self::load_adapter`]. Invokes the `listAdapters` callback
    /// registered at [`Self::register`] time. The callback is expected
    /// to return (or resolve to) an `Array<{ adapterId, scale, sourceDir,
    /// memoryBytes }>`, which is returned verbatim.
    ///
    /// If no `listAdapters` callback was registered, resolves to an
    /// empty array (truthful — "no adapters mounted" — not an error).
    ///
    /// # Errors
    ///
    /// Rejects only if the registered callback itself throws or rejects.
    ///
    /// # Panics
    ///
    /// Panics if the internal adapter-callbacks mutex is poisoned (should
    /// not happen in single-threaded WASM).
    #[wasm_bindgen(js_name = "listAdapters")]
    pub fn list_adapters(&self, id: String) -> Promise {
        let callbacks = self.adapter_callbacks.clone();
        future_to_promise(SendFuture(async move {
            let cb = {
                let map = callbacks.lock().expect("adapter_callbacks mutex poisoned");
                map.get(&id).and_then(|c| c.list.clone())
            };
            let Some(cb) = cb else {
                // No callback -> truthfully report an empty list.
                return Ok(js_sys::Array::new().into());
            };
            let result = cb.0.call0(&JsValue::NULL).map_err(|e| {
                JsValue::from_str(&format!(
                    "model '{id}' lifecycle.listAdapters() threw: {e:?}"
                ))
            })?;
            if result.has_type::<Promise>() {
                let promise: Promise = result.unchecked_into();
                let val = JsFuture::from(promise).await.map_err(|e| {
                    JsValue::from_str(&format!(
                        "model '{id}' lifecycle.listAdapters() rejected: {e:?}"
                    ))
                })?;
                Ok(val)
            } else {
                Ok(result)
            }
        }))
    }

    /// Auto-detect-and-register a Hugging Face repo. NOT SUPPORTED in the
    /// WASM SDK — the browser sandbox cannot reach `~/.cache/huggingface/`
    /// and cannot stream weight files into the in-WASM provider backends.
    ///
    /// The returned promise rejects immediately with a diagnostic message.
    /// Callers must download the weights out-of-band and feed them to the
    /// model via `lifecycle.load` at register time.
    ///
    /// @param _id      - Unused; accepted for signature parity with native.
    /// @param _repo    - Unused; accepted for signature parity with native.
    /// @param _options - Unused; accepted for signature parity with native.
    #[wasm_bindgen(js_name = "loadFromHf")]
    pub fn load_from_hf(
        &self,
        _id: String,
        _repo: String,
        _options: Option<JsValue>,
    ) -> Promise {
        Promise::reject(&JsValue::from_str(
            "WASM SDK does not download HF models in-browser; provide bytes via lifecycle.load",
        ))
    }

    #[wasm_bindgen(js_name = "trainLora")]
    pub fn train_lora(&self, _config: JsValue, _dataset: JsValue, _progress: JsValue) -> Promise {
        // Why: training requires GB of weights + hours of compute, infeasible in a browser tab.
        let msg = "WASM SDK does not train models in-browser; training requires multi-GB weights and hours of compute. Run training in a native Blazen process and load the resulting adapter via load_adapter.";
        Promise::reject(&js_sys::Error::new(msg).into())
    }

    #[wasm_bindgen(js_name = "trainDpo")]
    pub fn train_dpo(&self, _config: JsValue, _dataset: JsValue, _progress: JsValue) -> Promise {
        // Why: DPO needs the policy + a frozen reference model in memory simultaneously, plus
        // gradient state across all trainable params -- multi-GB working set, hours of compute.
        let msg = "WASM SDK does not train models in-browser; preference training requires multi-GB weights, reference-model loading, and hours of compute. Use the Python/Node binding on a server.";
        Promise::reject(&js_sys::Error::new(msg).into())
    }

    #[wasm_bindgen(js_name = "trainOrpo")]
    pub fn train_orpo(&self, _config: JsValue, _dataset: JsValue, _progress: JsValue) -> Promise {
        // Why: ORPO fuses SFT + odds-ratio preference loss on the policy model; still needs full
        // gradient state and multi-GB weights -- infeasible in a browser tab.
        let msg = "WASM SDK does not train models in-browser; preference training requires multi-GB weights, reference-model loading, and hours of compute. Use the Python/Node binding on a server.";
        Promise::reject(&js_sys::Error::new(msg).into())
    }

    #[wasm_bindgen(js_name = "trainSimpo")]
    pub fn train_simpo(&self, _config: JsValue, _dataset: JsValue, _progress: JsValue) -> Promise {
        // Why: SimPO is reference-free, so it skips the reference model -- but it still needs the
        // full policy weights + gradient state in memory, which is multi-GB and hours of compute.
        let msg = "WASM SDK does not train models in-browser; reference-free preference training (SimPO) still requires multi-GB policy weights and hours of compute. Use the Python/Node binding on a server.";
        Promise::reject(&js_sys::Error::new(msg).into())
    }

    #[wasm_bindgen(js_name = "trainKto")]
    pub fn train_kto(&self, _config: JsValue, _dataset: JsValue, _progress: JsValue) -> Promise {
        // Why: KTO uses unpaired desirable/undesirable signals but still trains against a reference
        // model and full policy gradients -- multi-GB state, hours of compute.
        let msg = "WASM SDK does not train models in-browser; preference training requires multi-GB weights, reference-model loading, and hours of compute. Use the Python/Node binding on a server.";
        Promise::reject(&js_sys::Error::new(msg).into())
    }

    #[wasm_bindgen(js_name = "fineTune")]
    pub fn fine_tune(&self, _config: JsValue, _dataset: JsValue, _progress: JsValue) -> Promise {
        // Why: full fine-tuning updates every parameter, so optimizer state alone is ~2-3x the
        // model size on top of the weights themselves -- many GB, infeasible in a browser tab.
        let msg = "WASM SDK does not perform full fine-tuning; this requires gradient computation across all model parameters (several GB of state) and hours of compute. Use the Python/Node binding on a server.";
        Promise::reject(&js_sys::Error::new(msg).into())
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
            Pool::Remote => (2, 0),
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
