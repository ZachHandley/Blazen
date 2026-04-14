//! WASM-native model manager with budget-aware tracking.
//!
//! This is a lightweight reimplementation of `blazen-manager`'s
//! `ModelManager` for WASM. Since WASM is single-threaded, it uses
//! `RefCell` instead of `tokio::sync::Mutex`, avoiding the tokio
//! dependency entirely.
//!
//! The manager tracks registered JS-backed models and their estimated
//! VRAM footprints. When loading a model that would exceed the
//! configured budget, the least-recently-used loaded model is unloaded
//! first.
//!
//! ```js
//! const manager = new ModelManager(8); // 8 GB budget
//!
//! const model = CompletionModel.webLlm('Llama-3.1-8B-Instruct-q4f32_1-MLC');
//! manager.register('llama-8b', model, 4_000_000_000, {
//!   load: async () => { /* load model */ },
//!   unload: async () => { /* unload model */ },
//! });
//!
//! await manager.load('llama-8b');
//! const status = manager.status();
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::pin::Pin;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as agent.rs)
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
// Internal state
// ---------------------------------------------------------------------------

/// A use counter that monotonically increases (simulates `Instant` which
/// is unavailable in WASM without `performance.now`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct UseTick(u64);

struct RegisteredModel {
    /// The `CompletionModel` or any JS value associated with this model.
    #[allow(dead_code)]
    model_value: JsValue,
    /// Estimated VRAM footprint in bytes.
    vram_estimate: f64,
    /// Whether the model is currently loaded.
    loaded: bool,
    /// Monotonic use counter for LRU eviction.
    last_used: Option<UseTick>,
    /// JS object with `load()` and `unload()` async methods.
    lifecycle: JsValue,
}

struct ManagerState {
    budget_bytes: f64,
    models: HashMap<String, RegisteredModel>,
    tick: u64,
}

impl ManagerState {
    fn next_tick(&mut self) -> UseTick {
        self.tick += 1;
        UseTick(self.tick)
    }

    fn used_bytes(&self) -> f64 {
        self.models
            .values()
            .filter(|m| m.loaded)
            .map(|m| m.vram_estimate)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// WasmModelManager
// ---------------------------------------------------------------------------

/// A WASM-native model manager with VRAM budget tracking and LRU eviction.
///
/// Tracks registered models and their estimated VRAM footprints. When
/// loading a model that would exceed the budget, the least-recently-used
/// loaded model is unloaded first.
///
/// ```js
/// const manager = new ModelManager(8); // 8 GB budget
/// manager.register('my-model', model, 5_000_000_000, {
///   load: async () => console.log('loading...'),
///   unload: async () => console.log('unloading...'),
/// });
/// await manager.load('my-model');
/// ```
#[wasm_bindgen(js_name = "ModelManager")]
pub struct WasmModelManager {
    state: Rc<RefCell<ManagerState>>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmModelManager {}
unsafe impl Sync for WasmModelManager {}

/// Call an async method on a JS lifecycle object and await its result.
async fn call_lifecycle_method(
    lifecycle: &JsValue,
    method: &str,
) -> Result<(), JsValue> {
    let func = js_sys::Reflect::get(lifecycle, &JsValue::from_str(method))
        .map_err(|e| JsValue::from_str(&format!("lifecycle.{method} not found: {e:?}")))?;

    if !func.is_function() {
        return Err(JsValue::from_str(&format!(
            "lifecycle.{method} is not a function"
        )));
    }

    let func: &js_sys::Function = func.unchecked_ref();
    let result = func
        .call0(&JsValue::NULL)
        .map_err(|e| JsValue::from_str(&format!("lifecycle.{method}() threw: {e:?}")))?;

    if result.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = result.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("lifecycle.{method}() rejected: {e:?}")))?;
    }

    Ok(())
}

#[wasm_bindgen(js_class = "ModelManager")]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value,
)]
impl WasmModelManager {
    /// Create a new model manager with the given VRAM budget in gigabytes.
    ///
    /// @param budgetGb - Total VRAM budget in gigabytes (e.g. `8` for 8 GB).
    #[wasm_bindgen(constructor)]
    pub fn new(budget_gb: f64) -> Self {
        Self {
            state: Rc::new(RefCell::new(ManagerState {
                budget_bytes: budget_gb * 1_073_741_824.0,
                models: HashMap::new(),
                tick: 0,
            })),
        }
    }

    /// Register a model with its estimated VRAM footprint.
    ///
    /// The model starts in the unloaded state. The `lifecycle` object must
    /// implement `load()` and `unload()` async methods that are called when
    /// the manager needs to load or unload the model.
    ///
    /// @param id            - Unique identifier for this model.
    /// @param model         - The model value (CompletionModel, etc.).
    /// @param vramEstimate  - Estimated VRAM footprint in bytes.
    /// @param lifecycle     - Object with `load()` and `unload()` async methods.
    pub fn register(
        &self,
        id: &str,
        model: JsValue,
        vram_estimate: f64,
        lifecycle: JsValue,
    ) -> Result<(), JsValue> {
        // Validate lifecycle has load and unload methods.
        if lifecycle.is_object() {
            let load = js_sys::Reflect::get(&lifecycle, &JsValue::from_str("load"))
                .unwrap_or(JsValue::UNDEFINED);
            let unload = js_sys::Reflect::get(&lifecycle, &JsValue::from_str("unload"))
                .unwrap_or(JsValue::UNDEFINED);
            if !load.is_function() || !unload.is_function() {
                return Err(JsValue::from_str(
                    "lifecycle must have 'load' and 'unload' functions",
                ));
            }
        } else {
            return Err(JsValue::from_str("lifecycle must be an object"));
        }

        let mut state = self.state.borrow_mut();
        state.models.insert(
            id.to_owned(),
            RegisteredModel {
                model_value: model,
                vram_estimate,
                loaded: false,
                last_used: None,
                lifecycle,
            },
        );
        Ok(())
    }

    /// Unregister a model, removing it from the manager.
    ///
    /// If the model is currently loaded, it is **not** unloaded first --
    /// call `unload()` before `unregister()` if cleanup is needed.
    pub fn unregister(&self, id: &str) {
        let mut state = self.state.borrow_mut();
        state.models.remove(id);
    }

    /// Load a model, evicting LRU models if the budget would be exceeded.
    ///
    /// Returns a `Promise<void>` that resolves when the model is loaded.
    pub fn load(&self, id: String) -> js_sys::Promise {
        let state_rc = Rc::clone(&self.state);

        future_to_promise(SendFuture(async move {
            // Check model exists and if already loaded.
            {
                let mut state = state_rc.borrow_mut();
                let entry = state.models.get(&id).ok_or_else(|| {
                    JsValue::from_str(&format!("model '{id}' is not registered"))
                })?;

                if entry.loaded {
                    let tick = state.next_tick();
                    let entry = state.models.get_mut(&id).unwrap();
                    entry.last_used = Some(tick);
                    return Ok(JsValue::UNDEFINED);
                }

                let needed = entry.vram_estimate;
                if needed > state.budget_bytes {
                    return Err(JsValue::from_str(&format!(
                        "model '{id}' requires {needed} bytes but total budget is {} bytes",
                        state.budget_bytes
                    )));
                }
            }

            // Evict LRU models until we have space.
            loop {
                let (need_evict, lru_id, lru_lifecycle) = {
                    let state = state_rc.borrow();
                    let needed = state.models.get(&id).unwrap().vram_estimate;
                    let used = state.used_bytes();

                    if used + needed <= state.budget_bytes {
                        (false, String::new(), JsValue::UNDEFINED)
                    } else {
                        // Find LRU loaded model.
                        let lru = state
                            .models
                            .iter()
                            .filter(|(k, v)| v.loaded && k.as_str() != id)
                            .min_by_key(|(_, v)| v.last_used)
                            .map(|(k, v)| (k.clone(), v.lifecycle.clone()));

                        match lru {
                            Some((lru_id, lru_lifecycle)) => {
                                (true, lru_id, lru_lifecycle)
                            }
                            None => {
                                return Err(JsValue::from_str(&format!(
                                    "cannot free enough VRAM to load model '{id}'"
                                )));
                            }
                        }
                    }
                };

                if !need_evict {
                    break;
                }

                // Unload the LRU model (outside borrow).
                call_lifecycle_method(&lru_lifecycle, "unload").await?;

                {
                    let mut state = state_rc.borrow_mut();
                    if let Some(e) = state.models.get_mut(&lru_id) {
                        e.loaded = false;
                        e.last_used = None;
                    }
                }
            }

            // Load the requested model (outside borrow).
            let lifecycle = {
                let state = state_rc.borrow();
                state
                    .models
                    .get(&id)
                    .ok_or_else(|| {
                        JsValue::from_str(&format!("model '{id}' disappeared during load"))
                    })?
                    .lifecycle
                    .clone()
            };

            call_lifecycle_method(&lifecycle, "load").await?;

            {
                let mut state = state_rc.borrow_mut();
                let tick = state.next_tick();
                if let Some(e) = state.models.get_mut(&id) {
                    e.loaded = true;
                    e.last_used = Some(tick);
                }
            }

            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Unload a model, freeing its VRAM budget.
    ///
    /// Returns a `Promise<void>` that resolves when the model is unloaded.
    /// Idempotent -- unloading an already-unloaded model is a no-op.
    pub fn unload(&self, id: String) -> js_sys::Promise {
        let state_rc = Rc::clone(&self.state);

        future_to_promise(SendFuture(async move {
            let lifecycle = {
                let state = state_rc.borrow();
                let entry = state.models.get(&id).ok_or_else(|| {
                    JsValue::from_str(&format!("model '{id}' is not registered"))
                })?;

                if !entry.loaded {
                    return Ok(JsValue::UNDEFINED);
                }

                entry.lifecycle.clone()
            };

            call_lifecycle_method(&lifecycle, "unload").await?;

            {
                let mut state = state_rc.borrow_mut();
                if let Some(e) = state.models.get_mut(&id) {
                    e.loaded = false;
                    e.last_used = None;
                }
            }

            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Check whether a model is currently loaded.
    #[wasm_bindgen(js_name = "isLoaded")]
    pub fn is_loaded(&self, id: &str) -> bool {
        let state = self.state.borrow();
        state
            .models
            .get(id)
            .is_some_and(|e| e.loaded)
    }

    /// Total VRAM currently used by loaded models (in bytes).
    #[wasm_bindgen(getter, js_name = "usedBytes")]
    pub fn used_bytes(&self) -> f64 {
        let state = self.state.borrow();
        state.used_bytes()
    }

    /// Available VRAM within the budget (in bytes).
    #[wasm_bindgen(getter, js_name = "availableBytes")]
    pub fn available_bytes(&self) -> f64 {
        let state = self.state.borrow();
        let used = state.used_bytes();
        (state.budget_bytes - used).max(0.0)
    }

    /// The total VRAM budget in bytes.
    #[wasm_bindgen(getter, js_name = "budgetBytes")]
    pub fn budget_bytes(&self) -> f64 {
        let state = self.state.borrow();
        state.budget_bytes
    }

    /// Status of all registered models.
    ///
    /// Returns an array of `{ id, loaded, vramEstimate }` objects.
    pub fn status(&self) -> JsValue {
        let state = self.state.borrow();
        let arr = js_sys::Array::new();

        for (id, entry) in &state.models {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(id));
            let _ = js_sys::Reflect::set(&obj, &"loaded".into(), &JsValue::from_bool(entry.loaded));
            let _ = js_sys::Reflect::set(
                &obj,
                &"vramEstimate".into(),
                &JsValue::from_f64(entry.vram_estimate),
            );
            arr.push(&obj);
        }

        arr.into()
    }
}
