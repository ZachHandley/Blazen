//! Node.js binding for the VRAM-aware model manager.

use napi::Status;
use napi::bindgen_prelude::{BigInt, Promise};
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use std::sync::Arc;

use blazen_llm::{BlazenError, LocalModel};
use blazen_manager::ModelManager;

use crate::providers::completion_model::JsCompletionModel;

// ---------------------------------------------------------------------------
// ThreadsafeFunction type aliases for the JS-callback `LocalModel` adapter.
// ---------------------------------------------------------------------------

/// Lifecycle callback returning `Promise<void>`. Used for `load` and `unload`.
type LifecycleTsfn = ThreadsafeFunction<(), Promise<()>, (), Status, false, true>;

/// Optional `isLoaded()` predicate returning `Promise<boolean>`.
type IsLoadedTsfn = ThreadsafeFunction<(), Promise<bool>, (), Status, false, true>;

/// Configuration for creating a [`JsModelManager`].
///
/// Exactly one of `budgetGb` or `budgetBytes` must be provided.
#[napi(object)]
pub struct ModelManagerConfig {
    /// VRAM budget in gigabytes (e.g. `8.0` for 8 GiB).
    pub budget_gb: Option<f64>,
    /// VRAM budget in bytes (pass as JS `BigInt` to support values >4 GiB).
    pub budget_bytes: Option<BigInt>,
}

/// Status snapshot for a single registered model.
#[napi(object)]
pub struct JsModelStatus {
    /// Model identifier.
    pub id: String,
    /// Whether the model is currently loaded into VRAM.
    pub loaded: bool,
    /// Estimated VRAM footprint in bytes.
    pub vram_estimate: BigInt,
}

/// VRAM budget-aware model manager with LRU eviction.
///
/// Tracks registered local models and their estimated VRAM footprint.
/// When loading a model that would exceed the budget, the least-recently-used
/// loaded model is unloaded first.
///
/// ```javascript
/// const manager = new ModelManager({ budgetGb: 8.0 });
/// await manager.register("llama-7b", model, 4_000_000_000n);  // BigInt
/// await manager.load("llama-7b");
/// ```
#[napi(js_name = "ModelManager")]
pub struct JsModelManager {
    inner: Arc<ModelManager>,
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsModelManager {
    /// Create a new model manager with the given VRAM budget.
    ///
    /// Provide either `budgetGb` (gigabytes as a float) or `budgetBytes`
    /// (exact byte count). If both are given, `budgetGb` takes precedence.
    #[napi(constructor)]
    pub fn new(config: ModelManagerConfig) -> napi::Result<Self> {
        let bytes = match (config.budget_gb, config.budget_bytes) {
            (Some(gb), _) => {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    (gb * 1_073_741_824.0) as u64
                }
            }
            (_, Some(b)) => b.get_u64().1,
            (None, None) => {
                return Err(napi::Error::from_reason(
                    "must provide either budgetGb or budgetBytes",
                ));
            }
        };
        Ok(Self {
            inner: Arc::new(ModelManager::new(bytes)),
        })
    }

    /// Register a `CompletionModel`-backed local model with the manager.
    ///
    /// The model starts in the unloaded state.  An optional
    /// `vramEstimateBytes` overrides the model's self-reported estimate.
    ///
    /// Only local in-process providers (mistral.rs, llama.cpp, candle) can be
    /// registered -- remote HTTP providers will throw. To register an
    /// arbitrary JS-managed resource (embedding model, tokenizer, custom
    /// runtime, …), use [`Self::register_local_model`] instead.
    #[napi]
    pub async fn register(
        &self,
        id: String,
        model: &JsCompletionModel,
        vram_estimate_bytes: Option<BigInt>,
    ) -> napi::Result<()> {
        let local_model = model
            .local_model
            .clone()
            .ok_or_else(|| napi::Error::from_reason("model does not support local loading"))?;
        let vram = vram_estimate_bytes.map_or(0, |b| b.get_u64().1);
        self.inner.register(&id, local_model, vram).await;
        Ok(())
    }

    /// Register an arbitrary JS-managed local model with the manager.
    ///
    /// Unlike [`Self::register`] -- which expects a [`JsCompletionModel`]
    /// backed by an in-process provider -- this entrypoint takes raw
    /// lifecycle callbacks. The manager will invoke `load()` when the model
    /// is brought into VRAM (potentially after evicting an LRU peer) and
    /// `unload()` when it is evicted or explicitly released.
    ///
    /// Both callbacks must return a `Promise<void>` (or be `async`). A
    /// rejection from `load()` aborts the load operation; a rejection from
    /// `unload()` is propagated as a manager error.
    ///
    /// `isLoaded()` is optional: when omitted, the manager's own
    /// loaded-flag bookkeeping is the source of truth.
    /// `vramEstimateBytes` reports the model's footprint so the manager
    /// can enforce the global budget; defaults to `0` when not provided.
    ///
    /// ```javascript
    /// let loaded = false;
    /// await manager.registerLocalModel(
    ///   "my-resource",
    ///   async () => { /* materialize */ loaded = true; },
    ///   async () => { /* release */    loaded = false; },
    ///   async () => loaded,
    ///   2_000_000_000n,
    /// );
    /// ```
    ///
    /// `isLoaded` is `null`-able (pass `null` or `undefined` to omit) and
    /// `vramEstimateBytes` may also be omitted.
    #[napi(js_name = "registerLocalModel")]
    pub async fn register_local_model(
        &self,
        id: String,
        load: LifecycleTsfn,
        unload: LifecycleTsfn,
        is_loaded: Option<IsLoadedTsfn>,
        vram_estimate_bytes: Option<BigInt>,
    ) -> napi::Result<()> {
        let vram = vram_estimate_bytes.map_or(0, |b| b.get_u64().1);
        let adapter = JsLocalModelAdapter {
            id: id.clone(),
            load: Arc::new(load),
            unload: Arc::new(unload),
            is_loaded_fn: is_loaded.map(Arc::new),
            vram_estimate: vram,
        };
        let local_model: Arc<dyn LocalModel> = Arc::new(adapter);
        self.inner.register(&id, local_model, vram).await;
        Ok(())
    }

    /// Load a model, evicting LRU models if the budget would be exceeded.
    ///
    /// Throws if the model is not registered or its VRAM estimate exceeds the
    /// total budget.
    #[napi]
    pub async fn load(&self, id: String) -> napi::Result<()> {
        self.inner
            .load(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Unload a model, freeing its VRAM budget.
    ///
    /// Idempotent: unloading an already-unloaded model is a no-op.
    #[napi]
    pub async fn unload(&self, id: String) -> napi::Result<()> {
        self.inner
            .unload(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Check whether a model is currently loaded.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self, id: String) -> bool {
        self.inner.is_loaded(&id).await
    }

    /// Ensure a model is loaded.
    ///
    /// If already loaded, updates the LRU timestamp. If not loaded, loads it
    /// (potentially evicting other models).
    #[napi(js_name = "ensureLoaded")]
    pub async fn ensure_loaded(&self, id: String) -> napi::Result<()> {
        self.inner
            .ensure_loaded(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Total VRAM currently used by loaded models (in bytes).
    #[napi(js_name = "usedBytes")]
    pub async fn used_bytes(&self) -> BigInt {
        BigInt::from(self.inner.used_bytes().await)
    }

    /// Available VRAM within the budget (in bytes).
    #[napi(js_name = "availableBytes")]
    pub async fn available_bytes(&self) -> BigInt {
        BigInt::from(self.inner.available_bytes().await)
    }

    /// Status of all registered models.
    #[napi]
    pub async fn status(&self) -> Vec<JsModelStatus> {
        self.inner
            .status()
            .await
            .into_iter()
            .map(|s| JsModelStatus {
                id: s.id,
                loaded: s.loaded,
                vram_estimate: BigInt::from(s.vram_estimate),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// JsLocalModelAdapter: bridges JS lifecycle callbacks to `LocalModel`.
// ---------------------------------------------------------------------------

/// Internal adapter that implements [`LocalModel`] by dispatching into the
/// `load` / `unload` / `isLoaded` JS callbacks captured at registration time.
///
/// Mirrors `crates/blazen-wasm-sdk/src/manager.rs::JsLocalModelAdapter`, but
/// uses napi-rs `ThreadsafeFunction`s instead of raw `js_sys::Function`s.
struct JsLocalModelAdapter {
    id: String,
    load: Arc<LifecycleTsfn>,
    unload: Arc<LifecycleTsfn>,
    is_loaded_fn: Option<Arc<IsLoadedTsfn>>,
    vram_estimate: u64,
}

impl std::fmt::Debug for JsLocalModelAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsLocalModelAdapter")
            .field("id", &self.id)
            .field("vram_estimate", &self.vram_estimate)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl LocalModel for JsLocalModelAdapter {
    async fn load(&self) -> Result<(), BlazenError> {
        let promise = self.load.call_async(()).await.map_err(|e| {
            BlazenError::provider(
                "node_local_model",
                format!("model '{}' load() dispatch failed: {e}", self.id),
            )
        })?;
        promise.await.map_err(|e| {
            BlazenError::provider(
                "node_local_model",
                format!("model '{}' load() rejected: {e}", self.id),
            )
        })?;
        Ok(())
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        let promise = self.unload.call_async(()).await.map_err(|e| {
            BlazenError::provider(
                "node_local_model",
                format!("model '{}' unload() dispatch failed: {e}", self.id),
            )
        })?;
        promise.await.map_err(|e| {
            BlazenError::provider(
                "node_local_model",
                format!("model '{}' unload() rejected: {e}", self.id),
            )
        })?;
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        // Without a JS-side `isLoaded()` callback we have no separate state
        // to query: defer to the upstream `ModelManager`'s loaded-flag
        // bookkeeping by conservatively reporting `false`. (Callers should
        // use `JsModelManager::is_loaded`, which queries the manager
        // directly.) Mirrors the WASM adapter's behaviour.
        let Some(ref is_loaded) = self.is_loaded_fn else {
            return false;
        };
        let Ok(promise) = is_loaded.call_async(()).await else {
            return false;
        };
        promise.await.unwrap_or(false)
    }

    async fn vram_bytes(&self) -> Option<u64> {
        Some(self.vram_estimate)
    }
}
