//! Node.js binding for the memory-budget-aware model manager.

use napi::Status;
use napi::bindgen_prelude::{BigInt, Promise};
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use std::collections::HashMap;
use std::sync::Arc;

use blazen_llm::{BlazenError, Device, LocalModel, Pool};
use blazen_manager::ModelManager;

use crate::providers::completion_model::JsCompletionModel;

// ---------------------------------------------------------------------------
// ThreadsafeFunction type aliases for the JS-callback `LocalModel` adapter.
// ---------------------------------------------------------------------------

/// Lifecycle callback returning `Promise<void>`. Used for `load` and `unload`.
type LifecycleTsfn = ThreadsafeFunction<(), Promise<()>, (), Status, false, true>;

/// Optional `isLoaded()` predicate returning `Promise<boolean>`.
type IsLoadedTsfn = ThreadsafeFunction<(), Promise<bool>, (), Status, false, true>;

// ---------------------------------------------------------------------------
// Pool label parsing
// ---------------------------------------------------------------------------

/// Parse a pool label string into a [`Pool`].
///
/// Accepted forms (case-sensitive):
///
/// | Input        | Result            |
/// |--------------|-------------------|
/// | `"cpu"`      | `Pool::Cpu`       |
/// | `"gpu"`      | `Pool::Gpu(0)`    |
/// | `"gpu:N"`    | `Pool::Gpu(N)`    |
fn parse_pool_label(label: &str) -> napi::Result<Pool> {
    if label == "cpu" {
        return Ok(Pool::Cpu);
    }
    if label == "gpu" {
        return Ok(Pool::Gpu(0));
    }
    if let Some(idx_str) = label.strip_prefix("gpu:")
        && let Ok(idx) = idx_str.parse::<usize>()
    {
        return Ok(Pool::Gpu(idx));
    }
    Err(napi::Error::from_reason(format!(
        "invalid pool label '{label}': expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
    )))
}

/// Configuration for creating a [`JsModelManager`].
///
/// Pass either the convenience pair (`cpuRamGb` / `gpuVramGb`) for the
/// common single-GPU desktop layout, or a fully explicit `poolBudgets`
/// map for multi-GPU or custom topologies. Omit everything to receive
/// the unlimited-budget defaults (useful for tests).
#[napi(object)]
pub struct ModelManagerConfig {
    /// Host RAM budget in gigabytes for `Pool::Cpu`.
    #[napi(js_name = "cpuRamGb")]
    pub cpu_ram_gb: Option<f64>,
    /// GPU VRAM budget in gigabytes for `Pool::Gpu(0)`.
    #[napi(js_name = "gpuVramGb")]
    pub gpu_vram_gb: Option<f64>,
    /// Explicit per-pool budget map. Keys: `"cpu"`, `"gpu"`, `"gpu:N"`.
    /// Values: bytes as `BigInt`. When provided, overrides
    /// `cpuRamGb` / `gpuVramGb`.
    #[napi(js_name = "poolBudgets")]
    pub pool_budgets: Option<HashMap<String, BigInt>>,
}

/// Status snapshot for a single registered model.
#[napi(object)]
pub struct JsModelStatus {
    /// Model identifier.
    pub id: String,
    /// Whether the model is currently loaded into its pool.
    pub loaded: bool,
    /// Estimated memory footprint in bytes (host RAM if `pool` is
    /// `"cpu"`, GPU VRAM otherwise).
    #[napi(js_name = "memoryEstimateBytes")]
    pub memory_estimate_bytes: BigInt,
    /// Pool label this model targets (`"cpu"` or `"gpu:N"`).
    pub pool: String,
}

/// Reported per-pool budget pair returned by [`JsModelManager::pools`].
#[napi(object)]
pub struct JsPoolBudget {
    /// Pool label (`"cpu"` or `"gpu:N"`).
    pub pool: String,
    /// Configured budget for the pool in bytes.
    #[napi(js_name = "budgetBytes")]
    pub budget_bytes: BigInt,
}

/// Memory-budget-aware model manager with per-pool LRU eviction.
///
/// Tracks registered local models and their estimated memory footprint.
/// When loading a model that would exceed its pool's budget, the
/// least-recently-used loaded model in the same pool is unloaded first.
///
/// ```javascript
/// // Single-GPU desktop layout:
/// const manager = new ModelManager({ cpuRamGb: 100, gpuVramGb: 24 });
///
/// // Multi-pool layout via explicit BigInt budgets:
/// const manager = new ModelManager({
///   poolBudgets: { "cpu": 100_000_000_000n, "gpu:0": 24_000_000_000n },
/// });
///
/// // No arguments: both `cpu` and `gpu:0` default to the unlimited sentinel.
/// const manager = new ModelManager();
///
/// await manager.register("llama-7b", model, 4_000_000_000n);
/// await manager.load("llama-7b");
/// ```
#[napi(js_name = "ModelManager")]
pub struct JsModelManager {
    inner: Arc<ModelManager>,
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsModelManager {
    /// Create a new model manager with per-pool memory budgets.
    ///
    /// If `poolBudgets` is provided, it is used verbatim (pool labels are
    /// parsed by [`parse_pool_label`]). Otherwise the manager uses
    /// `cpuRamGb` (default `0`) for `Pool::Cpu` and `gpuVramGb`
    /// (default `0`) for `Pool::Gpu(0)`.
    ///
    /// When **all** fields are omitted, both `Pool::Cpu` and
    /// `Pool::Gpu(0)` default to `u64::MAX` — the unlimited-budget
    /// sentinel used by integration tests that don't want to think
    /// about capacity.
    #[napi(constructor)]
    pub fn new(config: Option<ModelManagerConfig>) -> napi::Result<Self> {
        let inner = match config {
            Some(ModelManagerConfig {
                pool_budgets: Some(map),
                ..
            }) => {
                let mut budgets: HashMap<Pool, u64> = HashMap::with_capacity(map.len());
                for (label, value) in map {
                    let pool = parse_pool_label(&label)?;
                    budgets.insert(pool, value.get_u64().1);
                }
                ModelManager::new(budgets)
            }
            Some(ModelManagerConfig {
                cpu_ram_gb,
                gpu_vram_gb,
                pool_budgets: None,
            }) => {
                if cpu_ram_gb.is_none() && gpu_vram_gb.is_none() {
                    let mut budgets: HashMap<Pool, u64> = HashMap::with_capacity(2);
                    budgets.insert(Pool::Cpu, u64::MAX);
                    budgets.insert(Pool::Gpu(0), u64::MAX);
                    ModelManager::new(budgets)
                } else {
                    ModelManager::with_budgets_gb(
                        cpu_ram_gb.unwrap_or(0.0),
                        gpu_vram_gb.unwrap_or(0.0),
                    )
                }
            }
            None => {
                let mut budgets: HashMap<Pool, u64> = HashMap::with_capacity(2);
                budgets.insert(Pool::Cpu, u64::MAX);
                budgets.insert(Pool::Gpu(0), u64::MAX);
                ModelManager::new(budgets)
            }
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Register a `CompletionModel`-backed local model with the manager.
    ///
    /// The model starts in the unloaded state. An optional
    /// `memoryEstimateBytes` overrides the model's self-reported
    /// estimate.
    ///
    /// Only local in-process providers (mistral.rs, llama.cpp, candle)
    /// can be registered — remote HTTP providers will throw. To
    /// register an arbitrary JS-managed resource (embedding model,
    /// tokenizer, custom runtime, …), use
    /// [`Self::register_local_model`] instead.
    #[napi]
    pub async fn register(
        &self,
        id: String,
        model: &JsCompletionModel,
        memory_estimate_bytes: Option<BigInt>,
    ) -> napi::Result<()> {
        let local_model = model
            .local_model
            .clone()
            .ok_or_else(|| napi::Error::from_reason("model does not support local loading"))?;
        let memory = memory_estimate_bytes.map_or(0, |b| b.get_u64().1);
        self.inner.register(&id, local_model, memory).await;
        Ok(())
    }

    /// Register an arbitrary JS-managed local model with the manager.
    ///
    /// Unlike [`Self::register`] — which expects a [`JsCompletionModel`]
    /// backed by an in-process provider — this entrypoint takes raw
    /// lifecycle callbacks. The manager will invoke `load()` when the
    /// model is brought into memory (potentially after evicting an LRU
    /// peer) and `unload()` when it is evicted or explicitly released.
    ///
    /// Both callbacks must return a `Promise<void>` (or be `async`).
    /// A rejection from `load()` aborts the load operation; a rejection
    /// from `unload()` is propagated as a manager error.
    ///
    /// `isLoaded()` is optional: when omitted, the manager's own
    /// loaded-flag bookkeeping is the source of truth.
    /// `memoryEstimateBytes` reports the model's footprint so the
    /// manager can enforce the per-pool budget; defaults to `0` when
    /// not provided. `device` selects which pool the model targets
    /// (`"cpu"`, `"cuda:0"`, `"metal"`, …); defaults to `"cpu"` when
    /// omitted.
    ///
    /// ```javascript
    /// let loaded = false;
    /// await manager.registerLocalModel(
    ///   "my-resource",
    ///   async () => { /* materialize */ loaded = true; },
    ///   async () => { /* release */    loaded = false; },
    ///   async () => loaded,
    ///   2_000_000_000n,
    ///   "cuda:0",
    /// );
    /// ```
    ///
    /// `isLoaded`, `memoryEstimateBytes`, and `device` are all
    /// nullable / optional (pass `null` or `undefined` to omit).
    #[napi(js_name = "registerLocalModel")]
    pub async fn register_local_model(
        &self,
        id: String,
        load: LifecycleTsfn,
        unload: LifecycleTsfn,
        is_loaded: Option<IsLoadedTsfn>,
        memory_estimate_bytes: Option<BigInt>,
        device: Option<String>,
    ) -> napi::Result<()> {
        let memory = memory_estimate_bytes.map_or(0, |b| b.get_u64().1);
        let parsed_device = device
            .as_deref()
            .map_or(Device::Cpu, |s| Device::parse(s).unwrap_or(Device::Cpu));
        let adapter = JsLocalModelAdapter {
            id: id.clone(),
            load: Arc::new(load),
            unload: Arc::new(unload),
            is_loaded_fn: is_loaded.map(Arc::new),
            memory_estimate_bytes: memory,
            device: parsed_device,
        };
        let local_model: Arc<dyn LocalModel> = Arc::new(adapter);
        self.inner.register(&id, local_model, memory).await;
        Ok(())
    }

    /// Load a model, evicting LRU peers in the same pool if the budget
    /// would be exceeded.
    ///
    /// Throws if the model is not registered or its memory estimate
    /// exceeds the pool's total budget.
    #[napi]
    pub async fn load(&self, id: String) -> napi::Result<()> {
        self.inner
            .load(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Unload a model, freeing its slice of the pool budget.
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
    /// If already loaded, updates the LRU timestamp. If not loaded,
    /// loads it (potentially evicting other models in the same pool).
    #[napi(js_name = "ensureLoaded")]
    pub async fn ensure_loaded(&self, id: String) -> napi::Result<()> {
        self.inner
            .ensure_loaded(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Total memory currently used by loaded models in the given pool,
    /// in bytes. Defaults to `"cpu"` when no pool label is provided.
    #[napi(js_name = "usedBytes")]
    pub async fn used_bytes(&self, pool: Option<String>) -> napi::Result<BigInt> {
        let pool = parse_pool_label(pool.as_deref().unwrap_or("cpu"))?;
        Ok(BigInt::from(self.inner.used_bytes(pool).await))
    }

    /// Available memory within the given pool's budget, in bytes.
    /// Defaults to `"cpu"` when no pool label is provided.
    #[napi(js_name = "availableBytes")]
    pub async fn available_bytes(&self, pool: Option<String>) -> napi::Result<BigInt> {
        let pool = parse_pool_label(pool.as_deref().unwrap_or("cpu"))?;
        Ok(BigInt::from(self.inner.available_bytes(pool).await))
    }

    /// Snapshot of the configured per-pool budgets.
    ///
    /// Returns one entry per pool the manager knows about; each entry
    /// carries the pool label (`"cpu"` or `"gpu:N"`) and its budget in
    /// bytes.
    #[napi]
    #[must_use]
    pub fn pools(&self) -> Vec<JsPoolBudget> {
        self.inner
            .pools()
            .into_iter()
            .map(|(p, b)| JsPoolBudget {
                pool: format!("{p}"),
                budget_bytes: BigInt::from(b),
            })
            .collect()
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
                memory_estimate_bytes: BigInt::from(s.memory_estimate_bytes),
                pool: format!("{}", s.pool),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// JsLocalModelAdapter: bridges JS lifecycle callbacks to `LocalModel`.
// ---------------------------------------------------------------------------

/// Internal adapter that implements [`LocalModel`] by dispatching into
/// the `load` / `unload` / `isLoaded` JS callbacks captured at
/// registration time.
///
/// Mirrors `crates/blazen-wasm-sdk/src/manager.rs::JsLocalModelAdapter`,
/// but uses napi-rs `ThreadsafeFunction`s instead of raw
/// `js_sys::Function`s.
struct JsLocalModelAdapter {
    id: String,
    load: Arc<LifecycleTsfn>,
    unload: Arc<LifecycleTsfn>,
    is_loaded_fn: Option<Arc<IsLoadedTsfn>>,
    memory_estimate_bytes: u64,
    device: Device,
}

impl std::fmt::Debug for JsLocalModelAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsLocalModelAdapter")
            .field("id", &self.id)
            .field("memory_estimate_bytes", &self.memory_estimate_bytes)
            .field("device", &self.device)
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
        // Without a JS-side `isLoaded()` callback we have no separate
        // state to query: defer to the upstream `ModelManager`'s
        // loaded-flag bookkeeping by conservatively reporting `false`.
        // (Callers should use `JsModelManager::is_loaded`, which
        // queries the manager directly.) Mirrors the WASM adapter's
        // behaviour.
        let Some(ref is_loaded) = self.is_loaded_fn else {
            return false;
        };
        let Ok(promise) = is_loaded.call_async(()).await else {
            return false;
        };
        promise.await.unwrap_or(false)
    }

    fn device(&self) -> Device {
        self.device
    }

    async fn memory_bytes(&self) -> Option<u64> {
        Some(self.memory_estimate_bytes)
    }
}
