//! Node.js binding for the memory-budget-aware model manager.

use napi::Status;
use napi::bindgen_prelude::{BigInt, FnArgs, Promise};
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use std::collections::HashMap;
use std::sync::Arc;

use blazen_llm::{AdapterHandle, AdapterMountStrategy, BlazenError, Device, LocalModel, Pool};
use blazen_manager::ModelManager;
#[cfg(feature = "hf-loader")]
use blazen_manager::hf_loader::{BackendHint, HfLoadOptions};

use crate::providers::completion_model::JsCompletionModel;

// ---------------------------------------------------------------------------
// ThreadsafeFunction type aliases for the JS-callback `LocalModel` adapter.
// ---------------------------------------------------------------------------

/// Lifecycle callback returning `Promise<void>`. Used for `load` and `unload`.
type LifecycleTsfn = ThreadsafeFunction<(), Promise<()>, (), Status, false, true>;

/// Optional `isLoaded()` predicate returning `Promise<boolean>`.
type IsLoadedTsfn = ThreadsafeFunction<(), Promise<bool>, (), Status, false, true>;

/// Optional `loadAdapter(adapterDir, options)` callback returning
/// `Promise<JsAdapterHandle>`.
type LoadAdapterTsfn = ThreadsafeFunction<
    FnArgs<(String, AdapterOptions)>,
    Promise<JsAdapterHandle>,
    FnArgs<(String, AdapterOptions)>,
    Status,
    false,
    true,
>;

/// Optional `unloadAdapter(handle)` callback returning `Promise<void>`.
type UnloadAdapterTsfn =
    ThreadsafeFunction<JsAdapterHandle, Promise<()>, JsAdapterHandle, Status, false, true>;

/// Optional `listAdapters()` callback returning `Promise<JsAdapterStatus[]>`.
type ListAdaptersTsfn =
    ThreadsafeFunction<(), Promise<Vec<JsAdapterStatus>>, (), Status, false, true>;

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

/// Caller-supplied options when mounting a `LoRA` adapter via
/// [`JsModelManager::load_adapter`].
///
/// Mirrors [`blazen_llm::AdapterOptions`]; `scale` is optional and
/// defaults to `1.0` (full strength, PEFT convention) when omitted.
#[napi(object)]
pub struct AdapterOptions {
    /// Caller-chosen identifier for this adapter mount. Must be unique
    /// per `(model, adapter)` pair within a manager.
    #[napi(js_name = "adapterId")]
    pub adapter_id: String,
    /// Scaling factor applied to the adapter's delta-weights. Defaults
    /// to `1.0` when not provided.
    pub scale: Option<f64>,
}

/// Handle returned by [`JsModelManager::load_adapter`] and accepted by
/// JS-side `unloadAdapter` lifecycle callbacks (see
/// [`JsModelManager::register_local_model`]).
///
/// Mirrors [`blazen_llm::AdapterHandle`]; `mountStrategy` is one of
/// `"attached"`, `"rebuilt"`, or `"merged"`.
#[napi(object)]
pub struct JsAdapterHandle {
    /// Echoes [`AdapterOptions::adapter_id`].
    #[napi(js_name = "adapterId")]
    pub adapter_id: String,
    /// Bytes the adapter occupies on top of the base model.
    #[napi(js_name = "memoryBytes")]
    pub memory_bytes: BigInt,
    /// One of `"attached"`, `"rebuilt"`, or `"merged"` — what the
    /// backend actually did to honor the mount request.
    #[napi(js_name = "mountStrategy")]
    pub mount_strategy: String,
}

/// Snapshot of one mounted adapter, returned by
/// [`JsModelManager::list_adapters`]. Mirrors [`blazen_llm::AdapterStatus`].
#[napi(object)]
pub struct JsAdapterStatus {
    /// Caller-supplied adapter identifier.
    #[napi(js_name = "adapterId")]
    pub adapter_id: String,
    /// Scaling factor applied at mount time.
    pub scale: f64,
    /// Absolute filesystem path to the adapter directory.
    #[napi(js_name = "sourceDir")]
    pub source_dir: String,
    /// Bytes the adapter occupies on top of the base model.
    #[napi(js_name = "memoryBytes")]
    pub memory_bytes: BigInt,
}

/// Local-inference backend identifier returned by
/// [`JsModelManager::load_from_hf`] and accepted as a forced override on
/// [`JsHfLoadOptions::backend_hint`].
#[cfg(feature = "hf-loader")]
#[napi(string_enum)]
pub enum JsBackendHint {
    /// `mistral.rs` — broad architecture coverage, handles both safetensors
    /// and GGUF, supports vision/multimodal models.
    #[allow(non_camel_case_types)]
    mistralrs,
    /// `candle` — pure-Rust, supports safetensors and GGUF for the subset of
    /// architectures candle ships.
    #[allow(non_camel_case_types)]
    candle,
    /// `llama.cpp` — GGUF only, best CPU performance and lowest memory.
    #[allow(non_camel_case_types)]
    llamacpp,
}

#[cfg(feature = "hf-loader")]
impl From<JsBackendHint> for BackendHint {
    fn from(h: JsBackendHint) -> Self {
        match h {
            JsBackendHint::mistralrs => Self::Mistralrs,
            JsBackendHint::candle => Self::Candle,
            JsBackendHint::llamacpp => Self::Llamacpp,
        }
    }
}

#[cfg(feature = "hf-loader")]
impl From<BackendHint> for JsBackendHint {
    fn from(h: BackendHint) -> Self {
        match h {
            BackendHint::Mistralrs => Self::mistralrs,
            BackendHint::Candle => Self::candle,
            BackendHint::Llamacpp => Self::llamacpp,
        }
    }
}

/// Caller-supplied options for [`JsModelManager::load_from_hf`].
///
/// Mirrors [`blazen_manager::HfLoadOptions`]; every field is optional.
#[cfg(feature = "hf-loader")]
#[napi(object)]
pub struct JsHfLoadOptions {
    /// Force a specific backend; skips engine inference but still probes
    /// the repo for memory sizing.
    #[napi(js_name = "backendHint")]
    pub backend_hint: Option<JsBackendHint>,
    /// Git revision (branch, tag, or commit sha). Defaults to the repo's
    /// default branch.
    pub revision: Option<String>,
    /// Hugging Face access token. When omitted, falls back to the
    /// `HF_TOKEN` environment variable, then to anonymous access.
    #[napi(js_name = "hfToken")]
    pub hf_token: Option<String>,
    /// Override the on-disk cache directory used by `hf-hub`.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
    /// Device specifier forwarded to the chosen provider (`"cpu"`,
    /// `"cuda:0"`, `"metal"`, …).
    pub device: Option<String>,
    /// Explicit GGUF filename for repos that ship multiple quantizations.
    #[napi(js_name = "ggufFile")]
    pub gguf_file: Option<String>,
    /// Override the auto-derived memory estimate, in bytes.
    #[napi(js_name = "memoryEstimateBytes")]
    pub memory_estimate_bytes: Option<BigInt>,
    /// Pool label (`"cpu"`, `"gpu"`, `"gpu:N"`). Defaults to `"cpu"`.
    pub pool: Option<String>,
}

fn mount_strategy_label(strategy: AdapterMountStrategy) -> &'static str {
    match strategy {
        AdapterMountStrategy::Attached => "attached",
        AdapterMountStrategy::Rebuilt => "rebuilt",
        AdapterMountStrategy::Merged => "merged",
    }
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
#[allow(
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation
)]
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
    /// `isLoaded`, `memoryEstimateBytes`, `device`, `loadAdapter`,
    /// `unloadAdapter`, and `listAdapters` are all nullable / optional
    /// (pass `null` or `undefined` to omit). Omitted adapter callbacks
    /// cause [`JsModelManager::load_adapter`] / `unloadAdapter` /
    /// `listAdapters` to surface the upstream "backend does not support
    /// `LoRA` adapters" error for this model.
    #[napi(js_name = "registerLocalModel")]
    pub async fn register_local_model(
        &self,
        id: String,
        load: LifecycleTsfn,
        unload: LifecycleTsfn,
        is_loaded: Option<IsLoadedTsfn>,
        memory_estimate_bytes: Option<BigInt>,
        device: Option<String>,
        load_adapter: Option<LoadAdapterTsfn>,
        unload_adapter: Option<UnloadAdapterTsfn>,
        list_adapters: Option<ListAdaptersTsfn>,
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
            load_adapter_fn: load_adapter.map(Arc::new),
            unload_adapter_fn: unload_adapter.map(Arc::new),
            list_adapters_fn: list_adapters.map(Arc::new),
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

    /// Auto-detect the right local-inference backend for a Hugging Face
    /// repo, then register and budget the model with this manager.
    ///
    /// Performs a single metadata request against the Hub to enumerate
    /// the repo's siblings, picks a backend (mistral.rs / candle /
    /// llama.cpp) per the rules documented on
    /// [`blazen_manager::hf_loader::choose_backend`], computes a memory
    /// estimate from the sibling sizes, and registers the model under
    /// `id`. The model starts unloaded — call [`Self::load`] or
    /// [`Self::ensure_loaded`] to materialize it.
    ///
    /// Returns the chosen backend as a lower-case string
    /// (`"mistralrs"` / `"candle"` / `"llamacpp"`).
    ///
    /// Throws on empty repo id, gated/missing repo, PEFT-adapter-only
    /// repo (use [`Self::load_adapter`] instead), missing backend
    /// feature, or any provider construction failure.
    #[cfg(feature = "hf-loader")]
    #[napi(js_name = "loadFromHf")]
    pub async fn load_from_hf(
        &self,
        id: String,
        repo: String,
        options: Option<JsHfLoadOptions>,
    ) -> napi::Result<String> {
        let opts = options.unwrap_or(JsHfLoadOptions {
            backend_hint: None,
            revision: None,
            hf_token: None,
            cache_dir: None,
            device: None,
            gguf_file: None,
            memory_estimate_bytes: None,
            pool: None,
        });
        let pool = match opts.pool.as_deref() {
            Some(label) => Some(parse_pool_label(label)?),
            None => None,
        };
        let rust_opts = HfLoadOptions {
            backend_hint: opts.backend_hint.map(BackendHint::from),
            revision: opts.revision,
            hf_token: opts.hf_token,
            cache_dir: opts.cache_dir.map(std::path::PathBuf::from),
            device: opts.device,
            gguf_file: opts.gguf_file,
            memory_estimate_bytes: opts.memory_estimate_bytes.map(|b| b.get_u64().1),
            pool,
        };
        let backend = self
            .inner
            .load_from_hf(id, &repo, rust_opts)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(backend.as_str().to_string())
    }

    /// Mount a PEFT-format `LoRA` adapter onto a registered model.
    ///
    /// `adapterDir` must contain the canonical PEFT layout
    /// (`adapter_model.safetensors` + `adapter_config.json`). The base
    /// model is implicitly loaded (`ensureLoaded`) before mounting.
    ///
    /// Returns the adapter id assigned by the backend (echoes
    /// `options.adapterId`).
    ///
    /// Throws if the model is not registered, the adapter id is already
    /// mounted, the pool budget would be exceeded, or the backend does
    /// not support adapters.
    #[napi(js_name = "loadAdapter")]
    pub async fn load_adapter(
        &self,
        model_id: String,
        adapter_dir: String,
        options: AdapterOptions,
    ) -> napi::Result<String> {
        let llm_options = blazen_llm::AdapterOptions {
            adapter_id: options.adapter_id,
            scale: options.scale.map_or(1.0_f32, |s| s as f32),
        };
        let path = std::path::PathBuf::from(adapter_dir);
        let handle = self
            .inner
            .load_adapter(&model_id, &path, llm_options)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(handle.adapter_id)
    }

    /// Unmount a previously-loaded adapter from a registered model.
    ///
    /// Throws if the model is not registered or the adapter id is not
    /// currently mounted on it.
    #[napi(js_name = "unloadAdapter")]
    pub async fn unload_adapter(&self, model_id: String, adapter_id: String) -> napi::Result<()> {
        self.inner
            .unload_adapter(&model_id, &adapter_id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// List adapters currently mounted on a registered model.
    ///
    /// Throws if the model is not registered.
    #[napi(js_name = "listAdapters")]
    pub async fn list_adapters(&self, model_id: String) -> napi::Result<Vec<JsAdapterStatus>> {
        let statuses = self
            .inner
            .list_adapters(&model_id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(statuses
            .into_iter()
            .map(|s| JsAdapterStatus {
                adapter_id: s.adapter_id,
                scale: f64::from(s.scale),
                source_dir: s.source_dir.to_string_lossy().into_owned(),
                memory_bytes: BigInt::from(s.memory_bytes),
            })
            .collect())
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
    load_adapter_fn: Option<Arc<LoadAdapterTsfn>>,
    unload_adapter_fn: Option<Arc<UnloadAdapterTsfn>>,
    list_adapters_fn: Option<Arc<ListAdaptersTsfn>>,
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
#[allow(clippy::cast_possible_truncation)]
impl LocalModel for JsLocalModelAdapter {
    async fn load(&self) -> Result<(), BlazenError> {
        let promise = self.load.call_async_catch(()).await.map_err(|e| {
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
        let promise = self.unload.call_async_catch(()).await.map_err(|e| {
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
        let Ok(promise) = is_loaded.call_async_catch(()).await else {
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

    async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        options: blazen_llm::AdapterOptions,
    ) -> Result<AdapterHandle, BlazenError> {
        let Some(ref load_adapter) = self.load_adapter_fn else {
            return Err(BlazenError::unsupported(
                "JS lifecycle does not implement loadAdapter",
            ));
        };
        let js_options = AdapterOptions {
            adapter_id: options.adapter_id,
            scale: Some(f64::from(options.scale)),
        };
        let adapter_dir_str = adapter_dir.to_string_lossy().into_owned();
        let promise = load_adapter
            .call_async_catch(FnArgs::from((adapter_dir_str, js_options)))
            .await
            .map_err(|e| {
                BlazenError::provider(
                    "node_local_model",
                    format!("model '{}' loadAdapter() dispatch failed: {e}", self.id),
                )
            })?;
        let js_handle = promise.await.map_err(|e| {
            BlazenError::provider(
                "node_local_model",
                format!("model '{}' loadAdapter() rejected: {e}", self.id),
            )
        })?;
        let mount_strategy = match js_handle.mount_strategy.as_str() {
            "attached" => AdapterMountStrategy::Attached,
            "rebuilt" => AdapterMountStrategy::Rebuilt,
            "merged" => AdapterMountStrategy::Merged,
            other => {
                return Err(BlazenError::provider(
                    "node_local_model",
                    format!(
                        "model '{}' loadAdapter() returned unknown mountStrategy '{other}'; \
                         expected 'attached', 'rebuilt', or 'merged'",
                        self.id,
                    ),
                ));
            }
        };
        Ok(AdapterHandle {
            adapter_id: js_handle.adapter_id,
            memory_bytes: js_handle.memory_bytes.get_u64().1,
            mount_strategy,
        })
    }

    async fn unload_adapter(&self, handle: &AdapterHandle) -> Result<(), BlazenError> {
        let Some(ref unload_adapter) = self.unload_adapter_fn else {
            return Err(BlazenError::unsupported(
                "JS lifecycle does not implement unloadAdapter",
            ));
        };
        let js_handle = JsAdapterHandle {
            adapter_id: handle.adapter_id.clone(),
            memory_bytes: BigInt::from(handle.memory_bytes),
            mount_strategy: mount_strategy_label(handle.mount_strategy).to_string(),
        };
        let promise = unload_adapter
            .call_async_catch(js_handle)
            .await
            .map_err(|e| {
                BlazenError::provider(
                    "node_local_model",
                    format!("model '{}' unloadAdapter() dispatch failed: {e}", self.id),
                )
            })?;
        promise.await.map_err(|e| {
            BlazenError::provider(
                "node_local_model",
                format!("model '{}' unloadAdapter() rejected: {e}", self.id),
            )
        })?;
        Ok(())
    }

    async fn list_adapters(&self) -> Vec<blazen_llm::AdapterStatus> {
        let Some(ref list_adapters) = self.list_adapters_fn else {
            return Vec::new();
        };
        let Ok(promise) = list_adapters.call_async_catch(()).await else {
            return Vec::new();
        };
        let Ok(statuses) = promise.await else {
            return Vec::new();
        };
        statuses
            .into_iter()
            .map(|s| blazen_llm::AdapterStatus {
                adapter_id: s.adapter_id,
                scale: s.scale as f32,
                source_dir: std::path::PathBuf::from(s.source_dir),
                memory_bytes: s.memory_bytes.get_u64().1,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Training surface (feature = "training")
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
pub use training::{
    JsDpoConfig, JsFullFineTuneConfig, JsFullFineTuneResult, JsJsonlDataset, JsKtoConfig,
    JsLoraConfig, JsMixedPrecision, JsOptimConfig, JsOrpoConfig, JsPreferenceJsonlDataset,
    JsRatedJsonlDataset, JsSchedulerConfig, JsSchedulerKind, JsSimpoConfig, JsTrainConfig,
    JsTrainCoreConfig, JsTrainedAdapter, JsTrainingEvent,
};

#[cfg(feature = "training")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::needless_pass_by_value
)]
mod training {
    use std::path::PathBuf;
    use std::sync::Arc;

    use async_trait::async_trait;
    use napi::Status;
    use napi::bindgen_prelude::BigInt;
    use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
    use napi_derive::napi;

    use blazen_train::dataset::{JsonlDataset, PreferenceJsonlDataset, RatedJsonlDataset};
    use blazen_train::{
        BlazenTrainError, DpoConfig, FullFineTuneConfig, FullFineTuneResult, KtoConfig, LoraConfig,
        MixedPrecision, OptimConfig, OrpoConfig, PreferenceDataset, RatedDataset, SchedulerConfig,
        SchedulerKind, SimpoConfig, TrainConfig, TrainCoreConfig, TrainedAdapter, TrainingBatch,
        TrainingDataset, TrainingEvent, TrainingProgress,
    };
    use tokenizers::Tokenizer;

    use super::JsModelManager;

    // -----------------------------------------------------------------------
    // JsSchedulerKind / JsMixedPrecision
    // -----------------------------------------------------------------------

    /// Learning-rate schedule shape passed to [`JsSchedulerConfig`].
    #[napi(string_enum = "lowercase")]
    pub enum JsSchedulerKind {
        Constant,
        Linear,
        Cosine,
    }

    impl From<JsSchedulerKind> for SchedulerKind {
        fn from(k: JsSchedulerKind) -> Self {
            match k {
                JsSchedulerKind::Constant => Self::Constant,
                JsSchedulerKind::Linear => Self::Linear,
                JsSchedulerKind::Cosine => Self::Cosine,
            }
        }
    }

    /// Mixed-precision mode passed to [`JsTrainConfig`].
    #[napi(string_enum = "lowercase")]
    pub enum JsMixedPrecision {
        None,
        Bf16,
    }

    impl From<JsMixedPrecision> for MixedPrecision {
        fn from(m: JsMixedPrecision) -> Self {
            match m {
                JsMixedPrecision::None => Self::None,
                JsMixedPrecision::Bf16 => Self::Bf16,
            }
        }
    }

    // -----------------------------------------------------------------------
    // JsLoraConfig / JsOptimConfig / JsSchedulerConfig / JsTrainConfig
    // -----------------------------------------------------------------------

    /// LoRA hyperparameters.
    #[napi(object)]
    pub struct JsLoraConfig {
        /// Low-rank dimension (PEFT "r"). Default `16`.
        pub rank: Option<u32>,
        /// Scaling numerator; effective per-layer scale is `alpha / rank`. Default `32`.
        pub alpha: Option<f64>,
        /// Dropout applied to LoRA-A input. Default `0.0`.
        pub dropout: Option<f64>,
        /// Module-name suffixes to inject LoRA into. Default
        /// `["q_proj","k_proj","v_proj","o_proj"]`.
        #[napi(js_name = "targetModules")]
        pub target_modules: Option<Vec<String>>,
    }

    fn default_target_modules() -> Vec<String> {
        vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ]
    }

    impl TryFrom<JsLoraConfig> for LoraConfig {
        type Error = napi::Error;

        fn try_from(c: JsLoraConfig) -> Result<Self, Self::Error> {
            let rank = c.rank.unwrap_or(16) as usize;
            if rank == 0 {
                return Err(napi::Error::from_reason("LoraConfig.rank must be > 0"));
            }
            let alpha = c.alpha.unwrap_or(32.0);
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(napi::Error::from_reason("LoraConfig.alpha must be > 0"));
            }
            let dropout = c.dropout.unwrap_or(0.0);
            if !(0.0..1.0).contains(&dropout) {
                return Err(napi::Error::from_reason(
                    "LoraConfig.dropout must be in [0.0, 1.0)",
                ));
            }
            let target_modules = c.target_modules.unwrap_or_else(default_target_modules);
            if target_modules.is_empty() {
                return Err(napi::Error::from_reason(
                    "LoraConfig.targetModules must be non-empty",
                ));
            }
            Ok(Self {
                rank,
                alpha: alpha as f32,
                dropout: dropout as f32,
                target_modules,
            })
        }
    }

    /// AdamW optimizer hyperparameters.
    #[napi(object)]
    pub struct JsOptimConfig {
        /// Peak learning rate (applied at end of warmup). Default `2e-4`.
        #[napi(js_name = "learningRate")]
        pub learning_rate: Option<f64>,
        /// AdamW beta1. Default `0.9`.
        pub beta1: Option<f64>,
        /// AdamW beta2. Default `0.999`.
        pub beta2: Option<f64>,
        /// AdamW numerical-stability epsilon. Default `1e-8`.
        pub epsilon: Option<f64>,
        /// Decoupled weight decay. Default `0.0`.
        #[napi(js_name = "weightDecay")]
        pub weight_decay: Option<f64>,
        /// Global gradient L2-norm clip; `null` disables clipping. Default `1.0`.
        #[napi(js_name = "gradientClip")]
        pub gradient_clip: Option<f64>,
    }

    impl TryFrom<JsOptimConfig> for OptimConfig {
        type Error = napi::Error;

        fn try_from(c: JsOptimConfig) -> Result<Self, Self::Error> {
            let learning_rate = c.learning_rate.unwrap_or(2e-4);
            if !learning_rate.is_finite() || learning_rate <= 0.0 {
                return Err(napi::Error::from_reason(
                    "OptimConfig.learningRate must be > 0",
                ));
            }
            let beta1 = c.beta1.unwrap_or(0.9);
            let beta2 = c.beta2.unwrap_or(0.999);
            if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
                return Err(napi::Error::from_reason(
                    "OptimConfig.beta1 / beta2 must be in [0.0, 1.0)",
                ));
            }
            let epsilon = c.epsilon.unwrap_or(1e-8);
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(napi::Error::from_reason("OptimConfig.epsilon must be > 0"));
            }
            let weight_decay = c.weight_decay.unwrap_or(0.0);
            if !weight_decay.is_finite() || weight_decay < 0.0 {
                return Err(napi::Error::from_reason(
                    "OptimConfig.weightDecay must be >= 0",
                ));
            }
            let gradient_clip = match c.gradient_clip {
                Some(g) => {
                    if !g.is_finite() || g <= 0.0 {
                        return Err(napi::Error::from_reason(
                            "OptimConfig.gradientClip, when set, must be > 0",
                        ));
                    }
                    Some(g as f32)
                }
                None => Some(1.0_f32),
            };
            Ok(Self {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                gradient_clip,
            })
        }
    }

    /// Learning-rate scheduler configuration.
    #[napi(object)]
    pub struct JsSchedulerConfig {
        /// Schedule shape. Default `Cosine`.
        pub kind: Option<JsSchedulerKind>,
        /// Linear-warmup duration in steps applied before the main shape. Default `0`.
        #[napi(js_name = "warmupSteps")]
        pub warmup_steps: Option<u32>,
    }

    impl From<JsSchedulerConfig> for SchedulerConfig {
        fn from(c: JsSchedulerConfig) -> Self {
            Self {
                kind: c.kind.map_or(SchedulerKind::Cosine, Into::into),
                warmup_steps: c.warmup_steps.unwrap_or(0) as usize,
            }
        }
    }

    /// Full configuration for one training run.
    #[napi(object)]
    pub struct JsTrainConfig {
        /// HuggingFace repo id of the base model.
        #[napi(js_name = "baseModelRepo")]
        pub base_model_repo: String,
        /// Filesystem directory where the trained adapter and checkpoints land.
        #[napi(js_name = "outputDir")]
        pub output_dir: String,
        pub lora: Option<JsLoraConfig>,
        pub optim: Option<JsOptimConfig>,
        pub scheduler: Option<JsSchedulerConfig>,
        /// Total optimizer steps to run. Default `1000`.
        #[napi(js_name = "maxSteps")]
        pub max_steps: Option<u32>,
        /// Micro-batch size (per forward pass). Default `4`.
        #[napi(js_name = "batchSize")]
        pub batch_size: Option<u32>,
        /// Micro-batches accumulated before each optimizer step. Default `1`.
        #[napi(js_name = "gradientAccumulationSteps")]
        pub gradient_accumulation_steps: Option<u32>,
        /// Maximum tokenized sequence length per example. Default `2048`.
        #[napi(js_name = "maxSeqLen")]
        pub max_seq_len: Option<u32>,
        /// Run evaluation every N steps when set.
        #[napi(js_name = "evalSteps")]
        pub eval_steps: Option<u32>,
        /// Write a checkpoint every N steps when set.
        #[napi(js_name = "saveSteps")]
        pub save_steps: Option<u32>,
        /// RNG seed (dataset shuffling + LoRA `A` init). Default `42`.
        pub seed: Option<BigInt>,
        /// Mixed-precision mode for forward / backward. Default `Bf16`.
        #[napi(js_name = "mixedPrecision")]
        pub mixed_precision: Option<JsMixedPrecision>,
        /// Device string forwarded to the trainer (`"cpu"`, `"cuda:0"`, `"metal"`).
        pub device: Option<String>,
    }

    impl TryFrom<JsTrainConfig> for TrainConfig {
        type Error = napi::Error;

        fn try_from(c: JsTrainConfig) -> Result<Self, Self::Error> {
            if c.base_model_repo.trim().is_empty() {
                return Err(napi::Error::from_reason(
                    "TrainConfig.baseModelRepo must be non-empty",
                ));
            }
            if c.output_dir.trim().is_empty() {
                return Err(napi::Error::from_reason(
                    "TrainConfig.outputDir must be non-empty",
                ));
            }
            let max_steps = c.max_steps.unwrap_or(1000) as usize;
            if max_steps == 0 {
                return Err(napi::Error::from_reason("TrainConfig.maxSteps must be > 0"));
            }
            let batch_size = c.batch_size.unwrap_or(4) as usize;
            if batch_size == 0 {
                return Err(napi::Error::from_reason(
                    "TrainConfig.batchSize must be > 0",
                ));
            }
            let gradient_accumulation_steps = c.gradient_accumulation_steps.unwrap_or(1) as usize;
            if gradient_accumulation_steps == 0 {
                return Err(napi::Error::from_reason(
                    "TrainConfig.gradientAccumulationSteps must be > 0",
                ));
            }
            let max_seq_len = c.max_seq_len.unwrap_or(2048) as usize;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "TrainConfig.maxSeqLen must be > 0",
                ));
            }
            let lora = c
                .lora
                .map(LoraConfig::try_from)
                .transpose()?
                .unwrap_or_else(|| LoraConfig {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.0,
                    target_modules: default_target_modules(),
                });
            let optim = c
                .optim
                .map(OptimConfig::try_from)
                .transpose()?
                .unwrap_or(OptimConfig {
                    learning_rate: 2e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.0,
                    gradient_clip: Some(1.0),
                });
            let scheduler = c.scheduler.map_or(
                SchedulerConfig {
                    kind: SchedulerKind::Cosine,
                    warmup_steps: 0,
                },
                SchedulerConfig::from,
            );
            Ok(Self {
                base_model_repo: c.base_model_repo,
                output_dir: PathBuf::from(c.output_dir),
                lora,
                optim,
                scheduler,
                max_steps,
                batch_size,
                gradient_accumulation_steps,
                max_seq_len,
                eval_steps: c.eval_steps.map(|v| v as usize),
                save_steps: c.save_steps.map(|v| v as usize),
                seed: c.seed.map_or(42, |b| b.get_u64().1),
                mixed_precision: c.mixed_precision.map_or(MixedPrecision::Bf16, Into::into),
                device: c.device,
            })
        }
    }

    // -----------------------------------------------------------------------
    // JsTrainedAdapter
    // -----------------------------------------------------------------------

    /// Result of a completed training run.
    #[napi(object)]
    pub struct JsTrainedAdapter {
        /// Directory the PEFT-format adapter was written to.
        #[napi(js_name = "adapterDir")]
        pub adapter_dir: String,
        /// Final training loss.
        #[napi(js_name = "finalLoss")]
        pub final_loss: f64,
        /// Total optimizer steps executed.
        #[napi(js_name = "totalSteps")]
        pub total_steps: BigInt,
    }

    impl From<TrainedAdapter> for JsTrainedAdapter {
        fn from(a: TrainedAdapter) -> Self {
            Self {
                adapter_dir: a.adapter_dir.display().to_string(),
                final_loss: f64::from(a.final_loss),
                total_steps: BigInt::from(a.total_steps as u64),
            }
        }
    }

    // -----------------------------------------------------------------------
    // JsTrainingEvent (flat discriminated record)
    // -----------------------------------------------------------------------

    /// One observable event emitted during a training run.
    ///
    /// Switch on `kind` (`"started"` / `"stepCompleted"` / `"evaluating"` /
    /// `"evalCompleted"` / `"checkpointSaved"` / `"finished"`); other fields
    /// carry the per-variant payload and are absent for variants that do not
    /// populate them.
    #[napi(object)]
    pub struct JsTrainingEvent {
        pub kind: String,
        pub step: Option<BigInt>,
        pub loss: Option<f64>,
        #[napi(js_name = "learningRate")]
        pub learning_rate: Option<f64>,
        #[napi(js_name = "elapsedMs")]
        pub elapsed_ms: Option<f64>,
        #[napi(js_name = "totalSteps")]
        pub total_steps: Option<BigInt>,
        #[napi(js_name = "evalLoss")]
        pub eval_loss: Option<f64>,
        #[napi(js_name = "checkpointPath")]
        pub checkpoint_path: Option<String>,
        #[napi(js_name = "adapterDir")]
        pub adapter_dir: Option<String>,
        #[napi(js_name = "finalLoss")]
        pub final_loss: Option<f64>,
    }

    impl JsTrainingEvent {
        fn empty(kind: &'static str) -> Self {
            Self {
                kind: kind.to_string(),
                step: None,
                loss: None,
                learning_rate: None,
                elapsed_ms: None,
                total_steps: None,
                eval_loss: None,
                checkpoint_path: None,
                adapter_dir: None,
                final_loss: None,
            }
        }
    }

    impl From<TrainingEvent> for JsTrainingEvent {
        fn from(ev: TrainingEvent) -> Self {
            match ev {
                TrainingEvent::Started { total_steps } => Self {
                    total_steps: Some(BigInt::from(total_steps as u64)),
                    ..Self::empty("started")
                },
                TrainingEvent::StepCompleted {
                    step,
                    loss,
                    learning_rate,
                    elapsed,
                } => Self {
                    step: Some(BigInt::from(step as u64)),
                    loss: Some(f64::from(loss)),
                    learning_rate: Some(learning_rate),
                    elapsed_ms: Some(elapsed.as_secs_f64() * 1000.0),
                    ..Self::empty("stepCompleted")
                },
                TrainingEvent::Evaluating { step } => Self {
                    step: Some(BigInt::from(step as u64)),
                    ..Self::empty("evaluating")
                },
                TrainingEvent::EvalCompleted { step, eval_loss } => Self {
                    step: Some(BigInt::from(step as u64)),
                    eval_loss: Some(f64::from(eval_loss)),
                    ..Self::empty("evalCompleted")
                },
                TrainingEvent::CheckpointSaved { step, path } => Self {
                    step: Some(BigInt::from(step as u64)),
                    checkpoint_path: Some(path.display().to_string()),
                    ..Self::empty("checkpointSaved")
                },
                TrainingEvent::Finished {
                    final_loss,
                    total_steps,
                    adapter_dir,
                } => Self {
                    total_steps: Some(BigInt::from(total_steps as u64)),
                    final_loss: Some(f64::from(final_loss)),
                    adapter_dir: Some(adapter_dir.display().to_string()),
                    ..Self::empty("finished")
                },
            }
        }
    }

    // -----------------------------------------------------------------------
    // JsJsonlDataset (opaque)
    // -----------------------------------------------------------------------

    /// Optional knobs for [`JsJsonlDataset::from_path`].
    #[napi(object)]
    pub struct JsJsonlDatasetOptions {
        /// Jinja2 chat template (from `tokenizer_config.json`). Required when
        /// rows use the `messages` shape.
        #[napi(js_name = "chatTemplate")]
        pub chat_template: Option<String>,
        /// Maximum tokenized sequence length per example. Default `2048`.
        #[napi(js_name = "maxSeqLen")]
        pub max_seq_len: Option<u32>,
        /// Candle device string. Default `"cpu"`.
        pub device: Option<String>,
        /// Token id to write into padded positions. Default `0`.
        #[napi(js_name = "padTokenId")]
        pub pad_token_id: Option<u32>,
    }

    /// JSONL-backed training dataset.
    ///
    /// Each line of the input file must deserialize to either
    /// `{"messages": [{"role": ..., "content": ...}, ...]}` (OpenAI shape)
    /// or `{"prompt": "...", "completion": "..."}` (legacy SFT).
    #[napi(js_name = "JsonlDataset")]
    pub struct JsJsonlDataset {
        pub(super) inner: Arc<JsonlDataset>,
    }

    #[napi]
    impl JsJsonlDataset {
        /// Load a JSONL training file using the tokenizer at `tokenizerPath`.
        ///
        /// # Errors
        ///
        /// Throws if the tokenizer cannot be loaded, the device string is
        /// invalid, or the JSONL file fails to parse.
        #[napi(factory, js_name = "fromPath")]
        pub fn from_path(
            path: String,
            tokenizer_path: String,
            opts: Option<JsJsonlDatasetOptions>,
        ) -> napi::Result<Self> {
            let opts = opts.unwrap_or(JsJsonlDatasetOptions {
                chat_template: None,
                max_seq_len: None,
                device: None,
                pad_token_id: None,
            });
            let max_seq_len = opts.max_seq_len.unwrap_or(2048) as usize;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "JsonlDataset.maxSeqLen must be > 0",
                ));
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                napi::Error::from_reason(format!(
                    "failed to load tokenizer from {tokenizer_path:?}: {e}"
                ))
            })?;
            let device_str = opts.device.unwrap_or_else(|| String::from("cpu"));
            let cdev = parse_train_device_node(&device_str)?;
            let ds = JsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                opts.chat_template.as_deref(),
                max_seq_len,
                cdev,
                opts.pad_token_id.unwrap_or(0),
            )
            .map_err(|e| napi::Error::from_reason(format!("JsonlDataset load failed: {e}")))?;
            Ok(Self {
                inner: Arc::new(ds),
            })
        }
    }

    fn parse_train_device_node(device: &str) -> napi::Result<candle_core::Device> {
        let trimmed = device.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower == "cpu" {
            return Ok(candle_core::Device::Cpu);
        }
        if let Some(idx_str) = lower.strip_prefix("cuda:") {
            let idx: usize = idx_str.parse().map_err(|_| {
                napi::Error::from_reason(format!(
                    "invalid CUDA device {trimmed:?}: expected 'cuda:N'"
                ))
            })?;
            return candle_core::Device::new_cuda(idx).map_err(|e| {
                napi::Error::from_reason(format!("failed to open CUDA device {trimmed:?}: {e}"))
            });
        }
        if lower == "cuda" {
            return candle_core::Device::new_cuda(0)
                .map_err(|e| napi::Error::from_reason(format!("failed to open cuda:0: {e}")));
        }
        if lower == "metal" {
            return candle_core::Device::new_metal(0)
                .map_err(|e| napi::Error::from_reason(format!("failed to open metal:0: {e}")));
        }
        Err(napi::Error::from_reason(format!(
            "unrecognized device {trimmed:?}: expected 'cpu', 'cuda', 'cuda:N', or 'metal'"
        )))
    }

    // -----------------------------------------------------------------------
    // Bridges: dataset (Arc-wrapped) + progress callback
    // -----------------------------------------------------------------------

    struct ArcDataset(Arc<JsonlDataset>);

    #[async_trait]
    impl TrainingDataset for ArcDataset {
        fn len(&self) -> usize {
            self.0.len()
        }

        async fn batch(
            &self,
            batch_size: usize,
            idx: usize,
        ) -> Result<TrainingBatch, BlazenTrainError> {
            self.0.batch(batch_size, idx).await
        }
    }

    /// TSFN type for the `(event) => void` progress callback. The return
    /// value is ignored; only the queueing status is observable from Rust.
    type ProgressTsfn =
        ThreadsafeFunction<JsTrainingEvent, (), JsTrainingEvent, Status, false, true>;

    /// Bridge between the Rust [`TrainingProgress`] trait and a JS callback.
    ///
    /// Why: `TrainingProgress::on_event` is sync, but JS callbacks run on the
    /// main event-loop thread. We use the non-blocking `call()` API; a
    /// non-OK queueing status (closed function, queue full) is the only
    /// failure we can observe synchronously and triggers cancellation:
    /// `Err(BlazenTrainError::Cancelled)`. JS-side exceptions are fire-and-
    /// forget, matching the `runAgentWithCallback` event-emitter pattern.
    struct NodeTrainingProgressBridge {
        callback: Arc<ProgressTsfn>,
    }

    impl TrainingProgress for NodeTrainingProgressBridge {
        fn on_event(&self, event: TrainingEvent) -> Result<(), BlazenTrainError> {
            let js_event = JsTrainingEvent::from(event);
            let status = self
                .callback
                .call(js_event, ThreadsafeFunctionCallMode::Blocking);
            if status == Status::Ok {
                Ok(())
            } else {
                tracing::warn!(
                    ?status,
                    "training progress callback queueing failed; aborting run",
                );
                Err(BlazenTrainError::Cancelled)
            }
        }
    }

    // -----------------------------------------------------------------------
    // JsPreferenceJsonlDataset / JsRatedJsonlDataset (opaque)
    // -----------------------------------------------------------------------

    /// Preference-pair JSONL dataset for DPO / ORPO / SimPO.
    ///
    /// Each line of the input file must deserialize to either
    /// `{"prompt": "...", "chosen": "...", "rejected": "..."}` or
    /// `{"messages": [...], "chosen": "...", "rejected": "..."}` (chat shape).
    #[napi(js_name = "PreferenceJsonlDataset")]
    pub struct JsPreferenceJsonlDataset {
        pub(super) inner: Arc<PreferenceJsonlDataset>,
    }

    #[napi]
    impl JsPreferenceJsonlDataset {
        /// Load a preference JSONL file using the tokenizer at
        /// `tokenizerPath`.
        ///
        /// # Errors
        ///
        /// Throws if the tokenizer cannot be loaded, the device string is
        /// invalid, or the JSONL file fails to parse.
        #[napi(factory, js_name = "fromPath")]
        pub fn from_path(
            path: String,
            tokenizer_path: String,
            opts: Option<JsJsonlDatasetOptions>,
        ) -> napi::Result<Self> {
            let opts = opts.unwrap_or(JsJsonlDatasetOptions {
                chat_template: None,
                max_seq_len: None,
                device: None,
                pad_token_id: None,
            });
            let max_seq_len = opts.max_seq_len.unwrap_or(2048) as usize;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "PreferenceJsonlDataset.maxSeqLen must be > 0",
                ));
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                napi::Error::from_reason(format!(
                    "failed to load tokenizer from {tokenizer_path:?}: {e}"
                ))
            })?;
            let device_str = opts.device.unwrap_or_else(|| String::from("cpu"));
            let cdev = parse_train_device_node(&device_str)?;
            let ds = PreferenceJsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                opts.chat_template.as_deref(),
                max_seq_len,
                cdev,
                opts.pad_token_id.unwrap_or(0),
            )
            .map_err(|e| {
                napi::Error::from_reason(format!("PreferenceJsonlDataset load failed: {e}"))
            })?;
            Ok(Self {
                inner: Arc::new(ds),
            })
        }
    }

    /// Rated JSONL dataset for KTO.
    ///
    /// Each line of the input file must deserialize to
    /// `{"prompt"|"messages": ..., "completion": "...", "label": true|false}`.
    #[napi(js_name = "RatedJsonlDataset")]
    pub struct JsRatedJsonlDataset {
        pub(super) inner: Arc<RatedJsonlDataset>,
    }

    #[napi]
    impl JsRatedJsonlDataset {
        /// Load a rated JSONL file using the tokenizer at `tokenizerPath`.
        ///
        /// # Errors
        ///
        /// Throws if the tokenizer cannot be loaded, the device string is
        /// invalid, or the JSONL file fails to parse.
        #[napi(factory, js_name = "fromPath")]
        pub fn from_path(
            path: String,
            tokenizer_path: String,
            opts: Option<JsJsonlDatasetOptions>,
        ) -> napi::Result<Self> {
            let opts = opts.unwrap_or(JsJsonlDatasetOptions {
                chat_template: None,
                max_seq_len: None,
                device: None,
                pad_token_id: None,
            });
            let max_seq_len = opts.max_seq_len.unwrap_or(2048) as usize;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "RatedJsonlDataset.maxSeqLen must be > 0",
                ));
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                napi::Error::from_reason(format!(
                    "failed to load tokenizer from {tokenizer_path:?}: {e}"
                ))
            })?;
            let device_str = opts.device.unwrap_or_else(|| String::from("cpu"));
            let cdev = parse_train_device_node(&device_str)?;
            let ds = RatedJsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                opts.chat_template.as_deref(),
                max_seq_len,
                cdev,
                opts.pad_token_id.unwrap_or(0),
            )
            .map_err(|e| napi::Error::from_reason(format!("RatedJsonlDataset load failed: {e}")))?;
            Ok(Self {
                inner: Arc::new(ds),
            })
        }
    }

    // -----------------------------------------------------------------------
    // JsTrainCoreConfig + preference / fine-tune config structs
    // -----------------------------------------------------------------------

    /// Shared training hyperparameters for DPO/ORPO/SimPO/KTO and full
    /// fine-tune. Mirrors [`blazen_train::TrainCoreConfig`].
    #[napi(object)]
    pub struct JsTrainCoreConfig {
        /// HuggingFace repo id of the base model.
        #[napi(js_name = "baseModelRepo")]
        pub base_model_repo: String,
        /// Optional revision (branch / tag / commit) for the base model.
        #[napi(js_name = "baseModelRevision")]
        pub base_model_revision: Option<String>,
        /// Filesystem directory for trained weights and checkpoints.
        #[napi(js_name = "outputDir")]
        pub output_dir: String,
        /// Total optimizer steps to run. Default `1000`.
        #[napi(js_name = "maxSteps")]
        pub max_steps: Option<u32>,
        /// Micro-batch size (per forward pass). Default `1`.
        #[napi(js_name = "batchSize")]
        pub batch_size: Option<u32>,
        /// Micro-batches accumulated before each optimizer step. Default `8`.
        #[napi(js_name = "gradientAccumulationSteps")]
        pub gradient_accumulation_steps: Option<u32>,
        /// Maximum tokenized sequence length per example. Default `1024`.
        #[napi(js_name = "maxSeqLen")]
        pub max_seq_len: Option<u32>,
        /// Run evaluation every N steps when set.
        #[napi(js_name = "evalSteps")]
        pub eval_steps: Option<u32>,
        /// Write a checkpoint every N steps when set.
        #[napi(js_name = "saveSteps")]
        pub save_steps: Option<u32>,
        /// RNG seed. Default `42`.
        pub seed: Option<BigInt>,
        /// Mixed-precision mode for forward / backward. Default `Bf16`.
        #[napi(js_name = "mixedPrecision")]
        pub mixed_precision: Option<JsMixedPrecision>,
        /// Device string forwarded to the trainer (`"cpu"`, `"cuda:0"`, `"metal"`).
        pub device: Option<String>,
        /// Optimizer hyperparameters (AdamW).
        pub optim: Option<JsOptimConfig>,
        /// Learning-rate schedule.
        pub scheduler: Option<JsSchedulerConfig>,
    }

    impl TryFrom<JsTrainCoreConfig> for TrainCoreConfig {
        type Error = napi::Error;

        fn try_from(c: JsTrainCoreConfig) -> Result<Self, Self::Error> {
            if c.base_model_repo.trim().is_empty() {
                return Err(napi::Error::from_reason(
                    "TrainCoreConfig.baseModelRepo must be non-empty",
                ));
            }
            if c.output_dir.trim().is_empty() {
                return Err(napi::Error::from_reason(
                    "TrainCoreConfig.outputDir must be non-empty",
                ));
            }
            let max_steps = c.max_steps.unwrap_or(1000) as usize;
            if max_steps == 0 {
                return Err(napi::Error::from_reason(
                    "TrainCoreConfig.maxSteps must be > 0",
                ));
            }
            let batch_size = c.batch_size.unwrap_or(1) as usize;
            if batch_size == 0 {
                return Err(napi::Error::from_reason(
                    "TrainCoreConfig.batchSize must be > 0",
                ));
            }
            let gradient_accumulation_steps = c.gradient_accumulation_steps.unwrap_or(8) as usize;
            if gradient_accumulation_steps == 0 {
                return Err(napi::Error::from_reason(
                    "TrainCoreConfig.gradientAccumulationSteps must be > 0",
                ));
            }
            let max_seq_len = c.max_seq_len.unwrap_or(1024) as usize;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "TrainCoreConfig.maxSeqLen must be > 0",
                ));
            }
            let optim = c
                .optim
                .map(OptimConfig::try_from)
                .transpose()?
                .unwrap_or(OptimConfig {
                    learning_rate: 2e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.0,
                    gradient_clip: Some(1.0),
                });
            let scheduler = c.scheduler.map_or(
                SchedulerConfig {
                    kind: SchedulerKind::Cosine,
                    warmup_steps: 0,
                },
                SchedulerConfig::from,
            );
            Ok(Self {
                base_model_repo: c.base_model_repo,
                base_model_revision: c.base_model_revision,
                output_dir: PathBuf::from(c.output_dir),
                max_steps,
                batch_size,
                gradient_accumulation_steps,
                max_seq_len,
                eval_steps: c.eval_steps.map(|v| v as usize),
                save_steps: c.save_steps.map(|v| v as usize),
                seed: c.seed.map_or(42, |b| b.get_u64().1),
                mixed_precision: c.mixed_precision.map_or(MixedPrecision::Bf16, Into::into),
                device: c.device,
                optim,
                scheduler,
            })
        }
    }

    /// Direct Preference Optimization (DPO) configuration.
    #[napi(object)]
    pub struct JsDpoConfig {
        /// Shared training hyperparameters.
        pub core: JsTrainCoreConfig,
        /// LoRA hyperparameters applied to the policy model.
        pub lora: Option<JsLoraConfig>,
        /// KL-regularization strength. Default `0.1`.
        pub beta: Option<f64>,
        /// Conservative DPO label smoothing (cDPO). Default `0.0`.
        #[napi(js_name = "labelSmoothing")]
        pub label_smoothing: Option<f64>,
        /// Reference model repo. `null` reuses `core.baseModelRepo`.
        #[napi(js_name = "referenceModelRepo")]
        pub reference_model_repo: Option<String>,
        /// Optional revision for the reference model.
        #[napi(js_name = "referenceModelRevision")]
        pub reference_model_revision: Option<String>,
    }

    impl TryFrom<JsDpoConfig> for DpoConfig {
        type Error = napi::Error;

        fn try_from(c: JsDpoConfig) -> Result<Self, Self::Error> {
            let beta = c.beta.unwrap_or(0.1);
            if !beta.is_finite() || beta <= 0.0 {
                return Err(napi::Error::from_reason("DpoConfig.beta must be > 0"));
            }
            let label_smoothing = c.label_smoothing.unwrap_or(0.0);
            if !label_smoothing.is_finite() || !(0.0..=0.5).contains(&label_smoothing) {
                return Err(napi::Error::from_reason(
                    "DpoConfig.labelSmoothing must be in [0.0, 0.5]",
                ));
            }
            let core: TrainCoreConfig = c.core.try_into()?;
            let lora = c
                .lora
                .map(LoraConfig::try_from)
                .transpose()?
                .unwrap_or_else(|| LoraConfig {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.0,
                    target_modules: default_target_modules(),
                });
            Ok(Self {
                core,
                lora,
                beta: beta as f32,
                reference_model_repo: c.reference_model_repo,
                reference_model_revision: c.reference_model_revision,
                label_smoothing: label_smoothing as f32,
            })
        }
    }

    /// Odds Ratio Preference Optimization (ORPO) configuration.
    #[napi(object)]
    pub struct JsOrpoConfig {
        /// Shared training hyperparameters.
        pub core: JsTrainCoreConfig,
        /// LoRA hyperparameters.
        pub lora: Option<JsLoraConfig>,
        /// Weight of the odds-ratio term relative to the SFT term. Default `0.1`.
        pub lambda: Option<f64>,
    }

    impl TryFrom<JsOrpoConfig> for OrpoConfig {
        type Error = napi::Error;

        fn try_from(c: JsOrpoConfig) -> Result<Self, Self::Error> {
            let lambda = c.lambda.unwrap_or(0.1);
            if !lambda.is_finite() || lambda < 0.0 {
                return Err(napi::Error::from_reason("OrpoConfig.lambda must be >= 0"));
            }
            let core: TrainCoreConfig = c.core.try_into()?;
            let lora = c
                .lora
                .map(LoraConfig::try_from)
                .transpose()?
                .unwrap_or_else(|| LoraConfig {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.0,
                    target_modules: default_target_modules(),
                });
            Ok(Self {
                core,
                lora,
                lambda: lambda as f32,
            })
        }
    }

    /// Simple Preference Optimization (`SimPO`) configuration.
    #[napi(object)]
    pub struct JsSimpoConfig {
        /// Shared training hyperparameters.
        pub core: JsTrainCoreConfig,
        /// LoRA hyperparameters.
        pub lora: Option<JsLoraConfig>,
        /// Logit scaling for the length-normalized preference margin. Default `2.0`.
        pub beta: Option<f64>,
        /// Target reward margin between chosen and rejected. Default `1.0`.
        pub gamma: Option<f64>,
    }

    impl TryFrom<JsSimpoConfig> for SimpoConfig {
        type Error = napi::Error;

        fn try_from(c: JsSimpoConfig) -> Result<Self, Self::Error> {
            let beta = c.beta.unwrap_or(2.0);
            if !beta.is_finite() || beta <= 0.0 {
                return Err(napi::Error::from_reason("SimpoConfig.beta must be > 0"));
            }
            let gamma = c.gamma.unwrap_or(1.0);
            if !gamma.is_finite() || gamma < 0.0 {
                return Err(napi::Error::from_reason("SimpoConfig.gamma must be >= 0"));
            }
            let core: TrainCoreConfig = c.core.try_into()?;
            let lora = c
                .lora
                .map(LoraConfig::try_from)
                .transpose()?
                .unwrap_or_else(|| LoraConfig {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.0,
                    target_modules: default_target_modules(),
                });
            Ok(Self {
                core,
                lora,
                beta: beta as f32,
                gamma: gamma as f32,
            })
        }
    }

    /// Kahneman-Tversky Optimization (KTO) configuration.
    #[napi(object)]
    pub struct JsKtoConfig {
        /// Shared training hyperparameters.
        pub core: JsTrainCoreConfig,
        /// LoRA hyperparameters applied to the policy model.
        pub lora: Option<JsLoraConfig>,
        /// KL-regularization strength. Default `0.1`.
        pub beta: Option<f64>,
        /// Loss weight applied to desirable examples. Default `1.0`.
        #[napi(js_name = "lambdaD")]
        pub lambda_d: Option<f64>,
        /// Loss weight applied to undesirable examples. Default `1.0`.
        #[napi(js_name = "lambdaU")]
        pub lambda_u: Option<f64>,
        /// Reference model repo. `null` reuses `core.baseModelRepo`.
        #[napi(js_name = "referenceModelRepo")]
        pub reference_model_repo: Option<String>,
        /// Optional revision for the reference model.
        #[napi(js_name = "referenceModelRevision")]
        pub reference_model_revision: Option<String>,
    }

    impl TryFrom<JsKtoConfig> for KtoConfig {
        type Error = napi::Error;

        fn try_from(c: JsKtoConfig) -> Result<Self, Self::Error> {
            let beta = c.beta.unwrap_or(0.1);
            if !beta.is_finite() || beta <= 0.0 {
                return Err(napi::Error::from_reason("KtoConfig.beta must be > 0"));
            }
            let lambda_d = c.lambda_d.unwrap_or(1.0);
            if !lambda_d.is_finite() || lambda_d < 0.0 {
                return Err(napi::Error::from_reason("KtoConfig.lambdaD must be >= 0"));
            }
            let lambda_u = c.lambda_u.unwrap_or(1.0);
            if !lambda_u.is_finite() || lambda_u < 0.0 {
                return Err(napi::Error::from_reason("KtoConfig.lambdaU must be >= 0"));
            }
            let core: TrainCoreConfig = c.core.try_into()?;
            let lora = c
                .lora
                .map(LoraConfig::try_from)
                .transpose()?
                .unwrap_or_else(|| LoraConfig {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.0,
                    target_modules: default_target_modules(),
                });
            Ok(Self {
                core,
                lora,
                beta: beta as f32,
                lambda_d: lambda_d as f32,
                lambda_u: lambda_u as f32,
                reference_model_repo: c.reference_model_repo,
                reference_model_revision: c.reference_model_revision,
            })
        }
    }

    /// Full fine-tune configuration (every parameter trains; no LoRA adapter).
    ///
    /// `gradientCheckpointing = true` is accepted for forward compatibility
    /// but the trainer currently rejects it at init time because candle
    /// 0.10.2 has no activation-checkpointing primitive.
    #[napi(object)]
    pub struct JsFullFineTuneConfig {
        /// Shared training hyperparameters.
        pub core: JsTrainCoreConfig,
        /// Activation checkpointing (currently unsupported in the trainer).
        #[napi(js_name = "gradientCheckpointing")]
        pub gradient_checkpointing: Option<bool>,
    }

    impl TryFrom<JsFullFineTuneConfig> for FullFineTuneConfig {
        type Error = napi::Error;

        fn try_from(c: JsFullFineTuneConfig) -> Result<Self, Self::Error> {
            Ok(Self {
                core: c.core.try_into()?,
                gradient_checkpointing: c.gradient_checkpointing.unwrap_or(false),
            })
        }
    }

    /// Result of a completed full fine-tune run.
    #[napi(object)]
    pub struct JsFullFineTuneResult {
        /// Directory the trained model weights were written to.
        #[napi(js_name = "outputDir")]
        pub output_dir: String,
        /// Final training loss.
        #[napi(js_name = "finalLoss")]
        pub final_loss: f64,
        /// Total optimizer steps executed.
        #[napi(js_name = "stepsCompleted")]
        pub steps_completed: u32,
    }

    impl From<FullFineTuneResult> for JsFullFineTuneResult {
        fn from(r: FullFineTuneResult) -> Self {
            Self {
                output_dir: r.output_dir.display().to_string(),
                final_loss: f64::from(r.final_loss),
                steps_completed: r.steps_completed as u32,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Bridge for fine_tune: wrap Arc<JsonlDataset> as Arc<dyn TrainingDataset>
    // -----------------------------------------------------------------------

    /// Why: `train_lora` consumes `Box<dyn TrainingDataset>` (and we already
    /// have the local [`ArcDataset`] wrapper above), but `fine_tune` wants
    /// `Arc<dyn TrainingDataset>`. The blanket impl `impl TrainingDataset for
    /// JsonlDataset` is for the bare struct, not `Arc<JsonlDataset>`, so we
    /// need an Arc-aware wrapper that forwards to the underlying impl.
    struct ArcDatasetForFineTune(Arc<JsonlDataset>);

    #[async_trait]
    impl TrainingDataset for ArcDatasetForFineTune {
        fn len(&self) -> usize {
            self.0.len()
        }

        async fn batch(
            &self,
            batch_size: usize,
            idx: usize,
        ) -> Result<TrainingBatch, BlazenTrainError> {
            self.0.batch(batch_size, idx).await
        }
    }

    // -----------------------------------------------------------------------
    // JsModelManager::train_lora
    // -----------------------------------------------------------------------

    #[napi]
    impl JsModelManager {
        /// Train a `LoRA` adapter end-to-end on the configured base model.
        ///
        /// Downloads the base model from HuggingFace (cached), builds a
        /// VarMap, runs the AdamW + LoRA training loop driven by `dataset`,
        /// and writes the resulting PEFT-format adapter to
        /// `config.outputDir`. The returned `TrainedAdapter`'s `adapterDir`
        /// is immediately mountable via [`Self::load_adapter`] on a
        /// compatible backend.
        ///
        /// `progress`, when supplied, is invoked once per Started /
        /// StepCompleted / Evaluating / EvalCompleted / CheckpointSaved /
        /// Finished transition. The return value is ignored; throwing
        /// inside the callback does not abort the run. A failure to queue
        /// the call (closed function, etc.) cancels the run with a
        /// `BlazenError::cancelled`.
        ///
        /// # Errors
        ///
        /// Throws on invalid config, unrecognised device, HF download
        /// failure, dataset I/O failure, trainer failure, or queueing
        /// failure on the progress callback.
        #[napi(js_name = "trainLora", ts_return_type = "Promise<TrainedAdapter>")]
        pub async fn train_lora(
            &self,
            config: JsTrainConfig,
            dataset: &JsJsonlDataset,
            progress: Option<ProgressTsfn>,
        ) -> napi::Result<JsTrainedAdapter> {
            let rust_cfg: TrainConfig = config.try_into()?;
            let ds_arc = dataset.inner.clone();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = NodeTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });
            let dataset_box: Box<dyn TrainingDataset> = Box::new(ArcDataset(ds_arc));
            let adapter = self
                .inner
                .train_lora(rust_cfg, dataset_box, sink)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(JsTrainedAdapter::from(adapter))
        }

        /// Train a `LoRA` adapter via Direct Preference Optimization (DPO).
        ///
        /// Like [`Self::train_lora`] but consumes a preference-pair dataset
        /// of `(prompt, chosen, rejected)` triples and requires a frozen
        /// reference model (defaults to `config.core.baseModelRepo` when
        /// `config.referenceModelRepo` is `null`).
        ///
        /// # Errors
        ///
        /// Throws on invalid config, unrecognised device, HF download
        /// failure, dataset I/O failure, trainer failure, or queueing
        /// failure on the progress callback.
        #[napi(js_name = "trainDpo", ts_return_type = "Promise<TrainedAdapter>")]
        pub async fn train_dpo(
            &self,
            config: JsDpoConfig,
            dataset: &JsPreferenceJsonlDataset,
            progress: Option<ProgressTsfn>,
        ) -> napi::Result<JsTrainedAdapter> {
            let rust_cfg: DpoConfig = config.try_into()?;
            let ds_arc: Arc<dyn PreferenceDataset> = dataset.inner.clone();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = NodeTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });
            let adapter = self
                .inner
                .train_dpo(rust_cfg, ds_arc, sink)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(JsTrainedAdapter::from(adapter))
        }

        /// Train a `LoRA` adapter via Odds Ratio Preference Optimization (ORPO).
        ///
        /// Reference-free; combines a standard SFT loss on chosen
        /// completions with an odds-ratio preference term weighted by
        /// `config.lambda`.
        ///
        /// # Errors
        ///
        /// Same surface as [`Self::train_dpo`].
        #[napi(js_name = "trainOrpo", ts_return_type = "Promise<TrainedAdapter>")]
        pub async fn train_orpo(
            &self,
            config: JsOrpoConfig,
            dataset: &JsPreferenceJsonlDataset,
            progress: Option<ProgressTsfn>,
        ) -> napi::Result<JsTrainedAdapter> {
            let rust_cfg: OrpoConfig = config.try_into()?;
            let ds_arc: Arc<dyn PreferenceDataset> = dataset.inner.clone();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = NodeTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });
            let adapter = self
                .inner
                .train_orpo(rust_cfg, ds_arc, sink)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(JsTrainedAdapter::from(adapter))
        }

        /// Train a `LoRA` adapter via Simple Preference Optimization (`SimPO`).
        ///
        /// Reference-free and length-normalized. `config.beta` scales the
        /// preference logits and `config.gamma` sets the target reward
        /// margin.
        ///
        /// # Errors
        ///
        /// Same surface as [`Self::train_dpo`].
        #[napi(js_name = "trainSimpo", ts_return_type = "Promise<TrainedAdapter>")]
        pub async fn train_simpo(
            &self,
            config: JsSimpoConfig,
            dataset: &JsPreferenceJsonlDataset,
            progress: Option<ProgressTsfn>,
        ) -> napi::Result<JsTrainedAdapter> {
            let rust_cfg: SimpoConfig = config.try_into()?;
            let ds_arc: Arc<dyn PreferenceDataset> = dataset.inner.clone();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = NodeTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });
            let adapter = self
                .inner
                .train_simpo(rust_cfg, ds_arc, sink)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(JsTrainedAdapter::from(adapter))
        }

        /// Train a `LoRA` adapter via Kahneman-Tversky Optimization (KTO).
        ///
        /// Like DPO, KTO requires a frozen reference model — but the
        /// dataset schema differs: each row is a
        /// `(prompt, completion, desirable)` triple
        /// ([`JsRatedJsonlDataset`]), not a chosen/rejected pair.
        ///
        /// # Errors
        ///
        /// Same surface as [`Self::train_dpo`].
        #[napi(js_name = "trainKto", ts_return_type = "Promise<TrainedAdapter>")]
        pub async fn train_kto(
            &self,
            config: JsKtoConfig,
            dataset: &JsRatedJsonlDataset,
            progress: Option<ProgressTsfn>,
        ) -> napi::Result<JsTrainedAdapter> {
            let rust_cfg: KtoConfig = config.try_into()?;
            let ds_arc: Arc<dyn RatedDataset> = dataset.inner.clone();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = NodeTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });
            let adapter = self
                .inner
                .train_kto(rust_cfg, ds_arc, sink)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(JsTrainedAdapter::from(adapter))
        }

        /// Run a full fine-tune (every parameter trainable; no `LoRA`
        /// adapter).
        ///
        /// Returns [`JsFullFineTuneResult`] — not [`JsTrainedAdapter`] —
        /// because the output is a complete set of model weights in
        /// `config.core.outputDir` rather than a mountable PEFT delta.
        ///
        /// Setting `config.gradientCheckpointing = true` is rejected at
        /// init time because candle 0.10.2 has no activation-checkpointing
        /// primitive.
        ///
        /// # Errors
        ///
        /// Throws on invalid config, unrecognised device,
        /// `gradientCheckpointing = true`, HF download failure, dataset
        /// I/O failure, trainer failure, or queueing failure on the
        /// progress callback.
        #[napi(js_name = "fineTune", ts_return_type = "Promise<FullFineTuneResult>")]
        pub async fn fine_tune(
            &self,
            config: JsFullFineTuneConfig,
            dataset: &JsJsonlDataset,
            progress: Option<ProgressTsfn>,
        ) -> napi::Result<JsFullFineTuneResult> {
            let rust_cfg: FullFineTuneConfig = config.try_into()?;
            let ds_arc: Arc<dyn TrainingDataset> =
                Arc::new(ArcDatasetForFineTune(dataset.inner.clone()));
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = NodeTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });
            let result = self
                .inner
                .fine_tune(rust_cfg, ds_arc, sink)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Ok(JsFullFineTuneResult::from(result))
        }
    }
}
