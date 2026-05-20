//! Memory-budget-aware model manager with per-pool LRU eviction.
//!
//! Tracks registered [`LocalModel`] instances and their estimated memory
//! footprint, organised by [`Pool`] (one budget for host RAM, one per GPU).
//! Loading a model that would exceed its pool's budget evicts the
//! least-recently-used loaded model **in the same pool** until it fits.
//! Models in different pools never evict each other.
//!
//! # Capacity, not performance
//!
//! These budgets answer "will this fit?" — not "will this run fast?".
//! Whether a 70B model loaded on CPU is *useful* at 1–3 tok/s is a
//! workload-choice question that this manager intentionally does not
//! answer. It only prevents OOM.

pub mod hf_loader;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use blazen_llm::{AdapterHandle, AdapterOptions, AdapterStatus, BlazenError, LocalModel, Pool};
use tokio::sync::Mutex;

#[cfg(feature = "hf-loader")]
use crate::hf_loader::{BackendHint, HfLoadOptions};

/// Status of a registered model.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    /// Identifier of the model.
    pub id: String,
    /// Whether the model is currently loaded.
    pub loaded: bool,
    /// Estimated memory footprint in bytes. Includes the base model plus any
    /// mounted adapters.
    pub memory_estimate_bytes: u64,
    /// Pool this model is registered against.
    pub pool: Pool,
    /// Adapters currently mounted on this model (empty if none).
    pub adapters: Vec<AdapterStatus>,
}

struct RegisteredModel {
    model: Arc<dyn LocalModel>,
    memory_estimate_bytes: u64,
    pool: Pool,
    loaded: bool,
    last_used: Option<Instant>,
    adapters: HashMap<String, AdapterStatus>,
}

/// Memory-budget-aware model manager with per-pool LRU eviction.
///
/// See the crate-level docs for the capacity-vs-performance distinction.
pub struct ModelManager {
    pool_budgets: HashMap<Pool, u64>,
    state: Mutex<HashMap<String, RegisteredModel>>,
}

impl ModelManager {
    /// Create a new manager with the given per-pool budgets (in bytes).
    ///
    /// Pools not present in the map have an implicit budget of 0, meaning
    /// no model targeting that pool can ever load. Add explicit entries for
    /// every pool you expect to use.
    #[must_use]
    pub fn new(pool_budgets: HashMap<Pool, u64>) -> Self {
        Self {
            pool_budgets,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Convenience constructor for the common single-GPU desktop case:
    /// one CPU pool sized in GB and one GPU pool (`Pool::Gpu(0)`) sized in GB.
    ///
    /// Pass `0.0` for either argument to disable that pool. These are
    /// *capacity* budgets — they govern how many bytes of model weights can
    /// be resident concurrently, not how fast inference will run.
    #[must_use]
    pub fn with_budgets_gb(cpu_ram_gb: f64, gpu_vram_gb: f64) -> Self {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let cpu_bytes = (cpu_ram_gb * 1_073_741_824.0) as u64;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let gpu_bytes = (gpu_vram_gb * 1_073_741_824.0) as u64;
        let mut budgets = HashMap::new();
        budgets.insert(Pool::Cpu, cpu_bytes);
        budgets.insert(Pool::Gpu(0), gpu_bytes);
        Self::new(budgets)
    }

    /// Register a model with its estimated memory footprint in bytes.
    /// The pool is derived from `model.device()` at registration time.
    pub async fn register(&self, id: &str, model: Arc<dyn LocalModel>, memory_estimate_bytes: u64) {
        let pool: Pool = model.device().into();
        let mut state = self.state.lock().await;
        state.insert(
            id.to_owned(),
            RegisteredModel {
                model,
                memory_estimate_bytes,
                pool,
                loaded: false,
                last_used: None,
                adapters: HashMap::new(),
            },
        );
    }

    /// Load a model, evicting LRU models in the same pool if necessary.
    ///
    /// # Errors
    /// Returns [`BlazenError::Validation`] when the model is not registered,
    /// when its memory estimate exceeds its pool's total budget, or when
    /// no eviction can free enough space. Propagates errors from the
    /// underlying [`LocalModel::load`] / [`LocalModel::unload`] calls.
    ///
    /// # Panics
    /// Panics only if internal state becomes inconsistent (a model ID that
    /// was just verified to exist is no longer present). This cannot happen
    /// under normal operation.
    pub async fn load(&self, id: &str) -> Result<(), BlazenError> {
        let mut state = self.state.lock().await;

        let entry = state
            .get(id)
            .ok_or_else(|| BlazenError::validation(format!("model '{id}' is not registered")))?;

        if entry.loaded {
            let entry = state.get_mut(id).expect("checked above");
            entry.last_used = Some(Instant::now());
            return Ok(());
        }

        let needed = entry.memory_estimate_bytes;
        let pool = entry.pool;
        let budget = self.pool_budgets.get(&pool).copied().unwrap_or(0);

        if needed > budget {
            return Err(BlazenError::validation(format!(
                "model '{id}' requires {needed} bytes but pool {pool} budget is only {budget} bytes",
            )));
        }

        let mut used = Self::used_bytes_in_pool(&state, pool);
        while used + needed > budget {
            let lru_id = state
                .iter()
                .filter(|(k, v)| v.loaded && v.pool == pool && k.as_str() != id)
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone());

            let Some(lru_id) = lru_id else {
                return Err(BlazenError::validation(format!(
                    "cannot free enough memory to load model '{id}' in pool {pool} \
                     (need {needed}, used {used}, budget {budget})",
                )));
            };

            let lru_model = state
                .get(&lru_id)
                .expect("lru_id came from iteration")
                .model
                .clone();
            drop(state);
            lru_model.unload().await?;
            state = self.state.lock().await;
            if let Some(e) = state.get_mut(&lru_id) {
                e.loaded = false;
                e.last_used = None;
            }
            used = Self::used_bytes_in_pool(&state, pool);
        }

        let model = state.get(id).expect("checked at top").model.clone();
        drop(state);
        model.load().await?;
        let mut state = self.state.lock().await;
        if let Some(e) = state.get_mut(id) {
            e.loaded = true;
            e.last_used = Some(Instant::now());
        }
        Ok(())
    }

    /// Unload a model, freeing its budget within its pool.
    ///
    /// # Errors
    /// Returns [`BlazenError::Validation`] when the model is not registered.
    /// Propagates errors from the underlying [`LocalModel::unload`] call.
    pub async fn unload(&self, id: &str) -> Result<(), BlazenError> {
        let state = self.state.lock().await;
        let entry = state
            .get(id)
            .ok_or_else(|| BlazenError::validation(format!("model '{id}' is not registered")))?;
        if !entry.loaded {
            return Ok(());
        }
        let model = entry.model.clone();
        drop(state);
        model.unload().await?;
        let mut state = self.state.lock().await;
        if let Some(e) = state.get_mut(id) {
            e.loaded = false;
            e.last_used = None;
        }
        Ok(())
    }

    /// Check if a model is currently loaded.
    pub async fn is_loaded(&self, id: &str) -> bool {
        let state = self.state.lock().await;
        state.get(id).is_some_and(|e| e.loaded)
    }

    /// Ensure a model is loaded. Equivalent to [`Self::load`].
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn ensure_loaded(&self, id: &str) -> Result<(), BlazenError> {
        self.load(id).await
    }

    /// Total memory currently used by loaded models in the given pool.
    pub async fn used_bytes(&self, pool: Pool) -> u64 {
        let state = self.state.lock().await;
        Self::used_bytes_in_pool(&state, pool)
    }

    /// Available memory within the given pool's budget.
    pub async fn available_bytes(&self, pool: Pool) -> u64 {
        let used = self.used_bytes(pool).await;
        let budget = self.pool_budgets.get(&pool).copied().unwrap_or(0);
        budget.saturating_sub(used)
    }

    /// All configured pools and their budgets (in bytes).
    #[must_use]
    pub fn pools(&self) -> Vec<(Pool, u64)> {
        self.pool_budgets.iter().map(|(p, b)| (*p, *b)).collect()
    }

    /// Status of all registered models.
    pub async fn status(&self) -> Vec<ModelStatus> {
        let state = self.state.lock().await;
        state
            .iter()
            .map(|(id, entry)| ModelStatus {
                id: id.clone(),
                loaded: entry.loaded,
                memory_estimate_bytes: entry.memory_estimate_bytes,
                pool: entry.pool,
                adapters: entry.adapters.values().cloned().collect(),
            })
            .collect()
    }

    fn used_bytes_in_pool(state: &HashMap<String, RegisteredModel>, pool: Pool) -> u64 {
        state
            .values()
            .filter(|e| e.loaded && e.pool == pool)
            .map(|e| e.memory_estimate_bytes)
            .sum()
    }

    /// Mount a `PEFT`-format `LoRA` adapter on a previously-registered model.
    ///
    /// The base model is loaded automatically via [`Self::ensure_loaded`]
    /// before the adapter is mounted. Adapter file sizes are summed from
    /// `adapter_dir` and checked against the pool budget before forwarding
    /// to the backend; if the result would exceed the budget the call
    /// fails with [`BlazenError::validation`] and the pool is NOT auto-
    /// evicted (evicting another tenant's model to make room for this
    /// tenant's adapter is action-at-a-distance and breaks isolation).
    ///
    /// After the backend reports success the manager bumps the model's
    /// `memory_estimate_bytes` by the adapter's `memory_bytes` so future
    /// LRU decisions see the updated footprint.
    ///
    /// # Errors
    ///
    /// - [`BlazenError::validation`] if `model_id` is not registered, the
    ///   adapter directory is missing required files, the adapter id is
    ///   already mounted, or the pool budget would be exceeded.
    /// - [`BlazenError::unsupported`] if the backend cannot mount adapters.
    /// - Any error returned by [`LocalModel::load`] (during the
    ///   `ensure_loaded` pre-flight) or [`LocalModel::load_adapter`].
    pub async fn load_adapter(
        &self,
        model_id: &str,
        adapter_dir: &Path,
        options: AdapterOptions,
    ) -> Result<AdapterHandle, BlazenError> {
        self.ensure_loaded(model_id).await?;

        let adapter_id = options.adapter_id.clone();
        let scale = options.scale;
        let source_dir = adapter_dir.to_path_buf();

        let (model, pool, used, budget, on_disk_estimate) = {
            let state = self.state.lock().await;
            let entry = state.get(model_id).ok_or_else(|| {
                BlazenError::validation(format!("model '{model_id}' is not registered"))
            })?;
            if entry.adapters.contains_key(&adapter_id) {
                return Err(BlazenError::validation(format!(
                    "adapter '{adapter_id}' is already mounted on model '{model_id}'; \
                     unload it first to replace",
                )));
            }
            let pool = entry.pool;
            let budget = self.pool_budgets.get(&pool).copied().unwrap_or(0);
            let used = Self::used_bytes_in_pool(&state, pool);
            let model = entry.model.clone();
            let on_disk = sum_adapter_bytes(adapter_dir)?;
            (model, pool, used, budget, on_disk)
        };

        if used + on_disk_estimate > budget {
            return Err(BlazenError::validation(format!(
                "mounting adapter '{adapter_id}' on '{model_id}' would exceed pool {pool} budget \
                 (need {on_disk_estimate} more, used {used}, budget {budget})",
            )));
        }

        let handle = model.load_adapter(adapter_dir, options).await?;

        let status = AdapterStatus {
            adapter_id: handle.adapter_id.clone(),
            scale,
            source_dir,
            memory_bytes: handle.memory_bytes,
        };
        let mut state = self.state.lock().await;
        if let Some(entry) = state.get_mut(model_id) {
            entry.memory_estimate_bytes = entry
                .memory_estimate_bytes
                .saturating_add(handle.memory_bytes);
            entry.adapters.insert(handle.adapter_id.clone(), status);
            entry.last_used = Some(Instant::now());
        }
        Ok(handle)
    }

    /// Remove an adapter previously mounted via [`Self::load_adapter`].
    ///
    /// # Errors
    ///
    /// - [`BlazenError::validation`] if `model_id` is not registered or
    ///   `adapter_id` is not currently mounted on it.
    /// - Any error returned by [`LocalModel::unload_adapter`].
    pub async fn unload_adapter(
        &self,
        model_id: &str,
        adapter_id: &str,
    ) -> Result<(), BlazenError> {
        let (model, status) = {
            let state = self.state.lock().await;
            let entry = state.get(model_id).ok_or_else(|| {
                BlazenError::validation(format!("model '{model_id}' is not registered"))
            })?;
            let status = entry.adapters.get(adapter_id).cloned().ok_or_else(|| {
                BlazenError::validation(format!(
                    "adapter '{adapter_id}' is not mounted on model '{model_id}'",
                ))
            })?;
            (entry.model.clone(), status)
        };

        let handle = AdapterHandle {
            adapter_id: status.adapter_id.clone(),
            memory_bytes: status.memory_bytes,
            mount_strategy: blazen_llm::AdapterMountStrategy::Attached,
        };
        model.unload_adapter(&handle).await?;

        let mut state = self.state.lock().await;
        if let Some(entry) = state.get_mut(model_id) {
            entry.memory_estimate_bytes = entry
                .memory_estimate_bytes
                .saturating_sub(status.memory_bytes);
            entry.adapters.remove(adapter_id);
        }
        Ok(())
    }

    /// Auto-detect the right local-inference backend for the Hugging Face
    /// repo at `repo` and register a fresh provider under `id`.
    ///
    /// Hides backend choice from casual users; advanced users still
    /// construct providers directly and call [`Self::register`]. The chosen
    /// backend is returned so the caller can inspect or log it.
    ///
    /// See [`hf_loader::choose_backend`] for the selection rules. The repo
    /// metadata is probed once via [`hf_loader::detect_layout`] (one HTTP
    /// `GET /api/models/{repo}` — no weight downloads); selection plus
    /// memory estimation are then pure.
    ///
    /// # Errors
    ///
    /// - [`BlazenError::Unsupported`] when the backend the loader picked
    ///   is not compiled in (feature gate `mistralrs-provider`,
    ///   `candle-provider`, or `llamacpp-provider` — `live-models`
    ///   activates all three).
    /// - [`BlazenError::Validation`] when the repo has no recognisable
    ///   model weights, is PEFT-adapter-only (use [`Self::load_adapter`]
    ///   instead), or is otherwise unloadable as a base model.
    /// - Any error from [`hf_loader::detect_layout`] (network / parse) or
    ///   the chosen provider's own construction.
    #[cfg(feature = "hf-loader")]
    pub async fn load_from_hf(
        &self,
        id: String,
        repo: &str,
        options: HfLoadOptions,
    ) -> Result<BackendHint, BlazenError> {
        let layout = hf_loader::detect_layout(repo, &options).await?;
        let backend =
            hf_loader::choose_backend(&layout, options.backend_hint, options.gguf_file.as_deref())?;

        // Why: layout-derived sizes are only available when the Hub returned
        // them. The user override always wins; otherwise fall back to summing
        // siblings, then to layout.total_weight_bytes (rounded), then bail.
        let memory_bytes = options
            .memory_estimate_bytes
            .or_else(|| {
                hf_loader::estimate_backend_bytes(
                    backend,
                    &layout,
                    &layout.file_sizes,
                    options.gguf_file.as_deref(),
                )
            })
            .or_else(|| layout.total_weight_bytes.map(hf_loader::round_up_to_mb))
            .ok_or_else(|| {
                BlazenError::validation(format!(
                    "could not determine memory footprint for repo '{repo}'; \
                     pass HfLoadOptions::memory_estimate_bytes explicitly"
                ))
            })?;

        let model = build_provider(backend, repo, &options, &layout).await?;
        self.register(&id, model, memory_bytes).await;
        Ok(backend)
    }

    /// List adapters mounted on a registered model.
    ///
    /// # Errors
    /// [`BlazenError::validation`] if `model_id` is not registered.
    pub async fn list_adapters(&self, model_id: &str) -> Result<Vec<AdapterStatus>, BlazenError> {
        let state = self.state.lock().await;
        let entry = state.get(model_id).ok_or_else(|| {
            BlazenError::validation(format!("model '{model_id}' is not registered"))
        })?;
        Ok(entry.adapters.values().cloned().collect())
    }
}

/// Construct an `Arc<dyn LocalModel>` for the chosen backend, mapping
/// [`HfLoadOptions`] onto the backend's own options struct.
///
/// Each backend is independently feature-gated; calling [`ModelManager::load_from_hf`]
/// for a backend whose feature is off returns [`BlazenError::unsupported`]
/// naming the missing feature.
#[cfg(feature = "hf-loader")]
#[allow(unused_variables)] // Why: `layout` is consumed only by the candle/llamacpp arms.
#[allow(clippy::unused_async)] // Why: only the llamacpp arm awaits; the other arms are sync or disabled-feature error branches.
async fn build_provider(
    backend: BackendHint,
    repo: &str,
    options: &HfLoadOptions,
    layout: &hf_loader::DetectedLayout,
) -> Result<Arc<dyn LocalModel>, BlazenError> {
    match backend {
        BackendHint::Mistralrs => {
            #[cfg(feature = "mistralrs-provider")]
            {
                let mr_opts = blazen_llm_mistralrs::MistralRsOptions {
                    model_id: repo.to_string(),
                    quantization: None,
                    device: options.device.clone(),
                    context_length: None,
                    max_batch_size: None,
                    chat_template: None,
                    cache_dir: options.cache_dir.clone(),
                    vision: false,
                    initial_adapters: Vec::new(),
                };
                let p = blazen_llm_mistralrs::MistralRsProvider::from_options(mr_opts)
                    .map_err(|e| BlazenError::internal(format!("mistralrs build: {e}")))?;
                Ok(Arc::new(p))
            }
            #[cfg(not(feature = "mistralrs-provider"))]
            {
                Err(BlazenError::unsupported(
                    "load_from_hf chose the mistralrs backend but the \
                     `mistralrs-provider` feature is not enabled on blazen-manager",
                ))
            }
        }
        BackendHint::Candle => {
            #[cfg(feature = "candle-provider")]
            {
                let force_st =
                    !layout.gguf_files.is_empty() && !layout.safetensors_files.is_empty();
                let c_opts = blazen_llm_candle::CandleLlmOptions {
                    model_id: Some(repo.to_string()),
                    device: options.device.clone(),
                    quantization: None,
                    revision: options.revision.clone(),
                    context_length: None,
                    cache_dir: options.cache_dir.clone(),
                    initial_adapters: Vec::new(),
                    force_safetensors: force_st,
                };
                let p = blazen_llm_candle::CandleLlmProvider::from_options(c_opts)
                    .map_err(|e| BlazenError::internal(format!("candle build: {e}")))?;
                Ok(Arc::new(p))
            }
            #[cfg(not(feature = "candle-provider"))]
            {
                Err(BlazenError::unsupported(
                    "load_from_hf chose the candle backend but the \
                     `candle-provider` feature is not enabled on blazen-manager",
                ))
            }
        }
        BackendHint::Llamacpp => {
            #[cfg(feature = "llamacpp-provider")]
            {
                // Why: llama.cpp's LlamaCppOptions::model_path accepts either a
                // local path or the HF triple "org/repo/file.gguf". We must
                // pick a specific GGUF — either the caller-supplied gguf_file
                // or the first one in the repo — because the provider rejects
                // bare "org/repo" without a filename.
                let gguf = options
                    .gguf_file
                    .clone()
                    .or_else(|| layout.gguf_files.first().cloned())
                    .ok_or_else(|| {
                        BlazenError::validation(format!(
                            "llamacpp backend requires a *.gguf file but repo '{repo}' \
                             has none; pass HfLoadOptions::gguf_file or use a different backend"
                        ))
                    })?;
                let model_path = format!("{repo}/{gguf}");
                let l_opts = blazen_llm_llamacpp::LlamaCppOptions {
                    model_path: Some(model_path),
                    device: options.device.clone(),
                    quantization: None,
                    context_length: None,
                    n_gpu_layers: None,
                    cache_dir: options.cache_dir.clone(),
                    initial_adapters: Vec::new(),
                };
                let p = blazen_llm_llamacpp::LlamaCppProvider::from_options(l_opts)
                    .await
                    .map_err(|e| BlazenError::internal(format!("llamacpp build: {e}")))?;
                Ok(Arc::new(p))
            }
            #[cfg(not(feature = "llamacpp-provider"))]
            {
                Err(BlazenError::unsupported(
                    "load_from_hf chose the llamacpp backend but the \
                     `llamacpp-provider` feature is not enabled on blazen-manager",
                ))
            }
        }
    }
}

/// Sum the on-disk sizes of the canonical PEFT-layout files in
/// `adapter_dir`: `adapter_model.safetensors` (required) +
/// `adapter_config.json` (required) + `tokenizer.json` (optional). Used as
/// a conservative pre-flight estimate before calling
/// [`LocalModel::load_adapter`]. The backend reports the true runtime
/// footprint after the mount succeeds.
fn sum_adapter_bytes(adapter_dir: &Path) -> Result<u64, BlazenError> {
    let required = ["adapter_model.safetensors", "adapter_config.json"];
    let optional = ["tokenizer.json"];

    let mut total: u64 = 0;
    for name in required {
        let path: PathBuf = adapter_dir.join(name);
        let meta = std::fs::metadata(&path).map_err(|e| {
            BlazenError::validation(format!(
                "adapter directory missing required file '{name}' at {}: {e}",
                path.display(),
            ))
        })?;
        total = total.saturating_add(meta.len());
    }
    for name in optional {
        let path = adapter_dir.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            total = total.saturating_add(meta.len());
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_llm::{AdapterMountStrategy, Device};
    use std::sync::Mutex as StdMutex;

    const GB: u64 = 1_073_741_824;

    struct MockLocalModel {
        loaded: StdMutex<bool>,
        device: Device,
        adapters: StdMutex<HashMap<String, AdapterStatus>>,
        next_adapter_memory_bytes: StdMutex<u64>,
    }

    impl MockLocalModel {
        fn new(device: Device) -> Self {
            Self {
                loaded: StdMutex::new(false),
                device,
                adapters: StdMutex::new(HashMap::new()),
                next_adapter_memory_bytes: StdMutex::new(1024),
            }
        }

        fn with_adapter_memory(self, bytes: u64) -> Self {
            *self.next_adapter_memory_bytes.lock().unwrap() = bytes;
            self
        }
    }

    #[async_trait::async_trait]
    impl LocalModel for MockLocalModel {
        async fn load(&self) -> Result<(), BlazenError> {
            *self.loaded.lock().unwrap() = true;
            Ok(())
        }
        async fn unload(&self) -> Result<(), BlazenError> {
            *self.loaded.lock().unwrap() = false;
            Ok(())
        }
        async fn is_loaded(&self) -> bool {
            *self.loaded.lock().unwrap()
        }
        fn device(&self) -> Device {
            self.device
        }
        async fn load_adapter(
            &self,
            adapter_dir: &std::path::Path,
            options: AdapterOptions,
        ) -> Result<AdapterHandle, BlazenError> {
            let memory_bytes = *self.next_adapter_memory_bytes.lock().unwrap();
            let status = AdapterStatus {
                adapter_id: options.adapter_id.clone(),
                scale: options.scale,
                source_dir: adapter_dir.to_path_buf(),
                memory_bytes,
            };
            self.adapters
                .lock()
                .unwrap()
                .insert(options.adapter_id.clone(), status);
            Ok(AdapterHandle {
                adapter_id: options.adapter_id,
                memory_bytes,
                mount_strategy: AdapterMountStrategy::Attached,
            })
        }
        async fn unload_adapter(&self, handle: &AdapterHandle) -> Result<(), BlazenError> {
            let mut adapters = self.adapters.lock().unwrap();
            if adapters.remove(&handle.adapter_id).is_none() {
                return Err(BlazenError::validation(format!(
                    "mock backend has no adapter '{}'",
                    handle.adapter_id
                )));
            }
            Ok(())
        }
        async fn list_adapters(&self) -> Vec<AdapterStatus> {
            self.adapters.lock().unwrap().values().cloned().collect()
        }
    }

    fn make_peft_adapter_dir(bytes_a: usize, bytes_b: usize) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("adapter_model.safetensors"),
            vec![0u8; bytes_a],
        )
        .expect("write safetensors");
        std::fs::write(dir.path().join("adapter_config.json"), vec![0u8; bytes_b])
            .expect("write config");
        dir
    }

    fn cpu_gpu_mgr(cpu_gb: u64, gpu_gb: u64) -> ModelManager {
        let mut budgets = HashMap::new();
        budgets.insert(Pool::Cpu, cpu_gb * GB);
        budgets.insert(Pool::Gpu(0), gpu_gb * GB);
        ModelManager::new(budgets)
    }

    #[tokio::test]
    async fn test_register_and_load_cpu() {
        let mgr = cpu_gpu_mgr(24, 24);
        let model = Arc::new(MockLocalModel::new(Device::Cpu));

        mgr.register("m1", model.clone(), 4 * GB).await;
        mgr.load("m1").await.unwrap();

        assert!(mgr.is_loaded("m1").await);
        assert_eq!(mgr.used_bytes(Pool::Cpu).await, 4 * GB);
        assert_eq!(mgr.used_bytes(Pool::Gpu(0)).await, 0);
    }

    #[tokio::test]
    async fn test_register_and_load_gpu() {
        let mgr = cpu_gpu_mgr(24, 24);
        let model = Arc::new(MockLocalModel::new(Device::Cuda(0)));

        mgr.register("g1", model.clone(), 8 * GB).await;
        mgr.load("g1").await.unwrap();

        assert!(mgr.is_loaded("g1").await);
        assert_eq!(mgr.used_bytes(Pool::Gpu(0)).await, 8 * GB);
        assert_eq!(mgr.used_bytes(Pool::Cpu).await, 0);
    }

    #[tokio::test]
    async fn test_lru_eviction_within_pool() {
        let mgr = cpu_gpu_mgr(20, 0);
        let m1 = Arc::new(MockLocalModel::new(Device::Cpu));
        let m2 = Arc::new(MockLocalModel::new(Device::Cpu));
        let m3 = Arc::new(MockLocalModel::new(Device::Cpu));

        mgr.register("m1", m1.clone(), 4 * GB).await;
        mgr.register("m2", m2.clone(), 8 * GB).await;
        mgr.register("m3", m3.clone(), 12 * GB).await;

        mgr.load("m1").await.unwrap();
        mgr.load("m2").await.unwrap();
        assert_eq!(mgr.used_bytes(Pool::Cpu).await, 12 * GB);

        // Loading m3 (12 GB) on top of 12 GB used in a 20 GB pool needs to
        // evict m1 (oldest LRU, 4 GB) — leaves 8 GB used, 12 GB free, fits.
        mgr.load("m3").await.unwrap();

        assert!(
            !mgr.is_loaded("m1").await,
            "m1 should have been evicted (oldest LRU)"
        );
        assert!(mgr.is_loaded("m2").await, "m2 should still be loaded");
        assert!(mgr.is_loaded("m3").await, "m3 should now be loaded");
    }

    #[tokio::test]
    async fn test_cross_pool_no_eviction() {
        // CPU pool: 16 GB, GPU pool: 16 GB.
        let mgr = cpu_gpu_mgr(16, 16);
        let cpu = Arc::new(MockLocalModel::new(Device::Cpu));
        let gpu_a = Arc::new(MockLocalModel::new(Device::Cuda(0)));
        let gpu_b = Arc::new(MockLocalModel::new(Device::Cuda(0)));

        // Fill each pool to its max.
        mgr.register("cpu", cpu.clone(), 16 * GB).await;
        mgr.register("gpu_a", gpu_a.clone(), 16 * GB).await;
        mgr.register("gpu_b", gpu_b.clone(), 16 * GB).await;

        mgr.load("cpu").await.unwrap();
        mgr.load("gpu_a").await.unwrap();

        // Loading another GPU-sized model must evict gpu_a, never cpu —
        // even though cpu is older.
        mgr.load("gpu_b").await.unwrap();

        assert!(
            mgr.is_loaded("cpu").await,
            "CPU model must NOT be evicted when a GPU model is loaded"
        );
        assert!(
            !mgr.is_loaded("gpu_a").await,
            "GPU LRU (gpu_a) should have been evicted"
        );
        assert!(mgr.is_loaded("gpu_b").await, "gpu_b should be loaded");
        assert_eq!(mgr.used_bytes(Pool::Cpu).await, 16 * GB);
        assert_eq!(mgr.used_bytes(Pool::Gpu(0)).await, 16 * GB);
    }

    #[tokio::test]
    async fn test_model_exceeds_pool_budget() {
        let mgr = cpu_gpu_mgr(24, 0);
        let model = Arc::new(MockLocalModel::new(Device::Cpu));

        mgr.register("big", model.clone(), 32 * GB).await;

        let err = mgr
            .load("big")
            .await
            .expect_err("loading a model larger than the pool budget must fail");
        let msg = err.to_string();
        let needed_bytes = (32 * GB).to_string();
        let budget_bytes = (24 * GB).to_string();
        assert!(
            msg.contains(&needed_bytes) && msg.contains(&budget_bytes),
            "error should mention both needed and budget bytes, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_unload_frees_pool_budget() {
        let mgr = cpu_gpu_mgr(0, 24);
        let model = Arc::new(MockLocalModel::new(Device::Cuda(0)));

        mgr.register("g1", model.clone(), 8 * GB).await;
        mgr.load("g1").await.unwrap();
        assert_eq!(mgr.used_bytes(Pool::Gpu(0)).await, 8 * GB);
        assert_eq!(mgr.available_bytes(Pool::Gpu(0)).await, 16 * GB);

        mgr.unload("g1").await.unwrap();
        assert_eq!(mgr.used_bytes(Pool::Gpu(0)).await, 0);
        assert_eq!(mgr.available_bytes(Pool::Gpu(0)).await, 24 * GB);
    }

    #[tokio::test]
    async fn test_status_includes_pool() {
        let mgr = cpu_gpu_mgr(24, 24);
        let cpu = Arc::new(MockLocalModel::new(Device::Cpu));
        let gpu = Arc::new(MockLocalModel::new(Device::Cuda(0)));

        mgr.register("cpu", cpu.clone(), 4 * GB).await;
        mgr.register("gpu", gpu.clone(), 8 * GB).await;
        mgr.load("cpu").await.unwrap();

        let statuses = mgr.status().await;
        assert_eq!(statuses.len(), 2);

        let cpu_status = statuses
            .iter()
            .find(|s| s.id == "cpu")
            .expect("cpu missing");
        let gpu_status = statuses
            .iter()
            .find(|s| s.id == "gpu")
            .expect("gpu missing");

        assert_eq!(cpu_status.pool, Pool::Cpu);
        assert!(cpu_status.loaded);
        assert_eq!(cpu_status.memory_estimate_bytes, 4 * GB);

        assert_eq!(gpu_status.pool, Pool::Gpu(0));
        assert!(!gpu_status.loaded);
        assert_eq!(gpu_status.memory_estimate_bytes, 8 * GB);
    }

    #[tokio::test]
    async fn test_with_budgets_gb() {
        let mgr = ModelManager::with_budgets_gb(100.0, 24.0);
        assert_eq!(mgr.available_bytes(Pool::Cpu).await, 100 * GB);
        assert_eq!(mgr.available_bytes(Pool::Gpu(0)).await, 24 * GB);
    }

    #[tokio::test]
    async fn test_unknown_pool_implicit_zero_budget() {
        // Manager only knows about Pool::Cpu and Pool::Gpu(0).
        let mgr = cpu_gpu_mgr(24, 24);
        // But this model targets Pool::Gpu(7) which is implicitly 0.
        let model = Arc::new(MockLocalModel::new(Device::Cuda(7)));

        mgr.register("orphan", model.clone(), GB).await;

        let err = mgr
            .load("orphan")
            .await
            .expect_err("model on unbudgeted pool must fail to load");
        let msg = err.to_string();
        assert!(
            msg.contains("budget is only 0 bytes"),
            "error should mention zero budget, got: {msg}"
        );
        assert!(!mgr.is_loaded("orphan").await);
        assert_eq!(mgr.available_bytes(Pool::Gpu(7)).await, 0);
    }

    #[tokio::test]
    async fn test_load_adapter_records_memory_and_status() {
        let mgr = cpu_gpu_mgr(16, 0);
        let model = Arc::new(MockLocalModel::new(Device::Cpu).with_adapter_memory(2 * GB));
        mgr.register("base", model.clone(), 4 * GB).await;
        mgr.load("base").await.unwrap();

        let dir = make_peft_adapter_dir(1024, 256);
        let handle = mgr
            .load_adapter("base", dir.path(), AdapterOptions::new("a1"))
            .await
            .expect("load_adapter should succeed");

        assert_eq!(handle.adapter_id, "a1");
        assert_eq!(handle.memory_bytes, 2 * GB);
        assert_eq!(handle.mount_strategy, AdapterMountStrategy::Attached);

        let listed = mgr.list_adapters("base").await.unwrap();
        assert_eq!(listed.len(), 1);
        let entry = &listed[0];
        assert_eq!(entry.adapter_id, "a1");
        assert!((entry.scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(entry.source_dir, dir.path().to_path_buf());
        assert_eq!(entry.memory_bytes, 2 * GB);

        let statuses = mgr.status().await;
        let base_status = statuses.iter().find(|s| s.id == "base").expect("base");
        assert_eq!(base_status.memory_estimate_bytes, 4 * GB + 2 * GB);
        assert_eq!(base_status.adapters.len(), 1);
    }

    #[tokio::test]
    async fn test_load_adapter_refuses_duplicate_id() {
        let mgr = cpu_gpu_mgr(16, 0);
        let model = Arc::new(MockLocalModel::new(Device::Cpu).with_adapter_memory(GB));
        mgr.register("base", model.clone(), 4 * GB).await;
        mgr.load("base").await.unwrap();

        let dir = make_peft_adapter_dir(1024, 256);
        mgr.load_adapter("base", dir.path(), AdapterOptions::new("dup"))
            .await
            .expect("first mount should succeed");

        let dir2 = make_peft_adapter_dir(1024, 256);
        let err = mgr
            .load_adapter("base", dir2.path(), AdapterOptions::new("dup"))
            .await
            .expect_err("second mount with same id must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("already mounted"),
            "expected 'already mounted' in error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_load_adapter_fails_if_pool_full() {
        let mut budgets = HashMap::new();
        budgets.insert(Pool::Cpu, 8 * 1024_u64);
        budgets.insert(Pool::Gpu(0), 0);
        let mgr = ModelManager::new(budgets);

        let model = Arc::new(MockLocalModel::new(Device::Cpu));
        mgr.register("base", model.clone(), 4 * 1024).await;
        mgr.load("base").await.unwrap();

        let dir = make_peft_adapter_dir(4 * 1024, 4 * 1024);
        let err = mgr
            .load_adapter("base", dir.path(), AdapterOptions::new("oversized"))
            .await
            .expect_err("mounting an adapter that exceeds budget must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("would exceed pool"),
            "expected 'would exceed pool' in error, got: {msg}"
        );
        assert!(
            msg.contains("CPU") || msg.contains("cpu") || msg.contains("Cpu"),
            "expected pool name in error, got: {msg}"
        );

        assert!(mgr.list_adapters("base").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_unload_adapter_decrements_memory() {
        let mgr = cpu_gpu_mgr(16, 0);
        let model = Arc::new(MockLocalModel::new(Device::Cpu).with_adapter_memory(2 * GB));
        mgr.register("base", model.clone(), 4 * GB).await;
        mgr.load("base").await.unwrap();

        let dir = make_peft_adapter_dir(1024, 256);
        mgr.load_adapter("base", dir.path(), AdapterOptions::new("rm"))
            .await
            .unwrap();

        let statuses = mgr.status().await;
        let base = statuses.iter().find(|s| s.id == "base").expect("base");
        assert_eq!(base.memory_estimate_bytes, 4 * GB + 2 * GB);

        mgr.unload_adapter("base", "rm").await.unwrap();

        let statuses = mgr.status().await;
        let base = statuses.iter().find(|s| s.id == "base").expect("base");
        assert_eq!(base.memory_estimate_bytes, 4 * GB);
        assert!(base.adapters.is_empty());

        let err = mgr
            .unload_adapter("base", "rm")
            .await
            .expect_err("second unload must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("is not mounted"),
            "expected 'is not mounted' in error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_list_adapters_reflects_state() {
        use std::collections::HashSet;

        let mgr = cpu_gpu_mgr(16, 0);
        let model = Arc::new(MockLocalModel::new(Device::Cpu).with_adapter_memory(1024));
        mgr.register("base", model.clone(), 4 * GB).await;
        mgr.load("base").await.unwrap();

        for id in ["a", "b", "c"] {
            let dir = make_peft_adapter_dir(512, 128);
            mgr.load_adapter("base", dir.path(), AdapterOptions::new(id))
                .await
                .unwrap();
        }

        let list = mgr.list_adapters("base").await.unwrap();
        assert_eq!(list.len(), 3);
        let ids: HashSet<String> = list.iter().map(|a| a.adapter_id.clone()).collect();
        assert!(ids.contains("a") && ids.contains("b") && ids.contains("c"));

        mgr.unload_adapter("base", "b").await.unwrap();

        let list = mgr.list_adapters("base").await.unwrap();
        assert_eq!(list.len(), 2);
        let ids: HashSet<String> = list.iter().map(|a| a.adapter_id.clone()).collect();
        assert!(ids.contains("a") && ids.contains("c"));
        assert!(!ids.contains("b"));

        let err = mgr
            .list_adapters("nonexistent")
            .await
            .expect_err("list_adapters on unknown model must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("not registered"),
            "expected 'not registered' in error, got: {msg}"
        );
    }
}
