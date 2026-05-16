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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use blazen_llm::{BlazenError, LocalModel, Pool};
use tokio::sync::Mutex;

/// Status of a registered model.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    /// Identifier of the model.
    pub id: String,
    /// Whether the model is currently loaded.
    pub loaded: bool,
    /// Estimated memory footprint in bytes.
    pub memory_estimate_bytes: u64,
    /// Pool this model is registered against.
    pub pool: Pool,
}

struct RegisteredModel {
    model: Arc<dyn LocalModel>,
    memory_estimate_bytes: u64,
    pool: Pool,
    loaded: bool,
    last_used: Option<Instant>,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_llm::Device;
    use std::sync::Mutex as StdMutex;

    const GB: u64 = 1_073_741_824;

    struct MockLocalModel {
        loaded: StdMutex<bool>,
        device: Device,
    }

    impl MockLocalModel {
        fn new(device: Device) -> Self {
            Self {
                loaded: StdMutex::new(false),
                device,
            }
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
}
