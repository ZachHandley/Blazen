//! Memory-budget-aware model manager surface for the UniFFI bindings.
//!
//! Mirrors `crates/blazen-py/src/manager.rs` verb-for-verb: register a
//! foreign-implemented [`crate::local_model::ForeignLocalModel`] against a
//! [`UniffiModelManager`], then `load` / `unload` / `is_loaded` / `status` /
//! `pools` it, and mount PEFT LoRA adapters via `load_adapter` /
//! `unload_adapter` / `list_adapters`.
//!
//! Foreign callers don't see the upstream [`blazen_manager::ModelManager`] —
//! they get this opaque [`UniffiModelManager`] handle plus the
//! `ModelStatusRecord` / `PoolStatusRecord` / `AdapterStatusRecord` Records
//! defined here and in `local_model.rs`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use blazen_llm::Pool;
use blazen_manager::ModelManager;

use crate::errors::{BlazenError, BlazenResult};
use crate::local_model::{
    AdapterOptionsRecord, AdapterStatusRecord, ForeignLocalModel, ForeignLocalModelAdapter,
};
use crate::runtime::runtime;

/// Per-model state snapshot returned by [`UniffiModelManager::status`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelStatusRecord {
    pub id: String,
    pub loaded: bool,
    pub memory_estimate_bytes: u64,
    /// Pool label (`"cpu"` or `"gpu:N"`).
    pub pool: String,
    pub adapters: Vec<AdapterStatusRecord>,
}

/// Per-pool budget snapshot returned by [`UniffiModelManager::pools`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct PoolStatusRecord {
    /// Pool label (`"cpu"` or `"gpu:N"`).
    pub pool: String,
    pub budget_bytes: u64,
}

/// Parse a pool label (`"cpu"`, `"gpu"`, `"gpu:N"`) into a [`Pool`].
///
/// Mirrors `parse_pool_label` in the Python binding.
fn parse_pool_label(s: &str) -> BlazenResult<Pool> {
    let trimmed = s.trim();
    let lower = trimmed.to_ascii_lowercase();

    if let Some((name, idx)) = lower.split_once(':') {
        if name == "gpu" {
            let index = idx.parse::<usize>().map_err(|_| BlazenError::Validation {
                message: format!(
                    "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
                ),
            })?;
            return Ok(Pool::Gpu(index));
        }
        return Err(BlazenError::Validation {
            message: format!(
                "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
            ),
        });
    }
    match lower.as_str() {
        "cpu" => Ok(Pool::Cpu),
        "gpu" => Ok(Pool::Gpu(0)),
        _ => Err(BlazenError::Validation {
            message: format!(
                "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
            ),
        }),
    }
}

/// Memory-budget-aware model manager with per-pool LRU eviction.
///
/// Foreign code constructs one of these, registers
/// [`ForeignLocalModel`]-implementing handles against it, and drives loads /
/// unloads / adapter lifecycle from any thread / fiber / goroutine /
/// coroutine on the foreign side.
#[derive(uniffi::Object)]
pub struct UniffiModelManager {
    inner: Arc<ModelManager>,
}

#[uniffi::export]
impl UniffiModelManager {
    /// Construct a manager with no budget enforcement (both `Cpu` and
    /// `Gpu(0)` pools seeded with `u64::MAX`).
    ///
    /// Matches the Python binding's `ModelManager()` no-arg sentinel
    /// behaviour. For real deployments, prefer
    /// [`Self::with_budgets_gb`](Self::with_budgets_gb) or
    /// [`Self::with_pool_budgets`](Self::with_pool_budgets).
    #[uniffi::constructor]
    #[must_use]
    pub fn new() -> Arc<Self> {
        let mut budgets = HashMap::new();
        budgets.insert(Pool::Cpu, u64::MAX);
        budgets.insert(Pool::Gpu(0), u64::MAX);
        Arc::new(Self {
            inner: Arc::new(ModelManager::new(budgets)),
        })
    }

    /// Construct a manager with one CPU-pool budget and one GPU-pool
    /// (`Gpu(0)`) budget, both expressed in gigabytes.
    #[uniffi::constructor]
    #[must_use]
    pub fn with_budgets_gb(cpu_ram_gb: f64, gpu_vram_gb: f64) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(ModelManager::with_budgets_gb(cpu_ram_gb, gpu_vram_gb)),
        })
    }

    /// Construct a manager with explicit per-pool budgets.
    ///
    /// Keys are pool labels (`"cpu"`, `"gpu"`, `"gpu:0"`, `"gpu:1"`, ...);
    /// values are budgets in **gigabytes** (mirrors the Python binding's
    /// `pool_budgets` ergonomics — bytes-as-`u64` would force foreign
    /// callers to write `64 * 1024 * 1024 * 1024` for trivial values).
    #[uniffi::constructor]
    pub fn with_pool_budgets(per_pool_budgets: HashMap<String, f64>) -> BlazenResult<Arc<Self>> {
        let mut budgets = HashMap::with_capacity(per_pool_budgets.len());
        for (label, gb) in per_pool_budgets {
            let pool = parse_pool_label(&label)?;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let bytes = (gb * 1_073_741_824.0) as u64;
            budgets.insert(pool, bytes);
        }
        Ok(Arc::new(Self {
            inner: Arc::new(ModelManager::new(budgets)),
        }))
    }

    /// List configured pools and their budgets in bytes.
    #[must_use]
    pub fn pools(&self) -> Vec<PoolStatusRecord> {
        self.inner
            .pools()
            .into_iter()
            .map(|(pool, budget)| PoolStatusRecord {
                pool: pool.to_string(),
                budget_bytes: budget,
            })
            .collect()
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl UniffiModelManager {
    /// Register a foreign-implemented [`ForeignLocalModel`] under `id`.
    ///
    /// `memory_estimate_bytes` is the model's estimated footprint and is
    /// charged against the pool derived from the foreign model's `device()`
    /// when it's loaded.
    pub async fn register_local(
        self: Arc<Self>,
        id: String,
        model: Arc<dyn ForeignLocalModel>,
        memory_estimate_bytes: u64,
    ) -> BlazenResult<()> {
        let adapter: Arc<dyn blazen_llm::LocalModel> =
            Arc::new(ForeignLocalModelAdapter::new(model));
        self.inner
            .register(&id, adapter, memory_estimate_bytes)
            .await;
        Ok(())
    }

    pub async fn load(self: Arc<Self>, model_id: String) -> BlazenResult<()> {
        self.inner.load(&model_id).await.map_err(BlazenError::from)
    }

    pub async fn unload(self: Arc<Self>, model_id: String) -> BlazenResult<()> {
        self.inner
            .unload(&model_id)
            .await
            .map_err(BlazenError::from)
    }

    pub async fn is_loaded(self: Arc<Self>, model_id: String) -> bool {
        self.inner.is_loaded(&model_id).await
    }

    pub async fn ensure_loaded(self: Arc<Self>, model_id: String) -> BlazenResult<()> {
        self.inner
            .ensure_loaded(&model_id)
            .await
            .map_err(BlazenError::from)
    }

    pub async fn status(self: Arc<Self>) -> Vec<ModelStatusRecord> {
        self.inner
            .status()
            .await
            .into_iter()
            .map(|s| ModelStatusRecord {
                id: s.id,
                loaded: s.loaded,
                memory_estimate_bytes: s.memory_estimate_bytes,
                pool: s.pool.to_string(),
                adapters: s
                    .adapters
                    .into_iter()
                    .map(AdapterStatusRecord::from)
                    .collect(),
            })
            .collect()
    }

    pub async fn used_bytes(self: Arc<Self>, pool: String) -> BlazenResult<u64> {
        let pool = parse_pool_label(&pool)?;
        Ok(self.inner.used_bytes(pool).await)
    }

    pub async fn available_bytes(self: Arc<Self>, pool: String) -> BlazenResult<u64> {
        let pool = parse_pool_label(&pool)?;
        Ok(self.inner.available_bytes(pool).await)
    }

    /// Mount a PEFT-format LoRA adapter and return the adapter id reported
    /// by the backend.
    pub async fn load_adapter(
        self: Arc<Self>,
        model_id: String,
        adapter_dir: String,
        options: AdapterOptionsRecord,
    ) -> BlazenResult<String> {
        let path = PathBuf::from(adapter_dir);
        let handle = self
            .inner
            .load_adapter(&model_id, &path, options.into())
            .await
            .map_err(BlazenError::from)?;
        Ok(handle.adapter_id)
    }

    pub async fn unload_adapter(
        self: Arc<Self>,
        model_id: String,
        adapter_id: String,
    ) -> BlazenResult<()> {
        self.inner
            .unload_adapter(&model_id, &adapter_id)
            .await
            .map_err(BlazenError::from)
    }

    pub async fn list_adapters(
        self: Arc<Self>,
        model_id: String,
    ) -> BlazenResult<Vec<AdapterStatusRecord>> {
        let statuses = self
            .inner
            .list_adapters(&model_id)
            .await
            .map_err(BlazenError::from)?;
        Ok(statuses
            .into_iter()
            .map(AdapterStatusRecord::from)
            .collect())
    }
}

#[uniffi::export]
impl UniffiModelManager {
    /// Synchronous variant of [`Self::load`] — blocks the current thread on
    /// the shared Tokio runtime.
    pub fn load_blocking(self: Arc<Self>, model_id: String) -> BlazenResult<()> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.load(model_id).await })
    }

    /// Synchronous variant of [`Self::unload`].
    pub fn unload_blocking(self: Arc<Self>, model_id: String) -> BlazenResult<()> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.unload(model_id).await })
    }
}
