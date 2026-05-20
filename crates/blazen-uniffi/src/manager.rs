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

use crate::errors::{BlazenError, BlazenResult};
use crate::local_model::{
    AdapterOptionsRecord, AdapterStatusRecord, ForeignLocalModel, ForeignLocalModelAdapter,
};
use crate::runtime::runtime;
use blazen_llm::Pool;
use blazen_manager::ModelManager;
#[cfg(feature = "hf-loader")]
use blazen_manager::hf_loader::{BackendHint, HfLoadOptions};

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

/// Local-inference backend identifier returned by
/// [`UniffiModelManager::load_from_hf`] and accepted as a forced override on
/// [`HfLoadOptionsRecord::backend_hint`].
///
/// Mirrors [`blazen_manager::hf_loader::BackendHint`] as a UniFFI Enum.
#[cfg(feature = "hf-loader")]
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum BackendHintEnum {
    /// `mistral.rs` — broad architecture coverage, handles both safetensors
    /// and GGUF, supports vision/multimodal models.
    Mistralrs,
    /// `candle` — pure-Rust, supports safetensors and GGUF for the subset of
    /// architectures candle ships.
    Candle,
    /// `llama.cpp` — GGUF only, best CPU performance and lowest memory.
    Llamacpp,
}

#[cfg(feature = "hf-loader")]
impl From<BackendHintEnum> for BackendHint {
    fn from(h: BackendHintEnum) -> Self {
        match h {
            BackendHintEnum::Mistralrs => Self::Mistralrs,
            BackendHintEnum::Candle => Self::Candle,
            BackendHintEnum::Llamacpp => Self::Llamacpp,
        }
    }
}

#[cfg(feature = "hf-loader")]
impl From<BackendHint> for BackendHintEnum {
    fn from(h: BackendHint) -> Self {
        match h {
            BackendHint::Mistralrs => Self::Mistralrs,
            BackendHint::Candle => Self::Candle,
            BackendHint::Llamacpp => Self::Llamacpp,
        }
    }
}

/// Caller-supplied options for [`UniffiModelManager::load_from_hf`].
///
/// Mirrors [`blazen_manager::hf_loader::HfLoadOptions`]; every field is
/// optional. `pool` is a label (`"cpu"`, `"gpu"`, `"gpu:N"`) and is parsed
/// by `parse_pool_label`.
#[cfg(feature = "hf-loader")]
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct HfLoadOptionsRecord {
    /// Force a specific backend; skips engine inference but still probes
    /// the repo for memory sizing.
    pub backend_hint: Option<BackendHintEnum>,
    /// Git revision (branch, tag, or commit sha). Defaults to the repo's
    /// default branch.
    pub revision: Option<String>,
    /// Hugging Face access token. When omitted, falls back to the
    /// `HF_TOKEN` environment variable, then to anonymous access.
    pub hf_token: Option<String>,
    /// Override the on-disk cache directory used by `hf-hub`.
    pub cache_dir: Option<String>,
    /// Device specifier forwarded to the chosen provider (`"cpu"`,
    /// `"cuda:0"`, `"metal"`, …).
    pub device: Option<String>,
    /// Explicit GGUF filename for repos that ship multiple quantizations.
    pub gguf_file: Option<String>,
    /// Override the auto-derived memory estimate, in bytes.
    pub memory_estimate_bytes: Option<u64>,
    /// Pool label (`"cpu"`, `"gpu"`, `"gpu:N"`). Defaults to `"cpu"`.
    pub pool: Option<String>,
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

    /// Probe a Hugging Face repo, pick a local-inference backend, build the
    /// provider, and register it under `id`.
    ///
    /// Returns the chosen backend as a lower-case stable string
    /// (`"mistralrs"` / `"candle"` / `"llamacpp"`). The model starts unloaded
    /// — call [`Self::load`] or [`Self::ensure_loaded`] to materialize it.
    ///
    /// Errors on empty repo id, gated/missing repo, PEFT-adapter-only repo
    /// (use [`Self::load_adapter`] instead), missing backend feature, or any
    /// provider construction failure.
    #[cfg(feature = "hf-loader")]
    pub async fn load_from_hf(
        self: Arc<Self>,
        id: String,
        repo: String,
        options: HfLoadOptionsRecord,
    ) -> BlazenResult<String> {
        let pool = match options.pool.as_deref() {
            Some(label) => Some(parse_pool_label(label)?),
            None => None,
        };
        let rust_opts = HfLoadOptions {
            backend_hint: options.backend_hint.map(BackendHint::from),
            revision: options.revision,
            hf_token: options.hf_token,
            cache_dir: options.cache_dir.map(PathBuf::from),
            device: options.device,
            gguf_file: options.gguf_file,
            memory_estimate_bytes: options.memory_estimate_bytes,
            pool,
        };
        let backend = self
            .inner
            .load_from_hf(id, &repo, rust_opts)
            .await
            .map_err(BlazenError::from)?;
        Ok(backend.as_str().to_owned())
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
