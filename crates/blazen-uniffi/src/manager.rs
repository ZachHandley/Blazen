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

// ---------------------------------------------------------------------------
// Training surface (feature = "training")
//
// Mirrors `crates/blazen-py/src/manager.rs` Wave 3C. The Rust `TrainingProgress`
// trait is sync, the trainer drives it from a tokio worker via
// `Trainer::run`, and the foreign callback (`ForeignTrainingProgress`) is
// likewise modeled SYNC on the UniFFI surface. Why sync rather than async:
// `TrainingProgress::on_event` is itself sync, so an async foreign callback
// would force us to `block_on(...)` from a tokio worker thread — which
// panics for `Handle::block_on` and risks deadlocks for
// `futures::executor::block_on` once the foreign side schedules onto the
// same runtime. Sync foreign methods compose cleanly with `with_foreign`
// on every target language (Go: blocking goroutine; Swift / Kotlin: regular
// function; Ruby: blocking call) and match how `PyTrainingProgressBridge`
// already pumps events under the GIL.
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
pub use training::{
    DpoConfigRecord, ForeignTrainingProgress, FullFineTuneConfigRecord, FullFineTuneResultRecord,
    KtoConfigRecord, LoraConfigRecord, MixedPrecisionEnum, OptimConfigRecord, OrpoConfigRecord,
    SchedulerConfigRecord, SchedulerKindEnum, SimpoConfigRecord, TrainConfigRecord,
    TrainCoreConfigRecord, TrainedAdapterRecord, TrainingEventEnum, UniffiJsonlDataset,
    UniffiPreferenceJsonlDataset, UniffiRatedJsonlDataset,
};

#[cfg(feature = "training")]
mod training {
    use std::path::PathBuf;
    use std::sync::Arc;

    use async_trait::async_trait;

    use blazen_train::dataset::{JsonlDataset, PreferenceJsonlDataset, RatedJsonlDataset};
    use blazen_train::{
        BlazenTrainError, DpoConfig, FullFineTuneConfig, FullFineTuneResult, KtoConfig, LoraConfig,
        MixedPrecision, OptimConfig, OrpoConfig, PreferenceDataset, RatedDataset, SchedulerConfig,
        SchedulerKind, SimpoConfig, TrainConfig, TrainCoreConfig, TrainedAdapter, TrainingBatch,
        TrainingDataset, TrainingEvent, TrainingProgress,
    };
    use tokenizers::Tokenizer;

    use crate::errors::{BlazenError, BlazenResult};

    use super::UniffiModelManager;

    #[derive(Debug, Clone, Copy, uniffi::Enum)]
    pub enum SchedulerKindEnum {
        Constant,
        Linear,
        Cosine,
    }

    impl From<SchedulerKindEnum> for SchedulerKind {
        fn from(k: SchedulerKindEnum) -> Self {
            match k {
                SchedulerKindEnum::Constant => Self::Constant,
                SchedulerKindEnum::Linear => Self::Linear,
                SchedulerKindEnum::Cosine => Self::Cosine,
            }
        }
    }

    impl From<SchedulerKind> for SchedulerKindEnum {
        fn from(k: SchedulerKind) -> Self {
            match k {
                SchedulerKind::Constant => Self::Constant,
                SchedulerKind::Linear => Self::Linear,
                SchedulerKind::Cosine => Self::Cosine,
            }
        }
    }

    #[derive(Debug, Clone, Copy, uniffi::Enum)]
    pub enum MixedPrecisionEnum {
        None,
        Bf16,
    }

    impl From<MixedPrecisionEnum> for MixedPrecision {
        fn from(m: MixedPrecisionEnum) -> Self {
            match m {
                MixedPrecisionEnum::None => Self::None,
                MixedPrecisionEnum::Bf16 => Self::Bf16,
            }
        }
    }

    impl From<MixedPrecision> for MixedPrecisionEnum {
        fn from(m: MixedPrecision) -> Self {
            match m {
                MixedPrecision::None => Self::None,
                MixedPrecision::Bf16 => Self::Bf16,
            }
        }
    }

    /// LoRA hyperparameters.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct LoraConfigRecord {
        pub rank: u32,
        pub alpha: f32,
        pub dropout: f32,
        pub target_modules: Vec<String>,
    }

    impl From<LoraConfigRecord> for LoraConfig {
        fn from(c: LoraConfigRecord) -> Self {
            Self {
                rank: c.rank as usize,
                alpha: c.alpha,
                dropout: c.dropout,
                target_modules: c.target_modules,
            }
        }
    }

    impl From<LoraConfig> for LoraConfigRecord {
        fn from(c: LoraConfig) -> Self {
            Self {
                rank: u32::try_from(c.rank).unwrap_or(u32::MAX),
                alpha: c.alpha,
                dropout: c.dropout,
                target_modules: c.target_modules,
            }
        }
    }

    /// AdamW optimizer hyperparameters.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct OptimConfigRecord {
        pub learning_rate: f64,
        pub beta1: f64,
        pub beta2: f64,
        pub epsilon: f64,
        pub weight_decay: f64,
        pub gradient_clip: Option<f32>,
    }

    impl From<OptimConfigRecord> for OptimConfig {
        fn from(c: OptimConfigRecord) -> Self {
            Self {
                learning_rate: c.learning_rate,
                beta1: c.beta1,
                beta2: c.beta2,
                epsilon: c.epsilon,
                weight_decay: c.weight_decay,
                gradient_clip: c.gradient_clip,
            }
        }
    }

    impl From<OptimConfig> for OptimConfigRecord {
        fn from(c: OptimConfig) -> Self {
            Self {
                learning_rate: c.learning_rate,
                beta1: c.beta1,
                beta2: c.beta2,
                epsilon: c.epsilon,
                weight_decay: c.weight_decay,
                gradient_clip: c.gradient_clip,
            }
        }
    }

    /// Learning-rate scheduler configuration.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct SchedulerConfigRecord {
        pub kind: SchedulerKindEnum,
        pub warmup_steps: u32,
    }

    impl From<SchedulerConfigRecord> for SchedulerConfig {
        fn from(c: SchedulerConfigRecord) -> Self {
            Self {
                kind: c.kind.into(),
                warmup_steps: c.warmup_steps as usize,
            }
        }
    }

    impl From<SchedulerConfig> for SchedulerConfigRecord {
        fn from(c: SchedulerConfig) -> Self {
            Self {
                kind: c.kind.into(),
                warmup_steps: u32::try_from(c.warmup_steps).unwrap_or(u32::MAX),
            }
        }
    }

    /// Full configuration for one training run.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct TrainConfigRecord {
        pub base_model_repo: String,
        pub output_dir: String,
        pub lora: LoraConfigRecord,
        pub optim: OptimConfigRecord,
        pub scheduler: SchedulerConfigRecord,
        pub max_steps: u32,
        pub batch_size: u32,
        pub gradient_accumulation_steps: u32,
        pub max_seq_len: u32,
        pub eval_steps: Option<u32>,
        pub save_steps: Option<u32>,
        pub seed: u64,
        pub mixed_precision: MixedPrecisionEnum,
        pub device: Option<String>,
    }

    impl From<TrainConfigRecord> for TrainConfig {
        fn from(c: TrainConfigRecord) -> Self {
            Self {
                base_model_repo: c.base_model_repo,
                output_dir: PathBuf::from(c.output_dir),
                lora: c.lora.into(),
                optim: c.optim.into(),
                scheduler: c.scheduler.into(),
                max_steps: c.max_steps as usize,
                batch_size: c.batch_size as usize,
                gradient_accumulation_steps: c.gradient_accumulation_steps as usize,
                max_seq_len: c.max_seq_len as usize,
                eval_steps: c.eval_steps.map(|v| v as usize),
                save_steps: c.save_steps.map(|v| v as usize),
                seed: c.seed,
                mixed_precision: c.mixed_precision.into(),
                device: c.device,
            }
        }
    }

    impl From<TrainConfig> for TrainConfigRecord {
        fn from(c: TrainConfig) -> Self {
            Self {
                base_model_repo: c.base_model_repo,
                output_dir: c.output_dir.display().to_string(),
                lora: c.lora.into(),
                optim: c.optim.into(),
                scheduler: c.scheduler.into(),
                max_steps: u32::try_from(c.max_steps).unwrap_or(u32::MAX),
                batch_size: u32::try_from(c.batch_size).unwrap_or(u32::MAX),
                gradient_accumulation_steps: u32::try_from(c.gradient_accumulation_steps)
                    .unwrap_or(u32::MAX),
                max_seq_len: u32::try_from(c.max_seq_len).unwrap_or(u32::MAX),
                eval_steps: c.eval_steps.map(|v| u32::try_from(v).unwrap_or(u32::MAX)),
                save_steps: c.save_steps.map(|v| u32::try_from(v).unwrap_or(u32::MAX)),
                seed: c.seed,
                mixed_precision: c.mixed_precision.into(),
                device: c.device,
            }
        }
    }

    /// On-disk descriptor returned by [`UniffiModelManager::train_lora`].
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct TrainedAdapterRecord {
        pub adapter_dir: String,
        pub final_loss: f32,
        pub total_steps: u64,
    }

    impl From<TrainedAdapter> for TrainedAdapterRecord {
        fn from(a: TrainedAdapter) -> Self {
            Self {
                adapter_dir: a.adapter_dir.display().to_string(),
                final_loss: a.final_loss,
                total_steps: a.total_steps as u64,
            }
        }
    }

    /// One observable event emitted during a training run.
    #[derive(Debug, Clone, uniffi::Enum)]
    pub enum TrainingEventEnum {
        Started {
            total_steps: u64,
        },
        StepCompleted {
            step: u64,
            loss: f32,
            learning_rate: f64,
            elapsed_ms: u64,
        },
        Evaluating {
            step: u64,
        },
        EvalCompleted {
            step: u64,
            eval_loss: f32,
        },
        CheckpointSaved {
            step: u64,
            path: String,
        },
        Finished {
            final_loss: f32,
            total_steps: u64,
            adapter_dir: String,
        },
    }

    impl From<TrainingEvent> for TrainingEventEnum {
        fn from(ev: TrainingEvent) -> Self {
            match ev {
                TrainingEvent::Started { total_steps } => Self::Started {
                    total_steps: total_steps as u64,
                },
                TrainingEvent::StepCompleted {
                    step,
                    loss,
                    learning_rate,
                    elapsed,
                } => Self::StepCompleted {
                    step: step as u64,
                    loss,
                    learning_rate,
                    elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                },
                TrainingEvent::Evaluating { step } => Self::Evaluating { step: step as u64 },
                TrainingEvent::EvalCompleted { step, eval_loss } => Self::EvalCompleted {
                    step: step as u64,
                    eval_loss,
                },
                TrainingEvent::CheckpointSaved { step, path } => Self::CheckpointSaved {
                    step: step as u64,
                    path: path.display().to_string(),
                },
                TrainingEvent::Finished {
                    final_loss,
                    total_steps,
                    adapter_dir,
                } => Self::Finished {
                    final_loss,
                    total_steps: total_steps as u64,
                    adapter_dir: adapter_dir.display().to_string(),
                },
            }
        }
    }

    /// Foreign-implementable training-progress sink.
    ///
    /// Modeled SYNC so the bridge to [`TrainingProgress`] (also sync) is
    /// trivial — the upstream trainer calls `on_event` from a tokio worker
    /// and an async foreign hop would require `block_on` from inside the
    /// same runtime (panic / deadlock-prone). Returning `Err(_)` cancels
    /// the run; the trainer surfaces it as `BlazenError::Cancelled`.
    #[uniffi::export(with_foreign)]
    pub trait ForeignTrainingProgress: Send + Sync {
        fn on_event(&self, event: TrainingEventEnum) -> BlazenResult<()>;
    }

    /// Bridge between sync `TrainingProgress` and the (sync) foreign trait.
    struct ForeignProgressAdapter {
        inner: Arc<dyn ForeignTrainingProgress>,
    }

    impl TrainingProgress for ForeignProgressAdapter {
        fn on_event(&self, event: TrainingEvent) -> Result<(), BlazenTrainError> {
            let mapped = TrainingEventEnum::from(event);
            match self.inner.on_event(mapped) {
                Ok(()) => Ok(()),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "training progress callback returned error; aborting run",
                    );
                    Err(BlazenTrainError::Cancelled)
                }
            }
        }
    }

    /// JSONL-backed training dataset opaque handle.
    ///
    /// Construct via [`UniffiJsonlDataset::from_path`] and pass to
    /// [`UniffiModelManager::train_lora`]. The dataset is reference-counted
    /// (`Arc`-shared), so foreign callers can keep a handle around and
    /// re-use it across multiple training runs.
    #[derive(uniffi::Object)]
    pub struct UniffiJsonlDataset {
        inner: Arc<JsonlDataset>,
    }

    #[uniffi::export]
    impl UniffiJsonlDataset {
        /// Load a JSONL training file using the tokenizer at `tokenizer_path`.
        ///
        /// `chat_template` is optional Jinja2 from `tokenizer_config.json`;
        /// required if any row uses the OpenAI `messages` shape.
        /// `device` matches the trainer device strings — `"cpu"`,
        /// `"cuda"` / `"cuda:N"`, `"metal"` / `"metal:N"` (default `"cpu"`).
        #[uniffi::constructor]
        pub fn from_path(
            path: String,
            tokenizer_path: String,
            chat_template: Option<String>,
            max_seq_len: u32,
            device: Option<String>,
            pad_token_id: u32,
        ) -> BlazenResult<Arc<Self>> {
            if max_seq_len == 0 {
                return Err(BlazenError::Validation {
                    message: "JsonlDataset.max_seq_len must be > 0".into(),
                });
            }
            let tokenizer =
                Tokenizer::from_file(&tokenizer_path).map_err(|e| BlazenError::Validation {
                    message: format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
                })?;
            let device_str = device.as_deref().unwrap_or("cpu");
            let cdev = parse_train_device(device_str)?;
            let ds = JsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                chat_template.as_deref(),
                max_seq_len as usize,
                cdev,
                pad_token_id,
            )
            .map_err(|e| BlazenError::Validation {
                message: format!("JsonlDataset load failed: {e}"),
            })?;
            Ok(Arc::new(Self {
                inner: Arc::new(ds),
            }))
        }

        /// Number of examples in the dataset.
        pub fn len(&self) -> u64 {
            self.inner.len() as u64
        }

        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
    }

    /// Adapter so an `Arc<JsonlDataset>` satisfies `Box<dyn TrainingDataset>`.
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

    fn parse_train_device(spec: &str) -> BlazenResult<candle_core::Device> {
        let normalized = spec.trim().to_ascii_lowercase();
        if normalized == "cpu" {
            return Ok(candle_core::Device::Cpu);
        }
        let (kind, idx) = match normalized.split_once(':') {
            Some((k, rest)) => {
                let parsed = rest.parse::<usize>().map_err(|e| BlazenError::Validation {
                    message: format!(
                        "training device {spec:?} has non-numeric index {rest:?}: {e}"
                    ),
                })?;
                (k, parsed)
            }
            None => (normalized.as_str(), 0),
        };
        match kind {
            "cuda" => candle_core::Device::new_cuda(idx).map_err(|e| BlazenError::Validation {
                message: format!("cuda:{idx} unavailable: {e}"),
            }),
            "metal" => candle_core::Device::new_metal(idx).map_err(|e| BlazenError::Validation {
                message: format!("metal:{idx} unavailable: {e}"),
            }),
            other => Err(BlazenError::Validation {
                message: format!(
                    "unknown training device {other:?} (want one of: cpu, cuda[:N], metal[:N])"
                ),
            }),
        }
    }

    #[uniffi::export(async_runtime = "tokio")]
    impl UniffiModelManager {
        /// Train a LoRA adapter end-to-end on the configured base model.
        ///
        /// Downloads the base model from HuggingFace (cached), runs the
        /// AdamW + LoRA training loop driven by `dataset`, and writes the
        /// resulting PEFT-format adapter to `config.output_dir`. The
        /// returned [`TrainedAdapterRecord`] points at an on-disk adapter
        /// directory that's immediately mountable via
        /// [`UniffiModelManager::load_adapter`] on a compatible backend.
        ///
        /// If `progress` is provided, its `on_event` is called for each
        /// Started / StepCompleted / Evaluating / EvalCompleted /
        /// CheckpointSaved / Finished transition. Returning `Err(_)` from
        /// the callback cancels the run with [`BlazenError::Cancelled`].
        pub async fn train_lora(
            self: Arc<Self>,
            config: TrainConfigRecord,
            dataset: Arc<UniffiJsonlDataset>,
            progress: Option<Arc<dyn ForeignTrainingProgress>>,
        ) -> BlazenResult<TrainedAdapterRecord> {
            let rust_cfg: TrainConfig = config.into();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|p| {
                Arc::new(ForeignProgressAdapter { inner: p }) as Arc<dyn TrainingProgress>
            });
            let dataset_box: Box<dyn TrainingDataset> =
                Box::new(ArcDataset(Arc::clone(&dataset.inner)));
            let adapter = self
                .inner
                .train_lora(rust_cfg, dataset_box, sink)
                .await
                .map_err(BlazenError::from)?;
            Ok(TrainedAdapterRecord::from(adapter))
        }
    }

    // -----------------------------------------------------------------------
    // PR8 Wave 14 — DPO / ORPO / SimPO / KTO / full fine-tune
    //
    // `TrainCoreConfig` is the SFT-free hyperparameter bundle shared by all
    // four preference-optimization verbs plus full fine-tune. Defaults track
    // upstream `blazen_train::config::TrainCoreConfig::default`.
    // -----------------------------------------------------------------------

    /// Shared training hyperparameters used by DPO / ORPO / SimPO / KTO /
    /// full fine-tune. Mirrors [`TrainCoreConfig`] (`TrainConfig` minus the
    /// PEFT-specific [`LoraConfigRecord`]).
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct TrainCoreConfigRecord {
        pub base_model_repo: String,
        #[uniffi(default = None)]
        pub base_model_revision: Option<String>,
        pub output_dir: String,
        #[uniffi(default = 1000)]
        pub max_steps: u32,
        #[uniffi(default = 1)]
        pub batch_size: u32,
        #[uniffi(default = 8)]
        pub gradient_accumulation_steps: u32,
        #[uniffi(default = 1024)]
        pub max_seq_len: u32,
        #[uniffi(default = None)]
        pub eval_steps: Option<u32>,
        #[uniffi(default = None)]
        pub save_steps: Option<u32>,
        #[uniffi(default = 42)]
        pub seed: u64,
        pub mixed_precision: MixedPrecisionEnum,
        #[uniffi(default = None)]
        pub device: Option<String>,
        pub optim: OptimConfigRecord,
        pub scheduler: SchedulerConfigRecord,
    }

    impl From<TrainCoreConfigRecord> for TrainCoreConfig {
        fn from(c: TrainCoreConfigRecord) -> Self {
            Self {
                base_model_repo: c.base_model_repo,
                base_model_revision: c.base_model_revision,
                output_dir: PathBuf::from(c.output_dir),
                max_steps: c.max_steps as usize,
                batch_size: c.batch_size as usize,
                gradient_accumulation_steps: c.gradient_accumulation_steps as usize,
                max_seq_len: c.max_seq_len as usize,
                eval_steps: c.eval_steps.map(|v| v as usize),
                save_steps: c.save_steps.map(|v| v as usize),
                seed: c.seed,
                mixed_precision: c.mixed_precision.into(),
                device: c.device,
                optim: c.optim.into(),
                scheduler: c.scheduler.into(),
            }
        }
    }

    impl From<TrainCoreConfig> for TrainCoreConfigRecord {
        fn from(c: TrainCoreConfig) -> Self {
            Self {
                base_model_repo: c.base_model_repo,
                base_model_revision: c.base_model_revision,
                output_dir: c.output_dir.display().to_string(),
                max_steps: u32::try_from(c.max_steps).unwrap_or(u32::MAX),
                batch_size: u32::try_from(c.batch_size).unwrap_or(u32::MAX),
                gradient_accumulation_steps: u32::try_from(c.gradient_accumulation_steps)
                    .unwrap_or(u32::MAX),
                max_seq_len: u32::try_from(c.max_seq_len).unwrap_or(u32::MAX),
                eval_steps: c.eval_steps.map(|v| u32::try_from(v).unwrap_or(u32::MAX)),
                save_steps: c.save_steps.map(|v| u32::try_from(v).unwrap_or(u32::MAX)),
                seed: c.seed,
                mixed_precision: c.mixed_precision.into(),
                device: c.device,
                optim: c.optim.into(),
                scheduler: c.scheduler.into(),
            }
        }
    }

    /// Direct Preference Optimization (DPO) configuration.
    ///
    /// Requires a frozen reference model. If `reference_model_repo` is
    /// `None`, the trainer reuses `core.base_model_repo`.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct DpoConfigRecord {
        pub core: TrainCoreConfigRecord,
        pub lora: LoraConfigRecord,
        #[uniffi(default = 0.1)]
        pub beta: f32,
        #[uniffi(default = 0.0)]
        pub label_smoothing: f32,
        #[uniffi(default = None)]
        pub reference_model_repo: Option<String>,
        #[uniffi(default = None)]
        pub reference_model_revision: Option<String>,
    }

    impl From<DpoConfigRecord> for DpoConfig {
        fn from(c: DpoConfigRecord) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                beta: c.beta,
                reference_model_repo: c.reference_model_repo,
                reference_model_revision: c.reference_model_revision,
                label_smoothing: c.label_smoothing,
            }
        }
    }

    /// Odds Ratio Preference Optimization (ORPO) configuration.
    ///
    /// Reference-free — combines an SFT loss on chosen responses with an
    /// odds-ratio loss term weighted by `lambda`.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct OrpoConfigRecord {
        pub core: TrainCoreConfigRecord,
        pub lora: LoraConfigRecord,
        #[uniffi(default = 0.1)]
        pub lambda: f32,
    }

    impl From<OrpoConfigRecord> for OrpoConfig {
        fn from(c: OrpoConfigRecord) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                lambda: c.lambda,
            }
        }
    }

    /// Simple Preference Optimization (`SimPO`) configuration.
    ///
    /// Reference-free, length-normalized. Defaults follow TRL `main`
    /// (`beta = 2.0`, `gamma = 1.0`).
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct SimpoConfigRecord {
        pub core: TrainCoreConfigRecord,
        pub lora: LoraConfigRecord,
        #[uniffi(default = 2.0)]
        pub beta: f32,
        #[uniffi(default = 1.0)]
        pub gamma: f32,
    }

    impl From<SimpoConfigRecord> for SimpoConfig {
        fn from(c: SimpoConfigRecord) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                beta: c.beta,
                gamma: c.gamma,
            }
        }
    }

    /// Kahneman-Tversky Optimization (KTO) configuration.
    ///
    /// Like DPO, KTO requires a frozen reference model (defaults to
    /// `core.base_model_repo`) — but the dataset schema differs:
    /// each row is a `(prompt, completion, desirable)` triple
    /// ([`UniffiRatedJsonlDataset`]), not a chosen/rejected pair.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct KtoConfigRecord {
        pub core: TrainCoreConfigRecord,
        pub lora: LoraConfigRecord,
        #[uniffi(default = 0.1)]
        pub beta: f32,
        #[uniffi(default = 1.0)]
        pub lambda_d: f32,
        #[uniffi(default = 1.0)]
        pub lambda_u: f32,
        #[uniffi(default = None)]
        pub reference_model_repo: Option<String>,
        #[uniffi(default = None)]
        pub reference_model_revision: Option<String>,
    }

    impl From<KtoConfigRecord> for KtoConfig {
        fn from(c: KtoConfigRecord) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                beta: c.beta,
                lambda_d: c.lambda_d,
                lambda_u: c.lambda_u,
                reference_model_repo: c.reference_model_repo,
                reference_model_revision: c.reference_model_revision,
            }
        }
    }

    /// Full fine-tune configuration (no LoRA — every parameter trains).
    ///
    /// `gradient_checkpointing = true` is accepted for forward compatibility
    /// but the trainer rejects it at init time with
    /// [`BlazenError::Validation`] — candle 0.10.2 has no activation-
    /// checkpointing primitive.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct FullFineTuneConfigRecord {
        pub core: TrainCoreConfigRecord,
        #[uniffi(default = false)]
        pub gradient_checkpointing: bool,
    }

    impl From<FullFineTuneConfigRecord> for FullFineTuneConfig {
        fn from(c: FullFineTuneConfigRecord) -> Self {
            Self {
                core: c.core.into(),
                gradient_checkpointing: c.gradient_checkpointing,
            }
        }
    }

    /// On-disk descriptor returned by [`UniffiModelManager::fine_tune`].
    ///
    /// Unlike [`TrainedAdapterRecord`], no PEFT adapter is written — the
    /// entire model's weights are saved to `output_dir` directly.
    #[derive(Debug, Clone, uniffi::Record)]
    pub struct FullFineTuneResultRecord {
        pub output_dir: String,
        pub final_loss: f32,
        pub steps_completed: u64,
    }

    impl From<FullFineTuneResult> for FullFineTuneResultRecord {
        fn from(r: FullFineTuneResult) -> Self {
            Self {
                output_dir: r.output_dir.display().to_string(),
                final_loss: r.final_loss,
                steps_completed: r.steps_completed as u64,
            }
        }
    }

    /// JSONL-backed preference-pair dataset opaque handle for DPO / ORPO /
    /// `SimPO`.
    ///
    /// Each line of the input file must deserialize to either
    /// `{"prompt": "...", "chosen": "...", "rejected": "..."}` or
    /// `{"messages": [...], "chosen": "...", "rejected": "..."}` (the
    /// latter requires `chat_template`).
    #[derive(uniffi::Object)]
    pub struct UniffiPreferenceJsonlDataset {
        inner: Arc<PreferenceJsonlDataset>,
    }

    #[uniffi::export]
    impl UniffiPreferenceJsonlDataset {
        /// Load a preference-pair JSONL file using the tokenizer at
        /// `tokenizer_path`. Args mirror [`UniffiJsonlDataset::from_path`].
        #[uniffi::constructor]
        pub fn from_path(
            path: String,
            tokenizer_path: String,
            chat_template: Option<String>,
            max_seq_len: u32,
            device: Option<String>,
            pad_token_id: u32,
        ) -> BlazenResult<Arc<Self>> {
            if max_seq_len == 0 {
                return Err(BlazenError::Validation {
                    message: "PreferenceJsonlDataset.max_seq_len must be > 0".into(),
                });
            }
            let tokenizer =
                Tokenizer::from_file(&tokenizer_path).map_err(|e| BlazenError::Validation {
                    message: format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
                })?;
            let device_str = device.as_deref().unwrap_or("cpu");
            let cdev = parse_train_device(device_str)?;
            let ds = PreferenceJsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                chat_template.as_deref(),
                max_seq_len as usize,
                cdev,
                pad_token_id,
            )
            .map_err(|e| BlazenError::Validation {
                message: format!("PreferenceJsonlDataset load failed: {e}"),
            })?;
            Ok(Arc::new(Self {
                inner: Arc::new(ds),
            }))
        }

        /// Number of preference examples in the dataset.
        pub fn len(&self) -> u64 {
            self.inner.len() as u64
        }

        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
    }

    /// JSONL-backed rated single-completion dataset opaque handle for KTO.
    ///
    /// Each line of the input file must deserialize to either
    /// `{"prompt": "...", "completion": "...", "label": true|false}` or
    /// `{"messages": [...], "completion": "...", "label": ...}` (the
    /// latter requires `chat_template`).
    #[derive(uniffi::Object)]
    pub struct UniffiRatedJsonlDataset {
        inner: Arc<RatedJsonlDataset>,
    }

    #[uniffi::export]
    impl UniffiRatedJsonlDataset {
        /// Load a rated JSONL file using the tokenizer at `tokenizer_path`.
        /// Args mirror [`UniffiJsonlDataset::from_path`].
        #[uniffi::constructor]
        pub fn from_path(
            path: String,
            tokenizer_path: String,
            chat_template: Option<String>,
            max_seq_len: u32,
            device: Option<String>,
            pad_token_id: u32,
        ) -> BlazenResult<Arc<Self>> {
            if max_seq_len == 0 {
                return Err(BlazenError::Validation {
                    message: "RatedJsonlDataset.max_seq_len must be > 0".into(),
                });
            }
            let tokenizer =
                Tokenizer::from_file(&tokenizer_path).map_err(|e| BlazenError::Validation {
                    message: format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
                })?;
            let device_str = device.as_deref().unwrap_or("cpu");
            let cdev = parse_train_device(device_str)?;
            let ds = RatedJsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                chat_template.as_deref(),
                max_seq_len as usize,
                cdev,
                pad_token_id,
            )
            .map_err(|e| BlazenError::Validation {
                message: format!("RatedJsonlDataset load failed: {e}"),
            })?;
            Ok(Arc::new(Self {
                inner: Arc::new(ds),
            }))
        }

        /// Number of rated examples in the dataset.
        pub fn len(&self) -> u64 {
            self.inner.len() as u64
        }

        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
    }

    #[uniffi::export(async_runtime = "tokio")]
    impl UniffiModelManager {
        /// Train a `LoRA` adapter via Direct Preference Optimization (DPO).
        ///
        /// Downloads the base model from `HuggingFace` (cached) plus the
        /// reference model (defaults to `config.core.base_model_repo`),
        /// runs the DPO training loop driven by `dataset`, and writes the
        /// resulting PEFT-format adapter to `config.core.output_dir`.
        ///
        /// If `progress` is provided, its `on_event` is called for each
        /// transition. Returning `Err(_)` from the callback cancels the run
        /// with [`BlazenError::Cancelled`].
        pub async fn train_dpo(
            self: Arc<Self>,
            config: DpoConfigRecord,
            dataset: Arc<UniffiPreferenceJsonlDataset>,
            progress: Option<Arc<dyn ForeignTrainingProgress>>,
        ) -> BlazenResult<TrainedAdapterRecord> {
            let rust_cfg: DpoConfig = config.into();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|p| {
                Arc::new(ForeignProgressAdapter { inner: p }) as Arc<dyn TrainingProgress>
            });
            let ds: Arc<dyn PreferenceDataset> = dataset.inner.clone();
            let adapter = self
                .inner
                .train_dpo(rust_cfg, ds, sink)
                .await
                .map_err(BlazenError::from)?;
            Ok(TrainedAdapterRecord::from(adapter))
        }

        /// Train a `LoRA` adapter via Odds Ratio Preference Optimization
        /// (ORPO). Reference-free — combines an SFT loss on chosen
        /// completions with an odds-ratio preference term.
        pub async fn train_orpo(
            self: Arc<Self>,
            config: OrpoConfigRecord,
            dataset: Arc<UniffiPreferenceJsonlDataset>,
            progress: Option<Arc<dyn ForeignTrainingProgress>>,
        ) -> BlazenResult<TrainedAdapterRecord> {
            let rust_cfg: OrpoConfig = config.into();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|p| {
                Arc::new(ForeignProgressAdapter { inner: p }) as Arc<dyn TrainingProgress>
            });
            let ds: Arc<dyn PreferenceDataset> = dataset.inner.clone();
            let adapter = self
                .inner
                .train_orpo(rust_cfg, ds, sink)
                .await
                .map_err(BlazenError::from)?;
            Ok(TrainedAdapterRecord::from(adapter))
        }

        /// Train a `LoRA` adapter via Simple Preference Optimization
        /// (`SimPO`). Reference-free and length-normalized.
        pub async fn train_simpo(
            self: Arc<Self>,
            config: SimpoConfigRecord,
            dataset: Arc<UniffiPreferenceJsonlDataset>,
            progress: Option<Arc<dyn ForeignTrainingProgress>>,
        ) -> BlazenResult<TrainedAdapterRecord> {
            let rust_cfg: SimpoConfig = config.into();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|p| {
                Arc::new(ForeignProgressAdapter { inner: p }) as Arc<dyn TrainingProgress>
            });
            let ds: Arc<dyn PreferenceDataset> = dataset.inner.clone();
            let adapter = self
                .inner
                .train_simpo(rust_cfg, ds, sink)
                .await
                .map_err(BlazenError::from)?;
            Ok(TrainedAdapterRecord::from(adapter))
        }

        /// Train a `LoRA` adapter via Kahneman-Tversky Optimization (KTO).
        ///
        /// Like DPO, KTO requires a frozen reference model; the dataset
        /// schema differs: each row is a `(prompt, completion, desirable)`
        /// triple.
        pub async fn train_kto(
            self: Arc<Self>,
            config: KtoConfigRecord,
            dataset: Arc<UniffiRatedJsonlDataset>,
            progress: Option<Arc<dyn ForeignTrainingProgress>>,
        ) -> BlazenResult<TrainedAdapterRecord> {
            let rust_cfg: KtoConfig = config.into();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|p| {
                Arc::new(ForeignProgressAdapter { inner: p }) as Arc<dyn TrainingProgress>
            });
            let ds: Arc<dyn RatedDataset> = dataset.inner.clone();
            let adapter = self
                .inner
                .train_kto(rust_cfg, ds, sink)
                .await
                .map_err(BlazenError::from)?;
            Ok(TrainedAdapterRecord::from(adapter))
        }

        /// Run a full fine-tune (every parameter trainable; no `LoRA`
        /// adapter).
        ///
        /// Returns [`FullFineTuneResultRecord`] rather than
        /// [`TrainedAdapterRecord`] because the output is a complete set
        /// of model weights in `config.core.output_dir`, not a mountable
        /// PEFT delta. Setting `config.gradient_checkpointing = true` is
        /// rejected up-front because candle 0.10.2 has no activation-
        /// checkpointing primitive.
        pub async fn fine_tune(
            self: Arc<Self>,
            config: FullFineTuneConfigRecord,
            dataset: Arc<UniffiJsonlDataset>,
            progress: Option<Arc<dyn ForeignTrainingProgress>>,
        ) -> BlazenResult<FullFineTuneResultRecord> {
            let rust_cfg: FullFineTuneConfig = config.into();
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|p| {
                Arc::new(ForeignProgressAdapter { inner: p }) as Arc<dyn TrainingProgress>
            });
            // `JsonlDataset` impls `TrainingDataset` directly, so the
            // `Arc<JsonlDataset>` unsized-coerces to `Arc<dyn TrainingDataset>`
            // without needing the `ArcDataset` wrapper used by `train_lora`
            // (which takes `Box<dyn TrainingDataset>`).
            let ds: Arc<dyn TrainingDataset> = dataset.inner.clone();
            let result = self
                .inner
                .fine_tune(rust_cfg, ds, sink)
                .await
                .map_err(BlazenError::from)?;
            Ok(FullFineTuneResultRecord::from(result))
        }
    }

    #[cfg(test)]
    mod training_tests {
        use super::*;

        fn sample_record() -> TrainConfigRecord {
            TrainConfigRecord {
                base_model_repo: "Qwen/Qwen2.5-0.5B".into(),
                output_dir: "/tmp/blazen-train-test".into(),
                lora: LoraConfigRecord {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.05,
                    target_modules: vec!["q_proj".into(), "v_proj".into()],
                },
                optim: OptimConfigRecord {
                    learning_rate: 2e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.0,
                    gradient_clip: Some(1.0),
                },
                scheduler: SchedulerConfigRecord {
                    kind: SchedulerKindEnum::Cosine,
                    warmup_steps: 50,
                },
                max_steps: 1000,
                batch_size: 4,
                gradient_accumulation_steps: 2,
                max_seq_len: 2048,
                eval_steps: Some(100),
                save_steps: Some(200),
                seed: 42,
                mixed_precision: MixedPrecisionEnum::Bf16,
                device: Some("cpu".into()),
            }
        }

        #[tokio::test]
        async fn train_config_record_round_trips() {
            let original = sample_record();
            let as_rust: TrainConfig = original.clone().into();
            let back: TrainConfigRecord = as_rust.into();

            assert_eq!(back.base_model_repo, original.base_model_repo);
            assert_eq!(back.output_dir, original.output_dir);
            assert_eq!(back.lora.rank, original.lora.rank);
            assert!((back.lora.alpha - original.lora.alpha).abs() < f32::EPSILON);
            assert_eq!(back.lora.target_modules, original.lora.target_modules);
            assert!((back.optim.learning_rate - original.optim.learning_rate).abs() < f64::EPSILON);
            assert_eq!(back.scheduler.warmup_steps, original.scheduler.warmup_steps);
            assert!(matches!(back.scheduler.kind, SchedulerKindEnum::Cosine));
            assert_eq!(back.max_steps, original.max_steps);
            assert_eq!(back.batch_size, original.batch_size);
            assert_eq!(
                back.gradient_accumulation_steps,
                original.gradient_accumulation_steps
            );
            assert_eq!(back.max_seq_len, original.max_seq_len);
            assert_eq!(back.eval_steps, original.eval_steps);
            assert_eq!(back.save_steps, original.save_steps);
            assert_eq!(back.seed, original.seed);
            assert!(matches!(back.mixed_precision, MixedPrecisionEnum::Bf16));
            assert_eq!(back.device, original.device);
        }

        #[test]
        fn training_event_enum_maps_all_variants() {
            use std::time::Duration;

            let started = TrainingEventEnum::from(TrainingEvent::Started { total_steps: 100 });
            assert!(matches!(
                started,
                TrainingEventEnum::Started { total_steps: 100 }
            ));

            let step = TrainingEventEnum::from(TrainingEvent::StepCompleted {
                step: 3,
                loss: 0.42,
                learning_rate: 1e-4,
                elapsed: Duration::from_millis(750),
            });
            match step {
                TrainingEventEnum::StepCompleted {
                    step,
                    loss,
                    learning_rate,
                    elapsed_ms,
                } => {
                    assert_eq!(step, 3);
                    assert!((loss - 0.42).abs() < f32::EPSILON);
                    assert!((learning_rate - 1e-4).abs() < f64::EPSILON);
                    assert_eq!(elapsed_ms, 750);
                }
                other => panic!("expected StepCompleted, got {other:?}"),
            }

            let finished = TrainingEventEnum::from(TrainingEvent::Finished {
                final_loss: 0.1,
                total_steps: 5,
                adapter_dir: PathBuf::from("/tmp/adapter"),
            });
            match finished {
                TrainingEventEnum::Finished {
                    final_loss,
                    total_steps,
                    adapter_dir,
                } => {
                    assert!((final_loss - 0.1).abs() < f32::EPSILON);
                    assert_eq!(total_steps, 5);
                    assert_eq!(adapter_dir, "/tmp/adapter");
                }
                other => panic!("expected Finished, got {other:?}"),
            }
        }
    }
}
