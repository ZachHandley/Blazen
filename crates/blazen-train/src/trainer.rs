//! Training state machine.
//!
//! Wave 3A of PR7 fills in the real `step` / `run` bodies. The trainer
//! owns the [`VarMap`] of trainable LoRA params + the optimizer over the
//! same subset, and ties together the per-architecture wrapper, dataset
//! iterator, scheduler, gradient clipper, checkpoint writer, and PEFT
//! adapter exporter.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::Deserialize;

use crate::arch::{llama, mistral, qwen2};
use crate::checkpoint;
use crate::config::TrainConfig;
use crate::error::BlazenTrainError;
use crate::export;
use crate::grad_clip;
use crate::lora::freeze_base_params;
use crate::progress::{TrainingEvent, TrainingProgress};
use crate::schedulers;

/// Marker trait for a frozen reference model used by preference-optimization
/// losses (DPO / ORPO / SimPO / KTO).
///
/// PR8 implements this against the per-arch wrappers and feeds the result
/// into [`Trainer::with_reference_model`]. PR7's SFT path leaves the
/// reference slot `None`.
pub trait ReferenceModel: Send + Sync {
    /// Compute logits for an `input_ids` batch under the frozen reference
    /// model. The result must be detached from any autograd graph.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error.
    fn forward_logits(&self, input_ids: &Tensor) -> candle_core::Result<Tensor>;
}

/// Trainable model dispatch. Built by [`Trainer::load_base_model`] based
/// on the HF repo's `config.json` `model_type` and stored inline so the
/// per-step forward is a single virtual call.
pub(crate) enum TrainableModel {
    Qwen2(qwen2::TrainableQwen2),
    Llama(llama::TrainableLlama),
    Mistral(mistral::TrainableMistral),
}

impl TrainableModel {
    fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen2(m) => m.forward(input_ids),
            Self::Llama(m) => m.forward(input_ids),
            Self::Mistral(m) => m.forward(input_ids),
        }
    }
}

/// Stateful training loop.
///
/// Construct via [`Trainer::new`], optionally attach a reference model
/// and progress sink, then drive with [`Trainer::step`] (per-batch) or
/// [`Trainer::run`] (full dataset). The trainer owns the [`VarMap`] for
/// the training run and the [`AdamW`] optimizer over its LoRA subset.
pub struct Trainer {
    config: TrainConfig,
    varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    reference_model: Option<Arc<dyn ReferenceModel>>,
    progress: Option<Arc<dyn TrainingProgress>>,
    global_step: usize,
    model: Option<TrainableModel>,
    lr_scheduler: Box<dyn Fn(usize) -> f64 + Send + Sync>,
    total_steps: usize,
    accum_counter: usize,
}

impl Trainer {
    /// Construct a fresh trainer from a config + varmap + device.
    ///
    /// The varmap is expected to already contain both the frozen base
    /// weights and the trainable LoRA params (the per-arch wrappers
    /// in `crate::arch::*` will own that population in Wave 2A/B/C).
    /// `Trainer::new` extracts the LoRA subset via
    /// [`crate::lora::freeze_base_params`] and hands it to AdamW.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::InvalidConfig`] if `config.lora.rank`
    /// is zero, or [`BlazenTrainError::Optimizer`] if `AdamW::new` rejects
    /// the param set.
    pub fn new(
        config: TrainConfig,
        varmap: VarMap,
        device: Device,
    ) -> Result<Self, BlazenTrainError> {
        if config.lora.rank == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "lora.rank must be > 0".to_string(),
            ));
        }
        if config.max_steps == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "max_steps must be > 0".to_string(),
            ));
        }

        let target_refs: Vec<&str> = config
            .lora
            .target_modules
            .iter()
            .map(String::as_str)
            .collect();
        let trainable = freeze_base_params(&varmap, &target_refs);

        let params = ParamsAdamW {
            lr: config.optim.learning_rate,
            beta1: config.optim.beta1,
            beta2: config.optim.beta2,
            eps: config.optim.epsilon,
            weight_decay: config.optim.weight_decay,
        };
        let optimizer = AdamW::new(trainable, params)
            .map_err(|e| BlazenTrainError::Optimizer(e.to_string()))?;

        let lr_scheduler = schedulers::make_scheduler(
            &config.scheduler,
            config.optim.learning_rate,
            config.max_steps,
        );
        let total_steps = config.max_steps;

        Ok(Self {
            config,
            varmap,
            optimizer,
            device,
            reference_model: None,
            progress: None,
            global_step: 0,
            model: None,
            lr_scheduler,
            total_steps,
            accum_counter: 0,
        })
    }

    /// Attach a frozen reference model for PR8-era DPO-family losses.
    #[must_use]
    pub fn with_reference_model(mut self, reference: Arc<dyn ReferenceModel>) -> Self {
        self.reference_model = Some(reference);
        self
    }

    /// Attach a per-step progress sink.
    #[must_use]
    pub fn with_progress(mut self, progress: Arc<dyn TrainingProgress>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Borrow the active config.
    #[must_use]
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Borrow the [`VarMap`] backing the training run (for checkpoints).
    #[must_use]
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Borrow the device the training graph lives on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Current 0-indexed optimizer step counter.
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Inject a pre-built [`TrainableModel`] for unit tests that want to
    /// avoid the HF Hub download path entirely. Production code drives
    /// [`Self::load_base_model`] instead.
    #[cfg(test)]
    pub(crate) fn set_model_for_testing(&mut self, model: TrainableModel) {
        self.model = Some(model);
    }

    /// Download the base model from HF Hub, parse `config.json` to detect
    /// the architecture, mmap the safetensors into a frozen `VarBuilder`,
    /// and wrap the result in a `TrainableXxx` whose LoRA params land in
    /// `self.varmap` under PEFT-canonical paths.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::ModelLoad`] for HF Hub / parse / mmap
    /// failures, or [`BlazenTrainError::Candle`] for tensor errors during
    /// the per-arch wrapper's `load`.
    pub async fn load_base_model(&mut self) -> Result<(), BlazenTrainError> {
        let dtype = match self.config.mixed_precision {
            crate::config::MixedPrecision::None => DType::F32,
            crate::config::MixedPrecision::Bf16 => DType::BF16,
        };

        let repo_id = self.config.base_model_repo.clone();
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(false)
            .build()
            .map_err(|e| BlazenTrainError::ModelLoad(format!("HF API init failed: {e}")))?;
        let repo = api.model(repo_id.clone());

        let config_path = repo.get("config.json").await.map_err(|e| {
            BlazenTrainError::ModelLoad(format!("config.json download failed: {e}"))
        })?;
        // Why: tokenizer is downloaded so the user has it cached locally
        // alongside the adapter, even though the trainer itself does not
        // tokenize (datasets carry their own tokenizer).
        let _tokenizer_path = repo.get("tokenizer.json").await.map_err(|e| {
            BlazenTrainError::ModelLoad(format!("tokenizer.json download failed: {e}"))
        })?;

        let cfg_bytes = std::fs::read(&config_path).map_err(|e| {
            BlazenTrainError::ModelLoad(format!(
                "read config.json at {}: {e}",
                config_path.display()
            ))
        })?;

        let weight_paths = download_safetensors_shards(&repo).await?;

        let probe: ArchProbe = serde_json::from_slice(&cfg_bytes).map_err(|e| {
            BlazenTrainError::ModelLoad(format!("config.json arch probe failed: {e}"))
        })?;
        let arch_str = probe.model_type.as_deref().unwrap_or("");

        // Why: SAFETY — HF cache files are immutable once written; the
        // mmap stays valid for the lifetime of the model.
        #[allow(unsafe_code)]
        let base_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &self.device)
                .map_err(|e| BlazenTrainError::ModelLoad(format!("safetensors mmap failed: {e}")))?
        };
        let lora_vb = VarBuilder::from_varmap(&self.varmap, dtype, &self.device);

        let model = match arch_str {
            "qwen2" => {
                let shim: Qwen2ConfigShim = serde_json::from_slice(&cfg_bytes).map_err(|e| {
                    BlazenTrainError::ModelLoad(format!("qwen2 Config parse failed: {e}"))
                })?;
                let arch_cfg = shim.into_config();
                let m =
                    qwen2::TrainableQwen2::load(base_vb, lora_vb, &arch_cfg, &self.config.lora)?;
                TrainableModel::Qwen2(m)
            }
            "llama" => {
                let llama_cfg: llama::LlamaConfig =
                    serde_json::from_slice(&cfg_bytes).map_err(|e| {
                        BlazenTrainError::ModelLoad(format!("llama Config parse failed: {e}"))
                    })?;
                let arch_cfg = llama_cfg.into_config(false);
                let m =
                    llama::TrainableLlama::load(base_vb, lora_vb, &arch_cfg, &self.config.lora)?;
                TrainableModel::Llama(m)
            }
            "mistral" => {
                let arch_cfg: mistral::Config =
                    serde_json::from_slice(&cfg_bytes).map_err(|e| {
                        BlazenTrainError::ModelLoad(format!("mistral Config parse failed: {e}"))
                    })?;
                let m = mistral::TrainableMistral::load(
                    base_vb,
                    lora_vb,
                    &arch_cfg,
                    &self.config.lora,
                )?;
                TrainableModel::Mistral(m)
            }
            other => {
                return Err(BlazenTrainError::ModelLoad(format!(
                    "unsupported model_type '{other}' (wired: qwen2, llama, mistral)"
                )));
            }
        };

        self.model = Some(model);

        // Why: after the model lands, the LoRA vars are present in the
        // VarMap, so rebuild the optimizer over the freshly-populated
        // trainable subset. AdamW::new requires the vars at construction.
        let target_refs: Vec<&str> = self
            .config
            .lora
            .target_modules
            .iter()
            .map(String::as_str)
            .collect();
        let trainable = freeze_base_params(&self.varmap, &target_refs);
        let params = ParamsAdamW {
            lr: self.config.optim.learning_rate,
            beta1: self.config.optim.beta1,
            beta2: self.config.optim.beta2,
            eps: self.config.optim.epsilon,
            weight_decay: self.config.optim.weight_decay,
        };
        self.optimizer = AdamW::new(trainable, params)
            .map_err(|e| BlazenTrainError::Optimizer(e.to_string()))?;

        Ok(())
    }

    /// Run one training step on a batch.
    ///
    /// Forward + cross-entropy loss (with `-100` masking) + backward +
    /// optional gradient clipping + optimizer step. Loss accumulation
    /// across `gradient_accumulation_steps` micro-batches is handled by
    /// scaling the loss before backward and only stepping the optimizer
    /// every Nth call.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::InvalidConfig`] if the model has not
    /// been loaded yet; otherwise forwards candle errors.
    // Why: the public signature is async so callers can await device
    // synchronization (CUDA stream sync, MPS commit) and per-step dataset
    // shuffling without breaking the API; the current CPU-only body has
    // no await but the contract stays async.
    #[allow(clippy::unused_async)]
    pub async fn step(&mut self, batch: TrainingBatch) -> Result<f32, BlazenTrainError> {
        let model = self.model.as_ref().ok_or_else(|| {
            BlazenTrainError::InvalidConfig(
                "Trainer::step requires a loaded model — call load_base_model first".to_string(),
            )
        })?;

        let lr = (self.lr_scheduler)(self.global_step);
        self.optimizer.set_learning_rate(lr);

        let logits = model.forward(&batch.input_ids)?;
        let (b, t, v) = logits.dims3()?;
        let logits_2d = logits.reshape((b * t, v))?;
        let labels_1d = batch.labels.reshape((b * t,))?.to_dtype(DType::I64)?;

        let loss = masked_cross_entropy(&logits_2d, &labels_1d, -100)?;

        let accum = self.config.gradient_accumulation_steps.max(1);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / accum as f64;
        let scaled_loss = (&loss * scale)?;

        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        let mut grads = scaled_loss.backward()?;

        self.accum_counter += 1;
        if self.accum_counter >= accum {
            if let Some(max) = self.config.optim.gradient_clip {
                let target_refs: Vec<&str> = self
                    .config
                    .lora
                    .target_modules
                    .iter()
                    .map(String::as_str)
                    .collect();
                let vars = freeze_base_params(&self.varmap, &target_refs);
                let var_refs: Vec<&candle_core::Var> = vars.iter().collect();
                grad_clip::clip_grad_norm(&mut grads, &var_refs, max)?;
            }

            self.optimizer
                .step(&grads)
                .map_err(|e| BlazenTrainError::Optimizer(e.to_string()))?;
            self.accum_counter = 0;
        }

        self.global_step += 1;
        Ok(loss_value)
    }

    /// Full training run from a dataset.
    ///
    /// Loads the base model if not already loaded, then iterates
    /// `dataset.batch(...)` → `step(...)` until `global_step` reaches
    /// `config.max_steps`. Emits progress events, optional periodic
    /// checkpoints, and a final PEFT adapter export.
    ///
    /// # Errors
    ///
    /// Forwards any [`BlazenTrainError`] from [`Self::step`],
    /// [`crate::checkpoint::save_checkpoint`], or
    /// [`crate::export::save_peft_adapter`]. Returns
    /// [`BlazenTrainError::Cancelled`] if a progress callback aborts.
    pub async fn run(
        &mut self,
        dataset: Box<dyn TrainingDataset>,
    ) -> Result<TrainedAdapter, BlazenTrainError> {
        if self.model.is_none() {
            self.load_base_model().await?;
        }

        self.emit_event(TrainingEvent::Started {
            total_steps: self.total_steps,
        })?;

        let mut final_loss = 0.0_f32;
        while self.global_step < self.total_steps {
            let step_idx = self.global_step;
            let batch = dataset.batch(self.config.batch_size, step_idx).await?;
            let started = Instant::now();
            let loss = self.step(batch).await?;
            let elapsed = started.elapsed();
            final_loss = loss;

            let lr = (self.lr_scheduler)(step_idx);
            self.emit_event(TrainingEvent::StepCompleted {
                step: step_idx,
                loss,
                learning_rate: lr,
                elapsed,
            })?;

            if let Some(save_every) = self.config.save_steps
                && save_every > 0
                && self.global_step.is_multiple_of(save_every)
            {
                checkpoint::save_checkpoint(
                    &self.config.output_dir,
                    self.global_step,
                    &self.varmap,
                    &self.config,
                )?;
                let cp_path = self
                    .config
                    .output_dir
                    .join(format!("step_{}", self.global_step));
                self.emit_event(TrainingEvent::CheckpointSaved {
                    step: self.global_step,
                    path: cp_path,
                })?;
            }
        }

        export::save_peft_adapter(
            &self.varmap,
            &self.config.output_dir,
            &self.config.lora,
            &self.config.base_model_repo,
        )?;

        let adapter = TrainedAdapter {
            adapter_dir: self.config.output_dir.clone(),
            final_loss,
            total_steps: self.global_step,
        };

        self.emit_event(TrainingEvent::Finished {
            final_loss,
            total_steps: self.global_step,
            adapter_dir: self.config.output_dir.clone(),
        })?;

        Ok(adapter)
    }

    fn emit_event(&self, event: TrainingEvent) -> Result<(), BlazenTrainError> {
        if let Some(sink) = self.progress.as_ref() {
            sink.on_event(event)
                .map_err(|_| BlazenTrainError::Cancelled)?;
        }
        Ok(())
    }
}

/// Cross-entropy with an `ignore_index` mask.
///
/// `logits_2d` is `(N, V)` raw logits and `labels_1d` is `(N,)` i64
/// labels. Positions where `labels == ignore_index` contribute zero to
/// both the numerator and the denominator. The result is the mean over
/// non-ignored positions; if every position is ignored the loss is
/// reported as zero (which back-propagates a zero gradient — the
/// optimizer step on this batch becomes a no-op).
fn masked_cross_entropy(
    logits_2d: &Tensor,
    labels_1d: &Tensor,
    ignore_index: i64,
) -> candle_core::Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(&logits_2d.to_dtype(DType::F32)?, 1)?;

    // Why: gather rejects negative indices, so swap `ignore_index` with
    // 0 before the gather and zero out the contribution via the mask.
    let labels_i64 = labels_1d.to_dtype(DType::I64)?;
    let ignore_tensor = Tensor::new(ignore_index, labels_i64.device())?;
    let keep_mask_bool = labels_i64.ne(&ignore_tensor.broadcast_as(labels_i64.shape())?)?;
    let zero_tensor = Tensor::zeros(labels_i64.shape(), DType::I64, labels_i64.device())?;
    let safe_labels = keep_mask_bool.where_cond(&labels_i64, &zero_tensor)?;

    // Why: gather wants the index in u32; clamp + cast post-mask.
    let safe_labels_u32 = safe_labels.to_dtype(DType::U32)?;
    let gathered = log_probs
        .gather(&safe_labels_u32.unsqueeze(1)?, 1)?
        .squeeze(1)?;
    let nll_per_token = gathered.neg()?;

    let keep_f32 = keep_mask_bool.to_dtype(DType::F32)?;
    let masked = (nll_per_token * &keep_f32)?;

    let num_kept = keep_f32.sum_all()?.to_scalar::<f32>()?;
    if num_kept <= 0.0 {
        return Tensor::zeros((), DType::F32, logits_2d.device());
    }
    let sum = masked.sum_all()?;
    sum.affine(f64::from(1.0_f32 / num_kept), 0.0)
}

/// A single training batch (input ids + attention mask + label ids).
pub struct TrainingBatch {
    /// Input token ids `[batch, seq]`, dtype `i64`/`u32` per arch convention.
    pub input_ids: Tensor,
    /// Attention mask `[batch, seq]`, 1 for real tokens / 0 for padding.
    pub attention_mask: Tensor,
    /// Label ids `[batch, seq]`. `-100` masks the position from the loss.
    pub labels: Tensor,
}

/// Async source of training batches.
///
/// Wave 2 supplies a JSONL/parquet implementation in `crate::dataset`.
#[async_trait]
pub trait TrainingDataset: Send + Sync {
    /// Total number of examples in the dataset.
    fn len(&self) -> usize;

    /// Returns whether the dataset has zero examples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build the `idx`-th batch of size `batch_size`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for indexing / tokenization /
    /// I/O failures.
    async fn batch(&self, batch_size: usize, idx: usize)
    -> Result<TrainingBatch, BlazenTrainError>;
}

/// Final artifact produced by [`Trainer::run`].
#[derive(Debug, Clone)]
pub struct TrainedAdapter {
    /// Directory the PEFT-format adapter was written to.
    pub adapter_dir: PathBuf,
    /// Final training loss.
    pub final_loss: f32,
    /// Total optimizer steps executed.
    pub total_steps: usize,
}

/// Subset of `config.json` needed for architecture dispatch.
#[derive(Debug, Deserialize)]
struct ArchProbe {
    #[serde(default)]
    model_type: Option<String>,
}

/// Tolerant Qwen2 config wrapper: real Qwen2/2.5 configs may omit
/// optional fields that upstream `qwen2::Config` requires.
#[derive(Debug, Deserialize)]
struct Qwen2ConfigShim {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    max_window_layers: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    rope_theta: f64,
    rms_norm_eps: f64,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(default)]
    hidden_act: Option<candle_nn::Activation>,
}

impl Qwen2ConfigShim {
    fn into_config(self) -> qwen2::Config {
        let sliding_window = self.sliding_window.unwrap_or(self.max_position_embeddings);
        qwen2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window,
            max_window_layers: self.max_window_layers.unwrap_or(self.num_hidden_layers),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            hidden_act: self.hidden_act.unwrap_or(candle_nn::Activation::Silu),
        }
    }
}

/// Enumerate every `model-*.safetensors` shard in the HF repo (or the
/// single `model.safetensors`) and download each to the local cache.
async fn download_safetensors_shards(
    repo: &hf_hub::api::tokio::ApiRepo,
) -> Result<Vec<PathBuf>, BlazenTrainError> {
    let info = repo
        .info()
        .await
        .map_err(|e| BlazenTrainError::ModelLoad(format!("HF repo info failed: {e}")))?;

    let mut shard_names: Vec<String> = Vec::new();
    for sib in info.siblings {
        let name = sib.rfilename;
        if name == "adapter_model.safetensors" || name.ends_with("/adapter_model.safetensors") {
            continue;
        }
        if name == "model.safetensors" || is_sharded_safetensors(&name) {
            shard_names.push(name);
        }
    }
    shard_names.sort();

    if shard_names.is_empty() {
        return Err(BlazenTrainError::ModelLoad(
            "no safetensors shards found in base model repo".to_string(),
        ));
    }

    let mut paths: Vec<PathBuf> = Vec::with_capacity(shard_names.len());
    for name in &shard_names {
        let path = repo.get(name).await.map_err(|e| {
            BlazenTrainError::ModelLoad(format!("safetensors download failed ({name}): {e}"))
        })?;
        paths.push(path);
    }
    Ok(paths)
}

fn is_sharded_safetensors(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((shard, total)) = rest.split_once("-of-") else {
        return false;
    };
    !shard.is_empty()
        && !total.is_empty()
        && shard.bytes().all(|b| b.is_ascii_digit())
        && total.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Activation, Init, VarBuilder, VarMap};

    use crate::config::{
        LoraConfig, MixedPrecision, OptimConfig, SchedulerConfig, SchedulerKind, TrainConfig,
    };
    use crate::lora::LoraLinear;

    fn make_base_linear(in_dim: usize, out_dim: usize, device: &Device) -> candle_nn::Linear {
        let w = Tensor::ones((out_dim, in_dim), DType::F32, device).expect("base weight ones");
        candle_nn::Linear::new(w, None)
    }

    fn tiny_qwen2_config() -> qwen2::Config {
        qwen2::Config {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: 64,
            max_window_layers: 2,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        }
    }

    fn tiny_train_config(output_dir: PathBuf, max_steps: usize) -> TrainConfig {
        TrainConfig {
            base_model_repo: "test/local".to_string(),
            output_dir,
            lora: LoraConfig {
                rank: 4,
                alpha: 8.0,
                dropout: 0.0,
                target_modules: vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                ],
            },
            optim: OptimConfig {
                learning_rate: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                gradient_clip: Some(1.0),
            },
            scheduler: SchedulerConfig {
                kind: SchedulerKind::Constant,
                warmup_steps: 0,
            },
            max_steps,
            batch_size: 2,
            gradient_accumulation_steps: 1,
            max_seq_len: 8,
            eval_steps: None,
            save_steps: None,
            seed: 42,
            mixed_precision: MixedPrecision::None,
            device: None,
        }
    }

    fn build_tiny_qwen2(varmap: &VarMap, device: &Device, lora_cfg: &LoraConfig) -> TrainableModel {
        let cfg = tiny_qwen2_config();
        let base_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let lora_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let m = qwen2::TrainableQwen2::load(base_vb, lora_vb, &cfg, lora_cfg).expect("load");
        TrainableModel::Qwen2(m)
    }

    struct FixedBatchDataset {
        batch: TrainingBatch,
    }

    impl FixedBatchDataset {
        fn new(device: &Device, batch_size: usize, seq_len: usize, vocab: usize) -> Self {
            let ids: Vec<u32> = (0..batch_size * seq_len)
                .map(|i| u32::try_from(i % vocab).unwrap())
                .collect();
            let mask: Vec<u32> = vec![1; batch_size * seq_len];
            let labels: Vec<i64> = ids.iter().map(|&t| i64::from(t)).collect();
            let shape = (batch_size, seq_len);
            let input_ids = Tensor::from_vec(ids, shape, device).expect("ids");
            let attention_mask = Tensor::from_vec(mask, shape, device).expect("mask");
            let labels = Tensor::from_vec(labels, shape, device).expect("labels");
            Self {
                batch: TrainingBatch {
                    input_ids,
                    attention_mask,
                    labels,
                },
            }
        }

        fn clone_batch(&self) -> TrainingBatch {
            TrainingBatch {
                input_ids: self.batch.input_ids.clone(),
                attention_mask: self.batch.attention_mask.clone(),
                labels: self.batch.labels.clone(),
            }
        }
    }

    #[async_trait]
    impl TrainingDataset for FixedBatchDataset {
        fn len(&self) -> usize {
            1
        }
        async fn batch(
            &self,
            _batch_size: usize,
            _idx: usize,
        ) -> Result<TrainingBatch, BlazenTrainError> {
            Ok(self.clone_batch())
        }
    }

    #[test]
    fn trainer_new_rejects_zero_rank() {
        let cfg = TrainConfig {
            lora: LoraConfig {
                rank: 0,
                ..LoraConfig::default()
            },
            ..TrainConfig::default()
        };
        let vm = VarMap::new();
        match Trainer::new(cfg, vm, Device::Cpu) {
            Err(BlazenTrainError::InvalidConfig(msg)) => {
                assert!(msg.contains("rank"), "msg was: {msg}");
            }
            Err(other) => panic!("expected InvalidConfig, got {other}"),
            Ok(_) => panic!("zero rank must error"),
        }
    }

    #[test]
    fn trainer_new_rejects_zero_max_steps() {
        let cfg = TrainConfig {
            max_steps: 0,
            ..TrainConfig::default()
        };
        let vm = VarMap::new();
        match Trainer::new(cfg, vm, Device::Cpu) {
            Err(BlazenTrainError::InvalidConfig(msg)) => {
                assert!(msg.contains("max_steps"), "msg was: {msg}");
            }
            Err(other) => panic!("expected InvalidConfig, got {other}"),
            Ok(_) => panic!("zero max_steps must error"),
        }
    }

    #[test]
    fn trainer_new_picks_up_only_lora_vars_for_optimizer() {
        let device = Device::Cpu;
        let varmap = VarMap::new();

        let vb_lora = VarBuilder::from_varmap(&varmap, DType::F32, &device)
            .push_prefix("model.layers.0.self_attn.q_proj");
        let _ =
            LoraLinear::wrap(make_base_linear(4, 4, &device), 4, 4, 2, 4.0, vb_lora).expect("wrap");
        let _ = varmap
            .get(
                (4, 4),
                "model.layers.0.self_attn.q_proj.weight",
                Init::Const(0.0),
                DType::F32,
                &device,
            )
            .expect("register frozen base var");

        let trainer = Trainer::new(TrainConfig::default(), varmap, device).expect("ctor");
        assert_eq!(trainer.global_step(), 0);
        assert!(trainer.config().lora.rank > 0);
    }

    #[test]
    fn masked_cross_entropy_ignores_minus_100_labels() {
        let device = Device::Cpu;
        // 4 positions, vocab=3; positions 1 and 3 are -100 (ignored).
        let logits = Tensor::from_vec(
            vec![
                1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            ],
            (4, 3),
            &device,
        )
        .unwrap();
        let labels = Tensor::from_vec(vec![0_i64, -100, 2, -100], (4,), &device).unwrap();

        let masked = masked_cross_entropy(&logits, &labels, -100)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let logits_kept =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0], (2, 3), &device).unwrap();
        let labels_kept = Tensor::from_vec(vec![0_u32, 2], (2,), &device).unwrap();
        let reference = candle_nn::loss::cross_entropy(&logits_kept, &labels_kept)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            (masked - reference).abs() < 1e-5,
            "masked={masked} reference={reference}"
        );
    }

    #[test]
    fn masked_cross_entropy_all_ignored_is_zero() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.5_f32, 0.5, 0.5, 0.5], (2, 2), &device).unwrap();
        let labels = Tensor::from_vec(vec![-100_i64, -100], (2,), &device).unwrap();
        let masked = masked_cross_entropy(&logits, &labels, -100)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (masked - 0.0).abs() < 1e-7,
            "all-ignored loss not zero: {masked}"
        );
    }

    #[tokio::test]
    async fn trainer_step_decreases_loss_on_fixed_batch() {
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_train_config(tmp.path().to_path_buf(), 30);
        let varmap = VarMap::new();

        // Why: build the model first so the LoRA vars are registered, then
        // construct the trainer over the (now-populated) varmap.
        let model = build_tiny_qwen2(&varmap, &device, &cfg.lora);

        let mut trainer = Trainer::new(cfg, varmap, device.clone()).expect("ctor");
        trainer.set_model_for_testing(model);

        let ds = FixedBatchDataset::new(&device, 2, 8, 128);

        let initial = trainer.step(ds.clone_batch()).await.unwrap();
        let mut last = initial;
        for _ in 1..25 {
            last = trainer.step(ds.clone_batch()).await.unwrap();
        }

        assert!(
            last < initial * 0.7,
            "loss did not decrease enough: initial={initial} final={last}"
        );
    }

    #[tokio::test]
    async fn trainer_run_writes_peft_adapter() {
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_train_config(tmp.path().to_path_buf(), 5);
        let varmap = VarMap::new();
        let model = build_tiny_qwen2(&varmap, &device, &cfg.lora);

        let mut trainer = Trainer::new(cfg, varmap, device.clone()).expect("ctor");
        trainer.set_model_for_testing(model);

        let ds: Box<dyn TrainingDataset> = Box::new(FixedBatchDataset::new(&device, 2, 8, 128));
        let result = trainer.run(ds).await.expect("run");

        assert_eq!(result.total_steps, 5);
        assert!(tmp.path().join("adapter_config.json").exists());
        assert!(tmp.path().join("adapter_model.safetensors").exists());
    }
}
