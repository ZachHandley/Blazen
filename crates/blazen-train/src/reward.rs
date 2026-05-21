//! Reward model training (Bradley-Terry pairwise loss on preference data).
//!
//! A "reward model" is a base LLM with a scalar regression head bolted on
//! top of the final non-pad token's post-norm hidden state. The training
//! signal is a preference pair `(prompt, chosen, rejected)`: the model is
//! pushed to assign higher reward to the chosen completion than to the
//! rejected one via the standard pairwise loss
//!
//! ```text
//! loss = -mean_i log( sigmoid( r_chosen_i - r_rejected_i ) )
//! ```
//!
//! Once trained, the reward model is the scorer that drives RLHF-family
//! algorithms — most directly [`crate::grpo::GrpoTrainer`], whose
//! group-relative advantages are computed over reward-model scores rather
//! than over a learned critic.
//!
//! Phase 1 scope (this module):
//!  - [`RewardModel`] = `TrainableLlama` + `Linear(hidden, 1)` head.
//!  - [`RewardTrainer`] consumes [`crate::trainer::PreferenceBatch`] rows
//!    (the same JSONL format DPO/ORPO/SimPO already use) and minimizes the
//!    Bradley-Terry loss over LoRA-on-base or full-fine-tune-on-base.
//!  - Llama-family base models only — Qwen2/Mistral reward heads require
//!    the equivalent `forward_hidden_states` entry point on those arches.
//!    Deferred to phase 2 alongside HF-Hub reward-model loading and
//!    binding parity.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::arch::llama::{Config as LlamaConfig, TrainableLlama};
use crate::checkpoint;
use crate::config::{LoraConfig, MixedPrecision, RewardConfig, TrainConfig};
use crate::error::BlazenTrainError;
use crate::grad_clip;
use crate::lora::freeze_base_params;
use crate::progress::{TrainingEvent, TrainingProgress};
use crate::schedulers;
use crate::trainer::{PreferenceBatch, PreferenceDataset, TrainedAdapter};

/// VarMap key under which the reward head's weight is registered.
///
/// Public so callers exporting / merging the adapter can find the row
/// without re-running the trainer (e.g. the GRPO trainer loading a
/// pretrained reward model). Matches the candle convention of a single
/// `weight` row per [`Linear`] with no bias.
pub const REWARD_HEAD_WEIGHT_KEY: &str = "reward_head.weight";

/// Llama-backed reward model: hidden-state encoder + scalar head.
///
/// The encoder is a [`TrainableLlama`] (so LoRA / FFT plumbing inherits
/// for free), and the scalar head is a single `Linear(hidden, 1)` with no
/// bias whose weights live in the same `VarMap` as the encoder. The head
/// reads the last *non-pad* token's hidden state per row, projects it to
/// a scalar, and returns a `[B]` `f32` reward.
///
/// Last-non-pad selection uses the attention mask sum minus one — matches
/// HuggingFace's `LlamaForSequenceClassification` convention and keeps
/// gradients flowing through the encoder at every token (the gather is
/// pure routing).
pub struct RewardModel {
    encoder: TrainableLlama,
    head: Linear,
}

impl RewardModel {
    /// Wire up a reward model on top of a freshly-loaded [`TrainableLlama`].
    ///
    /// The reward-head weight is registered in `head_vb` (typically the
    /// same `VarBuilder::from_varmap` the encoder's LoRA params landed in)
    /// under the leaf [`REWARD_HEAD_WEIGHT_KEY`]. The init uses candle's
    /// default `linear_no_bias` initializer (Kaiming-uniform), which keeps
    /// initial rewards tightly clustered around zero so the pairwise loss
    /// starts near `ln(2)` (sigmoid of zero) for any prompt.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error from the `linear_no_bias` call.
    // Why: candle's `linear_no_bias(in, out, VarBuilder)` consumes the
    // VarBuilder by value, so it's idiomatic to take ours by value too —
    // callers can `.pp(...)` clone on the way in if they need to keep
    // the original alive.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        encoder: TrainableLlama,
        cfg: &LlamaConfig,
        head_vb: VarBuilder,
    ) -> Result<Self, BlazenTrainError> {
        let head = candle_nn::linear_no_bias(cfg.hidden_size, 1, head_vb.pp("reward_head"))?;
        Ok(Self { encoder, head })
    }

    /// Borrow the underlying encoder (for tests / introspection).
    #[must_use]
    pub fn encoder(&self) -> &TrainableLlama {
        &self.encoder
    }

    /// Borrow the scalar reward head.
    #[must_use]
    pub fn head(&self) -> &Linear {
        &self.head
    }

    /// Forward pass: returns one scalar reward per row, `[B]` f32.
    ///
    /// `input_ids` is `[B, T]` u32. `attention_mask` is `[B, T]` u32/i64
    /// with `1` at real tokens and `0` at right-side padding. The reward
    /// is read from the position `sum(mask) - 1` per row, mirroring HF's
    /// `LlamaForSequenceClassification` behavior (pad-on-right datasets
    /// only — the trainer's batchers always right-pad).
    ///
    /// All intermediate ops are autograd-live, so `loss.backward()` reaches
    /// both the encoder's params and the reward head.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Candle`] for any tensor shape/dtype
    /// mismatch. Returns [`BlazenTrainError::Forward`] if a row's attention
    /// mask sums to zero (no real tokens — caller bug).
    pub fn forward_rewards(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, BlazenTrainError> {
        // [B, T, H], dtype = encoder's dtype (typically f32 for tests, bf16 for prod).
        let hidden = self.encoder.forward_hidden_states(input_ids)?;
        let (b, t, _h) = hidden.dims3()?;
        let (mb, mt) = attention_mask.dims2()?;
        if mb != b || mt != t {
            return Err(BlazenTrainError::Forward(format!(
                "reward forward: attention_mask shape [{mb}, {mt}] != hidden [{b}, {t}]"
            )));
        }

        // Per-row last-real-token index = sum(mask) - 1. We cast to f32 first
        // so the sum is a smooth tensor op; the index gather only needs the
        // scalar values, not the autograd path, but we still keep the result
        // alive in case future callers want it differentiable.
        let mask_f32 = attention_mask.to_dtype(DType::F32)?;
        let lengths = mask_f32.sum(D::Minus1)?; // [B] f32

        // Guard: every row must have at least one real token. We collect to
        // a host Vec so the index is concrete for the per-row slice below;
        // the gather operator would also work but its API is harder to use
        // for this "one position per row" shape.
        let lengths_vec: Vec<f32> = lengths.to_vec1::<f32>()?;
        let mut last_idx: Vec<usize> = Vec::with_capacity(b);
        for (row, &len) in lengths_vec.iter().enumerate() {
            if len < 0.5 {
                return Err(BlazenTrainError::Forward(format!(
                    "reward forward: row {row} has zero real tokens in its attention mask"
                )));
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let idx = (len as usize).saturating_sub(1);
            last_idx.push(idx.min(t.saturating_sub(1)));
        }

        // Per-row slice: build [B, 1, H] by stacking `hidden[row, idx, :]`.
        let mut per_row: Vec<Tensor> = Vec::with_capacity(b);
        for (row, &idx) in last_idx.iter().enumerate() {
            // hidden[row, idx, :] -> [H]; unsqueeze to [1, H] for stacking.
            let slice = hidden.i((row, idx, ..))?.unsqueeze(0)?;
            per_row.push(slice);
        }
        let last_hidden = Tensor::cat(&per_row, 0)?; // [B, H]
        let scores = self.head.forward(&last_hidden)?; // [B, 1]
        let scores_f32 = scores.to_dtype(DType::F32)?;
        Ok(scores_f32.squeeze(D::Minus1)?) // [B]
    }
}

/// Bradley-Terry pairwise reward trainer.
///
/// Owns the [`RewardModel`], the [`VarMap`] backing its trainable params
/// (encoder LoRA params + reward head), and an [`AdamW`] optimizer over
/// the same subset. Drives [`Self::step`] (per-batch) or [`Self::run`]
/// (full dataset). Mirrors [`crate::trainer::Trainer`]'s gradient-
/// accumulation + clipping + LR-schedule plumbing — just with a different
/// loss and a smaller surface area.
pub struct RewardTrainer {
    config: TrainConfig,
    varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    model: RewardModel,
    progress: Option<Arc<dyn TrainingProgress>>,
    global_step: usize,
    lr_scheduler: Box<dyn Fn(usize) -> f64 + Send + Sync>,
    total_steps: usize,
    accum_counter: usize,
}

impl RewardTrainer {
    /// Construct a reward trainer from a [`RewardConfig`] + [`VarMap`] +
    /// [`Device`] + a built [`RewardModel`].
    ///
    /// The caller is responsible for building the [`RewardModel`] against
    /// the supplied varmap (the encoder's LoRA params and the reward head
    /// must already be registered in the varmap before this constructor
    /// runs, since AdamW captures the param set up-front).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::InvalidConfig`] if `lora.rank == 0` or
    /// `core.max_steps == 0`, or [`BlazenTrainError::Optimizer`] if
    /// `AdamW::new` rejects the trainable param set.
    pub fn new(
        cfg: RewardConfig,
        varmap: VarMap,
        device: Device,
        model: RewardModel,
        progress: Option<Arc<dyn TrainingProgress>>,
    ) -> Result<Self, BlazenTrainError> {
        if cfg.lora.rank == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "reward: lora.rank must be > 0".to_string(),
            ));
        }
        if cfg.core.max_steps == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "reward: core.max_steps must be > 0".to_string(),
            ));
        }

        // Synthesize a TrainConfig so the per-step plumbing (checkpoint,
        // grad-clip, scheduler) can reuse the existing helpers verbatim.
        let synthesized = TrainConfig {
            base_model_repo: cfg.core.base_model_repo,
            output_dir: cfg.core.output_dir,
            lora: cfg.lora,
            optim: cfg.core.optim,
            scheduler: cfg.core.scheduler,
            max_steps: cfg.core.max_steps,
            batch_size: cfg.core.batch_size,
            gradient_accumulation_steps: cfg.core.gradient_accumulation_steps,
            max_seq_len: cfg.core.max_seq_len,
            eval_steps: cfg.core.eval_steps,
            save_steps: cfg.core.save_steps,
            seed: cfg.core.seed,
            mixed_precision: cfg.core.mixed_precision,
            device: cfg.core.device,
        };

        let trainable = collect_reward_trainable_params(&varmap, &synthesized.lora);
        let params = ParamsAdamW {
            lr: synthesized.optim.learning_rate,
            beta1: synthesized.optim.beta1,
            beta2: synthesized.optim.beta2,
            eps: synthesized.optim.epsilon,
            weight_decay: synthesized.optim.weight_decay,
        };
        let optimizer = AdamW::new(trainable, params)
            .map_err(|e| BlazenTrainError::Optimizer(e.to_string()))?;

        let lr_scheduler = schedulers::make_scheduler(
            &synthesized.scheduler,
            synthesized.optim.learning_rate,
            synthesized.max_steps,
        );
        let total_steps = synthesized.max_steps;

        Ok(Self {
            config: synthesized,
            varmap,
            optimizer,
            device,
            model,
            progress: None.or(progress),
            global_step: 0,
            lr_scheduler,
            total_steps,
            accum_counter: 0,
        })
    }

    /// Borrow the active config.
    #[must_use]
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Borrow the [`VarMap`] backing the training run.
    #[must_use]
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Borrow the device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Borrow the trained reward model.
    #[must_use]
    pub fn model(&self) -> &RewardModel {
        &self.model
    }

    /// Current 0-indexed optimizer step counter.
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Run one preference-pair training step.
    ///
    /// Computes scalar rewards `r_c`, `r_r` for the chosen and rejected
    /// completions, then minimizes
    ///
    /// ```text
    /// loss = -mean_i log_sigmoid( r_chosen_i - r_rejected_i )
    /// ```
    ///
    /// Gradient accumulation, clipping, and optimizer stepping mirror
    /// [`crate::trainer::Trainer::step_dpo`] verbatim.
    ///
    /// # Errors
    ///
    /// Forwards any candle / optimizer / forward error.
    pub fn step(&mut self, batch: &PreferenceBatch) -> Result<f32, BlazenTrainError> {
        let lr = (self.lr_scheduler)(self.global_step);
        self.optimizer.set_learning_rate(lr);

        let r_chosen = self
            .model
            .forward_rewards(&batch.chosen_input_ids, &batch.chosen_attn)?;
        let r_rejected = self
            .model
            .forward_rewards(&batch.rejected_input_ids, &batch.rejected_attn)?;

        // margin = r_c - r_r, shape [B] f32.
        let margin = (&r_chosen - &r_rejected)?;
        // loss = -mean log_sigmoid(margin)
        let lsig = log_sigmoid(&margin)?;
        let loss = lsig.neg()?.mean_all()?;

        let accum = self.config.gradient_accumulation_steps.max(1);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / accum as f64;
        let scaled_loss = (&loss * scale)?;

        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let mut grads = scaled_loss.backward()?;

        self.accum_counter += 1;
        if self.accum_counter >= accum {
            if let Some(max) = self.config.optim.gradient_clip {
                let vars = collect_reward_trainable_params(&self.varmap, &self.config.lora);
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

    /// Drive the trainer over a [`PreferenceDataset`] until `max_steps`
    /// is reached.
    ///
    /// Emits the usual [`TrainingEvent`]s through any attached progress
    /// sink and writes optional periodic checkpoints. Returns a
    /// [`TrainedAdapter`] pointing at `core.output_dir` — note that
    /// PR-R phase 1 does NOT write a PEFT-style `adapter_config.json`
    /// because the reward head is a non-PEFT extra row; the on-disk
    /// shape is finalized in phase 2 alongside the loader.
    ///
    /// # Errors
    ///
    /// Forwards any [`BlazenTrainError`] from [`Self::step`],
    /// [`crate::checkpoint::save_checkpoint`], or a cancelling progress
    /// callback.
    pub async fn run(
        &mut self,
        dataset: Arc<dyn PreferenceDataset>,
    ) -> Result<TrainedAdapter, BlazenTrainError> {
        emit_event(
            self.progress.as_ref(),
            TrainingEvent::Started {
                total_steps: self.total_steps,
            },
        )?;

        std::fs::create_dir_all(&self.config.output_dir).map_err(|e| {
            BlazenTrainError::Export(format!(
                "create reward output dir {}: {e}",
                self.config.output_dir.display()
            ))
        })?;

        let mut final_loss = 0.0_f32;
        while self.global_step < self.total_steps {
            let step_idx = self.global_step;
            let batch = dataset.batch(self.config.batch_size, step_idx).await?;
            let started = Instant::now();
            let loss = self.step(&batch)?;
            let elapsed = started.elapsed();
            final_loss = loss;

            let lr = (self.lr_scheduler)(step_idx);
            emit_event(
                self.progress.as_ref(),
                TrainingEvent::StepCompleted {
                    step: step_idx,
                    loss,
                    learning_rate: lr,
                    elapsed,
                },
            )?;

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
                emit_event(
                    self.progress.as_ref(),
                    TrainingEvent::CheckpointSaved {
                        step: self.global_step,
                        path: cp_path,
                    },
                )?;
            }
        }

        let adapter = TrainedAdapter {
            adapter_dir: self.config.output_dir.clone(),
            final_loss,
            total_steps: self.global_step,
        };
        emit_event(
            self.progress.as_ref(),
            TrainingEvent::Finished {
                final_loss,
                total_steps: self.global_step,
                adapter_dir: self.config.output_dir.clone(),
            },
        )?;

        Ok(adapter)
    }

    /// Helper for the GRPO trainer: borrow the reward model immutably so it
    /// can be scored against without taking ownership.
    #[must_use]
    pub fn into_model_and_varmap(self) -> (RewardModel, VarMap, Device, PathBuf) {
        (self.model, self.varmap, self.device, self.config.output_dir)
    }
}

/// Filter the varmap down to the params the reward trainer should update:
/// LoRA A/B rows plus the reward head row. Mirrors
/// [`freeze_base_params`] in spirit but adds the reward-head key.
fn collect_reward_trainable_params(
    varmap: &VarMap,
    lora_cfg: &LoraConfig,
) -> Vec<candle_core::Var> {
    let target_refs: Vec<&str> = lora_cfg.target_modules.iter().map(String::as_str).collect();
    let mut params = freeze_base_params(varmap, &target_refs);

    let guard = varmap
        .data()
        .lock()
        .expect("varmap mutex poisoned by another thread");
    for (name, var) in guard.iter() {
        if name.ends_with("reward_head.weight") {
            params.push(var.clone());
        }
    }
    params
}

fn emit_event(
    progress: Option<&Arc<dyn TrainingProgress>>,
    event: TrainingEvent,
) -> Result<(), BlazenTrainError> {
    match progress {
        Some(sink) => sink
            .on_event(event)
            .map_err(|_| BlazenTrainError::Cancelled),
        None => Ok(()),
    }
}

/// Numerically stable `log(sigmoid(x))` — same identity used in
/// `trainer.rs`. Duplicated here so this module doesn't have to depend
/// on a `pub(crate)` symbol that lives next to unrelated DPO/KTO state.
fn log_sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = x.zeros_like()?;
    let min_x_zero = x.minimum(&zeros)?;
    let abs_x = x.abs()?;
    let inner = (abs_x.neg()?.exp()? + 1.0_f64)?.log()?;
    min_x_zero - inner
}

/// Build a [`RewardModel`] from disk by mmap'ing a Llama checkpoint and
/// constructing a fresh reward head in `head_varmap`.
///
/// Convenience helper for callers that already have a config + safetensors
/// on disk (e.g. the GRPO trainer wiring up a pretrained reward model).
/// HF-Hub download is deferred to phase 2.
///
/// # Errors
///
/// Forwards any candle / IO / parse error from the underlying load.
pub fn build_reward_model_from_llama(
    cfg: &LlamaConfig,
    base_vb: VarBuilder,
    head_vb: VarBuilder,
    lora_cfg: &LoraConfig,
) -> Result<RewardModel, BlazenTrainError> {
    let encoder = TrainableLlama::load(base_vb, head_vb.clone(), cfg, lora_cfg)?;
    RewardModel::new(encoder, cfg, head_vb)
}

/// Resolve the candle [`DType`] for a [`MixedPrecision`] mode. Re-exported
/// for tests / external callers that build VarBuilders by hand.
#[must_use]
pub fn dtype_for_mixed_precision(mp: MixedPrecision) -> DType {
    match mp {
        MixedPrecision::None => DType::F32,
        MixedPrecision::Bf16 => DType::BF16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::llama::Config as LlamaCfg;
    use crate::config::{OptimConfig, SchedulerConfig, SchedulerKind, TrainCoreConfig};
    use std::path::PathBuf;

    fn tiny_llama_config() -> LlamaCfg {
        LlamaCfg {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 32,
            tie_word_embeddings: false,
        }
    }

    fn tiny_reward_config(output_dir: PathBuf, max_steps: usize) -> RewardConfig {
        RewardConfig {
            core: TrainCoreConfig {
                base_model_repo: "test/local-llama".to_string(),
                base_model_revision: None,
                output_dir,
                max_steps,
                batch_size: 2,
                gradient_accumulation_steps: 1,
                max_seq_len: 8,
                eval_steps: None,
                save_steps: None,
                seed: 42,
                mixed_precision: MixedPrecision::None,
                device: None,
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
            },
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
        }
    }

    fn build_tiny_reward(varmap: &VarMap, device: &Device, lora_cfg: &LoraConfig) -> RewardModel {
        let cfg = tiny_llama_config();
        let base_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let lora_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let encoder = TrainableLlama::load(base_vb, lora_vb, &cfg, lora_cfg).expect("load encoder");
        let head_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        RewardModel::new(encoder, &cfg, head_vb).expect("reward head")
    }

    fn make_pref_batch(device: &Device) -> PreferenceBatch {
        // Chosen and rejected have the same shape and identical attention masks
        // (every position is real), so the forward sees both as length-8
        // sequences. Different token offsets ensure the encoder cannot reduce
        // the rewards to zero by ignoring inputs.
        let batch_size = 2;
        let seq_len = 6;
        let total = batch_size * seq_len;
        let chosen: Vec<u32> = (0..total)
            .map(|i| u32::try_from(i % 128).unwrap())
            .collect();
        let rejected: Vec<u32> = (0..total)
            .map(|i| u32::try_from((i + 17) % 128).unwrap())
            .collect();
        let chosen_labels: Vec<i64> = chosen.iter().map(|&t| i64::from(t)).collect();
        let rejected_labels: Vec<i64> = rejected.iter().map(|&t| i64::from(t)).collect();
        let mask: Vec<u32> = vec![1; total];
        let shape = (batch_size, seq_len);
        PreferenceBatch {
            chosen_input_ids: Tensor::from_vec(chosen, shape, device).unwrap(),
            chosen_labels: Tensor::from_vec(chosen_labels, shape, device).unwrap(),
            chosen_attn: Tensor::from_vec(mask.clone(), shape, device).unwrap(),
            rejected_input_ids: Tensor::from_vec(rejected, shape, device).unwrap(),
            rejected_labels: Tensor::from_vec(rejected_labels, shape, device).unwrap(),
            rejected_attn: Tensor::from_vec(mask, shape, device).unwrap(),
        }
    }

    #[test]
    fn reward_forward_shape_correct() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let cfg = tiny_reward_config(PathBuf::from("/tmp-unused"), 1);
        let model = build_tiny_reward(&varmap, &device, &cfg.lora);

        let batch = make_pref_batch(&device);
        let rewards = model
            .forward_rewards(&batch.chosen_input_ids, &batch.chosen_attn)
            .expect("forward");
        assert_eq!(rewards.dims(), &[2]);
    }

    #[test]
    fn reward_head_registered_in_varmap() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let cfg = tiny_reward_config(PathBuf::from("/tmp-unused"), 1);
        let _model = build_tiny_reward(&varmap, &device, &cfg.lora);

        let guard = varmap.data().lock().expect("varmap mutex");
        let has_head = guard
            .keys()
            .any(|k| k.ends_with("reward_head.weight") || k == REWARD_HEAD_WEIGHT_KEY);
        assert!(
            has_head,
            "expected reward_head.weight in varmap, keys: {:?}",
            guard.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn reward_loss_chosen_higher_than_rejected_yields_negative_grad() {
        // With LoRA-B zero-initialized and the reward head Kaiming-uniform,
        // initial r_c - r_r is close to zero, so loss ≈ ln(2). After a few
        // optimization steps on a fixed batch where we hand the trainer a
        // consistent (chosen, rejected) signal, the loss must decrease — i.e.
        // the gradient on the head pushed r_chosen up relative to r_rejected.
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_reward_config(tmp.path().to_path_buf(), 30);
        let varmap = VarMap::new();
        let model = build_tiny_reward(&varmap, &device, &cfg.lora);

        let mut trainer = RewardTrainer::new(cfg, varmap, device.clone(), model, None)
            .expect("RewardTrainer::new");
        let batch = make_pref_batch(&device);
        let initial = trainer.step(&batch).expect("initial step");
        let mut last = initial;
        for _ in 1..30 {
            last = trainer.step(&batch).expect("step");
        }
        eprintln!("reward loss: initial={initial} final={last}");
        assert!(
            last < initial * 0.9,
            "reward loss did not decrease: initial={initial} final={last}"
        );
    }
}
