//! Group Relative Policy Optimization (GRPO).
//!
//! DeepSeek's critic-free PPO replacement. Each training step:
//!
//! 1. For each prompt in the batch, sample `K = group_size` completions
//!    from the current policy (mix of greedy + temperature-stochastic).
//! 2. Score every completion with a frozen reward model
//!    ([`crate::reward::RewardModel`]).
//! 3. Compute per-group baselined advantages:
//!    `advantage_i = (r_i - mean_group_r) / (std_group_r + eps)`.
//! 4. Forward the policy on each completion (autograd-live), gather the
//!    per-row log-probability of the sampled tokens.
//! 5. Forward a frozen reference copy of the policy and compute the
//!    per-token approximate KL between the two distributions.
//! 6. Minimize
//!    `loss = -mean_i( advantage_i * log_prob_i ) + beta * mean_i( KL_i )`.
//! 7. Backward + AdamW step + accumulation/clipping.
//!
//! In plain English: increase the probability of completions that scored
//! better than their group's average, decrease the probability of those
//! that scored worse, and never wander too far from the frozen reference
//! policy.
//!
//! Phase 1 scope (this module):
//!  - [`GrpoTrainer`] holds the policy, reference, and reward model; owns
//!    the AdamW optimizer over the policy's LoRA params; runs the GRPO
//!    step against a caller-provided [`GrpoBatch`] of pre-sampled
//!    completions.
//!  - In-trainer completion sampling, HF-Hub reward-model loading, and
//!    binding parity (Python/Node/WASM/UniFFI/CABI) are deferred to
//!    phase 2 — phase 1 keeps the loss kernel and gradient plumbing in
//!    one focused module so the math can be exercised end-to-end with a
//!    tiny CPU model.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::arch::llama::{Config as LlamaConfig, TrainableLlama};
use crate::config::{GrpoConfig, LoraConfig, TrainConfig};
use crate::error::BlazenTrainError;
use crate::grad_clip;
use crate::lora::freeze_base_params;
use crate::progress::TrainingProgress;
use crate::reward::RewardModel;
use crate::schedulers;

/// One GRPO training batch.
///
/// Holds `B * K` rows of *prompt + sampled completion* concatenated into
/// `[B*K, T]` token tensors, plus the per-row labels that indicate which
/// positions are the sampled-completion tokens (everything else is
/// [`IGNORE_INDEX`] = `-100`), an attention mask, and a `[B*K]` group id
/// telling the loss which K-tuple each row belongs to.
///
/// The trainer computes per-row log-probs of the labeled tokens, groups
/// them by `group_id`, and applies the GRPO loss against the per-row
/// reward scores produced by the reward model in [`crate::reward`].
///
/// In phase 2 the trainer will own a sampler that builds this batch from
/// a [`PromptDataset`]; phase 1 leaves construction to callers and tests.
#[derive(Debug, Clone)]
pub struct GrpoBatch {
    /// Token IDs `[B*K, T]`, dtype `u32`. Each row is a prompt followed by
    /// one sampled completion, right-padded to `T`.
    pub input_ids: Tensor,
    /// Labels `[B*K, T]`, dtype `i64`. Sampled-completion positions hold
    /// the actual token IDs; prompt + pad positions are
    /// [`IGNORE_INDEX`] (-100) and are excluded from both the log-prob
    /// gather and the KL term.
    pub labels: Tensor,
    /// Attention mask `[B*K, T]`, dtype `u32`. `1` at real tokens, `0` at
    /// right-side padding. Same convention as `PreferenceBatch`.
    pub attention_mask: Tensor,
    /// Group assignment `[B*K]`, dtype `u32`. Rows with the same value
    /// belong to the same prompt's K-tuple and share an advantage
    /// normalizer. Group ids must be 0..G-1 dense.
    pub group_ids: Tensor,
    /// Reward scores `[B*K]`, dtype `f32`. Computed by the caller with
    /// the frozen reward model; the trainer treats these as constants
    /// (no autograd reaches them).
    pub rewards: Tensor,
}

/// `-100`: HF/PEFT convention for "ignore at the loss reduction".
/// Duplicated here so this module doesn't have to depend on a
/// `pub(crate)` constant living next to unrelated DPO/KTO state.
pub const IGNORE_INDEX: i64 = -100;

/// GRPO trainer.
///
/// Owns:
///  - the policy [`TrainableLlama`] and its [`VarMap`] (the trainable
///    LoRA params live here),
///  - a frozen reference [`TrainableLlama`] (a copy of the policy at
///    step 0, used for the KL regularizer),
///  - a frozen [`RewardModel`] (only required at sampling time; phase 1
///    accepts pre-scored batches so the model is optional in the
///    constructor and surfaced via [`Self::set_reward_model`] for use
///    by the phase-2 sampler).
///
/// The reference is built by deep-copying the policy's varmap before
/// any gradient steps; subsequent updates to the policy varmap have no
/// effect on the reference. This mirrors DPO's reference-model pattern
/// from [`crate::trainer::Trainer::load_models_dpo`].
pub struct GrpoTrainer {
    config: TrainConfig,
    grpo_cfg: GrpoCoreState,
    policy_varmap: VarMap,
    policy: TrainableLlama,
    reference: TrainableLlama,
    /// Frozen reward model; held in an [`Arc`] so the phase-2 sampler can
    /// share it across worker threads. Phase 1 leaves it optional — the
    /// trainer's `step` ignores it because batches arrive pre-scored.
    reward_model: Option<Arc<RewardModel>>,
    optimizer: AdamW,
    device: Device,
    progress: Option<Arc<dyn TrainingProgress>>,
    global_step: usize,
    lr_scheduler: Box<dyn Fn(usize) -> f64 + Send + Sync>,
    total_steps: usize,
    accum_counter: usize,
}

/// GRPO-specific state extracted from [`GrpoConfig`].
struct GrpoCoreState {
    /// K — completions per prompt.
    group_size: usize,
    /// KL-regularization strength.
    beta: f32,
    /// Per-group std-dev floor.
    advantage_epsilon: f32,
}

impl GrpoTrainer {
    /// Construct a GRPO trainer.
    ///
    /// The caller is responsible for building the policy [`TrainableLlama`]
    /// against `varmap` (its LoRA params must already be registered in the
    /// varmap before AdamW captures them) and a frozen reference
    /// [`TrainableLlama`] whose weights are a snapshot of the policy.
    ///
    /// Llama-only in phase 1; multi-arch dispatch lands in phase 2 alongside
    /// HF-Hub loading.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::InvalidConfig`] if `group_size < 2`,
    /// `lora.rank == 0`, or `core.max_steps == 0`. Returns
    /// [`BlazenTrainError::Optimizer`] if `AdamW::new` rejects the trainable
    /// param set.
    pub fn new(
        cfg: GrpoConfig,
        varmap: VarMap,
        device: Device,
        policy: TrainableLlama,
        reference: TrainableLlama,
        progress: Option<Arc<dyn TrainingProgress>>,
    ) -> Result<Self, BlazenTrainError> {
        if cfg.group_size < 2 {
            return Err(BlazenTrainError::InvalidConfig(
                "grpo: group_size must be >= 2 (a singleton group has zero variance)".to_string(),
            ));
        }
        if cfg.lora.rank == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "grpo: lora.rank must be > 0".to_string(),
            ));
        }
        if cfg.core.max_steps == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "grpo: core.max_steps must be > 0".to_string(),
            ));
        }

        let GrpoConfig {
            core,
            lora,
            group_size,
            beta,
            advantage_epsilon,
            sampling_temperature: _,
            reward_model_repo: _,
            reward_model_revision: _,
        } = cfg;

        let synthesized = TrainConfig {
            base_model_repo: core.base_model_repo,
            output_dir: core.output_dir,
            lora,
            optim: core.optim,
            scheduler: core.scheduler,
            max_steps: core.max_steps,
            batch_size: core.batch_size,
            gradient_accumulation_steps: core.gradient_accumulation_steps,
            max_seq_len: core.max_seq_len,
            eval_steps: core.eval_steps,
            save_steps: core.save_steps,
            seed: core.seed,
            mixed_precision: core.mixed_precision,
            device: core.device,
        };

        let target_refs: Vec<&str> = synthesized
            .lora
            .target_modules
            .iter()
            .map(String::as_str)
            .collect();
        let trainable = freeze_base_params(&varmap, &target_refs);
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
            grpo_cfg: GrpoCoreState {
                group_size,
                beta,
                advantage_epsilon,
            },
            policy_varmap: varmap,
            policy,
            reference,
            reward_model: None,
            optimizer,
            device,
            progress,
            global_step: 0,
            lr_scheduler,
            total_steps,
            accum_counter: 0,
        })
    }

    /// Attach a frozen reward model. Required by the phase-2 in-trainer
    /// sampler; phase 1 leaves it optional (pre-scored batches only).
    pub fn set_reward_model(&mut self, reward: Arc<RewardModel>) {
        self.reward_model = Some(reward);
    }

    /// Borrow the active config.
    #[must_use]
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Borrow the policy varmap.
    #[must_use]
    pub fn varmap(&self) -> &VarMap {
        &self.policy_varmap
    }

    /// Borrow the device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Current 0-indexed optimizer step counter.
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    // --- Distributed-training accessors (used by AllReduceGrpoTrainer) ---

    /// Evaluate the LR schedule at `step` without mutating optimizer state.
    /// Used by [`crate::distributed::AllReduceGrpoTrainer`].
    #[must_use]
    pub fn lr_for_step(&self, step: usize) -> f64 {
        (self.lr_scheduler)(step)
    }

    /// Apply `lr` to the inner optimizer. Used by
    /// [`crate::distributed::AllReduceGrpoTrainer`] right before the
    /// per-step optimizer kick.
    pub fn set_optimizer_lr(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    /// GRPO KL-regularization coefficient (β).
    #[must_use]
    pub(crate) fn beta_coeff(&self) -> f32 {
        self.grpo_cfg.beta
    }

    /// GRPO advantage-normalization epsilon (per-group std floor).
    #[must_use]
    pub(crate) fn advantage_epsilon(&self) -> f32 {
        self.grpo_cfg.advantage_epsilon
    }

    /// Borrow the policy [`TrainableLlama`] (autograd-live forward pass).
    pub(crate) fn policy(&self) -> &TrainableLlama {
        &self.policy
    }

    /// Borrow the frozen reference [`TrainableLlama`] (used for the KL term).
    pub(crate) fn reference(&self) -> &TrainableLlama {
        &self.reference
    }

    /// Run the gradient-clip + optimizer step on `grads` if the accumulation
    /// counter has reached `gradient_accumulation_steps`. Returns `true` when
    /// the optimizer actually stepped.
    ///
    /// Used by [`crate::distributed::AllReduceGrpoTrainer`] after the
    /// AllReduce splice so the wrapper can decide whether to bump
    /// `global_step`.
    ///
    /// # Errors
    /// - [`BlazenTrainError::Optimizer`] when the AdamW step fails.
    /// - [`BlazenTrainError::Candle`] for grad-clip tensor errors.
    pub fn maybe_step_optimizer(
        &mut self,
        grads: &mut candle_core::backprop::GradStore,
        vars: &[candle_core::Var],
    ) -> Result<bool, BlazenTrainError> {
        let accum = self.config.gradient_accumulation_steps.max(1);
        self.accum_counter += 1;
        if self.accum_counter < accum {
            return Ok(false);
        }
        if let Some(max) = self.config.optim.gradient_clip {
            let var_refs: Vec<&candle_core::Var> = vars.iter().collect();
            grad_clip::clip_grad_norm(grads, &var_refs, max)?;
        }
        self.optimizer
            .step(grads)
            .map_err(|e| BlazenTrainError::Optimizer(e.to_string()))?;
        self.accum_counter = 0;
        Ok(true)
    }

    /// Increment the global step counter. Used by
    /// [`crate::distributed::AllReduceGrpoTrainer`] after a successful
    /// optimizer step.
    pub fn bump_global_step(&mut self) {
        self.global_step += 1;
    }

    /// Compute group-relative advantages from raw per-row rewards.
    ///
    /// `rewards` is `[B*K]` f32 and `group_ids` is `[B*K]` u32 with values
    /// in `0..G`. For each group g the advantage is
    /// `(r_i - mean_g) / (std_g + eps)` for every row `i` assigned to g.
    /// The result is `[B*K]` f32 with the same row ordering as the inputs.
    ///
    /// Exposed publicly so tests + the phase-2 sampler can verify the
    /// invariant `mean(adv_per_group) ≈ 0` and `std(adv_per_group) ≈ 1`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Forward`] if `rewards` and `group_ids`
    /// disagree on shape, or if any group is empty.
    pub fn group_relative_advantages(
        rewards: &Tensor,
        group_ids: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, BlazenTrainError> {
        let rewards_vec: Vec<f32> = rewards.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let group_ids_vec: Vec<u32> = group_ids.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        if rewards_vec.len() != group_ids_vec.len() {
            return Err(BlazenTrainError::Forward(format!(
                "grpo: rewards len {} != group_ids len {}",
                rewards_vec.len(),
                group_ids_vec.len()
            )));
        }
        let num_groups = group_ids_vec
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        if num_groups == 0 {
            return Err(BlazenTrainError::Forward(
                "grpo: empty rewards tensor".to_string(),
            ));
        }

        let mut means = vec![0.0_f64; num_groups];
        let mut counts = vec![0_u64; num_groups];
        for (&r, &g) in rewards_vec.iter().zip(group_ids_vec.iter()) {
            means[g as usize] += f64::from(r);
            counts[g as usize] += 1;
        }
        for g in 0..num_groups {
            if counts[g] == 0 {
                return Err(BlazenTrainError::Forward(format!(
                    "grpo: group {g} is empty"
                )));
            }
            #[allow(clippy::cast_precision_loss)]
            let denom = counts[g] as f64;
            means[g] /= denom;
        }

        let mut variances = vec![0.0_f64; num_groups];
        for (&r, &g) in rewards_vec.iter().zip(group_ids_vec.iter()) {
            let d = f64::from(r) - means[g as usize];
            variances[g as usize] += d * d;
        }
        let mut stds = vec![0.0_f64; num_groups];
        for g in 0..num_groups {
            #[allow(clippy::cast_precision_loss)]
            let denom = counts[g] as f64;
            stds[g] = (variances[g] / denom).sqrt();
        }

        let eps = f64::from(epsilon);
        let advantages: Vec<f32> = rewards_vec
            .iter()
            .zip(group_ids_vec.iter())
            .map(|(&r, &g)| {
                let mean = means[g as usize];
                let std = stds[g as usize];
                #[allow(clippy::cast_possible_truncation)]
                let adv = ((f64::from(r) - mean) / (std + eps)) as f32;
                adv
            })
            .collect();
        Tensor::from_vec(advantages, rewards_vec.len(), rewards.device()).map_err(Into::into)
    }

    /// Run one GRPO step against a pre-built batch.
    ///
    /// The batch must already carry reward scores (typically computed by
    /// the caller via [`RewardModel::forward_rewards`]). The trainer
    /// computes group-relative advantages, the policy / reference log-probs,
    /// the approximate KL, and the combined GRPO loss; then backprops
    /// through the policy.
    ///
    /// Loss formula:
    /// ```text
    /// adv_i  = (r_i - mean_group) / (std_group + eps)         # detached
    /// lp_i   = mean over completion tokens of log π_pol(t|...)
    /// lr_i   = mean over completion tokens of log π_ref(t|...) # detached
    /// kl_i   = lr_i - lp_i                                    # k1 approx
    /// loss   = -mean_i( adv_i * lp_i ) + beta * mean_i( kl_i )
    /// ```
    ///
    /// The advantage tensor is detached (no autograd through the reward
    /// model). The reference log-probs are detached. Only the policy's
    /// LoRA params receive gradient.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Forward`] for shape mismatches; forwards
    /// any candle tensor error.
    pub fn step(&mut self, batch: &GrpoBatch) -> Result<f32, BlazenTrainError> {
        let lr = (self.lr_scheduler)(self.global_step);
        self.optimizer.set_learning_rate(lr);

        // Advantages: computed on the host (no autograd needed — the reward
        // signal is treated as a constant by GRPO).
        let advantages = Self::group_relative_advantages(
            &batch.rewards,
            &batch.group_ids,
            self.grpo_cfg.advantage_epsilon,
        )?
        .detach();

        // Policy forward — WITH autograd.
        let policy_logits = self.policy.forward(&batch.input_ids)?;
        // Reference forward — autograd severed via .detach().
        let ref_logits = self.reference.forward(&batch.input_ids)?.detach();

        // Per-row mean log-prob of the labeled (completion) tokens.
        let lp_policy = mean_label_log_probs(&policy_logits, &batch.labels)?;
        let lp_ref = mean_label_log_probs(&ref_logits, &batch.labels)?.detach();

        // Policy-gradient term: -mean( adv * lp_policy ).
        let pg = (&advantages * &lp_policy)?.mean_all()?.neg()?;

        // KL term (approx, k1): mean( lp_ref - lp_policy ). Drops to zero
        // when policy == reference. We follow DeepSeek-Math: take the
        // straightforward difference rather than the more expensive k3
        // estimator — it's cheap and the gradient sign is identical.
        let kl_per_row = (&lp_ref - &lp_policy)?;
        let kl_term = kl_per_row.mean_all()?;
        let kl_scaled = (&kl_term * f64::from(self.grpo_cfg.beta))?;

        let loss = (pg + kl_scaled)?;
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
                let vars = freeze_base_params(&self.policy_varmap, &target_refs);
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

    /// Drive the trainer over a caller-provided iterator of pre-built
    /// batches until `max_steps` is reached.
    ///
    /// Phase 1's stand-in for a full `run(dataset)` driver — once the
    /// phase-2 in-trainer sampler lands the `Vec<GrpoBatch>` argument
    /// becomes a `PromptDataset` and this method gets renamed to `run`.
    ///
    /// Emits the usual [`TrainingEvent`]s through any attached progress
    /// sink. Returns the final per-step loss observed.
    ///
    /// # Errors
    ///
    /// Forwards any [`BlazenTrainError`] from [`Self::step`] or a
    /// cancelling progress callback. Returns
    /// [`BlazenTrainError::Dataset`] if the batch iterator runs dry
    /// before `max_steps`.
    pub fn run_batches(&mut self, batches: &[GrpoBatch]) -> Result<f32, BlazenTrainError> {
        use crate::progress::TrainingEvent;

        emit_event(
            self.progress.as_ref(),
            TrainingEvent::Started {
                total_steps: self.total_steps,
            },
        )?;

        let mut final_loss = 0.0_f32;
        let mut idx = 0;
        while self.global_step < self.total_steps {
            let batch = batches.get(idx).ok_or_else(|| {
                BlazenTrainError::Dataset(format!(
                    "grpo run_batches: ran out of batches at step {} (need {})",
                    self.global_step, self.total_steps
                ))
            })?;
            let step_idx = self.global_step;
            let started = std::time::Instant::now();
            let loss = self.step(batch)?;
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
            idx = (idx + 1) % batches.len().max(1);
        }

        emit_event(
            self.progress.as_ref(),
            TrainingEvent::Finished {
                final_loss,
                total_steps: self.global_step,
                adapter_dir: self.config.output_dir.clone(),
            },
        )?;
        Ok(final_loss)
    }

    /// Borrow the active GRPO group size (for callers that need to match
    /// it when constructing batches).
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.grpo_cfg.group_size
    }

    /// Borrow the active GRPO beta (KL coefficient).
    #[must_use]
    pub fn beta(&self) -> f32 {
        self.grpo_cfg.beta
    }

    /// Output directory the trainer will write checkpoints to.
    #[must_use]
    pub fn output_dir(&self) -> PathBuf {
        self.config.output_dir.clone()
    }
}

fn emit_event(
    progress: Option<&Arc<dyn TrainingProgress>>,
    event: crate::progress::TrainingEvent,
) -> Result<(), BlazenTrainError> {
    match progress {
        Some(sink) => sink
            .on_event(event)
            .map_err(|_| BlazenTrainError::Cancelled),
        None => Ok(()),
    }
}

/// Compute per-row mean log-prob of the *labeled* (non-`-100`) tokens.
///
/// `logits` is `[B, T, V]` (autograd-live or detached). `labels` is `[B, T]`
/// `i64` with `-100` at positions to skip. Returns `[B]` `f32`:
///
/// ```text
/// mean_i = ( sum_{t : label_t != -100} log p(label_t | ...) ) / count_t
/// ```
///
/// Rows with zero labeled tokens get a zero result rather than NaN, matching
/// the trainer's `masked_cross_entropy` convention.
///
/// Implementation mirrors `trainer.rs::sequence_logprobs` but with a per-row
/// average instead of a sum — GRPO normalizes by completion length so longer
/// completions don't dominate the advantage signal.
pub(crate) fn mean_label_log_probs(
    logits: &Tensor,
    labels: &Tensor,
) -> Result<Tensor, BlazenTrainError> {
    let log_probs = candle_nn::ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;

    let labels_i64 = labels.to_dtype(DType::I64)?;
    let keep_mask_bool = labels_i64.ne(IGNORE_INDEX)?;
    let zero_tensor = Tensor::zeros(labels_i64.shape(), DType::I64, labels_i64.device())?;
    let safe_labels = keep_mask_bool.where_cond(&labels_i64, &zero_tensor)?;
    let safe_labels_u32 = safe_labels.to_dtype(DType::U32)?;

    let gathered = log_probs
        .gather(&safe_labels_u32.unsqueeze(2)?, 2)?
        .squeeze(2)?; // [B, T]

    let keep_f32 = keep_mask_bool.to_dtype(DType::F32)?;
    let masked = (gathered * &keep_f32)?;
    let sum_per_row = masked.sum(D::Minus1)?; // [B]
    let counts_per_row = keep_f32.sum(D::Minus1)?; // [B]

    // Avoid division by zero: replace zeros with ones (the corresponding
    // sum is already zero, so the resulting mean is zero). Building the
    // "ones-where-zero" mask via a where_cond keeps everything autograd-
    // friendly.
    let zero = Tensor::zeros(counts_per_row.shape(), DType::F32, counts_per_row.device())?;
    let one = Tensor::ones(counts_per_row.shape(), DType::F32, counts_per_row.device())?;
    let nonzero_mask = counts_per_row.gt(&zero)?;
    let safe_counts = nonzero_mask.where_cond(&counts_per_row, &one)?;
    Ok((sum_per_row / safe_counts)?)
}

/// Build a frozen reference [`TrainableLlama`] by copying the policy
/// weights through a fresh `VarBuilder::from_varmap` into a dedicated
/// reference varmap. Tests + the phase-2 sampler use this to spin up a
/// pristine reference without re-reading weights from disk.
///
/// The reference's varmap is intentionally separate from the policy's so
/// optimizer updates on the policy do not bleed into the reference. The
/// LoRA targets are empty — the reference is just the base model.
///
/// # Errors
///
/// Forwards any candle tensor error from the underlying `load`.
///
/// # Panics
///
/// Panics if either the policy or the reference [`VarMap`]'s internal
/// mutex has been poisoned by a concurrent panic in another thread.
pub fn build_reference_from_policy(
    policy_varmap: &VarMap,
    cfg: &LlamaConfig,
    dtype: DType,
    device: &Device,
) -> Result<(TrainableLlama, VarMap), BlazenTrainError> {
    // Why: we need a snapshot of the policy. Easiest is to clone the
    // policy's varmap into a fresh one and reload llama against it. The
    // reference has no LoRA adapters (empty target_modules), so any
    // optimizer steps on the policy LoRA cannot reach it.
    let ref_varmap = VarMap::new();
    {
        let src = policy_varmap
            .data()
            .lock()
            .expect("policy varmap mutex poisoned");
        let mut dst = ref_varmap
            .data()
            .lock()
            .expect("reference varmap mutex poisoned");
        for (name, var) in src.iter() {
            // Skip LoRA params — the reference is the base-only model.
            if name.contains(".lora_A.") || name.contains(".lora_B.") {
                continue;
            }
            // Clone the underlying tensor into a fresh Var so subsequent
            // policy updates don't alias.
            let cloned_tensor = var.as_tensor().detach().copy()?;
            let new_var = candle_core::Var::from_tensor(&cloned_tensor)?;
            dst.insert(name.clone(), new_var);
        }
    }
    let base_vb = VarBuilder::from_varmap(&ref_varmap, dtype, device);
    let lora_vb = VarBuilder::from_varmap(&ref_varmap, dtype, device);
    let empty_lora = LoraConfig {
        rank: 1,
        alpha: 1.0,
        dropout: 0.0,
        target_modules: Vec::new(),
    };
    let model = TrainableLlama::load(base_vb, lora_vb, cfg, &empty_lora)?;
    Ok((model, ref_varmap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::llama::Config as LlamaCfg;
    use crate::config::{
        MixedPrecision, OptimConfig, SchedulerConfig, SchedulerKind, TrainCoreConfig,
    };
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

    fn tiny_grpo_config(output_dir: PathBuf, max_steps: usize) -> GrpoConfig {
        GrpoConfig {
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
                    learning_rate: 1e-3,
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
            group_size: 4,
            beta: 0.04,
            advantage_epsilon: 1e-8,
            sampling_temperature: 1.0,
            reward_model_repo: None,
            reward_model_revision: None,
        }
    }

    fn build_tiny_policy_and_ref(
        device: &Device,
        lora_cfg: &LoraConfig,
    ) -> (VarMap, TrainableLlama, TrainableLlama) {
        let cfg = tiny_llama_config();
        let policy_vm = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&policy_vm, DType::F32, device);
        let lora_vb = VarBuilder::from_varmap(&policy_vm, DType::F32, device);
        let policy = TrainableLlama::load(base_vb, lora_vb, &cfg, lora_cfg).expect("policy load");
        let (reference, _ref_vm) =
            build_reference_from_policy(&policy_vm, &cfg, DType::F32, device).expect("ref build");
        (policy_vm, policy, reference)
    }

    fn make_grpo_batch(
        device: &Device,
        num_prompts: usize,
        group_size: usize,
        seq_len: usize,
        prompt_len: usize,
        rewards: &[f32],
    ) -> GrpoBatch {
        // batch = num_prompts * group_size rows. labels[0..prompt_len] = -100,
        // labels[prompt_len..] = token id at that position (so the model is
        // asked to "predict the next token" of the completion).
        let bk = num_prompts * group_size;
        assert_eq!(rewards.len(), bk, "rewards must have {bk} entries");

        let mut ids: Vec<u32> = Vec::with_capacity(bk * seq_len);
        let mut labels: Vec<i64> = Vec::with_capacity(bk * seq_len);
        let mut group_ids: Vec<u32> = Vec::with_capacity(bk);
        for r in 0..bk {
            #[allow(clippy::cast_possible_truncation)]
            let group_id = (r / group_size) as u32;
            group_ids.push(group_id);
            for t in 0..seq_len {
                #[allow(clippy::cast_possible_truncation)]
                let id = ((r * seq_len + t) % 128) as u32;
                ids.push(id);
                if t < prompt_len {
                    labels.push(IGNORE_INDEX);
                } else {
                    labels.push(i64::from(id));
                }
            }
        }
        let mask: Vec<u32> = vec![1; bk * seq_len];
        let shape = (bk, seq_len);
        GrpoBatch {
            input_ids: Tensor::from_vec(ids, shape, device).unwrap(),
            labels: Tensor::from_vec(labels, shape, device).unwrap(),
            attention_mask: Tensor::from_vec(mask, shape, device).unwrap(),
            group_ids: Tensor::from_vec(group_ids, (bk,), device).unwrap(),
            rewards: Tensor::from_vec(rewards.to_vec(), (bk,), device).unwrap(),
        }
    }

    #[test]
    fn grpo_advantages_zero_mean_unit_var_per_group() {
        // Two groups of four. With distinct values inside each group, the
        // normalized advantages should sum to ~0 (mean = 0) and the sum of
        // squares should be ~K (variance = 1).
        let device = Device::Cpu;
        let rewards = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0],
            (8,),
            &device,
        )
        .unwrap();
        let group_ids = Tensor::from_vec(vec![0u32, 0, 0, 0, 1, 1, 1, 1], (8,), &device).unwrap();
        let advs =
            GrpoTrainer::group_relative_advantages(&rewards, &group_ids, 1e-8).expect("advs");
        let advs_vec: Vec<f32> = advs.to_vec1::<f32>().unwrap();

        // Group 0: indices 0..4.
        let g0_mean: f32 = advs_vec[0..4].iter().copied().sum::<f32>() / 4.0;
        let g0_var: f32 = advs_vec[0..4].iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!(g0_mean.abs() < 1e-4, "group 0 mean = {g0_mean}, expected 0");
        assert!(
            (g0_var - 1.0).abs() < 1e-3,
            "group 0 variance = {g0_var}, expected 1"
        );

        // Group 1: indices 4..8.
        let g1_mean: f32 = advs_vec[4..8].iter().copied().sum::<f32>() / 4.0;
        let g1_var: f32 = advs_vec[4..8].iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!(g1_mean.abs() < 1e-4, "group 1 mean = {g1_mean}, expected 0");
        assert!(
            (g1_var - 1.0).abs() < 1e-3,
            "group 1 variance = {g1_var}, expected 1"
        );
    }

    #[test]
    fn grpo_advantages_degenerate_group_does_not_explode() {
        // When every reward in a group is identical, std == 0 and the
        // advantage would be 0/0 without the epsilon. The epsilon term
        // keeps the result finite (and exactly zero, since the numerator
        // is also zero).
        let device = Device::Cpu;
        let rewards = Tensor::from_vec(vec![3.0_f32, 3.0, 3.0, 3.0], (4,), &device).unwrap();
        let group_ids = Tensor::from_vec(vec![0u32, 0, 0, 0], (4,), &device).unwrap();
        let advs =
            GrpoTrainer::group_relative_advantages(&rewards, &group_ids, 1e-8).expect("advs");
        let advs_vec: Vec<f32> = advs.to_vec1::<f32>().unwrap();
        for (i, a) in advs_vec.iter().enumerate() {
            assert!(
                a.is_finite() && a.abs() < 1e-3,
                "advantage[{i}] = {a}, expected ~0 (finite)"
            );
        }
    }

    #[tokio::test]
    async fn grpo_one_step_does_not_panic() {
        // End-to-end smoke test: build a tiny llama policy + reference,
        // construct a 2-prompt × 4-completion GRPO batch with synthetic
        // rewards, take one optimizer step, assert the loss is finite.
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_grpo_config(tmp.path().to_path_buf(), 1);
        let (policy_vm, policy, reference) = build_tiny_policy_and_ref(&device, &cfg.lora);

        let mut trainer = GrpoTrainer::new(cfg, policy_vm, device.clone(), policy, reference, None)
            .expect("GrpoTrainer::new");

        // 2 prompts × 4 completions = 8 rows.
        let batch = make_grpo_batch(
            &device,
            2,
            4,
            6,
            2,
            &[0.1, 0.9, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0],
        );
        let loss = trainer.step(&batch).expect("step_grpo");
        assert!(loss.is_finite(), "GRPO loss not finite: {loss}");
        assert_eq!(trainer.global_step(), 1);
    }

    #[tokio::test]
    async fn grpo_loss_finite_across_many_steps() {
        // Stress test: 10 steps on the same batch — the loss should stay
        // finite (no NaN drift from the KL term or the advantage gather).
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_grpo_config(tmp.path().to_path_buf(), 10);
        let (policy_vm, policy, reference) = build_tiny_policy_and_ref(&device, &cfg.lora);

        let mut trainer = GrpoTrainer::new(cfg, policy_vm, device.clone(), policy, reference, None)
            .expect("GrpoTrainer::new");

        let batch = make_grpo_batch(
            &device,
            2,
            4,
            6,
            2,
            &[0.1, 0.9, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0],
        );
        for step in 0..10 {
            let loss = trainer.step(&batch).expect("step");
            assert!(
                loss.is_finite(),
                "GRPO loss at step {step} was NaN/inf: {loss}"
            );
        }
        assert_eq!(trainer.global_step(), 10);
    }
}
