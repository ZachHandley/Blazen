//! Proximal Policy Optimization (PPO).
//!
//! Classical actor-critic RLHF: each training step consumes a rollout
//! batch of `(prompt, completion, old_log_prob, old_value, reward,
//! ref_log_prob)` tuples and updates the policy via the clipped surrogate
//! objective plus a value-function regression plus an entropy bonus
//! (plus an optional KL-to-reference penalty):
//!
//! ```text
//! ratio_t   = exp( log π_pol(a_t|...) - log π_old(a_t|...) )
//! clip_t    = clip(ratio_t, 1 - eps, 1 + eps)
//! pg_loss   = -mean_t( min(ratio_t * adv_t, clip_t * adv_t) )
//! vf_loss   = mean_t( (V(s_t) - return_t)^2 )
//! ent_loss  = -mean_t( H(π_pol(·|s_t)) )
//! loss      = pg_loss + value_coef * vf_loss + entropy_coef * ent_loss
//!             + kl_coef * mean_t( KL(π_pol || π_ref) )       # optional
//! ```
//!
//! All `mean_t` reductions are taken only over completion positions
//! (prompt + pad positions are masked out via `completion_mask`).
//!
//! GAE returns:
//!
//! ```text
//! δ_t       = r_t + γ * V(s_{t+1}) - V(s_t)
//! adv_t     = δ_t + γλ * adv_{t+1}
//! return_t  = adv_t + V(s_t)
//! ```
//!
//! Rewards are placed at the **last completion position per row**
//! (terminal-reward / RLHF convention); intermediate positions get
//! `r_t = 0`. The terminal value `V(s_{end+1}) = 0`. This matches the
//! TRL `PPOTrainer.compute_advantages` formulation.
//!
//! Phase 1 scope (this module):
//!  - [`ValueModel`] = `TrainableLlama` + `Linear(hidden, 1)` head, exact
//!    mirror of [`crate::reward::RewardModel`] except the head reads
//!    per-token (not just the last non-pad position).
//!  - [`PpoBatch`] = pre-rolled buffer of rollout tensors.
//!  - [`PpoTrainer::step`] = one PPO update over a [`PpoBatch`].
//!  - GAE advantages computed on the host (no autograd reaches the
//!    reward signal or `V_old`).
//!  - Reuses [`crate::grpo::build_reference_from_policy`] for the
//!    reference snapshot.
//!
//! Phase 2 deferrals:
//!  - In-trainer `PromptDataset → PpoBatch` rollout loop (the GRPO
//!    sampler lands at the same time).
//!  - Bindings (Python/Node/WASM/UniFFI/CABI).
//!  - HF-Hub reward / critic loading.

use std::sync::Arc;

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::arch::llama::{Config as LlamaConfig, TrainableLlama};
use crate::config::{LoraConfig, PpoConfig, TrainConfig};
use crate::error::BlazenTrainError;
use crate::grad_clip;
use crate::lora::freeze_base_params;
use crate::progress::TrainingProgress;
use crate::schedulers;

/// VarMap key under which the critic's scalar value head is registered.
///
/// Public so callers exporting / merging the critic can find the row
/// without re-running the trainer. Matches the candle convention of a
/// single `weight` row per [`Linear`] with no bias. Distinct from
/// [`crate::reward::REWARD_HEAD_WEIGHT_KEY`] so a single varmap can host
/// both heads (e.g. when initializing the critic from the reward model).
pub const VALUE_HEAD_WEIGHT_KEY: &str = "value_head.weight";

/// Llama-backed value (critic) model: hidden-state encoder + per-token
/// scalar head.
///
/// The encoder is a [`TrainableLlama`] (LoRA / FFT plumbing inherited
/// for free). The scalar head is a single `Linear(hidden, 1)` with no
/// bias whose weights live in the same `VarMap` as the encoder. Unlike
/// [`crate::reward::RewardModel`] which reads only the last non-pad
/// token, the value head is applied at **every** position so PPO's
/// per-token value estimates `V(s_t)` are available for GAE.
pub struct ValueModel {
    encoder: TrainableLlama,
    head: Linear,
}

impl ValueModel {
    /// Wire up a value model on top of a freshly-loaded [`TrainableLlama`].
    ///
    /// The value-head weight is registered in `head_vb` (typically the
    /// same `VarBuilder::from_varmap` the encoder's LoRA params landed in)
    /// under the leaf [`VALUE_HEAD_WEIGHT_KEY`]. Init uses candle's
    /// default `linear_no_bias` initializer (Kaiming-uniform), which keeps
    /// initial values tightly clustered around zero.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error from the `linear_no_bias` call.
    // Why: candle's `linear_no_bias(in, out, VarBuilder)` consumes the
    // VarBuilder by value, so it's idiomatic to take ours by value too —
    // mirrors [`crate::reward::RewardModel::new`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        encoder: TrainableLlama,
        cfg: &LlamaConfig,
        head_vb: VarBuilder,
    ) -> Result<Self, BlazenTrainError> {
        let head = candle_nn::linear_no_bias(cfg.hidden_size, 1, head_vb.pp("value_head"))?;
        Ok(Self { encoder, head })
    }

    /// Borrow the underlying encoder (for tests / introspection).
    #[must_use]
    pub fn encoder(&self) -> &TrainableLlama {
        &self.encoder
    }

    /// Borrow the scalar value head.
    #[must_use]
    pub fn head(&self) -> &Linear {
        &self.head
    }

    /// Forward pass: returns one scalar value per (row, position),
    /// shape `[B, T]` f32, autograd-live.
    ///
    /// `input_ids` is `[B, T]` u32. The head is applied independently at
    /// every position; callers mask out prompt + pad positions downstream
    /// via the completion mask before reducing.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Candle`] for any tensor shape/dtype
    /// mismatch from the underlying forward.
    pub fn forward_values(&self, input_ids: &Tensor) -> Result<Tensor, BlazenTrainError> {
        // [B, T, H] in encoder dtype.
        let hidden = self.encoder.forward_hidden_states(input_ids)?;
        // Project to scalar per position: [B, T, H] -> [B, T, 1].
        let head_in = hidden;
        let scores = self.head.forward(&head_in)?;
        let scores_f32 = scores.to_dtype(DType::F32)?;
        // Drop the trailing 1-dim: [B, T, 1] -> [B, T].
        Ok(scores_f32.squeeze(D::Minus1)?)
    }
}

/// Build a [`ValueModel`] from a freshly-loaded [`TrainableLlama`] +
/// matching `VarBuilder`s. Convenience for the common path where the
/// caller already has the encoder in hand.
///
/// `base_vb` and `lora_vb` are forwarded to [`TrainableLlama::load`];
/// `head_vb` is where the value-head weight lands. All three may point
/// at the same underlying varmap.
///
/// # Errors
///
/// Forwards any candle / model-load error.
pub fn build_value_model_from_llama(
    cfg: &LlamaConfig,
    base_vb: VarBuilder,
    lora_vb: VarBuilder,
    head_vb: VarBuilder,
    lora_cfg: &LoraConfig,
) -> Result<ValueModel, BlazenTrainError> {
    let encoder = TrainableLlama::load(base_vb, lora_vb, cfg, lora_cfg)?;
    ValueModel::new(encoder, cfg, head_vb)
}

/// One PPO training batch.
///
/// Holds `B` rows of *prompt + sampled completion* concatenated into
/// `[B, T]` token tensors, plus per-token tensors carrying the rollout
/// snapshot the loss needs: old log-probs, old value estimates, and a
/// mask indicating which positions are completion tokens. Per-row
/// scalar rewards (terminal-reward / RLHF convention) round out the
/// buffer.
///
/// All tensors live on the same device as the policy. The rollout
/// snapshot fields (`old_log_probs`, `old_values`, `rewards`,
/// `ref_log_probs`) are treated as constants by the trainer — no
/// gradient reaches them.
#[derive(Debug, Clone)]
pub struct PpoBatch {
    /// Token IDs `[B, T]`, dtype `u32`. Each row is a prompt followed by
    /// one sampled completion, right-padded to `T`.
    pub input_ids: Tensor,
    /// Completion mask `[B, T]`, dtype `f32`. `1.0` at completion
    /// positions (where the loss is reduced); `0.0` at prompt + pad
    /// positions. The trainer reduces every per-token tensor via this
    /// mask so prompt tokens never enter the loss.
    pub completion_mask: Tensor,
    /// Sampled action token IDs `[B, T]`, dtype `u32`. At every position
    /// `t`, this is the token the policy emitted at step `t` — i.e. the
    /// token that should be looked up in `log_softmax(logits_t)` to
    /// compute `log π(a_t | s_t)`. At prompt + pad positions the value
    /// is ignored (masked by `completion_mask`); callers may pad with
    /// any token id (typically the input id itself).
    pub actions: Tensor,
    /// Log-prob of the sampled token under the *behavior* policy at
    /// rollout time, `[B, T]` f32. PPO's ratio `π_pol / π_old` is
    /// computed against these.
    pub old_log_probs: Tensor,
    /// Value-function estimate at rollout time, `[B, T]` f32. Used as
    /// the GAE baseline.
    pub old_values: Tensor,
    /// Per-row scalar reward `[B]` f32. Placed at the last completion
    /// position per row (intermediate positions get `r_t = 0`). This
    /// matches the TRL terminal-reward convention for RLHF.
    pub rewards: Tensor,
    /// Reference-policy log-probs at the sampled tokens `[B, T]` f32.
    /// Only consulted when `kl_coef > 0`; callers may pass a tensor of
    /// zeros when no KL term is requested.
    pub ref_log_probs: Tensor,
}

/// PPO trainer.
///
/// Owns the policy [`TrainableLlama`], the critic [`ValueModel`], a
/// frozen reference [`TrainableLlama`] for the optional KL term, the
/// shared [`VarMap`] backing both the policy and critic trainables, and
/// the [`AdamW`] optimizer over LoRA params + value-head row.
///
/// The reference is built by deep-copying the policy varmap before any
/// gradient steps; subsequent updates to the policy varmap have no
/// effect on the reference. This mirrors DPO's reference-model pattern
/// and GRPO's [`crate::grpo::build_reference_from_policy`].
pub struct PpoTrainer {
    config: TrainConfig,
    ppo_cfg: PpoCoreState,
    /// Single varmap holding the policy's LoRA params **and** the
    /// critic's LoRA + value-head params. AdamW captures both subsets
    /// at construction time; downstream LR scheduling and gradient
    /// clipping touch the same param list.
    varmap: VarMap,
    policy: TrainableLlama,
    critic: ValueModel,
    reference: TrainableLlama,
    optimizer: AdamW,
    device: Device,
    progress: Option<Arc<dyn TrainingProgress>>,
    global_step: usize,
    lr_scheduler: Box<dyn Fn(usize) -> f64 + Send + Sync>,
    /// Captured at construction from `config.max_steps`. Reserved for the
    /// phase-2 rollout driver (`run_batches` / `run`) so it can emit the
    /// `TrainingEvent::Started { total_steps }` header. Phase-1 callers
    /// drive `step` directly and don't consult this field.
    #[allow(dead_code)]
    total_steps: usize,
    accum_counter: usize,
}

/// PPO-specific state extracted from [`PpoConfig`].
struct PpoCoreState {
    clip_epsilon: f32,
    value_coef: f32,
    entropy_coef: f32,
    gae_lambda: f32,
    gamma: f32,
    kl_coef: f32,
}

impl PpoTrainer {
    /// Construct a PPO trainer.
    ///
    /// The caller is responsible for building the policy + critic
    /// `TrainableLlama` / `ValueModel` against `varmap` (LoRA params plus
    /// the value head must already be registered before AdamW captures
    /// them), and for handing in a frozen reference `TrainableLlama`
    /// whose weights are a snapshot of the policy.
    ///
    /// Simplest construction path:
    ///
    /// 1. Build the policy against `varmap`.
    /// 2. Build the critic against the same `varmap` with a different
    ///    prefix so the value head's `weight` row doesn't collide with
    ///    any reward head from the same varmap.
    /// 3. [`crate::grpo::build_reference_from_policy`] for the reference.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::InvalidConfig`] if `clip_epsilon <= 0`,
    /// `lora.rank == 0`, or `core.max_steps == 0`. Returns
    /// [`BlazenTrainError::Optimizer`] if `AdamW::new` rejects the trainable
    /// param set.
    pub fn new(
        cfg: PpoConfig,
        varmap: VarMap,
        device: Device,
        policy: TrainableLlama,
        critic: ValueModel,
        reference: TrainableLlama,
        progress: Option<Arc<dyn TrainingProgress>>,
    ) -> Result<Self, BlazenTrainError> {
        if cfg.clip_epsilon <= 0.0 {
            return Err(BlazenTrainError::InvalidConfig(
                "ppo: clip_epsilon must be > 0".to_string(),
            ));
        }
        if cfg.lora.rank == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "ppo: lora.rank must be > 0".to_string(),
            ));
        }
        if cfg.core.max_steps == 0 {
            return Err(BlazenTrainError::InvalidConfig(
                "ppo: core.max_steps must be > 0".to_string(),
            ));
        }
        if cfg.gae_lambda < 0.0 || cfg.gae_lambda > 1.0 {
            return Err(BlazenTrainError::InvalidConfig(format!(
                "ppo: gae_lambda must be in [0, 1], got {}",
                cfg.gae_lambda
            )));
        }
        if cfg.gamma < 0.0 || cfg.gamma > 1.0 {
            return Err(BlazenTrainError::InvalidConfig(format!(
                "ppo: gamma must be in [0, 1], got {}",
                cfg.gamma
            )));
        }

        let PpoConfig {
            core,
            lora,
            clip_epsilon,
            value_coef,
            entropy_coef,
            gae_lambda,
            gamma,
            kl_coef,
            reward_model_repo: _,
            reward_model_revision: _,
            value_model_init: _,
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

        let trainable = collect_ppo_trainable_params(&varmap, &synthesized.lora);
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
            ppo_cfg: PpoCoreState {
                clip_epsilon,
                value_coef,
                entropy_coef,
                gae_lambda,
                gamma,
                kl_coef,
            },
            varmap,
            policy,
            critic,
            reference,
            optimizer,
            device,
            progress,
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

    /// Borrow the shared varmap (policy + critic params).
    #[must_use]
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Borrow the device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Borrow the policy model.
    #[must_use]
    pub fn policy(&self) -> &TrainableLlama {
        &self.policy
    }

    /// Borrow the critic.
    #[must_use]
    pub fn critic(&self) -> &ValueModel {
        &self.critic
    }

    /// Borrow the frozen reference policy.
    #[must_use]
    pub fn reference(&self) -> &TrainableLlama {
        &self.reference
    }

    /// Optional progress sink (mostly relevant once the rollout loop
    /// lands in phase 2; phase-1 callers can drop it).
    pub fn set_progress(&mut self, progress: Option<Arc<dyn TrainingProgress>>) {
        self.progress = progress;
    }

    /// Current 0-indexed optimizer step counter.
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Current learning rate (from the LR scheduler at the active step).
    #[must_use]
    pub fn current_learning_rate(&self) -> f64 {
        (self.lr_scheduler)(self.global_step)
    }

    /// Compute GAE advantages + returns from a host-side rollout.
    ///
    /// All four inputs are flat row-major slices of length `B * T`:
    /// `rewards_per_token[r, t]`, `values[r, t]`, `mask[r, t]`. The
    /// terminal value `V(s_{T}) = 0` per row. Padding / prompt positions
    /// are skipped via `mask` (they get advantage / return = 0).
    ///
    /// The recurrence is the standard GAE-λ:
    ///
    /// ```text
    /// δ_t      = r_t + γ * V(s_{t+1}) - V(s_t)
    /// adv_t    = δ_t + γλ * adv_{t+1} * mask_{t+1}
    /// return_t = adv_t + V(s_t)
    /// ```
    ///
    /// `mask_{t+1}` zeros the bootstrap across episode boundaries — at
    /// the last completion position (or before prompt-only padding),
    /// the bootstrap `adv_{t+1}` is gated off.
    ///
    /// Exposed publicly so tests + the phase-2 rollout sampler can
    /// verify the recurrence against a hand-computed reference.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Forward`] if the input lengths are
    /// inconsistent with `(batch, seq_len)`.
    pub fn compute_gae(
        rewards_per_token: &[f32],
        values: &[f32],
        mask: &[f32],
        batch: usize,
        seq_len: usize,
        gamma: f32,
        lambda: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), BlazenTrainError> {
        let expected = batch * seq_len;
        if rewards_per_token.len() != expected || values.len() != expected || mask.len() != expected
        {
            return Err(BlazenTrainError::Forward(format!(
                "ppo gae: expected {expected} entries per tensor, got rewards={} values={} mask={}",
                rewards_per_token.len(),
                values.len(),
                mask.len()
            )));
        }

        let mut advantages = vec![0.0_f32; expected];
        let mut returns = vec![0.0_f32; expected];

        for row in 0..batch {
            let base = row * seq_len;
            let mut next_adv: f32 = 0.0;
            // Walk backwards. The bootstrap V(s_{t+1}) reads the value at
            // t+1 if it's a real completion position, else 0 (terminal).
            for t in (0..seq_len).rev() {
                let m = mask[base + t];
                if m < 0.5 {
                    advantages[base + t] = 0.0;
                    returns[base + t] = 0.0;
                    // A masked position never bootstraps into earlier
                    // positions: reset the carrier.
                    next_adv = 0.0;
                    continue;
                }
                let next_value = if t + 1 < seq_len && mask[base + t + 1] > 0.5 {
                    values[base + t + 1]
                } else {
                    0.0
                };
                let next_adv_gated = if t + 1 < seq_len && mask[base + t + 1] > 0.5 {
                    next_adv
                } else {
                    0.0
                };
                let delta = rewards_per_token[base + t] + gamma * next_value - values[base + t];
                let adv = delta + gamma * lambda * next_adv_gated;
                advantages[base + t] = adv;
                returns[base + t] = adv + values[base + t];
                next_adv = adv;
            }
        }

        Ok((advantages, returns))
    }

    /// Run one PPO optimizer step over a pre-rolled batch.
    ///
    /// Returns the scalar loss before gradient accumulation scaling. Loss
    /// formula and masking conventions are documented at the module level.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Forward`] for shape mismatches; forwards
    /// any candle / optimizer error.
    // Why: PPO's loss is a sum of four named terms (clipped surrogate,
    // value regression, entropy, optional KL) plus the host-side GAE
    // setup. Splitting it across helpers would scatter the autograd
    // graph and make the masking convention harder to audit; keep the
    // single-pass implementation and silence the line-count lint.
    #[allow(clippy::too_many_lines)]
    pub fn step(&mut self, batch: &PpoBatch) -> Result<f32, BlazenTrainError> {
        let lr = (self.lr_scheduler)(self.global_step);
        self.optimizer.set_learning_rate(lr);

        let (b, t) = batch.input_ids.dims2()?;
        check_shape("completion_mask", &batch.completion_mask, b, t)?;
        check_shape("actions", &batch.actions, b, t)?;
        check_shape("old_log_probs", &batch.old_log_probs, b, t)?;
        check_shape("old_values", &batch.old_values, b, t)?;
        check_shape("ref_log_probs", &batch.ref_log_probs, b, t)?;
        let (br,) = batch.rewards.dims1().map(|n| (n,))?;
        if br != b {
            return Err(BlazenTrainError::Forward(format!(
                "ppo step: rewards has length {br}, expected batch {b}"
            )));
        }

        // ----- Host-side GAE (no autograd) ---------------------------
        // Distribute the per-row terminal reward to the last completion
        // position; everything else is r_t = 0.
        let mask_f32: Vec<f32> = batch
            .completion_mask
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let old_values_host: Vec<f32> = batch
            .old_values
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let rewards_host: Vec<f32> = batch.rewards.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        let mut rewards_per_token = vec![0.0_f32; b * t];
        for (row, &reward) in rewards_host.iter().enumerate().take(b) {
            // Find the last position with mask == 1 (terminal reward placement).
            let row_base = row * t;
            let last = (0..t).rev().find(|&col| mask_f32[row_base + col] > 0.5);
            if let Some(col) = last {
                rewards_per_token[row_base + col] = reward;
            }
        }

        let (advantages_host, returns_host) = Self::compute_gae(
            &rewards_per_token,
            &old_values_host,
            &mask_f32,
            b,
            t,
            self.ppo_cfg.gamma,
            self.ppo_cfg.gae_lambda,
        )?;

        // Normalize advantages over the unmasked entries (standard PPO
        // trick; stabilizes training when rewards have arbitrary scale).
        let advantages_host = normalize_advantages(&advantages_host, &mask_f32);

        let advantages_t = Tensor::from_vec(advantages_host, (b, t), &self.device)?.detach();
        let returns_t = Tensor::from_vec(returns_host, (b, t), &self.device)?.detach();
        let mask_t = Tensor::from_vec(mask_f32.clone(), (b, t), &self.device)?.detach();
        let mask_sum = mask_t.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        if mask_sum < 0.5 {
            return Err(BlazenTrainError::Forward(
                "ppo step: completion_mask is all zeros — no positions to reduce over".to_string(),
            ));
        }
        let mask_sum_t = Tensor::new(mask_sum, &self.device)?.detach();

        // ----- Policy forward (autograd-live) -----------------------
        let policy_logits = self.policy.forward(&batch.input_ids)?;
        let log_probs_all =
            candle_nn::ops::log_softmax(&policy_logits.to_dtype(DType::F32)?, D::Minus1)?;
        // Gather log π(a_t | s_t) at the sampled action ids.
        let actions_u32 = batch.actions.to_dtype(DType::U32)?;
        let new_log_probs = log_probs_all
            .gather(&actions_u32.unsqueeze(2)?, 2)?
            .squeeze(2)?; // [B, T] f32

        // ----- Critic forward (autograd-live) -----------------------
        let new_values = self.critic.forward_values(&batch.input_ids)?; // [B, T]

        // ----- PPO surrogate ----------------------------------------
        let ratio = (&new_log_probs - &batch.old_log_probs)?.exp()?; // [B, T]
        let surr1 = (&ratio * &advantages_t)?;
        let lo = f64::from(1.0_f32 - self.ppo_cfg.clip_epsilon);
        let hi = f64::from(1.0_f32 + self.ppo_cfg.clip_epsilon);
        let ratio_clamped = ratio.clamp(lo, hi)?;
        let surr2 = (&ratio_clamped * &advantages_t)?;
        // min(surr1, surr2) per element. Use .minimum() — candle supports it.
        let surr_min = surr1.minimum(&surr2)?;
        let surr_masked = (&surr_min * &mask_t)?;
        let pg_loss = surr_masked.sum_all()?.neg()?.broadcast_div(&mask_sum_t)?;

        // ----- Value-function regression ----------------------------
        let value_diff = (&new_values - &returns_t)?;
        let vf_per = value_diff.sqr()?;
        let vf_masked = (&vf_per * &mask_t)?;
        let vf_loss = vf_masked.sum_all()?.broadcast_div(&mask_sum_t)?;

        // ----- Entropy bonus ----------------------------------------
        // H = -Σ_a p_a * log p_a, computed per position.
        let probs = log_probs_all.exp()?;
        let neg_p_logp = (&probs * &log_probs_all)?.neg()?;
        let entropy_per_pos = neg_p_logp.sum(D::Minus1)?; // [B, T]
        let entropy_masked = (&entropy_per_pos * &mask_t)?;
        let ent_mean = entropy_masked.sum_all()?.broadcast_div(&mask_sum_t)?;
        // We want to *maximize* entropy → subtract entropy_coef * H.
        let ent_term = (ent_mean * f64::from(-self.ppo_cfg.entropy_coef))?;

        // ----- Optional KL-to-reference -----------------------------
        let kl_term = if self.ppo_cfg.kl_coef.abs() > f32::EPSILON {
            // k1 estimator: KL ≈ log π_pol(a) - log π_ref(a) per sampled token.
            let kl_per = (&new_log_probs - &batch.ref_log_probs)?;
            let kl_masked = (&kl_per * &mask_t)?;
            let kl_mean = kl_masked.sum_all()?.broadcast_div(&mask_sum_t)?;
            (kl_mean * f64::from(self.ppo_cfg.kl_coef))?
        } else {
            Tensor::zeros((), DType::F32, &self.device)?
        };

        // ----- Total loss -------------------------------------------
        let vf_scaled = (vf_loss * f64::from(self.ppo_cfg.value_coef))?;
        let loss = ((pg_loss + vf_scaled)? + ent_term)?;
        let loss = (loss + kl_term)?;

        let accum = self.config.gradient_accumulation_steps.max(1);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / accum as f64;
        let scaled_loss = (&loss * scale)?;
        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        let mut grads = scaled_loss.backward()?;

        self.accum_counter += 1;
        if self.accum_counter >= accum {
            if let Some(max) = self.config.optim.gradient_clip {
                let vars = collect_ppo_trainable_params(&self.varmap, &self.config.lora);
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
}

/// Mean / unit-variance normalize advantages, but only over positions
/// where the mask is set. Masked entries stay at their original value
/// (zero, since GAE returned zero for them) — the normalizer is computed
/// over the active subset.
fn normalize_advantages(advantages: &[f32], mask: &[f32]) -> Vec<f32> {
    let mut count = 0_u64;
    let mut sum = 0.0_f64;
    for (&a, &m) in advantages.iter().zip(mask.iter()) {
        if m > 0.5 {
            sum += f64::from(a);
            count += 1;
        }
    }
    if count == 0 {
        return advantages.to_vec();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = count as f64;
    let mean = sum / n;
    let mut var = 0.0_f64;
    for (&a, &m) in advantages.iter().zip(mask.iter()) {
        if m > 0.5 {
            let d = f64::from(a) - mean;
            var += d * d;
        }
    }
    let std = (var / n).sqrt();
    let denom = std + 1e-8;
    advantages
        .iter()
        .zip(mask.iter())
        .map(|(&a, &m)| {
            if m > 0.5 {
                #[allow(clippy::cast_possible_truncation)]
                let z = ((f64::from(a) - mean) / denom) as f32;
                z
            } else {
                a
            }
        })
        .collect()
}

fn check_shape(name: &str, t: &Tensor, b: usize, len: usize) -> Result<(), BlazenTrainError> {
    let (tb, tt) = t.dims2()?;
    if tb != b || tt != len {
        return Err(BlazenTrainError::Forward(format!(
            "ppo step: {name} has shape [{tb}, {tt}], expected [{b}, {len}]"
        )));
    }
    Ok(())
}

/// Filter the varmap down to the params PPO should update: LoRA A/B
/// rows (policy + critic encoder) plus the value-head row. Mirrors
/// [`crate::reward::collect_reward_trainable_params`] in spirit but
/// targets the value-head key.
fn collect_ppo_trainable_params(varmap: &VarMap, lora_cfg: &LoraConfig) -> Vec<candle_core::Var> {
    let target_refs: Vec<&str> = lora_cfg.target_modules.iter().map(String::as_str).collect();
    let mut params = freeze_base_params(varmap, &target_refs);

    let guard = varmap
        .data()
        .lock()
        .expect("varmap mutex poisoned by another thread");
    for (name, var) in guard.iter() {
        if name.ends_with("value_head.weight") {
            params.push(var.clone());
        }
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::llama::Config as LlamaCfg;
    use crate::config::{
        MixedPrecision, OptimConfig, SchedulerConfig, SchedulerKind, TrainCoreConfig,
        ValueModelInit,
    };
    use crate::grpo::build_reference_from_policy;
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

    fn tiny_ppo_config(output_dir: PathBuf, max_steps: usize) -> PpoConfig {
        PpoConfig {
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
            clip_epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            gae_lambda: 0.95,
            gamma: 1.0,
            kl_coef: 0.0,
            reward_model_repo: None,
            reward_model_revision: None,
            value_model_init: ValueModelInit::FromPolicy,
        }
    }

    fn build_tiny_value_model(
        varmap: &VarMap,
        device: &Device,
        lora_cfg: &LoraConfig,
    ) -> ValueModel {
        let cfg = tiny_llama_config();
        let base_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let lora_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let head_vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        build_value_model_from_llama(&cfg, base_vb, lora_vb, head_vb, lora_cfg)
            .expect("build value model")
    }

    fn build_tiny_ppo_models(
        device: &Device,
        lora_cfg: &LoraConfig,
    ) -> (VarMap, TrainableLlama, ValueModel, TrainableLlama) {
        let cfg = tiny_llama_config();
        let varmap = VarMap::new();
        // Policy: lives at the root of the varmap (no prefix).
        let pol_base = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let pol_lora = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let policy = TrainableLlama::load(pol_base, pol_lora, &cfg, lora_cfg).expect("policy load");
        // Critic: nest under a "critic" prefix so its base + LoRA params
        // don't collide with the policy's. The value head sits at
        // "critic.value_head.weight".
        let crit_base = VarBuilder::from_varmap(&varmap, DType::F32, device).pp("critic");
        let crit_lora = VarBuilder::from_varmap(&varmap, DType::F32, device).pp("critic");
        let crit_head = VarBuilder::from_varmap(&varmap, DType::F32, device).pp("critic");
        let critic = build_value_model_from_llama(&cfg, crit_base, crit_lora, crit_head, lora_cfg)
            .expect("critic build");
        let (reference, _ref_vm) =
            build_reference_from_policy(&varmap, &cfg, DType::F32, device).expect("ref build");
        (varmap, policy, critic, reference)
    }

    fn make_ppo_batch(
        device: &Device,
        batch: usize,
        seq_len: usize,
        prompt_len: usize,
    ) -> PpoBatch {
        let total = batch * seq_len;
        let ids: Vec<u32> = (0..total)
            .map(|i| u32::try_from(i % 128).unwrap())
            .collect();
        let actions: Vec<u32> = ids.clone(); // any valid token id; masked at prompt positions
        // mask = 1 at completion positions only
        let mut mask = vec![0.0_f32; total];
        for r in 0..batch {
            for t in prompt_len..seq_len {
                mask[r * seq_len + t] = 1.0;
            }
        }
        let old_log_probs = vec![-2.0_f32; total];
        let old_values = vec![0.1_f32; total];
        let ref_log_probs = vec![-2.0_f32; total];
        let rewards: Vec<f32> = (0..batch)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let scaled = 0.25 * (i as f32 + 1.0);
                scaled
            })
            .collect();
        let shape = (batch, seq_len);
        PpoBatch {
            input_ids: Tensor::from_vec(ids, shape, device).unwrap(),
            completion_mask: Tensor::from_vec(mask, shape, device).unwrap(),
            actions: Tensor::from_vec(actions, shape, device).unwrap(),
            old_log_probs: Tensor::from_vec(old_log_probs, shape, device).unwrap(),
            old_values: Tensor::from_vec(old_values, shape, device).unwrap(),
            rewards: Tensor::from_vec(rewards, (batch,), device).unwrap(),
            ref_log_probs: Tensor::from_vec(ref_log_probs, shape, device).unwrap(),
        }
    }

    #[test]
    fn value_forward_shape_correct() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let cfg = tiny_ppo_config(PathBuf::from("/tmp-unused"), 1);
        let model = build_tiny_value_model(&varmap, &device, &cfg.lora);

        let b = 1;
        let t = 4;
        let ids: Vec<u32> = (0..(b * t)).map(|i| u32::try_from(i).unwrap()).collect();
        let input = Tensor::from_vec(ids, (b, t), &device).unwrap();
        let values = model.forward_values(&input).expect("forward values");
        assert_eq!(values.dims(), &[b, t]);
    }

    #[test]
    fn value_head_registered_in_varmap() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let cfg = tiny_ppo_config(PathBuf::from("/tmp-unused"), 1);
        let _model = build_tiny_value_model(&varmap, &device, &cfg.lora);

        let guard = varmap.data().lock().expect("varmap mutex");
        let has_head = guard
            .keys()
            .any(|k| k.ends_with("value_head.weight") || k == VALUE_HEAD_WEIGHT_KEY);
        assert!(
            has_head,
            "expected value_head.weight in varmap, keys: {:?}",
            guard.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn gae_advantages_match_reference_formula() {
        // Single row, seq_len = 4, all positions are completion tokens.
        // values = [0, 0, 0, 0]; rewards_per_token = [1, 0, 0, 1].
        // gamma = 1.0, lambda = 0.95.
        //
        // Walking backwards with V = 0 and γ = 1:
        //   δ_3 = r_3 + γ * V_4 - V_3 = 1 + 0 - 0 = 1
        //   adv_3 = δ_3 = 1
        //   δ_2 = 0 + 0 - 0 = 0
        //   adv_2 = 0 + 1*0.95 * adv_3 = 0.95
        //   δ_1 = 0
        //   adv_1 = 0 + 0.95 * adv_2 = 0.95^2 = 0.9025
        //   δ_0 = 1
        //   adv_0 = 1 + 0.95 * adv_1 = 1 + 0.95 * 0.9025 = 1 + 0.857375 = 1.857375
        //
        // Returns = adv + V = adv (since V = 0).
        let rewards = [1.0_f32, 0.0, 0.0, 1.0];
        let values = [0.0_f32, 0.0, 0.0, 0.0];
        let mask = [1.0_f32, 1.0, 1.0, 1.0];
        let (advs, rets) =
            PpoTrainer::compute_gae(&rewards, &values, &mask, 1, 4, 1.0, 0.95).unwrap();

        let expected_advs = [1.857_375_f32, 0.902_5, 0.95, 1.0];
        for (i, (got, want)) in advs.iter().zip(expected_advs.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "advantage[{i}] = {got}, expected {want}"
            );
        }
        // Returns == advantages when V = 0.
        for (i, (got, want)) in rets.iter().zip(expected_advs.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "return[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn gae_masked_positions_get_zero() {
        // 1 row, seq_len = 5, last 2 positions are padding.
        let rewards = [0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let values = [0.5_f32, 0.5, 0.5, 0.0, 0.0];
        let mask = [1.0_f32, 1.0, 1.0, 0.0, 0.0];
        let (advs, rets) =
            PpoTrainer::compute_gae(&rewards, &values, &mask, 1, 5, 1.0, 1.0).unwrap();
        assert!(advs[3].abs() < f32::EPSILON, "advs[3] = {}", advs[3]);
        assert!(advs[4].abs() < f32::EPSILON, "advs[4] = {}", advs[4]);
        assert!(rets[3].abs() < f32::EPSILON, "rets[3] = {}", rets[3]);
        assert!(rets[4].abs() < f32::EPSILON, "rets[4] = {}", rets[4]);
        // At t=2 (last completion), δ = 1 + 0 (V_{t+1} masked) - 0.5 = 0.5; adv = 0.5.
        assert!((advs[2] - 0.5).abs() < 1e-6, "advs[2] = {}", advs[2]);
    }

    #[test]
    fn ppo_clip_epsilon_caps_ratio() {
        // Build a 1×1 toy batch by hand: old_log_prob = log(0.1) so the
        // ratio when new_log_prob = log(1.0) is 10. With clip_epsilon =
        // 0.2 the surrogate uses clamp(10, 0.8, 1.2) = 1.2, so for a
        // positive advantage we expect the clipped branch to win and the
        // contribution = -min(10*A, 1.2*A) = -1.2*A (since A > 0, min
        // picks 1.2*A).
        //
        // This is a pure math check on the clip semantics; we exercise
        // it through the tensor primitives PpoTrainer::step uses.
        let device = Device::Cpu;
        let ratio = Tensor::from_vec(vec![10.0_f32], (1,), &device).unwrap();
        let adv = Tensor::from_vec(vec![1.0_f32], (1,), &device).unwrap();
        let eps = 0.2_f32;
        let lo = f64::from(1.0 - eps);
        let hi = f64::from(1.0 + eps);
        let r_clamped = ratio.clamp(lo, hi).unwrap();
        let surr1 = (&ratio * &adv).unwrap();
        let surr2 = (&r_clamped * &adv).unwrap();
        let chosen = surr1.minimum(&surr2).unwrap();
        let v = chosen.to_vec1::<f32>().unwrap()[0];
        assert!(
            (v - 1.2).abs() < 1e-5,
            "expected clipped surrogate to be 1.2, got {v}"
        );
    }

    #[test]
    fn ppo_loss_negative_advantage_pushes_policy_away() {
        // Toy 1×1 cell: with a negative advantage, the gradient of the
        // un-clipped surrogate w.r.t. log π is `-ratio * advantage`,
        // which is positive (since adv < 0 and ratio > 0). That means
        // taking a gradient *descent* step on the loss reduces log π —
        // i.e. pushes the policy *away* from the action. We verify this
        // numerically: compute the gradient sign of pg_loss w.r.t. the
        // policy log-prob; it must be opposite the sign of the
        // advantage (positive grad ↔ negative adv).
        let device = Device::Cpu;
        let new_lp =
            candle_core::Var::from_tensor(&Tensor::from_vec(vec![0.0_f32], (1,), &device).unwrap())
                .unwrap();
        let old_lp = Tensor::from_vec(vec![0.0_f32], (1,), &device).unwrap();
        let adv = Tensor::from_vec(vec![-1.0_f32], (1,), &device).unwrap();
        let ratio = (new_lp.as_tensor() - &old_lp).unwrap().exp().unwrap();
        let surr = (&ratio * &adv).unwrap();
        let loss = surr.neg().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g = grads
            .get(new_lp.as_tensor())
            .expect("grad for new_lp")
            .to_vec1::<f32>()
            .unwrap()[0];
        // d/d(new_lp) [ -ratio * adv ] = -ratio * adv (since d ratio/d lp = ratio).
        // With ratio = 1 and adv = -1, expected gradient = -(1 * -1) = 1.
        assert!(
            g > 0.0,
            "negative-advantage gradient should push log π down (positive grad), got {g}"
        );
    }

    #[test]
    fn ppo_one_step_does_not_panic() {
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_ppo_config(tmp.path().to_path_buf(), 1);
        let (varmap, policy, critic, reference) = build_tiny_ppo_models(&device, &cfg.lora);

        let mut trainer =
            PpoTrainer::new(cfg, varmap, device.clone(), policy, critic, reference, None)
                .expect("PpoTrainer::new");

        let batch = make_ppo_batch(&device, 2, 6, 2);
        let loss = trainer.step(&batch).expect("ppo step");
        assert!(loss.is_finite(), "PPO loss not finite: {loss}");
        assert_eq!(trainer.global_step(), 1);
    }

    #[test]
    fn ppo_loss_finite_across_many_steps() {
        let device = Device::Cpu;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tiny_ppo_config(tmp.path().to_path_buf(), 5);
        let (varmap, policy, critic, reference) = build_tiny_ppo_models(&device, &cfg.lora);

        let mut trainer =
            PpoTrainer::new(cfg, varmap, device.clone(), policy, critic, reference, None)
                .expect("PpoTrainer::new");

        let batch = make_ppo_batch(&device, 2, 6, 2);
        for step in 0..5 {
            let loss = trainer.step(&batch).expect("step");
            assert!(
                loss.is_finite(),
                "PPO loss at step {step} was NaN/inf: {loss}"
            );
        }
        assert_eq!(trainer.global_step(), 5);
    }
}
