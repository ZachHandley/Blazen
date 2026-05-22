//! `AllReduceGrpoTrainer` — wraps a [`crate::GrpoTrainer`] with
//! ring-AllReduce gradient averaging across an N-worker ring before each
//! optimizer step.
//!
//! Phase 2 of the distributed-training rollout. Mirrors the structure of
//! [`super::trainer::AllReduceTrainer`] (which wraps [`crate::Trainer`])
//! but splices AllReduce into the GRPO-specific loss kernel: per-group
//! advantage normalization, per-row policy log-probs, reference KL term,
//! and a combined policy-gradient + KL loss.

use std::sync::Arc;

use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};

use crate::error::BlazenTrainError;
use crate::grpo::{GrpoBatch, GrpoTrainer, mean_label_log_probs};
use crate::lora::freeze_base_params;

use super::allreduce::{average_by, ring_all_reduce};
use super::ring::RingTopology;
use super::transport::RingTransport;

/// Distributed-training wrapper around an inner [`GrpoTrainer`].
///
/// Owns:
/// - the inner single-device GRPO trainer (policy + reference + reward
///   model + AdamW optimizer over the policy's LoRA params),
/// - the ring [`RingTopology`] for this worker,
/// - a [`RingTransport`] to push / pull gradient chunks.
///
/// Each `step` computes the local GRPO loss (group-relative advantages,
/// per-row policy/reference log-probs, KL regularizer) via the inner
/// trainer's policy + reference forwards, runs [`ring_all_reduce`] on
/// every trainable parameter's gradient, divides by `world_size` to
/// recover the mean, then performs the optimizer step on the averaged
/// gradients.
///
/// Identical to [`GrpoTrainer::step`] when `topology.world_size == 1`
/// (the AllReduce kernel short-circuits, and the wrapper delegates to
/// the inner step directly to avoid duplicating its accumulator state).
pub struct AllReduceGrpoTrainer {
    inner: GrpoTrainer,
    topology: RingTopology,
    transport: Arc<dyn RingTransport>,
}

impl AllReduceGrpoTrainer {
    /// Wrap an existing [`GrpoTrainer`].
    ///
    /// The inner trainer must already have its policy + reference loaded —
    /// the wrapper does not call any model-load logic itself so callers
    /// can share base-weight loading across the ring.
    #[must_use]
    pub fn new(
        inner: GrpoTrainer,
        topology: RingTopology,
        transport: Arc<dyn RingTransport>,
    ) -> Self {
        Self {
            inner,
            topology,
            transport,
        }
    }

    /// Borrow the underlying single-device trainer (read-only access for
    /// callers that need its config / progress hooks).
    #[must_use]
    pub fn inner(&self) -> &GrpoTrainer {
        &self.inner
    }

    /// Mutably borrow the underlying trainer (e.g. to attach a reward
    /// model, swap the batch source, or load weights before the first
    /// step).
    pub fn inner_mut(&mut self) -> &mut GrpoTrainer {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner single-device trainer.
    ///
    /// Useful at end-of-training when callers need to reclaim the inner
    /// trainer's [`candle_nn::VarMap`] for adapter export. Dropping the
    /// wrapper releases the [`RingTransport`] handle and (for the gRPC
    /// transport) closes the next-peer client connection.
    #[must_use]
    pub fn into_inner(self) -> GrpoTrainer {
        self.inner
    }

    /// Borrow this worker's ring topology.
    #[must_use]
    pub fn topology(&self) -> &RingTopology {
        &self.topology
    }

    /// Run one distributed GRPO training step.
    ///
    /// Mirrors [`GrpoTrainer::step`] but splices ring-AllReduce in
    /// between `loss.backward()` and the optimizer step:
    ///
    /// 1. Compute group-relative advantages on the local batch.
    /// 2. Policy + reference forwards; per-row mean log-probs of the
    ///    labeled (completion) tokens.
    /// 3. Combined loss: `-mean(adv * lp_policy) + beta * mean(lp_ref - lp_policy)`.
    /// 4. `scaled_loss.backward()` to populate the local `GradStore`.
    /// 5. For each trainable LoRA param, AllReduce the per-param gradient
    ///    across the ring and divide by `world_size`.
    /// 6. Optimizer step on the averaged gradients.
    ///
    /// Identical to [`GrpoTrainer::step`] when `topology.world_size == 1`
    /// (short-circuits to the inner sync step).
    ///
    /// # Errors
    /// Propagates [`BlazenTrainError`] from forward, backward, transport,
    /// or optimizer paths.
    pub async fn step(&mut self, batch: &GrpoBatch) -> Result<f32, BlazenTrainError> {
        // World size 1 → delegate to the inner trainer directly.
        if !self.topology.is_distributed() {
            return self.inner.step(batch);
        }

        // Mirror GrpoTrainer::step's body up to (but not through) the
        // optimizer step. The AllReduce splices in between backward and
        // the optimizer kick.
        let lr = self.inner.lr_for_step(self.inner.global_step());
        self.inner.set_optimizer_lr(lr);

        // Advantages: detached (no autograd through the reward signal).
        let advantages = GrpoTrainer::group_relative_advantages(
            &batch.rewards,
            &batch.group_ids,
            self.inner.advantage_epsilon(),
        )?
        .detach();

        // Policy forward — WITH autograd. Reference forward is detached.
        let policy_logits = self.inner.policy().forward(&batch.input_ids)?;
        let ref_logits = self.inner.reference().forward(&batch.input_ids)?.detach();

        // Per-row mean log-prob of the labeled (completion) tokens.
        let lp_policy = mean_label_log_probs(&policy_logits, &batch.labels)?;
        let lp_ref = mean_label_log_probs(&ref_logits, &batch.labels)?.detach();

        // Policy-gradient term: -mean( adv * lp_policy ).
        let pg = (&advantages * &lp_policy)?.mean_all()?.neg()?;

        // KL term (k1 approx): mean( lp_ref - lp_policy ).
        let kl_per_row = (&lp_ref - &lp_policy)?;
        let kl_term = kl_per_row.mean_all()?;
        let kl_scaled = (&kl_term * f64::from(self.inner.beta_coeff()))?;

        let loss = (pg + kl_scaled)?;
        let accum = self.inner.config().gradient_accumulation_steps.max(1);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / accum as f64;
        let scaled_loss = (&loss * scale)?;
        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        let mut grads: GradStore = scaled_loss.backward()?;

        let target_refs: Vec<&str> = self
            .inner
            .config()
            .lora
            .target_modules
            .iter()
            .map(String::as_str)
            .collect();
        let vars: Vec<Var> = freeze_base_params(self.inner.varmap(), &target_refs);

        // AllReduce each trainable gradient. Tag = enumerate index so
        // every param has a distinct (step, chunk_id) namespace inside
        // the same optimizer step.
        for (tag, var) in vars.iter().enumerate() {
            let Some(grad) = grads.remove(var.as_tensor()) else {
                // No gradient flowed through this param this batch.
                // Participate in the ring with a zero tensor so peers
                // don't deadlock on a missing chunk.
                let zero = Tensor::zeros_like(var.as_tensor())?;
                let mut z = zero;
                ring_all_reduce(
                    &mut z,
                    &self.topology,
                    self.transport.as_ref(),
                    u32::try_from(tag).unwrap_or(0),
                )
                .await?;
                average_by(&mut z, self.topology.world_size)?;
                grads.insert(var.as_tensor(), z);
                continue;
            };
            let mut g = grad;
            ring_all_reduce(
                &mut g,
                &self.topology,
                self.transport.as_ref(),
                u32::try_from(tag).unwrap_or(0),
            )
            .await?;
            average_by(&mut g, self.topology.world_size)?;
            grads.insert(var.as_tensor(), g);
        }

        let did_step = self.inner.maybe_step_optimizer(&mut grads, &vars)?;
        if did_step {
            self.inner.bump_global_step();
        }

        Ok(loss_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::llama::Config as LlamaCfg;
    use crate::config::{
        GrpoConfig, LoraConfig, MixedPrecision, OptimConfig, SchedulerConfig, SchedulerKind,
        TrainCoreConfig,
    };
    use crate::distributed::transport::InMemoryRingTransport;
    use crate::grpo::build_reference_from_policy;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
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
                target_modules: vec!["q_proj".to_string()],
            },
            group_size: 4,
            beta: 0.04,
            advantage_epsilon: 1e-8,
            sampling_temperature: 1.0,
            reward_model_repo: None,
            reward_model_revision: None,
        }
    }

    #[tokio::test]
    async fn all_reduce_grpo_trainer_constructs_with_world_2_inmemory() {
        let ring = InMemoryRingTransport::ring(2);
        let device = Device::Cpu;
        let cfg = tiny_grpo_config(PathBuf::from("./_pr_d_grpo_test_out"), 1);
        let llama_cfg = tiny_llama_config();
        let policy_vm = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&policy_vm, DType::F32, &device);
        let lora_vb = VarBuilder::from_varmap(&policy_vm, DType::F32, &device);
        let policy =
            crate::arch::llama::TrainableLlama::load(base_vb, lora_vb, &llama_cfg, &cfg.lora)
                .expect("policy load");
        let (reference, _ref_vm) =
            build_reference_from_policy(&policy_vm, &llama_cfg, DType::F32, &device)
                .expect("reference build");
        let trainer =
            GrpoTrainer::new(cfg, policy_vm, device, policy, reference, None).expect("trainer");
        let topo =
            RingTopology::new(0, vec!["a:1".to_string(), "b:1".to_string()]).expect("topology");
        let dist = AllReduceGrpoTrainer::new(trainer, topo, ring[0].clone());
        assert_eq!(dist.topology().world_size, 2);
        assert_eq!(dist.topology().rank, 0);
        assert!(dist.topology().is_distributed());
    }
}
