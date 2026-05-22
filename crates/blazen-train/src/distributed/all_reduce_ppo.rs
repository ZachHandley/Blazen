//! `AllReducePpoTrainer` — wraps a [`crate::ppo::PpoTrainer`] with ring-
//! AllReduce gradient averaging across an N-worker ring before each
//! optimizer step.
//!
//! Mirrors [`super::trainer::AllReduceTrainer`] one-for-one but targets
//! PPO's actor + critic + value-head trainable set instead of the SFT /
//! LoRA trainable set. The reduction surface (LoRA A/B rows for both the
//! policy and the critic encoder, plus the scalar value-head row) is
//! collected from the shared varmap by
//! [`crate::ppo::PpoTrainer::collect_trainable_params`].

use std::sync::Arc;

use candle_core::Tensor;
use candle_core::backprop::GradStore;

use crate::error::BlazenTrainError;
use crate::ppo::{PpoBatch, PpoTrainer};

use super::allreduce::{average_by, ring_all_reduce};
use super::ring::RingTopology;
use super::transport::RingTransport;

/// Distributed-training wrapper around an inner [`PpoTrainer`].
///
/// Owns:
/// - the inner single-device PPO trainer (forward / backward / optimizer),
/// - the ring [`RingTopology`] for this worker,
/// - a [`RingTransport`] to push / pull gradient chunks.
///
/// Each [`AllReducePpoTrainer::step`] computes the PPO loss locally via
/// [`PpoTrainer::forward_loss`], runs `loss.backward()` to populate the
/// local `GradStore`, then ring-AllReduces every trainable parameter's
/// gradient (LoRA A/B rows on both the policy and critic encoders, plus
/// the scalar value-head row), divides by `world_size` to recover the
/// mean, and finally drives the optimizer through
/// [`PpoTrainer::maybe_step_optimizer`]. PPO uses a single shared varmap
/// for both the actor and the critic, so a single trainable-param sweep
/// covers both subnets — no separate actor / critic ring is needed.
pub struct AllReducePpoTrainer {
    inner: PpoTrainer,
    topology: RingTopology,
    transport: Arc<dyn RingTransport>,
}

impl AllReducePpoTrainer {
    /// Wrap an existing [`PpoTrainer`].
    ///
    /// The inner trainer must already have its policy / critic / reference
    /// models loaded — the wrapper does not call the inner constructors
    /// itself so callers can share a model-load step across the ring.
    #[must_use]
    pub fn new(
        inner: PpoTrainer,
        topology: RingTopology,
        transport: Arc<dyn RingTransport>,
    ) -> Self {
        Self {
            inner,
            topology,
            transport,
        }
    }

    /// Borrow the underlying single-device PPO trainer (read-only access
    /// for callers that need its config / progress hooks / inspectors).
    #[must_use]
    pub fn inner(&self) -> &PpoTrainer {
        &self.inner
    }

    /// Mutably borrow the underlying PPO trainer (e.g. to install a
    /// progress sink or swap the reference snapshot before the first step).
    pub fn inner_mut(&mut self) -> &mut PpoTrainer {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner single-device PPO trainer.
    ///
    /// Useful at end-of-training when callers need to reclaim the inner
    /// trainer's [`candle_nn::VarMap`] for adapter export. Dropping the
    /// wrapper releases the [`RingTransport`] handle.
    #[must_use]
    pub fn into_inner(self) -> PpoTrainer {
        self.inner
    }

    /// Borrow this worker's ring topology.
    #[must_use]
    pub fn topology(&self) -> &RingTopology {
        &self.topology
    }

    /// Run one distributed PPO training step.
    ///
    /// Mirrors [`PpoTrainer::step`] but splices ring-AllReduce in between
    /// `loss.backward()` and the optimizer step:
    ///
    /// 1. Forward pass + PPO surrogate / VF / entropy / KL loss on the
    ///    local batch via [`PpoTrainer::forward_loss`].
    /// 2. `scaled_loss.backward()` to populate the local `GradStore`.
    /// 3. For each trainable param (policy LoRA + critic LoRA + value
    ///    head), AllReduce the per-param gradient across the ring and
    ///    divide by `world_size`.
    /// 4. [`PpoTrainer::maybe_step_optimizer`] on the averaged gradients
    ///    (respects gradient accumulation + grad-clip), then
    ///    [`PpoTrainer::bump_global_step`] (PPO's convention: the global
    ///    counter advances every step, not just on optimizer-fire steps).
    ///
    /// Identical to [`PpoTrainer::step`] when `topology.world_size == 1`
    /// (the AllReduce kernel short-circuits inside `ring_all_reduce`, but
    /// we still skip the extra Var collection by delegating directly).
    ///
    /// # Errors
    /// Propagates [`BlazenTrainError`] from forward, backward, transport,
    /// or optimizer paths.
    pub async fn step(&mut self, batch: &PpoBatch) -> Result<f32, BlazenTrainError> {
        // World size 1 → delegate to the inner trainer directly.
        if !self.topology.is_distributed() {
            return self.inner.step(batch);
        }

        // Mirror PpoTrainer::step's LR-then-forward ordering so the
        // scheduler observes the same global_step value it would on a
        // single-worker run.
        let lr = self.inner.lr_for_step(self.inner.global_step());
        self.inner.set_optimizer_lr(lr);

        let loss = self.inner.forward_loss(batch)?;
        let accum = self.inner.config().gradient_accumulation_steps.max(1);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / accum as f64;
        let scaled_loss = (&loss * scale)?;
        let loss_value = loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;

        let mut grads: GradStore = scaled_loss.backward()?;

        let vars = self.inner.collect_trainable_params();

        // AllReduce each trainable gradient. Tag = enumerate index so
        // every param has a distinct (step, chunk_id) namespace inside
        // the same optimizer step. Includes the value-head row — PPO's
        // critic shares the optimizer with the policy and the row must
        // ride the same ring or workers will desync on the value head.
        for (tag, var) in vars.iter().enumerate() {
            let Some(grad) = grads.remove(var.as_tensor()) else {
                // No gradient flowed through this param this batch
                // (e.g. dropout, masked tokens, or the KL term being
                // disabled). Still participate in the ring with a zero
                // tensor so peers don't deadlock on a missing chunk.
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

        // maybe_step_optimizer handles the gradient-accumulation counter
        // and grad-clip; we mirror PpoTrainer::step's contract of bumping
        // the global step every call (even when the optimizer didn't fire).
        let _stepped = self.inner.maybe_step_optimizer(&mut grads, &vars)?;
        self.inner.bump_global_step();

        Ok(loss_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::transport::InMemoryRingTransport;

    #[test]
    fn all_reduce_ppo_trainer_topology_wiring() {
        // We don't drive a full PPO forward here — that requires loaded
        // tiny llamas which the existing `ppo::tests` cover. This test
        // exercises construction + topology wiring on a world-size-2
        // ring, mirroring the AllReduceTrainer construction smoke test.
        let ring = InMemoryRingTransport::ring(2);
        let topo =
            RingTopology::new(0, vec!["a:1".to_string(), "b:1".to_string()]).expect("topology");
        assert_eq!(topo.world_size, 2);
        assert_eq!(topo.rank, 0);
        assert!(topo.is_distributed());
        // Transport reference held to keep `ring` alive across both ends.
        let _t0 = ring[0].clone();
    }
}
