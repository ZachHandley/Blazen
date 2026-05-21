//! `AllReduceTrainer` — wraps a [`crate::Trainer`] with ring-AllReduce
//! gradient averaging across an N-worker ring before each optimizer step.
//!
//! Phase 1 ships the SFT/LoRA wrapper around [`crate::Trainer`]. Phase 2
//! extends the same pattern to `QloraTrainer`, `GrpoTrainer`, and the
//! upcoming `PpoTrainer`.

use std::sync::Arc;

use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};

use crate::error::BlazenTrainError;
use crate::lora::freeze_base_params;
use crate::trainer::{Trainer, TrainingBatch};

use super::allreduce::{average_by, ring_all_reduce};
use super::ring::RingTopology;
use super::transport::RingTransport;

/// Distributed-training wrapper around an inner [`Trainer`].
///
/// Owns:
/// - the inner single-device trainer (forward / backward / optimizer),
/// - the ring [`RingTopology`] for this worker,
/// - a [`RingTransport`] to push / pull gradient chunks.
///
/// Each `step` computes local gradients via the inner trainer's forward
/// + `loss.backward()`, runs [`ring_all_reduce`] on every trainable
///   parameter's gradient, divides by `world_size` to recover the mean,
///   then performs the optimizer step on the averaged gradients.
pub struct AllReduceTrainer {
    inner: Trainer,
    topology: RingTopology,
    transport: Arc<dyn RingTransport>,
}

impl AllReduceTrainer {
    /// Wrap an existing [`Trainer`].
    ///
    /// The inner trainer must already have its base model loaded — the
    /// wrapper does not call `load_base_model` itself so callers can
    /// share a model-load step across the ring.
    #[must_use]
    pub fn new(inner: Trainer, topology: RingTopology, transport: Arc<dyn RingTransport>) -> Self {
        Self {
            inner,
            topology,
            transport,
        }
    }

    /// Borrow the underlying single-device trainer (read-only access for
    /// callers that need its config / progress hooks).
    #[must_use]
    pub fn inner(&self) -> &Trainer {
        &self.inner
    }

    /// Mutably borrow the underlying trainer (e.g. to install progress
    /// callbacks, swap the dataset, or call `load_base_model` before the
    /// first step).
    pub fn inner_mut(&mut self) -> &mut Trainer {
        &mut self.inner
    }

    /// Borrow this worker's ring topology.
    #[must_use]
    pub fn topology(&self) -> &RingTopology {
        &self.topology
    }

    /// Run one distributed training step.
    ///
    /// Mirrors [`Trainer::step`] but splices ring-AllReduce in between
    /// `loss.backward()` and the optimizer step:
    ///
    /// 1. Forward pass + masked cross-entropy on the local batch.
    /// 2. `scaled_loss.backward()` to populate the local `GradStore`.
    /// 3. For each trainable param, AllReduce the per-param gradient
    ///    across the ring and divide by `world_size`.
    /// 4. Optimizer step on the averaged gradients.
    ///
    /// Identical to [`Trainer::step`] when `topology.world_size == 1`
    /// (the AllReduce kernel short-circuits).
    ///
    /// # Errors
    /// Propagates [`BlazenTrainError`] from forward, backward, transport,
    /// or optimizer paths.
    pub async fn step(&mut self, batch: TrainingBatch) -> Result<f32, BlazenTrainError> {
        // World size 1 → delegate to the inner trainer directly.
        if !self.topology.is_distributed() {
            return self.inner.step(batch).await;
        }

        // Reach into the inner trainer's loss / grad pipeline. We
        // duplicate the kernel of `Trainer::step` rather than calling
        // it because the optimizer-step boundary is exactly where the
        // AllReduce splices in, and the inner method bundles them.
        let lr = self.inner.lr_for_step(self.inner.global_step());
        self.inner.set_optimizer_lr(lr);

        let loss = self.inner.forward_loss(&batch)?;
        let accum = self.inner.config().gradient_accumulation_steps.max(1);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / accum as f64;
        let scaled_loss = (&loss * scale)?;
        let loss_value = loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;

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
                // No gradient flowed through this param this batch
                // (e.g. dropout or masked tokens zeroed it). Still
                // participate in the ring with a zero tensor so peers
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
    use crate::config::{LoraConfig, OptimConfig, SchedulerConfig, SchedulerKind, TrainConfig};
    use crate::distributed::transport::InMemoryRingTransport;
    use crate::trainer::Trainer;
    use std::path::PathBuf;

    fn tiny_train_config() -> TrainConfig {
        TrainConfig {
            base_model_repo: "tiny".to_string(),
            output_dir: PathBuf::from("./_pr_d_test_out"),
            lora: LoraConfig {
                rank: 2,
                alpha: 4.0,
                dropout: 0.0,
                target_modules: vec!["q_proj".to_string()],
            },
            optim: OptimConfig {
                learning_rate: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                gradient_clip: None,
            },
            scheduler: SchedulerConfig {
                kind: SchedulerKind::Constant,
                warmup_steps: 0,
            },
            max_steps: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            max_seq_len: 16,
            eval_steps: None,
            save_steps: None,
            seed: 0,
            mixed_precision: crate::config::MixedPrecision::None,
            device: None,
        }
    }

    #[tokio::test]
    async fn all_reduce_trainer_constructs_with_world_2_inmemory() {
        // We don't drive a real forward pass here — the Trainer
        // forward path requires a loaded base model, which would need
        // HF download / weights. The test exercises construction +
        // topology wiring (the rest is exercised by the ring/transport
        // tests + the kernel tests). Once the bindings test surface
        // grows a tiny fake model we'll layer a full one-step test on top.
        let ring = InMemoryRingTransport::ring(2);
        let cfg = tiny_train_config();
        let varmap = candle_nn::VarMap::new();
        let device = candle_core::Device::Cpu;
        let trainer = Trainer::new(cfg, varmap, device).expect("trainer");
        let topo =
            RingTopology::new(0, vec!["a:1".to_string(), "b:1".to_string()]).expect("topology");
        let dist = AllReduceTrainer::new(trainer, topo.clone(), ring[0].clone());
        assert_eq!(dist.topology().world_size, 2);
        assert_eq!(dist.topology().rank, 0);
        assert!(dist.topology().is_distributed());
    }
}
