//! Training configuration types.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Full configuration for a single training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// HuggingFace repo id of the base model (e.g. `"Qwen/Qwen2.5-0.5B"`).
    pub base_model_repo: String,
    /// Filesystem directory where the trained adapter and checkpoints land.
    pub output_dir: PathBuf,
    /// LoRA-specific hyperparameters.
    pub lora: LoraConfig,
    /// Optimizer hyperparameters (AdamW).
    pub optim: OptimConfig,
    /// Learning-rate schedule.
    pub scheduler: SchedulerConfig,
    /// Total optimizer steps to run.
    pub max_steps: usize,
    /// Micro-batch size (per forward pass).
    pub batch_size: usize,
    /// Number of micro-batches to accumulate before each optimizer step.
    pub gradient_accumulation_steps: usize,
    /// Maximum tokenized sequence length per example.
    pub max_seq_len: usize,
    /// If `Some`, run evaluation every N steps.
    pub eval_steps: Option<usize>,
    /// If `Some`, write a checkpoint every N steps.
    pub save_steps: Option<usize>,
    /// RNG seed (controls dataset shuffling and LoRA `A` init).
    pub seed: u64,
    /// Mixed-precision mode for forward / backward.
    pub mixed_precision: MixedPrecision,
    /// Device to place the training graph on, parsed by the caller:
    /// `"cpu"`, `"cuda:0"`, `"metal"`. `None` defers to the caller's
    /// default (e.g. `ModelManager::train_lora` treats `None` as `"cpu"`).
    /// The trainer itself takes a `candle_core::Device` directly; this
    /// field exists so configs carried across FFI boundaries can name a
    /// device without depending on `candle_core`.
    #[serde(default)]
    pub device: Option<String>,
}

/// LoRA hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Low-rank dimension (PEFT "r").
    pub rank: usize,
    /// Scaling numerator. Effective per-layer scale is `alpha / rank`.
    pub alpha: f32,
    /// Dropout probability applied to LoRA-A input (PEFT default 0.05).
    pub dropout: f32,
    /// Module-name suffixes to inject LoRA into
    /// (e.g. `["q_proj", "k_proj", "v_proj", "o_proj"]`).
    pub target_modules: Vec<String>,
}

/// AdamW optimizer hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimConfig {
    /// Peak learning rate (applied at end of warmup).
    pub learning_rate: f64,
    /// AdamW beta1.
    pub beta1: f64,
    /// AdamW beta2.
    pub beta2: f64,
    /// AdamW numerical-stability epsilon.
    pub epsilon: f64,
    /// AdamW weight decay (decoupled).
    pub weight_decay: f64,
    /// If `Some`, clip the global gradient L2-norm to this value.
    pub gradient_clip: Option<f32>,
}

/// Learning-rate scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Schedule shape.
    pub kind: SchedulerKind,
    /// Linear-warmup duration in steps (applied before the main shape).
    pub warmup_steps: usize,
}

/// Supported LR-schedule shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerKind {
    /// LR stays at peak after warmup.
    Constant,
    /// LR linearly decays from peak to 0 over the remaining steps.
    Linear,
    /// Half-cosine decay from peak to 0 over the remaining steps.
    Cosine,
}

/// Mixed-precision mode for the training graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixedPrecision {
    /// Full-precision fp32 weights and gradients.
    None,
    /// bf16 forward / backward, fp32 master weights and optimizer state.
    Bf16,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            base_model_repo: "Qwen/Qwen2.5-0.5B".to_string(),
            output_dir: PathBuf::from("./blazen-train-out"),
            lora: LoraConfig::default(),
            optim: OptimConfig::default(),
            scheduler: SchedulerConfig::default(),
            max_steps: 1000,
            batch_size: 1,
            gradient_accumulation_steps: 8,
            max_seq_len: 2048,
            eval_steps: None,
            save_steps: Some(200),
            seed: 42,
            mixed_precision: MixedPrecision::Bf16,
            device: None,
        }
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            gradient_clip: Some(1.0),
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            kind: SchedulerKind::Cosine,
            warmup_steps: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_config_serde_roundtrip() {
        let original = TrainConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: TrainConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.base_model_repo, original.base_model_repo);
        assert_eq!(parsed.output_dir, original.output_dir);
        assert_eq!(parsed.lora.rank, original.lora.rank);
        assert!((parsed.lora.alpha - original.lora.alpha).abs() < f32::EPSILON);
        assert_eq!(parsed.lora.target_modules, original.lora.target_modules);
        assert!((parsed.optim.learning_rate - original.optim.learning_rate).abs() < f64::EPSILON);
        assert_eq!(parsed.scheduler.kind, original.scheduler.kind);
        assert_eq!(
            parsed.scheduler.warmup_steps,
            original.scheduler.warmup_steps
        );
        assert_eq!(parsed.max_steps, original.max_steps);
        assert_eq!(parsed.mixed_precision, original.mixed_precision);
    }
}
