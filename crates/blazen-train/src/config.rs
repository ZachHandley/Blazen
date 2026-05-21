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

/// Common training hyperparameters shared by every training mode
/// (SFT, DPO, ORPO, `SimPO`, KTO, full fine-tune).
///
/// New preference / full-finetune configs compose this as their `core`
/// field rather than duplicating the same dozen fields. [`TrainConfig`]
/// stays flat for backward compatibility with PR7 users.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainCoreConfig {
    /// HuggingFace repo id of the base model.
    pub base_model_repo: String,
    /// Optional revision (branch / tag / commit) for the base model.
    #[serde(default)]
    pub base_model_revision: Option<String>,
    /// Filesystem directory for trained weights and checkpoints.
    pub output_dir: PathBuf,
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
    /// RNG seed.
    pub seed: u64,
    /// Mixed-precision mode for forward / backward.
    pub mixed_precision: MixedPrecision,
    /// Device name (`"cpu"`, `"cuda:0"`, `"metal"`); `None` defers to caller.
    #[serde(default)]
    pub device: Option<String>,
    /// Optimizer hyperparameters (AdamW).
    pub optim: OptimConfig,
    /// Learning-rate schedule.
    pub scheduler: SchedulerConfig,
}

impl Default for TrainCoreConfig {
    fn default() -> Self {
        Self {
            base_model_repo: "Qwen/Qwen2.5-0.5B".to_string(),
            base_model_revision: None,
            output_dir: PathBuf::from("./train_output"),
            max_steps: 1000,
            batch_size: 1,
            gradient_accumulation_steps: 8,
            max_seq_len: 1024,
            eval_steps: None,
            save_steps: None,
            seed: 42,
            mixed_precision: MixedPrecision::Bf16,
            device: None,
            optim: OptimConfig::default(),
            scheduler: SchedulerConfig::default(),
        }
    }
}

/// Direct Preference Optimization (DPO) configuration.
///
/// DPO requires a frozen reference model. If `reference_model_repo` is
/// `None`, the trainer uses `core.base_model_repo` as the reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters applied to the policy model.
    pub lora: LoraConfig,
    /// KL-regularization strength (TRL default 0.1).
    pub beta: f32,
    /// Reference model repo. `None` reuses `core.base_model_repo`.
    #[serde(default)]
    pub reference_model_repo: Option<String>,
    /// Optional revision for the reference model.
    #[serde(default)]
    pub reference_model_revision: Option<String>,
    /// Conservative DPO label smoothing (cDPO). 0.0 disables.
    pub label_smoothing: f32,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            core: TrainCoreConfig::default(),
            lora: LoraConfig::default(),
            beta: 0.1,
            reference_model_repo: None,
            reference_model_revision: None,
            label_smoothing: 0.0,
        }
    }
}

/// Odds Ratio Preference Optimization (ORPO) configuration.
///
/// Reference-free — combines an SFT loss on chosen responses with an
/// odds-ratio loss term weighted by `lambda`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrpoConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters.
    pub lora: LoraConfig,
    /// Weight of the odds-ratio term relative to the SFT term.
    pub lambda: f32,
}

impl Default for OrpoConfig {
    fn default() -> Self {
        Self {
            core: TrainCoreConfig::default(),
            lora: LoraConfig::default(),
            lambda: 0.1,
        }
    }
}

/// Simple Preference Optimization (`SimPO`) configuration.
///
/// Reference-free, length-normalized. Defaults follow TRL `main`
/// (beta=2.0, gamma=1.0), not the older TRL release (0.1 / 0.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpoConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters.
    pub lora: LoraConfig,
    /// Logit scaling for the length-normalized preference margin.
    pub beta: f32,
    /// Target reward margin between chosen and rejected.
    pub gamma: f32,
}

impl Default for SimpoConfig {
    fn default() -> Self {
        Self {
            core: TrainCoreConfig::default(),
            lora: LoraConfig::default(),
            beta: 2.0,
            gamma: 1.0,
        }
    }
}

/// Kahneman-Tversky Optimization (KTO) configuration.
///
/// Requires a frozen reference model. Unlike DPO, KTO consumes a
/// `(prompt, completion, desirable)` triple rather than chosen/rejected pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KtoConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters applied to the policy model.
    pub lora: LoraConfig,
    /// KL-regularization strength.
    pub beta: f32,
    /// Loss weight applied to desirable examples.
    pub lambda_d: f32,
    /// Loss weight applied to undesirable examples.
    pub lambda_u: f32,
    /// Reference model repo. `None` reuses `core.base_model_repo`.
    #[serde(default)]
    pub reference_model_repo: Option<String>,
    /// Optional revision for the reference model.
    #[serde(default)]
    pub reference_model_revision: Option<String>,
}

impl Default for KtoConfig {
    fn default() -> Self {
        Self {
            core: TrainCoreConfig::default(),
            lora: LoraConfig::default(),
            beta: 0.1,
            lambda_d: 1.0,
            lambda_u: 1.0,
            reference_model_repo: None,
            reference_model_revision: None,
        }
    }
}

/// Reward model training configuration.
///
/// The reward model is a base LM (today: Llama) with a scalar reward head
/// — a single `Linear(hidden_size, 1)` projection applied to the final
/// non-pad token's post-norm hidden state. Training uses the standard
/// Bradley-Terry pairwise loss `-log(sigmoid(r_chosen - r_rejected))` on
/// preference-pair data (the same JSONL format consumed by DPO/ORPO/SimPO).
///
/// Only Llama-family base models are supported in PR-R phase 1 — Qwen2 /
/// Mistral reward heads are deferred to a later phase (the `forward_hidden_states`
/// entry point is currently only implemented on `TrainableLlama`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters applied to the base model. Set
    /// `target_modules = []` to train every base weight (full fine-tune).
    pub lora: LoraConfig,
}

/// Group Relative Policy Optimization (GRPO) configuration.
///
/// DeepSeek's critic-free PPO replacement. Each training step samples
/// `group_size` completions per prompt from the current policy, scores
/// each with a frozen reward model, computes group-relative advantages
/// `(r_i - mean_group) / (std_group + eps)`, then minimizes
/// `-mean_i(advantage_i * log_prob_i) + beta * KL(policy || reference)`.
///
/// The reference model is a frozen copy of the policy at step 0 (same
/// pattern as DPO's reference).
///
/// Phase 1 deferrals (documented as `Unsupported` at runtime):
/// - reward-model loading from disk: the trainer accepts a constructed
///   [`crate::reward::RewardModel`] via `set_reward_model`; HF Hub
///   download of a reward-model adapter is deferred to phase 2 alongside
///   bindings.
/// - on-policy completion sampling: phase 1's `step_grpo` consumes a
///   caller-provided `GrpoBatch` of pre-sampled completions; the
///   in-trainer sampler is wired up in phase 2 when the manager-level
///   sampling APIs land.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters applied to the policy model.
    pub lora: LoraConfig,
    /// Number of sampled completions per prompt. 4-8 is the typical range;
    /// DeepSeek's published recipe uses 16 for math reasoning. Must be >= 2
    /// (a singleton group has zero variance and the advantage normalizer
    /// collapses to 0/0).
    pub group_size: usize,
    /// KL-regularization strength against the frozen reference policy.
    /// DeepSeek default 0.04; TRL defaults to 0.1. Pick by your taste for
    /// behavioral drift vs. reward chasing.
    pub beta: f32,
    /// Numerical-stability epsilon added to the per-group standard
    /// deviation before normalizing advantages. Prevents NaN when every
    /// completion in a group scores identically.
    pub advantage_epsilon: f32,
    /// Sampling temperature for stochastic completion generation in
    /// phase 2's in-trainer sampler. Carried on the config so callers can
    /// pin reproducibility ahead of the sampler landing.
    pub sampling_temperature: f32,
    /// Reward model repo. `None` reuses `core.base_model_repo` as the
    /// reward-model architecture (the reward head still needs to be
    /// trained or loaded separately via [`crate::reward::RewardTrainer`]).
    #[serde(default)]
    pub reward_model_repo: Option<String>,
    /// Optional revision for the reward model.
    #[serde(default)]
    pub reward_model_revision: Option<String>,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            core: TrainCoreConfig::default(),
            lora: LoraConfig::default(),
            group_size: 4,
            beta: 0.04,
            advantage_epsilon: 1e-8,
            sampling_temperature: 1.0,
            reward_model_repo: None,
            reward_model_revision: None,
        }
    }
}

/// How to initialize the value (critic) model in a PPO run.
///
/// PPO's critic is a `LlamaModel` + scalar value head (mirrors the reward
/// head). The encoder weights can be seeded from three places:
///
/// - [`ValueModelInit::FromPolicy`] — clone the policy's base weights (the
///   most common choice; gives the critic a head start at modeling the
///   same token distribution).
/// - [`ValueModelInit::FromReward`] — clone the reward model's encoder
///   (common in TRL when a reward model is already on disk; the critic
///   shares its inductive bias with the scoring function).
/// - [`ValueModelInit::Random`] — initialize from scratch via the
///   `VarBuilder`'s default initializer (useful for tests / ablations).
///
/// The scalar value head itself is always freshly initialized; only the
/// encoder weights are affected by this choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ValueModelInit {
    /// Clone the policy's base weights (recommended default).
    #[default]
    FromPolicy,
    /// Clone the reward model's encoder weights.
    FromReward,
    /// Random init via the `VarBuilder`'s default initializer.
    Random,
}

/// Proximal Policy Optimization (PPO) configuration.
///
/// Classical actor-critic RLHF: each step rolls out completions from the
/// current policy, scores them with a frozen reward model, computes GAE
/// advantages against a learned critic ("value model"), and updates the
/// policy with the PPO clipped surrogate objective + a value-function
/// regression + an entropy bonus.
///
/// Loss (per-token, masked to completion positions):
///
/// ```text
/// ratio_t   = exp( log π_pol(a_t|...) - log π_old(a_t|...) )
/// clip_t    = clip(ratio_t, 1 - eps, 1 + eps)
/// pg_loss   = -mean_t( min(ratio_t * adv_t, clip_t * adv_t) )
/// vf_loss   = mean_t( (V(s_t) - return_t)^2 )
/// ent_loss  = -mean_t( H(π_pol(·|s_t)) )
/// loss      = pg_loss + value_coef * vf_loss + entropy_coef * ent_loss
///             + kl_coef * mean_t( KL(π_pol || π_ref) )       # optional
/// ```
///
/// GAE returns: `adv_t = δ_t + γλ * adv_{t+1}` where
/// `δ_t = r_t + γ * V(s_{t+1}) - V(s_t)`; `return_t = adv_t + V(s_t)`.
///
/// Phase 1 deferrals:
/// - Full `PromptDataset → PpoBatch` rollout loop (the trainer's `step`
///   consumes pre-rolled batches; the in-trainer sampler is wired up in
///   phase 2 alongside the GRPO sampler).
/// - HF-Hub reward-model loading.
/// - Bindings (Python/Node/WASM/UniFFI/CABI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters applied to the policy + critic encoder.
    pub lora: LoraConfig,
    /// PPO clip range epsilon. The OpenAI default is 0.2; values in
    /// `[0.1, 0.3]` are common. Smaller = more conservative updates.
    pub clip_epsilon: f32,
    /// Weight of the value-function regression term in the combined loss.
    /// OpenAI default 0.5.
    pub value_coef: f32,
    /// Weight of the entropy bonus. Encourages exploration; the
    /// `spinningup` / OpenAI default is 0.01.
    pub entropy_coef: f32,
    /// GAE smoothing parameter `λ ∈ [0, 1]`. OpenAI default 0.95.
    pub gae_lambda: f32,
    /// Discount factor `γ ∈ [0, 1]`. RLHF typically uses 1.0 (episodic).
    pub gamma: f32,
    /// Optional KL-to-reference penalty coefficient. `0.0` disables (the
    /// clipping objective already constrains policy drift); some recipes
    /// add a small KL term on top.
    pub kl_coef: f32,
    /// Reward model repo. `None` means the trainer expects a pre-built
    /// [`crate::reward::RewardModel`] supplied at construction time
    /// (phase 1 path; HF-Hub loading is phase 2).
    #[serde(default)]
    pub reward_model_repo: Option<String>,
    /// Optional revision for the reward model.
    #[serde(default)]
    pub reward_model_revision: Option<String>,
    /// Where to seed the critic's encoder weights from.
    #[serde(default)]
    pub value_model_init: ValueModelInit,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            core: TrainCoreConfig::default(),
            lora: LoraConfig::default(),
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
}

/// Full fine-tune configuration (no LoRA — every parameter trains).
///
/// `gradient_checkpointing = true` is accepted for forward compatibility
/// but candle 0.10.2 has no checkpointing primitive, so the trainer will
/// return `Unsupported` at init time when this is set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FullFineTuneConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// Activation checkpointing (currently unsupported in the trainer).
    pub gradient_checkpointing: bool,
}

/// Quantization dtype for the frozen base weights in a QLoRA run.
///
/// Mirrors a subset of [`candle_core::quantized::GgmlDType`] but lives in
/// the config crate (no candle dependency) so it can cross FFI boundaries.
/// The trainer's QLoRA wrapper maps each variant 1:1 onto the candle dtype
/// at model-build time.
///
/// All variants are integer-quantization formats with no learnable scale
/// — the dequant-on-matmul kernel reads the packed blocks and produces an
/// f32/bf16 output the LoRA adapters add into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QloraQuantDtype {
    /// 4-bit, 32-element blocks, single scale per block (`GgmlDType::Q4_0`).
    /// QLoRA's reference uses NF4 which candle doesn't ship; `Q4_0` is the
    /// closest 4-bit GGUF analogue and what mainstream candle inference
    /// quantizes Llama-family checkpoints to. Default for the same reason —
    /// NF4 is unavailable in candle 0.10.2 and `Q4_0` is the lowest-VRAM
    /// 4-bit format we can use today.
    #[default]
    Q4_0,
    /// 4-bit, K-quants superblock layout (`GgmlDType::Q4K`). Slightly higher
    /// effective bit-rate than `Q4_0` for the same accuracy budget.
    Q4K,
    /// 5-bit, 32-element blocks (`GgmlDType::Q5_0`). Higher fidelity than
    /// `Q4_0` at the cost of ~25% more VRAM for the frozen base.
    Q5_0,
    /// 8-bit, 32-element blocks (`GgmlDType::Q8_0`). The most accurate
    /// integer format candle ships; useful as a precision-baseline run.
    Q8_0,
}

/// QLoRA configuration: 4-bit (or other GGUF-integer) frozen base
/// weights + bf16/f32 trainable LoRA adapters.
///
/// The trainer reuses every SFT plumbing path (dataset, optimizer,
/// scheduler, grad-clip, checkpoint, exporter) and only swaps in
/// [`crate::qlora::QLoraLinear`] in place of [`crate::lora::LoraLinear`]
/// at the per-arch wrapper. Memory savings come from the 4-bit base,
/// not from any change to the gradient path: only the LoRA A/B matrices
/// are trainable and only their gradients live in optimizer state.
///
/// `base_quant` controls the integer format used for the frozen base
/// weights; `lora` controls the trainable adapter shape (rank, alpha,
/// dropout, target modules). `target_modules` is honored exactly the
/// way SFT honors it — non-target linears stay as plain (dequantized)
/// `Linear`s, target linears become [`crate::qlora::QLoraLinear`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QloraConfig {
    /// Shared training hyperparameters.
    pub core: TrainCoreConfig,
    /// LoRA hyperparameters applied to the policy model.
    pub lora: LoraConfig,
    /// Quantization format for the frozen base weights.
    pub base_quant: QloraQuantDtype,
}

/// Configuration for a multi-GPU / multi-node distributed training run.
///
/// Mirrors PyTorch's `torchrun` env vars (`RANK`, `WORLD_SIZE`,
/// `MASTER_ADDR`, `MASTER_PORT`) but in struct form so the same shape
/// can ride across the bindings without an extra layer of env parsing.
///
/// `peers` is the full ordered endpoint list — `peers[i]` is the gRPC
/// AllReduce endpoint (`"host:port"`) of rank `i`. `master_addr` /
/// `master_port` identify the bootstrap node (typically `peers[0]`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// 0-indexed rank of this worker. Must satisfy `rank < world_size`.
    pub rank: usize,
    /// Total number of workers in the ring.
    pub world_size: usize,
    /// Ordered list of `"host:port"` gRPC endpoints, indexed by rank.
    #[serde(default)]
    pub peers: Vec<String>,
    /// Bootstrap address (typically `peers[0]`'s host).
    pub master_addr: String,
    /// Bootstrap port.
    pub master_port: u16,
}

impl DistributedConfig {
    /// `true` when `world_size > 1` — the trainer should AllReduce.
    #[must_use]
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
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

    fn assert_core_eq(a: &TrainCoreConfig, b: &TrainCoreConfig) {
        assert_eq!(a.base_model_repo, b.base_model_repo);
        assert_eq!(a.base_model_revision, b.base_model_revision);
        assert_eq!(a.output_dir, b.output_dir);
        assert_eq!(a.max_steps, b.max_steps);
        assert_eq!(a.batch_size, b.batch_size);
        assert_eq!(a.gradient_accumulation_steps, b.gradient_accumulation_steps);
        assert_eq!(a.max_seq_len, b.max_seq_len);
        assert_eq!(a.eval_steps, b.eval_steps);
        assert_eq!(a.save_steps, b.save_steps);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.mixed_precision, b.mixed_precision);
        assert_eq!(a.device, b.device);
        assert!((a.optim.learning_rate - b.optim.learning_rate).abs() < f64::EPSILON);
        assert_eq!(a.scheduler.kind, b.scheduler.kind);
        assert_eq!(a.scheduler.warmup_steps, b.scheduler.warmup_steps);
    }

    fn assert_lora_eq(a: &LoraConfig, b: &LoraConfig) {
        assert_eq!(a.rank, b.rank);
        assert!((a.alpha - b.alpha).abs() < f32::EPSILON);
        assert!((a.dropout - b.dropout).abs() < f32::EPSILON);
        assert_eq!(a.target_modules, b.target_modules);
    }

    #[test]
    fn train_core_config_default_has_reasonable_values() {
        let cfg = TrainCoreConfig::default();
        assert!(cfg.max_steps > 0);
        assert!(cfg.batch_size > 0);
        assert!(cfg.gradient_accumulation_steps > 0);
        assert!(cfg.max_seq_len > 0);
        assert_ne!(cfg.seed, 0);
        assert!(!cfg.base_model_repo.is_empty());
    }

    #[test]
    fn dpo_config_default_beta_is_0_1() {
        let cfg = DpoConfig::default();
        assert!((cfg.beta - 0.1).abs() < f32::EPSILON);
        assert!(cfg.reference_model_repo.is_none());
        assert!((cfg.label_smoothing - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dpo_config_serde_roundtrip() {
        let original = DpoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: DpoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert!((parsed.beta - original.beta).abs() < f32::EPSILON);
        assert_eq!(parsed.reference_model_repo, original.reference_model_repo);
        assert_eq!(
            parsed.reference_model_revision,
            original.reference_model_revision
        );
        assert!((parsed.label_smoothing - original.label_smoothing).abs() < f32::EPSILON);
    }

    #[test]
    fn orpo_config_default_lambda_is_0_1() {
        let cfg = OrpoConfig::default();
        assert!((cfg.lambda - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn orpo_config_serde_roundtrip() {
        let original = OrpoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: OrpoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert!((parsed.lambda - original.lambda).abs() < f32::EPSILON);
    }

    #[test]
    fn simpo_config_default_beta_2_0_gamma_1_0() {
        let cfg = SimpoConfig::default();
        assert!((cfg.beta - 2.0).abs() < f32::EPSILON);
        assert!((cfg.gamma - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn simpo_config_serde_roundtrip() {
        let original = SimpoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: SimpoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert!((parsed.beta - original.beta).abs() < f32::EPSILON);
        assert!((parsed.gamma - original.gamma).abs() < f32::EPSILON);
    }

    #[test]
    fn kto_config_default_lambdas_both_1_0() {
        let cfg = KtoConfig::default();
        assert!((cfg.lambda_d - 1.0).abs() < f32::EPSILON);
        assert!((cfg.lambda_u - 1.0).abs() < f32::EPSILON);
        assert!((cfg.beta - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn kto_config_serde_roundtrip() {
        let original = KtoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: KtoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert!((parsed.beta - original.beta).abs() < f32::EPSILON);
        assert!((parsed.lambda_d - original.lambda_d).abs() < f32::EPSILON);
        assert!((parsed.lambda_u - original.lambda_u).abs() < f32::EPSILON);
        assert_eq!(parsed.reference_model_repo, original.reference_model_repo);
        assert_eq!(
            parsed.reference_model_revision,
            original.reference_model_revision
        );
    }

    #[test]
    fn full_finetune_config_default_grad_checkpointing_disabled() {
        let cfg = FullFineTuneConfig::default();
        assert!(!cfg.gradient_checkpointing);
    }

    #[test]
    fn full_finetune_config_serde_roundtrip() {
        let original = FullFineTuneConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: FullFineTuneConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_eq!(
            parsed.gradient_checkpointing,
            original.gradient_checkpointing
        );
    }

    #[test]
    fn train_core_config_serde_roundtrip() {
        let original = TrainCoreConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: TrainCoreConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed, &original);
    }

    #[test]
    fn qlora_config_default_is_q4_0() {
        let cfg = QloraConfig::default();
        assert_eq!(cfg.base_quant, QloraQuantDtype::Q4_0);
    }

    #[test]
    fn qlora_config_serde_roundtrip() {
        let original = QloraConfig {
            base_quant: QloraQuantDtype::Q4K,
            ..QloraConfig::default()
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: QloraConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert_eq!(parsed.base_quant, original.base_quant);
    }

    #[test]
    fn reward_config_default_inherits_core_defaults() {
        let cfg = RewardConfig::default();
        assert!(cfg.core.max_steps > 0);
        assert!(cfg.lora.rank > 0);
    }

    #[test]
    fn reward_config_serde_roundtrip() {
        let original = RewardConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: RewardConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
    }

    #[test]
    fn distributed_config_default_is_single_worker() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.world_size, 0);
        assert!(!cfg.is_distributed());
    }

    #[test]
    fn distributed_config_serde_roundtrip() {
        let original = DistributedConfig {
            rank: 2,
            world_size: 4,
            peers: vec![
                "host0:50051".to_string(),
                "host1:50051".to_string(),
                "host2:50051".to_string(),
                "host3:50051".to_string(),
            ],
            master_addr: "host0".to_string(),
            master_port: 50051,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: DistributedConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
        assert!(parsed.is_distributed());
    }

    #[test]
    fn grpo_config_default_group_size_is_four() {
        let cfg = GrpoConfig::default();
        assert_eq!(cfg.group_size, 4);
        assert!((cfg.beta - 0.04).abs() < f32::EPSILON);
        assert!(cfg.advantage_epsilon > 0.0);
        assert!((cfg.sampling_temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn grpo_config_serde_roundtrip() {
        let original = GrpoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: GrpoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert_eq!(parsed.group_size, original.group_size);
        assert!((parsed.beta - original.beta).abs() < f32::EPSILON);
        assert!((parsed.advantage_epsilon - original.advantage_epsilon).abs() < f32::EPSILON);
        assert!((parsed.sampling_temperature - original.sampling_temperature).abs() < f32::EPSILON);
        assert_eq!(parsed.reward_model_repo, original.reward_model_repo);
        assert_eq!(parsed.reward_model_revision, original.reward_model_revision);
    }

    #[test]
    fn ppo_config_default_has_openai_hyperparams() {
        let cfg = PpoConfig::default();
        assert!((cfg.clip_epsilon - 0.2).abs() < f32::EPSILON);
        assert!((cfg.value_coef - 0.5).abs() < f32::EPSILON);
        assert!((cfg.entropy_coef - 0.01).abs() < f32::EPSILON);
        assert!((cfg.gae_lambda - 0.95).abs() < f32::EPSILON);
        assert!((cfg.gamma - 1.0).abs() < f32::EPSILON);
        assert!((cfg.kl_coef - 0.0).abs() < f32::EPSILON);
        assert_eq!(cfg.value_model_init, ValueModelInit::FromPolicy);
    }

    #[test]
    fn ppo_config_serde_roundtrip() {
        let original = PpoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: PpoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_core_eq(&parsed.core, &original.core);
        assert_lora_eq(&parsed.lora, &original.lora);
        assert!((parsed.clip_epsilon - original.clip_epsilon).abs() < f32::EPSILON);
        assert!((parsed.value_coef - original.value_coef).abs() < f32::EPSILON);
        assert!((parsed.entropy_coef - original.entropy_coef).abs() < f32::EPSILON);
        assert!((parsed.gae_lambda - original.gae_lambda).abs() < f32::EPSILON);
        assert!((parsed.gamma - original.gamma).abs() < f32::EPSILON);
        assert!((parsed.kl_coef - original.kl_coef).abs() < f32::EPSILON);
        assert_eq!(parsed.reward_model_repo, original.reward_model_repo);
        assert_eq!(parsed.reward_model_revision, original.reward_model_revision);
        assert_eq!(parsed.value_model_init, original.value_model_init);
    }

    #[test]
    fn ppo_config_serde_roundtrip_non_default_init() {
        // Cover every `ValueModelInit` variant through serde so a future
        // rename of the enum doesn't silently break the wire format.
        for init in [
            ValueModelInit::FromPolicy,
            ValueModelInit::FromReward,
            ValueModelInit::Random,
        ] {
            let original = PpoConfig {
                value_model_init: init,
                kl_coef: 0.05,
                ..PpoConfig::default()
            };
            let json = serde_json::to_string(&original).expect("serialize");
            let parsed: PpoConfig = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed.value_model_init, init);
            assert!((parsed.kl_coef - 0.05).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn ppo_config_serde_accepts_legacy_no_init_field() {
        // The `value_model_init` field is `#[serde(default)]` so older
        // configs that pre-date the field still parse — the default is
        // `FromPolicy`.
        let json = r#"{
            "core": {
                "base_model_repo": "Qwen/Qwen2.5-0.5B",
                "output_dir": "./out",
                "max_steps": 100,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_seq_len": 1024,
                "eval_steps": null,
                "save_steps": null,
                "seed": 42,
                "mixed_precision": "Bf16",
                "device": null,
                "optim": {
                    "learning_rate": 2e-4,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "weight_decay": 0.0,
                    "gradient_clip": 1.0
                },
                "scheduler": { "kind": "Cosine", "warmup_steps": 50 }
            },
            "lora": {
                "rank": 16,
                "alpha": 32.0,
                "dropout": 0.05,
                "target_modules": ["q_proj","k_proj","v_proj","o_proj"]
            },
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "gae_lambda": 0.95,
            "gamma": 1.0,
            "kl_coef": 0.0
        }"#;
        let parsed: PpoConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.value_model_init, ValueModelInit::FromPolicy);
        assert!(parsed.reward_model_repo.is_none());
    }
}
