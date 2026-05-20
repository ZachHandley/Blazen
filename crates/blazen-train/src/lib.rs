//! Local fine-tuning for Blazen. SFT + LoRA today; preference optimization
//! (DPO/ORPO/SimPO/KTO) in PR8.
//!
//! See the per-module docs for the wave-by-wave fill-in plan; PR7 Wave 1
//! ships the public surface (config, error, progress, `LoraLinear`,
//! `Trainer` state machine) and stubs the per-arch wrappers, dataset,
//! export, checkpoint, and scheduler modules with `unimplemented!()`
//! markers for the trailing waves.

// Why: this crate's docs are saturated with ML acronyms (LoRA, PEFT,
// AdamW, SFT, DPO, ORPO, SimPO, KTO, JSONL, GGUF, FP32, BF16, ...) that
// aren't Rust symbols. Backticking each is busywork that hurts readability;
// blazen-py uses the same allow for the same reason.
#![allow(clippy::doc_markdown)]

pub mod config;
pub mod error;
pub mod progress;

#[cfg(feature = "engine")]
pub mod arch;
#[cfg(feature = "engine")]
pub mod checkpoint;
#[cfg(feature = "engine")]
pub mod dataset;
#[cfg(feature = "engine")]
pub mod export;
#[cfg(feature = "engine")]
pub mod grad_clip;
#[cfg(feature = "engine")]
pub mod lora;
#[cfg(feature = "engine")]
pub mod mixed_precision;
#[cfg(feature = "engine")]
pub mod schedulers;
#[cfg(feature = "engine")]
pub mod trainer;

pub use config::{
    LoraConfig, MixedPrecision, OptimConfig, SchedulerConfig, SchedulerKind, TrainConfig,
};
pub use error::BlazenTrainError;
pub use progress::{TrainingEvent, TrainingProgress};

#[cfg(feature = "engine")]
pub use lora::{LoraLinear, freeze_base_params, lora_param_names};
#[cfg(feature = "engine")]
pub use trainer::{ReferenceModel, TrainedAdapter, Trainer, TrainingBatch, TrainingDataset};
