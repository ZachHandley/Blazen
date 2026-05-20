//! Error type for the training surface.

use thiserror::Error;

/// Top-level error type for all `blazen-train` operations.
#[derive(Debug, Error)]
pub enum BlazenTrainError {
    /// Configuration values failed validation before the trainer ran.
    #[error("invalid training config: {0}")]
    InvalidConfig(String),

    /// Failure inside the dataset / batch loader.
    #[error("dataset error: {0}")]
    Dataset(String),

    /// Failure while loading the base model weights / tokenizer.
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Failure during the forward pass.
    #[error("forward pass failed: {0}")]
    Forward(String),

    /// Failure during autograd (backward pass).
    #[error("backward pass failed: {0}")]
    Backward(String),

    /// Failure during the optimizer step.
    #[error("optimizer step failed: {0}")]
    Optimizer(String),

    /// Failure writing the trained adapter to disk.
    #[error("adapter export failed: {0}")]
    Export(String),

    /// Failure reading or writing a checkpoint.
    #[error("checkpoint failed: {0}")]
    Checkpoint(String),

    /// A progress callback returned `Err(...)` to cancel the training run.
    #[error("training cancelled by progress callback")]
    Cancelled,

    /// Forwarded from candle-core (tensor op, dtype/shape mismatch, etc.).
    #[cfg(feature = "engine")]
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// I/O error (filesystem, network, etc.).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization error.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

// Why: lossless mapping into the framework-wide `BlazenError` so `?`
// composes through `ModelManager::train_lora` without callers writing
// boilerplate. Cancellation maps to the dedicated cancelled() constructor
// so cancellation can be discriminated from genuine failures; configuration
// validation surfaces as Validation; everything else (HF download, candle
// tensor ops, optimizer/export/checkpoint I/O) collapses to internal(...)
// because there is no granular sibling on BlazenError today.
#[cfg(feature = "blazen-llm-interop")]
impl From<BlazenTrainError> for blazen_llm::BlazenError {
    fn from(e: BlazenTrainError) -> Self {
        match e {
            BlazenTrainError::Cancelled => Self::cancelled(),
            BlazenTrainError::InvalidConfig(msg) => Self::validation(format!("training: {msg}")),
            other => Self::internal(format!("training error: {other}")),
        }
    }
}
