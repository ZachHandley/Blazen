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

    /// A feature was requested that this release does not implement yet.
    /// Used for honest, documented deferrals (e.g. activation
    /// checkpointing, multi-shard safetensors, full fine-tunes of
    /// models above the safe-on-consumer-GPU param count).
    #[error("unsupported: {0}")]
    Unsupported(String),

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

    /// Offline adapter merge failed (single-adapter merge or multi-adapter
    /// weighted blend). See [`MergeError`] for the underlying classification.
    #[error("adapter merge failed: {0}")]
    Merge(#[from] MergeError),
}

/// Classifies failures from the offline LoRA merge surface
/// ([`crate::merge::merge_lora_into_base`] and
/// [`crate::merge::merge_lora_blend`]).
///
/// All variants carry a human-readable message naming the offending tensor
/// (or path) so a caller staring at a stderr log can identify which adapter
/// or base shard caused the failure without re-running under a debugger.
#[derive(Debug, Error)]
pub enum MergeError {
    /// Base model safetensors did not contain the target tensor that an
    /// adapter wanted to patch (i.e. the LoRA targets a module that does
    /// not exist in the supplied base — typically a base-model mismatch).
    #[error("base tensor missing for adapter target: {0}")]
    MissingBaseTensor(String),

    /// `B @ A` shape did not match the base linear's weight shape
    /// (typically a mid-training shape edit or a corrupt adapter file).
    #[error("shape mismatch merging {module}: base={base:?}, delta={delta:?}")]
    ShapeMismatch {
        /// Stripped module path (e.g. `model.layers.0.self_attn.q_proj`).
        module: String,
        /// Base weight shape (rows, cols) — i.e. (`out_dim`, `in_dim`).
        base: Vec<usize>,
        /// Computed `B @ A` shape (rows, cols).
        delta: Vec<usize>,
    },

    /// Two adapters in a blend disagree about the rank of the same target
    /// module. The merge math itself does not require equal ranks (the
    /// `B @ A` product collapses rank), but mixing ranks is almost always
    /// a sign the caller paired adapters trained against different base
    /// models — surfaced as an error so the failure mode is explicit.
    #[error(
        "adapter rank inconsistency on {module}: adapter 0 has rank {first_rank}, adapter {other_index} has rank {other_rank}"
    )]
    RankMismatch {
        /// Stripped module path the two adapters disagree on.
        module: String,
        /// Rank reported by the first adapter that touched `module`.
        first_rank: usize,
        /// Index (into the input slice) of the offending adapter.
        other_index: usize,
        /// Rank reported by the offending adapter.
        other_rank: usize,
    },

    /// An adapter has `lora_A` but no matching `lora_B` (or vice versa)
    /// for a given module. Mirrors the inference-loader's pairing rule
    /// in `crates/blazen-llm-candle/src/lora.rs::build_layers`.
    #[error("adapter has unpaired LoRA halves for module '{module}': missing {missing}")]
    UnpairedLora {
        /// Stripped module path.
        module: String,
        /// Which half is missing (`lora_A.weight` or `lora_B.weight`).
        missing: &'static str,
    },

    /// Adapter `adapter_model.safetensors` could not be parsed (wrong key
    /// shape, malformed tensor metadata, etc.).
    #[error("malformed adapter at {path}: {reason}")]
    MalformedAdapter {
        /// Filesystem path of the adapter that failed to parse.
        path: String,
        /// Underlying parse / shape diagnostic.
        reason: String,
    },

    /// Empty adapter list passed to [`crate::merge::merge_lora_blend`]
    /// (zero adapters cannot yield a meaningful blend).
    #[error("merge_lora_blend requires at least one adapter, got zero")]
    EmptyBlend,

    /// Underlying candle tensor failure (dtype mismatch in `B @ A`,
    /// device transfer issue, etc.).
    #[cfg(feature = "engine")]
    #[error("candle tensor op failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// Underlying safetensors serialization / parse error.
    #[error("safetensors I/O failed: {0}")]
    Safetensors(String),

    /// Filesystem I/O error (reading a base shard, writing the merged
    /// output, etc.).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
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
            BlazenTrainError::InvalidConfig(msg) | BlazenTrainError::Unsupported(msg) => {
                Self::validation(format!("training: {msg}"))
            }
            other => Self::internal(format!("training error: {other}")),
        }
    }
}
