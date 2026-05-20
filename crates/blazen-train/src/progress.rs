//! Training-progress callback surface.

use std::path::PathBuf;
use std::time::Duration;

use crate::error::BlazenTrainError;

/// Sink for per-step training events.
///
/// Called from [`crate::trainer::Trainer::step`] and
/// [`crate::trainer::Trainer::run`]. Returning `Err(...)` cancels
/// training; the trainer surfaces it as [`BlazenTrainError::Cancelled`].
pub trait TrainingProgress: Send + Sync {
    /// Receive one event. Return `Err(_)` to abort the run.
    ///
    /// # Errors
    ///
    /// Implementations may return any [`BlazenTrainError`] to signal
    /// cancellation; the trainer rewrites it into
    /// [`BlazenTrainError::Cancelled`].
    fn on_event(&self, event: TrainingEvent) -> Result<(), BlazenTrainError>;
}

/// One observable event emitted during a training run.
#[derive(Debug, Clone)]
pub enum TrainingEvent {
    /// Fired exactly once when training begins.
    Started {
        /// Total number of optimizer steps the run will execute.
        total_steps: usize,
    },
    /// Fired after each optimizer step completes.
    StepCompleted {
        /// 0-indexed step counter.
        step: usize,
        /// Loss value from this step's forward pass.
        loss: f32,
        /// Learning rate applied by the scheduler at this step.
        learning_rate: f64,
        /// Wall-clock time spent on this step.
        elapsed: Duration,
    },
    /// Fired immediately before an evaluation pass begins.
    Evaluating {
        /// Step counter at which evaluation was triggered.
        step: usize,
    },
    /// Fired when an evaluation pass finishes.
    EvalCompleted {
        /// Step counter at which evaluation finished.
        step: usize,
        /// Average loss over the eval split.
        eval_loss: f32,
    },
    /// Fired after a checkpoint has been written to disk.
    CheckpointSaved {
        /// Step counter at which the checkpoint was saved.
        step: usize,
        /// Filesystem path the checkpoint was written to.
        path: PathBuf,
    },
    /// Fired exactly once when training completes successfully.
    Finished {
        /// Final training loss.
        final_loss: f32,
        /// Total number of optimizer steps executed.
        total_steps: usize,
        /// Directory the exported adapter was written to.
        adapter_dir: PathBuf,
    },
}
