//! Error type for `blazen-train-tune`.

use thiserror::Error;

/// Errors raised by the search, journaling, or runner layers.
#[derive(Debug, Error)]
pub enum TuneError {
    /// A hyperparameter name was referenced that the `SearchSpace` does not
    /// declare. Typically a typo in `add` / `sample` / TPE post-hoc lookup.
    #[error("unknown hyperparameter: {0}")]
    UnknownParam(String),

    /// `GridSearch` was asked for a parameter whose distribution it cannot
    /// enumerate (e.g. continuous `Uniform`). Use `Discrete` or `Categorical`
    /// for grid axes.
    #[error("grid search cannot enumerate distribution `{0}`: {1}")]
    UngriddableDistribution(String, &'static str),

    /// `GridSearch` exhausted its product enumeration before the budget did.
    /// Treated as a graceful loop terminator by `Runner`.
    #[error("grid search exhausted: all {0} combinations have been proposed")]
    GridExhausted(usize),

    /// `Distribution::Uniform` / `LogUniform` / `IntUniform` constructed
    /// with `low >= high`, or `LogUniform` with non-positive bounds.
    #[error("invalid distribution bounds: {0}")]
    InvalidBounds(String),

    /// A trial config could not be coerced to the type the caller expected
    /// (e.g. `as_f64()` on a string field). Carries the offending key.
    #[error("type mismatch reading param `{key}`: expected {expected}, got {got}")]
    TypeMismatch {
        key: String,
        expected: &'static str,
        got: String,
    },

    /// I/O error from the JSONL journal.
    #[error("journal I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization failure when persisting / replaying a trial record.
    #[error("journal serde: {0}")]
    Serde(#[from] serde_json::Error),

    /// The user-supplied `Evaluator` returned an error. The runner records
    /// the trial as `Failed` and continues.
    #[error("evaluator failed: {0}")]
    Evaluator(String),
}
