//! AutoML / hyperparameter search on top of `blazen-train`.
//!
//! This crate is intentionally separate from `blazen-train` itself so that
//! the heavy training engine (candle, hf-hub, tokenizers) doesn't have to
//! be compiled in to use the searchers — the `Evaluator` trait is the only
//! coupling point, and callers wire their own training loop into it.
//!
//! Pieces:
//! - [`space`] — `SearchSpace` + `Distribution` (Categorical / IntUniform /
//!   Uniform / LogUniform / Discrete). The space defines what hyperparams
//!   exist and what each one's prior looks like.
//! - [`trial`] — `Trial` record + status state machine.
//! - [`searcher`] — `Searcher` trait + three real implementations:
//!   `RandomSearch`, `GridSearch`, and `TpeSearch` (Tree-structured
//!   Parzen Estimator, Bergstra et al. 2011, Algorithm 1).
//! - [`journal`] — JSONL append-only `TrialJournal` for crash recovery.
//! - [`runner`] — `Runner` glues a `SearchSpace` + `Searcher` + `Evaluator`
//!   + `TrialJournal` into a budgeted (max-trials / time-budget) loop,
//!   optionally fan-out across N tokio workers.
//!
//! See `examples/lora_sft_search.rs` for an end-to-end LoRA SFT search.
//!
//! ML acronyms (LoRA, TPE, KDE, SFT, ...) saturate the docs; backticking
//! each one would hurt readability without adding clarity.
#![allow(clippy::doc_markdown)]
#![allow(clippy::doc_lazy_continuation)]
// pedantic warns about returning Result from internal helpers; the API
// shape is intentional (errors bubble up to the runner).
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod journal;
pub mod runner;
pub mod searcher;
pub mod space;
pub mod trial;

pub use error::TuneError;
pub use journal::TrialJournal;
pub use runner::{EvalFuture, Evaluator, Runner, RunnerBudget};
pub use searcher::{GridSearch, RandomSearch, Searcher, TpeSearch};
pub use space::{Distribution, SearchSpace};
pub use trial::{Trial, TrialId, TrialStatus};
