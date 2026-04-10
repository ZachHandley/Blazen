//! Local inference backend bridges.
//!
//! Each sub-module gates behind a feature flag and implements the appropriate
//! `blazen-llm` trait (e.g. [`EmbeddingModel`](crate::EmbeddingModel)) for
//! the backing crate's model type.

#[cfg(feature = "fastembed")]
pub mod fastembed;

#[cfg(feature = "mistralrs")]
pub mod mistralrs;

#[cfg(feature = "whispercpp")]
pub mod whispercpp;
