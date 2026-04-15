//! Local inference backend bridges.
//!
//! Each sub-module gates behind a feature flag and implements the appropriate
//! `blazen-llm` trait (e.g. [`EmbeddingModel`](crate::EmbeddingModel)) for
//! the backing crate's model type.

#[cfg(feature = "candle-embed")]
pub mod candle_embed;

#[cfg(feature = "candle-llm")]
pub mod candle_llm;

#[cfg(feature = "embed")]
pub mod embed;

#[cfg(feature = "mistralrs")]
pub mod mistralrs;

#[cfg(feature = "whispercpp")]
pub mod whispercpp;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;
