//! Local embedding model backend for Blazen using [`fastembed`].
//!
//! This crate wraps the [`fastembed`] Rust crate (ONNX Runtime) to provide
//! fully local, offline vector embeddings with no API keys required.
//!
//! When used through `blazen-llm` with the `fastembed` feature flag, this
//! crate's [`FastEmbedModel`] automatically implements
//! `blazen_llm::EmbeddingModel`.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use blazen_embed_fastembed::{FastEmbedModel, FastEmbedOptions};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = FastEmbedModel::from_options(FastEmbedOptions::default())?;
//! let response = model.embed(&["hello".into(), "world".into()]).await?;
//! assert_eq!(response.embeddings.len(), 2);
//! # Ok(())
//! # }
//! ```

mod options;
mod provider;

pub use options::FastEmbedOptions;
pub use provider::{FastEmbedError, FastEmbedModel, FastEmbedResponse};
