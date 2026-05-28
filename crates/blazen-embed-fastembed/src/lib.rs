//! Local embedding model backend for Blazen using [`fastembed`].
//!
//! This crate wraps the [`fastembed`] Rust crate (ONNX Runtime) to provide
//! fully local, offline vector embeddings with no API keys required.
//!
//! When used through `blazen-llm` with the `fastembed` feature flag, this
//! crate's [`FastEmbedModel`] automatically implements
//! `blazen_llm::EmbeddingModel`.
//!
//! # Target gating
//!
//! As of `ort-sys` 2.0.0-rc.12, pyke dropped `x86_64-apple-darwin` from
//! their prebuilt distribution matrix. On that target this crate compiles
//! to a stub (see `provider_stub`) — the public API remains, but
//! [`FastEmbedModel::from_options`] always returns
//! [`FastEmbedError::UnsupportedTarget`]. Consumers should route through
//! `blazen-embed`'s facade, which picks `blazen-embed-tract` (pure-Rust
//! ONNX) on those targets automatically.
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
pub use options::FastEmbedOptions;

#[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
mod provider;
#[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
pub use provider::{FastEmbedError, FastEmbedModel, FastEmbedResponse};

#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
mod provider_stub;
#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
pub use provider_stub::{FastEmbedError, FastEmbedModel, FastEmbedResponse};
