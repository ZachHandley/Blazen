//! Local image generation backend for Blazen using [`diffusion-rs`](https://github.com/huggingface/diffusion-rs).
//!
//! This crate wraps the `diffusion-rs` pure-Rust Stable Diffusion inference
//! engine to provide fully local, offline image generation with no API keys
//! required.
//!
//! When used through `blazen-llm` with the `diffusion` feature flag, this
//! crate's [`DiffusionProvider`] will implement `blazen_llm::ImageGeneration`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                      |
//! |----------|--------------------------------------------------|
//! | `engine` | Links the actual `diffusion-rs` runtime (CPU)    |
//! | `cuda`   | NVIDIA CUDA GPU acceleration                     |
//! | `metal`  | Apple Silicon GPU acceleration (Metal)            |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run image generation. This keeps workspace
//! builds fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::{DiffusionOptions, DiffusionScheduler};
pub use provider::{DiffusionError, DiffusionProvider};
