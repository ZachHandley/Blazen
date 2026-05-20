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
//!
//! # Streaming support
//!
//! Image generation through this crate is **single-call**: one
//! [`DiffusionProvider::generate_image`] invocation returns one complete
//! [`blazen_llm::ImageResult`]. There is no per-step callback or partial
//! result stream.
//!
//! Upstream `diffusion-rs` does expose a
//! `gen_img_with_progress(&Config, &mut ModelConfig, Sender<Progress>)`
//! entry point that emits `(step, total_steps, time)` triples over a
//! [`std::sync::mpsc::Sender`], but the `Progress` fields are private
//! (only `Debug` is implemented) so they can be logged but not meaningfully
//! surfaced through Blazen's typed APIs. Wiring richer progress events
//! would require either upstream exposing public accessors or building a
//! progress facade in this crate that re-formats the `Debug` output --
//! both are out of scope for the initial engine wire-up.

#[cfg(feature = "engine")]
pub mod engine;
mod options;
mod provider;

#[cfg(feature = "engine")]
pub use engine::GeneratedImage;
pub use options::{DiffusionOptions, DiffusionScheduler};
pub use provider::{DiffusionError, DiffusionProvider};
