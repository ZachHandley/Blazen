//! Local LLM backend for Blazen using [`llama.cpp`](https://github.com/ggerganov/llama.cpp).
//!
//! This crate wraps the `llama.cpp` Rust bindings to provide fully local,
//! offline LLM inference with no API keys required.
//!
//! When used through `blazen-llm` with the `llamacpp` feature flag, this
//! crate's [`LlamaCppProvider`] will implement `blazen_llm::CompletionModel`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                      |
//! |----------|--------------------------------------------------|
//! | `engine` | Links the actual `llama.cpp` runtime             |
//! | `cpu`    | CPU inference (default)                           |
//! | `cuda`   | NVIDIA CUDA GPU acceleration                     |
//! | `metal`  | Apple Silicon GPU acceleration (Metal)            |
//! | `vulkan` | Vulkan GPU acceleration                          |
//! | `rocm`   | AMD `ROCm` GPU acceleration                        |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run inference. This keeps workspace builds
//! fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::LlamaCppOptions;
pub use provider::{LlamaCppError, LlamaCppProvider};
