//! Local LLM backend for Blazen using [`candle`](https://github.com/huggingface/candle).
//!
//! This crate wraps the `candle` Rust framework to provide fully local,
//! offline LLM inference with no API keys required.
//!
//! When used through `blazen-llm` with the `candle-llm` feature flag, this
//! crate's [`CandleLlmProvider`] will implement `blazen_llm::CompletionModel`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                      |
//! |----------|--------------------------------------------------|
//! | `engine` | Links the actual `candle` runtime                |
//! | `cpu`    | CPU inference (default)                           |
//! | `cuda`   | NVIDIA CUDA GPU acceleration                     |
//! | `metal`  | Apple Silicon GPU acceleration (Metal)            |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run inference. This keeps workspace builds
//! fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::CandleLlmOptions;
pub use provider::{CandleLlmError, CandleLlmProvider};
