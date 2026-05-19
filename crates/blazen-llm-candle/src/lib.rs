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

#[cfg(feature = "engine")]
pub mod lora;
#[cfg(feature = "engine")]
pub mod lora_backend;
mod options;
mod provider;
#[cfg(feature = "engine")]
pub mod safetensors_engine;

pub use options::CandleLlmOptions;
pub use provider::{
    CandleInferenceResult, CandleLlmError, CandleLlmProvider, MountedAdapterRecord,
};
#[cfg(feature = "engine")]
pub use safetensors_engine::SafetensorsEngine;
