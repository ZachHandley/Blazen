//! Local LLM backend for Blazen using [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs).
//!
//! This crate wraps the `mistralrs` Rust crate to provide fully local,
//! offline LLM inference with no API keys required.
//!
//! When used through `blazen-llm` with the `mistralrs` feature flag, this
//! crate's [`MistralRsProvider`] will implement `blazen_llm::CompletionModel`.
//!
//! # Feature flags
//!
//! | Feature      | Description                                      |
//! |--------------|--------------------------------------------------|
//! | `engine`     | Links the actual `mistralrs` runtime (CPU)       |
//! | `cuda`       | NVIDIA CUDA GPU acceleration                     |
//! | `metal`      | Apple Silicon GPU acceleration (Metal)            |
//! | `accelerate` | Apple Accelerate framework for CPU math           |
//! | `mkl`        | Intel MKL for CPU math                            |
//! | `flash-attn` | Flash Attention (requires CUDA)                   |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run inference. This keeps workspace builds
//! fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::MistralRsOptions;
pub use provider::{
    ChatMessageInput, ChatRole, InferenceChunk, InferenceChunkStream, InferenceImage,
    InferenceImageSource, InferenceResult, InferenceToolCall, InferenceUsage, MistralRsError,
    MistralRsProvider,
};
