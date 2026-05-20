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
//!
//! # GPU acceleration
//!
//! `cuda`, `metal`, `accelerate`, and `mkl` are marker features at this
//! library level -- they do **not** force-enable the matching `mistralrs/*`
//! feature, because doing so would break `cargo --all-features` on hosts
//! that lack the corresponding toolchain (CUDA on Linux without NVIDIA,
//! Metal on Linux, etc.). To actually engage GPU acceleration, declare
//! `mistralrs` directly in your binary crate with the desired feature; the
//! resulting Cargo feature unification will be picked up automatically.
//!
//! `flash-attn` is the one exception: it really does forward to
//! `mistralrs/flash-attn`, which builds FlashAttention-2 kernels via nvcc
//! at compile time. Hardware / toolchain requirements:
//!
//! | Requirement                | Minimum                                  |
//! |----------------------------|------------------------------------------|
//! | CUDA toolkit               | 11.8 or newer                             |
//! | GPU compute capability     | `sm_80` (Ampere) or newer; Hopper / Ada ok |
//! | Unsupported GPUs           | Turing (`sm_75`) and older -- use eager attn |
//!
//! CI cannot validate this end-to-end (the live-models runner is CPU-only),
//! so flash-attn builds must be exercised manually on GPU hosts.

mod options;
mod provider;

pub use options::MistralRsOptions;
pub use provider::{
    ChatMessageInput, ChatRole, InferenceChunk, InferenceChunkStream, InferenceImage,
    InferenceImageSource, InferenceResult, InferenceToolCall, InferenceUsage, MistralRsError,
    MistralRsProvider, MountedAdapter,
};
