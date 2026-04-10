//! Local embedding backend for Blazen using [`candle`](https://github.com/huggingface/candle).
//!
//! This crate wraps `HuggingFace`'s Candle ML framework to provide fully local,
//! offline text embedding with no API keys required.
//!
//! When used through `blazen-llm` with the `candle-embed` feature flag, this
//! crate's [`CandleEmbedModel`] will implement `blazen_llm::EmbeddingModel`.
//!
//! # Feature flags
//!
//! | Feature      | Description                                      |
//! |--------------|--------------------------------------------------|
//! | `engine`     | Links the actual candle runtime                  |
//! | `cpu`        | CPU inference (default)                           |
//! | `cuda`       | NVIDIA CUDA GPU acceleration                     |
//! | `metal`      | Apple Silicon GPU acceleration (Metal)            |
//! | `accelerate` | Apple Accelerate framework                        |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run embedding. This keeps workspace builds
//! fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::CandleEmbedOptions;
pub use provider::{CandleEmbedError, CandleEmbedModel};
