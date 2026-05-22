//! Shared candle primitives for video diffusion backends.
//!
//! This is the video parallel of [`blazen-audio-core`](../blazen_audio_core)
//! and [`blazen-3d-core`](../blazen_3d_core). It hosts the building
//! blocks that future native video diffusion backends (CogVideoX,
//! HunyuanVideo, Mochi, …) will reuse without pulling each other's
//! backend-specific configs or I/O code.
//!
//! # What lives here
//!
//! - [`temporal_attention`] — a temporal multi-head self-attention
//!   block that mixes information along the time axis while leaving
//!   the spatial layout fused into the batch dimension. This is the
//!   primitive every modern video DiT bolts on top of its spatial
//!   attention pass.
//! - [`video_vae`] — building-block primitives for the latent video
//!   VAE family used by CogVideoX / HunyuanVideo / Mochi. The first
//!   primitive shipped here is
//!   [`TemporalCausalConv3d`](video_vae::TemporalCausalConv3d), a
//!   3D convolution with asymmetric past-only padding along the time
//!   axis (so the decoder remains streaming-safe) and symmetric
//!   padding along height / width.
//! - [`time_embeddings`] — sinusoidal time-step embeddings used by
//!   every diffusion model to condition the denoiser on the current
//!   noise level.
//!
//! # What does NOT live here
//!
//! - Backend-specific configs (CogVideoX / HunyuanVideo / Mochi
//!   hyper-parameters, weight layouts, sampler / scheduler choices,
//!   text encoder wiring, etc.). Those live in the consuming backend
//!   crate.
//! - I/O. Nothing in this crate touches the filesystem, the network,
//!   or any model registry. Weights are loaded by the caller via
//!   `candle_nn::VarBuilder` and handed in.
//! - Backend-specific U-Net / DiT block recipes. Those live in the
//!   consuming backend; this crate exposes pure candle primitives.
//!
//! These primitives are intentionally feature-free; consuming crates
//! gate their own backends as needed.

#![deny(missing_docs)]
// The candle-error / shape-mismatch failure modes of every function in
// this crate are documented inline in the function bodies and through
// the candle docs; repeating an `# Errors` section on every wrapper
// would be pure noise. Same for `# Panics` — none of the public APIs
// here can panic on shape-valid input. `clippy::doc_markdown` fires on
// every math abbreviation (`RoPE`, `MLP`, `VAE`, `B`, `T`, `H`, `W`, …)
// — terms-of-art that look wrong in backticks; suppress the lint at
// the crate level rather than papering over every doc line.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

pub mod temporal_attention;
pub mod time_embeddings;
pub mod video_vae;
