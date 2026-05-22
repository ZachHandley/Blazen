//! Shared candle primitives for 3D reconstruction / generation backends.
//!
//! This is the parallel of [`blazen-audio-core`](../blazen_audio_core)
//! for the 3D modality. It hosts the building blocks that future native
//! 3D backends (TripoSR, TRELLIS, Hunyuan3D, …) will reuse without
//! pulling each other's backend-specific configs or I/O code.
//!
//! # What lives here
//!
//! - [`triplane`] — [`Triplane`](triplane::Triplane) feature container
//!   (three orthogonal 2D feature planes covering the XY / YZ / XZ
//!   axes) plus per-point sampling that returns the per-point feature
//!   vector consumed by an NeRF-style MLP head.
//! - [`marching_cubes`] — thin candle-friendly wrapper around the
//!   `isosurface` crate's classic marching-cubes implementation;
//!   converts a 3D scalar field (SDF or occupancy) into a triangle
//!   mesh.
//! - [`gsplat`] — public types for the Gaussian-splat representation
//!   ([`GaussianSplat`](gsplat::GaussianSplat),
//!   [`CameraIntrinsics`](gsplat::CameraIntrinsics),
//!   [`CameraExtrinsics`](gsplat::CameraExtrinsics)) plus the
//!   [`GsplatRasterizer`](gsplat::GsplatRasterizer) entry point.
//!   The rasterizer itself is a scaffolding stub until the native
//!   tile-based renderer lands.
//! - [`image_encoders`] — thin candle-transformers DINOv2 wrapper
//!   ([`DinoV2Encoder`](image_encoders::DinoV2Encoder)) used by the
//!   first 3D port (TripoSR) for image conditioning.
//!
//! # What does NOT live here
//!
//! - Backend-specific configs (TripoSR / TRELLIS / Hunyuan hyper-
//!   parameters, weight layouts, etc.). Those live in the consuming
//!   backend crate.
//! - I/O. Nothing in this crate touches the filesystem, the network,
//!   or any model registry. Weights are loaded by the caller via
//!   `candle_nn::VarBuilder` and handed in.
//! - Backend-specific MLP heads, transformer blocks, or U-Nets. Those
//!   live in the consuming backend; this crate exposes pure candle
//!   primitives and a couple of thin wrappers around upstream crates.
//!
//! These primitives are intentionally feature-free; consuming crates
//! gate their own backends as needed.

#![deny(missing_docs)]
// The candle-error / shape-mismatch failure modes of every function in
// this crate are documented inline in the function bodies and through
// the candle docs; repeating an `# Errors` section on every wrapper
// would be pure noise. Same for `# Panics` — none of the public APIs
// here can panic on shape-valid input. `clippy::doc_markdown` fires on
// every math abbreviation (`SDF`, `MLP`, `XY`, `YZ`, `XZ`, `RGB`, …) —
// terms-of-art that look wrong in backticks; suppress the lint at the
// crate level rather than papering over every doc line.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

pub mod gsplat;
pub mod image_encoders;
pub mod marching_cubes;
pub mod triplane;
