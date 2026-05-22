//! Multi-backend 3D-pipeline surface for Blazen.
//!
//! This crate exposes the four capability traits that make up the
//! post-generation 3D pipeline. Each trait targets one stage of the
//! "raw mesh → finished asset" flow and can be implemented
//! independently by different backends (HTTP-proxy services, native
//! engines, etc.) so callers can mix-and-match per stage.
//!
//! # Pipeline stages
//!
//! 1. **Texturizer** — [`Texturizer3dBackend`]. Apply or generate a
//!    texture/material for an existing mesh, optionally producing PBR
//!    material maps (albedo / normal / roughness / metallic).
//! 2. **Rigger** — [`Rigger3dBackend`]. Auto-rig a mesh by placing a
//!    skeletal armature and optionally painting skin weights.
//! 3. **Refiner** — [`Refiner3dBackend`]. Mesh-cleanup passes:
//!    decimation, hole-filling (poisson reconstruction), UV unwrapping,
//!    retopology, smoothing.
//! 4. **Animator** — [`Animator3dBackend`]. Drive a rigged mesh from a
//!    text prompt, motion-capture clip, or driving video.
//!
//! The complementary **generator** stage (text-to-mesh) currently lives
//! on `blazen_llm::compute::traits::ThreeDGeneration`; a future commit
//! may decide whether to migrate that into this crate as a fifth
//! `Generator3dBackend` trait.
//!
//! # Feature flags
//!
//! | Feature        | Default | Description                                       |
//! |----------------|---------|---------------------------------------------------|
//! | `compat-proxy` | no      | Compiles the [`backends::compat`] HTTP-proxy module. |

#![deny(missing_docs)]

pub mod backends;
mod errors;
mod traits;

pub use errors::{Animator3dError, Refiner3dError, Rigger3dError, Texturizer3dError};
pub use traits::animator::{AnimateRequest, AnimateResult, Animator3dBackend};
pub use traits::refiner::{RefineRequest, RefineResult, RefineStats, Refiner3dBackend};
pub use traits::rigger::{RigRequest, RigResult, Rigger3dBackend};
pub use traits::texturizer::{PbrMaps, TexturizeRequest, TexturizeResult, Texturizer3dBackend};

/// Re-exports of the most common public items for downstream callers
/// (the language-binding crates in particular).
pub mod prelude {
    pub use crate::{
        AnimateRequest, AnimateResult, Animator3dBackend, Animator3dError, PbrMaps, RefineRequest,
        RefineResult, RefineStats, Refiner3dBackend, Refiner3dError, RigRequest, RigResult,
        Rigger3dBackend, Rigger3dError, TexturizeRequest, TexturizeResult, Texturizer3dBackend,
        Texturizer3dError,
    };
}
