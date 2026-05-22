//! 3D backend implementations live here.
//!
//! Each backend is gated by its own Cargo feature so a build only
//! pulls in the engine, runtime, and model-loader code it actually
//! uses. The capability traits themselves (
//! [`crate::Texturizer3dBackend`], [`crate::Rigger3dBackend`],
//! [`crate::Refiner3dBackend`], [`crate::Animator3dBackend`]) live in
//! `crate::traits` and are always available.

#[cfg(feature = "compat-proxy")]
pub mod compat;
