//! Capability traits for the four post-generation 3D pipeline stages.
//!
//! Each submodule defines one trait plus its associated request/result
//! types. Backends implement whichever subset of these traits they
//! support — there is no requirement that a single backend cover the
//! whole pipeline.

pub mod animator;
pub mod refiner;
pub mod rigger;
pub mod texturizer;
