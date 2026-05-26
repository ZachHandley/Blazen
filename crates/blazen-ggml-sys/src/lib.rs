//! Vendored ggml C library — link-only crate.
//!
//! This crate has no Rust API. Its sole purpose is to compile and link a
//! single canonical copy of `ggml-org/ggml` so that the downstream
//! `-sys` crates that previously each vendored their own copy can share
//! one instance. Cargo's `links = "ggml"` attribute on the package
//! enforces uniqueness in the dependency graph.
//!
//! ## Consumers
//!
//! - `llama-cpp-sys-2` (via its existing `system-ggml-static` feature)
//! - `blazen-whisper-sys` (Blazen fork of `whisper-rs-sys` with a
//!   `system-ggml` feature added — patched into the workspace via
//!   `[patch.crates-io]`)
//! - `blazen-diffusion-sys` (same surgery on `diffusion-rs-sys`)
//!
//! ## How consumers find this build
//!
//! `build.rs` runs cmake's install step so the artifacts land at
//! `${OUT_DIR}/install/{lib,include,lib/cmake/ggml}`. We expose:
//!
//! - `cargo:include={OUT_DIR}/install/include`
//! - `cargo:lib={OUT_DIR}/install/lib`
//! - `cargo:prefix={OUT_DIR}/install`
//!
//! Downstream build scripts read these via the cargo-set env vars
//! `DEP_GGML_INCLUDE`, `DEP_GGML_LIB`, and `DEP_GGML_PREFIX`. Each
//! consumer passes `CMAKE_PREFIX_PATH=$DEP_GGML_PREFIX` to its own
//! cmake invocation so its `find_package(ggml REQUIRED)` resolves
//! against our build instead of fetching/building its own.
//!
//! ## License
//!
//! Vendored `ggml-org/ggml` source is MIT-licensed; see
//! `vendor/ggml/LICENSE`.

#![no_std]
