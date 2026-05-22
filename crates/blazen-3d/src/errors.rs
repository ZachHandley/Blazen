//! Error types for the four 3D-pipeline capability traits.
//!
//! Each trait gets its own parallel error enum with the same variant
//! set, so a backend that implements multiple stages can map
//! engine-specific failures into the stage's error type consistently.
//! Backend-specific error types (HTTP-client errors, ONNX-runtime
//! errors, etc.) are flattened into one of these variants with their
//! own `Display` text preserved, so the public surface does not leak
//! engine types.

use thiserror::Error;

/// Errors returned by [`crate::Texturizer3dBackend`] implementations.
#[derive(Debug, Error)]
pub enum Texturizer3dError {
    /// An I/O failure while reading a mesh file, reference image, or
    /// model file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The selected backend is not available in this build (e.g. the
    /// `compat-proxy` feature is disabled but the caller requested
    /// the HTTP-proxy backend).
    #[error("texturizer engine not available: {0}")]
    EngineNotAvailable(String),

    /// The caller-supplied input was malformed — invalid mesh bytes,
    /// unsupported container format, malformed request fields, etc.
    #[error("invalid texturize input: {0}")]
    InvalidInput(String),

    /// The backend reported a runtime failure (HTTP error, inference
    /// error, etc.).
    #[error("texturize backend error: {0}")]
    Backend(String),

    /// The capability requested is not supported by the active
    /// backend (e.g. PBR maps from an albedo-only texturizer).
    #[error("texturize capability not supported: {0}")]
    Unsupported(String),
}

/// Errors returned by [`crate::Rigger3dBackend`] implementations.
#[derive(Debug, Error)]
pub enum Rigger3dError {
    /// An I/O failure while reading a mesh file or model file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The selected backend is not available in this build.
    #[error("rigger engine not available: {0}")]
    EngineNotAvailable(String),

    /// The caller-supplied input was malformed — invalid mesh bytes,
    /// unrecognised rig template, etc.
    #[error("invalid rig input: {0}")]
    InvalidInput(String),

    /// The backend reported a runtime failure.
    #[error("rig backend error: {0}")]
    Backend(String),

    /// The capability requested is not supported by the active
    /// backend (e.g. quadruped template on a humanoid-only rigger).
    #[error("rig capability not supported: {0}")]
    Unsupported(String),
}

/// Errors returned by [`crate::Refiner3dBackend`] implementations.
#[derive(Debug, Error)]
pub enum Refiner3dError {
    /// An I/O failure while reading a mesh file or model file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The selected backend is not available in this build.
    #[error("refiner engine not available: {0}")]
    EngineNotAvailable(String),

    /// The caller-supplied input was malformed — invalid mesh bytes,
    /// malformed request fields, etc.
    #[error("invalid refine input: {0}")]
    InvalidInput(String),

    /// The backend reported a runtime failure.
    #[error("refine backend error: {0}")]
    Backend(String),

    /// The capability requested is not supported by the active
    /// backend (e.g. retopology on a decimation-only refiner).
    #[error("refine capability not supported: {0}")]
    Unsupported(String),
}

/// Errors returned by [`crate::Animator3dBackend`] implementations.
#[derive(Debug, Error)]
pub enum Animator3dError {
    /// An I/O failure while reading a rigged mesh, driving video,
    /// BVH clip, or model file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The selected backend is not available in this build.
    #[error("animator engine not available: {0}")]
    EngineNotAvailable(String),

    /// The caller-supplied input was malformed — unrigged mesh,
    /// invalid video/BVH bytes, conflicting request fields, etc.
    #[error("invalid animate input: {0}")]
    InvalidInput(String),

    /// The backend reported a runtime failure.
    #[error("animate backend error: {0}")]
    Backend(String),

    /// The capability requested is not supported by the active
    /// backend (e.g. video-driven motion on a text-only animator).
    #[error("animate capability not supported: {0}")]
    Unsupported(String),
}
