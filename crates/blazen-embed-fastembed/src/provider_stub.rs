//! Stub `FastEmbedModel` impl for targets where the ORT (`ort-sys`) prebuilt
//! is unavailable.
//!
//! As of `ort-sys` 2.0.0-rc.12, pyke dropped `x86_64-apple-darwin` from
//! their prebuilt distribution matrix (only `aarch64-apple-darwin` remains
//! for Apple targets). Compiling fastembed on Intel mac would require
//! either source-building ONNX Runtime or shipping a custom libonnxruntime
//! at distribution time — neither acceptable for the `local-all` default.
//!
//! On those targets this crate compiles to a stub: the public types still
//! exist (so consumers across `blazen-llm`, `blazen-py`, `blazen-node`,
//! `blazen-uniffi`, etc. need no `#[cfg]` gates), but `from_options`
//! returns [`FastEmbedError::UnsupportedTarget`] at runtime. Consumers
//! should route through `blazen-embed`'s facade, which auto-selects the
//! `blazen-embed-tract` (pure-Rust ONNX) backend on these targets.

use std::fmt;

use crate::FastEmbedOptions;

/// Error type for fastembed operations on the stub target.
///
/// The variants mirror the real [`provider::FastEmbedError`] so consumer
/// `match` arms compile unchanged; only [`Self::UnsupportedTarget`] is
/// ever actually returned by the stub.
#[derive(Debug)]
pub enum FastEmbedError {
    /// This target does not have a pyke `ort-sys` prebuilt. Use the
    /// `blazen-embed-tract` backend via the `blazen-embed` facade.
    UnsupportedTarget,
    /// The model name was not recognised by fastembed. (Stub: unused.)
    UnknownModel(String),
    /// The fastembed model failed to initialise. (Stub: unused.)
    Init(String),
    /// An embedding operation failed. (Stub: unused.)
    Embed(String),
    /// The internal mutex was poisoned. (Stub: unused.)
    MutexPoisoned(String),
    /// A blocking task panicked. (Stub: unused.)
    TaskPanicked(String),
}

impl fmt::Display for FastEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTarget => write!(
                f,
                "fastembed backend unavailable on this target (no ort-sys \
                 prebuilt); use the tract backend via blazen-embed"
            ),
            Self::UnknownModel(msg) => write!(f, "unknown fastembed model: {msg}"),
            Self::Init(msg) => write!(f, "fastembed init failed: {msg}"),
            Self::Embed(msg) => write!(f, "fastembed embed failed: {msg}"),
            Self::MutexPoisoned(msg) => write!(f, "fastembed mutex poisoned: {msg}"),
            Self::TaskPanicked(msg) => write!(f, "fastembed blocking task panicked: {msg}"),
        }
    }
}

impl std::error::Error for FastEmbedError {}

/// Response from a fastembed embedding operation. (Stub: never constructed.)
#[derive(Debug, Clone)]
pub struct FastEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
}

/// Stub embedding model — always errors on construction with
/// [`FastEmbedError::UnsupportedTarget`].
pub struct FastEmbedModel {
    _never: std::convert::Infallible,
}

impl FastEmbedModel {
    /// Stub constructor — always returns [`FastEmbedError::UnsupportedTarget`].
    ///
    /// # Errors
    ///
    /// Always returns [`FastEmbedError::UnsupportedTarget`].
    pub fn from_options(_opts: FastEmbedOptions) -> Result<Self, FastEmbedError> {
        tracing::warn!(
            "FastEmbedModel::from_options called on a target without an \
             ort-sys prebuilt (e.g. x86_64-apple-darwin) — returning \
             UnsupportedTarget. Switch to blazen-embed-tract."
        );
        Err(FastEmbedError::UnsupportedTarget)
    }

    /// Stub accessor — unreachable (model instance can't be constructed).
    #[must_use]
    pub fn model_id(&self) -> &str {
        match self._never {}
    }

    /// Stub accessor — unreachable.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        match self._never {}
    }

    /// Stub embed — unreachable.
    ///
    /// # Errors
    ///
    /// Unreachable; the only way to acquire a `FastEmbedModel` value on a
    /// stub target is through `from_options`, which always errors.
    pub async fn embed(&self, _texts: &[String]) -> Result<FastEmbedResponse, FastEmbedError> {
        match self._never {}
    }
}
