//! Facade over the concrete embedding backend for the current target.
//!
//! - On non-musl targets, re-exports `blazen-embed-fastembed` (fast C++ ONNX Runtime).
//! - On musl targets, re-exports `blazen-embed-tract` (pure-Rust ONNX inference via tract).
//!
//! Downstream code imports ONLY from this crate. The backend selection is invisible.

#[cfg(not(target_env = "musl"))]
pub use blazen_embed_fastembed::{
    FastEmbedError as EmbedError, FastEmbedModel as EmbedModel, FastEmbedOptions as EmbedOptions,
    FastEmbedResponse as EmbedResponse,
};

#[cfg(target_env = "musl")]
pub use blazen_embed_tract::{
    TractEmbedModel as EmbedModel, TractError as EmbedError, TractOptions as EmbedOptions,
    TractResponse as EmbedResponse,
};
