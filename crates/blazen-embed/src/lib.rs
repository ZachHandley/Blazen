//! Facade over the concrete embedding backend for the current target.
//!
//! - On non-musl, non-Intel-macOS targets, re-exports `blazen-embed-fastembed`
//!   (fast C++ ONNX Runtime).
//! - On musl Linux and Intel macOS (`x86_64-apple-darwin`), re-exports
//!   `blazen-embed-tract` (pure-Rust ONNX inference via tract). ORT has no
//!   prebuilt binaries for either platform.
//!
//! Downstream code imports ONLY from this crate. The backend selection is invisible.

#[cfg(not(any(target_env = "musl", all(target_os = "macos", target_arch = "x86_64"))))]
pub use blazen_embed_fastembed::{
    FastEmbedError as EmbedError, FastEmbedModel as EmbedModel, FastEmbedOptions as EmbedOptions,
    FastEmbedResponse as EmbedResponse,
};

#[cfg(any(target_env = "musl", all(target_os = "macos", target_arch = "x86_64")))]
pub use blazen_embed_tract::{
    TractEmbedModel as EmbedModel, TractError as EmbedError, TractOptions as EmbedOptions,
    TractResponse as EmbedResponse,
};
