//! Facade over the concrete embedding backend for the current target.
//!
//! Backend selection priority:
//! 1. `--features embed-fastembed` forces the fastembed (ORT) backend.
//! 2. `--features embed-tract` forces the pure-Rust tract backend.
//! 3. Otherwise the target-cfg in `Cargo.toml` picks the default:
//!    - fastembed on `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`,
//!      `x86_64-unknown-linux-gnu` (the only triples with ORT prebuilts).
//!    - tract on every other target (aarch64-linux, musl, all wasm,
//!      aarch64-windows, AND x86_64-apple-darwin — pyke dropped the Intel-mac
//!      ORT prebuilt, so the facade routes Intel macs to tract here).
//!
//! Downstream code imports ONLY from this crate. The backend selection is invisible.
//!
//! `embed-fastembed` and `embed-tract` are documented as mutually exclusive.
//! If both are set, fastembed wins (the re-export below picks it first); this
//! is a soft guarantee — prefer enabling exactly one.

// --- Forced fastembed (feature wins over target-cfg default) ---
#[cfg(feature = "embed-fastembed")]
pub use blazen_embed_fastembed::{
    FastEmbedError as EmbedError, FastEmbedModel as EmbedModel, FastEmbedOptions as EmbedOptions,
    FastEmbedResponse as EmbedResponse,
};

// --- Forced tract (only when fastembed is NOT also forced) ---
#[cfg(all(feature = "embed-tract", not(feature = "embed-fastembed")))]
pub use blazen_embed_tract::{
    TractEmbedModel as EmbedModel, TractError as EmbedError, TractOptions as EmbedOptions,
    TractResponse as EmbedResponse,
};

// --- Default fastembed on ORT-supported triples (no override feature set) ---
#[cfg(all(
    not(feature = "embed-fastembed"),
    not(feature = "embed-tract"),
    not(target_family = "wasm"),
    not(target_env = "musl"),
    any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "windows"),
        all(target_arch = "aarch64", target_os = "macos"),
    )
))]
pub use blazen_embed_fastembed::{
    FastEmbedError as EmbedError, FastEmbedModel as EmbedModel, FastEmbedOptions as EmbedOptions,
    FastEmbedResponse as EmbedResponse,
};

// --- Default tract on every other target (no override feature set) ---
#[cfg(all(
    not(feature = "embed-fastembed"),
    not(feature = "embed-tract"),
    any(
        target_family = "wasm",
        target_env = "musl",
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "windows"),
        all(target_arch = "x86_64", target_os = "macos"),
    )
))]
pub use blazen_embed_tract::{
    TractEmbedModel as EmbedModel, TractError as EmbedError, TractOptions as EmbedOptions,
    TractResponse as EmbedResponse,
};
