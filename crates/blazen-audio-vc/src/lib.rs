//! Multi-backend voice-conversion surface for Blazen.
//!
//! This crate provides one capability trait —
//! [`VoiceConversionBackend`] — that extends
//! [`blazen_audio::AudioBackend`] with methods for converting a source
//! utterance into the voice of a registered target speaker. The trait
//! covers file-based conversion plus a streaming variant for low-latency
//! real-time use cases, and exposes a small voice-management surface
//! (`list_target_voices` / `register_target_voice`).
//!
//! Backends live under [`backends`] and are gated by Cargo features:
//!
//! - [`backends::rvc`] — reserved for the Retrieval-based Voice
//!   Conversion engine. Currently a placeholder module; the F0
//!   extractor, content encoder, retrieval index, generator, decoding
//!   pipeline, and weights loader land in Wave D.2. Gated by the `rvc`
//!   feature flag.
//!
//! # Feature flags
//!
//! | Feature | Default | Description                                   |
//! |---------|---------|-----------------------------------------------|
//! | `rvc`   | no      | Compiles the [`backends::rvc`] module tree.   |

#![deny(missing_docs)]

pub mod backends;
mod error;
mod traits;

pub use error::VcError;
pub use traits::{TargetVoice, VoiceConversionBackend};

#[cfg(feature = "rvc")]
pub use backends::rvc::RvcBackend;

/// Re-exports of the most common public items for downstream callers
/// (the language-binding crates in particular).
pub mod prelude {
    pub use crate::{TargetVoice, VcError, VoiceConversionBackend};

    #[cfg(feature = "rvc")]
    pub use crate::RvcBackend;
}
