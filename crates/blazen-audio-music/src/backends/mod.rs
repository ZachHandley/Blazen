//! Concrete music + SFX backend implementations.
//!
//! Each backend lives behind a feature flag. With no features enabled the
//! crate compiles as a thin no-op layer that surfaces every entry point as
//! [`MusicError::EngineNotAvailable`] or [`MusicError::NotYetImplemented`]
//! depending on the underlying engine's status.
//!
//! [`MusicError::EngineNotAvailable`]: crate::MusicError::EngineNotAvailable
//! [`MusicError::NotYetImplemented`]: crate::MusicError::NotYetImplemented

#[cfg(feature = "audiogen")]
pub mod audiogen;
pub mod musicgen;
pub mod stable_audio;
#[cfg(any(feature = "musicgen", feature = "stable-audio"))]
pub mod wav;
