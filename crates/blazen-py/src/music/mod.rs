//! Music + sound-effect (SFX) generation Python surface.
//!
//! Wraps the [`blazen_audio_music`] capability crate:
//!
//! - [`PyMusicChunk`](chunk::PyMusicChunk) — a streamed PCM chunk plus
//!   `is_final` flag and (for the non-streaming path) the producing
//!   backend's sample rate.
//! - [`PyMusicModel`](model::PyMusicModel) — opaque handle around a
//!   `dyn MusicBackend` constructed via feature-gated `MusicModel.musicgen()`
//!   / `MusicModel.audiogen()` / `MusicModel.stable_audio()` static factories.
//! - [`PyMusicStream`](stream::PyMusicStream) — lazy async iterator over
//!   streamed [`MusicChunk`](blazen_audio_music::MusicChunk) emissions,
//!   mirroring the pattern used by `PyLazyCompletionStream`.
//!
//! The module is gated on any of the three backend features
//! (`audio-music-musicgen`, `audio-music-audiogen`, `audio-music-stable-audio`).
//! With none enabled, none of these types compile in or are registered on
//! the Python module.

pub mod chunk;
pub mod model;
pub mod stream;

pub use chunk::PyMusicChunk;
pub use model::PyMusicModel;
pub use stream::PyMusicStream;
