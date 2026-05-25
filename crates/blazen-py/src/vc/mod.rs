//! Voice-conversion (RVC) Python surface.
//!
//! Wraps the [`blazen_audio_vc`] capability crate:
//!
//! - [`PyVcChunk`](chunk::PyVcChunk) — a streamed PCM chunk plus
//!   `is_final` flag, optional measured latency, and sample-rate metadata.
//! - [`PyTargetVoice`](target_voice::PyTargetVoice) — a registered target
//!   speaker descriptor (`id`, optional `label`, native `sample_rate_hz`).
//! - [`PyVcModel`](model::PyVcModel) — opaque handle around a
//!   `dyn VoiceConversionBackend` constructed via the feature-gated
//!   `VcModel.rvc(...)` static factory.
//! - [`PyVcStream`](stream::PyVcStream) — lazy async iterator over
//!   streamed converted PCM chunks, mirroring the pattern used by
//!   [`PyMusicStream`](crate::music::PyMusicStream).
//!
//! The module is gated on the `audio-vc-rvc` feature. With that feature
//! disabled, none of these types compile in or are registered on the
//! Python module.

pub mod chunk;
pub mod model;
pub mod stream;
pub mod target_voice;

pub use chunk::PyVcChunk;
pub use model::PyVcModel;
pub use stream::PyVcStream;
pub use target_voice::PyTargetVoice;
