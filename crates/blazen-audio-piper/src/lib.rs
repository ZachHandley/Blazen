//! **Deprecated** — this crate has been renamed to
//! [`blazen-audio-tts`](https://crates.io/crates/blazen-audio-tts).
//!
//! The original `blazen-audio-piper` was a phase-9 stub targeting the
//! Piper ONNX runtime. The PR6 audio pivot replaces that with the
//! `any-tts` crate (Kokoro-82M, `VibeVoice`, Qwen3-TTS), so the crate has
//! been renamed to reflect its broader scope. This shim re-exports every
//! public symbol from `blazen-audio-tts` under the old `Piper*` names so
//! existing downstream code keeps compiling for one release; new code
//! should depend on `blazen-audio-tts` directly.
//!
//! Each re-export carries a `#[deprecated]` warning pointing at the new
//! name so the upgrade path surfaces during `cargo build`.

#![deprecated(
    since = "0.0.0-dev",
    note = "renamed to `blazen-audio-tts`; this crate is a transitional re-export shim and will be removed in the next release"
)]

#[deprecated(
    since = "0.0.0-dev",
    note = "renamed to `blazen_audio_tts::TtsOptions`"
)]
pub use blazen_audio_tts::TtsOptions as PiperOptions;

#[deprecated(
    since = "0.0.0-dev",
    note = "renamed to `blazen_audio_tts::TtsProvider`"
)]
pub use blazen_audio_tts::TtsProvider as PiperProvider;

#[deprecated(since = "0.0.0-dev", note = "renamed to `blazen_audio_tts::TtsError`")]
pub use blazen_audio_tts::TtsError as PiperError;

// Re-export the model enum + audio struct under their new names too so
// downstream code that already moved to the richer API keeps working
// regardless of which crate they pick up.
pub use blazen_audio_tts::{SynthesizedAudio, TtsModel};
