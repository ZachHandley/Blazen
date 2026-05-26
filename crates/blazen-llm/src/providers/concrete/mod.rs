//! Concrete per-engine provider classes — the consumer-facing
//! `<Engine>Provider` newtypes that implement the polymorphic
//! [`crate::providers::BaseProvider`] root + their respective
//! capability sub-trait
//! ([`crate::providers::TtsProvider`],
//! [`crate::providers::SttProvider`], etc.).
//!
//! Each concrete `<Engine>Provider`:
//! - Holds an inner backend handle (typed `Arc<<Engine>Backend>` or the
//!   type-erased `Arc<Dyn*Provider>` from the audio sub-crates).
//! - Stamps a [`crate::providers::ProviderMetadata`] at construction
//!   time so the [`crate::providers::BaseProvider::metadata`] /
//!   `provider_id` / `capability` lookups are O(1).
//! - Translates the capability-trait request/result DTOs
//!   ([`crate::compute::requests::SpeechRequest`],
//!   [`crate::compute::results::AudioResult`], etc.) into the
//!   backend's native shape via the existing
//!   `Dyn<Capability>Provider` adapters in
//!   [`crate::backends`].
//!
//! These types are the public consumer API and the foundation for the
//! binding-side `PiperProvider` / `SparkTtsProvider` / `TripoSrProvider`
//! / etc. exported via napi-rs / `PyO3` / `UniFFI` / cabi / Ruby / WASM.

// Capability-grouped modules — one per capability, gated by the matching
// upstream feature so a non-`-all-features` build still compiles.

#[cfg(feature = "audio-tts")]
pub mod tts;

#[cfg(feature = "whispercpp")]
pub mod stt;

#[cfg(feature = "audio-music-musicgen")]
pub mod music;

#[cfg(feature = "audio-vc")]
pub mod vc;

#[cfg(feature = "triposr")]
pub mod three_d;

// LLM provider impls live alongside their existing provider definitions
// in `crate::providers::{openai,anthropic,...}` — added by sub-wave
// P4.1.i.
