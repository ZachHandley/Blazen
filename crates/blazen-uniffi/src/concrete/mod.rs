//! Per-engine `#[uniffi::Object]` provider classes exported across
//! foreign bindings.
//!
//! Mirrors the canonical Rust hierarchy from
//! `blazen_llm::providers::concrete::*`. Each engine is its own
//! `#[uniffi::Object]` wrapping `Arc<blazen_llm::<Engine>Provider>` so
//! UniFFI's foreign-language bindgens emit a real concrete class per
//! engine (Kotlin `class PiperProvider`, Swift `class PiperProvider`,
//! Go `*PiperProvider`, Ruby `Blazen::PiperProvider`).
//!
//! UniFFI doesn't model Rust trait inheritance for `Object` types, so
//! foreign-language consumers see independent per-engine classes whose
//! method names happen to match the [`TtsProvider`] / [`SttProvider`] /
//! [`MusicProvider`] / [`VcProvider`] / [`ThreeDProvider`] capability
//! contracts. Real polymorphic-base inheritance lands separately in
//! the napi-rs / `PyO3` shims (P4.3).
//!
//! ## Why this lives alongside the existing `*Model` types
//!
//! The central `TtsModel` / `SttModel` / `MusicModel` / `VcModel` /
//! `ThreeDModel` opaque types in [`crate::compute`] /
//! [`crate::compute_music`] / [`crate::compute_vc`] stay in place
//! during P4.2 so the cabi/Ruby/Py/Node bindings continue to compile.
//! They deprecate when those bindings switch over in P4.3 / P4.4 / P4.5.

// Capability-gated per-engine modules. Each module declaration is
// feature-gated so a non-all-features build still compiles.

#[cfg(feature = "tts")]
pub mod tts;

#[cfg(feature = "whispercpp")]
pub mod stt;

#[cfg(feature = "audio-music-musicgen")]
pub mod music;

#[cfg(feature = "audio-vc")]
pub mod vc;

#[cfg(feature = "triposr")]
pub mod three_d;

// LLM concrete providers land in a follow-up sub-wave (P4.2.llm).
