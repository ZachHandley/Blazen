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
//! The polymorphic capability-base traits in [`bases`] give Kotlin /
//! Swift / Go consumers real interface conformance — a Kotlin
//! `interface TtsProvider`, a Swift `protocol TtsProvider`, a Go
//! `TtsProvider` interface — that every per-engine concrete implements
//! via a sibling `impl <Capability>Provider for <Engine>Provider`
//! block. Those impl blocks land in the per-capability files in
//! P4.2.x.3.{tts,stt,music,vc,three_d} (this sub-task only declares
//! the traits).
//!
//! Bindings-layer hierarchies for PyO3 / napi-rs / Ruby / cabi follow
//! in P4.3+ — they project the same shape using each binding's native
//! inheritance mechanism (Python ABCs, JS class inheritance, Ruby
//! modules, C ABI v-tables in cabi).
//!
//! ## Why this lives alongside the existing `*Model` types
//!
//! The central `TtsModel` / `SttModel` / `MusicModel` / `VcModel` /
//! `ThreeDModel` opaque types in [`crate::compute`] /
//! [`crate::compute_music`] / [`crate::compute_vc`] stay in place
//! during P4.2 so the cabi/Ruby/Py/Node bindings continue to compile.
//! They deprecate when those bindings switch over in P4.3 / P4.4 / P4.5.

// Polymorphic capability-base traits. Unconditional — the file itself
// feature-gates individual traits to match their concretes' gates.
pub mod bases;

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

// 3D module covers both the `TripoSrProvider` (gated on `triposr`) and
// the HTTP-proxy `Compat3dProvider` (gated on `threed-compat-proxy`).
// Activate the module under the umbrella `threed` feature so either
// concrete can land independently. Individual structs inside still
// carry their per-engine `#[cfg(feature = ...)]` gates.
#[cfg(feature = "threed")]
pub mod three_d;

// LLM concrete providers land in a follow-up sub-wave (P4.2.llm).
