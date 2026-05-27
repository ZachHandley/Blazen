//! Polymorphic capability-base traits exported across the UniFFI
//! foreign-language surface.
//!
//! These traits give Kotlin / Swift / Go consumers real interface
//! conformance: a Kotlin `interface TtsProvider`, a Swift
//! `protocol TtsProvider`, a Go `type TtsProvider interface`. Per-engine
//! `#[uniffi::Object]` concretes in the sibling per-capability files
//! (`tts.rs`, `stt.rs`, `music.rs`, `vc.rs`, `three_d.rs`) implement the
//! matching trait in subsequent sub-tasks
//! (P4.2.x.3.{tts,stt,music,vc,three_d}) so foreign code can hold a
//! `TtsProvider` reference and dispatch polymorphically across engines.
//!
//! ## Why the capability traits do NOT inherit from [`BaseProvider`]
//!
//! UniFFI 0.31 injects a hidden `uniffi_foreign_handle` default method
//! into every `#[uniffi::export]` trait it processes. If
//! `TtsProvider: BaseProvider`, both traits contribute the same hidden
//! method and rustc raises `E0034: multiple applicable items in scope`
//! at the trait-impl site. The workaround is to make every capability
//! trait independent (`: Send + Sync`) and *duplicate* the
//! `provider_id` / `capability` accessors on each trait body so foreign
//! callers still have a uniform identity surface. The concrete classes
//! implement [`BaseProvider`] as well, so Rust-side `Arc<dyn BaseProvider>`
//! collections remain possible — this only affects the foreign-bindgen
//! shape, where each capability interface is self-contained.
//!
//! ## Why trait methods have NO default implementations
//!
//! UniFFI 0.31's `#[uniffi::export]` on traits rejects `fn` bodies
//! ("uniffi::export'd trait methods can't have a default
//! implementation"). The plan called for `Unsupported`-returning
//! default impls on optional methods (`generate_sfx`, `clone_voice`,
//! `list_target_voices`); those defaults are instead provided on the
//! per-engine concrete `impl <Capability>Provider for <Engine>Provider`
//! blocks landing in the per-capability sub-tasks. The trait body here
//! declares only required signatures.
//!
//! Methods use `&self` receivers (UniFFI trait methods do not support
//! `Arc<Self>` — the existing per-engine `Arc<Self>` methods continue
//! to exist alongside as concrete-class-specific surface).
//!
//! These are *Rust-side polymorphism bases* (`#[uniffi::export]`, NOT
//! `with_foreign`) — they are implementable only by Blazen's per-engine
//! Rust types, not by foreign user code. Foreign-implementable provider
//! callbacks live on [`crate::provider_custom::CustomProvider`].

use async_trait::async_trait;

use crate::errors::BlazenError;

#[cfg(feature = "diffusion")]
use crate::compute::ImageGenResult;
#[cfg(feature = "whispercpp")]
use crate::compute::SttResult;
#[cfg(feature = "threed")]
use crate::compute::ThreeDGenerateResult;
#[cfg(feature = "tts")]
use crate::compute::TtsResult;
#[cfg(feature = "audio-music-musicgen")]
use crate::compute_music::MusicResult;
#[cfg(feature = "audio-vc")]
use crate::compute_vc::{TargetVoice, VcResult};

// 3D post-processing DTOs live alongside the per-engine concretes
// (`concrete::three_d`). They're available whenever the `threed`
// umbrella feature is on (which both `triposr` and
// `threed-compat-proxy` imply).
#[cfg(feature = "threed")]
use crate::concrete::three_d::{
    AnimateRequest, AnimateResult, RefineRequest, RefineResult, RigRequest, RigResult,
    TexturizeRequest, TexturizeResult,
};

use crate::llm::{ModelRequest, ModelResponse};

// ---------------------------------------------------------------------------
// CapabilityKind
// ---------------------------------------------------------------------------

/// Coarse categorization of what a provider does.
///
/// Mirrors [`blazen_llm::providers::root::CapabilityKind`] but is
/// re-declared here so it can carry `#[derive(uniffi::Enum)]` for the
/// FFI surface (the upstream enum uses `serde` derives that are not
/// `uniffi::Enum`-compatible).
#[derive(uniffi::Enum, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CapabilityKind {
    /// Large language model — chat / completion / streaming.
    Llm,
    /// Text-to-speech audio synthesis.
    Tts,
    /// Speech-to-text transcription.
    Stt,
    /// Text-to-music / text-to-sfx audio generation.
    Music,
    /// Voice conversion (source utterance + target voice → re-voiced audio).
    Vc,
    /// 3D mesh generation (image-to-3D, text-to-3D).
    ThreeD,
    /// 2D image generation (text-to-image, image-to-image, upscale).
    ImageGen,
    /// Vector embedding generation.
    Embedding,
    /// Neural audio codec (PCM ↔ discrete codebook tokens).
    Codec,
    /// Background removal on existing images.
    BackgroundRemoval,
    /// Video generation (text-to-video, image-to-video).
    Video,
}

// ---------------------------------------------------------------------------
// BaseProvider
// ---------------------------------------------------------------------------

/// Shared root of the polymorphic provider hierarchy.
///
/// Every concrete `<Engine>Provider` implements [`BaseProvider`] in
/// addition to its capability trait so Rust-side code can collect
/// providers into capability-erased `Arc<dyn BaseProvider>` containers
/// for telemetry, routing, and registry lookups.
#[uniffi::export]
pub trait BaseProvider: Send + Sync {
    /// Stable engine identifier (e.g. `"piper"`, `"kokoro"`, `"musicgen"`).
    fn provider_id(&self) -> String;

    /// Capability bucket this provider serves.
    fn capability(&self) -> CapabilityKind;
}

// ---------------------------------------------------------------------------
// TtsProvider
// ---------------------------------------------------------------------------

/// Text-to-speech capability trait.
///
/// Implemented by every concrete TTS engine
/// (`PiperProvider`, `KokoroProvider`, `VibeVoiceProvider`,
/// `Qwen3TtsProvider`, `SparkTtsProvider`, `BarkProvider`, `F5Provider`,
/// `FalTtsProvider`) so foreign code can dispatch synthesis through a
/// single `TtsProvider` interface regardless of the backing engine.
#[cfg(feature = "tts")]
#[uniffi::export]
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::Tts`] for this trait;
    /// duplicated for binding-surface uniformity).
    fn capability(&self) -> CapabilityKind;

    /// Synthesize `text` into an audio payload.
    async fn synthesize(
        &self,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// SttProvider
// ---------------------------------------------------------------------------

/// Speech-to-text capability trait.
#[cfg(feature = "whispercpp")]
#[uniffi::export]
#[async_trait]
pub trait SttProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::Stt`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Transcribe the audio at `audio_source` (local path or URL) into
    /// text. `language` is an optional ISO-639 hint for the recognizer.
    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// MusicProvider
// ---------------------------------------------------------------------------

/// Text-to-music / text-to-sfx capability trait.
///
/// Engines that only support music generation get the default
/// [`generate_sfx`](MusicProvider::generate_sfx) impl which surfaces
/// `Unsupported`. SFX-only engines override `generate_sfx` and inherit
/// the default `generate_music` (which also surfaces `Unsupported`).
#[cfg(feature = "audio-music-musicgen")]
#[uniffi::export]
#[async_trait]
pub trait MusicProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::Music`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Generate a music clip of approximately `duration_seconds`.
    ///
    /// Engines that do not support music generation surface
    /// `BlazenError::Unsupported` from the per-engine `impl` block
    /// (default trait bodies are not permitted by `#[uniffi::export]`).
    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError>;

    /// Generate a sound-effect clip of approximately `duration_seconds`.
    ///
    /// Engines that do not support SFX generation surface
    /// `BlazenError::Unsupported` from the per-engine `impl` block.
    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// VcProvider
// ---------------------------------------------------------------------------

/// Voice-conversion capability trait.
#[cfg(feature = "audio-vc")]
#[uniffi::export]
#[async_trait]
pub trait VcProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::Vc`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Convert `input_path` (a source utterance) into the voice
    /// identified by `target_voice_id`.
    async fn convert_voice(
        &self,
        input_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError>;

    /// Register a new target voice from a reference audio file.
    ///
    /// Engines that do not support cloning surface
    /// `BlazenError::Unsupported` from the per-engine `impl` block
    /// (default trait bodies are not permitted by `#[uniffi::export]`).
    async fn clone_voice(
        &self,
        voice_id: String,
        reference_path: String,
    ) -> Result<(), BlazenError>;

    /// List the target voices known to this backend.
    ///
    /// Engines without an enumerable voice library return an empty
    /// `Vec` from the per-engine `impl` block.
    async fn list_target_voices(&self) -> Result<Vec<TargetVoice>, BlazenError>;
}

// ---------------------------------------------------------------------------
// ThreeDProvider
// ---------------------------------------------------------------------------

/// Image-to-3D mesh generation **and** post-processing capability trait.
///
/// Implemented by every concrete 3D engine
/// (`TripoSrProvider`, `Compat3dProvider`) so foreign code can dispatch
/// the full 3D pipeline — generation plus the texturize / rig / refine
/// / animate post-processing stages — through a single `ThreeDProvider`
/// interface regardless of the backing engine.
///
/// Engines that only cover one half of the pipeline surface
/// `BlazenError::Unsupported` from the other half (e.g. `TripoSrProvider`
/// is generation-only and returns `Unsupported` from `texturize`; the
/// HTTP-proxy `Compat3dProvider` is post-proc-only and returns
/// `Unsupported` from `generate_from_image`). UniFFI 0.31 forbids
/// default method bodies on exported traits, so the `Unsupported`
/// paths live in the per-engine `impl` blocks rather than the trait body.
#[cfg(feature = "threed")]
#[uniffi::export]
#[async_trait]
pub trait ThreeDProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::ThreeD`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Generate a 3D mesh from a single image.
    ///
    /// Post-proc-only engines (e.g. the HTTP-proxy
    /// [`crate::concrete::three_d::Compat3dProvider`]) surface
    /// [`BlazenError::Unsupported`] from this method.
    async fn generate_from_image(
        &self,
        image_bytes: Vec<u8>,
        mesh_resolution: u32,
    ) -> Result<ThreeDGenerateResult, BlazenError>;

    /// Apply or generate a texture / material for an existing mesh.
    ///
    /// Generation-only engines (e.g.
    /// [`crate::concrete::three_d::TripoSrProvider`]) surface
    /// [`BlazenError::Unsupported`] from this method.
    async fn texturize(
        &self,
        mesh_glb: Vec<u8>,
        request: TexturizeRequest,
    ) -> Result<TexturizeResult, BlazenError>;

    /// Auto-rig a mesh, producing a GLB with skeletal armature
    /// (and optional skin weights) embedded.
    async fn rig(&self, mesh_glb: Vec<u8>, request: RigRequest) -> Result<RigResult, BlazenError>;

    /// Refine a mesh: decimate, fill holes, unwrap UVs, retopologize, smooth.
    async fn refine(
        &self,
        mesh_glb: Vec<u8>,
        request: RefineRequest,
    ) -> Result<RefineResult, BlazenError>;

    /// Animate a rigged mesh from a text prompt, mocap clip, or driving video.
    async fn animate(
        &self,
        rigged_glb: Vec<u8>,
        request: AnimateRequest,
    ) -> Result<AnimateResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// ImageGenProvider
// ---------------------------------------------------------------------------

/// 2D image-generation capability trait.
#[cfg(feature = "diffusion")]
#[uniffi::export]
#[async_trait]
pub trait ImageGenProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::ImageGen`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Generate one or more images from a text prompt.
    async fn generate_image(
        &self,
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<ImageGenResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// EmbeddingProvider
// ---------------------------------------------------------------------------

/// Vector-embedding capability trait.
///
/// `dimensions` is a synchronous metadata accessor — `#[async_trait]`
/// (used here for `embed`) tolerates sync methods alongside async
/// methods in the same trait body, matching the pattern used by
/// [`crate::provider_custom::CustomProvider`].
#[uniffi::export]
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::Embedding`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Compute embedding vectors for each input string.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError>;

    /// Return the dimensionality of vectors produced by [`embed`](Self::embed).
    fn dimensions(&self) -> u32;
}

// ---------------------------------------------------------------------------
// LlmProvider
// ---------------------------------------------------------------------------

/// Large-language-model capability trait — non-streaming completion.
///
/// Streaming completion uses the existing
/// [`crate::provider_custom::CustomProvider::stream`] surface
/// (foreign-implementable callback shape) and is not duplicated here.
#[uniffi::export]
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stable engine identifier (mirrors [`BaseProvider::provider_id`]).
    fn provider_id(&self) -> String;

    /// Capability bucket (always [`CapabilityKind::Llm`] for this trait).
    fn capability(&self) -> CapabilityKind;

    /// Perform a non-streaming chat / completion request.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError>;
}
