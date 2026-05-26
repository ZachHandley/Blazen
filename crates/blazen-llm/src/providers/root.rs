//! Polymorphic root for the Blazen provider class hierarchy.
//!
//! Every concrete provider (LLM, TTS, STT, Music, VC, 3D, Image-gen,
//! Embedding, Codec) implements [`BaseProvider`] either directly or via
//! one of the capability sub-traits ([`crate::providers::LLMProvider`],
//! [`crate::providers::TtsProvider`], etc. — defined in `mod.rs` /
//! sibling files).
//!
//! The trait is object-safe (`&str` / `Copy` / `&ProviderMetadata`
//! returns only) so consumers can store providers behind
//! `Arc<dyn BaseProvider>` for capability-erased collections.
//!
//! Downstream proprietary engines can implement [`BaseProvider`] from
//! external crates without forking Blazen — this is the foundation of
//! the public-base / private-engine extensibility model.

use std::sync::Arc;

// ---------------------------------------------------------------------------
// CapabilityKind
// ---------------------------------------------------------------------------

/// Coarse categorization of what a provider does.
///
/// Used by [`BaseProvider::capability`] for runtime dispatch / filtering
/// when callers hold an `Arc<dyn BaseProvider>` and need to know whether
/// it's safe to downcast to a specific capability trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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
    /// Vector embedding generation (text / image / multi-modal).
    Embedding,
    /// Neural audio codec (PCM ↔ discrete codebook tokens).
    Codec,
    /// Background removal on existing images.
    BackgroundRemoval,
    /// Video generation (text-to-video, image-to-video).
    Video,
}

impl CapabilityKind {
    /// Stable string identifier for the capability (matches the
    /// `serde(rename_all = "snake_case")` wire form).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            CapabilityKind::Llm => "llm",
            CapabilityKind::Tts => "tts",
            CapabilityKind::Stt => "stt",
            CapabilityKind::Music => "music",
            CapabilityKind::Vc => "vc",
            CapabilityKind::ThreeD => "three_d",
            CapabilityKind::ImageGen => "image_gen",
            CapabilityKind::Embedding => "embedding",
            CapabilityKind::Codec => "codec",
            CapabilityKind::BackgroundRemoval => "background_removal",
            CapabilityKind::Video => "video",
        }
    }
}

impl std::fmt::Display for CapabilityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// ProviderMetadata
// ---------------------------------------------------------------------------

/// Static metadata describing a provider instance.
///
/// Lightweight — owned by the provider, shared by reference through
/// [`BaseProvider::metadata`]. Carries the canonical provider id (used
/// for routing, telemetry, billing keys), capability kind, optional
/// human-readable display name, and optional version pin.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProviderMetadata {
    /// Canonical provider identifier — stable across binding surfaces.
    /// Examples: `"openai"`, `"anthropic"`, `"fal"`, `"piper"`,
    /// `"triposr"`, `"spark-tts"`, `"faster-whisper"`.
    pub provider_id: String,
    /// What this provider does.
    pub capability: CapabilityKind,
    /// Optional human-readable name shown in UIs / logs. Defaults to
    /// `provider_id` if unset.
    pub display_name: Option<String>,
    /// Optional version pin — typically the model id / weights revision
    /// (e.g. `"gpt-4o-2024-08-06"`, `"en_US-amy-medium"`,
    /// `"facebook/musicgen-small"`).
    pub version: Option<String>,
}

impl ProviderMetadata {
    /// Build a minimal metadata record. Use [`Self::with_display_name`] /
    /// [`Self::with_version`] for the optional fields.
    #[must_use]
    pub fn new(provider_id: impl Into<String>, capability: CapabilityKind) -> Self {
        Self {
            provider_id: provider_id.into(),
            capability,
            display_name: None,
            version: None,
        }
    }

    /// Attach a human-readable display name.
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Attach a version pin (model id, weights revision, etc.).
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Returns [`Self::display_name`] when set, otherwise
    /// [`Self::provider_id`].
    #[must_use]
    pub fn display(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.provider_id)
    }
}

// ---------------------------------------------------------------------------
// BaseProvider trait
// ---------------------------------------------------------------------------

/// Polymorphic root for every Blazen provider.
///
/// Object-safe: `Arc<dyn BaseProvider>` is valid. Concrete provider
/// classes implement this directly via the capability sub-traits
/// ([`crate::providers::LLMProvider`], [`crate::providers::TtsProvider`],
/// etc.), each of which is `: BaseProvider + Send + Sync`.
///
/// Method invariants:
/// - [`provider_id`](BaseProvider::provider_id) MUST equal
///   `metadata().provider_id` (cached for cheap dispatch without
///   metadata-allocation overhead).
/// - [`capability`](BaseProvider::capability) MUST equal
///   `metadata().capability`.
/// - Both invariants are upheld by the default convenience impls;
///   downstream implementors should override the convenience methods
///   only when they have a reason to.
pub trait BaseProvider: Send + Sync + std::fmt::Debug {
    /// Static metadata describing this provider instance.
    fn metadata(&self) -> &ProviderMetadata;

    /// Canonical provider identifier. Default delegates to
    /// `metadata().provider_id`.
    fn provider_id(&self) -> &str {
        &self.metadata().provider_id
    }

    /// Capability kind this provider serves. Default delegates to
    /// `metadata().capability`.
    fn capability(&self) -> CapabilityKind {
        self.metadata().capability
    }
}

/// Convenience: every `Arc<P>` where `P: BaseProvider` is itself a
/// `BaseProvider`. Lets downstream code thread `Arc`-wrapped providers
/// through capability-erased pipelines without manual newtype wrappers.
impl<P: BaseProvider + ?Sized> BaseProvider for Arc<P> {
    fn metadata(&self) -> &ProviderMetadata {
        (**self).metadata()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FakeProvider {
        meta: ProviderMetadata,
    }

    impl BaseProvider for FakeProvider {
        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }
    }

    #[test]
    fn capability_kind_round_trips_str() {
        for kind in [
            CapabilityKind::Llm,
            CapabilityKind::Tts,
            CapabilityKind::Stt,
            CapabilityKind::Music,
            CapabilityKind::Vc,
            CapabilityKind::ThreeD,
            CapabilityKind::ImageGen,
            CapabilityKind::Embedding,
            CapabilityKind::Codec,
            CapabilityKind::BackgroundRemoval,
            CapabilityKind::Video,
        ] {
            assert_eq!(kind.to_string(), kind.as_str());
        }
    }

    #[test]
    fn provider_metadata_display_falls_back_to_provider_id() {
        let m = ProviderMetadata::new("openai", CapabilityKind::Llm);
        assert_eq!(m.display(), "openai");
        let m2 = m.with_display_name("OpenAI Chat Completions");
        assert_eq!(m2.display(), "OpenAI Chat Completions");
    }

    #[test]
    fn base_provider_default_delegates_to_metadata() {
        let p = FakeProvider {
            meta: ProviderMetadata::new("piper", CapabilityKind::Tts)
                .with_version("en_US-amy-medium"),
        };
        assert_eq!(p.provider_id(), "piper");
        assert_eq!(p.capability(), CapabilityKind::Tts);
        assert_eq!(p.metadata().version.as_deref(), Some("en_US-amy-medium"));
    }

    #[test]
    fn arc_passthrough_impl_works() {
        let p = Arc::new(FakeProvider {
            meta: ProviderMetadata::new("triposr", CapabilityKind::ThreeD),
        });
        let p_dyn: Arc<dyn BaseProvider> = p.clone();
        assert_eq!(p_dyn.provider_id(), "triposr");
        assert_eq!(p_dyn.capability(), CapabilityKind::ThreeD);
    }

    #[test]
    fn capability_kind_serde_roundtrip() {
        let json = serde_json::to_string(&CapabilityKind::ThreeD).unwrap();
        assert_eq!(json, "\"three_d\"");
        let back: CapabilityKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, CapabilityKind::ThreeD);
    }
}
