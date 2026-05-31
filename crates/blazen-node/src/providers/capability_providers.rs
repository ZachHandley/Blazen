//! Per-capability provider classes for user subclassing.
//!
//! Each class exposes one media-generation capability. Users subclass
//! in TypeScript/JavaScript and override the async methods to plug in
//! their own backends.
//!
//! All capability methods accept a `serde_json::Value` (plain JS
//! object) matching the shape of the corresponding Blazen request type
//! and return a `serde_json::Value` matching the result type. This
//! avoids type-conversion friction at the napi boundary while keeping
//! the API self-documenting through TypeScript type annotations.

use napi_derive::napi;

use blazen_llm::providers::{CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// CapabilityKind
// ---------------------------------------------------------------------------

/// The capability a provider serves. Mirrors
/// [`blazen_llm::providers::CapabilityKind`].
#[napi(string_enum, js_name = "CapabilityKind")]
#[derive(Debug, Clone, Copy)]
pub enum JsCapabilityKind {
    /// Large language model — chat / completion / streaming.
    Llm,
    /// Text-to-speech audio synthesis.
    Tts,
    /// Speech-to-text transcription.
    Stt,
    /// Text-to-music / text-to-sfx audio generation.
    Music,
    /// Voice conversion.
    Vc,
    /// 3D mesh generation.
    ThreeD,
    /// 2D image generation.
    ImageGen,
    /// Vector embedding generation.
    Embedding,
    /// Neural audio codec.
    Codec,
    /// Background removal on existing images.
    BackgroundRemoval,
    /// Video generation.
    Video,
}

impl From<JsCapabilityKind> for CapabilityKind {
    fn from(k: JsCapabilityKind) -> Self {
        match k {
            JsCapabilityKind::Llm => CapabilityKind::Llm,
            JsCapabilityKind::Tts => CapabilityKind::Tts,
            JsCapabilityKind::Stt => CapabilityKind::Stt,
            JsCapabilityKind::Music => CapabilityKind::Music,
            JsCapabilityKind::Vc => CapabilityKind::Vc,
            JsCapabilityKind::ThreeD => CapabilityKind::ThreeD,
            JsCapabilityKind::ImageGen => CapabilityKind::ImageGen,
            JsCapabilityKind::Embedding => CapabilityKind::Embedding,
            JsCapabilityKind::Codec => CapabilityKind::Codec,
            JsCapabilityKind::BackgroundRemoval => CapabilityKind::BackgroundRemoval,
            JsCapabilityKind::Video => CapabilityKind::Video,
        }
    }
}

impl From<CapabilityKind> for JsCapabilityKind {
    fn from(k: CapabilityKind) -> Self {
        match k {
            CapabilityKind::Llm => JsCapabilityKind::Llm,
            CapabilityKind::Tts => JsCapabilityKind::Tts,
            CapabilityKind::Stt => JsCapabilityKind::Stt,
            CapabilityKind::Music => JsCapabilityKind::Music,
            CapabilityKind::Vc => JsCapabilityKind::Vc,
            CapabilityKind::ThreeD => JsCapabilityKind::ThreeD,
            CapabilityKind::ImageGen => JsCapabilityKind::ImageGen,
            CapabilityKind::Embedding => JsCapabilityKind::Embedding,
            CapabilityKind::Codec => JsCapabilityKind::Codec,
            CapabilityKind::BackgroundRemoval => JsCapabilityKind::BackgroundRemoval,
            CapabilityKind::Video => JsCapabilityKind::Video,
        }
    }
}

// ---------------------------------------------------------------------------
// ProviderMetadata
// ---------------------------------------------------------------------------

/// Static metadata describing a provider instance. Mirrors
/// [`blazen_llm::providers::ProviderMetadata`].
#[napi(object, js_name = "ProviderMetadata")]
pub struct JsProviderMetadata {
    /// Canonical provider identifier — stable across binding surfaces
    /// (e.g. `"openai"`, `"fal"`, `"spark-tts"`).
    pub provider_id: String,
    /// What this provider does.
    pub capability: JsCapabilityKind,
    /// Optional human-readable name shown in UIs / logs. Defaults to
    /// `providerId` when unset.
    pub display_name: Option<String>,
    /// Optional version pin — typically the model id / weights revision.
    pub version: Option<String>,
}

impl From<JsProviderMetadata> for ProviderMetadata {
    fn from(m: JsProviderMetadata) -> Self {
        let mut meta = ProviderMetadata::new(m.provider_id, m.capability.into());
        if let Some(name) = m.display_name {
            meta = meta.with_display_name(name);
        }
        if let Some(version) = m.version {
            meta = meta.with_version(version);
        }
        meta
    }
}

impl From<ProviderMetadata> for JsProviderMetadata {
    fn from(m: ProviderMetadata) -> Self {
        Self {
            provider_id: m.provider_id,
            capability: m.capability.into(),
            display_name: m.display_name,
            version: m.version,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared config object
// ---------------------------------------------------------------------------

/// Configuration passed to any capability provider constructor.
#[napi(object)]
pub struct CapabilityProviderConfig {
    /// Short identifier for this provider (e.g. `"elevenlabs"`, `"fal"`).
    pub provider_id: String,
    /// Optional base URL for HTTP-based providers.
    pub base_url: Option<String>,
    /// Optional estimated memory footprint in bytes when loaded
    /// (host RAM if the provider targets the CPU, GPU VRAM otherwise).
    pub memory_estimate_bytes: Option<u32>,
}

// ---------------------------------------------------------------------------
// Helper macro
// ---------------------------------------------------------------------------

/// Declares a napi class with a `ProviderConfig` field, constructor,
/// common getters, and capability-specific async methods that throw
/// `napi::Error` by default (users override them in JS/TS subclasses).
///
/// `$extra_methods` is a brace-delimited block of additional method
/// items spliced into the `#[napi] impl` alongside the constructor
/// and getters.
macro_rules! capability_provider {
    (
        $(#[$meta:meta])*
        $js_name:literal, $struct_name:ident,
        extra { $($extra:tt)* }
    ) => {
        $(#[$meta])*
        #[napi(js_name = $js_name)]
        pub struct $struct_name {
            config: blazen_llm::ProviderConfig,
        }

        #[napi]
        #[allow(
            clippy::must_use_candidate,
            clippy::missing_errors_doc,
            clippy::needless_pass_by_value,
            clippy::unused_async,
        )]
        impl $struct_name {
            // -- constructor -------------------------------------------------

            #[napi(constructor)]
            pub fn new(config: CapabilityProviderConfig) -> Self {
                Self {
                    config: blazen_llm::ProviderConfig {
                        provider_id: Some(config.provider_id),
                        base_url: config.base_url,
                        memory_estimate_bytes: config.memory_estimate_bytes.map(u64::from),
                        ..Default::default()
                    },
                }
            }

            // -- common getters ----------------------------------------------

            /// The provider identifier.
            #[napi(getter)]
            pub fn provider_id(&self) -> Option<String> {
                self.config.provider_id.clone()
            }

            /// The base URL, if set.
            #[napi(getter)]
            pub fn base_url(&self) -> Option<String> {
                self.config.base_url.clone()
            }

            /// Estimated memory footprint in bytes (host RAM if the
            /// provider targets the CPU, GPU VRAM otherwise), if set.
            #[napi(getter)]
            pub fn memory_estimate_bytes(&self) -> Option<u32> {
                self.config
                    .memory_estimate_bytes
                    .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
            }

            // -- capability-specific methods ---------------------------------

            $($extra)*
        }
    };
}

// ---------------------------------------------------------------------------
// 1. TTSProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for text-to-speech providers.
    ///
    /// Subclass and override `textToSpeech()` to implement a custom TTS
    /// backend.
    "TTSProvider", JsTTSProvider,
    extra {
        /// Synthesize speech from text.
        #[napi(js_name = "textToSpeech")]
        pub async fn text_to_speech(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override textToSpeech()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 2. MusicProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for music generation providers.
    ///
    /// Subclass and override `generateMusic()` and `generateSfx()` to
    /// implement a custom music/SFX backend.
    "MusicProvider", JsMusicProvider,
    extra {
        /// Generate music from a prompt.
        #[napi(js_name = "generateMusic")]
        pub async fn generate_music(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override generateMusic()",
            ))
        }

        /// Generate a sound effect from a prompt.
        #[napi(js_name = "generateSfx")]
        pub async fn generate_sfx(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override generateSfx()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 3. ImageProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for image generation providers.
    ///
    /// Subclass and override `generateImage()` and `upscaleImage()` to
    /// implement a custom image backend.
    "ImageProvider", JsImageProvider,
    extra {
        /// Generate an image from a prompt.
        #[napi(js_name = "generateImage")]
        pub async fn generate_image(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override generateImage()",
            ))
        }

        /// Upscale an existing image.
        #[napi(js_name = "upscaleImage")]
        pub async fn upscale_image(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override upscaleImage()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 4. VideoProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for video generation providers.
    ///
    /// Subclass and override `textToVideo()` and `imageToVideo()` to
    /// implement a custom video backend.
    "VideoProvider", JsVideoProvider,
    extra {
        /// Generate a video from a text prompt.
        #[napi(js_name = "textToVideo")]
        pub async fn text_to_video(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override textToVideo()",
            ))
        }

        /// Generate a video from an image (image-to-video).
        #[napi(js_name = "imageToVideo")]
        pub async fn image_to_video(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override imageToVideo()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 5. ThreeDProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for 3D model generation providers.
    ///
    /// Subclass and override `generate3d()` to implement a custom 3D
    /// backend.
    "ThreeDProvider", JsThreeDProvider,
    extra {
        /// Generate a 3D model from a prompt or image.
        #[napi(js_name = "generate3d")]
        pub async fn generate_3d(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override generate3d()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 6. BackgroundRemovalProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for background removal providers.
    ///
    /// Subclass and override `removeBackground()` to implement a custom
    /// background-removal backend.
    "BackgroundRemovalProvider", JsBackgroundRemovalProvider,
    extra {
        /// Remove the background from an image.
        #[napi(js_name = "removeBackground")]
        pub async fn remove_background(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override removeBackground()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 7. VoiceProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for voice cloning providers.
    ///
    /// Subclass and override `cloneVoice()`, `listVoices()`, and
    /// `deleteVoice()` to implement a custom voice-cloning backend.
    "VoiceProvider", JsVoiceProvider,
    extra {
        /// Clone a voice from audio samples.
        #[napi(js_name = "cloneVoice")]
        pub async fn clone_voice(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override cloneVoice()",
            ))
        }

        /// List all available voices.
        #[napi(js_name = "listVoices")]
        pub async fn list_voices(&self) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override listVoices()",
            ))
        }

        /// Delete a previously-cloned voice.
        #[napi(js_name = "deleteVoice")]
        pub async fn delete_voice(
            &self,
            _voice: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override deleteVoice()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 8. LLMProvider (canonical capability trait surface)
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for large-language-model providers.
    ///
    /// Mirrors the [`blazen_llm::providers::LLMProvider`] capability trait.
    /// Subclass and override `complete()` (and optionally `stream()`) to
    /// implement a custom chat/completion backend.
    "LLMProvider", JsLLMProvider,
    extra {
        /// Non-streaming completion. Receives a `ModelRequest`-shaped object
        /// and returns a `ModelResponse`-shaped object.
        #[napi(js_name = "complete")]
        pub async fn complete(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override complete()",
            ))
        }

        /// Streaming completion. Receives a `ModelRequest`-shaped object and
        /// returns the accumulated stream chunks.
        #[napi(js_name = "stream")]
        pub async fn stream(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override stream()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 9. EmbeddingProvider (canonical capability trait surface)
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for vector-embedding providers.
    ///
    /// Mirrors the [`blazen_llm::providers::EmbeddingProvider`] capability
    /// trait. Subclass and override `embed()` to implement a custom embedding
    /// backend.
    "EmbeddingProvider", JsEmbeddingProvider,
    extra {
        /// Embed a batch of texts. Receives an array of strings and returns
        /// an array of float vectors (one per input).
        #[napi(js_name = "embed")]
        pub async fn embed(
            &self,
            _texts: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override embed()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 10. ImageGenProvider (canonical capability trait surface)
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for 2D image-generation providers.
    ///
    /// Mirrors the [`blazen_llm::providers::ImageGenProvider`] capability
    /// trait. Subclass and override `generateImage()` (and optionally
    /// `upscaleImage()`) to implement a custom image backend.
    "ImageGenProvider", JsImageGenProvider,
    extra {
        /// Generate images from a text prompt.
        #[napi(js_name = "generateImage")]
        pub async fn generate_image(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override generateImage()",
            ))
        }

        /// Upscale an existing image.
        #[napi(js_name = "upscaleImage")]
        pub async fn upscale_image(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override upscaleImage()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 11. VcProvider (canonical voice-conversion capability trait surface)
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for voice-conversion providers.
    ///
    /// Mirrors the [`blazen_llm::providers::VcProvider`] capability trait —
    /// source utterance + target voice → re-voiced audio, plus voice
    /// cloning. Subclass and override `convertVoice()` (and optionally
    /// `cloneVoice()`, `listVoices()`, `deleteVoice()`).
    "VcProvider", JsVcProvider,
    extra {
        /// Convert the source utterance into the target voice.
        #[napi(js_name = "convertVoice")]
        pub async fn convert_voice(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override convertVoice()",
            ))
        }

        /// Clone a voice from reference audio clips.
        #[napi(js_name = "cloneVoice")]
        pub async fn clone_voice(
            &self,
            _request: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override cloneVoice()",
            ))
        }

        /// List all voices known to this provider.
        #[napi(js_name = "listVoices")]
        pub async fn list_voices(&self) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override listVoices()",
            ))
        }

        /// Delete a previously-cloned voice.
        #[napi(js_name = "deleteVoice")]
        pub async fn delete_voice(
            &self,
            _voice: serde_json::Value,
        ) -> napi::Result<serde_json::Value> {
            Err(napi::Error::from_reason(
                "subclass must override deleteVoice()",
            ))
        }
    }
}
