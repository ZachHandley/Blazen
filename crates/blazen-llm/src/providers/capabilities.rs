//! Capability sub-traits for the provider class hierarchy.
//!
//! Each trait extends [`crate::providers::BaseProvider`] and adds the
//! capability-specific async methods that concrete per-engine providers
//! implement. Method signatures mirror the existing
//! [`crate::compute::traits`] surface (`AudioGeneration`,
//! `Transcription`, `ThreeDGeneration`, …) but split the multi-modal
//! ones cleanly:
//!
//! - Old `AudioGeneration` (TTS + music + SFX) → new [`TtsProvider`] +
//!   [`MusicProvider`].
//! - Old `Transcription` → new [`SttProvider`].
//! - Old `ThreeDGeneration` → new [`ThreeDProvider`].
//! - Old `ImageGeneration` → new [`ImageGenProvider`].
//! - Old `VoiceCloning` → new [`VcProvider`].
//! - Old `Codec` → new [`CodecProvider`].
//! - Old `BackgroundRemoval` → new [`BackgroundRemovalProvider`].
//! - Old `VideoGeneration` → new [`VideoProvider`].
//! - New [`LLMProvider`] wraps the existing [`crate::traits::Model`]
//!   chat / completion / streaming methods.
//! - New [`EmbeddingProvider`] wraps text-embedding capability.
//!
//! The existing [`crate::compute::traits`] traits stay in place as the
//! compute-job (`submit`/`status`/`result`/`cancel`) shape used by
//! fal/Replicate/RunPod-style HTTP backends. They are orthogonal to
//! these capability traits — concrete providers can implement both.
//!
//! Object-safety: every trait method takes `&self` and returns owned
//! data, so each trait is object-safe (`Arc<dyn TtsProvider>` works).

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

#[cfg(feature = "audio-music")]
use crate::MusicChunk;
#[cfg(feature = "threed")]
use crate::compute::requests::{AnimateRequest, RefineRequest, RigRequest, TexturizeRequest};
use crate::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
#[cfg(feature = "threed")]
use crate::compute::results::{AnimateResult, RefineResult, RigResult, TexturizeResult};
use crate::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use crate::error::BlazenError;
use crate::providers::root::BaseProvider;
use crate::types::{ModelRequest, ModelResponse, StreamChunk};

// ---------------------------------------------------------------------------
// LLMProvider — chat / completion / streaming
// ---------------------------------------------------------------------------

/// LLM provider capability — chat completion + token streaming.
///
/// Mirrors the existing [`crate::traits::Model`] surface but slots
/// into the polymorphic [`BaseProvider`] hierarchy. Concrete LLM
/// providers (`OpenAI`, Anthropic, Fal, `Llama.cpp`, …) implement BOTH
/// [`Model`](crate::traits::Model) (for backwards compatibility) AND
/// [`LLMProvider`] (the canonical capability surface).
#[async_trait]
pub trait LLMProvider: BaseProvider {
    /// Non-streaming completion.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError>;

    /// Streaming completion — yields `StreamChunk`s until the stream
    /// terminates with either an `EndOfStream` chunk or an error.
    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>;

    /// The model identifier the provider routes to by default. Mirrors
    /// [`crate::traits::Model::model_id`].
    fn model_id(&self) -> &str {
        // Default: synthesize a "<provider_id>/<version>" id when the
        // provider hasn't overridden. Real providers should override to
        // return the wire-format model id their API expects.
        self.metadata()
            .version
            .as_deref()
            .unwrap_or_else(|| self.provider_id())
    }
}

// ---------------------------------------------------------------------------
// TtsProvider — text-to-speech
// ---------------------------------------------------------------------------

/// Text-to-speech audio synthesis capability.
#[async_trait]
pub trait TtsProvider: BaseProvider {
    /// Synthesize speech from text.
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError>;

    /// List the voice presets this provider exposes. Returns an empty
    /// list by default for providers that don't enumerate (most local
    /// model backends).
    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        Ok(Vec::new())
    }
}

// ---------------------------------------------------------------------------
// SttProvider — speech-to-text
// ---------------------------------------------------------------------------

/// Speech-to-text transcription capability.
#[async_trait]
pub trait SttProvider: BaseProvider {
    /// Transcribe audio to text with optional diarization.
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// MusicProvider — text-to-music / text-to-sfx
// ---------------------------------------------------------------------------

/// Text-to-music / text-to-sfx audio generation capability.
///
/// Split from the old `AudioGeneration` trait so providers that ONLY
/// do music (Stable Audio, `MusicGen`) don't have to implement TTS, and
/// vice versa.
#[async_trait]
pub trait MusicProvider: BaseProvider {
    /// Generate music from a prompt.
    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError>;

    /// Generate sound effects from a prompt. Defaults to
    /// `BlazenError::Unsupported` so music-only providers don't have
    /// to implement.
    async fn generate_sfx(&self, _request: MusicRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "sound effect generation not supported by this provider",
        ))
    }

    /// Stream music generation for low-latency / progressive playback.
    ///
    /// Yields [`MusicChunk`]s as the backend produces them; the
    /// concatenated samples equal a single
    /// [`generate_music`](MusicProvider::generate_music) call. Defaults
    /// to `BlazenError::Unsupported`; providers that wrap a streaming
    /// [`MusicBackend`](crate::MusicBackend) override this.
    #[cfg(feature = "audio-music")]
    async fn stream_generate_music(
        &self,
        _prompt: String,
        _duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::unsupported(
            "streaming music generation not supported by this provider",
        ))
    }

    /// Stream SFX generation for low-latency / progressive playback.
    ///
    /// Yields [`MusicChunk`]s as the backend produces them; the
    /// concatenated samples equal a single
    /// [`generate_sfx`](MusicProvider::generate_sfx) call. Defaults to
    /// `BlazenError::Unsupported`; providers that wrap a streaming
    /// [`MusicBackend`](crate::MusicBackend) override this.
    #[cfg(feature = "audio-music")]
    async fn stream_generate_sfx(
        &self,
        _prompt: String,
        _duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::unsupported(
            "streaming sfx generation not supported by this provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// VcProvider — voice conversion
// ---------------------------------------------------------------------------

/// Voice conversion capability — source utterance + target voice id
/// → re-voiced audio. Also supports cloning a new target voice from
/// reference clips (the old `VoiceCloning` trait absorbed).
#[async_trait]
pub trait VcProvider: BaseProvider {
    /// Convert the source utterance into the target voice and return
    /// the rendered audio.
    async fn convert_voice(&self, request: VoiceCloneRequest) -> Result<AudioResult, BlazenError>;

    /// Clone a voice from reference audio clips, returning a handle
    /// the caller can later pass as the target on
    /// [`convert_voice`](VcProvider::convert_voice). Defaults to
    /// `BlazenError::Unsupported` for providers that only convert
    /// to pre-registered voices.
    async fn clone_voice(&self, _request: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "voice cloning not supported by this provider",
        ))
    }

    /// List all voices known to this provider. Defaults to an empty
    /// list.
    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        Ok(Vec::new())
    }

    /// Delete a previously cloned voice. Defaults to
    /// `BlazenError::Unsupported`.
    async fn delete_voice(&self, _voice: &VoiceHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "delete_voice not supported by this provider",
        ))
    }

    /// Stream voice conversion for low-latency / real-time use.
    ///
    /// Consumes a single buffer of 32-bit float PCM samples
    /// (`input_pcm`) at the backend's expected source sample rate and
    /// yields converted PCM sample chunks at the target voice's native
    /// rate. Defaults to `BlazenError::Unsupported`; providers that wrap
    /// a streaming
    /// [`VoiceConversionBackend`](crate::VoiceConversionBackend)
    /// override this.
    ///
    /// # Item type
    ///
    /// The yielded item is the raw `Vec<f32>` PCM frame straight from the
    /// backend's
    /// [`stream_convert`](crate::VoiceConversionBackend::stream_convert),
    /// not a richer chunk record. The `blazen-uniffi` wrapper re-wraps
    /// each frame into its FFI `VcChunk` (adding `is_final` /
    /// `latency_seconds`); there is no `VcChunk` type at this layer, so
    /// the faithful mapping is the backend's native sample buffer.
    #[cfg(feature = "audio-vc")]
    async fn stream_convert_pcm(
        &self,
        _input_pcm: Vec<f32>,
        _target_voice_id: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<f32>, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::unsupported(
            "streaming voice conversion not supported by this provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// ThreeDProvider — 3D mesh generation
// ---------------------------------------------------------------------------

/// 3D mesh generation + post-processing capability.
///
/// Carries one *generation* method (`generate_3d`, which produces a fresh
/// mesh from a text prompt or image) plus four *post-processing* methods —
/// `texturize`, `rig`, `refine`, `animate` — that operate on an existing
/// mesh. Post-proc methods default to `BlazenError::Unsupported`, so a
/// generation-only backend (e.g. `TripoSrProvider`) doesn't need to
/// override them; HTTP-proxy / multi-stage backends like `Compat3dProvider`
/// override them by delegating to the inner `blazen-3d` capability traits.
///
/// Post-proc DTOs (`TexturizeRequest` / `TexturizeResult` / `PbrMaps` /
/// `RigRequest` / `RigResult` / `RefineRequest` / `RefineResult` /
/// `RefineStats` / `AnimateRequest` / `AnimateResult`) live in
/// [`crate::compute::requests`] + [`crate::compute::results`] as
/// re-exports of the canonical `blazen-3d` types.
#[async_trait]
pub trait ThreeDProvider: BaseProvider {
    /// Generate a 3D mesh from a generic [`ThreeDRequest`] (carries
    /// either a text prompt or a source image; the provider picks
    /// whichever mode it supports).
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError>;

    /// Apply or generate a texture / material for an existing 3D mesh.
    /// Defaults to `BlazenError::Unsupported`.
    #[cfg(feature = "threed")]
    async fn texturize(
        &self,
        _mesh_glb: &[u8],
        _request: TexturizeRequest,
    ) -> Result<TexturizeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "ThreeDProvider::texturize not implemented by this provider",
        ))
    }

    /// Auto-rig a 3D mesh (skeletal armature + optional skin weights).
    /// Defaults to `BlazenError::Unsupported`.
    #[cfg(feature = "threed")]
    async fn rig(&self, _mesh_glb: &[u8], _request: RigRequest) -> Result<RigResult, BlazenError> {
        Err(BlazenError::unsupported(
            "ThreeDProvider::rig not implemented by this provider",
        ))
    }

    /// Refine a 3D mesh (decimate / fill holes / unwrap UVs /
    /// retopologize / smooth). Defaults to `BlazenError::Unsupported`.
    #[cfg(feature = "threed")]
    async fn refine(
        &self,
        _mesh_glb: &[u8],
        _request: RefineRequest,
    ) -> Result<RefineResult, BlazenError> {
        Err(BlazenError::unsupported(
            "ThreeDProvider::refine not implemented by this provider",
        ))
    }

    /// Animate a rigged 3D mesh from a text prompt, motion-capture
    /// clip, or driving video. Defaults to `BlazenError::Unsupported`.
    #[cfg(feature = "threed")]
    async fn animate(
        &self,
        _rigged_glb: &[u8],
        _request: AnimateRequest,
    ) -> Result<AnimateResult, BlazenError> {
        Err(BlazenError::unsupported(
            "ThreeDProvider::animate not implemented by this provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// ImageGenProvider — 2D image generation
// ---------------------------------------------------------------------------

/// 2D image generation capability — text-to-image, image-to-image,
/// upscale.
#[async_trait]
pub trait ImageGenProvider: BaseProvider {
    /// Generate images from a text prompt.
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError>;

    /// Upscale an existing image. Defaults to `BlazenError::Unsupported`
    /// for providers that only generate fresh images.
    async fn upscale_image(&self, _request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        Err(BlazenError::unsupported(
            "image upscaling not supported by this provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProvider — vector embeddings
// ---------------------------------------------------------------------------

/// Vector embedding generation capability.
///
/// Distinct from the existing [`crate::compute::traits`] surface
/// because embedding has no compute-job submit/poll semantics — it's
/// strictly request → response.
#[async_trait]
pub trait EmbeddingProvider: BaseProvider {
    /// Embed a batch of texts. Each input maps to one `Vec<f32>` of
    /// length [`dimensions`](EmbeddingProvider::dimensions).
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError>;

    /// Embedding vector dimensionality. Returns the per-vector length
    /// of every output from [`embed`](EmbeddingProvider::embed).
    fn dimensions(&self) -> usize;
}

// ---------------------------------------------------------------------------
// CodecProvider — neural audio codec
// ---------------------------------------------------------------------------

/// Neural audio codec capability — PCM ↔ discrete codebook tokens.
///
/// Both methods default to `BlazenError::Unsupported` so existing
/// non-codec providers do not need to override them.
#[async_trait]
pub trait CodecProvider: BaseProvider {
    /// Encode mono PCM samples (`f32` in `[-1.0, 1.0]`) into discrete
    /// codebook tokens.
    async fn encode_audio(&self, _pcm: &[f32], _sample_rate: u32) -> Result<Vec<u32>, BlazenError> {
        Err(BlazenError::unsupported(
            "audio codec encode_audio not supported by this provider",
        ))
    }

    /// Decode flat row-major codebook tokens back into mono PCM samples.
    async fn decode_audio(
        &self,
        _tokens: &[u32],
        _num_codebooks: usize,
    ) -> Result<Vec<f32>, BlazenError> {
        Err(BlazenError::unsupported(
            "audio codec decode_audio not supported by this provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// BackgroundRemovalProvider — alpha-matte extraction on existing images
// ---------------------------------------------------------------------------

/// Background removal capability — extract an alpha matte / cut out
/// the subject from an existing image.
#[async_trait]
pub trait BackgroundRemovalProvider: BaseProvider {
    /// Remove the background from an image and return the result.
    async fn remove_background(
        &self,
        request: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError>;
}

// ---------------------------------------------------------------------------
// VideoProvider — text-to-video / image-to-video
// ---------------------------------------------------------------------------

/// Video generation capability.
#[async_trait]
pub trait VideoProvider: BaseProvider {
    /// Generate a video from a text prompt.
    async fn text_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError>;

    /// Generate a video from a source image and prompt. Defaults to
    /// `BlazenError::Unsupported` for text-only providers.
    async fn image_to_video(&self, _request: VideoRequest) -> Result<VideoResult, BlazenError> {
        Err(BlazenError::unsupported(
            "image_to_video not supported by this provider",
        ))
    }
}

// ---------------------------------------------------------------------------
// Arc passthrough impls
// ---------------------------------------------------------------------------
//
// Mirror the BaseProvider passthrough in `root.rs` so consumers can
// hold capability providers behind Arc without manual newtype wrappers.

macro_rules! impl_arc_passthrough {
    ($trait:ident, $($method:ident($($arg:ident: $ty:ty),*) -> $ret:ty),+ $(,)?) => {
        #[async_trait]
        impl<P: $trait + ?Sized> $trait for Arc<P> {
            $(
                async fn $method(&self, $($arg: $ty),*) -> $ret {
                    (**self).$method($($arg),*).await
                }
            )+
        }
    };
}

// LLM
#[async_trait]
impl<P: LLMProvider + ?Sized> LLMProvider for Arc<P> {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        (**self).complete(request).await
    }
    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        (**self).stream(request).await
    }
    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}

// TTS — passes through synthesize + list_voices
impl_arc_passthrough!(
    TtsProvider,
    synthesize(request: SpeechRequest) -> Result<AudioResult, BlazenError>,
    list_voices() -> Result<Vec<VoiceHandle>, BlazenError>,
);

// STT
impl_arc_passthrough!(
    SttProvider,
    transcribe(request: TranscriptionRequest) -> Result<TranscriptionResult, BlazenError>,
);

// Music — cfg-gated streaming methods don't fit the simple macro shape;
// hand-expand so `Arc<P>` forwards the overridden streaming impls instead of
// silently falling back to the trait defaults.
#[async_trait]
impl<P: MusicProvider + ?Sized> MusicProvider for Arc<P> {
    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        (**self).generate_music(request).await
    }
    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        (**self).generate_sfx(request).await
    }
    #[cfg(feature = "audio-music")]
    async fn stream_generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        (**self)
            .stream_generate_music(prompt, duration_seconds)
            .await
    }
    #[cfg(feature = "audio-music")]
    async fn stream_generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>, BlazenError>
    {
        (**self).stream_generate_sfx(prompt, duration_seconds).await
    }
}

// VC — VcProvider's list_voices/delete_voice take refs that don't fit the
// simple macro shape; expand by hand.
#[async_trait]
impl<P: VcProvider + ?Sized> VcProvider for Arc<P> {
    async fn convert_voice(&self, request: VoiceCloneRequest) -> Result<AudioResult, BlazenError> {
        (**self).convert_voice(request).await
    }
    async fn clone_voice(&self, request: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        (**self).clone_voice(request).await
    }
    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        (**self).list_voices().await
    }
    async fn delete_voice(&self, voice: &VoiceHandle) -> Result<(), BlazenError> {
        (**self).delete_voice(voice).await
    }
    #[cfg(feature = "audio-vc")]
    async fn stream_convert_pcm(
        &self,
        input_pcm: Vec<f32>,
        target_voice_id: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<f32>, BlazenError>> + Send>>, BlazenError>
    {
        (**self)
            .stream_convert_pcm(input_pcm, target_voice_id)
            .await
    }
}

// ThreeD
impl_arc_passthrough!(
    ThreeDProvider,
    generate_3d(request: ThreeDRequest) -> Result<ThreeDResult, BlazenError>,
);

// ImageGen
impl_arc_passthrough!(
    ImageGenProvider,
    generate_image(request: ImageRequest) -> Result<ImageResult, BlazenError>,
    upscale_image(request: UpscaleRequest) -> Result<ImageResult, BlazenError>,
);

// Embedding — has both async + sync methods; hand-expand.
#[async_trait]
impl<P: EmbeddingProvider + ?Sized> EmbeddingProvider for Arc<P> {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        (**self).embed(texts).await
    }
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }
}

// Codec — takes slice refs; hand-expand.
#[async_trait]
impl<P: CodecProvider + ?Sized> CodecProvider for Arc<P> {
    async fn encode_audio(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<u32>, BlazenError> {
        (**self).encode_audio(pcm, sample_rate).await
    }
    async fn decode_audio(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> Result<Vec<f32>, BlazenError> {
        (**self).decode_audio(tokens, num_codebooks).await
    }
}

// BackgroundRemoval
impl_arc_passthrough!(
    BackgroundRemovalProvider,
    remove_background(request: BackgroundRemovalRequest) -> Result<ImageResult, BlazenError>,
);

// Video
impl_arc_passthrough!(
    VideoProvider,
    text_to_video(request: VideoRequest) -> Result<VideoResult, BlazenError>,
    image_to_video(request: VideoRequest) -> Result<VideoResult, BlazenError>,
);

// ---------------------------------------------------------------------------
// Tests — smoke checks: each trait is object-safe and Arc-passthrough works.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::root::{CapabilityKind, ProviderMetadata};

    #[derive(Debug)]
    struct FakeTts {
        meta: ProviderMetadata,
    }

    impl BaseProvider for FakeTts {
        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }
    }

    #[async_trait]
    impl TtsProvider for FakeTts {
        async fn synthesize(&self, _request: SpeechRequest) -> Result<AudioResult, BlazenError> {
            Err(BlazenError::unsupported("fake"))
        }
    }

    #[test]
    fn tts_provider_is_object_safe() {
        let p = FakeTts {
            meta: ProviderMetadata::new("fake-tts", CapabilityKind::Tts),
        };
        // The mere act of building a `dyn TtsProvider` proves object-safety.
        let erased: &dyn TtsProvider = &p;
        assert_eq!(erased.provider_id(), "fake-tts");
    }

    #[test]
    fn capability_traits_are_object_safe() {
        // Compile-time proof: each Box<dyn ...> is well-formed only if the
        // trait is object-safe. Failure to compile = trait isn't object-safe.
        fn _llm(_: Box<dyn LLMProvider>) {}
        fn _stt(_: Box<dyn SttProvider>) {}
        fn _music(_: Box<dyn MusicProvider>) {}
        fn _vc(_: Box<dyn VcProvider>) {}
        fn _three_d(_: Box<dyn ThreeDProvider>) {}
        fn _image_gen(_: Box<dyn ImageGenProvider>) {}
        fn _embedding(_: Box<dyn EmbeddingProvider>) {}
        fn _codec(_: Box<dyn CodecProvider>) {}
        fn _bg(_: Box<dyn BackgroundRemovalProvider>) {}
        fn _video(_: Box<dyn VideoProvider>) {}
    }

    #[tokio::test]
    async fn arc_passthrough_for_tts_provider() {
        let p = Arc::new(FakeTts {
            meta: ProviderMetadata::new("fake-tts", CapabilityKind::Tts),
        });
        let voices = p.list_voices().await.unwrap();
        assert!(voices.is_empty());
    }
}
