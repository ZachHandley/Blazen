// See `musicgen.rs` for the rationale behind these crate-wide allows.
#![allow(clippy::doc_markdown, clippy::unused_async)]

//! `MusicModel` — unified aggregator exposing `.musicgen()`,
//! `.audioGen()`, and `.stableAudio()` factories that return a single
//! `MusicModel` whose `generateMusic` / `generateSfx` / `streamGenerateMusic`
//! / `streamGenerateSfx` methods dispatch through a dyn-trait
//! `MusicBackend`.
//!
//! This mirrors the convenience aggregator pattern used by `Model` /
//! `EmbeddingModel` in the rest of the binding so callers can swap
//! backends without touching call sites.

use std::sync::Arc;

use blazen_audio_music::MusicBackend;
use napi::Result;
use napi_derive::napi;

use crate::error::music_error_to_napi;
use crate::music::StreamMusicChunkCallbackTsfn;
use crate::music::audiogen::{JsAudioGenBackend, JsAudioGenOptions};
use crate::music::chunk::{JsMusicResult, build_music_result};
use crate::music::musicgen::{JsMusicgenBackend, JsMusicgenOptions};
use crate::music::stable_audio::{JsStableAudioBackend, JsStableAudioOptions};

#[cfg(feature = "audio-music-stable-audio")]
use blazen_audio_music::backends::stable_audio::StableAudioBackend;
#[cfg(feature = "audio-music-stable-audio")]
use std::path::PathBuf;
#[cfg(feature = "audio-music-stable-audio")]
use tokio::sync::OnceCell;

#[cfg(feature = "audio-music-musicgen")]
use crate::music::musicgen::drive_music_stream;

// ---------------------------------------------------------------------------
// Aggregator wrapper
// ---------------------------------------------------------------------------

/// Backing storage strategy for [`JsMusicModel`].
///
/// MusicGen and AudioGen wrap a single `Arc<dyn MusicBackend>` shared with
/// the per-backend `#[napi]` class; Stable Audio carries its lazy-loader
/// (and `StableAudioConfig`) so the unified aggregator still constructs
/// synchronously.
enum Inner {
    /// MusicGen + AudioGen: ready-to-use `Arc<dyn MusicBackend>` (or a
    /// stub that always returns an error when the feature is OFF).
    Ready(Arc<dyn MusicBackend + Send + Sync>),
    /// Stable Audio: lazy-loaded backend driven by an internal `OnceCell`.
    #[cfg(feature = "audio-music-stable-audio")]
    StableAudio {
        backend: Arc<OnceCell<Arc<StableAudioBackend>>>,
        config: blazen_audio_music::backends::stable_audio::StableAudioConfig,
    },
    /// Stable Audio in stub mode (feature OFF) — every call returns
    /// `MusicNotYetImplementedError`.
    #[cfg(not(feature = "audio-music-stable-audio"))]
    StableAudioStub(Arc<StableAudioStubBackend>),
}

#[cfg(not(feature = "audio-music-stable-audio"))]
struct StableAudioStubBackend {
    inner: blazen_audio_music::backends::stable_audio::StableAudioBackend,
}

#[cfg(not(feature = "audio-music-stable-audio"))]
impl StableAudioStubBackend {
    fn new() -> Self {
        Self {
            inner: blazen_audio_music::backends::stable_audio::StableAudioBackend::new(),
        }
    }
}

/// Unified music + SFX backend aggregator.
///
/// ```javascript
/// // Pick a backend at construction time:
/// const m = MusicModel.musicgen({ variant: "small" });
/// const wav = await m.generateMusic("uplifting piano", 8);
///
/// // Or swap to AudioGen / Stable Audio with the same method surface:
/// const sfx = MusicModel.audioGen({});
/// const ambient = MusicModel.stableAudio({});
/// ```
#[napi(js_name = "MusicModel")]
pub struct JsMusicModel {
    inner: Inner,
    model_id: String,
}

#[napi]
impl JsMusicModel {
    /// Build a [`JsMusicModel`] backed by MusicGen.
    #[napi(factory)]
    #[must_use]
    pub fn musicgen(options: Option<JsMusicgenOptions>) -> Self {
        let backend = JsMusicgenBackend::create(options);
        let model_id = backend.model_id();
        let inner = build_musicgen_inner(&backend);
        Self { inner, model_id }
    }

    /// Build a [`JsMusicModel`] backed by AudioGen.
    #[napi(factory, js_name = "audioGen")]
    #[must_use]
    pub fn audio_gen(options: Option<JsAudioGenOptions>) -> Self {
        let backend = JsAudioGenBackend::create(options);
        let model_id = backend.model_id();
        let inner = build_audiogen_inner(&backend);
        Self { inner, model_id }
    }

    /// Build a [`JsMusicModel`] backed by Stable Audio Open.
    #[napi(factory, js_name = "stableAudio")]
    #[must_use]
    pub fn stable_audio(options: Option<JsStableAudioOptions>) -> Self {
        // We materialize the per-backend `JsStableAudioBackend` for its
        // `modelId` but cannot reuse its private fields, so we re-derive
        // the inner storage from the same options to keep behaviour
        // bit-identical to the per-backend class.
        let backend = JsStableAudioBackend::create(options);
        let model_id = backend.model_id();
        let inner = build_stable_audio_inner();
        Self { inner, model_id }
    }

    /// Backend identifier — same value `modelId` returns on the per-
    /// backend `#[napi]` class (e.g. `"musicgen-small"`,
    /// `"audiogen-medium"`, `"stable-audio"`).
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Generate music conditioned on `prompt`.
    ///
    /// # Errors
    /// See per-backend documentation
    /// ([`JsMusicgenBackend::generate_music`], etc.).
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        let backend = self.resolve_backend().await?;
        #[allow(clippy::cast_possible_truncation)]
        let generated = backend
            .generate_music(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        Ok(build_music_result(generated))
    }

    /// Generate sound-effect audio conditioned on `prompt`.
    ///
    /// # Errors
    /// See per-backend documentation.
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        let backend = self.resolve_backend().await?;
        #[allow(clippy::cast_possible_truncation)]
        let generated = backend
            .generate_sfx(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        Ok(build_music_result(generated))
    }

    /// Stream music generation, invoking `onChunk` for each emitted
    /// `JsMusicChunk` until the final chunk arrives (`isFinal === true`).
    ///
    /// # Errors
    /// See per-backend documentation.
    #[napi(js_name = "streamGenerateMusic")]
    pub async fn stream_generate_music(
        &self,
        prompt: String,
        duration_seconds: f64,
        on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        let backend = self.resolve_backend().await?;
        #[allow(clippy::cast_possible_truncation)]
        let stream = backend
            .stream_generate_music(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        drive(stream, &on_chunk).await
    }

    /// Stream SFX generation, invoking `onChunk` for each emitted
    /// `JsMusicChunk` until the final chunk arrives (`isFinal === true`).
    ///
    /// # Errors
    /// See per-backend documentation.
    #[napi(js_name = "streamGenerateSfx")]
    pub async fn stream_generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f64,
        on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        let backend = self.resolve_backend().await?;
        #[allow(clippy::cast_possible_truncation)]
        let stream = backend
            .stream_generate_sfx(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        drive(stream, &on_chunk).await
    }
}

impl JsMusicModel {
    async fn resolve_backend(&self) -> Result<Arc<dyn MusicBackend + Send + Sync>> {
        match &self.inner {
            Inner::Ready(b) => Ok(Arc::clone(b)),
            #[cfg(feature = "audio-music-stable-audio")]
            Inner::StableAudio { backend, config } => {
                let loaded = backend
                    .get_or_try_init(|| async {
                        StableAudioBackend::load(config.clone()).await.map(Arc::new)
                    })
                    .await
                    .map_err(clone_and_map_music_error)?;
                Ok(Arc::clone(loaded) as Arc<dyn MusicBackend + Send + Sync>)
            }
            #[cfg(not(feature = "audio-music-stable-audio"))]
            Inner::StableAudioStub(stub) => {
                let arc: Arc<dyn MusicBackend + Send + Sync> = Arc::new(StableAudioStubAdapter {
                    inner: Arc::clone(stub),
                });
                Ok(arc)
            }
        }
    }
}

#[cfg(not(feature = "audio-music-stable-audio"))]
struct StableAudioStubAdapter {
    inner: Arc<StableAudioStubBackend>,
}

#[cfg(not(feature = "audio-music-stable-audio"))]
#[async_trait::async_trait]
impl blazen_audio::AudioBackend for StableAudioStubAdapter {
    fn id(&self) -> &'static str {
        "stable-audio"
    }

    fn provider_kind(&self) -> &'static str {
        "music"
    }
}

#[cfg(not(feature = "audio-music-stable-audio"))]
#[async_trait::async_trait]
impl MusicBackend for StableAudioStubAdapter {
    async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> std::result::Result<blazen_audio::GeneratedAudio, blazen_audio_music::MusicError> {
        self.inner
            .inner
            .generate_music(prompt, duration_seconds)
            .await
    }

    async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> std::result::Result<blazen_audio::GeneratedAudio, blazen_audio_music::MusicError> {
        self.inner
            .inner
            .generate_sfx(prompt, duration_seconds)
            .await
    }
}

#[cfg(feature = "audio-music-stable-audio")]
#[cfg(feature = "audio-music-stable-audio")]
fn clone_and_map_music_error(err: blazen_audio_music::MusicError) -> napi::Error {
    music_error_to_napi(err)
}

// ---------------------------------------------------------------------------
// Per-backend Inner builders
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-music-musicgen")]
fn build_musicgen_inner(backend: &JsMusicgenBackend) -> Inner {
    Inner::Ready(backend.arc() as Arc<dyn MusicBackend + Send + Sync>)
}

#[cfg(not(feature = "audio-music-musicgen"))]
fn build_musicgen_inner(_backend: &JsMusicgenBackend) -> Inner {
    Inner::Ready(Arc::new(EngineUnavailableBackend {
        id: "musicgen-small",
    }))
}

#[cfg(feature = "audio-music-audiogen")]
fn build_audiogen_inner(backend: &JsAudioGenBackend) -> Inner {
    Inner::Ready(backend.arc() as Arc<dyn MusicBackend + Send + Sync>)
}

#[cfg(not(feature = "audio-music-audiogen"))]
fn build_audiogen_inner(_backend: &JsAudioGenBackend) -> Inner {
    Inner::Ready(Arc::new(EngineUnavailableBackend {
        id: "audiogen-medium",
    }))
}

#[cfg(feature = "audio-music-stable-audio")]
fn build_stable_audio_inner() -> Inner {
    // The per-backend `JsStableAudioBackend::create` already built a
    // `StableAudioConfig` from the same options bag; we reconstruct one
    // here from defaults so the aggregator is independent of the
    // per-backend wrapper's private state. Callers who need custom HF
    // repo / tokenizer paths should use `StableAudioBackend.create({...})`
    // directly.
    let variant = blazen_audio_music::backends::stable_audio::StableAudioVariant::Small;
    let config = blazen_audio_music::backends::stable_audio::StableAudioConfig {
        hf_repo: variant.hf_repo().to_string(),
        local_weights_path: None,
        tokenizer_path: PathBuf::from("tokenizer.json"),
        device: candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
        variant,
    };
    Inner::StableAudio {
        backend: Arc::new(OnceCell::new()),
        config,
    }
}

#[cfg(not(feature = "audio-music-stable-audio"))]
fn build_stable_audio_inner() -> Inner {
    Inner::StableAudioStub(Arc::new(StableAudioStubBackend::new()))
}

// ---------------------------------------------------------------------------
// EngineUnavailableBackend (used when a backend's engine feature is OFF)
// ---------------------------------------------------------------------------

#[cfg(any(
    not(feature = "audio-music-musicgen"),
    not(feature = "audio-music-audiogen")
))]
struct EngineUnavailableBackend {
    id: &'static str,
}

#[cfg(any(
    not(feature = "audio-music-musicgen"),
    not(feature = "audio-music-audiogen")
))]
#[async_trait::async_trait]
impl blazen_audio::AudioBackend for EngineUnavailableBackend {
    fn id(&self) -> &'static str {
        self.id
    }

    fn provider_kind(&self) -> &'static str {
        "music"
    }
}

#[cfg(any(
    not(feature = "audio-music-musicgen"),
    not(feature = "audio-music-audiogen")
))]
#[async_trait::async_trait]
impl MusicBackend for EngineUnavailableBackend {
    async fn generate_music(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> std::result::Result<blazen_audio::GeneratedAudio, blazen_audio_music::MusicError> {
        Err(blazen_audio_music::MusicError::EngineNotAvailable)
    }

    async fn generate_sfx(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> std::result::Result<blazen_audio::GeneratedAudio, blazen_audio_music::MusicError> {
        Err(blazen_audio_music::MusicError::EngineNotAvailable)
    }
}

// ---------------------------------------------------------------------------
// Shared streaming driver — delegates to the musicgen-feature version
// when available; otherwise drops in a local copy.
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-music-musicgen")]
async fn drive(
    stream: std::pin::Pin<
        Box<
            dyn futures_util::Stream<
                    Item = std::result::Result<
                        blazen_audio_music::MusicChunk,
                        blazen_audio_music::MusicError,
                    >,
                > + Send,
        >,
    >,
    on_chunk: &StreamMusicChunkCallbackTsfn,
) -> Result<()> {
    drive_music_stream(stream, on_chunk).await
}

#[cfg(not(feature = "audio-music-musicgen"))]
async fn drive(
    stream: std::pin::Pin<
        Box<
            dyn futures_util::Stream<
                    Item = std::result::Result<
                        blazen_audio_music::MusicChunk,
                        blazen_audio_music::MusicError,
                    >,
                > + Send,
        >,
    >,
    on_chunk: &StreamMusicChunkCallbackTsfn,
) -> Result<()> {
    use crate::music::chunk::build_music_chunk;
    use futures_util::StreamExt;
    use napi::threadsafe_function::ThreadsafeFunctionCallMode;
    let mut stream = std::pin::pin!(stream);
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                on_chunk.call(
                    build_music_chunk(chunk),
                    ThreadsafeFunctionCallMode::Blocking,
                );
            }
            Err(e) => return Err(music_error_to_napi(e)),
        }
    }
    Ok(())
}
