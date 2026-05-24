// Product names referenced in docs (MusicGen, Hugging Face, ...) match
// the convention used by the sibling `blazen-audio-music` crate, which
// also allows this lint crate-wide.
//
// `unused_async`: stub fallbacks (when the engine feature is OFF) return
// an error immediately without awaiting; we still need the async signature
// so the public API is shape-identical across feature modes.
#![allow(clippy::doc_markdown, clippy::unused_async)]

//! `MusicgenBackend` napi wrapper — Meta MusicGen text-to-music + SFX.
//!
//! Construction is cheap (weights download lazily on the first
//! `generateMusic` / `generateSfx` call). With the `audio-music-musicgen`
//! cargo feature OFF the engine surfaces
//! `MusicEngineNotAvailableError` from every entry point but the typed
//! class still exists so JS callers can use `instanceof` regardless.

#[cfg(feature = "audio-music-musicgen")]
use std::path::PathBuf;
#[cfg(feature = "audio-music-musicgen")]
use std::sync::Arc;

#[cfg(feature = "audio-music-musicgen")]
use blazen_audio_music::MusicBackend;
use napi::Result;
#[cfg(feature = "audio-music-musicgen")]
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;

#[cfg(feature = "audio-music-musicgen")]
use blazen_audio_music::backends::musicgen::{MusicgenBackend, MusicgenConfig, MusicgenVariant};

use crate::error::music_error_to_napi;
use crate::music::StreamMusicChunkCallbackTsfn;
use crate::music::chunk::JsMusicResult;
#[cfg(feature = "audio-music-musicgen")]
use crate::music::chunk::{build_music_chunk, build_music_result};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Available MusicGen checkpoints on Hugging Face Hub.
#[napi(string_enum)]
#[derive(Debug, Clone, Copy)]
pub enum JsMusicgenVariant {
    /// `facebook/musicgen-small` -- ~300M params, 32 kHz mono.
    #[napi(value = "small")]
    Small,
    /// `facebook/musicgen-medium` -- ~1.5B params, 32 kHz mono.
    #[napi(value = "medium")]
    Medium,
    /// `facebook/musicgen-large` -- ~3.3B params, 32 kHz mono.
    #[napi(value = "large")]
    Large,
}

#[cfg(feature = "audio-music-musicgen")]
impl From<JsMusicgenVariant> for MusicgenVariant {
    fn from(value: JsMusicgenVariant) -> Self {
        match value {
            JsMusicgenVariant::Small => Self::Small,
            JsMusicgenVariant::Medium => Self::Medium,
            JsMusicgenVariant::Large => Self::Large,
        }
    }
}

/// Construction-time options for [`JsMusicgenBackend`]. All fields
/// optional — defaults match the small CPU-friendly variant.
#[napi(object)]
pub struct JsMusicgenOptions {
    /// Which checkpoint to load. Defaults to `"small"`.
    pub variant: Option<JsMusicgenVariant>,
    /// Optional override for the Hugging Face cache directory.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
    /// Hard safety cap on the requested duration (seconds). Defaults to
    /// 30 s. The absolute upper bound enforced by MusicGen itself is
    /// 60 s — requests past either limit surface `MusicInvalidInputError`.
    #[napi(js_name = "maxDurationSeconds")]
    pub max_duration_seconds: Option<f64>,
}

// ---------------------------------------------------------------------------
// Backend wrapper (real impl)
// ---------------------------------------------------------------------------

/// MusicGen text-to-music + text-to-SFX backend.
///
/// Use the [`JsMusicgenBackend::create`] factory to construct an instance.
#[cfg(feature = "audio-music-musicgen")]
#[napi(js_name = "MusicgenBackend")]
pub struct JsMusicgenBackend {
    inner: Arc<MusicgenBackend>,
    model_id: String,
}

#[cfg(feature = "audio-music-musicgen")]
impl JsMusicgenBackend {
    pub(crate) fn arc(&self) -> Arc<MusicgenBackend> {
        Arc::clone(&self.inner)
    }
}

#[cfg(feature = "audio-music-musicgen")]
#[napi]
impl JsMusicgenBackend {
    /// Construct a MusicGen backend handle.
    ///
    /// # Errors
    /// Returns the resulting `napi::Error` if option conversion fails;
    /// in practice always succeeds.
    #[napi(factory)]
    #[must_use]
    pub fn create(options: Option<JsMusicgenOptions>) -> Self {
        let opts = options.unwrap_or(JsMusicgenOptions {
            variant: None,
            cache_dir: None,
            max_duration_seconds: None,
        });
        let variant = opts.variant.map_or(MusicgenVariant::Small, Into::into);
        let model_id = musicgen_model_id(variant).to_string();
        let mut config = MusicgenConfig {
            variant,
            device: None,
            cache_dir: opts.cache_dir.map(PathBuf::from),
            ..MusicgenConfig::default()
        };
        #[allow(clippy::cast_possible_truncation)]
        if let Some(cap) = opts.max_duration_seconds {
            config.max_duration_seconds = cap as f32;
        }
        Self {
            inner: Arc::new(MusicgenBackend::new(config)),
            model_id,
        }
    }

    /// Backend identifier, e.g. `"musicgen-small"`.
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Generate music conditioned on `prompt`.
    ///
    /// # Errors
    /// Returns `MusicInvalidInputError` for empty prompts or non-positive
    /// / out-of-range durations, `MusicHfHubError` on weight-download
    /// failure, `MusicCandleError` on inference failure, or
    /// `MusicEngineNotAvailableError` when the engine feature was
    /// compiled out.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        #[allow(clippy::cast_possible_truncation)]
        let generated = self
            .inner
            .generate_music(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        Ok(build_music_result(generated))
    }

    /// Generate sound-effect audio conditioned on `prompt`.
    ///
    /// MusicGen treats music and SFX as the same autoregressive pipeline
    /// (the prompt is the only discriminator).
    ///
    /// # Errors
    /// Same surface as [`Self::generate_music`].
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        #[allow(clippy::cast_possible_truncation)]
        let generated = self
            .inner
            .generate_sfx(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        Ok(build_music_result(generated))
    }

    /// Stream music generation, invoking `onChunk` for each emitted
    /// `JsMusicChunk` until the final chunk arrives (`isFinal === true`).
    ///
    /// # Errors
    /// Same surface as [`Self::generate_music`].
    #[napi(js_name = "streamGenerateMusic")]
    pub async fn stream_generate_music(
        &self,
        prompt: String,
        duration_seconds: f64,
        on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        #[allow(clippy::cast_possible_truncation)]
        let stream = self
            .inner
            .stream_generate_music(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        drive_music_stream(stream, &on_chunk).await
    }

    /// Stream SFX generation, invoking `onChunk` for each emitted
    /// `JsMusicChunk` until the final chunk arrives (`isFinal === true`).
    ///
    /// # Errors
    /// Same surface as [`Self::generate_music`].
    #[napi(js_name = "streamGenerateSfx")]
    pub async fn stream_generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f64,
        on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        #[allow(clippy::cast_possible_truncation)]
        let stream = self
            .inner
            .stream_generate_sfx(&prompt, duration_seconds as f32)
            .await
            .map_err(music_error_to_napi)?;
        drive_music_stream(stream, &on_chunk).await
    }
}

// ---------------------------------------------------------------------------
// Backend wrapper (stub when `audio-music-musicgen` feature is OFF)
// ---------------------------------------------------------------------------

/// MusicGen text-to-music + text-to-SFX backend — stub fallback.
///
/// With the `audio-music-musicgen` cargo feature OFF every `generate*`
/// entry point surfaces `MusicEngineNotAvailableError`. The class still
/// exists so `instanceof MusicgenBackend` continues to type-check from
/// JS regardless of the build's feature set.
#[cfg(not(feature = "audio-music-musicgen"))]
#[napi(js_name = "MusicgenBackend")]
pub struct JsMusicgenBackend {
    model_id: String,
}

#[cfg(not(feature = "audio-music-musicgen"))]
#[napi]
impl JsMusicgenBackend {
    /// Construct a MusicGen backend handle (stub fallback).
    #[napi(factory)]
    #[must_use]
    pub fn create(options: Option<JsMusicgenOptions>) -> Self {
        let variant = options
            .and_then(|o| o.variant)
            .unwrap_or(JsMusicgenVariant::Small);
        let model_id = match variant {
            JsMusicgenVariant::Small => "musicgen-small",
            JsMusicgenVariant::Medium => "musicgen-medium",
            JsMusicgenVariant::Large => "musicgen-large",
        };
        Self {
            model_id: model_id.to_string(),
        }
    }

    /// Backend identifier, e.g. `"musicgen-small"`.
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Generate music — always surfaces `MusicEngineNotAvailableError`
    /// in stub mode.
    ///
    /// # Errors
    /// Always returns `MusicEngineNotAvailableError`.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(
        &self,
        _prompt: String,
        _duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        Err(music_error_to_napi(
            blazen_audio_music::MusicError::EngineNotAvailable,
        ))
    }

    /// Generate SFX — always surfaces `MusicEngineNotAvailableError` in
    /// stub mode.
    ///
    /// # Errors
    /// Always returns `MusicEngineNotAvailableError`.
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(
        &self,
        _prompt: String,
        _duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        Err(music_error_to_napi(
            blazen_audio_music::MusicError::EngineNotAvailable,
        ))
    }

    /// Stream music — always surfaces `MusicEngineNotAvailableError` in
    /// stub mode.
    ///
    /// # Errors
    /// Always returns `MusicEngineNotAvailableError`.
    #[napi(js_name = "streamGenerateMusic")]
    pub async fn stream_generate_music(
        &self,
        _prompt: String,
        _duration_seconds: f64,
        _on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        Err(music_error_to_napi(
            blazen_audio_music::MusicError::EngineNotAvailable,
        ))
    }

    /// Stream SFX — always surfaces `MusicEngineNotAvailableError` in
    /// stub mode.
    ///
    /// # Errors
    /// Always returns `MusicEngineNotAvailableError`.
    #[napi(js_name = "streamGenerateSfx")]
    pub async fn stream_generate_sfx(
        &self,
        _prompt: String,
        _duration_seconds: f64,
        _on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        Err(music_error_to_napi(
            blazen_audio_music::MusicError::EngineNotAvailable,
        ))
    }
}

// ---------------------------------------------------------------------------
// Shared streaming driver
// ---------------------------------------------------------------------------

/// Drive an upstream music stream, invoking `on_chunk` for each item and
/// surfacing the first inference error as a `napi::Error`.
#[cfg(feature = "audio-music-musicgen")]
pub(crate) async fn drive_music_stream(
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
    use futures_util::StreamExt;
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the canonical model id string for a MusicGen variant. Mirrors
/// the `AudioBackend::id` returned by [`MusicgenBackend::id`] so JS-side
/// callers can compare against either source without surprises.
#[cfg(feature = "audio-music-musicgen")]
pub(crate) fn musicgen_model_id(variant: MusicgenVariant) -> &'static str {
    match variant {
        MusicgenVariant::Small => "musicgen-small",
        MusicgenVariant::Medium => "musicgen-medium",
        MusicgenVariant::Large => "musicgen-large",
    }
}
