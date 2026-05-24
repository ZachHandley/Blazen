// See `musicgen.rs` for the rationale behind these crate-wide allows.
#![allow(clippy::doc_markdown, clippy::unused_async)]

//! `AudioGenBackend` napi wrapper — Meta AudioGen text-to-SFX (also
//! text-to-music; the model class is identical to MusicGen and the
//! prompt is the only discriminator).
//!
//! Construction is cheap (weights download lazily on the first
//! `generateMusic` / `generateSfx` call). With the `audio-music-audiogen`
//! cargo feature OFF the engine surfaces
//! `MusicEngineNotAvailableError` from every entry point but the typed
//! class still exists so JS callers can use `instanceof` regardless.

#[cfg(feature = "audio-music-audiogen")]
use std::path::PathBuf;
#[cfg(feature = "audio-music-audiogen")]
use std::sync::Arc;

#[cfg(feature = "audio-music-audiogen")]
use blazen_audio_music::MusicBackend;
use napi::Result;
use napi_derive::napi;

#[cfg(feature = "audio-music-audiogen")]
use blazen_audio_music::{AudioGenBackend, AudioGenConfig};

use crate::error::music_error_to_napi;
use crate::music::StreamMusicChunkCallbackTsfn;
use crate::music::chunk::JsMusicResult;
#[cfg(feature = "audio-music-audiogen")]
use crate::music::chunk::build_music_result;
#[cfg(feature = "audio-music-audiogen")]
use crate::music::musicgen::drive_music_stream;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Construction-time options for [`JsAudioGenBackend`]. All fields
/// optional — defaults match `facebook/audiogen-medium`.
#[napi(object)]
pub struct JsAudioGenOptions {
    /// Override the Hugging Face Hub repo id. Defaults to
    /// `"facebook/audiogen-medium"`.
    #[napi(js_name = "repoId")]
    pub repo_id: Option<String>,
    /// Optional pinned revision (commit SHA or tag) for the HF repo.
    pub revision: Option<String>,
    /// Optional override for the Hugging Face cache directory.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
    /// Hard safety cap on the requested duration (seconds). Defaults to
    /// 30 s. AudioGen-medium's absolute upper bound is 30 s; requests past
    /// either limit surface `MusicInvalidInputError`.
    #[napi(js_name = "maxDurationSeconds")]
    pub max_duration_seconds: Option<f64>,
}

// ---------------------------------------------------------------------------
// Backend wrapper (real impl)
// ---------------------------------------------------------------------------

/// AudioGen text-to-SFX + text-to-music backend.
///
/// Use the [`JsAudioGenBackend::create`] factory to construct an instance.
#[cfg(feature = "audio-music-audiogen")]
#[napi(js_name = "AudioGenBackend")]
pub struct JsAudioGenBackend {
    inner: Arc<AudioGenBackend>,
    model_id: String,
}

#[cfg(feature = "audio-music-audiogen")]
impl JsAudioGenBackend {
    pub(crate) fn arc(&self) -> Arc<AudioGenBackend> {
        Arc::clone(&self.inner)
    }
}

#[cfg(feature = "audio-music-audiogen")]
#[napi]
impl JsAudioGenBackend {
    /// Construct an AudioGen backend handle.
    #[napi(factory)]
    #[must_use]
    pub fn create(options: Option<JsAudioGenOptions>) -> Self {
        let opts = options.unwrap_or(JsAudioGenOptions {
            repo_id: None,
            revision: None,
            cache_dir: None,
            max_duration_seconds: None,
        });
        let mut config = AudioGenConfig::default();
        if let Some(repo) = opts.repo_id {
            config.repo_id = repo;
        }
        if let Some(rev) = opts.revision {
            config.revision = Some(rev);
        }
        if let Some(dir) = opts.cache_dir {
            config.cache_dir = Some(PathBuf::from(dir));
        }
        #[allow(clippy::cast_possible_truncation)]
        if let Some(cap) = opts.max_duration_seconds {
            config.max_duration_seconds = cap as f32;
        }
        Self {
            inner: Arc::new(AudioGenBackend::new(config)),
            model_id: "audiogen-medium".to_string(),
        }
    }

    /// Backend identifier, always `"audiogen-medium"`.
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
// Backend wrapper (stub when `audio-music-audiogen` feature is OFF)
// ---------------------------------------------------------------------------

/// AudioGen text-to-SFX + text-to-music backend — stub fallback.
///
/// With the `audio-music-audiogen` cargo feature OFF every `generate*`
/// entry point surfaces `MusicEngineNotAvailableError`. The class still
/// exists so `instanceof AudioGenBackend` continues to type-check from
/// JS regardless of the build's feature set.
#[cfg(not(feature = "audio-music-audiogen"))]
#[napi(js_name = "AudioGenBackend")]
pub struct JsAudioGenBackend {
    model_id: String,
}

#[cfg(not(feature = "audio-music-audiogen"))]
#[napi]
impl JsAudioGenBackend {
    /// Construct an AudioGen backend handle (stub fallback).
    #[napi(factory)]
    #[must_use]
    pub fn create(_options: Option<JsAudioGenOptions>) -> Self {
        Self {
            model_id: "audiogen-medium".to_string(),
        }
    }

    /// Backend identifier, always `"audiogen-medium"`.
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
