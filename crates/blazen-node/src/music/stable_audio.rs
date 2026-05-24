// See `musicgen.rs` for the rationale behind these crate-wide allows.
#![allow(clippy::doc_markdown, clippy::unused_async)]

//! `StableAudioBackend` napi wrapper — Stability AI's Stable Audio Open
//! native candle port.
//!
//! Construction is cheap (sync). When the `audio-music-stable-audio`
//! cargo feature is ON the first `generateMusic` / `generateSfx` call
//! lazily downloads weights via `hf-hub` and loads them through the
//! upstream [`blazen_audio_music::backends::stable_audio::StableAudioBackend`].
//! When the feature is OFF, every entry point surfaces
//! `MusicNotYetImplementedError` (the upstream crate's stub fallback).

#[cfg(feature = "audio-music-stable-audio")]
use std::sync::Arc;

use blazen_audio_music::MusicBackend;
use napi::Result;
use napi_derive::napi;

use blazen_audio_music::backends::stable_audio::StableAudioBackend;
#[cfg(feature = "audio-music-stable-audio")]
use blazen_audio_music::backends::stable_audio::{StableAudioConfig, StableAudioVariant};
#[cfg(feature = "audio-music-stable-audio")]
use std::path::PathBuf;
#[cfg(feature = "audio-music-stable-audio")]
use tokio::sync::OnceCell;

use crate::error::music_error_to_napi;
use crate::music::StreamMusicChunkCallbackTsfn;
use crate::music::chunk::{JsMusicResult, build_music_result};
#[cfg(all(feature = "audio-music-musicgen", feature = "audio-music-stable-audio"))]
use crate::music::musicgen::drive_music_stream;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Hyperparameter pack describing which Stable Audio Open checkpoint to load.
#[napi(string_enum)]
#[derive(Debug, Clone, Copy)]
pub enum JsStableAudioVariant {
    /// `stabilityai/stable-audio-open-small` -- 341 M params, 8-step
    /// distilled sampler, 11 s output cap.
    #[napi(value = "small")]
    Small,
    /// `stabilityai/stable-audio-open-1.0` -- 1.21 B params, 100-step
    /// DPM-Solver++, 47 s output cap.
    #[napi(value = "open1_0")]
    Open10,
}

#[cfg(feature = "audio-music-stable-audio")]
impl From<JsStableAudioVariant> for StableAudioVariant {
    fn from(value: JsStableAudioVariant) -> Self {
        match value {
            JsStableAudioVariant::Small => Self::Small,
            JsStableAudioVariant::Open10 => Self::Open1_0,
        }
    }
}

/// Construction-time options for [`JsStableAudioBackend`]. All fields
/// optional — defaults to the Small variant on CPU with F32 precision.
#[napi(object)]
pub struct JsStableAudioOptions {
    /// Which variant to load. Defaults to `"small"`.
    pub variant: Option<JsStableAudioVariant>,
    /// Override the Hugging Face Hub repo id. Defaults to the variant's
    /// canonical repo (`stabilityai/stable-audio-open-{small,1.0}`).
    #[napi(js_name = "hfRepo")]
    pub hf_repo: Option<String>,
    /// Path to a local `tokenizer.json` for the T5 conditioner. Required
    /// when the `audio-music-stable-audio` feature is enabled; ignored in
    /// stub mode.
    #[napi(js_name = "tokenizerPath")]
    pub tokenizer_path: Option<String>,
    /// Optional override for a pre-downloaded safetensors weights file.
    /// When `None`, weights are pulled from the configured HF repo on
    /// first generation.
    #[napi(js_name = "localWeightsPath")]
    pub local_weights_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Backend wrapper
// ---------------------------------------------------------------------------

/// Stable Audio Open backend.
///
/// Use the [`JsStableAudioBackend::create`] factory to construct an
/// instance. In stub mode (feature `audio-music-stable-audio` OFF), every
/// `generate*` entry point surfaces `MusicNotYetImplementedError`.
#[napi(js_name = "StableAudioBackend")]
pub struct JsStableAudioBackend {
    #[cfg(not(feature = "audio-music-stable-audio"))]
    stub: StableAudioBackend,
    #[cfg(feature = "audio-music-stable-audio")]
    config: StableAudioConfig,
    #[cfg(feature = "audio-music-stable-audio")]
    backend: Arc<OnceCell<Arc<StableAudioBackend>>>,
    model_id: String,
}

#[cfg(feature = "audio-music-stable-audio")]
impl JsStableAudioBackend {
    async fn ensure_loaded(&self) -> Result<Arc<StableAudioBackend>> {
        let backend = self
            .backend
            .get_or_try_init(|| async {
                let cfg = self.config.clone();
                StableAudioBackend::load(cfg).await.map(Arc::new)
            })
            .await
            .map_err(music_error_to_napi)?;
        Ok(Arc::clone(backend))
    }
}

#[napi]
impl JsStableAudioBackend {
    /// Construct a Stable Audio backend handle.
    ///
    /// In stub mode (`audio-music-stable-audio` OFF), the returned
    /// handle's `generate*` calls all surface
    /// `MusicNotYetImplementedError`. With the feature ON, the first
    /// `generate*` call lazily downloads weights and loads the model.
    #[napi(factory)]
    #[must_use]
    pub fn create(options: Option<JsStableAudioOptions>) -> Self {
        let opts = options.unwrap_or(JsStableAudioOptions {
            variant: None,
            hf_repo: None,
            tokenizer_path: None,
            local_weights_path: None,
        });
        let js_variant = opts.variant.unwrap_or(JsStableAudioVariant::Small);

        #[cfg(not(feature = "audio-music-stable-audio"))]
        {
            let _ = (
                js_variant,
                &opts.hf_repo,
                &opts.tokenizer_path,
                &opts.local_weights_path,
            );
            Self {
                stub: StableAudioBackend::new(),
                model_id: "stable-audio".to_string(),
            }
        }

        #[cfg(feature = "audio-music-stable-audio")]
        {
            let variant: StableAudioVariant = js_variant.into();
            let hf_repo = opts
                .hf_repo
                .unwrap_or_else(|| variant.hf_repo().to_string());
            let tokenizer_path = opts
                .tokenizer_path
                .map_or_else(|| PathBuf::from("tokenizer.json"), PathBuf::from);
            let local_weights_path = opts.local_weights_path.map(PathBuf::from);
            let config = StableAudioConfig {
                hf_repo,
                local_weights_path,
                tokenizer_path,
                device: candle_core::Device::Cpu,
                dtype: candle_core::DType::F32,
                variant,
            };
            Self {
                config,
                backend: Arc::new(OnceCell::new()),
                model_id: "stable-audio".to_string(),
            }
        }
    }

    /// Backend identifier, always `"stable-audio"`.
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Generate music conditioned on `prompt`.
    ///
    /// # Errors
    /// Returns `MusicNotYetImplementedError` in stub mode (feature
    /// `audio-music-stable-audio` OFF). With the feature ON, may return
    /// `MusicInvalidInputError`, `MusicHfHubError`, or `MusicCandleError`.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f64,
    ) -> Result<JsMusicResult> {
        #[cfg(not(feature = "audio-music-stable-audio"))]
        {
            #[allow(clippy::cast_possible_truncation)]
            let generated = self
                .stub
                .generate_music(&prompt, duration_seconds as f32)
                .await
                .map_err(music_error_to_napi)?;
            Ok(build_music_result(generated))
        }
        #[cfg(feature = "audio-music-stable-audio")]
        {
            let backend = self.ensure_loaded().await?;
            #[allow(clippy::cast_possible_truncation)]
            let generated = backend
                .generate_music(&prompt, duration_seconds as f32)
                .await
                .map_err(music_error_to_napi)?;
            Ok(build_music_result(generated))
        }
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
        #[cfg(not(feature = "audio-music-stable-audio"))]
        {
            #[allow(clippy::cast_possible_truncation)]
            let generated = self
                .stub
                .generate_sfx(&prompt, duration_seconds as f32)
                .await
                .map_err(music_error_to_napi)?;
            Ok(build_music_result(generated))
        }
        #[cfg(feature = "audio-music-stable-audio")]
        {
            let backend = self.ensure_loaded().await?;
            #[allow(clippy::cast_possible_truncation)]
            let generated = backend
                .generate_sfx(&prompt, duration_seconds as f32)
                .await
                .map_err(music_error_to_napi)?;
            Ok(build_music_result(generated))
        }
    }

    /// Stream music generation, invoking `onChunk` for each emitted
    /// `JsMusicChunk` until the final chunk arrives (`isFinal === true`).
    ///
    /// # Errors
    /// Same surface as [`Self::generate_music`]. In stub mode (without
    /// the streaming path on the upstream trait), this surfaces
    /// `MusicNotYetImplementedError` because the trait default
    /// implementation routes there.
    #[napi(js_name = "streamGenerateMusic")]
    pub async fn stream_generate_music(
        &self,
        prompt: String,
        duration_seconds: f64,
        on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        #[cfg(not(feature = "audio-music-stable-audio"))]
        {
            let _ = (prompt, duration_seconds, on_chunk);
            Err(music_error_to_napi(
                blazen_audio_music::MusicError::not_yet_implemented(
                    "Stable Audio streaming requires the `audio-music-stable-audio` cargo feature",
                ),
            ))
        }
        #[cfg(feature = "audio-music-stable-audio")]
        {
            let backend = self.ensure_loaded().await?;
            #[allow(clippy::cast_possible_truncation)]
            let stream = backend
                .stream_generate_music(&prompt, duration_seconds as f32)
                .await
                .map_err(music_error_to_napi)?;
            #[cfg(feature = "audio-music-musicgen")]
            {
                drive_music_stream(stream, &on_chunk).await
            }
            #[cfg(not(feature = "audio-music-musicgen"))]
            {
                drive_stream_local(stream, &on_chunk).await
            }
        }
    }

    /// Stream SFX generation, invoking `onChunk` for each emitted
    /// `JsMusicChunk` until the final chunk arrives (`isFinal === true`).
    ///
    /// # Errors
    /// Same surface as [`Self::stream_generate_music`].
    #[napi(js_name = "streamGenerateSfx")]
    pub async fn stream_generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f64,
        on_chunk: StreamMusicChunkCallbackTsfn,
    ) -> Result<()> {
        #[cfg(not(feature = "audio-music-stable-audio"))]
        {
            let _ = (prompt, duration_seconds, on_chunk);
            Err(music_error_to_napi(
                blazen_audio_music::MusicError::not_yet_implemented(
                    "Stable Audio streaming requires the `audio-music-stable-audio` cargo feature",
                ),
            ))
        }
        #[cfg(feature = "audio-music-stable-audio")]
        {
            let backend = self.ensure_loaded().await?;
            #[allow(clippy::cast_possible_truncation)]
            let stream = backend
                .stream_generate_sfx(&prompt, duration_seconds as f32)
                .await
                .map_err(music_error_to_napi)?;
            #[cfg(feature = "audio-music-musicgen")]
            {
                drive_music_stream(stream, &on_chunk).await
            }
            #[cfg(not(feature = "audio-music-musicgen"))]
            {
                drive_stream_local(stream, &on_chunk).await
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback streaming driver used when `audio-music-musicgen` is OFF but
// `audio-music-stable-audio` is ON (musicgen.rs's helper is gated on the
// musicgen feature). Functionally identical.
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "audio-music-stable-audio",
    not(feature = "audio-music-musicgen")
))]
async fn drive_stream_local(
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
