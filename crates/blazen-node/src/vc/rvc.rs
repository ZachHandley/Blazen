// See `mod.rs` for the rationale behind these crate-wide allows.
//
// `unused_async`: stub fallbacks (when the engine feature is OFF)
// return an error immediately without awaiting; we still need the async
// signature so the public API is shape-identical across feature modes.
#![allow(clippy::doc_markdown, clippy::unused_async)]

//! `RvcBackend` napi wrapper — Retrieval-based Voice Conversion.
//!
//! Construction is cheap (weights download lazily on the first
//! `convertVoice` / `streamConvertPcm` call). With the `audio-vc-rvc`
//! cargo feature OFF the engine surfaces `VcEngineNotAvailableError`
//! from every entry point but the typed class still exists so JS
//! callers can use `instanceof` regardless.

#[cfg(feature = "audio-vc-rvc")]
use std::path::Path;
#[cfg(feature = "audio-vc-rvc")]
use std::sync::Arc;

#[cfg(feature = "audio-vc-rvc")]
use blazen_audio_vc::VoiceConversionBackend;
use napi::Result;
#[cfg(feature = "audio-vc-rvc")]
use napi::bindgen_prelude::Float32Array;
#[cfg(feature = "audio-vc-rvc")]
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;

#[cfg(feature = "audio-vc-rvc")]
use blazen_audio_vc::RvcBackend;

#[cfg(feature = "audio-vc")]
use crate::error::vc_error_to_napi;
use crate::vc::StreamVcChunkCallbackTsfn;
use crate::vc::chunk::{JsTargetVoice, JsVcResult};
#[cfg(feature = "audio-vc-rvc")]
use crate::vc::chunk::{build_target_voice, build_vc_chunk, build_vc_result};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Construction-time options for [`JsRvcBackend`]. All fields optional
/// — defaults match the upstream RVC reference (top-k = 8, blend = 0.75,
/// V2 content encoder).
#[napi(object)]
pub struct JsRvcOptions {
    /// kNN neighbour count for the retrieval blend (`top_k`). Defaults
    /// to 8. Clamped to `>= 1` at query time.
    #[napi(js_name = "topK")]
    pub top_k: Option<u32>,
    /// Retrieval blend factor (`index_rate` in the upstream
    /// reference). Defaults to 0.75. Clamped into `[0.0, 1.0]`.
    #[napi(js_name = "retrievalBlend")]
    pub retrieval_blend: Option<f64>,
    /// Which ContentVec family to use for the shared HuBERT encoder.
    /// One of `"v1"` or `"v2"` (case-insensitive). Defaults to `"v2"`
    /// — the family contemporary RVC checkpoints target.
    #[napi(js_name = "rvcVersion")]
    pub rvc_version: Option<String>,
}

// ---------------------------------------------------------------------------
// Backend wrapper (real impl)
// ---------------------------------------------------------------------------

/// Retrieval-based Voice Conversion backend.
///
/// Use the [`JsRvcBackend::create`] factory to construct an instance.
#[cfg(feature = "audio-vc-rvc")]
#[napi(js_name = "RvcBackend")]
pub struct JsRvcBackend {
    inner: Arc<RvcBackend>,
    model_id: String,
}

#[cfg(feature = "audio-vc-rvc")]
impl JsRvcBackend {
    pub(crate) fn arc(&self) -> Arc<RvcBackend> {
        Arc::clone(&self.inner)
    }
}

#[cfg(feature = "audio-vc-rvc")]
#[napi]
impl JsRvcBackend {
    /// Construct an RVC backend handle.
    #[napi(factory)]
    #[must_use]
    pub fn create(options: Option<JsRvcOptions>) -> Self {
        let opts = options.unwrap_or(JsRvcOptions {
            top_k: None,
            retrieval_blend: None,
            rvc_version: None,
        });
        let mut backend = RvcBackend::new();
        if let Some(k) = opts.top_k {
            backend = backend.with_top_k(k as usize);
        }
        #[allow(clippy::cast_possible_truncation)]
        if let Some(blend) = opts.retrieval_blend {
            backend = backend.with_retrieval_blend(blend as f32);
        }
        if let Some(version) = opts.rvc_version {
            let parsed = parse_rvc_version(&version);
            backend = backend.with_rvc_version(parsed);
        }
        Self {
            inner: Arc::new(backend),
            model_id: "rvc".to_string(),
        }
    }

    /// Backend identifier, always `"rvc"`.
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Convert a source utterance to the voice of a registered target
    /// speaker, returning the rendered audio as a self-describing WAV
    /// payload + parsed sample-rate / duration metadata.
    ///
    /// # Errors
    /// Returns `VcVoiceNotFoundError` when `targetVoiceId` is not
    /// registered, `VcIoError` on file-read failures, `VcModelLoadError`
    /// on weight-load failures, `VcConversionError` on inference
    /// failures, or `VcEngineNotAvailableError` when the engine
    /// feature was compiled out.
    #[napi(js_name = "convertVoice")]
    pub async fn convert_voice(
        &self,
        input_audio_path: String,
        target_voice_id: String,
    ) -> Result<JsVcResult> {
        let bytes = self
            .inner
            .convert_voice(Path::new(&input_audio_path), &target_voice_id)
            .await
            .map_err(vc_error_to_napi)?;
        Ok(build_vc_result(bytes))
    }

    /// Stream voice conversion over an in-memory PCM buffer, invoking
    /// `onChunk` for each emitted [`crate::vc::JsVcChunk`] until the
    /// stream ends (the last chunk arrives with `isFinal === true`).
    ///
    /// The input samples are wrapped in a single-item stream and fed to
    /// the backend's chunked streaming entry point; the backend
    /// internally buffers windows (typically 2 seconds at 16 kHz) and
    /// emits the converted PCM at the target voice's native sample
    /// rate.
    ///
    /// # Errors
    /// Same surface as [`Self::convert_voice`]; additionally surfaces
    /// `VcUnsupportedError` from a backend that does not support
    /// streaming (the default-impl path).
    #[napi(js_name = "streamConvertPcm")]
    pub async fn stream_convert_pcm(
        &self,
        input_samples: Float32Array,
        target_voice_id: String,
        on_chunk: StreamVcChunkCallbackTsfn,
    ) -> Result<()> {
        let samples = input_samples.to_vec();
        let input_stream = futures_util::stream::iter([samples]);
        let stream = self
            .inner
            .stream_convert(Box::pin(input_stream), &target_voice_id)
            .await
            .map_err(vc_error_to_napi)?;
        drive_vc_stream(stream, &on_chunk).await
    }

    /// List the target voices this backend can currently render.
    ///
    /// # Errors
    /// Returns `VcUnsupportedError` from backends that don't expose a
    /// voice catalogue; `VcIoError` when probing the voice directory
    /// fails.
    #[napi(js_name = "listTargetVoices")]
    pub async fn list_target_voices(&self) -> Result<Vec<JsTargetVoice>> {
        let voices = self
            .inner
            .list_target_voices()
            .await
            .map_err(vc_error_to_napi)?;
        Ok(voices.into_iter().map(build_target_voice).collect())
    }

    /// Register a new target voice from a reference utterance.
    ///
    /// RVC voice registration is intentionally unsupported at runtime
    /// (training a voice profile requires an offline pipeline of 1+
    /// hours); this method therefore surfaces
    /// `VcUnsupportedError`. Pre-trained voice profiles can be placed
    /// under `$BLAZEN_RVC_VOICE_DIR/<voice_id>/` and will surface
    /// through [`Self::list_target_voices`] / [`Self::convert_voice`].
    ///
    /// # Errors
    /// Returns `VcUnsupportedError` from RVC; other backends may
    /// override this with a real implementation.
    #[napi(js_name = "registerTargetVoice")]
    pub async fn register_target_voice(
        &self,
        voice_id: String,
        reference_audio_path: String,
    ) -> Result<()> {
        self.inner
            .register_target_voice(&voice_id, Path::new(&reference_audio_path))
            .await
            .map_err(vc_error_to_napi)
    }
}

// ---------------------------------------------------------------------------
// Backend wrapper (stub when `audio-vc-rvc` feature is OFF)
// ---------------------------------------------------------------------------

/// Retrieval-based Voice Conversion backend — stub fallback.
///
/// With the `audio-vc-rvc` cargo feature OFF every `convert*` /
/// `list*` / `register*` entry point surfaces
/// `VcEngineNotAvailableError`. The class still exists so
/// `instanceof RvcBackend` continues to type-check from JS regardless
/// of the build's feature set.
#[cfg(not(feature = "audio-vc-rvc"))]
#[napi(js_name = "RvcBackend")]
pub struct JsRvcBackend {
    model_id: String,
}

#[cfg(not(feature = "audio-vc-rvc"))]
#[napi]
impl JsRvcBackend {
    /// Construct an RVC backend handle (stub fallback).
    #[napi(factory)]
    #[must_use]
    pub fn create(_options: Option<JsRvcOptions>) -> Self {
        Self {
            model_id: "rvc".to_string(),
        }
    }

    /// Backend identifier, always `"rvc"`.
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Convert a source utterance — always surfaces
    /// `VcEngineNotAvailableError` in stub mode.
    ///
    /// # Errors
    /// Always returns `VcEngineNotAvailableError`.
    #[napi(js_name = "convertVoice")]
    pub async fn convert_voice(
        &self,
        _input_audio_path: String,
        _target_voice_id: String,
    ) -> Result<JsVcResult> {
        Err(stub_engine_not_available())
    }

    /// Stream voice conversion — always surfaces
    /// `VcEngineNotAvailableError` in stub mode.
    ///
    /// # Errors
    /// Always returns `VcEngineNotAvailableError`.
    #[napi(js_name = "streamConvertPcm")]
    pub async fn stream_convert_pcm(
        &self,
        _input_samples: napi::bindgen_prelude::Float32Array,
        _target_voice_id: String,
        _on_chunk: StreamVcChunkCallbackTsfn,
    ) -> Result<()> {
        Err(stub_engine_not_available())
    }

    /// List target voices — always surfaces
    /// `VcEngineNotAvailableError` in stub mode.
    ///
    /// # Errors
    /// Always returns `VcEngineNotAvailableError`.
    #[napi(js_name = "listTargetVoices")]
    pub async fn list_target_voices(&self) -> Result<Vec<JsTargetVoice>> {
        Err(stub_engine_not_available())
    }

    /// Register a target voice — always surfaces
    /// `VcEngineNotAvailableError` in stub mode.
    ///
    /// # Errors
    /// Always returns `VcEngineNotAvailableError`.
    #[napi(js_name = "registerTargetVoice")]
    pub async fn register_target_voice(
        &self,
        _voice_id: String,
        _reference_audio_path: String,
    ) -> Result<()> {
        Err(stub_engine_not_available())
    }
}

#[cfg(not(feature = "audio-vc-rvc"))]
fn stub_engine_not_available() -> napi::Error {
    napi::Error::with_class(
        "VcEngineNotAvailableError",
        "voice-conversion engine not available: rvc feature disabled at build time",
    )
}

// ---------------------------------------------------------------------------
// Shared streaming driver
// ---------------------------------------------------------------------------

/// Drive an upstream voice-conversion stream, invoking `on_chunk` for
/// each yielded sample buffer and surfacing the first inference error
/// as a `napi::Error`. The final emission carries `isFinal = true`;
/// intermediate emissions carry `isFinal = false`.
#[cfg(feature = "audio-vc-rvc")]
pub(crate) async fn drive_vc_stream(
    stream: std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = std::result::Result<Vec<f32>, blazen_audio_vc::VcError>>
                + Send,
        >,
    >,
    on_chunk: &StreamVcChunkCallbackTsfn,
) -> Result<()> {
    use futures_util::StreamExt;
    let mut stream = std::pin::pin!(stream);
    // We need to know whether each yielded chunk is the last so the
    // `isFinal` flag is set correctly. Buffer one item ahead.
    let mut pending: Option<Vec<f32>> = None;
    while let Some(item) = stream.next().await {
        let samples = item.map_err(vc_error_to_napi)?;
        if let Some(prev) = pending.take() {
            on_chunk.call(
                build_vc_chunk(prev, false),
                ThreadsafeFunctionCallMode::Blocking,
            );
        }
        pending = Some(samples);
    }
    if let Some(last) = pending {
        on_chunk.call(
            build_vc_chunk(last, true),
            ThreadsafeFunctionCallMode::Blocking,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a JS-supplied RVC content-encoder version string into the
/// upstream enum. Unrecognised values fall back to V2 (the default).
#[cfg(feature = "audio-vc-rvc")]
fn parse_rvc_version(s: &str) -> blazen_audio_vc::backends::rvc::content::RvcVersion {
    use blazen_audio_vc::backends::rvc::content::RvcVersion;
    match s.trim().to_ascii_lowercase().as_str() {
        "v1" | "1" => RvcVersion::V1,
        _ => RvcVersion::V2,
    }
}
