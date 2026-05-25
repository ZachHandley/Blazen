// See `rvc.rs` for the rationale behind these crate-wide allows.
#![allow(clippy::doc_markdown, clippy::unused_async)]

//! `VcModel` — unified aggregator exposing a `.rvc(opts)` factory that
//! returns a single `VcModel` whose `convertVoice` / `streamConvertPcm`
//! / `listTargetVoices` / `registerTargetVoice` methods dispatch
//! through a dyn-trait `VoiceConversionBackend`.
//!
//! Mirrors the convenience aggregator pattern used by `MusicModel` /
//! `Model` / `EmbeddingModel` in the rest of the binding so callers can
//! swap backends without touching call sites.

use napi::Result;
use napi::bindgen_prelude::Float32Array;
use napi_derive::napi;

#[cfg(feature = "audio-vc-rvc")]
use std::path::Path;
#[cfg(feature = "audio-vc-rvc")]
use std::sync::Arc;

#[cfg(feature = "audio-vc-rvc")]
use blazen_audio_vc::VoiceConversionBackend;

#[cfg(feature = "audio-vc")]
use crate::error::vc_error_to_napi;
use crate::vc::StreamVcChunkCallbackTsfn;
use crate::vc::chunk::{JsTargetVoice, JsVcResult};
#[cfg(feature = "audio-vc-rvc")]
use crate::vc::chunk::{build_target_voice, build_vc_result};
#[cfg(feature = "audio-vc-rvc")]
use crate::vc::rvc::drive_vc_stream;
use crate::vc::rvc::{JsRvcBackend, JsRvcOptions};

// ---------------------------------------------------------------------------
// Aggregator wrapper
// ---------------------------------------------------------------------------

/// Backing storage strategy for [`JsVcModel`].
///
/// Currently only the RVC backend is wired; both real and stub modes
/// collapse to a single `Ready` variant that holds a shared backend
/// handle (or, in stub mode, just the model-id string the stub class
/// reports).
enum Inner {
    /// Real backend handle when `audio-vc-rvc` is ON.
    #[cfg(feature = "audio-vc-rvc")]
    Ready(Arc<dyn VoiceConversionBackend + Send + Sync>),
    /// Stub mode: no backend, every call surfaces
    /// `VcEngineNotAvailableError`.
    #[cfg(not(feature = "audio-vc-rvc"))]
    EngineUnavailable,
}

/// Unified voice-conversion backend aggregator.
///
/// ```javascript
/// // Pick a backend at construction time:
/// const m = VcModel.rvc({ topK: 8, retrievalBlend: 0.75 });
/// const result = await m.convertVoice('input.wav', 'speaker-01');
/// ```
#[napi(js_name = "VcModel")]
pub struct JsVcModel {
    inner: Inner,
    model_id: String,
}

#[napi]
impl JsVcModel {
    /// Build a [`JsVcModel`] backed by RVC.
    #[napi(factory)]
    #[must_use]
    pub fn rvc(options: Option<JsRvcOptions>) -> Self {
        let backend = JsRvcBackend::create(options);
        let model_id = backend.model_id();
        let inner = build_rvc_inner(&backend);
        Self { inner, model_id }
    }

    /// Backend identifier — same value `modelId` returns on the per-
    /// backend `#[napi]` class (e.g. `"rvc"`).
    #[napi(js_name = "modelId", getter)]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Convert a source utterance to the voice of a registered target
    /// speaker.
    ///
    /// # Errors
    /// See per-backend documentation
    /// ([`JsRvcBackend::convert_voice`]).
    #[napi(js_name = "convertVoice")]
    pub async fn convert_voice(
        &self,
        input_audio_path: String,
        target_voice_id: String,
    ) -> Result<JsVcResult> {
        #[cfg(feature = "audio-vc-rvc")]
        {
            let backend = self.resolve_backend();
            let bytes = backend
                .convert_voice(Path::new(&input_audio_path), &target_voice_id)
                .await
                .map_err(vc_error_to_napi)?;
            Ok(build_vc_result(bytes))
        }
        #[cfg(not(feature = "audio-vc-rvc"))]
        {
            let _ = (input_audio_path, target_voice_id);
            Err(stub_engine_not_available())
        }
    }

    /// Stream voice conversion over an in-memory PCM buffer.
    ///
    /// # Errors
    /// See per-backend documentation
    /// ([`JsRvcBackend::stream_convert_pcm`]).
    #[napi(js_name = "streamConvertPcm")]
    pub async fn stream_convert_pcm(
        &self,
        input_samples: Float32Array,
        target_voice_id: String,
        on_chunk: StreamVcChunkCallbackTsfn,
    ) -> Result<()> {
        #[cfg(feature = "audio-vc-rvc")]
        {
            let backend = self.resolve_backend();
            let samples = input_samples.to_vec();
            let input_stream = futures_util::stream::iter([samples]);
            let stream = backend
                .stream_convert(Box::pin(input_stream), &target_voice_id)
                .await
                .map_err(vc_error_to_napi)?;
            drive_vc_stream(stream, &on_chunk).await
        }
        #[cfg(not(feature = "audio-vc-rvc"))]
        {
            let _ = (input_samples, target_voice_id, on_chunk);
            Err(stub_engine_not_available())
        }
    }

    /// List the target voices the active backend can currently render.
    ///
    /// # Errors
    /// See per-backend documentation.
    #[napi(js_name = "listTargetVoices")]
    pub async fn list_target_voices(&self) -> Result<Vec<JsTargetVoice>> {
        #[cfg(feature = "audio-vc-rvc")]
        {
            let backend = self.resolve_backend();
            let voices = backend
                .list_target_voices()
                .await
                .map_err(vc_error_to_napi)?;
            Ok(voices.into_iter().map(build_target_voice).collect())
        }
        #[cfg(not(feature = "audio-vc-rvc"))]
        {
            Err(stub_engine_not_available())
        }
    }

    /// Register a new target voice from a reference utterance.
    ///
    /// # Errors
    /// See per-backend documentation.
    #[napi(js_name = "registerTargetVoice")]
    pub async fn register_target_voice(
        &self,
        voice_id: String,
        reference_audio_path: String,
    ) -> Result<()> {
        #[cfg(feature = "audio-vc-rvc")]
        {
            let backend = self.resolve_backend();
            backend
                .register_target_voice(&voice_id, Path::new(&reference_audio_path))
                .await
                .map_err(vc_error_to_napi)
        }
        #[cfg(not(feature = "audio-vc-rvc"))]
        {
            let _ = (voice_id, reference_audio_path);
            Err(stub_engine_not_available())
        }
    }
}

impl JsVcModel {
    #[cfg(feature = "audio-vc-rvc")]
    fn resolve_backend(&self) -> Arc<dyn VoiceConversionBackend + Send + Sync> {
        match &self.inner {
            Inner::Ready(b) => Arc::clone(b),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-backend Inner builders
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-vc-rvc")]
fn build_rvc_inner(backend: &JsRvcBackend) -> Inner {
    Inner::Ready(backend.arc() as Arc<dyn VoiceConversionBackend + Send + Sync>)
}

#[cfg(not(feature = "audio-vc-rvc"))]
fn build_rvc_inner(_backend: &JsRvcBackend) -> Inner {
    Inner::EngineUnavailable
}

#[cfg(not(feature = "audio-vc-rvc"))]
fn stub_engine_not_available() -> napi::Error {
    napi::Error::with_class(
        "VcEngineNotAvailableError",
        "voice-conversion engine not available: rvc feature disabled at build time",
    )
}
