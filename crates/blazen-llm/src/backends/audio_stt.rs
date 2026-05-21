//! Bridge between [`blazen_audio_stt::DynSttProvider`] and the
//! [`Transcription`](crate::compute::Transcription) trait.
//!
//! Any [`blazen_audio_stt::SttBackend`] erased into a `DynSttProvider`
//! is plugged into `blazen-llm`'s compute facade through this bridge.
//! The provider is constructed up-front (typically with
//! [`WhisperCppBackend`](blazen_audio_stt::backends::WhisperCppBackend)
//! when the `whispercpp` feature is on) and transcription is driven
//! through the trait's [`transcribe`](Transcription::transcribe) method.
//!
//! [`ComputeProvider::submit`] returns
//! [`BlazenError::Unsupported`] — STT in this stack is synchronous and
//! does not use the asynchronous job-queue surface.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use blazen_audio_stt::{DynSttProvider, TranscriptionResult as EngineTranscriptionResult};

use crate::compute::{
    ComputeProvider, ComputeRequest, ComputeResult, JobHandle, JobStatus, Transcription,
    TranscriptionRequest, TranscriptionResult, TranscriptionSegment,
};
use crate::error::BlazenError;
use crate::traits::LocalModel;
use crate::types::{MediaSource, RequestTiming};

const PROVIDER_ID: &str = "blazen-audio-stt";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the on-disk audio path from a [`TranscriptionRequest`].
///
/// Prefers `audio_source` when it is a [`MediaSource::File`]; otherwise falls
/// back to interpreting `audio_url` as a local path (this bridge targets
/// offline backends, primarily whisper.cpp, which do not fetch URLs).
fn resolve_audio_path(request: &TranscriptionRequest) -> Result<PathBuf, BlazenError> {
    if let Some(source) = &request.audio_source {
        match source {
            MediaSource::File { path } => return Ok(path.clone()),
            MediaSource::Url { .. } => {
                return Err(BlazenError::unsupported(
                    "blazen-audio-stt does not support URL audio sources -- \
                     provide a local file via `TranscriptionRequest::from_file` \
                     or set `audio_source` to `MediaSource::File`",
                ));
            }
            MediaSource::Base64 { .. } => {
                return Err(BlazenError::unsupported(
                    "blazen-audio-stt does not support base64 audio sources -- \
                     write the audio to a temporary file and use `MediaSource::File`",
                ));
            }
            MediaSource::ProviderFile { provider, id } => {
                return Err(BlazenError::unsupported(format!(
                    "blazen-audio-stt: ProviderFile audio sources require a content store \
                     to materialize bytes; fetch and supply as File first \
                     (provider={provider:?}, id={id})"
                )));
            }
            MediaSource::Handle { handle } => {
                return Err(BlazenError::unsupported(format!(
                    "blazen-audio-stt: ContentHandle audio sources require a wired \
                     ContentStore to materialize bytes (handle.id={}, handle.kind={:?})",
                    handle.id, handle.kind
                )));
            }
        }
    }

    if request.audio_url.is_empty() {
        return Err(BlazenError::provider(
            PROVIDER_ID,
            "transcription request has neither `audio_source` nor `audio_url`",
        ));
    }

    if request.audio_url.starts_with("http://") || request.audio_url.starts_with("https://") {
        return Err(BlazenError::unsupported(
            "blazen-audio-stt does not support remote URLs -- \
             download the audio first and pass a local path via \
             `TranscriptionRequest::from_file`",
        ));
    }

    Ok(PathBuf::from(&request.audio_url))
}

/// Convert the audio-stt engine result into the public blazen-llm result type.
fn to_blazen_result(engine: EngineTranscriptionResult, total_ms: u64) -> TranscriptionResult {
    #[allow(clippy::cast_precision_loss)]
    let segments: Vec<TranscriptionSegment> = engine
        .segments
        .into_iter()
        .map(|s| TranscriptionSegment {
            text: s.text,
            start: (s.start_ms as f64) / 1000.0,
            end: (s.end_ms as f64) / 1000.0,
            speaker: None,
        })
        .collect();

    let audio_seconds: f64 = if segments.is_empty() {
        0.0
    } else {
        let max_end = segments.iter().map(|s| s.end).fold(0.0_f64, f64::max);
        let min_start = segments.iter().map(|s| s.start).fold(f64::MAX, f64::min);
        (max_end - min_start).max(0.0)
    };

    TranscriptionResult {
        text: engine.text,
        segments,
        language: engine.language,
        timing: RequestTiming {
            queue_ms: None,
            execution_ms: Some(total_ms),
            total_ms: Some(total_ms),
        },
        cost: None,
        usage: None,
        audio_seconds,
        metadata: serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for DynSttProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-stt runs synchronously and does not use the \
             ComputeRequest job API; call `Transcription::transcribe` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-stt does not expose a job queue -- transcription is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-stt does not expose a job queue -- transcription is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-stt transcription is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

#[async_trait]
impl Transcription for DynSttProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        if request.diarize {
            return Err(BlazenError::unsupported(
                "speaker diarization is not yet supported by blazen-audio-stt backends",
            ));
        }

        let audio_path = resolve_audio_path(&request)?;
        let audio_path: &Path = audio_path.as_ref();

        tracing::info!(
            path = %audio_path.display(),
            lang = ?request.language,
            "running blazen-audio-stt transcription"
        );

        let start = std::time::Instant::now();

        let engine_result =
            DynSttProvider::transcribe(self, audio_path, request.language.as_deref())
                .await
                .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))?;

        #[allow(clippy::cast_possible_truncation)]
        let total_ms = start.elapsed().as_millis() as u64;

        Ok(to_blazen_result(engine_result, total_ms))
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: delegates to the wrapped backend's `AudioBackend`
/// lifecycle methods. Backends with weight loading (whisper.cpp) honor
/// their own `load`/`unload`; stateless backends are always loaded.
#[async_trait]
impl LocalModel for DynSttProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        DynSttProvider::load(self)
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        DynSttProvider::unload(self)
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        DynSttProvider::is_loaded(self).await
    }

    fn device(&self) -> crate::device::Device {
        // The audio-stt surface does not yet expose a device query.
        crate::device::Device::Cpu
    }

    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-stt does not support LoRA adapters",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use blazen_audio::AudioBackend;
    use blazen_audio_stt::traits::{SttBackend, TranscriptionResult as EngineResult};
    use blazen_audio_stt::{SttError, SttProvider};

    struct StubBackend;

    #[async_trait]
    impl AudioBackend for StubBackend {
        fn id(&self) -> &'static str {
            "stub:stt"
        }
        fn provider_kind(&self) -> &'static str {
            "stt"
        }
    }

    #[async_trait]
    impl SttBackend for StubBackend {
        async fn transcribe(
            &self,
            _audio_path: &Path,
            _language: Option<&str>,
        ) -> Result<EngineResult, SttError> {
            Ok(EngineResult {
                text: "stub transcript".into(),
                segments: vec![],
                language: Some("en".into()),
            })
        }
    }

    fn provider() -> DynSttProvider {
        SttProvider::new(StubBackend).into_dyn()
    }

    #[tokio::test]
    async fn provider_id_is_blazen_audio_stt() {
        let p = provider();
        assert_eq!(ComputeProvider::provider_id(&p), PROVIDER_ID);
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let p = provider();
        let request = ComputeRequest {
            model: "stub".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = p.submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn diarization_is_unsupported() {
        let p = provider();
        let request = TranscriptionRequest::from_file("/tmp/fake.wav").with_diarize(true);
        let err = Transcription::transcribe(&p, request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn http_url_is_rejected() {
        let p = provider();
        let request = TranscriptionRequest::new("https://example.com/audio.wav");
        let err = Transcription::transcribe(&p, request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn empty_source_is_rejected() {
        let p = provider();
        let request = TranscriptionRequest::default();
        let err = Transcription::transcribe(&p, request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Provider { .. }));
    }

    #[tokio::test]
    async fn stub_local_path_transcribes() {
        let p = provider();
        let request = TranscriptionRequest::from_file("/dev/null");
        let result = Transcription::transcribe(&p, request).await.unwrap();
        assert_eq!(result.text, "stub transcript");
        assert_eq!(result.language.as_deref(), Some("en"));
    }

    #[tokio::test]
    async fn lora_adapter_unsupported() {
        let p = provider();
        let err = LocalModel::load_adapter(
            &p,
            std::path::Path::new("/dev/null"),
            crate::AdapterOptions::new("test"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }
}
