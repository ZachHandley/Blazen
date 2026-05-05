//! Bridge between [`blazen_audio_whispercpp::WhisperCppProvider`] and the
//! [`Transcription`](crate::compute::Transcription) trait.
//!
//! Whisper.cpp runs fully on-device, so there is no job queue or asynchronous
//! polling: [`ComputeProvider::submit`] executes the transcription inline and
//! returns a completed [`JobHandle`].  [`ComputeProvider::status`],
//! [`ComputeProvider::result`], and [`ComputeProvider::cancel`] are stubs that
//! return [`BlazenError::Unsupported`] because the pure job-API model does not
//! map cleanly onto a local, synchronous engine -- consumers should call
//! [`Transcription::transcribe`] directly.
//!
//! When the `engine` feature on `blazen-audio-whispercpp` is not enabled, the
//! underlying provider cannot actually run inference and every call will
//! surface as a [`BlazenError::Provider`] containing the
//! `EngineNotAvailable` message.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use blazen_audio_whispercpp::{
    TranscriptionResult as EngineTranscriptionResult,
    TranscriptionSegment as EngineTranscriptionSegment, WhisperCppProvider,
};

use crate::compute::{
    ComputeProvider, ComputeRequest, ComputeResult, JobHandle, JobStatus, Transcription,
    TranscriptionRequest, TranscriptionResult, TranscriptionSegment,
};
use crate::error::BlazenError;
use crate::traits::LocalModel;
use crate::types::{MediaSource, RequestTiming};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the on-disk audio path from a [`TranscriptionRequest`].
///
/// Prefers `audio_source` when it is a [`MediaSource::File`]; otherwise falls
/// back to interpreting `audio_url` as a local path (whisper.cpp is an
/// offline backend and does not fetch URLs).
fn resolve_audio_path(request: &TranscriptionRequest) -> Result<PathBuf, BlazenError> {
    if let Some(source) = &request.audio_source {
        match source {
            MediaSource::File { path } => return Ok(path.clone()),
            MediaSource::Url { .. } => {
                return Err(BlazenError::unsupported(
                    "whisper.cpp does not support URL audio sources -- \
                     provide a local file via `TranscriptionRequest::from_file` \
                     or set `audio_source` to `MediaSource::File`",
                ));
            }
            MediaSource::Base64 { .. } => {
                return Err(BlazenError::unsupported(
                    "whisper.cpp does not support base64 audio sources -- \
                     write the audio to a temporary file and use `MediaSource::File`",
                ));
            }
            MediaSource::ProviderFile { provider, id } => {
                tracing::warn!(
                    ?provider,
                    %id,
                    "whisper.cpp received ProviderFile audio source; cannot fetch remote bytes",
                );
                return Err(BlazenError::unsupported(format!(
                    "whisper.cpp: ProviderFile audio sources require a content store to \
                     materialize bytes; fetch and supply as Base64 or File first \
                     (provider={provider:?}, id={id})"
                )));
            }
            MediaSource::Handle { handle } => {
                tracing::warn!(
                    handle.id = %handle.id,
                    handle.kind = ?handle.kind,
                    "whisper.cpp received ContentHandle audio source; no ContentStore wired here",
                );
                return Err(BlazenError::unsupported(format!(
                    "whisper.cpp: ContentHandle audio sources require a wired ContentStore to \
                     materialize bytes (handle.id={}, handle.kind={:?})",
                    handle.id, handle.kind
                )));
            }
        }
    }

    if request.audio_url.is_empty() {
        return Err(BlazenError::provider(
            "whispercpp",
            "transcription request has neither `audio_source` nor `audio_url`",
        ));
    }

    // Treat a non-empty `audio_url` as a local path for offline usage. If the
    // caller really wanted a URL we reject it to avoid surprising behaviour.
    if request.audio_url.starts_with("http://") || request.audio_url.starts_with("https://") {
        return Err(BlazenError::unsupported(
            "whisper.cpp does not support remote URLs -- \
             download the audio first and pass a local path via \
             `TranscriptionRequest::from_file`",
        ));
    }

    Ok(PathBuf::from(&request.audio_url))
}

/// Convert a whisper-engine segment into the blazen-llm
/// [`TranscriptionSegment`] shape (seconds instead of milliseconds).
fn to_blazen_segment(seg: EngineTranscriptionSegment) -> TranscriptionSegment {
    #[allow(clippy::cast_precision_loss)]
    TranscriptionSegment {
        text: seg.text,
        start: (seg.start_ms as f64) / 1000.0,
        end: (seg.end_ms as f64) / 1000.0,
        speaker: None,
    }
}

/// Convert the provider-level result into the blazen-llm public result type.
fn to_blazen_result(engine: EngineTranscriptionResult, total_ms: u64) -> TranscriptionResult {
    let audio_seconds: f64 = if engine.segments.is_empty() {
        0.0
    } else {
        let max_end = engine.segments.iter().map(|s| s.end_ms).max().unwrap_or(0);
        let min_start = engine
            .segments
            .iter()
            .map(|s| s.start_ms)
            .min()
            .unwrap_or(0);
        let span_ms = (max_end - min_start).max(0);
        #[allow(clippy::cast_precision_loss)]
        let secs = span_ms as f64 / 1000.0;
        secs
    };
    TranscriptionResult {
        text: engine.text,
        segments: engine.segments.into_iter().map(to_blazen_segment).collect(),
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
impl ComputeProvider for WhisperCppProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        "whispercpp"
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "whisper.cpp runs locally and does not use the ComputeRequest job API; \
             call `Transcription::transcribe` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "whisper.cpp does not expose a job queue -- transcription is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "whisper.cpp does not expose a job queue -- transcription is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "whisper.cpp transcription is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

#[async_trait]
impl Transcription for WhisperCppProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        if request.diarize {
            return Err(BlazenError::unsupported(
                "speaker diarization is not yet supported by the whisper.cpp backend",
            ));
        }

        let audio_path = resolve_audio_path(&request)?;
        let audio_path: &Path = audio_path.as_ref();

        tracing::info!(
            path = %audio_path.display(),
            lang = ?request.language,
            "running whisper.cpp transcription"
        );

        let start = std::time::Instant::now();

        let engine_result =
            WhisperCppProvider::transcribe(self, audio_path, request.language.as_deref())
                .await
                .map_err(|e| BlazenError::provider("whispercpp", e.to_string()))?;

        #[allow(clippy::cast_possible_truncation)]
        let total_ms = start.elapsed().as_millis() as u64;

        Ok(to_blazen_result(engine_result, total_ms))
    }
}

// ---------------------------------------------------------------------------
// LocalModel implementation
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: gives callers explicit `load`/`unload` control over
/// the underlying whisper.cpp context while preserving the existing lazy
/// auto-load-on-first-transcribe behavior provided by
/// [`WhisperCppProvider::transcribe`].
///
/// The impl forwards to the inherent methods on [`WhisperCppProvider`] and
/// wraps [`blazen_audio_whispercpp::WhisperError`] into
/// [`BlazenError::Provider`] via [`BlazenError::provider`]. The upstream
/// crate does not define a `From<WhisperError> for BlazenError` conversion
/// (and cannot, because `blazen-audio-whispercpp` does not depend on
/// `blazen-llm` -- the dependency edge runs the other way), so we do the
/// conversion inline here.
///
/// Without the upstream `engine` feature, the inherent `load`, `unload`,
/// and `is_loaded` methods on [`WhisperCppProvider`] are stubs that return
/// [`blazen_audio_whispercpp::WhisperError::EngineNotAvailable`] (for
/// `load`), succeed as no-ops (for `unload`), or return `false` (for
/// `is_loaded`). This mirrors the behavior of `transcribe` and lets
/// downstream crates depend on `LocalModel` without unconditionally
/// pulling in the heavy whisper.cpp runtime.
#[async_trait]
impl LocalModel for WhisperCppProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        WhisperCppProvider::load(self)
            .await
            .map_err(|e| BlazenError::provider("whispercpp", e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        WhisperCppProvider::unload(self)
            .await
            .map_err(|e| BlazenError::provider("whispercpp", e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        WhisperCppProvider::is_loaded(self).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_audio_whispercpp::WhisperOptions;

    #[tokio::test]
    async fn provider_id_is_whispercpp() {
        let provider = WhisperCppProvider::from_options(WhisperOptions::default())
            .await
            .expect("construction should succeed");
        assert_eq!(ComputeProvider::provider_id(&provider), "whispercpp");
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let provider = WhisperCppProvider::from_options(WhisperOptions::default())
            .await
            .expect("construction should succeed");
        let request = ComputeRequest {
            model: "whispercpp".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = provider.submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn cancel_is_unsupported() {
        let provider = WhisperCppProvider::from_options(WhisperOptions::default())
            .await
            .expect("construction should succeed");
        let handle = JobHandle {
            id: "fake".into(),
            provider: "whispercpp".into(),
            model: "whispercpp".into(),
            submitted_at: chrono::Utc::now(),
        };
        let err = provider.cancel(&handle).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn diarization_is_unsupported() {
        let provider = WhisperCppProvider::from_options(WhisperOptions::default())
            .await
            .expect("construction should succeed");
        let request = TranscriptionRequest::from_file("/tmp/fake.wav").with_diarize(true);
        let err = Transcription::transcribe(&provider, request)
            .await
            .unwrap_err();
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "expected Unsupported for diarize, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn http_url_is_rejected() {
        let provider = WhisperCppProvider::from_options(WhisperOptions::default())
            .await
            .expect("construction should succeed");
        let request = TranscriptionRequest::new("https://example.com/audio.wav");
        let err = Transcription::transcribe(&provider, request)
            .await
            .unwrap_err();
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "expected Unsupported for https URL, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn empty_source_is_rejected() {
        let provider = WhisperCppProvider::from_options(WhisperOptions::default())
            .await
            .expect("construction should succeed");
        let request = TranscriptionRequest::default();
        let err = Transcription::transcribe(&provider, request)
            .await
            .unwrap_err();
        assert!(
            matches!(err, BlazenError::Provider { .. }),
            "expected Provider error for empty source, got: {err:?}"
        );
    }

    #[test]
    fn segment_conversion_converts_ms_to_seconds() {
        let engine_seg = EngineTranscriptionSegment {
            start_ms: 1_250,
            end_ms: 3_500,
            text: "hello world".into(),
        };
        let blazen = to_blazen_segment(engine_seg);
        assert!((blazen.start - 1.25).abs() < f64::EPSILON);
        assert!((blazen.end - 3.5).abs() < f64::EPSILON);
        assert_eq!(blazen.text, "hello world");
        assert!(blazen.speaker.is_none());
    }

    #[test]
    fn result_conversion_preserves_fields() {
        let engine = EngineTranscriptionResult {
            text: "complete transcript".into(),
            segments: vec![EngineTranscriptionSegment {
                start_ms: 0,
                end_ms: 1_000,
                text: "complete transcript".into(),
            }],
            language: Some("en".into()),
        };
        let blazen = to_blazen_result(engine, 42);
        assert_eq!(blazen.text, "complete transcript");
        assert_eq!(blazen.segments.len(), 1);
        assert_eq!(blazen.language.as_deref(), Some("en"));
        assert_eq!(blazen.timing.execution_ms, Some(42));
        assert_eq!(blazen.timing.total_ms, Some(42));
        assert!(blazen.cost.is_none());
    }
}
