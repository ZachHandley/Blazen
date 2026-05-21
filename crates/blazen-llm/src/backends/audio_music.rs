//! Bridge between [`blazen_audio_music::DynMusicProvider`] and the
//! [`AudioGeneration`](crate::compute::AudioGeneration) trait.
//!
//! Any [`blazen_audio_music::MusicBackend`] erased into an
//! `Arc<dyn MusicBackend>` (i.e. `DynMusicProvider`) is plugged into
//! `blazen-llm`'s compute facade through this bridge. Music + SFX
//! generation are driven through the
//! [`AudioGeneration::generate_music`] / [`AudioGeneration::generate_sfx`]
//! methods.
//!
//! [`ComputeProvider::submit`] returns
//! [`BlazenError::Unsupported`] — music generation in this stack is
//! synchronous and does not use the asynchronous job-queue surface.
//!
//! `text_to_speech` is intentionally `Unsupported` here — music backends
//! do not synthesize speech; use the TTS bridge for that.

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use blazen_audio_music::{DynMusicProvider, MusicBackend, MusicError};

use crate::compute::{
    AudioGeneration, AudioResult, ComputeProvider, ComputeRequest, ComputeResult, JobHandle,
    JobStatus, MusicRequest, SpeechRequest,
};
use crate::error::BlazenError;
use crate::media::{GeneratedAudio, MediaOutput, MediaType};
use crate::traits::LocalModel;
use crate::types::RequestTiming;

const PROVIDER_ID: &str = "blazen-audio-music";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn media_type_for(format: blazen_audio::AudioFormat) -> MediaType {
    use blazen_audio::AudioFormat;
    match format {
        AudioFormat::Wav | AudioFormat::Pcm => MediaType::Wav,
        AudioFormat::Mp3 => MediaType::Mp3,
        AudioFormat::Flac => MediaType::Flac,
        AudioFormat::Opus => MediaType::Ogg,
    }
}

fn to_blazen_error(err: MusicError) -> BlazenError {
    match err {
        MusicError::EngineNotAvailable | MusicError::NotYetImplemented(_) => {
            BlazenError::unsupported(err.to_string())
        }
        MusicError::InvalidInput(msg) => BlazenError::provider(PROVIDER_ID, msg),
        other => BlazenError::provider(PROVIDER_ID, other.to_string()),
    }
}

fn audio_to_result(audio: &blazen_audio::GeneratedAudio, total_ms: u64) -> AudioResult {
    let audio_len = u64::try_from(audio.bytes.len()).unwrap_or(u64::MAX);
    let mut media =
        MediaOutput::from_base64(BASE64.encode(&audio.bytes), media_type_for(audio.format));
    media.file_size = Some(audio_len);

    let clip = GeneratedAudio {
        media,
        duration_seconds: audio.duration_seconds,
        sample_rate: Some(audio.sample_rate),
        channels: Some(u8::try_from(audio.channels).unwrap_or(1)),
    };

    let audio_seconds = f64::from(audio.duration_seconds.unwrap_or(0.0));

    AudioResult {
        audio: vec![clip],
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
impl ComputeProvider for DynMusicProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-music runs synchronously and does not use the \
             ComputeRequest job API; call `AudioGeneration::generate_music` / \
             `generate_sfx` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-music does not expose a job queue -- generation is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-music does not expose a job queue -- generation is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-music generation is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioGeneration for DynMusicProvider {
    async fn text_to_speech(&self, _request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-music is for music + sound-effect generation; \
             for text-to-speech use the blazen-audio-tts bridge instead",
        ))
    }

    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        if request.prompt.is_empty() {
            return Err(BlazenError::provider(
                PROVIDER_ID,
                "generate_music request has empty `prompt`",
            ));
        }

        let duration = request.duration_seconds.unwrap_or(10.0);
        let start = std::time::Instant::now();

        let backend: &dyn MusicBackend = self.as_ref();
        let audio = backend
            .generate_music(&request.prompt, duration)
            .await
            .map_err(to_blazen_error)?;

        #[allow(clippy::cast_possible_truncation)]
        let total_ms = start.elapsed().as_millis() as u64;

        Ok(audio_to_result(&audio, total_ms))
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        if request.prompt.is_empty() {
            return Err(BlazenError::provider(
                PROVIDER_ID,
                "generate_sfx request has empty `prompt`",
            ));
        }

        let duration = request.duration_seconds.unwrap_or(5.0);
        let start = std::time::Instant::now();

        let backend: &dyn MusicBackend = self.as_ref();
        let audio = backend
            .generate_sfx(&request.prompt, duration)
            .await
            .map_err(to_blazen_error)?;

        #[allow(clippy::cast_possible_truncation)]
        let total_ms = start.elapsed().as_millis() as u64;

        Ok(audio_to_result(&audio, total_ms))
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: delegates to the wrapped backend's `AudioBackend`
/// lifecycle methods. Backends that hold weights (`MusicGen`) honor their
/// own `load`/`unload`; stateless / not-yet-implemented backends are
/// no-op `load`/`unload`.
#[async_trait]
impl LocalModel for DynMusicProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        let backend: &dyn MusicBackend = self.as_ref();
        backend
            .load()
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        let backend: &dyn MusicBackend = self.as_ref();
        backend
            .unload()
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        let backend: &dyn MusicBackend = self.as_ref();
        backend.is_loaded().await
    }

    fn device(&self) -> crate::device::Device {
        // The audio-music surface does not yet expose a device query.
        crate::device::Device::Cpu
    }

    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-music does not support LoRA adapters",
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
    use blazen_audio::{AudioBackend, AudioFormat, GeneratedAudio as RawAudio};
    use std::sync::Arc;

    struct StubMusicBackend;

    #[async_trait]
    impl AudioBackend for StubMusicBackend {
        fn id(&self) -> &'static str {
            "stub:music"
        }
        fn provider_kind(&self) -> &'static str {
            "music"
        }
    }

    #[async_trait]
    impl MusicBackend for StubMusicBackend {
        async fn generate_music(
            &self,
            _prompt: &str,
            duration_seconds: f32,
        ) -> Result<RawAudio, MusicError> {
            Ok(RawAudio {
                bytes: vec![0_u8; 16],
                format: AudioFormat::Wav,
                sample_rate: 32_000,
                channels: 1,
                duration_seconds: Some(duration_seconds),
            })
        }

        async fn generate_sfx(
            &self,
            _prompt: &str,
            duration_seconds: f32,
        ) -> Result<RawAudio, MusicError> {
            Ok(RawAudio {
                bytes: vec![0_u8; 8],
                format: AudioFormat::Wav,
                sample_rate: 32_000,
                channels: 1,
                duration_seconds: Some(duration_seconds),
            })
        }
    }

    fn provider() -> DynMusicProvider {
        Arc::new(StubMusicBackend)
    }

    #[tokio::test]
    async fn provider_id_is_blazen_audio_music() {
        let p = provider();
        assert_eq!(ComputeProvider::provider_id(&p), PROVIDER_ID);
    }

    #[tokio::test]
    async fn tts_is_unsupported() {
        let p = provider();
        let err = AudioGeneration::text_to_speech(&p, SpeechRequest::new("hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn generate_music_returns_clip() {
        let p = provider();
        let req = MusicRequest::new("upbeat jazz").with_duration(3.0);
        let result = AudioGeneration::generate_music(&p, req).await.unwrap();
        assert_eq!(result.audio.len(), 1);
        assert_eq!(result.audio[0].sample_rate, Some(32_000));
        assert!((result.audio_seconds - 3.0).abs() < 1e-3);
    }

    #[tokio::test]
    async fn generate_sfx_returns_clip() {
        let p = provider();
        let req = MusicRequest::new("rain on a tin roof").with_duration(2.0);
        let result = AudioGeneration::generate_sfx(&p, req).await.unwrap();
        assert_eq!(result.audio.len(), 1);
        assert!((result.audio_seconds - 2.0).abs() < 1e-3);
    }

    #[tokio::test]
    async fn empty_prompt_music_is_rejected() {
        let p = provider();
        let req = MusicRequest::new("");
        let err = AudioGeneration::generate_music(&p, req).await.unwrap_err();
        assert!(matches!(err, BlazenError::Provider { .. }));
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let p = provider();
        let request = ComputeRequest {
            model: "stub-music".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = p.submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
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
