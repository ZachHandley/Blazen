//! Bridge between [`blazen_audio_tts::DynTtsProvider`] and the
//! [`AudioGeneration`](crate::compute::AudioGeneration) trait.
//!
//! Any [`blazen_audio_tts::TtsBackend`] erased into a `DynTtsProvider`
//! is plugged into `blazen-llm`'s compute facade through this bridge.
//! The provider is constructed up-front (typically with
//! [`AnyTtsBackend`](blazen_audio_tts::backends::AnyTtsBackend) when
//! the `anytts` feature is on, or
//! [`OpenAiTtsBackend`](blazen_audio_tts::backends::OpenAiTtsBackend)
//! when `openai` is on) and synthesis is driven through the trait's
//! [`text_to_speech`](AudioGeneration::text_to_speech) method.
//!
//! [`ComputeProvider::submit`] returns
//! [`BlazenError::Unsupported`] — TTS in this stack is synchronous and
//! does not use the asynchronous job-queue surface.
//!
//! `generate_music` and `generate_sfx` fall back to the trait-default
//! `Unsupported` implementations because none of the wrapped TTS
//! backends generate music or SFX.

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use blazen_audio_tts::{DynTtsProvider, TtsOptions};

use crate::compute::{
    AudioGeneration, AudioResult, ComputeProvider, ComputeRequest, ComputeResult, JobHandle,
    JobStatus, SpeechRequest,
};
use crate::error::BlazenError;
use crate::media::{GeneratedAudio, MediaOutput, MediaType};
use crate::traits::LocalModel;
use crate::types::RequestTiming;

const PROVIDER_ID: &str = "blazen-audio-tts";

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for DynTtsProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-tts runs synchronously and does not use the \
             ComputeRequest job API; call `AudioGeneration::text_to_speech` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-tts does not expose a job queue -- synthesis is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-tts does not expose a job queue -- synthesis is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-tts synthesis is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration
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

#[async_trait]
impl AudioGeneration for DynTtsProvider {
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        if request.text.is_empty() {
            return Err(BlazenError::provider(
                PROVIDER_ID,
                "text_to_speech request has empty `text`",
            ));
        }

        if request.voice_url.is_some() {
            tracing::warn!("blazen-audio-tts: voice_url is not honored by this bridge; ignoring");
        }

        let options = TtsOptions {
            voice: request.voice.clone(),
            language: request.language.clone(),
            model_id: request.model.clone(),
            speed: request.speed,
            ..TtsOptions::default()
        };

        let start = std::time::Instant::now();

        let synth = self
            .synthesize(&request.text, &options)
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))?;

        #[allow(clippy::cast_possible_truncation)]
        let total_ms = start.elapsed().as_millis() as u64;

        let audio_len = u64::try_from(synth.bytes.len()).unwrap_or(u64::MAX);
        let mut media =
            MediaOutput::from_base64(BASE64.encode(&synth.bytes), media_type_for(synth.format));
        media.file_size = Some(audio_len);

        let clip = GeneratedAudio {
            media,
            duration_seconds: synth.duration_seconds,
            sample_rate: Some(synth.sample_rate),
            channels: Some(u8::try_from(synth.channels).unwrap_or(1)),
        };

        let audio_seconds = f64::from(synth.duration_seconds.unwrap_or(0.0));

        Ok(AudioResult {
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
        })
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: delegates to the underlying backend's
/// `AudioBackend` lifecycle methods. Stateless HTTP backends are always
/// "loaded"; backends that hold weights honor their own `load`/`unload`.
#[async_trait]
impl LocalModel for DynTtsProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        DynTtsProvider::load(self)
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        DynTtsProvider::unload(self)
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        DynTtsProvider::is_loaded(self).await
    }

    fn device(&self) -> crate::device::Device {
        // The wrapped backend picks its own device internally; we report
        // CPU as the conservative default until the audio surface exposes
        // a device query.
        crate::device::Device::Cpu
    }

    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-tts does not support LoRA adapters",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_audio_tts::PiperBackend;

    fn provider() -> DynTtsProvider {
        // Use the always-available reserved Piper stub so tests don't
        // require any feature flag combination.
        DynTtsProvider::erase(PiperBackend::new())
    }

    #[tokio::test]
    async fn provider_id_is_blazen_audio_tts() {
        let p = provider();
        assert_eq!(ComputeProvider::provider_id(&p), "blazen-audio-tts");
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let p = provider();
        let request = ComputeRequest {
            model: "any-tts".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = p.submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn cancel_is_unsupported() {
        let p = provider();
        let handle = JobHandle {
            id: "fake".into(),
            provider: "any-tts".into(),
            model: "any-tts".into(),
            submitted_at: chrono::Utc::now(),
        };
        let err = p.cancel(&handle).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn empty_text_is_rejected() {
        let p = provider();
        let request = SpeechRequest::new("");
        let err = AudioGeneration::text_to_speech(&p, request)
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Provider { .. }));
    }

    #[tokio::test]
    async fn music_generation_unsupported_by_default() {
        use crate::compute::MusicRequest;
        let p = provider();
        let err = AudioGeneration::generate_music(&p, MusicRequest::new("piano"))
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn sfx_generation_unsupported_by_default() {
        use crate::compute::MusicRequest;
        let p = provider();
        let err = AudioGeneration::generate_sfx(&p, MusicRequest::new("rain"))
            .await
            .unwrap_err();
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
