//! Bridge between [`blazen_audio_tts::TtsProvider`] and the
//! [`AudioGeneration`](crate::compute::AudioGeneration) trait.
//!
//! Local TTS via the `any-tts` crate runs fully on-device, so there is no
//! job queue or asynchronous polling: [`ComputeProvider::submit`] returns
//! [`BlazenError::Unsupported`] and consumers are expected to call
//! [`AudioGeneration::text_to_speech`] directly.
//!
//! `generate_music` and `generate_sfx` fall back to the trait-default
//! `Unsupported` implementations because Kokoro/VibeVoice/Qwen3-TTS are
//! TTS-only models — there is no music-generation route through any-tts.
//!
//! Without the `engine` feature on `blazen-audio-tts`, the underlying
//! provider cannot actually run inference and every `text_to_speech` call
//! surfaces as a [`BlazenError::Provider`] carrying the
//! `EngineNotAvailable` message.

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use blazen_audio_tts::TtsProvider;

use crate::compute::{
    AudioGeneration, AudioResult, ComputeProvider, ComputeRequest, ComputeResult, JobHandle,
    JobStatus, SpeechRequest,
};
use crate::error::BlazenError;
use crate::media::{GeneratedAudio, MediaOutput, MediaType};
use crate::traits::LocalModel;
use crate::types::RequestTiming;

const PROVIDER_ID: &str = "any-tts";

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for TtsProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "any-tts runs locally and does not use the ComputeRequest job API; \
             call `AudioGeneration::text_to_speech` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "any-tts does not expose a job queue -- synthesis is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "any-tts does not expose a job queue -- synthesis is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "any-tts synthesis is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioGeneration for TtsProvider {
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        if request.text.is_empty() {
            return Err(BlazenError::provider(
                PROVIDER_ID,
                "text_to_speech request has empty `text`",
            ));
        }

        // SpeechRequest carries `voice`, `language`, `speed`, `model`, but
        // the live provider is constructed up-front from `TtsOptions` so we
        // only honour the per-call fields if they match the configured
        // model / voice. The bindings layer (PR-TTS-HTTP follow-up) will
        // handle per-call overrides; for v1 we log a warning and proceed.
        if request.voice_url.is_some() {
            tracing::warn!(
                "any-tts: voice_url is not supported on the local TTS backend; ignoring"
            );
        }
        if request.speed.is_some() {
            tracing::warn!("any-tts: per-call speed override not yet plumbed; ignoring");
        }

        let start = std::time::Instant::now();

        let synth = TtsProvider::synthesize(self, &request.text)
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))?;

        #[allow(clippy::cast_possible_truncation)]
        let total_ms = start.elapsed().as_millis() as u64;

        let wav_len = u64::try_from(synth.wav_bytes.len()).unwrap_or(u64::MAX);
        let mut media = MediaOutput::from_base64(BASE64.encode(&synth.wav_bytes), MediaType::Wav);
        media.file_size = Some(wav_len);

        let clip = GeneratedAudio {
            media,
            duration_seconds: Some(synth.duration_secs),
            sample_rate: Some(synth.sample_rate_hz),
            channels: Some(1),
        };

        Ok(AudioResult {
            audio: vec![clip],
            timing: RequestTiming {
                queue_ms: None,
                execution_ms: Some(total_ms),
                total_ms: Some(total_ms),
            },
            cost: None,
            usage: None,
            audio_seconds: f64::from(synth.duration_secs),
            metadata: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: the any-tts provider lazy-loads its underlying
/// model on first `synthesize`, so `load`/`unload` here are coarse hooks.
/// `load` returns `Ok(())` if the `engine` feature is active (real load
/// happens on first synthesis call); without it `load` surfaces the
/// `EngineNotAvailable` error from the provider's own synthesize path.
/// `unload` is a no-op because the upstream library does not expose a
/// drop-the-weights API today; once it does we will wire it in (see
/// follow-up tracked in the parent PR6 plan).
#[async_trait]
impl LocalModel for TtsProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        if !self.engine_available() {
            return Err(BlazenError::provider(
                PROVIDER_ID,
                "tts engine not available: compile blazen-audio-tts with the `engine` feature",
            ));
        }
        // No eager preload — any-tts loads weights on first synthesize().
        Ok(())
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        // any-tts has no exposed unload entrypoint; nothing to do.
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        // Cannot introspect any-tts model state today; treat the provider
        // as "loaded" iff the engine feature is compiled in.
        self.engine_available()
    }

    fn device(&self) -> crate::device::Device {
        // any-tts picks its own device internally via `DeviceSelection`;
        // we report CPU as the conservative default until upstream exposes
        // a query API.
        crate::device::Device::Cpu
    }

    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "any-tts does not support LoRA adapters",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_audio_tts::TtsOptions;

    #[tokio::test]
    async fn provider_id_is_any_tts() {
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        assert_eq!(ComputeProvider::provider_id(&provider), "any-tts");
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        let request = ComputeRequest {
            model: "any-tts".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = provider.submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn cancel_is_unsupported() {
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        let handle = JobHandle {
            id: "fake".into(),
            provider: "any-tts".into(),
            model: "any-tts".into(),
            submitted_at: chrono::Utc::now(),
        };
        let err = provider.cancel(&handle).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn empty_text_is_rejected() {
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        let request = SpeechRequest::new("");
        let err = AudioGeneration::text_to_speech(&provider, request)
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Provider { .. }));
    }

    #[tokio::test]
    async fn music_generation_unsupported_by_default() {
        use crate::compute::MusicRequest;
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        let err = AudioGeneration::generate_music(&provider, MusicRequest::new("piano"))
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn sfx_generation_unsupported_by_default() {
        use crate::compute::MusicRequest;
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        let err = AudioGeneration::generate_sfx(&provider, MusicRequest::new("rain"))
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn lora_adapter_unsupported() {
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("construct");
        let err = LocalModel::load_adapter(
            &provider,
            std::path::Path::new("/dev/null"),
            crate::AdapterOptions::new("test"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }
}
