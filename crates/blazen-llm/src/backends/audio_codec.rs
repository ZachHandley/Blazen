//! Bridge between [`blazen_audio_codec::DynCodecProvider`] and the
//! [`Codec`](crate::compute::Codec) capability trait.
//!
//! Any [`blazen_audio_codec::CodecBackend`] erased into a
//! [`DynCodecProvider`] is plugged into `blazen-llm`'s compute facade
//! through this bridge. The codec API does not fit a job-queue surface,
//! so [`ComputeProvider::submit`] returns
//! [`BlazenError::Unsupported`] — callers should drive `encode_audio` /
//! `decode_audio` directly through the [`Codec`] trait.

use async_trait::async_trait;
use blazen_audio_codec::{CodecError, DynCodecProvider};

use crate::compute::{Codec, ComputeProvider, ComputeRequest, ComputeResult, JobHandle, JobStatus};
use crate::error::BlazenError;
use crate::traits::LocalModel;

const PROVIDER_ID: &str = "blazen-audio-codec";

// ---------------------------------------------------------------------------
// Error translation
// ---------------------------------------------------------------------------

fn to_blazen_error(err: CodecError) -> BlazenError {
    match err {
        CodecError::EngineNotAvailable | CodecError::NotYetImplemented(_) => {
            BlazenError::unsupported(err.to_string())
        }
        CodecError::InvalidInput(msg) => BlazenError::provider(PROVIDER_ID, msg),
        other => BlazenError::provider(PROVIDER_ID, other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for DynCodecProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-codec is a deterministic codec; it does not use the \
             ComputeRequest job API. Call `Codec::encode_audio` / `decode_audio` \
             directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-codec does not expose a job queue",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-codec does not expose a job queue",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-codec is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

#[async_trait]
impl Codec for DynCodecProvider {
    async fn encode_audio(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<u32>, BlazenError> {
        self.encode_pcm(pcm, sample_rate)
            .await
            .map_err(to_blazen_error)
    }

    async fn decode_audio(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> Result<Vec<f32>, BlazenError> {
        self.decode_tokens(tokens, num_codebooks)
            .await
            .map_err(to_blazen_error)
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: codec backends honor their own load/unload
/// lifecycle via the wrapped `AudioBackend`. The underlying codec models
/// (`EnCodec`, `DAC`, `SNAC`, …) often lazy-load on first encode; calling
/// `load` explicitly is supported but not required.
#[async_trait]
impl LocalModel for DynCodecProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        self.backend()
            .load()
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        self.backend()
            .unload()
            .await
            .map_err(|e| BlazenError::provider(PROVIDER_ID, e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        self.backend().is_loaded().await
    }

    fn device(&self) -> crate::device::Device {
        crate::device::Device::Cpu
    }

    async fn load_adapter(
        &self,
        _adapter_dir: &std::path::Path,
        _options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-audio-codec does not support LoRA adapters",
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
    use blazen_audio_codec::CodecBackend;
    use std::sync::Arc;

    struct StubCodec;

    #[async_trait]
    impl AudioBackend for StubCodec {
        fn id(&self) -> &'static str {
            "stub:codec"
        }
        fn provider_kind(&self) -> &'static str {
            "codec"
        }
    }

    #[async_trait]
    impl CodecBackend for StubCodec {
        async fn encode_pcm(
            &self,
            samples: &[f32],
            _sample_rate: u32,
        ) -> Result<Vec<u32>, CodecError> {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Ok(samples.iter().map(|s| (s.abs() * 1000.0) as u32).collect())
        }

        async fn decode_tokens(
            &self,
            tokens: &[u32],
            num_codebooks: usize,
        ) -> Result<Vec<f32>, CodecError> {
            if !tokens.len().is_multiple_of(num_codebooks) {
                return Err(CodecError::invalid_input("misaligned"));
            }
            #[allow(clippy::cast_precision_loss)]
            Ok(tokens.iter().map(|&t| (t as f32) / 1000.0).collect())
        }

        fn num_codebooks(&self) -> usize {
            1
        }
    }

    fn provider() -> DynCodecProvider {
        DynCodecProvider::new(Arc::new(StubCodec) as Arc<dyn CodecBackend>)
    }

    #[tokio::test]
    async fn provider_id_is_blazen_audio_codec() {
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
    async fn encode_and_decode_roundtrip() {
        let p = provider();
        let tokens = Codec::encode_audio(&p, &[0.1, 0.2, 0.3], 24_000)
            .await
            .unwrap();
        assert_eq!(tokens.len(), 3);
        let pcm = Codec::decode_audio(&p, &tokens, 1).await.unwrap();
        assert_eq!(pcm.len(), 3);
    }

    #[tokio::test]
    async fn decode_invalid_alignment_surfaces_provider_error() {
        let p = provider();
        // 3 tokens with 2 codebooks → misaligned
        let err = Codec::decode_audio(&p, &[1, 2, 3], 2).await.unwrap_err();
        assert!(matches!(err, BlazenError::Provider { .. }));
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

    /// Trait-object dispatch through `&dyn Codec` — proves the trait is
    /// actually object-safe (the test simply fails to compile otherwise).
    #[tokio::test]
    async fn dyn_codec_dispatch_works() {
        let p = provider();
        let codec: &dyn Codec = &p;
        let tokens = codec.encode_audio(&[0.5], 24_000).await.unwrap();
        assert_eq!(tokens.len(), 1);
    }
}
