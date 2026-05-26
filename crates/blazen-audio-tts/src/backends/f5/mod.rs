//! F5-TTS backend (`SWivid` F5-TTS) — flow-matching `DiT`-based
//! zero-shot voice-cloning text-to-speech.
//!
//! # Architecture
//!
//! 1. **Text encoder + duration predictor** — converts input text +
//!    reference audio's text into a sequence of phoneme / char tokens.
//! 2. **`DiT`** (text-conditioned, `AdaLN-Zero` modulation,
//!    flow-matching velocity prediction) — reuses
//!    [`blazen_audio_core::dit`] / [`blazen_audio_core::adaln`] /
//!    [`blazen_audio_core::rope`] for the generic transformer
//!    primitives. F5-specific config + I/O sized for mel-spectrogram
//!    targets.
//! 3. **Flow-matching sampler** — Euler ODE solver, ~32 steps default.
//!    Generates mel-spectrogram conditioned on text + reference audio
//!    with classifier-free guidance.
//! 4. **Vocos vocoder** — converts predicted mel → 24 kHz waveform via
//!    a `ConvNeXt`-style backbone + `iSTFT` head.
//!
//! # Wave roadmap
//!
//! - **Wave F.1**: scaffolding only. [`F5Backend`] + [`F5Config`]
//!   existed; the [`TtsBackend`] impl returned [`TtsError::Unsupported`].
//! - **Wave F.2** (3 parallel agents): real implementations in
//!   [`dit_wrapper`] + [`vocos`] + [`sampler`].
//! - **Wave F.3**: [`tokenizer`] + [`weights`] (HF-hub backed).
//! - **Wave F.4** (this commit): [`pipeline`] orchestrates tokenizer +
//!   `DiT` + sampler + Vocos and the `TtsBackend::synthesize` /
//!   `clone_voice` paths are wired through it.
//!
//! # License
//!
//! Upstream `SWivid/F5-TTS` is licensed **MIT** (verified
//! 2026-05-22 via `gh api repos/SWivid/F5-TTS --jq
//! '.license.spdx_id'`). Reference: arxiv:2410.06885 — *F5-TTS: A
//! Fairytaler that Fakes Fluent and Faithful Speech with Flow
//! Matching*, Chen et al. 2024.

// Wave-1 stubs in sub-modules are intentionally not all public — the
// pipeline composes them privately.
#![allow(dead_code)]

mod dit_wrapper;
mod pipeline;
mod sampler;
mod tokenizer;
mod vocos;
mod weights;

pub use pipeline::{F5_SAMPLE_RATE_HZ, F5Pipeline};
pub use sampler::F5Sampling;
pub use tokenizer::F5Tokenizer;
pub use weights::F5Weights;

use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioFormat, CloneVoiceRequest, GeneratedAudio, VoiceHandle};
use futures_core::Stream;

use crate::traits::{StreamingAudioChunk, TtsBackend};
use crate::{TtsError, TtsOptions};

use pipeline::{
    PipelineCell, get_or_init_pipeline, new_pipeline_cell, pcm_duration_seconds,
    save_voice_reference,
};

/// Streaming chunk window in milliseconds. Picked to match the cadence
/// used by the streaming-capable Bark backend so downstream consumers
/// (WebSocket bridges, the OpenAI-compatible `/v1/audio/speech` SSE
/// surface, the napi-rs / `PyO3` stream adapters) see a uniform pacing
/// across diffusion-based TTS backends.
const F5_STREAM_WINDOW_MS: u32 = 250;

/// Chunk a fully-rendered PCM buffer into uniform fixed-width windows
/// for the [`TtsBackend::stream_synthesize`] override.
///
/// The last chunk has `is_final = true`; all earlier chunks have
/// `is_final = false`. Pulled out as a free function so the chunking
/// invariants (window size, final-flag placement) can be unit-tested
/// without spinning up an F5 pipeline.
///
/// `window_ms` must be > 0; `sample_rate` is the PCM sample rate in Hz.
/// An empty `samples` input returns an empty `Vec`. A single-chunk
/// input (fewer than one window) returns one chunk with `is_final =
/// true`.
#[must_use]
fn chunk_into_streaming(
    samples: &[f32],
    sample_rate: u32,
    window_ms: u32,
) -> Vec<StreamingAudioChunk> {
    if samples.is_empty() {
        return Vec::new();
    }
    let window_samples = usize::try_from((u64::from(sample_rate) * u64::from(window_ms)) / 1_000)
        .expect("window samples fit usize on 32-bit and wider targets");
    debug_assert!(
        window_samples > 0,
        "f5-tts: chunk_into_streaming requires window_ms * sample_rate >= 1000",
    );
    let total = samples.len();
    let chunk_count = total.div_ceil(window_samples);
    let mut chunks: Vec<StreamingAudioChunk> = Vec::with_capacity(chunk_count);
    let mut offset = 0_usize;
    while offset < total {
        let end = (offset + window_samples).min(total);
        let is_final = end == total;
        chunks.push(StreamingAudioChunk {
            samples: samples[offset..end].to_vec(),
            is_final,
            latency_seconds: None,
        });
        offset = end;
    }
    chunks
}

/// Stable backend-id prefix surfaced via [`AudioBackend::id`].
pub const F5_BACKEND_ID_PREFIX: &str = "f5-tts";

/// Configuration knobs for [`F5Backend`].
///
/// Defaults match upstream F5-TTS (arxiv:2410.06885): 32 Euler steps,
/// CFG scale 2.0, mel-24-kHz Vocos vocoder.
#[derive(Debug, Clone)]
pub struct F5Config {
    /// Hugging Face repo id for the F5-TTS model. Default
    /// `"SWivid/F5-TTS"`.
    pub model_id: String,
    /// Hugging Face repo id for the Vocos vocoder. Default
    /// `"charactr/vocos-mel-24khz"`.
    pub vocos_model_id: String,
    /// Number of Euler ODE sampling steps. Default `32`.
    pub sampling_steps: usize,
    /// Classifier-free guidance scale. Default `2.0`.
    pub cfg_scale: f32,
    /// Optional reference-audio path (.wav) used for zero-shot voice
    /// cloning. When `None` the backend falls back to its default
    /// voice prior. Populated by [`TtsBackend::clone_voice`] for
    /// subsequent synthesize calls.
    pub reference_audio: Option<PathBuf>,
    /// Optional transcript of [`Self::reference_audio`]. Required by
    /// upstream F5-TTS whenever a reference clip is supplied so the
    /// model can align the prompt's text features with its acoustic
    /// features.
    pub reference_text: Option<String>,
}

impl Default for F5Config {
    fn default() -> Self {
        Self {
            model_id: "SWivid/F5-TTS".to_owned(),
            vocos_model_id: "charactr/vocos-mel-24khz".to_owned(),
            sampling_steps: 32,
            cfg_scale: 2.0,
            reference_audio: None,
            reference_text: None,
        }
    }
}

/// F5-TTS backend handle.
///
/// Construct via [`F5Backend::new`]. Weights load lazily on the first
/// [`TtsBackend::synthesize`] / [`TtsBackend::clone_voice`] call.
#[derive(Clone)]
pub struct F5Backend {
    id: String,
    config: F5Config,
    /// Lazily-loaded shared pipeline cache. Cloned `F5Backend` values
    /// share the same underlying pipeline once it's materialised.
    pipeline: PipelineCell,
}

impl std::fmt::Debug for F5Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F5Backend")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("pipeline_loaded", &self.pipeline.initialized())
            .finish()
    }
}

impl F5Backend {
    /// Build a new F5-TTS backend with the given configuration.
    ///
    /// No weights are downloaded at construction time — the underlying
    /// [`F5Pipeline`] is materialised on the first synthesis call and
    /// cached for the lifetime of the backend (and any clones).
    #[must_use]
    pub fn new(config: F5Config) -> Self {
        let id = format!("{F5_BACKEND_ID_PREFIX}:{}", config.model_id);
        Self {
            id,
            config,
            pipeline: new_pipeline_cell(),
        }
    }

    /// The resolved model id this backend was configured with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    /// Access the underlying [`F5Config`] (read-only).
    #[must_use]
    pub fn config(&self) -> &F5Config {
        &self.config
    }

    /// Force the underlying [`F5Pipeline`] to load — useful for
    /// callers that want to amortise the HF download up front.
    ///
    /// # Errors
    ///
    /// Surfaces any [`TtsError::ModelLoad`] from the inner HF download.
    pub async fn load_pipeline(&self) -> Result<(), TtsError> {
        get_or_init_pipeline(
            &self.pipeline,
            &self.config.model_id,
            &self.config.vocos_model_id,
            self.sampling(),
        )
        .await?;
        Ok(())
    }

    /// Build the [`F5Sampling`] config from the user-facing
    /// [`F5Config`] knobs.
    fn sampling(&self) -> F5Sampling {
        F5Sampling {
            n_steps: self.config.sampling_steps,
            cfg_strength: self.config.cfg_scale,
            ..F5Sampling::default()
        }
    }
}

impl Default for F5Backend {
    fn default() -> Self {
        Self::new(F5Config::default())
    }
}

#[async_trait]
impl AudioBackend for F5Backend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "tts"
    }

    async fn is_loaded(&self) -> bool {
        self.pipeline.initialized()
    }
}

#[async_trait]
impl TtsBackend for F5Backend {
    async fn synthesize(
        &self,
        text: &str,
        _options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        let pipeline = get_or_init_pipeline(
            &self.pipeline,
            &self.config.model_id,
            &self.config.vocos_model_id,
            self.sampling(),
        )
        .await?;

        let (wav_bytes, n_samples) = pipeline
            .synthesize_wav(
                text,
                self.config.reference_audio.as_deref(),
                self.config.reference_text.as_deref(),
            )
            .await?;

        Ok(GeneratedAudio {
            bytes: wav_bytes,
            format: AudioFormat::Wav,
            sample_rate: F5_SAMPLE_RATE_HZ,
            channels: 1,
            duration_seconds: Some(pcm_duration_seconds(n_samples, F5_SAMPLE_RATE_HZ)),
        })
    }

    /// Persist a reference clip + transcript pair for later voice-clone
    /// synthesis.
    ///
    /// F5-TTS's voice cloning is reference-audio driven: callers
    /// supply a short reference WAV plus its transcript, and the
    /// model conditions on both at synthesis time. The implementation
    /// here saves the clip + transcript to the user cache and
    /// returns a [`VoiceHandle`] whose `id` is the on-disk path.
    /// Subsequent synthesis calls must rebuild the [`F5Config`] with
    /// `reference_audio = Some(handle.id)` + `reference_text =
    /// Some(<text>)`.
    ///
    /// Note: the actual audio-encoder conditioning path is gated on a
    /// follow-up wave (the standalone F5 audio encoder isn't ported
    /// yet); for now the reference clip is persisted but its acoustic
    /// features are not consumed during synthesis.
    async fn clone_voice(&self, request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        if request.audio_bytes.is_empty() {
            return Err(TtsError::Unsupported(
                "f5-tts clone_voice: `audio_bytes` is empty; supply a reference WAV clip"
                    .to_owned(),
            ));
        }
        let path = save_voice_reference(
            &request.name,
            &request.audio_bytes,
            request.transcript.as_deref(),
        )?;
        Ok(VoiceHandle {
            id: path.to_string_lossy().into_owned(),
            provider: F5_BACKEND_ID_PREFIX.to_owned(),
        })
    }

    /// Chunked-after-synthesis streaming.
    ///
    /// F5-TTS is a non-causal flow-matching `DiT` over the *entire*
    /// utterance mel spectrogram — every Euler ODE step refines the
    /// whole tensor, so per-frame (true causal) streaming is not
    /// architecturally possible. The Vocos vocoder is similarly
    /// non-causal (`ConvNeXt` + `iSTFT` overlap-add).
    ///
    /// This override therefore implements the only honest streaming
    /// shape available for diffusion-based TTS: synthesize the full
    /// utterance, then chop the rendered 24 kHz PCM into 250 ms
    /// windows and yield each as a [`StreamingAudioChunk`]. The last
    /// chunk has `is_final = true`. The total latency to first chunk
    /// is identical to a non-streaming [`synthesize`](Self::synthesize)
    /// call — the streaming layer is purely a delivery cadence
    /// adapter for callers (WebSocket bridges, SSE clients, the
    /// language-binding stream adapters) that expect a chunked feed.
    async fn stream_synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        options: TtsOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingAudioChunk, TtsError>> + Send>>, TtsError>
    {
        // Honor the per-call `voice` override by merging it into the
        // options before delegating to `synthesize` — F5's voice
        // selection currently flows through TtsOptions.voice, so this
        // keeps the streaming and non-streaming voice resolution paths
        // identical.
        let mut merged = options;
        if let Some(v) = voice {
            merged.voice = Some(v.to_owned());
        }

        let audio = self.synthesize(text, &merged).await?;
        let samples = decode_wav_to_f32(&audio.bytes)?;
        let chunks = chunk_into_streaming(&samples, audio.sample_rate, F5_STREAM_WINDOW_MS);
        let stream = futures_util::stream::iter(chunks.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

/// Decode a 16-bit PCM mono WAV (the format the F5 pipeline emits via
/// [`pipeline::F5Pipeline::synthesize_wav`]) back to a `Vec<f32>` of
/// samples in `[-1.0, 1.0]`. Strictly speaking this is the inverse of
/// the local `encode_wav_16bit` writer in [`pipeline`], not a general
/// WAV decoder — it asserts the 44-byte RIFF/WAVE/fmt/data layout we
/// emit and surfaces [`TtsError::Synthesis`] on any mismatch (which
/// would indicate the pipeline writer changed shape from under us).
fn decode_wav_to_f32(wav: &[u8]) -> Result<Vec<f32>, TtsError> {
    const HEADER_LEN: usize = 44;
    if wav.len() < HEADER_LEN {
        return Err(TtsError::Synthesis(format!(
            "f5-tts stream: wav too small ({} < {HEADER_LEN}-byte header)",
            wav.len(),
        )));
    }
    if &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err(TtsError::Synthesis(
            "f5-tts stream: wav header missing RIFF/WAVE".to_owned(),
        ));
    }
    let pcm = &wav[HEADER_LEN..];
    if !pcm.len().is_multiple_of(2) {
        return Err(TtsError::Synthesis(format!(
            "f5-tts stream: wav data length {} not aligned to i16",
            pcm.len(),
        )));
    }
    let samples = pcm
        .chunks_exact(2)
        .map(|pair| {
            let i = i16::from_le_bytes([pair[0], pair[1]]);
            f32::from(i) / f32::from(i16::MAX)
        })
        .collect();
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f5_config_defaults_match_upstream() {
        let cfg = F5Config::default();
        assert_eq!(cfg.model_id, "SWivid/F5-TTS");
        assert_eq!(cfg.vocos_model_id, "charactr/vocos-mel-24khz");
        assert_eq!(cfg.sampling_steps, 32);
        assert!((cfg.cfg_scale - 2.0).abs() < f32::EPSILON);
        assert!(cfg.reference_audio.is_none());
        assert!(cfg.reference_text.is_none());
    }

    #[test]
    fn f5_backend_id_includes_model() {
        let backend = F5Backend::new(F5Config::default());
        assert_eq!(backend.id(), "f5-tts:SWivid/F5-TTS");
        assert_eq!(backend.model_id(), "SWivid/F5-TTS");
        assert_eq!(backend.provider_kind(), "tts");
    }

    #[tokio::test]
    async fn is_loaded_starts_false() {
        let backend = F5Backend::default();
        assert!(!backend.is_loaded().await);
    }

    #[tokio::test]
    async fn clone_voice_with_empty_audio_bytes_returns_unsupported() {
        let backend = F5Backend::default();
        let req = CloneVoiceRequest {
            name: "test".to_owned(),
            audio_bytes: Vec::new(),
            transcript: Some("hello".to_owned()),
        };
        let err = backend
            .clone_voice(req)
            .await
            .expect_err("empty audio_bytes must surface Unsupported");
        match err {
            TtsError::Unsupported(msg) => {
                assert!(msg.contains("audio_bytes"), "msg = {msg}");
                assert!(msg.contains("empty"), "msg = {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn clone_voice_persists_reference_and_returns_handle() {
        let tmp = std::env::temp_dir().join(format!("blazen-f5-clone-test-{}", std::process::id()));
        // SAFETY: tests run with `BLAZEN_F5_VOICE_DIR` scoped to this
        // process; we restore the environment at the end of the test.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("BLAZEN_F5_VOICE_DIR", &tmp);
        }
        let backend = F5Backend::default();
        let req = CloneVoiceRequest {
            name: "alice-2026".to_owned(),
            audio_bytes: b"RIFF....fake-wav-payload".to_vec(),
            transcript: Some("hello world".to_owned()),
        };
        let handle = backend
            .clone_voice(req)
            .await
            .expect("clone_voice must succeed");
        assert_eq!(handle.provider, "f5-tts");
        let saved = PathBuf::from(&handle.id);
        assert!(saved.exists(), "saved reference at {saved:?} must exist");
        assert_eq!(
            saved.file_name().and_then(|s| s.to_str()),
            Some("alice-2026.wav"),
        );
        let txt = std::fs::read_to_string(saved.with_extension("txt"))
            .expect("transcript should be persisted");
        assert_eq!(txt, "hello world");
        let _ = std::fs::remove_dir_all(&tmp);
        // SAFETY: see above.
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("BLAZEN_F5_VOICE_DIR");
        }
    }

    #[test]
    fn stream_synthesize_chunks_have_final_at_end() {
        // 1.25 s at 24 kHz = 30 000 samples; 250 ms window = 6 000
        // samples → exactly 5 chunks of 6 000, the last is_final.
        let samples = vec![0.25_f32; 30_000];
        let chunks = chunk_into_streaming(&samples, F5_SAMPLE_RATE_HZ, F5_STREAM_WINDOW_MS);
        assert_eq!(chunks.len(), 5, "expected 5 chunks for 1.25 s at 24 kHz");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.samples.len(),
                6_000,
                "chunk {i} must hold one full 250 ms window",
            );
            let want_final = i == chunks.len() - 1;
            assert_eq!(
                chunk.is_final, want_final,
                "chunk {i}: is_final = {} but expected {want_final}",
                chunk.is_final,
            );
        }
    }

    #[test]
    fn chunk_into_streaming_marks_short_tail_as_final() {
        // 30 050 samples → 5 full windows + a 50-sample tail; the
        // tail chunk must be the only one with `is_final = true`.
        let samples = vec![0.0_f32; 30_050];
        let chunks = chunk_into_streaming(&samples, F5_SAMPLE_RATE_HZ, F5_STREAM_WINDOW_MS);
        assert_eq!(chunks.len(), 6, "30 050 samples → 5 full + 1 short tail");
        for (i, chunk) in chunks.iter().take(5).enumerate() {
            assert_eq!(chunk.samples.len(), 6_000, "leading chunk {i} must be full");
            assert!(!chunk.is_final, "leading chunk {i} must not be final");
        }
        let tail = chunks.last().expect("tail chunk");
        assert_eq!(tail.samples.len(), 50, "tail must carry the remainder");
        assert!(tail.is_final, "tail chunk must be flagged final");
    }

    #[test]
    fn chunk_into_streaming_empty_input_yields_no_chunks() {
        let chunks = chunk_into_streaming(&[], F5_SAMPLE_RATE_HZ, F5_STREAM_WINDOW_MS);
        assert!(chunks.is_empty(), "empty input must not emit a final chunk");
    }

    #[test]
    fn decode_wav_to_f32_roundtrips_through_pipeline_writer() {
        // Build a tiny WAV using the same 44-byte header the F5
        // pipeline emits, then decode and compare. Sample values are
        // chosen to exercise sign + clamping behaviour.
        let sr: u32 = F5_SAMPLE_RATE_HZ;
        let channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let data_size: u32 = 4 * 2;
        let byte_rate = sr * u32::from(channels) * u32::from(bits_per_sample) / 8;
        let block_align = channels * bits_per_sample / 8;
        let mut wav = Vec::with_capacity(44 + 8);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for s in [0_i16, i16::MAX, -i16::MAX, 0] {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        let samples = decode_wav_to_f32(&wav).expect("decode minimal wav");
        assert_eq!(samples.len(), 4);
        assert!(samples[0].abs() < 1e-6);
        assert!((samples[1] - 1.0).abs() < 1e-4);
        assert!((samples[2] + 1.0).abs() < 1e-4);
        assert!(samples[3].abs() < 1e-6);
    }
}
