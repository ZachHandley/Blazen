//! Bark TTS backend (Suno AI) — 3-stage autoregressive transformer
//! producing `EnCodec` acoustic tokens, with zero-shot voice cloning via
//! prompt-conditioned generation.
//!
//! Upstream: <https://github.com/suno-ai/bark> (MIT-licensed). Canonical
//! HF checkpoint: `suno/bark-small`; the full-size variant is `suno/bark`.
//!
//! # Architecture
//!
//! Bark synthesises speech in three stages:
//!
//! 1. **Semantic** (12-layer GPT-style decoder, [`semantic`]): text +
//!    optional voice-prompt history → semantic tokens.
//! 2. **Coarse acoustic** ([`coarse`]): semantic tokens → first two
//!    codebooks of `EnCodec` acoustic tokens (autoregressive).
//! 3. **Fine acoustic** ([`fine`]): coarse tokens → remaining six
//!    codebooks (non-autoregressive, two refinement steps).
//!
//! The complete 8-codebook tensor is decoded to a 24 kHz waveform by
//! the `EnCodec` backend in `blazen-audio-codec`.
//!
//! # Wave plan
//!
//! - **Wave B.1** (this commit): scaffolding only. The backend struct
//!   exists, the [`TtsBackend`] impl returns [`TtsError::Unsupported`]
//!   from `synthesize` and `clone_voice` until the real pipeline lands.
//! - **Wave B.2**: real `SemanticDecoder` / `CoarseDecoder` /
//!   `FineDecoder` in [`semantic`] / [`coarse`] / [`fine`].
//! - **Wave B.3**: [`tokenizer`] + [`weights`] — BERT-style tokenizer
//!   and the HF Hub weight loader for `suno/bark-small`.
//! - **Wave B.4** (this commit): [`pipeline`] orchestrates the three
//!   stages → `EnCodec` → 24 kHz WAV and wires the live `synthesize` /
//!   `clone_voice` paths.

// A handful of constants surfaced by the lower stages (e.g.
// `SEMANTIC_VOCAB_SIZE`, `BlockConfig.block_size`) document upstream
// invariants and are referenced from prose; suppressed crate-wide rather
// than littering each call site.
#![allow(dead_code)]

mod coarse;
mod fine;
mod gpt_block;
mod pipeline;
mod semantic;
mod tokenizer;
mod weights;

pub use pipeline::{BarkPipeline, BarkSamplingConfig, DEFAULT_ENCODEC_MODEL_ID};
pub use tokenizer::BarkTokenizer;
pub use weights::BarkWeights;

use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioFormat, CloneVoiceRequest, GeneratedAudio, VoiceHandle};
use futures_core::Stream;

use crate::traits::{StreamingAudioChunk, TtsBackend};
use crate::{TtsError, TtsOptions};

use pipeline::{
    PipelineCell, get_or_init_pipeline, load_voice_prompt, new_pipeline_cell, pcm_duration_seconds,
    save_voice_prompt,
};

/// Stable backend identifier, exposed at runtime via
/// [`AudioBackend::id`]. The trailing colon is the same shape the
/// other backends use (`piper:vendored`, `openai:<host>`).
pub const BARK_BACKEND_ID_PREFIX: &str = "bark";

/// Bark text-to-speech backend.
///
/// Construct via [`BarkBackend::new`]. Weights load lazily on the first
/// [`TtsBackend::synthesize`] / [`TtsBackend::clone_voice`] call (matching
/// the lazy-load convention used by `whisper-streaming` and `EnCodec`).
#[derive(Clone)]
pub struct BarkBackend {
    id: String,
    config: BarkConfig,
    /// Lazily-loaded shared pipeline cache. Cloned `BarkBackend` values
    /// share the same underlying pipeline once it's been materialised.
    pipeline: PipelineCell,
}

impl std::fmt::Debug for BarkBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarkBackend")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("pipeline_loaded", &self.pipeline.initialized())
            .finish()
    }
}

/// Configuration knobs for [`BarkBackend`].
///
/// All sampling temperatures default to the values used by upstream
/// `suno-ai/bark` (`0.7` for semantic + coarse, `0.5` for fine — the
/// latter being more deterministic since it merely refines existing
/// coarse tokens).
#[derive(Debug, Clone)]
pub struct BarkConfig {
    /// Hugging Face repo id for the Bark model. Defaults to
    /// `"suno/bark-small"`; set to `"suno/bark"` for the full-size
    /// (~1.5 GB) variant.
    pub model_id: String,
    /// Hugging Face repo id for the `EnCodec` waveform codec the Bark
    /// pipeline decodes through. Defaults to
    /// [`DEFAULT_ENCODEC_MODEL_ID`] (`"facebook/encodec_24khz"`).
    pub encodec_model_id: String,
    /// Sampling temperature for the semantic stage. Default `0.7`.
    pub semantic_temperature: f32,
    /// Sampling temperature for the coarse acoustic stage. Default `0.7`.
    pub coarse_temperature: f32,
    /// Sampling temperature for the fine acoustic stage. Default `0.5`.
    pub fine_temperature: f32,
    /// Optional path to a Blazen-format voice-prompt file (a raw `[u32]`
    /// little-endian dump of semantic tokens) produced by
    /// [`TtsBackend::clone_voice`]. When set, the semantic stage seeds
    /// generation with the prompt's history, enabling zero-shot voice
    /// cloning.
    pub voice_prompt: Option<PathBuf>,
}

impl Default for BarkConfig {
    fn default() -> Self {
        Self {
            model_id: "suno/bark-small".to_owned(),
            encodec_model_id: DEFAULT_ENCODEC_MODEL_ID.to_owned(),
            semantic_temperature: 0.7,
            coarse_temperature: 0.7,
            fine_temperature: 0.5,
            voice_prompt: None,
        }
    }
}

impl BarkBackend {
    /// Construct a new Bark backend with the given configuration.
    ///
    /// No weights are downloaded at construction time — the underlying
    /// [`BarkPipeline`] is materialised on the first synthesis call and
    /// cached for the lifetime of the backend (and any clones).
    #[must_use]
    pub fn new(config: BarkConfig) -> Self {
        let id = format!("{BARK_BACKEND_ID_PREFIX}:{}", config.model_id);
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

    /// Force the underlying [`BarkPipeline`] to load — useful for
    /// callers that want to amortise the HF download up front.
    ///
    /// # Errors
    ///
    /// Surfaces any [`TtsError::ModelLoad`] from the inner HF download.
    pub async fn load_pipeline(&self) -> Result<(), TtsError> {
        get_or_init_pipeline(
            &self.pipeline,
            &self.config.model_id,
            &self.config.encodec_model_id,
        )
        .await?;
        Ok(())
    }
}

impl Default for BarkBackend {
    fn default() -> Self {
        Self::new(BarkConfig::default())
    }
}

#[async_trait]
impl AudioBackend for BarkBackend {
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
impl TtsBackend for BarkBackend {
    async fn synthesize(
        &self,
        text: &str,
        _options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        let pipeline = get_or_init_pipeline(
            &self.pipeline,
            &self.config.model_id,
            &self.config.encodec_model_id,
        )
        .await?;

        let history_prompt: Option<Vec<u32>> = match &self.config.voice_prompt {
            Some(path) => Some(load_voice_prompt(path)?),
            None => None,
        };

        let sampling = BarkSamplingConfig {
            semantic_temperature: self.config.semantic_temperature,
            coarse_temperature: self.config.coarse_temperature,
            fine_temperature: self.config.fine_temperature,
            ..BarkSamplingConfig::default()
        };

        let (wav_bytes, n_samples) = pipeline
            .synthesize_wav(text, &sampling, history_prompt.as_deref())
            .await?;

        Ok(GeneratedAudio {
            bytes: wav_bytes,
            format: AudioFormat::Wav,
            sample_rate: pipeline::BARK_SAMPLE_RATE_HZ,
            channels: 1,
            duration_seconds: Some(pcm_duration_seconds(
                n_samples,
                pipeline::BARK_SAMPLE_RATE_HZ,
            )),
        })
    }

    /// Persist a pre-tokenized semantic prompt under `request.name` for
    /// later re-use as a `voice_prompt`.
    ///
    /// Bark's voice-cloning protocol takes a sequence of **semantic
    /// tokens** (not raw audio) extracted from a reference clip and
    /// concatenates them before the text tokens at synthesis time.
    /// Full reference-audio → semantic-token ingestion requires the
    /// Bark semantic *encoder* (a separate model not yet ported), so
    /// the API surfaced here accepts a pre-tokenized prompt via the
    /// transcript field interpreted as a hex-encoded little-endian u32
    /// stream. Callers without pre-tokenized prompts should fail
    /// gracefully with [`TtsError::Unsupported`].
    async fn clone_voice(&self, request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        // Bark clone protocol: we accept a pre-tokenized semantic-token
        // prompt encoded as raw little-endian u32s in `audio_bytes`.
        // Raw reference audio → semantic encoding requires the Bark
        // semantic encoder which is not yet ported; surface a clear
        // `Unsupported` instead of silently producing garbage.
        if !request.audio_bytes.len().is_multiple_of(4) || request.audio_bytes.is_empty() {
            return Err(TtsError::Unsupported(
                "bark clone_voice: pass pre-tokenized semantic tokens (raw LE u32 stream) in \
                 `audio_bytes`; reference-audio → semantic encoding is not yet ported"
                    .to_owned(),
            ));
        }
        let mut tokens = Vec::with_capacity(request.audio_bytes.len() / 4);
        for chunk in request.audio_bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().expect("chunks_exact(4) guarantees len 4");
            tokens.push(u32::from_le_bytes(arr));
        }
        let path = save_voice_prompt(&request.name, &tokens)?;
        Ok(VoiceHandle {
            id: path.to_string_lossy().into_owned(),
            provider: BARK_BACKEND_ID_PREFIX.to_owned(),
        })
    }

    /// Streaming synthesis for the Bark backend.
    ///
    /// This is a **chunked-after-synthesis** implementation: the full
    /// utterance is synthesised via [`Self::synthesize`] (which runs the
    /// semantic → coarse → fine → `EnCodec`-decode pipeline end-to-end),
    /// the returned 16-bit PCM WAV is decoded back to mono f32 samples
    /// at 24 kHz, then the buffer is sliced into ~250 ms windows
    /// (`STREAM_CHUNK_MS` × `BARK_SAMPLE_RATE_HZ` / 1000 samples per
    /// chunk) and yielded through a [`futures_util::stream::iter`]. The
    /// last chunk carries `is_final = true`. `latency_seconds` is
    /// reported as `None` because all chunks are emitted post-synthesis
    /// and therefore reflect post-hoc slicing rather than measured
    /// per-frame model latency. True per-frame streaming — yielding
    /// audio while the fine acoustic stage is still generating
    /// codebook frames through `EnCodec` — is a follow-up enhancement
    /// (it requires restructuring the pipeline to expose an incremental
    /// fine-decode → codec loop rather than the current one-shot path).
    async fn stream_synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        mut options: TtsOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingAudioChunk, TtsError>> + Send>>, TtsError>
    {
        if let Some(v) = voice {
            options.voice = Some(v.to_owned());
        }
        let generated = self.synthesize(text, &options).await?;
        let pcm = decode_wav_16bit_to_f32(&generated.bytes)?;
        let chunks = chunk_pcm_into_windows(&pcm, pipeline::BARK_SAMPLE_RATE_HZ, STREAM_CHUNK_MS);
        Ok(Box::pin(futures_util::stream::iter(
            chunks.into_iter().map(Ok),
        )))
    }

    // `list_voices`, `design_voice`, `delete_voice` inherit the
    // `TtsBackend` default impls (all returning `Unsupported`).
}

/// Streaming chunk window size in milliseconds. 250 ms strikes a
/// reasonable balance between perceived responsiveness and per-chunk
/// overhead for the chunked-after-synthesis stream emitter.
const STREAM_CHUNK_MS: u32 = 250;

/// Decode the 16-bit signed-PCM `data` chunk of a Blazen-emitted Bark
/// WAV (produced by [`pipeline::synthesize_wav`]) back to a mono f32
/// PCM buffer in [-1.0, 1.0].
///
/// Bark always emits WAV with a 44-byte canonical RIFF header followed
/// by little-endian `i16` samples (see `encode_wav_16bit` in
/// [`pipeline`]). Any malformed buffer surfaces as
/// [`TtsError::Synthesis`].
fn decode_wav_16bit_to_f32(wav: &[u8]) -> Result<Vec<f32>, TtsError> {
    const HEADER_LEN: usize = 44;
    if wav.len() < HEADER_LEN || &wav[..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err(TtsError::Synthesis(
            "bark stream_synthesize: synthesised WAV missing RIFF/WAVE header".to_owned(),
        ));
    }
    let body = &wav[HEADER_LEN..];
    if !body.len().is_multiple_of(2) {
        return Err(TtsError::Synthesis(
            "bark stream_synthesize: WAV data chunk length is not a multiple of 2 bytes".to_owned(),
        ));
    }
    let mut out = Vec::with_capacity(body.len() / 2);
    for pair in body.chunks_exact(2) {
        let raw: [u8; 2] = pair.try_into().expect("chunks_exact(2) guarantees len 2");
        let i = i16::from_le_bytes(raw);
        out.push(f32::from(i) / f32::from(i16::MAX));
    }
    Ok(out)
}

/// Slice a contiguous mono f32 PCM buffer into successive windows of
/// `window_ms` milliseconds at `sample_rate` Hz. The final window may
/// be short; only it carries `is_final = true`. An empty input yields
/// a single empty final chunk so consumers always observe a stream
/// terminator.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "window_ms * sample_rate / 1000 fits in usize for any realistic TTS clip"
)]
fn chunk_pcm_into_windows(
    pcm: &[f32],
    sample_rate: u32,
    window_ms: u32,
) -> Vec<StreamingAudioChunk> {
    let window_samples = ((u64::from(sample_rate) * u64::from(window_ms)) / 1000) as usize;
    let window_samples = window_samples.max(1);
    if pcm.is_empty() {
        return vec![StreamingAudioChunk {
            samples: Vec::new(),
            is_final: true,
            latency_seconds: None,
        }];
    }
    let total = pcm.len();
    let mut out: Vec<StreamingAudioChunk> = Vec::with_capacity(total.div_ceil(window_samples));
    let mut offset = 0;
    while offset < total {
        let end = (offset + window_samples).min(total);
        let is_final = end == total;
        out.push(StreamingAudioChunk {
            samples: pcm[offset..end].to_vec(),
            is_final,
            latency_seconds: None,
        });
        offset = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_suno_reference() {
        let cfg = BarkConfig::default();
        assert_eq!(cfg.model_id, "suno/bark-small");
        assert!((cfg.semantic_temperature - 0.7).abs() < f32::EPSILON);
        assert!((cfg.coarse_temperature - 0.7).abs() < f32::EPSILON);
        assert!((cfg.fine_temperature - 0.5).abs() < f32::EPSILON);
        assert!(cfg.voice_prompt.is_none());
    }

    #[test]
    fn new_sets_id_from_model_id() {
        let backend = BarkBackend::new(BarkConfig::default());
        assert_eq!(backend.id(), "bark:suno/bark-small");
        assert_eq!(backend.model_id(), "suno/bark-small");
        assert_eq!(backend.provider_kind(), "tts");
    }

    #[test]
    fn new_honors_custom_model_id() {
        let backend = BarkBackend::new(BarkConfig {
            model_id: "suno/bark".to_owned(),
            ..BarkConfig::default()
        });
        assert_eq!(backend.id(), "bark:suno/bark");
    }

    #[tokio::test]
    async fn clone_voice_with_empty_audio_bytes_returns_unsupported_with_clear_hint() {
        // Empty audio_bytes hits the not-yet-ported reference-audio path.
        let backend = BarkBackend::default();
        let req = CloneVoiceRequest {
            name: "test".to_owned(),
            audio_bytes: Vec::new(),
            transcript: None,
        };
        let err = backend
            .clone_voice(req)
            .await
            .expect_err("empty audio_bytes must error");
        match err {
            TtsError::Unsupported(msg) => {
                assert!(msg.contains("pre-tokenized"), "msg = {msg}");
                assert!(msg.contains("semantic"), "msg = {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn clone_voice_with_misaligned_audio_bytes_returns_unsupported() {
        let backend = BarkBackend::default();
        let req = CloneVoiceRequest {
            name: "bad".to_owned(),
            audio_bytes: vec![1, 2, 3], // 3 bytes — not a multiple of 4
            transcript: None,
        };
        let err = backend
            .clone_voice(req)
            .await
            .expect_err("misaligned audio_bytes must error");
        assert!(matches!(err, TtsError::Unsupported(_)));
    }

    #[tokio::test]
    async fn clone_voice_persists_pretokenized_prompt_and_returns_handle() {
        // Hermetic: override the voice cache dir to a temp path.
        let tmp =
            std::env::temp_dir().join(format!("blazen-bark-clone-test-{}", std::process::id()));
        let prev = std::env::var_os("BLAZEN_BARK_VOICE_DIR");
        // SAFETY: single-threaded mutation of a process-wide env var;
        // this test path is the only writer in the module.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("BLAZEN_BARK_VOICE_DIR", &tmp);
        }

        let backend = BarkBackend::default();
        let tokens: [u32; 4] = [10, 20, 30, 40];
        let mut bytes: Vec<u8> = Vec::new();
        for t in &tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        let req = CloneVoiceRequest {
            name: "alice".to_owned(),
            audio_bytes: bytes,
            transcript: None,
        };
        let handle = backend
            .clone_voice(req)
            .await
            .expect("clone_voice should succeed");
        assert_eq!(handle.provider, BARK_BACKEND_ID_PREFIX);
        assert!(
            handle.id.contains("alice"),
            "handle id should include voice name: {}",
            handle.id
        );
        // The file on disk should exist and round-trip the tokens.
        let loaded = load_voice_prompt(std::path::Path::new(&handle.id)).expect("loadable");
        assert_eq!(loaded, tokens.to_vec());

        let _ = std::fs::remove_dir_all(&tmp);
        #[allow(unsafe_code)]
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLAZEN_BARK_VOICE_DIR", v),
                None => std::env::remove_var("BLAZEN_BARK_VOICE_DIR"),
            }
        }
    }

    #[tokio::test]
    async fn is_loaded_starts_false_until_pipeline_is_built() {
        // Without calling load_pipeline / synthesize, the lazy cell
        // stays empty.
        let backend = BarkBackend::default();
        assert!(!backend.is_loaded().await);
    }

    #[test]
    fn config_defaults_include_encodec_model_id() {
        let cfg = BarkConfig::default();
        assert_eq!(cfg.encodec_model_id, "facebook/encodec_24khz");
    }

    /// Chunking helper round-trip: a synthetic 1-second mono f32 buffer
    /// at 24 kHz must split into 4 × 250 ms windows whose concatenation
    /// reproduces the input bit-for-bit, with `is_final` set only on
    /// the last chunk.
    #[test]
    fn chunk_pcm_into_windows_splits_one_second_into_four_quarter_second_chunks() {
        let sr = pipeline::BARK_SAMPLE_RATE_HZ;
        let total = sr as usize;
        let pcm: Vec<f32> = (0..total)
            .map(|i| {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "test fixture indexes well below f32 precision limit"
                )]
                let x = i as f32 / total as f32;
                x
            })
            .collect();
        let chunks = chunk_pcm_into_windows(&pcm, sr, STREAM_CHUNK_MS);

        assert_eq!(chunks.len(), 4, "1s ÷ 250ms = 4 chunks");
        assert_eq!(chunks[0].samples.len(), (sr as usize) / 4);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.is_final, i == chunks.len() - 1, "chunk {i} is_final");
            assert!(c.latency_seconds.is_none());
        }
        let concat: Vec<f32> = chunks.into_iter().flat_map(|c| c.samples).collect();
        assert_eq!(concat, pcm);
    }

    /// Trailing-window correctness: 1.1 s of audio produces 5 chunks
    /// (four full 250 ms windows + one short 100 ms tail), and only
    /// the tail carries `is_final = true`.
    #[test]
    fn chunk_pcm_into_windows_handles_trailing_partial_window() {
        let sr = pipeline::BARK_SAMPLE_RATE_HZ;
        let total = (sr as usize) + (sr as usize) / 10;
        let pcm = vec![0.25_f32; total];
        let chunks = chunk_pcm_into_windows(&pcm, sr, STREAM_CHUNK_MS);

        assert_eq!(chunks.len(), 5);
        for c in chunks.iter().take(4) {
            assert!(!c.is_final);
            assert_eq!(c.samples.len(), (sr as usize) / 4);
        }
        let tail = chunks.last().expect("non-empty");
        assert!(tail.is_final);
        assert_eq!(tail.samples.len(), (sr as usize) / 10);
    }

    /// Empty PCM input still produces one terminating empty chunk so
    /// downstream consumers always see an end-of-stream marker.
    #[test]
    fn chunk_pcm_into_windows_yields_single_empty_final_chunk_for_empty_input() {
        let chunks = chunk_pcm_into_windows(&[], pipeline::BARK_SAMPLE_RATE_HZ, STREAM_CHUNK_MS);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].samples.is_empty());
        assert!(chunks[0].is_final);
    }

    /// `decode_wav_16bit_to_f32` reverses the WAV encoder used by the
    /// Bark pipeline. Encoding a synthetic ramp and decoding it back
    /// must recover the input within `i16` quantisation error
    /// (1/32767 ≈ 3e-5).
    #[test]
    fn decode_wav_16bit_to_f32_round_trips_against_pipeline_encoder() {
        // Synthesize a tiny 3-sample buffer and round-trip via the
        // pipeline's WAV encoder (re-exposed through `synthesize_wav`
        // is not necessary here — we mirror its 16-bit-PCM frame
        // layout manually).
        let samples = [0.0_f32, 0.5, -0.5];
        let nbytes = u32::try_from(samples.len() * 2).expect("3-sample fixture fits u32 trivially");
        let mut wav = Vec::with_capacity(44 + samples.len() * 2);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36_u32 + nbytes).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&pipeline::BARK_SAMPLE_RATE_HZ.to_le_bytes());
        wav.extend_from_slice(&(pipeline::BARK_SAMPLE_RATE_HZ * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&nbytes.to_le_bytes());
        for &s in &samples {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "clamped sample fits i16 by construction"
            )]
            let i = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            wav.extend_from_slice(&i.to_le_bytes());
        }

        let decoded = decode_wav_16bit_to_f32(&wav).expect("decode succeeds");
        assert_eq!(decoded.len(), samples.len());
        for (got, want) in decoded.iter().zip(samples.iter()) {
            assert!((got - want).abs() < 1.0e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn decode_wav_16bit_to_f32_rejects_buffer_without_riff_header() {
        let err = decode_wav_16bit_to_f32(&[0_u8; 50]).expect_err("must reject non-RIFF buffer");
        assert!(matches!(err, TtsError::Synthesis(_)));
    }

    /// End-to-end streaming test against the real Bark model.
    ///
    /// Gated by `BLAZEN_TEST_BARK=1` because it downloads ~400 MB of
    /// weights and runs a full 3-stage synthesis (semantic + coarse +
    /// fine + `EnCodec` decode). Marked `#[ignore]` so the default
    /// `cargo nextest run --features bark` invocation skips it.
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_BARK=1 and pulls ~400 MB of Bark weights from HF Hub"]
    async fn stream_synthesize_returns_stream_with_final_chunk_at_end() {
        use futures_util::StreamExt;

        if std::env::var("BLAZEN_TEST_BARK").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_BARK != 1");
            return;
        }

        let backend = BarkBackend::default();
        let stream = backend
            .stream_synthesize("hi", None, TtsOptions::default())
            .await
            .expect("stream_synthesize returns Ok");

        let chunks: Vec<StreamingAudioChunk> = stream
            .collect::<Vec<Result<StreamingAudioChunk, TtsError>>>()
            .await
            .into_iter()
            .map(|r| r.expect("each chunk Ok"))
            .collect();

        assert!(!chunks.is_empty(), "expected at least one chunk");
        let last = chunks.last().expect("non-empty");
        assert!(last.is_final, "final chunk must have is_final = true");
        for (i, c) in chunks
            .iter()
            .enumerate()
            .take(chunks.len().saturating_sub(1))
        {
            assert!(!c.is_final, "non-terminal chunk {i} must not be final");
        }
        let concat: Vec<f32> = chunks.into_iter().flat_map(|c| c.samples).collect();
        assert!(!concat.is_empty(), "concatenated samples must be non-empty");
    }
}
