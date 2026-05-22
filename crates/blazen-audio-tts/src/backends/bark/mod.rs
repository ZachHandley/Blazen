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

#![cfg(feature = "bark")]
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

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioFormat, CloneVoiceRequest, GeneratedAudio, VoiceHandle};

use crate::traits::TtsBackend;
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

    // `list_voices`, `design_voice`, `delete_voice` inherit the
    // `TtsBackend` default impls (all returning `Unsupported`).
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
}
