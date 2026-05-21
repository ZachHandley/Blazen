//! Pure-Rust Whisper [`SttBackend`] powered by `candle-transformers`.
//!
//! Wraps [`candle_transformers::models::whisper`] for `OpenAI` Whisper
//! inference without a C/C++ build dependency. Weights are fetched from
//! the `openai/whisper-{size}` `HuggingFace` repositories via [`hf_hub`].
//!
//! ## Pipeline
//!
//! 1. Lazy-load `model.safetensors` + `tokenizer.json` + `config.json`
//!    from the chosen HF repo (cached on disk by `hf-hub`).
//! 2. Resample-aware decode: callers passing 16 kHz mono PCM via
//!    [`CandleWhisperBackend::transcribe_inherent`] skip resampling.
//!    File-based [`SttBackend::transcribe`] callers must supply a 16 kHz
//!    mono WAV (matching the whisper.cpp backend's contract).
//! 3. Build the 80- or 128-bin log-mel spectrogram (chunked into 30-second
//!    windows) using the precomputed mel filter banks shipped under
//!    `whisper_assets/`.
//! 4. Run encoder → decoder loop with the appropriate `<|lang|>` /
//!    `<|transcribe|>` or `<|translate|>` prompt tokens, doing greedy
//!    argmax decoding (temperature = 0; no fallback) until `<|endoftext|>`.
//! 5. Detokenize and emit a [`TranscriptionResult`] with one
//!    [`TranscriptionSegment`] per 30-second chunk.
//!
//! ## Scope
//!
//! Basic transcription only. `LoRA` adapter support (training- and
//! inference-time) is deferred to **PR-WL** (task #38). The
//! [`CandleWhisperConfig`] surface intentionally omits adapter fields so
//! the follow-up PR can add them additively without breaking callers.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError};
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use candle_transformers::models::whisper::{
    self as m, Config, audio as whisper_audio, model as whisper_model,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

use crate::error::SttError;
use crate::traits::{SttBackend, TranscriptionResult, TranscriptionSegment};

// ---------------------------------------------------------------------------
// Mel filter assets (precomputed by the candle-examples whisper crate).
// 80-bin filters are used by tiny/base/small/medium/large-v3-turbo;
// large-v3 / large-v3-turbo upstream uses 128-bin — `Config.num_mel_bins`
// in the downloaded repo selects which array to slice.
// ---------------------------------------------------------------------------

const MEL_FILTERS_80: &[u8] = include_bytes!("whisper_assets/melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("whisper_assets/melfilters128.bytes");

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Whisper model variants hosted under `openai/whisper-*` on `HuggingFace`.
///
/// All sizes share the same encoder/decoder architecture; the only
/// runtime difference is parameter count (and therefore RAM/VRAM
/// footprint and accuracy). The `LargeV3Turbo` variant is `OpenAI`'s
/// October-2024 distillation of `LargeV3`: identical encoder, decoder
/// trimmed from 32 → 4 layers, ~8× faster end-to-end at near-V3 quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum WhisperModel {
    /// 39M params, ~75 MB.
    Tiny,
    /// 74M params, ~140 MB.
    Base,
    /// 244M params, ~460 MB.
    #[default]
    Small,
    /// 769M params, ~1.5 GB.
    Medium,
    /// 1550M params, ~3 GB. Highest accuracy.
    LargeV3,
    /// 809M params, ~1.6 GB. ~8× faster than `LargeV3` at near-identical
    /// quality (Oct 2024 distillation).
    LargeV3Turbo,
}

impl WhisperModel {
    /// `HuggingFace` model repository id (`openai/whisper-*`).
    #[must_use]
    pub const fn hf_repo_id(self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::Base => "openai/whisper-base",
            Self::Small => "openai/whisper-small",
            Self::Medium => "openai/whisper-medium",
            Self::LargeV3 => "openai/whisper-large-v3",
            Self::LargeV3Turbo => "openai/whisper-large-v3-turbo",
        }
    }
}

impl std::fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Tiny => "tiny",
            Self::Base => "base",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::LargeV3 => "large-v3",
            Self::LargeV3Turbo => "large-v3-turbo",
        };
        f.write_str(name)
    }
}

/// Task to perform on the decoded audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum WhisperTask {
    /// Transcribe in the source language. (Default.)
    #[default]
    Transcribe,
    /// Translate the source language → English.
    Translate,
}

/// Construction-time configuration for [`CandleWhisperBackend`].
///
/// Defaults: `Small` model on CPU, transcribe task, auto-detect language,
/// HF default cache (`~/.cache/huggingface/hub`).
#[derive(Debug, Clone)]
pub struct CandleWhisperConfig {
    /// Whisper model size to download + load.
    pub model: WhisperModel,
    /// Inference device. Use [`Device::Cpu`], [`Device::new_cuda(idx)`],
    /// or [`Device::new_metal(idx)`].
    pub device: Device,
    /// ISO 639-1 language hint (e.g. `"en"`, `"es"`). `None` triggers
    /// Whisper's built-in language detection from the first encoder
    /// segment.
    pub language: Option<String>,
    /// Whether to transcribe (default) or translate to English.
    pub task: WhisperTask,
    /// Optional override for the HF cache directory. `None` uses the HF
    /// hub default (`$HF_HOME` or `~/.cache/huggingface`).
    pub cache_dir: Option<PathBuf>,
}

impl Default for CandleWhisperConfig {
    fn default() -> Self {
        Self {
            model: WhisperModel::default(),
            device: Device::Cpu,
            language: None,
            task: WhisperTask::default(),
            cache_dir: None,
        }
    }
}

/// Loaded model + tokenizer + mel filter bank, ready for inference.
struct LoadedWhisper {
    model: whisper_model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
}

/// Local speech-to-text backend powered by `candle-transformers`'s
/// Whisper implementation.
///
/// Construct with [`CandleWhisperBackend::new`]; weights are downloaded
/// and the model loaded lazily on the first `transcribe` call (or
/// eagerly via [`AudioBackend::load`]).
pub struct CandleWhisperBackend {
    /// Stable backend identifier, e.g. `"candle-whisper:small"`.
    id: String,
    /// Configuration preserved for transcription calls.
    config: CandleWhisperConfig,
    /// Lazily-loaded model. Wrapped in `OnceCell` so `load()` and the
    /// first `transcribe()` race-safely converge on a single weight load.
    inner: Arc<OnceCell<LoadedWhisper>>,
}

impl std::fmt::Debug for CandleWhisperBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleWhisperBackend")
            .field("id", &self.id)
            .field("model", &self.config.model)
            .field("language", &self.config.language)
            .field("task", &self.config.task)
            .finish_non_exhaustive()
    }
}

impl CandleWhisperBackend {
    /// Build a new backend from configuration. Does **not** download
    /// weights or load the model — that happens on first `transcribe`
    /// call (or via [`AudioBackend::load`]).
    #[must_use]
    pub fn new(config: CandleWhisperConfig) -> Self {
        let id = format!("candle-whisper:{}", config.model);
        Self {
            id,
            config,
            inner: Arc::new(OnceCell::new()),
        }
    }

    /// The model size configured at construction time.
    #[must_use]
    pub const fn model(&self) -> WhisperModel {
        self.config.model
    }

    /// The configured language hint, if any.
    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.config.language.as_deref()
    }

    /// The configured task (transcribe or translate).
    #[must_use]
    pub const fn task(&self) -> WhisperTask {
        self.config.task
    }

    /// Transcribe a raw PCM buffer of `f32` samples in `[-1.0, 1.0]`.
    ///
    /// The supplied `sample_rate` must equal `16_000` — Whisper's fixed
    /// input sample rate. Callers with audio at other rates should
    /// resample upstream (e.g. with `rubato` or `symphonia`).
    ///
    /// This bypasses the `SttBackend::transcribe` file-decoding path,
    /// which is useful for streaming callers and for testing with
    /// synthesised audio.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::InvalidOptions`] if `sample_rate != 16_000`,
    /// [`SttError::ModelLoad`] if weight download/load fails, or
    /// [`SttError::Transcription`] for inference failures.
    pub async fn transcribe_inherent(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> Result<TranscriptionResult, SttError> {
        if sample_rate != m::SAMPLE_RATE as u32 {
            return Err(SttError::InvalidOptions(format!(
                "expected {}Hz mono audio, got {sample_rate}Hz — resample upstream",
                m::SAMPLE_RATE
            )));
        }
        let samples = audio_samples.to_vec();
        self.transcribe_pcm(samples, self.config.language.clone())
            .await
    }

    /// Eagerly download + load weights. Internal helper for
    /// [`AudioBackend::load`] and the lazy path.
    async fn get_or_load(&self) -> Result<&LoadedWhisper, SttError> {
        self.inner
            .get_or_try_init(|| async {
                let cfg = self.config.clone();
                // Heavy work (HF download + safetensors mmap +
                // VarBuilder + Tokenizer parse) is sync and CPU-bound —
                // run on a blocking thread to keep the runtime free.
                tokio::task::spawn_blocking(move || load_blocking(&cfg))
                    .await
                    .map_err(|e| SttError::ModelLoad(format!("join error: {e}")))?
            })
            .await
    }

    /// Run inference on a PCM buffer (must already be 16 kHz mono f32 in
    /// `[-1.0, 1.0]`).
    async fn transcribe_pcm(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
    ) -> Result<TranscriptionResult, SttError> {
        // Load weights first (own future borrow ends before spawn_blocking).
        self.get_or_load().await?;
        // Re-fetch a Clone-friendly handle for the blocking thread.
        // `LoadedWhisper` is not `Clone`, so we route the inference call
        // through a dedicated blocking closure that captures `Arc<OnceCell>`.
        let inner = Arc::clone(&self.inner);
        let task = self.config.task;
        tokio::task::spawn_blocking(move || {
            let loaded = inner
                .get()
                .ok_or_else(|| SttError::ModelLoad("model unexpectedly unloaded".into()))?;
            run_inference(loaded, &samples, language.as_deref(), task)
        })
        .await
        .map_err(|e| SttError::Transcription(format!("join error: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioBackend for CandleWhisperBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "stt"
    }

    async fn load(&self) -> Result<(), AudioError> {
        let _ = self
            .get_or_load()
            .await
            .map_err(|e| AudioError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn unload(&self) -> Result<(), AudioError> {
        // `OnceCell` doesn't expose a reset method on the tokio variant,
        // so unload is a no-op — the loaded weights live for the
        // lifetime of the backend. Callers wanting to drop weights
        // should drop the `CandleWhisperBackend` itself.
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        self.inner.initialized()
    }
}

#[async_trait]
impl SttBackend for CandleWhisperBackend {
    async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        let raw_bytes = tokio::fs::read(audio_path).await?;
        let samples = decode_wav_16k_mono(&raw_bytes)?;
        let lang = language
            .map(String::from)
            .or_else(|| self.config.language.clone());
        self.transcribe_pcm(samples, lang).await
    }
}

// ---------------------------------------------------------------------------
// Blocking helpers (CPU-bound work)
// ---------------------------------------------------------------------------

/// Synchronously download weights + tokenizer + config, then build the
/// `Whisper` model. Runs on `spawn_blocking`.
fn load_blocking(cfg: &CandleWhisperConfig) -> Result<LoadedWhisper, SttError> {
    let repo_id = cfg.model.hf_repo_id();

    let api = build_api(cfg.cache_dir.as_deref())?;
    let repo = api.model(repo_id.to_string());

    tracing::info!(repo = repo_id, "candle-whisper: fetching files");

    let config_path = repo
        .get("config.json")
        .map_err(|e| SttError::ModelLoad(format!("config.json: {e}")))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| SttError::ModelLoad(format!("tokenizer.json: {e}")))?;
    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| SttError::ModelLoad(format!("model.safetensors: {e}")))?;

    let config_json = std::fs::read_to_string(&config_path)
        .map_err(|e| SttError::ModelLoad(format!("read config.json: {e}")))?;
    let config: Config = serde_json::from_str(&config_json)
        .map_err(|e| SttError::ModelLoad(format!("parse config.json: {e}")))?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| SttError::ModelLoad(format!("tokenizer.json: {e}")))?;

    let mel_bytes: &[u8] = match config.num_mel_bins {
        80 => MEL_FILTERS_80,
        128 => MEL_FILTERS_128,
        n => {
            return Err(SttError::ModelLoad(format!(
                "unexpected num_mel_bins {n} (expected 80 or 128)"
            )));
        }
    };
    if !mel_bytes.len().is_multiple_of(4) {
        return Err(SttError::ModelLoad(
            "mel filter byte length not a multiple of 4".into(),
        ));
    }
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    for (i, chunk) in mel_bytes.chunks_exact(4).enumerate() {
        mel_filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    let device = cfg.device.clone();
    // SAFETY: candle's `from_mmaped_safetensors` requires `unsafe`
    // because the safetensors file must outlive the mmap and its
    // contents must not change underneath us. We pass a path rooted in
    // the hf-hub cache whose contents are immutable by convention
    // (hf-hub writes via atomic-rename and never mutates downloaded
    // blobs in place).
    #[allow(unsafe_code)]
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.as_path()], m::DTYPE, &device)
            .map_err(|e| SttError::ModelLoad(format!("safetensors mmap: {e}")))?
    };
    let model = whisper_model::Whisper::load(&vb, config.clone())
        .map_err(|e| SttError::ModelLoad(format!("Whisper::load: {e}")))?;

    tracing::info!(repo = repo_id, "candle-whisper: model loaded");

    Ok(LoadedWhisper {
        model,
        tokenizer,
        config,
        mel_filters,
        device,
    })
}

fn build_api(cache_dir: Option<&Path>) -> Result<hf_hub::api::sync::Api, SttError> {
    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(dir) = cache_dir {
        builder = builder.with_cache_dir(dir.to_path_buf());
    }
    builder
        .build()
        .map_err(|e| SttError::ModelLoad(format!("hf-hub api: {e}")))
}

/// Run the full Whisper pipeline (mel spec → encoder → decoder loop).
fn run_inference(
    loaded: &LoadedWhisper,
    samples: &[f32],
    language: Option<&str>,
    task: WhisperTask,
) -> Result<TranscriptionResult, SttError> {
    if samples.is_empty() {
        return Err(SttError::Transcription("empty audio input".into()));
    }

    // Compute log-mel spectrogram for the entire utterance.
    let mel = whisper_audio::pcm_to_mel(&loaded.config, samples, &loaded.mel_filters);
    let mel_len = mel.len();
    let n_mel = loaded.config.num_mel_bins;
    let mel = Tensor::from_vec(mel, (1, n_mel, mel_len / n_mel), &loaded.device)
        .map_err(|e| SttError::Transcription(format!("mel tensor: {e}")))?;

    // We mutate a Clone of the loaded model so KV cache state doesn't
    // accumulate across calls. `Whisper` derives `Clone`, so this is
    // cheap (Arc-share of weights, fresh KV buffers).
    let mut model = loaded.model.clone();
    let tokenizer = &loaded.tokenizer;
    let cfg = &loaded.config;

    let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
    let transcribe_token = token_id(tokenizer, m::TRANSCRIBE_TOKEN)?;
    let translate_token = token_id(tokenizer, m::TRANSLATE_TOKEN)?;
    let eot_token = token_id(tokenizer, m::EOT_TOKEN)?;
    let no_timestamps_token = token_id(tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

    // Build the per-decode `suppress_tokens` mask once.
    let suppress_tokens: Vec<f32> = (0..cfg.vocab_size as u32)
        .map(|i| {
            if cfg.suppress_tokens.contains(&i) {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &loaded.device)
        .map_err(|e| SttError::Transcription(format!("suppress mask: {e}")))?;

    // Determine the language token. `language` arg ("en", "es", ...) is
    // converted to `<|en|>` etc.; `None` triggers detection from the
    // first encoder segment.
    let (language_token, detected_lang_code) =
        resolve_language_token(&mut model, tokenizer, &mel, language)?;

    let (_, _, content_frames) = mel
        .dims3()
        .map_err(|e| SttError::Transcription(format!("mel dims: {e}")))?;

    let mut full_text = String::new();
    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut seek = 0;
    while seek < content_frames {
        let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
        let segment_start_ms =
            i64::try_from(seek * m::HOP_LENGTH * 1000 / m::SAMPLE_RATE).unwrap_or(i64::MAX);
        let segment_end_ms =
            i64::try_from((seek + segment_size) * m::HOP_LENGTH * 1000 / m::SAMPLE_RATE)
                .unwrap_or(i64::MAX);
        let mel_segment = mel
            .narrow(2, seek, segment_size)
            .map_err(|e| SttError::Transcription(format!("mel narrow: {e}")))?;

        let tokens = decode_segment(
            &mut model,
            tokenizer,
            &mel_segment,
            &suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            no_timestamps_token,
            eot_token,
            language_token,
            task,
            cfg.max_target_positions,
        )?;

        let text = tokenizer
            .decode(&tokens, true)
            .map_err(|e| SttError::Transcription(format!("detokenize: {e}")))?;
        let text = text.trim().to_string();
        if !text.is_empty() {
            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&text);
            segments.push(TranscriptionSegment {
                start_ms: segment_start_ms,
                end_ms: segment_end_ms,
                text,
            });
        }

        seek += segment_size;
    }

    Ok(TranscriptionResult {
        text: full_text,
        segments,
        language: detected_lang_code,
    })
}

/// Greedy decode loop for a single 30-second mel segment. Returns the
/// list of generated text tokens (special tokens stripped by the
/// tokenizer's `skip_special_tokens=true` in the caller).
#[allow(clippy::too_many_arguments)]
fn decode_segment(
    model: &mut whisper_model::Whisper,
    _tokenizer: &Tokenizer,
    mel_segment: &Tensor,
    suppress_tokens: &Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    no_timestamps_token: u32,
    eot_token: u32,
    language_token: Option<u32>,
    task: WhisperTask,
    max_target_positions: usize,
) -> Result<Vec<u32>, SttError> {
    let device = mel_segment.device().clone();
    let audio_features = model
        .encoder
        .forward(mel_segment, true)
        .map_err(|e| SttError::Transcription(format!("encoder: {e}")))?;

    let mut tokens: Vec<u32> = vec![sot_token];
    if let Some(lt) = language_token {
        tokens.push(lt);
    }
    match task {
        WhisperTask::Transcribe => tokens.push(transcribe_token),
        WhisperTask::Translate => tokens.push(translate_token),
    }
    tokens.push(no_timestamps_token);

    let sample_len = max_target_positions / 2;
    for i in 0..sample_len {
        let tokens_t = Tensor::new(tokens.as_slice(), &device)
            .map_err(|e| SttError::Transcription(format!("tokens tensor: {e}")))?;
        let tokens_t = tokens_t
            .unsqueeze(0)
            .map_err(|e| SttError::Transcription(format!("unsqueeze: {e}")))?;
        let ys = model
            .decoder
            .forward(&tokens_t, &audio_features, i == 0)
            .map_err(|e| SttError::Transcription(format!("decoder: {e}")))?;

        let (_, seq_len, _) = ys
            .dims3()
            .map_err(|e| SttError::Transcription(format!("ys dims: {e}")))?;
        let logits = model
            .decoder
            .final_linear(
                &ys.i((..1, seq_len - 1..))
                    .map_err(|e| SttError::Transcription(format!("ys slice: {e}")))?,
            )
            .map_err(|e| SttError::Transcription(format!("final_linear: {e}")))?
            .i(0)
            .map_err(|e| SttError::Transcription(format!("logits i0: {e}")))?
            .i(0)
            .map_err(|e| SttError::Transcription(format!("logits i0i0: {e}")))?;
        let logits = logits
            .broadcast_add(suppress_tokens)
            .map_err(|e| SttError::Transcription(format!("suppress add: {e}")))?;

        // Greedy argmax over the masked logits.
        let logits_v: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| SttError::Transcription(format!("logits to_vec1: {e}")))?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| SttError::Transcription("empty logits".into()))?;

        if next_token == eot_token {
            break;
        }
        tokens.push(next_token);
        if tokens.len() >= max_target_positions {
            break;
        }
    }
    Ok(tokens)
}

/// Resolve the language token id from either a user-supplied ISO 639-1
/// code or by running Whisper's detection routine on the first encoder
/// segment.
///
/// Returns `(Option<token_id>, Option<iso_639_1_code>)`. `language_token`
/// is `None` for English-only model variants (which don't have any
/// `<|lang|>` tokens); `iso_639_1_code` is the human-friendly code we
/// surface on [`TranscriptionResult::language`].
fn resolve_language_token(
    model: &mut whisper_model::Whisper,
    tokenizer: &Tokenizer,
    mel: &Tensor,
    requested: Option<&str>,
) -> Result<(Option<u32>, Option<String>), SttError> {
    if let Some(lang) = requested {
        let token = format!("<|{lang}|>");
        return match tokenizer.token_to_id(&token) {
            Some(id) => Ok((Some(id), Some(lang.to_string()))),
            None => Err(SttError::InvalidOptions(format!(
                "language code `{lang}` not in tokenizer (model may be English-only)"
            ))),
        };
    }

    // Auto-detect — but only for multilingual models. Detect by
    // probing the tokenizer for `<|en|>`; English-only variants
    // don't carry per-language tokens.
    if tokenizer.token_to_id("<|en|>").is_none() {
        // English-only; no language token needed.
        return Ok((None, Some("en".into())));
    }

    let language_codes = whisper_language_codes();
    let language_token_ids: Vec<u32> = language_codes
        .iter()
        .map(|code| {
            tokenizer
                .token_to_id(&format!("<|{code}|>"))
                .ok_or_else(|| SttError::Transcription(format!("missing <|{code}|> token")))
        })
        .collect::<Result<_, _>>()?;

    let (_bsize, _, seq_len) = mel
        .dims3()
        .map_err(|e| SttError::Transcription(format!("mel dims: {e}")))?;
    let detection_mel = mel
        .narrow(2, 0, usize::min(seq_len, m::N_FRAMES))
        .map_err(|e| SttError::Transcription(format!("detection narrow: {e}")))?;
    let device = mel.device();

    let audio_features = model
        .encoder
        .forward(&detection_mel, true)
        .map_err(|e| SttError::Transcription(format!("detect encoder: {e}")))?;

    let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
    let tokens = Tensor::new(&[[sot_token]], device)
        .map_err(|e| SttError::Transcription(format!("detect tokens: {e}")))?;
    let language_token_ids_t = Tensor::new(language_token_ids.as_slice(), device)
        .map_err(|e| SttError::Transcription(format!("lang token ids: {e}")))?;
    let ys = model
        .decoder
        .forward(&tokens, &audio_features, true)
        .map_err(|e| SttError::Transcription(format!("detect decoder: {e}")))?;
    let logits = model
        .decoder
        .final_linear(
            &ys.i(..1)
                .map_err(|e| SttError::Transcription(format!("detect ys slice: {e}")))?,
        )
        .map_err(|e| SttError::Transcription(format!("detect final_linear: {e}")))?
        .i(0)
        .map_err(|e| SttError::Transcription(format!("detect logits i0: {e}")))?
        .i(0)
        .map_err(|e| SttError::Transcription(format!("detect logits i0i0: {e}")))?;
    let logits = logits
        .index_select(&language_token_ids_t, 0)
        .map_err(|e| SttError::Transcription(format!("detect index_select: {e}")))?;
    let probs = softmax(&logits, D::Minus1)
        .map_err(|e| SttError::Transcription(format!("detect softmax: {e}")))?;
    let probs: Vec<f32> = probs
        .to_vec1()
        .map_err(|e| SttError::Transcription(format!("detect to_vec1: {e}")))?;
    let (best_idx, _) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or_else(|| SttError::Transcription("empty language probs".into()))?;
    let code = language_codes[best_idx];
    Ok((Some(language_token_ids[best_idx]), Some(code.into())))
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, SttError> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| SttError::ModelLoad(format!("missing token `{token}` in tokenizer")))
}

/// The 99 ISO 639-1 language codes supported by multilingual Whisper.
/// Mirrors the list in candle-examples/examples/whisper/multilingual.rs.
const fn whisper_language_codes() -> &'static [&'static str] {
    &[
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ]
}

// ---------------------------------------------------------------------------
// WAV decoding (mirrors blazen-audio-stt::backends::whispercpp)
// ---------------------------------------------------------------------------

/// Decode a 16 kHz mono 16-bit PCM WAV into `f32` samples in `[-1.0, 1.0]`.
///
/// This is the bare-minimum WAV decoder that matches the Whisper input
/// contract — for richer formats wire `symphonia` or `hound` in upstream
/// pipeline code.
fn decode_wav_16k_mono(raw_bytes: &[u8]) -> Result<Vec<f32>, SttError> {
    if raw_bytes.len() < 44 {
        return Err(SttError::Transcription(
            "audio file too small to be a valid WAV".into(),
        ));
    }
    if &raw_bytes[..4] != b"RIFF" || &raw_bytes[8..12] != b"WAVE" {
        return Err(SttError::Transcription(
            "audio file is not a valid WAV (missing RIFF/WAVE header)".into(),
        ));
    }

    let pcm_data = &raw_bytes[44..];
    let mut samples = Vec::with_capacity(pcm_data.len() / 2);
    for chunk in pcm_data.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(f32::from(s) / 32_768.0);
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_small_cpu_transcribe_auto_detect() {
        let cfg = CandleWhisperConfig::default();
        assert_eq!(cfg.model, WhisperModel::Small);
        assert!(matches!(cfg.device, Device::Cpu));
        assert!(cfg.language.is_none());
        assert_eq!(cfg.task, WhisperTask::Transcribe);
        assert!(cfg.cache_dir.is_none());
    }

    #[test]
    fn whisper_model_display_round_trip() {
        assert_eq!(WhisperModel::Tiny.to_string(), "tiny");
        assert_eq!(WhisperModel::Base.to_string(), "base");
        assert_eq!(WhisperModel::Small.to_string(), "small");
        assert_eq!(WhisperModel::Medium.to_string(), "medium");
        assert_eq!(WhisperModel::LargeV3.to_string(), "large-v3");
        assert_eq!(WhisperModel::LargeV3Turbo.to_string(), "large-v3-turbo");
    }

    #[test]
    fn whisper_model_hf_repo_ids_are_openai() {
        for m in [
            WhisperModel::Tiny,
            WhisperModel::Base,
            WhisperModel::Small,
            WhisperModel::Medium,
            WhisperModel::LargeV3,
            WhisperModel::LargeV3Turbo,
        ] {
            assert!(m.hf_repo_id().starts_with("openai/whisper-"));
        }
        assert_eq!(
            WhisperModel::LargeV3Turbo.hf_repo_id(),
            "openai/whisper-large-v3-turbo"
        );
    }

    #[test]
    fn whisper_task_default_is_transcribe() {
        assert_eq!(WhisperTask::default(), WhisperTask::Transcribe);
    }

    #[test]
    fn backend_id_includes_model_size() {
        let backend = CandleWhisperBackend::new(CandleWhisperConfig {
            model: WhisperModel::Tiny,
            ..CandleWhisperConfig::default()
        });
        assert_eq!(
            <CandleWhisperBackend as AudioBackend>::id(&backend),
            "candle-whisper:tiny"
        );
        assert_eq!(backend.model(), WhisperModel::Tiny);
        assert_eq!(
            <CandleWhisperBackend as AudioBackend>::provider_kind(&backend),
            "stt",
        );
    }

    #[test]
    fn backend_starts_unloaded() {
        let backend = CandleWhisperBackend::new(CandleWhisperConfig::default());
        // `is_loaded` is async — we can poll the underlying OnceCell.
        assert!(!backend.inner.initialized());
    }

    #[test]
    fn language_codes_has_99_entries() {
        // Whisper supports 99 languages — this guards against accidental
        // truncation when refactoring the language array.
        assert_eq!(whisper_language_codes().len(), 99);
    }

    #[test]
    fn mel_filter_assets_are_aligned() {
        // 80-bin filters: 80 mel bands × 201 fft bins × 4 bytes (f32).
        assert_eq!(MEL_FILTERS_80.len() % 4, 0);
        assert_eq!(MEL_FILTERS_128.len() % 4, 0);
        // 128-bin file should be exactly 128/80 × the 80-bin file size.
        assert_eq!(MEL_FILTERS_128.len(), MEL_FILTERS_80.len() * 128 / 80);
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let backend = CandleWhisperBackend::new(CandleWhisperConfig {
            model: WhisperModel::Tiny,
            ..CandleWhisperConfig::default()
        });
        let result = futures_lite_block_on(backend.transcribe_inherent(&[0.0_f32; 16], 22_050));
        assert!(matches!(result, Err(SttError::InvalidOptions(_))));
    }

    #[test]
    fn decode_wav_rejects_truncated_input() {
        let result = decode_wav_16k_mono(&[0u8; 10]);
        assert!(matches!(result, Err(SttError::Transcription(_))));
    }

    #[test]
    fn decode_wav_rejects_non_riff() {
        let mut buf = vec![0u8; 64];
        buf[..4].copy_from_slice(b"OGGS");
        let result = decode_wav_16k_mono(&buf);
        assert!(matches!(result, Err(SttError::Transcription(_))));
    }

    #[test]
    fn decode_wav_parses_minimal_header() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(b"WAVE");
        // 32 more bytes to reach the 44-byte header, then a single sample.
        buf.extend_from_slice(&[0; 32]);
        buf.extend_from_slice(&i16::MAX.to_le_bytes());
        let samples = decode_wav_16k_mono(&buf).expect("decode");
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.999_969_5).abs() < 1e-6);
    }

    // ---- Smoke + live-models gated tests ------------------------------

    // Synthesise a 1-second 16 kHz 440 Hz sine wave and try to
    // transcribe it. Downloads ~75MB on first run, so mark `#[ignore]`
    // and document the opt-in flag. The transcribed text is allowed to
    // be empty/gibberish — we only assert that the pipeline runs
    // end-to-end without panicking.
    #[tokio::test]
    #[ignore = "downloads ~75MB whisper-tiny safetensors from HuggingFace"]
    async fn smoke_transcribe_sine_wave_tiny() {
        let backend = CandleWhisperBackend::new(CandleWhisperConfig {
            model: WhisperModel::Tiny,
            language: Some("en".into()),
            ..CandleWhisperConfig::default()
        });
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 / 16_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.1)
            .collect();
        let result = backend
            .transcribe_inherent(&samples, 16_000)
            .await
            .expect("transcribe should not panic");
        // No assertion on `text` — sine waves transcribe to whatever
        // Whisper hallucinates. The point is the pipeline returns.
        let _ = result.text;
    }

    /// Single-threaded synchronous poller for the tokio-free unit tests
    /// above. The crate already pulls in `tokio` with `rt` features, so
    /// we just spin up a one-shot current-thread runtime.
    fn futures_lite_block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt build")
            .block_on(f)
    }

    // Live-models test: build the candle Whisper tiny backend, load
    // real weights from HuggingFace (~75 MB) and run a 1-second 16 kHz
    // sine wave through the full pipeline. Gated behind `live-models`
    // because it fetches `model.safetensors` + `tokenizer.json` +
    // `config.json` from `openai/whisper-tiny` on first run.
    //
    // Mirrors the SNAC live-models template in
    // `crates/blazen-audio-codec/src/backends/snac.rs`.
    //
    // A pure sine wave will not produce real speech text — we only
    // assert that the pipeline runs end-to-end without panicking and
    // yields a well-formed `String` (which may be empty or contain
    // nospeech-marker hallucinations).
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn live_transcribe_sine_wave_tiny() {
        let backend = CandleWhisperBackend::new(CandleWhisperConfig {
            model: WhisperModel::Tiny,
            language: Some("en".into()),
            ..CandleWhisperConfig::default()
        });

        <CandleWhisperBackend as AudioBackend>::load(&backend)
            .await
            .expect("load whisper-tiny weights");

        // 1 second of a 440 Hz sine wave at 16 kHz, mono, amplitude 0.5.
        // The cast-precision-loss allow is fine here: `i` runs over
        // [0, 16_000) which is well within f32's exact-integer range
        // (<= 2^23 ≈ 1.6e7).
        let len = 16_000usize;
        let mut samples: Vec<f32> = Vec::with_capacity(len);
        for i in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 16_000.0;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        let result = backend
            .transcribe_inherent(&samples, 16_000)
            .await
            .expect("transcribe sine wave");

        // Weak assertion: the result must carry a `String` for `text`.
        // Content is intentionally not checked — sine waves transcribe
        // to whatever Whisper hallucinates, frequently the empty
        // string or a nospeech marker.
        let _: String = result.text;
    }
}
