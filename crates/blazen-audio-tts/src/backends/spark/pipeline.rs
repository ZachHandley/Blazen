//! End-to-end Spark-TTS orchestration (Wave S.2.4): text →
//! [`SparkTokenizer`] → [`SparkLlm`] → parse → [`BiCodec::detokenize`]
//! → WAV.
//!
//! Mirrors upstream `cli/SparkTTS.py::SparkTTS.inference` (the
//! "no reference audio" branch — voice cloning, which additionally
//! runs `BiCodec.tokenize` over a reference clip and injects pre-computed
//! global tokens into the prompt, is gated on a future wave).
//!
//! # Threading model
//!
//! The wrapped [`SparkLlm::generate`] mutates the Qwen2 KV cache, so the
//! decoder is held inside a [`tokio::sync::Mutex`]. The
//! [`SparkTokenizer`] and [`BiCodec`] are immutable once loaded —
//! tokenization and decode go through `&self` directly.
//!
//! The pipeline is built once and then shared across `&SparkTtsBackend`
//! clones via an `Arc<OnceCell<Arc<SparkPipeline>>>` in the parent
//! module ([`super::SparkTtsBackend`]). This mirrors the lazy-load
//! convention used by the Bark and F5 backends.

#![cfg(feature = "spark-tts")]

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use candle_transformers::models::qwen2::Config as Qwen2Config;
use futures_core::Stream;
use tokio::sync::mpsc;

use super::bicodec::{BiCodec, BiCodecConfig};
use super::decoder::{SparkLlm, SparkLlmConfig};
use super::tokenizer::{SparkTokenizer, parse_generation_str};
use crate::TtsError;
use crate::traits::StreamingAudioChunk;

/// `BiCodec`'s speaker-encoder emits exactly 32 global tokens per
/// utterance (the `token_num` field in `BiCodecConfig::spark_tts_05b`),
/// each carrying a single `ResidualFSQ` quantiser index
/// (`fsq_num_quantizers = 1`). The LLM is expected to echo exactly
/// these 32 tokens between `<|start_global_token|>` and
/// `<|end_global_token|>`; we surface a clear pipeline error if the
/// generation came back short or long instead of silently reshaping
/// garbage.
const SPARK_GLOBAL_TOKEN_COUNT: usize = 32;
const SPARK_FSQ_NUM_QUANTIZERS: usize = 1;

/// Spark-TTS output is hard-coded to 16 kHz mono — both the
/// `BiCodec` vocoder rates (`[8, 5, 4, 2]` total = 320x upsample) and
/// the semantic token rate (50 Hz) target this rate. Mirrors upstream
/// `cli/SparkTTS.py::SparkTTS.sample_rate`.
pub(super) const SPARK_TTS_SAMPLE_RATE_HZ: u32 = 16_000;

/// End-to-end Spark-TTS pipeline. Built once via [`SparkPipeline::load`]
/// and reused across every synthesis call.
pub(super) struct SparkPipeline {
    tokenizer: Arc<SparkTokenizer>,
    // `Arc<Mutex<_>>` because `SparkLlm::generate` takes `&mut self`
    // (KV cache mutation) AND the streaming path needs to hand an
    // `OwnedMutexGuard` to `tokio::task::spawn_blocking` so the AR loop
    // can drive a non-Send token sink from a blocking worker.
    llm: Arc<tokio::sync::Mutex<SparkLlm>>,
    bicodec: Arc<BiCodec>,
    device: Device,
}

/// Errors surfaced by [`SparkPipeline::load`] / [`SparkPipeline::synthesize_pcm`].
#[derive(thiserror::Error, Debug)]
pub(super) enum PipelineError {
    /// Weights resolution failed (download, cache, missing files).
    #[error("weights: {0}")]
    Weights(#[from] super::weights::WeightsError),
    /// Tokenizer load / encode / parse failed.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// LLM load or generation failed.
    #[error("llm: {0}")]
    Llm(String),
    /// `BiCodec` load or detokenize failed.
    #[error("bicodec: {0}")]
    BiCodec(String),
    /// Reading / parsing a config file (LLM `config.json`) failed.
    #[error("config: {0}")]
    Config(String),
    /// The LLM emitted a number of `<|bicodec_global_K|>` tokens that
    /// does not divide evenly into the expected
    /// `(token_num, fsq_num_quantizers)` global tensor.
    #[error(
        "expected {expected} global tokens (token_num={token_num} × fsq_num_quantizers={quantizers}), got {actual}"
    )]
    UnexpectedGlobalCount {
        /// Tokens we needed for the global tensor.
        expected: usize,
        /// `BiCodecConfig::token_num`.
        token_num: usize,
        /// `BiCodecConfig::fsq_num_quantizers`.
        quantizers: usize,
        /// Tokens we actually parsed.
        actual: usize,
    },
}

impl From<PipelineError> for TtsError {
    fn from(e: PipelineError) -> Self {
        match e {
            PipelineError::Weights(inner) => inner.into(),
            // Config failures happen during `load` and are model-load issues.
            PipelineError::Config(_) => TtsError::ModelLoad(e.to_string()),
            // Everything else is a synthesis-time failure.
            _ => TtsError::Synthesis(e.to_string()),
        }
    }
}

/// Parse `LLM/config.json` into a [`Qwen2Config`].
///
/// The HF transformers `config.json` carries many fields candle's
/// [`Qwen2Config`] doesn't model (e.g. `architectures`, `model_type`,
/// `torch_dtype`); serde's default behaviour is to ignore unknown
/// fields, so the deserialise succeeds as long as the required ones
/// are present.
fn parse_llm_config(path: &Path) -> Result<Qwen2Config, PipelineError> {
    let bytes = std::fs::read(path)
        .map_err(|e| PipelineError::Config(format!("read {}: {e}", path.display())))?;
    serde_json::from_slice::<Qwen2Config>(&bytes)
        .map_err(|e| PipelineError::Config(format!("parse {}: {e}", path.display())))
}

/// Walk `dir` and return every `*.safetensors` shard in ascending name
/// order. Single-file checkpoints surface as a one-element vec.
fn collect_safetensors_shards(dir: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    let mut out: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| PipelineError::Config(format!("read_dir {}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| PipelineError::Config(format!("read_dir entry {}: {e}", dir.display())))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors")
            && !path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|name| name.ends_with(".index.safetensors"))
        {
            out.push(path);
        }
    }
    out.sort();
    if out.is_empty() {
        return Err(PipelineError::Config(format!(
            "no *.safetensors files found under {}",
            dir.display()
        )));
    }
    Ok(out)
}

impl SparkPipeline {
    /// Load + initialise a pipeline from a Spark-TTS bundle directory
    /// containing `LLM/` and `BiCodec/` subdirectories.
    ///
    /// Use [`super::weights::ensure_downloaded`] to resolve the bundle
    /// directory before calling this.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::Config`] when `LLM/config.json` is missing /
    ///   malformed or no safetensors shards are discoverable under
    ///   `LLM/`.
    /// * [`PipelineError::Tokenizer`] when `LLM/tokenizer.json` cannot
    ///   be loaded.
    /// * [`PipelineError::Llm`] when the Qwen2 safetensors cannot be
    ///   mmap'd or its tensor shapes don't match `config.json`.
    /// * [`PipelineError::BiCodec`] when the `BiCodec` safetensors
    ///   cannot be loaded or its sub-module shapes are wrong.
    #[allow(
        clippy::unused_async,
        reason = "The body is currently all sync I/O (file reads + candle \
                  safetensors mmap + sync sub-module loaders), but the async \
                  signature is part of the public surface so future moves to \
                  background-thread loading or async-friendly device init \
                  don't churn callers. Mirrors weights::ensure_downloaded's \
                  async signature."
    )]
    pub(super) async fn load(model_dir: &Path, device: Device) -> Result<Self, PipelineError> {
        // 1. Parse `LLM/config.json` → Qwen2Config.
        let llm_dir = model_dir.join("LLM");
        let qwen_config = parse_llm_config(&llm_dir.join("config.json"))?;

        // 2. Discover the Qwen2 safetensors shards. `SparkAudio/Spark-TTS-0.5B`
        //    ships a single ~990 MB file (`LLM/model.safetensors`); sharded
        //    forks (`model-00001-of-00002.safetensors`, …) are also handled
        //    via `collect_safetensors_shards` returning every `*.safetensors`
        //    in name order.
        let llm_shards = collect_safetensors_shards(&llm_dir)?;

        // 3. Load the tokenizer.
        let tokenizer = SparkTokenizer::load(&llm_dir.join("tokenizer.json"))
            .map_err(|e| PipelineError::Tokenizer(e.to_string()))?;

        // 4. Load the Qwen2.5 LLM.
        let llm = SparkLlm::load_from_paths(
            &llm_shards,
            &qwen_config,
            device.clone(),
            SparkLlmConfig::default(),
        )
        .map_err(|e| PipelineError::Llm(e.to_string()))?;

        // 5. Load the BiCodec. `BiCodec::from_safetensors` accepts the
        //    bundle root (it walks down into `BiCodec/`) so we hand it
        //    `model_dir` directly.
        let bicodec = BiCodec::from_safetensors(model_dir, BiCodecConfig::spark_tts_05b(), &device)
            .map_err(|e| PipelineError::BiCodec(e.to_string()))?;

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            llm: Arc::new(tokio::sync::Mutex::new(llm)),
            bicodec: Arc::new(bicodec),
            device,
        })
    }

    /// Run text-to-speech end-to-end.
    ///
    /// Returns 16 kHz mono `f32` PCM in `[-1, 1]`.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::Tokenizer`] when the prompt cannot be encoded
    ///   or the LLM output cannot be parsed back into `BiCodec`
    ///   indices.
    /// * [`PipelineError::Llm`] when the AR loop fails (OOM, sampling
    ///   error, …).
    /// * [`PipelineError::UnexpectedGlobalCount`] when the LLM emits
    ///   the wrong number of global tokens (expected exactly
    ///   `token_num * fsq_num_quantizers = 32` for the canonical
    ///   `SparkAudio/Spark-TTS-0.5B` config).
    /// * [`PipelineError::BiCodec`] when codec detokenization fails.
    pub(super) async fn synthesize_pcm(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        // 1. Build the TTS prompt token ids.
        let prompt_ids = self
            .tokenizer
            .build_tts_prompt(text)
            .map_err(|e| PipelineError::Tokenizer(e.to_string()))?;
        let prompt_len = prompt_ids.len();
        let prompt_tensor = Tensor::from_vec(prompt_ids, (1, prompt_len), &self.device)
            .map_err(|e| PipelineError::Llm(format!("prompt tensor: {e}")))?;

        // 2. Run the AR loop. Mutex held only for the generation —
        //    BiCodec decode + WAV encode happen lock-free below.
        let generated = {
            let mut llm = self.llm.lock().await;
            llm.generate(&prompt_tensor)
                .map_err(|e| PipelineError::Llm(e.to_string()))?
        };

        // 3. Strip the prompt — we only feed the LLM-generated suffix to
        //    the BiCodec parser.
        let all_ids: Vec<u32> = generated
            .flatten_all()
            .and_then(|t| t.to_vec1::<u32>())
            .map_err(|e| PipelineError::Llm(format!("flatten generated: {e}")))?;
        if all_ids.len() < prompt_len {
            return Err(PipelineError::Llm(format!(
                "generate returned {} ids which is shorter than prompt ({prompt_len})",
                all_ids.len()
            )));
        }
        let new_tokens = &all_ids[prompt_len..];

        // 4. Parse out semantic + global indices.
        let (semantic_ids, global_ids) = self
            .tokenizer
            .parse_generation(new_tokens)
            .map_err(|e| PipelineError::Tokenizer(e.to_string()))?;

        // 5. Build the tensors BiCodec::detokenize wants.
        //
        // Semantic: (1, T_sem). Global: (1, token_num, fsq_num_quantizers).
        //
        // `token_num` and `fsq_num_quantizers` are hard-coded against
        // the canonical `SparkAudio/Spark-TTS-0.5B` BiCodec config (32 ×
        // 1) — the in-tree
        // `bicodec_config_default_spark_tts_05b_matches_upstream`
        // unit test in `super::bicodec::tests` guards the values against
        // drift. The BiCodec fields themselves are module-private (the
        // `speaker: SpeakerEncoderConfig` field is `pub(super)`-scoped),
        // so we keep the constants in lock-step with the codec config
        // via that compile-time test rather than a runtime read.
        let t_sem = semantic_ids.len();
        let semantic = Tensor::from_vec(semantic_ids, (1, t_sem), &self.device)
            .map_err(|e| PipelineError::BiCodec(format!("semantic tensor: {e}")))?;
        let expected_global = SPARK_GLOBAL_TOKEN_COUNT * SPARK_FSQ_NUM_QUANTIZERS;
        if global_ids.len() != expected_global {
            return Err(PipelineError::UnexpectedGlobalCount {
                expected: expected_global,
                token_num: SPARK_GLOBAL_TOKEN_COUNT,
                quantizers: SPARK_FSQ_NUM_QUANTIZERS,
                actual: global_ids.len(),
            });
        }
        let global = Tensor::from_vec(
            global_ids,
            (1, SPARK_GLOBAL_TOKEN_COUNT, SPARK_FSQ_NUM_QUANTIZERS),
            &self.device,
        )
        .map_err(|e| PipelineError::BiCodec(format!("global tensor: {e}")))?;

        // 6. Detokenize → (1, 1, T_wav) f32 in [-1, 1] (tanh-bounded by
        //    the WaveGenerator).
        let wav_tensor = self
            .bicodec
            .detokenize(&semantic, &global)
            .map_err(|e| PipelineError::BiCodec(e.to_string()))?;

        // 7. Flatten to Vec<f32>.
        let pcm = wav_tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| PipelineError::BiCodec(format!("flatten wav: {e}")))?;
        Ok(pcm)
    }

    /// Same as [`Self::synthesize_pcm`] but encodes the result as a
    /// 16 kHz mono 16-bit-PCM WAV.
    ///
    /// # Errors
    ///
    /// See [`Self::synthesize_pcm`].
    pub(super) async fn synthesize_wav(&self, text: &str) -> Result<Vec<u8>, PipelineError> {
        let pcm = self.synthesize_pcm(text).await?;
        Ok(pcm_to_wav(&pcm, SPARK_TTS_SAMPLE_RATE_HZ, 1))
    }

    /// Per-batch streaming variant of [`Self::synthesize_pcm`]. Yields
    /// audio chunks as soon as the LLM has produced
    /// [`BICODEC_STREAM_BATCH`] fresh semantic tokens (≈
    /// `BICODEC_STREAM_BATCH * 320 / 16_000` seconds of audio per
    /// chunk).
    ///
    /// # Architecture
    ///
    /// The wave decomposes into three concurrent stages:
    ///
    /// 1. **AR loop** (background, blocking): runs on a
    ///    [`tokio::task::spawn_blocking`] worker via
    ///    [`SparkLlm::generate_streaming`]. Each freshly sampled token
    ///    id is pushed into a bounded mpsc channel; sender errors
    ///    (consumer drop) cause the AR loop to break out early.
    /// 2. **Token-to-chunk classifier** (async, on the consumer task):
    ///    forwards each id from the channel, decodes the running
    ///    suffix to text via [`SparkTokenizer`], runs the same
    ///    `bicodec_semantic_(\d+)` / `bicodec_global_(\d+)` regexes
    ///    that the offline pipeline uses ([`parse_generation_str`]),
    ///    and accumulates indices. Once
    ///    [`SPARK_GLOBAL_TOKEN_COUNT`] global tokens are known **and**
    ///    [`BICODEC_STREAM_BATCH`] fresh semantic tokens have
    ///    accumulated, dispatches a [`BiCodec::detokenize`] for that
    ///    chunk on a blocking worker. The final partial batch is
    ///    flushed when the AR loop terminates.
    /// 3. **Output stream**: each [`BiCodec::detokenize`] result is
    ///    converted into a [`StreamingAudioChunk`]. The last emitted
    ///    chunk carries `is_final = true`.
    ///
    /// # Audio quality caveat
    ///
    /// `BiCodec`'s [`super::bicodec::WaveGenerator`] is a stack of
    /// dilated convolutions whose total receptive field spans several
    /// hundred input frames. Decoding semantic-index chunks
    /// independently therefore yields per-chunk edge artefacts (clicks
    /// at the seams) where the convolutions would otherwise have seen
    /// neighbouring context. This first cut ships the simple
    /// per-chunk decode and documents the artefact — a follow-up wave
    /// can add an overlap-add window with pre-padding worth of the
    /// vocoder's receptive field to mask the seams. The artefacts are
    /// audible but the words remain intelligible, which is the
    /// intended trade-off for sub-second latency-to-first-chunk.
    ///
    /// # Errors
    ///
    /// Errors surface as `Err(PipelineError)` items inside the stream.
    /// A failure aborts the stream (no subsequent items will arrive).
    /// The same error variants as [`Self::synthesize_pcm`] are used.
    pub(super) fn synthesize_pcm_stream(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingAudioChunk, PipelineError>> + Send>> {
        let (tx, rx) =
            mpsc::channel::<Result<StreamingAudioChunk, PipelineError>>(STREAM_CHANNEL_CAPACITY);

        let tokenizer = Arc::clone(&self.tokenizer);
        let bicodec = Arc::clone(&self.bicodec);
        let llm = Arc::clone(&self.llm);
        let device = self.device.clone();
        let text = text.to_owned();

        tokio::spawn(async move {
            if let Err(err) =
                run_stream_pipeline(text, tokenizer, bicodec, llm, device, tx.clone()).await
            {
                // Best-effort: if the consumer dropped rx we just drop
                // the trailing error too.
                let _ = tx.send(Err(err)).await;
            }
        });

        receiver_into_stream(rx)
    }
}

/// Per-yielded-chunk semantic-token batch size. With `BiCodec`'s
/// `320x` temporal upsample (`upsample_rates = [8, 5, 4, 2]`) and a
/// `16 000 Hz` output rate, `8` semantic tokens correspond to
/// `8 * 320 / 16_000 = 0.16` s = **160 ms** of audio per chunk. This
/// chooses the sweet spot between latency-to-first-chunk and the
/// vocoder edge-effect overhead documented on
/// [`SparkPipeline::synthesize_pcm_stream`]. Smaller chunks would
/// reduce first-token latency further but worsen the seam clicks; the
/// chosen value matches the order-of-magnitude latency users expect
/// from a "true streaming" TTS backend.
pub(super) const BICODEC_STREAM_BATCH: usize = 8;

/// Bounded mpsc capacity feeding the output stream. Four chunks ≈ 640 ms
/// of buffered audio worth of back-pressure headroom; matches the
/// faster-whisper streaming pipeline's `OUTPUT_CHANNEL_CAPACITY`.
const STREAM_CHANNEL_CAPACITY: usize = 4;

/// Convert an mpsc receiver into a `Pin<Box<dyn Stream>>` of its items.
/// Mirrors the helper that lives in the faster-whisper streaming
/// pipeline — same shape, kept local to avoid a cross-crate re-export
/// for a four-line helper.
fn receiver_into_stream<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| (v, rx))
    }))
}

/// Drive one streaming synthesis run end-to-end. Forwards every chunk
/// it produces through `out`. Returning `Err(_)` signals the caller
/// to forward one final error item on `out`.
#[allow(
    clippy::too_many_lines,
    reason = "single-purpose orchestration that reads top-to-bottom; splitting hurts readability more than the linter's heuristic helps"
)]
async fn run_stream_pipeline(
    text: String,
    tokenizer: Arc<SparkTokenizer>,
    bicodec: Arc<BiCodec>,
    llm: Arc<tokio::sync::Mutex<SparkLlm>>,
    device: Device,
    out: mpsc::Sender<Result<StreamingAudioChunk, PipelineError>>,
) -> Result<(), PipelineError> {
    // 1. Build the TTS prompt token ids.
    let prompt_ids = tokenizer
        .build_tts_prompt(&text)
        .map_err(|e| PipelineError::Tokenizer(e.to_string()))?;
    let prompt_len = prompt_ids.len();
    let prompt_tensor = Tensor::from_vec(prompt_ids, (1, prompt_len), &device)
        .map_err(|e| PipelineError::Llm(format!("prompt tensor: {e}")))?;

    // 2. Spawn the AR loop on a blocking worker. We hand it an
    //    `OwnedMutexGuard` so it can mutate the KV cache without
    //    holding a `&` into the pipeline. The token-id channel is
    //    bounded so the AR loop applies back-pressure when the
    //    consumer falls behind on `BiCodec::detokenize` dispatches.
    let llm_guard = llm.clone().lock_owned().await;
    let (id_tx, mut id_rx) = mpsc::channel::<u32>(AR_TOKEN_CHANNEL_CAPACITY);
    let ar_handle = tokio::task::spawn_blocking(move || {
        let mut guard = llm_guard;
        guard.generate_streaming(&prompt_tensor, |tok| {
            // `blocking_send` parks the AR thread when the consumer
            // can't keep up; if the channel is closed we error so the
            // AR loop can break out cleanly.
            id_tx
                .blocking_send(tok)
                .map_err(|_| candle_core::Error::Msg("stream consumer dropped".into()))
        })
    });

    // 3. Consume the id stream, classify tokens, and dispatch
    //    BiCodec.detokenize once we have a full chunk worth of new
    //    semantic indices (with the global tokens fully known).
    let mut tokens_consumed: usize = 0;
    let mut decoded_buf = String::with_capacity(text.len() * 4);
    let mut semantic_acc: Vec<u32> = Vec::new();
    let mut semantic_emitted: usize = 0;
    let mut global_acc: Vec<u32> = Vec::new();
    let mut global_locked: bool = false;
    let mut last_chunk_was_sent: bool = false;

    while let Some(tok) = id_rx.recv().await {
        tokens_consumed += 1;
        // Skip the prompt portion — `generate_streaming` only fires
        // for newly sampled tokens, but `parse_generation_str` is
        // applied to the LLM's output stream, not the prompt, so we
        // don't have to filter here. (The hook in
        // `generate_streaming` is documented as new-tokens-only.)

        // Decode the single new id to its text form so we can extend
        // the rolling buffer. With `skip_special_tokens=false` the
        // `<|bicodec_*|>` markers survive as literal text — mirrors
        // the offline `parse_generation` path.
        let piece = match tokenizer.decode_single(tok) {
            Ok(s) => s,
            Err(e) => {
                // Decode failures are fatal to the stream — the regex
                // can't run against garbage. Forward and bail.
                let _ = ar_handle.await;
                return Err(PipelineError::Tokenizer(e.to_string()));
            }
        };
        decoded_buf.push_str(&piece);

        // Re-extract every chunk-boundary check. `parse_generation_str`
        // returns NoSemanticTokens until we've seen the first
        // `<|bicodec_semantic_K|>` marker, which is fine — we treat it
        // as "no chunk to emit yet" rather than as a hard error.
        if let Ok((semantics_so_far, globals_so_far)) = parse_generation_str(&decoded_buf) {
            // Once we see all 32 globals, lock them in (the LLM emits
            // them in a contiguous block at the start of generation).
            if !global_locked && globals_so_far.len() >= SPARK_GLOBAL_TOKEN_COUNT {
                global_acc = globals_so_far[..SPARK_GLOBAL_TOKEN_COUNT].to_vec();
                global_locked = true;
            }
            semantic_acc = semantics_so_far;
        }

        // Chunk-boundary check: dispatch once we have enough semantic
        // tokens for the next batch AND the global tokens are locked.
        while global_locked && semantic_acc.len() >= semantic_emitted + BICODEC_STREAM_BATCH {
            let batch_end = semantic_emitted + BICODEC_STREAM_BATCH;
            let batch = semantic_acc[semantic_emitted..batch_end].to_vec();
            semantic_emitted = batch_end;

            let chunk = match dispatch_chunk(&bicodec, &device, &batch, &global_acc, false).await {
                Ok(c) => c,
                Err(e) => {
                    let _ = ar_handle.await;
                    return Err(e);
                }
            };
            last_chunk_was_sent = true;
            if out.send(Ok(chunk)).await.is_err() {
                // Consumer dropped — abort. The AR loop's blocking_send
                // will already be failing the next time it fires.
                let _ = ar_handle.await;
                return Ok(());
            }
        }
    }

    // 4. AR loop is done (channel closed). Join it to surface any
    //    candle errors that fired AFTER the closure short-circuit.
    let ar_result = match ar_handle.await {
        Ok(r) => r,
        Err(join_err) => {
            return Err(PipelineError::Llm(format!(
                "generate_streaming join: {join_err}"
            )));
        }
    };
    if let Err(e) = ar_result {
        // "consumer dropped" is the canonical clean-stop path; any
        // other candle error is real.
        let msg = e.to_string();
        if !msg.contains("stream consumer dropped") {
            return Err(PipelineError::Llm(msg));
        }
    }
    let _ = tokens_consumed; // diagnostic only; suppress unused-warn.

    // 5. Flush any partial trailing batch. If `semantic_acc` has
    //    leftover indices we haven't emitted yet, decode them as the
    //    final chunk; otherwise emit a single empty `is_final = true`
    //    terminator so consumers always observe the end-of-stream
    //    marker.
    if !global_locked {
        return Err(PipelineError::UnexpectedGlobalCount {
            expected: SPARK_GLOBAL_TOKEN_COUNT * SPARK_FSQ_NUM_QUANTIZERS,
            token_num: SPARK_GLOBAL_TOKEN_COUNT,
            quantizers: SPARK_FSQ_NUM_QUANTIZERS,
            actual: global_acc.len(),
        });
    }
    if semantic_emitted == 0 && semantic_acc.is_empty() {
        return Err(PipelineError::Tokenizer(
            "no semantic tokens emitted by the LLM".to_owned(),
        ));
    }
    let tail = if semantic_acc.len() > semantic_emitted {
        semantic_acc[semantic_emitted..].to_vec()
    } else {
        Vec::new()
    };
    if !tail.is_empty() {
        match dispatch_chunk(&bicodec, &device, &tail, &global_acc, true).await {
            Ok(c) => {
                let _ = out.send(Ok(c)).await;
            }
            Err(e) => return Err(e),
        }
    } else if last_chunk_was_sent {
        // Emit a final empty terminator so consumers observe
        // is_final=true. Mirrors Bark's behaviour for empty trailing
        // windows.
        let _ = out
            .send(Ok(StreamingAudioChunk {
                samples: Vec::new(),
                is_final: true,
                latency_seconds: None,
            }))
            .await;
    } else {
        // Pathological: zero chunks ever emitted. Surface as an error
        // rather than silently producing an empty stream.
        return Err(PipelineError::Tokenizer(
            "no semantic tokens emitted by the LLM".to_owned(),
        ));
    }

    Ok(())
}

/// Dispatch one [`BiCodec::detokenize`] call for a single semantic
/// chunk + the already-known global tokens. Returns a
/// [`StreamingAudioChunk`] ready to forward downstream.
async fn dispatch_chunk(
    bicodec: &Arc<BiCodec>,
    device: &Device,
    semantic_batch: &[u32],
    global_tokens: &[u32],
    is_final: bool,
) -> Result<StreamingAudioChunk, PipelineError> {
    let t_sem = semantic_batch.len();
    let semantic = Tensor::from_vec(semantic_batch.to_vec(), (1, t_sem), device)
        .map_err(|e| PipelineError::BiCodec(format!("semantic tensor: {e}")))?;
    let global = Tensor::from_vec(
        global_tokens.to_vec(),
        (1, SPARK_GLOBAL_TOKEN_COUNT, SPARK_FSQ_NUM_QUANTIZERS),
        device,
    )
    .map_err(|e| PipelineError::BiCodec(format!("global tensor: {e}")))?;

    let bicodec = Arc::clone(bicodec);
    let wav_tensor = tokio::task::spawn_blocking(move || bicodec.detokenize(&semantic, &global))
        .await
        .map_err(|e| PipelineError::BiCodec(format!("detokenize join: {e}")))?
        .map_err(|e| PipelineError::BiCodec(e.to_string()))?;
    let pcm = wav_tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| PipelineError::BiCodec(format!("flatten wav: {e}")))?;
    Ok(StreamingAudioChunk {
        samples: pcm,
        is_final,
        latency_seconds: None,
    })
}

/// Bounded capacity for the AR-loop -> classifier channel. 64 tokens
/// is ~1.3 s of audio worth of buffered ids (at the 50 Hz semantic
/// rate) — plenty for the classifier to keep up with the AR loop on
/// a CPU-only host without blocking the AR worker for long.
const AR_TOKEN_CHANNEL_CAPACITY: usize = 64;

/// Pack `f32` PCM in `[-1, 1]` into a 16-bit-PCM WAV byte vector
/// (16 kHz, mono by default but `sample_rate` / `channels` are
/// parameterised for clarity).
///
/// Sibling backends (`musicgen`, `stable_audio`, `bark`) keep a copy of
/// this helper in their respective crates; sharing across feature
/// flags would require a public re-export through `blazen-audio` which
/// is out of scope for Wave S.2.4. The implementation mirrors them
/// byte-for-byte (clamp → scale to `i16` → little-endian RIFF/WAVE
/// container).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "WAV `data` chunk size is a u32 by spec; sample buffers above 2^32 bytes \
              are impossible in practice. Clamped to [-1.0, 1.0] before scaling, so \
              multiplying by i16::MAX stays inside i16 range by construction."
)]
fn pcm_to_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let data_size = (samples.len() * usize::from(bits_per_sample / 8)) as u32;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1_u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * f32::from(i16::MAX)) as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_error_weights_routes_to_model_load() {
        // A bare WeightsError → PipelineError (via From) → TtsError must
        // land in ModelLoad (download-style failures are model-load
        // issues by the public surface's convention).
        let weights_err =
            super::super::weights::WeightsError::CacheDir("could not determine home dir".into());
        let pipe: PipelineError = weights_err.into();
        match &pipe {
            PipelineError::Weights(_) => {}
            other => panic!("expected Weights variant, got {other:?}"),
        }
        let tts: TtsError = pipe.into();
        match tts {
            TtsError::ModelLoad(msg) => {
                assert!(msg.contains("could not determine home dir"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn pipeline_error_synthesis_variants_route_to_synthesis() {
        let err = PipelineError::Llm("rope theta is wrong".into());
        let tts: TtsError = err.into();
        assert!(matches!(tts, TtsError::Synthesis(_)));

        let err = PipelineError::BiCodec("shape mismatch".into());
        let tts: TtsError = err.into();
        assert!(matches!(tts, TtsError::Synthesis(_)));

        let err = PipelineError::Tokenizer("regex returned nothing".into());
        let tts: TtsError = err.into();
        assert!(matches!(tts, TtsError::Synthesis(_)));

        let err = PipelineError::UnexpectedGlobalCount {
            expected: 32,
            token_num: 32,
            quantizers: 1,
            actual: 7,
        };
        let tts: TtsError = err.into();
        match tts {
            TtsError::Synthesis(msg) => {
                assert!(msg.contains("expected 32"), "msg = {msg}");
                assert!(msg.contains("got 7"), "msg = {msg}");
            }
            other => panic!("expected Synthesis, got {other:?}"),
        }
    }

    #[test]
    fn pipeline_error_config_routes_to_model_load() {
        let err = PipelineError::Config("config.json: invalid utf-8".into());
        let tts: TtsError = err.into();
        match tts {
            TtsError::ModelLoad(msg) => {
                assert!(msg.contains("config.json"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn synthesize_wav_with_invalid_model_dir_returns_helpful_error() {
        // No network needed: parse_llm_config fails immediately on a
        // nonexistent path before any weight/codec load is attempted.
        let bogus = Path::new("/nonexistent/spark-tts-pipeline-test");
        let result = SparkPipeline::load(bogus, Device::Cpu).await;
        match result {
            Ok(_) => panic!("loading from a nonexistent path must fail"),
            Err(PipelineError::Config(msg)) => {
                assert!(
                    msg.contains("/nonexistent/spark-tts-pipeline-test"),
                    "Config error should mention the bogus path; got: {msg}"
                );
                assert!(msg.contains("config.json"), "msg = {msg}");
            }
            Err(other) => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn bicodec_stream_batch_yields_about_160ms_at_16khz() {
        // 8 semantic tokens × 320× temporal upsample / 16 000 Hz ≈
        // 0.16 s = 160 ms of audio per yielded chunk. Sanity check
        // the per-chunk-batch geometry — see the
        // `synthesize_pcm_stream` doc-comment for context.
        let upsample = 320_u32; // BiCodec.upsample_rates = [8,5,4,2]
        let chunk_samples = u32::try_from(BICODEC_STREAM_BATCH).unwrap() * upsample;
        let chunk_seconds = f64::from(chunk_samples) / f64::from(SPARK_TTS_SAMPLE_RATE_HZ);
        assert!(
            (chunk_seconds - 0.16).abs() < 1e-9,
            "expected ≈0.16 s per chunk; got {chunk_seconds}"
        );
        // Lock the batch constant against accidental churn — changing
        // it without re-justifying the seam-artefact / latency
        // trade-off documented on `synthesize_pcm_stream` would be a
        // surprise.
        assert_eq!(BICODEC_STREAM_BATCH, 8);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn receiver_into_stream_terminates_on_sender_drop() {
        use futures_util::StreamExt;
        let (tx, rx) = mpsc::channel::<u32>(4);
        drop(tx);
        let mut s = receiver_into_stream(rx);
        assert!(s.next().await.is_none(), "stream must end when tx drops");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn receiver_into_stream_forwards_items_in_order() {
        use futures_util::StreamExt;
        let (tx, rx) = mpsc::channel::<u32>(4);
        tx.send(10).await.unwrap();
        tx.send(20).await.unwrap();
        tx.send(30).await.unwrap();
        drop(tx);
        let collected: Vec<u32> = receiver_into_stream(rx).collect().await;
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn pcm_to_wav_emits_riff_wave_header_at_16khz_mono() {
        let pcm = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let wav = pcm_to_wav(&pcm, SPARK_TTS_SAMPLE_RATE_HZ, 1);
        // 44-byte header + 2 bytes per sample (16-bit PCM).
        assert_eq!(wav.len(), 44 + pcm.len() * 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // Channels (u16 LE at offset 22).
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1);
        // Sample rate (u32 LE at offset 24).
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            SPARK_TTS_SAMPLE_RATE_HZ
        );
    }

    /// End-to-end live test — gated behind `BLAZEN_TEST_SPARK_TTS=1`
    /// because it downloads ~1.6 GB of model bytes from HF Hub and runs
    /// a few seconds of inference. Verifies the full
    /// text → tokenizer → LLM → `BiCodec` → WAV pipeline shape (no
    /// silence assertion: a 32-token global mismatch from a freshly
    /// shipped checkpoint would otherwise wedge CI on a transient
    /// upstream change).
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_SPARK_TTS=1 + downloads ~1.6 GB of Spark-TTS bytes from HF Hub"]
    async fn synthesize_wav_end_to_end_produces_non_silent_audio() {
        use blazen_audio::AudioBackend;

        if std::env::var("BLAZEN_TEST_SPARK_TTS").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_SPARK_TTS != 1");
            return;
        }

        let backend = super::super::SparkTtsBackend::default();
        let audio = crate::traits::TtsBackend::synthesize(
            &backend,
            "Hello world",
            &crate::TtsOptions::default(),
        )
        .await
        .expect("end-to-end synthesize must succeed");
        // The 44-byte RIFF header + a non-trivial PCM payload.
        assert!(audio.bytes.len() > 44, "wav is just a header");
        assert_eq!(audio.sample_rate, SPARK_TTS_SAMPLE_RATE_HZ);
        assert_eq!(audio.channels, 1);
        // Non-silent assertion: at least one sample magnitude > 1/256 LSB.
        let mut nonzero = 0_usize;
        for chunk in audio.bytes[44..].chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            if sample.abs() > 128 {
                nonzero += 1;
            }
        }
        assert!(
            nonzero > 0,
            "WAV body must contain at least one non-silent sample"
        );
        // Side-channel sanity check on the backend id surface.
        assert!(backend.id().starts_with("spark-tts:"));
    }

    /// End-to-end streaming variant — gated behind
    /// `BLAZEN_TEST_SPARK_TTS=1`. Drives the streaming pipeline,
    /// collects chunks, and asserts (a) at least one chunk arrived,
    /// (b) exactly one chunk has `is_final = true`, and (c) the total
    /// concatenated sample count is within ±25% of the non-streamed
    /// `synthesize_pcm` length (the tail chunk's exact boundary
    /// depends on where the LLM finishes emitting semantic tokens).
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_SPARK_TTS=1 + downloads ~1.6 GB of Spark-TTS bytes from HF Hub"]
    async fn synthesize_pcm_stream_matches_offline_length() {
        use blazen_audio::AudioBackend;
        use futures_util::StreamExt;

        if std::env::var("BLAZEN_TEST_SPARK_TTS").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_SPARK_TTS != 1");
            return;
        }

        let backend = super::super::SparkTtsBackend::default();
        // Force the lazy loader to materialise so we can call the
        // pipeline directly.
        let _ = crate::traits::TtsBackend::synthesize(
            &backend,
            "warmup",
            &crate::TtsOptions::default(),
        )
        .await
        .expect("warmup synthesise");
        // Reach into the backend's lazy cell now that it's initialised.
        let pipe = backend
            .inner
            .get()
            .expect("pipeline initialised after warmup")
            .clone();

        let text = "Hello streaming world";
        let offline_len = pipe.synthesize_pcm(text).await.expect("offline pcm").len();

        let mut stream = pipe.synthesize_pcm_stream(text);
        let mut chunks: Vec<StreamingAudioChunk> = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.expect("chunk ok"));
        }
        assert!(!chunks.is_empty(), "stream must emit at least one chunk");
        let finals = chunks.iter().filter(|c| c.is_final).count();
        assert_eq!(finals, 1, "exactly one chunk must be marked final");
        let total: usize = chunks.iter().map(|c| c.samples.len()).sum();
        let lo = offline_len.saturating_sub(offline_len / 4);
        let hi = offline_len + offline_len / 4;
        assert!(
            (lo..=hi).contains(&total),
            "streaming total {total} should be within ±25% of offline {offline_len}"
        );
        assert!(backend.id().starts_with("spark-tts:"));
    }
}
