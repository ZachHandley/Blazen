//! Chunked candle Whisper decode loop with sliding-window KV-cache.
//!
//! Wraps the existing [`CandleWhisperBackend`] with a sliding-window
//! decoder that operates on overlapping 30 s audio chunks. Each chunk
//! is decoded independently by the underlying candle Whisper (Whisper
//! has a fixed 30 s context — there's no efficient KV-cache reuse
//! across chunks), and a [`LocalAgreement`] policy decides which words
//! at the chunk boundary are stable enough to "commit" as final vs.
//! which remain partial for the next chunk's overlap region.
//!
//! [`LocalAgreement`]: https://aclanthology.org/2020.aacl-srw.13/
//!
//! # Algorithm
//!
//! 1. Maintain a buffer of words from the *tail* of the previous chunk
//!    that fell inside the overlap region.
//! 2. After decoding the current chunk, split its prefix words (the
//!    region overlapping with the previous chunk) and compare against
//!    the tail buffer.
//! 3. Common words (longest common prefix at the word level) are
//!    committed as `final_text`; everything beyond is `partial_text`
//!    and will be reconsidered on the next chunk.
//! 4. On [`ChunkedWhisperDecoder::finalize`] the remaining partial is
//!    promoted to final and returned.
//!
//! The decoder is **not** Wave-A.4 itself — it exposes a `decode_chunk`
//! API that the pipeline wave (Wave A.4) will drive from a VAD-gated
//! frame stream. The candle backend is constructed lazily on the first
//! call so unit tests can exercise the `LocalAgreement` logic without
//! touching `HuggingFace`.

#![cfg(feature = "whisper-streaming")]

use std::sync::Arc;

use candle_core::Device;

use crate::backends::candle::{CandleWhisperBackend, CandleWhisperConfig, WhisperModel};
use crate::error::SttError;
use crate::traits::StreamingTranscript;

/// Whisper's fixed input sample rate. Re-exported for downstream
/// callers so they don't need to import `candle-transformers` to find it.
const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;

/// Static configuration for [`ChunkedWhisperDecoder`].
#[derive(Debug, Clone)]
pub struct ChunkedWhisperConfig {
    /// HF model repo, e.g. `"openai/whisper-base"`. The decoder maps
    /// the trailing path component back to a [`WhisperModel`].
    pub model_id: String,
    /// Chunk length in seconds. Whisper's native context is 30 s; we
    /// default to that.
    pub chunk_seconds: f32,
    /// Overlap between successive chunks in seconds. Default `5.0`.
    pub chunk_overlap_seconds: f32,
    /// Minimum number of consecutive words at the head of the current
    /// chunk that must match the tail of the previous chunk before
    /// they're promoted from partial to final. Default `2`.
    pub local_agreement_min_consensus: usize,
    /// Inference device (CPU/CUDA/Metal). Defaults to CPU.
    pub device: Device,
    /// Optional ISO 639-1 language hint forwarded to the underlying
    /// candle backend.
    pub language: Option<String>,
}

impl Default for ChunkedWhisperConfig {
    fn default() -> Self {
        Self {
            model_id: "openai/whisper-base".into(),
            chunk_seconds: 30.0,
            chunk_overlap_seconds: 5.0,
            local_agreement_min_consensus: 2,
            device: Device::Cpu,
            language: None,
        }
    }
}

/// One emission from [`ChunkedWhisperDecoder::decode_chunk`].
#[derive(Debug, Clone)]
pub struct DecodedChunk {
    /// Newly-committed final words for this chunk. These won't be
    /// revised by subsequent chunks.
    pub final_text: String,
    /// Currently-partial words at the tail of this chunk. May be
    /// revised on the next chunk.
    pub partial_text: String,
    /// Wall-clock latency for this chunk in seconds (from
    /// `decode_chunk` entry to return).
    pub latency_seconds: f32,
}

/// Streaming-aware wrapper around the candle Whisper backend.
///
/// Holds a lazily-constructed [`CandleWhisperBackend`] plus the
/// `LocalAgreement` word-level buffer used to reconcile overlapping
/// chunks.
pub struct ChunkedWhisperDecoder {
    config: ChunkedWhisperConfig,
    /// Underlying candle Whisper backend. `Arc` so the pipeline (Wave
    /// A.4) can share it with the file-based path if it ever needs to.
    backend: Arc<CandleWhisperBackend>,
    /// Words held over from the previous chunk's tail that haven't yet
    /// been committed. The next `decode_chunk` will compare its head
    /// against this buffer.
    pending_tail: Vec<String>,
}

impl std::fmt::Debug for ChunkedWhisperDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkedWhisperDecoder")
            .field("config", &self.config)
            .field("pending_tail", &self.pending_tail)
            .finish_non_exhaustive()
    }
}

impl ChunkedWhisperDecoder {
    /// Build a new decoder.
    ///
    /// Weights are **not** downloaded here — the underlying candle
    /// backend loads on first `decode_chunk` call.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::InvalidOptions`] when `chunk_seconds` is
    /// non-positive or `chunk_overlap_seconds >= chunk_seconds`, or
    /// when `model_id` doesn't map to a known [`WhisperModel`].
    pub fn new(config: ChunkedWhisperConfig) -> Result<Self, SttError> {
        if config.chunk_seconds <= 0.0 {
            return Err(SttError::InvalidOptions(format!(
                "chunk_seconds must be > 0, got {}",
                config.chunk_seconds
            )));
        }
        if config.chunk_overlap_seconds < 0.0
            || config.chunk_overlap_seconds >= config.chunk_seconds
        {
            return Err(SttError::InvalidOptions(format!(
                "chunk_overlap_seconds must satisfy 0 <= overlap < chunk; got overlap={} chunk={}",
                config.chunk_overlap_seconds, config.chunk_seconds
            )));
        }
        let model = parse_model_id(&config.model_id)?;

        let candle_cfg = CandleWhisperConfig {
            model,
            device: config.device.clone(),
            language: config.language.clone(),
            ..CandleWhisperConfig::default()
        };
        let backend = Arc::new(CandleWhisperBackend::new(candle_cfg));

        Ok(Self {
            config,
            backend,
            pending_tail: Vec::new(),
        })
    }

    /// Decode one audio chunk (16 kHz mono f32 in `[-1, 1]`).
    ///
    /// `chunk_index` is the zero-based position of this chunk in the
    /// stream. The first chunk (`chunk_index == 0`) skips the
    /// `LocalAgreement` merge because there is no prior tail to align
    /// against.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::InvalidOptions`] when `audio.len()` differs
    /// from the configured chunk length by more than 10%; otherwise
    /// any error from the underlying candle backend.
    pub async fn decode_chunk(
        &mut self,
        audio: &[f32],
        chunk_index: usize,
    ) -> Result<DecodedChunk, SttError> {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let expected_samples =
            (self.config.chunk_seconds * f32::from(WHISPER_SAMPLE_RATE_HZ as u16)) as usize;
        let tolerance = expected_samples / 10;
        if audio.len() + tolerance < expected_samples || audio.len() > expected_samples + tolerance
        {
            return Err(SttError::InvalidOptions(format!(
                "chunk audio length {} not within 10% of expected {} samples \
                 (chunk_seconds={}, sample_rate={WHISPER_SAMPLE_RATE_HZ})",
                audio.len(),
                expected_samples,
                self.config.chunk_seconds
            )));
        }

        let start = std::time::Instant::now();
        let result = self
            .backend
            .transcribe_inherent(audio, WHISPER_SAMPLE_RATE_HZ)
            .await?;
        #[allow(clippy::cast_precision_loss)]
        let latency_seconds = start.elapsed().as_secs_f32();

        let chunk_words: Vec<String> = split_words(&result.text);

        let (final_text, new_pending) = if chunk_index == 0 {
            // No prior tail — everything becomes pending; nothing
            // commits yet (we don't know which words are stable).
            (String::new(), chunk_words)
        } else {
            self.local_agreement_merge(&chunk_words)
        };

        let partial_text = join_words(&new_pending);
        self.pending_tail = new_pending;

        Ok(DecodedChunk {
            final_text,
            partial_text,
            latency_seconds,
        })
    }

    /// Promote any pending partial text to final at end-of-stream.
    ///
    /// Returns a [`StreamingTranscript`] flagged as final containing
    /// whatever words were still hovering in the partial buffer. After
    /// this call the decoder is empty and may be reused for a new
    /// stream.
    pub fn finalize(&mut self) -> StreamingTranscript {
        let text = join_words(&self.pending_tail);
        self.pending_tail.clear();
        StreamingTranscript {
            text,
            is_final: true,
            confidence: None,
            latency_seconds: None,
        }
    }

    /// Borrow the decoder's configuration.
    #[must_use]
    pub const fn config(&self) -> &ChunkedWhisperConfig {
        &self.config
    }

    /// Underlying candle backend handle (Arc-shared). Pipeline code
    /// (Wave A.4) uses this to call `load()` eagerly if it wants to
    /// front-load the HF download.
    #[must_use]
    pub fn backend(&self) -> Arc<CandleWhisperBackend> {
        Arc::clone(&self.backend)
    }

    /// `LocalAgreement` merge of the current chunk's words against the
    /// pending tail buffer. Returns `(committed_text, new_pending)`.
    ///
    /// The committed prefix is the longest run of words at the head of
    /// `chunk_words` that matches the *tail* of `self.pending_tail`,
    /// provided the run is at least
    /// `local_agreement_min_consensus` long. The remaining tail of
    /// `chunk_words` becomes the new pending buffer.
    fn local_agreement_merge(&self, chunk_words: &[String]) -> (String, Vec<String>) {
        let consensus = longest_word_match(&self.pending_tail, chunk_words);
        let min = self.config.local_agreement_min_consensus.max(1);

        if consensus < min {
            // Not enough agreement — keep prior pending as still
            // partial, but slide it forward by setting it equal to
            // this chunk's words (the new "best guess").
            return (String::new(), chunk_words.to_vec());
        }

        // Commit the consensus prefix; everything after stays partial.
        let committed = chunk_words[..consensus].join(" ");
        let new_pending = chunk_words[consensus..].to_vec();
        (committed, new_pending)
    }
}

/// Map an `openai/whisper-{size}` model id back to a
/// [`WhisperModel`]. Returns [`SttError::InvalidOptions`] for unknown
/// ids.
fn parse_model_id(model_id: &str) -> Result<WhisperModel, SttError> {
    let size = model_id.strip_prefix("openai/whisper-").ok_or_else(|| {
        SttError::InvalidOptions(format!(
            "model_id must start with `openai/whisper-`, got `{model_id}`"
        ))
    })?;
    match size {
        "tiny" => Ok(WhisperModel::Tiny),
        "base" => Ok(WhisperModel::Base),
        "small" => Ok(WhisperModel::Small),
        "medium" => Ok(WhisperModel::Medium),
        "large-v3" => Ok(WhisperModel::LargeV3),
        "large-v3-turbo" => Ok(WhisperModel::LargeV3Turbo),
        other => Err(SttError::InvalidOptions(format!(
            "unknown whisper size `{other}` (expected tiny/base/small/medium/large-v3/large-v3-turbo)"
        ))),
    }
}

/// Whitespace-split into normalised lowercase tokens. Punctuation is
/// kept attached to its word — the `LocalAgreement` comparison treats
/// `hello.` and `hello` as different on purpose: a punctuation flip
/// usually means the model is still uncertain.
fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect()
}

fn join_words(words: &[String]) -> String {
    words.join(" ")
}

/// Compute the longest run of consecutive words at the *head* of
/// `head` that matches at the *tail* of `tail`. Conceptually: given
/// `tail = ["a", "b", "c"]` and `head = ["b", "c", "d", "e"]`, the
/// answer is 2 (the `["b", "c"]` overlap).
///
/// We search by trying decreasing match lengths from
/// `min(tail.len(), head.len())` down to 1.
fn longest_word_match(tail: &[String], head: &[String]) -> usize {
    let max_match = tail.len().min(head.len());
    for n in (1..=max_match).rev() {
        if tail[tail.len() - n..] == head[..n] {
            return n;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ChunkedWhisperConfig {
        ChunkedWhisperConfig::default()
    }

    #[test]
    fn default_config_matches_spec() {
        let c = cfg();
        assert_eq!(c.model_id, "openai/whisper-base");
        assert!((c.chunk_seconds - 30.0).abs() < f32::EPSILON);
        assert!((c.chunk_overlap_seconds - 5.0).abs() < f32::EPSILON);
        assert_eq!(c.local_agreement_min_consensus, 2);
        assert!(matches!(c.device, Device::Cpu));
        assert!(c.language.is_none());
    }

    #[test]
    fn parse_model_id_round_trip() {
        assert_eq!(
            parse_model_id("openai/whisper-tiny").unwrap(),
            WhisperModel::Tiny
        );
        assert_eq!(
            parse_model_id("openai/whisper-base").unwrap(),
            WhisperModel::Base
        );
        assert_eq!(
            parse_model_id("openai/whisper-large-v3-turbo").unwrap(),
            WhisperModel::LargeV3Turbo
        );
    }

    #[test]
    fn parse_model_id_rejects_unknown_repo_prefix() {
        let err = parse_model_id("custom/whisper-base").unwrap_err();
        assert!(matches!(err, SttError::InvalidOptions(_)));
    }

    #[test]
    fn parse_model_id_rejects_unknown_size() {
        let err = parse_model_id("openai/whisper-huge").unwrap_err();
        assert!(matches!(err, SttError::InvalidOptions(_)));
    }

    #[test]
    fn constructor_rejects_zero_chunk() {
        let bad = ChunkedWhisperConfig {
            chunk_seconds: 0.0,
            ..cfg()
        };
        let err = ChunkedWhisperDecoder::new(bad).unwrap_err();
        assert!(matches!(err, SttError::InvalidOptions(_)));
    }

    #[test]
    fn constructor_rejects_overlap_exceeding_chunk() {
        let bad = ChunkedWhisperConfig {
            chunk_seconds: 10.0,
            chunk_overlap_seconds: 11.0,
            ..cfg()
        };
        let err = ChunkedWhisperDecoder::new(bad).unwrap_err();
        assert!(matches!(err, SttError::InvalidOptions(_)));
    }

    #[test]
    fn constructor_accepts_overlap_equal_to_chunk_minus_epsilon() {
        let good = ChunkedWhisperConfig {
            chunk_seconds: 10.0,
            chunk_overlap_seconds: 9.0,
            ..cfg()
        };
        ChunkedWhisperDecoder::new(good).expect("valid config");
    }

    #[test]
    fn split_words_lowercases_and_keeps_punctuation() {
        assert_eq!(split_words("Hello WORLD."), vec!["hello", "world."]);
    }

    #[test]
    fn longest_word_match_finds_full_overlap() {
        let tail = vec!["a".into(), "b".into(), "c".into()];
        let head = vec!["b".into(), "c".into(), "d".into()];
        assert_eq!(longest_word_match(&tail, &head), 2);
    }

    #[test]
    fn longest_word_match_returns_zero_when_no_overlap() {
        let tail = vec!["a".into(), "b".into()];
        let head = vec!["x".into(), "y".into()];
        assert_eq!(longest_word_match(&tail, &head), 0);
    }

    #[test]
    fn longest_word_match_prefers_longer_overlap_when_multiple_match() {
        // tail "a b a b", head "a b a b ..." — the longest overlap that
        // ends at tail and starts at head is the whole length.
        let tail = vec!["a".into(), "b".into(), "a".into(), "b".into()];
        let head = vec!["a".into(), "b".into(), "a".into(), "b".into(), "c".into()];
        assert_eq!(longest_word_match(&tail, &head), 4);
    }

    #[test]
    fn local_agreement_commits_consensus_prefix() {
        let mut dec = ChunkedWhisperDecoder::new(cfg()).expect("ctor");
        dec.pending_tail = vec!["the".into(), "quick".into(), "brown".into()];
        let chunk_words = vec!["quick".into(), "brown".into(), "fox".into(), "jumps".into()];
        let (committed, new_pending) = dec.local_agreement_merge(&chunk_words);
        assert_eq!(committed, "quick brown");
        assert_eq!(new_pending, vec!["fox".to_string(), "jumps".to_string()]);
    }

    #[test]
    fn local_agreement_skips_commit_below_min_consensus() {
        let mut dec = ChunkedWhisperDecoder::new(ChunkedWhisperConfig {
            local_agreement_min_consensus: 3,
            ..cfg()
        })
        .expect("ctor");
        dec.pending_tail = vec!["the".into(), "quick".into(), "brown".into()];
        // Only 2 words of agreement; min_consensus = 3 → commit nothing.
        let chunk_words = vec!["quick".into(), "brown".into(), "fox".into()];
        let (committed, new_pending) = dec.local_agreement_merge(&chunk_words);
        assert!(committed.is_empty());
        assert_eq!(new_pending, chunk_words);
    }

    #[test]
    fn finalize_promotes_pending_to_final() {
        let mut dec = ChunkedWhisperDecoder::new(cfg()).expect("ctor");
        dec.pending_tail = vec!["foo".into(), "bar".into()];
        let final_emission = dec.finalize();
        assert!(final_emission.is_final);
        assert_eq!(final_emission.text, "foo bar");
        assert!(dec.pending_tail.is_empty());
        // Confidence and latency are intentionally None — the decoder
        // has no information about either at end-of-stream.
        assert!(final_emission.confidence.is_none());
        assert!(final_emission.latency_seconds.is_none());
    }

    #[test]
    fn finalize_on_empty_decoder_returns_empty_final() {
        let mut dec = ChunkedWhisperDecoder::new(cfg()).expect("ctor");
        let e = dec.finalize();
        assert!(e.is_final);
        assert_eq!(e.text, "");
    }

    #[test]
    fn backend_accessor_returns_arc_to_inner() {
        let dec = ChunkedWhisperDecoder::new(cfg()).expect("ctor");
        let a = dec.backend();
        let b = dec.backend();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn decode_chunk_rejects_wildly_wrong_length() {
        let mut dec = ChunkedWhisperDecoder::new(cfg()).expect("ctor");
        // 1 second of audio when 30 seconds were expected — way out of
        // tolerance, so we should reject before touching the backend.
        let audio = vec![0.0_f32; 16_000];
        let err = dec.decode_chunk(&audio, 0).await.unwrap_err();
        assert!(matches!(err, SttError::InvalidOptions(_)));
    }
}
