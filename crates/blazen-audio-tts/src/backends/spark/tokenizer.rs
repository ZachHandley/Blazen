//! Spark-TTS text frontend: prompt construction + tokenizer wrapper +
//! generated-output parser.
//!
//! Upstream reference: `SparkAudio/Spark-TTS` —
//! `cli/SparkTTS.py::process_prompt` (voice cloning) and the
//! `runtime/triton_trtllm/model_repo/spark_tts/1/model.py` mirror.
//!
//! ## Prompt format
//!
//! The LLM (Qwen2.5-0.5B with Spark's custom added tokens) consumes a
//! plain-text string assembled from special markers:
//!
//! ```text
//! <|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>
//! ```
//!
//! …for pure text-to-speech with no reference audio (the LLM is free to
//! emit its own global tokens). For voice cloning we additionally inject
//! pre-computed `<|bicodec_global_K|>` tokens between the global-start
//! and global-end markers, then append `<|start_semantic_token|>` so the
//! decoder can roll out the semantic stream. Matches upstream
//! `process_prompt` (`cli/SparkTTS.py:78-104`).
//!
//! ## Generation parsing
//!
//! The LLM emits a mix of natural-language tokens (which we ignore) and
//! `<|bicodec_semantic_K|>` / `<|bicodec_global_K|>` markers. Upstream
//! extracts them with two regexes — see `cli/SparkTTS.py:217` and
//! `cli/SparkTTS.py:224`:
//!
//! ```python
//! re.findall(r"bicodec_semantic_(\d+)", predicts)
//! re.findall(r"bicodec_global_(\d+)", predicts)
//! ```
//!
//! We mirror that exactly here using the `regex` crate, so the integer
//! indices we return are byte-for-byte compatible with what `BiCodec`'s
//! `detokenize` expects.

#![cfg(feature = "spark-tts")]
#![allow(
    dead_code,
    reason = "Wave S.2.3 lands the tokenizer + prompt-construction surface; \
              the pipeline (S.2.4) that consumes it ships in a follow-up \
              wave. Until then the symbols are exercised only by the \
              in-module unit tests."
)]

use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use tokenizers::Tokenizer;

/// `<|task_tts|>` — task selector from `sparktts/utils/token_parser.py::TASK_TOKEN_MAP`.
pub(super) const TASK_TTS: &str = "<|task_tts|>";
/// `<|start_content|>` — opens the text-content section of the prompt.
pub(super) const START_CONTENT: &str = "<|start_content|>";
/// `<|end_content|>` — closes the text-content section.
pub(super) const END_CONTENT: &str = "<|end_content|>";
/// `<|start_global_token|>` — opens the global-token (speaker style) section.
pub(super) const START_GLOBAL: &str = "<|start_global_token|>";
/// `<|end_global_token|>` — closes the global-token section.
pub(super) const END_GLOBAL: &str = "<|end_global_token|>";
/// `<|start_semantic_token|>` — opens the semantic-token (content) section.
pub(super) const START_SEMANTIC: &str = "<|start_semantic_token|>";

fn semantic_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"bicodec_semantic_(\d+)").expect("static regex compiles"))
}

fn global_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"bicodec_global_(\d+)").expect("static regex compiles"))
}

/// Errors surfaced by [`SparkTokenizer`] load / encode / parse paths.
#[derive(thiserror::Error, Debug)]
pub(super) enum TokenizerError {
    /// `tokenizer.json` could not be read or parsed (missing file,
    /// truncated payload, schema mismatch, …).
    #[error("tokenizer load failed: {0}")]
    Load(String),
    /// `Tokenizer::encode` (or decode) returned an error.
    #[error("tokenization failed: {0}")]
    Tokenize(String),
    /// The LLM's generated output contained zero `<|bicodec_semantic_K|>`
    /// markers — without semantic indices `BiCodec` cannot synthesise any
    /// waveform, so the caller should retry / surface a synthesis error.
    #[error("no semantic tokens emitted by the LLM")]
    NoSemanticTokens,
    /// `Tokenizer::decode` returned a string we could not interpret.
    #[error("invalid generation output: {0}")]
    InvalidGeneration(String),
}

/// Spark-TTS text tokenizer.
///
/// Wraps a `HuggingFace` [`tokenizers::Tokenizer`] loaded from the
/// `LLM/tokenizer.json` shipped inside the `SparkAudio/Spark-TTS-0.5B`
/// HF repository. The tokenizer carries the full Qwen2.5 vocab plus
/// Spark's custom special tokens (task selectors, content delimiters,
/// `<|bicodec_semantic_K|>`, `<|bicodec_global_K|>`, etc.).
#[derive(Debug)]
pub(super) struct SparkTokenizer {
    inner: Tokenizer,
}

impl SparkTokenizer {
    /// Load `LLM/tokenizer.json` from a Spark-TTS bundle directory.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::Load`] when the file is missing,
    /// unreadable, or fails the `tokenizers` crate's JSON validation.
    pub(super) fn load(tokenizer_path: &Path) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| TokenizerError::Load(format!("{}: {e}", tokenizer_path.display())))?;
        Ok(Self { inner })
    }

    /// Build a pure text-to-speech prompt (no voice cloning, no
    /// controllable attributes).
    ///
    /// The returned token id sequence ends with `<|start_global_token|>`
    /// so the LLM is free to emit its own `<|bicodec_global_K|>` tokens
    /// before transitioning to the semantic stream. This corresponds to
    /// the "no reference audio" code path that voice-cloning callers
    /// would otherwise feed pre-computed global tokens into.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::Tokenize`] when the underlying
    /// [`tokenizers::Tokenizer`] cannot encode the assembled prompt.
    pub(super) fn build_tts_prompt(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let prompt_text = format!("{TASK_TTS}{START_CONTENT}{text}{END_CONTENT}{START_GLOBAL}");
        self.encode(&prompt_text)
    }

    /// Build a voice-cloning prompt: text + pre-computed global tokens
    /// from `BiCodec::tokenize(reference_audio)`.
    ///
    /// Matches the `prompt_text is None` branch of upstream
    /// `process_prompt` (`cli/SparkTTS.py:95-104`): the global tokens
    /// are interpolated between `<|start_global_token|>` and
    /// `<|end_global_token|>`, then the prompt is closed with
    /// `<|start_semantic_token|>` so the LLM can roll out the semantic
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::Tokenize`] when the underlying
    /// [`tokenizers::Tokenizer`] cannot encode the assembled prompt.
    pub(super) fn build_clone_prompt(
        &self,
        text: &str,
        global_token_ids: &[u32],
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut global_str = String::with_capacity(global_token_ids.len() * 24);
        for g in global_token_ids {
            // Matches upstream: f"<|bicodec_global_{i}|>".
            global_str.push_str("<|bicodec_global_");
            // u32::to_string allocates a tiny scratch buffer; this avoids
            // the format! macro overhead in the inner loop.
            global_str.push_str(&g.to_string());
            global_str.push_str("|>");
        }
        let prompt_text = format!(
            "{TASK_TTS}{START_CONTENT}{text}{END_CONTENT}\
             {START_GLOBAL}{global_str}{END_GLOBAL}{START_SEMANTIC}",
        );
        self.encode(&prompt_text)
    }

    /// Parse the LLM's generated token ids back into `BiCodec` semantic +
    /// global indices.
    ///
    /// Decodes `generated_ids` to a text string via the underlying
    /// [`tokenizers::Tokenizer`] (with `skip_special_tokens=false`, so
    /// the `<|bicodec_*|>` markers survive), then runs the same two
    /// regexes upstream uses (`cli/SparkTTS.py:217,224`) to harvest the
    /// integer indices.
    ///
    /// Returns `(semantic_indices, global_indices)` where each vector
    /// contains the codebook entries to feed into
    /// `BiCodec::detokenize`. `global_indices` may be empty when the
    /// caller supplied pre-computed global tokens via
    /// [`Self::build_clone_prompt`] (the LLM only emits new global
    /// tokens in the no-reference path).
    ///
    /// # Errors
    ///
    /// * [`TokenizerError::InvalidGeneration`] when
    ///   [`tokenizers::Tokenizer::decode`] fails.
    /// * [`TokenizerError::NoSemanticTokens`] when the decoded string
    ///   contains zero `<|bicodec_semantic_K|>` markers — `BiCodec` needs
    ///   at least one semantic index per output frame, so an empty
    ///   stream means the LLM produced no usable audio.
    pub(super) fn parse_generation(
        &self,
        generated_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), TokenizerError> {
        let decoded = self
            .inner
            .decode(generated_ids, false)
            .map_err(|e| TokenizerError::InvalidGeneration(e.to_string()))?;
        parse_generation_str(&decoded)
    }

    fn encode(&self, prompt_text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(prompt_text, false)
            .map_err(|e| TokenizerError::Tokenize(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}

/// Pure-function variant of [`SparkTokenizer::parse_generation`] that
/// operates on an already-decoded string. Split out so unit tests can
/// exercise the regex/extraction logic without loading a real
/// `tokenizer.json`.
pub(super) fn parse_generation_str(decoded: &str) -> Result<(Vec<u32>, Vec<u32>), TokenizerError> {
    let semantic: Vec<u32> = semantic_regex()
        .captures_iter(decoded)
        .filter_map(|c| c.get(1).and_then(|m| m.as_str().parse::<u32>().ok()))
        .collect();
    let global: Vec<u32> = global_regex()
        .captures_iter(decoded)
        .filter_map(|c| c.get(1).and_then(|m| m.as_str().parse::<u32>().ok()))
        .collect();
    if semantic.is_empty() {
        return Err(TokenizerError::NoSemanticTokens);
    }
    Ok((semantic, global))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn tokenizer_load_returns_error_for_missing_file() {
        let nowhere = PathBuf::from("/nonexistent/spark-tts/tokenizer.json");
        let err = SparkTokenizer::load(&nowhere).expect_err("missing file must error");
        match err {
            TokenizerError::Load(msg) => {
                assert!(
                    msg.contains("/nonexistent/spark-tts/tokenizer.json"),
                    "expected path in error, got: {msg}"
                );
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn parse_generation_extracts_semantic_and_global_indices() {
        let synthetic = "<|start_global_token|>\
                         <|bicodec_global_42|><|bicodec_global_7|>\
                         <|end_global_token|>\
                         <|start_semantic_token|>\
                         <|bicodec_semantic_100|><|bicodec_semantic_200|><|bicodec_semantic_300|>\
                         <|im_end|>";
        let (semantic, global) =
            parse_generation_str(synthetic).expect("synthetic stream must parse");
        assert_eq!(semantic, vec![100, 200, 300]);
        assert_eq!(global, vec![42, 7]);
    }

    #[test]
    fn parse_generation_returns_no_semantic_tokens_error_for_empty_output() {
        let empty = "<|start_global_token|><|bicodec_global_5|><|end_global_token|>";
        let err = parse_generation_str(empty).expect_err("missing semantic markers must error");
        assert!(matches!(err, TokenizerError::NoSemanticTokens));
    }

    #[test]
    fn parse_generation_ignores_natural_text_between_markers() {
        // The LLM occasionally interleaves human-readable text amongst
        // the bicodec markers (the Qwen2.5 backbone is a general LM).
        // The upstream regex skips that text and grabs just the integer
        // suffixes; we must do the same.
        let noisy = "hello world <|bicodec_semantic_11|> filler \
                     <|bicodec_semantic_22|> end <|bicodec_global_3|>";
        let (semantic, global) =
            parse_generation_str(noisy).expect("noisy stream must still parse");
        assert_eq!(semantic, vec![11, 22]);
        assert_eq!(global, vec![3]);
    }

    #[test]
    fn parse_generation_skips_malformed_suffixes() {
        // u32-overflowing suffixes are dropped silently (mirrors
        // upstream's `int(token)` which would raise — we prefer to keep
        // the well-formed tokens we did see, so the caller still gets
        // *some* audio out).
        let mixed = "<|bicodec_semantic_99999999999999999999|>\
                     <|bicodec_semantic_5|>";
        let (semantic, _) = parse_generation_str(mixed).expect("at least one well-formed");
        assert_eq!(semantic, vec![5]);
    }

    #[test]
    #[ignore = "requires LLM/tokenizer.json from SparkAudio/Spark-TTS-0.5B \
                (gate via BLAZEN_TEST_SPARK_TTS=1 + cached bundle)"]
    fn build_tts_prompt_constructs_expected_token_sequence() {
        // Live-asset test: locate a cached tokenizer.json under
        // ~/.cache/spark-tts-research/Spark-TTS-0.5B/LLM/tokenizer.json
        // (or wherever the model-cache pulled it), load it, encode a
        // "Hello world" prompt, and assert the first token is
        // <|task_tts|> and that <|start_content|> appears in the output.
        let path = std::env::var("SPARK_TTS_TOKENIZER_JSON")
            .map(PathBuf::from)
            .expect("set SPARK_TTS_TOKENIZER_JSON to tokenizer.json path");
        let tok = SparkTokenizer::load(&path).expect("real tokenizer.json must load");
        let ids = tok
            .build_tts_prompt("Hello world")
            .expect("real tokenizer must encode the prompt");
        assert!(!ids.is_empty(), "encoded prompt must be non-empty");
        // Round-trip through decode (special tokens preserved) and
        // verify the prompt skeleton survives.
        let decoded = tok.inner.decode(&ids, false).expect("decode must succeed");
        assert!(decoded.starts_with(TASK_TTS), "decoded = {decoded}");
        assert!(decoded.contains(START_CONTENT), "decoded = {decoded}");
        assert!(decoded.contains(END_CONTENT), "decoded = {decoded}");
        assert!(decoded.contains(START_GLOBAL), "decoded = {decoded}");
    }

    #[test]
    #[ignore = "requires LLM/tokenizer.json from SparkAudio/Spark-TTS-0.5B \
                (gate via BLAZEN_TEST_SPARK_TTS=1 + cached bundle)"]
    fn build_clone_prompt_inlines_bicodec_global_tokens() {
        let path = std::env::var("SPARK_TTS_TOKENIZER_JSON")
            .map(PathBuf::from)
            .expect("set SPARK_TTS_TOKENIZER_JSON to tokenizer.json path");
        let tok = SparkTokenizer::load(&path).expect("real tokenizer.json must load");
        let ids = tok
            .build_clone_prompt("Hi.", &[12, 34, 56])
            .expect("encode must succeed");
        let decoded = tok.inner.decode(&ids, false).expect("decode must succeed");
        assert!(
            decoded.contains("<|bicodec_global_12|>"),
            "decoded = {decoded}"
        );
        assert!(
            decoded.contains("<|bicodec_global_34|>"),
            "decoded = {decoded}"
        );
        assert!(
            decoded.contains("<|bicodec_global_56|>"),
            "decoded = {decoded}"
        );
        assert!(decoded.contains(START_SEMANTIC), "decoded = {decoded}");
    }
}
