//! F5-TTS character-level text tokenizer.
//!
//! F5-TTS uses a plain character-level tokenizer (no BPE, no
//! subword). The vocab file (`F5TTS_Base/vocab.txt` on
//! `SWivid/F5-TTS`) ships as a UTF-8 text file with one character per
//! line. Line index → token id, with **space at index 0** doubling as
//! the out-of-vocabulary fallback.
//!
//! Upstream reference:
//! `SWivid/F5-TTS/src/f5_tts/model/utils.py::list_str_to_idx` and
//! `SWivid/F5-TTS/src/f5_tts/infer/utils_infer.py::load_vocab`.
//!
//! Key parity points with upstream:
//!
//! - **No lowercasing.** Upstream feeds the raw character through
//!   `vocab_char_map.get(c, 0)`; casing is preserved verbatim.
//! - **Vocab is parsed line-by-line.** Each line's trailing newline is
//!   stripped (`char[:-1]` in upstream) and the surviving content is
//!   treated as a single token. Empty lines yield empty-string tokens
//!   (legitimate when the file encodes a literal `'\n'` token via a
//!   blank line followed by the next token).
//! - **OOV characters map to index 0** (which is asserted to be the
//!   space token by upstream `assert vocab_char_map[" "] == 0`).
//! - **PAD semantics.** Upstream pads tensors with the sentinel value
//!   `-1` outside the vocab; inside the vocab the only "fallback" token
//!   is the index-0 space. We surface [`F5Tokenizer::pad_token`] as
//!   `0` (the space token) so callers that need an in-vocab pad id can
//!   reuse the same fallback the encoder uses for OOV. Callers that
//!   want the true `-1` mask sentinel should use a separate mask
//!   tensor, mirroring upstream `lens_to_mask`.

#![cfg(feature = "f5-tts")]

use std::collections::HashMap;
use std::path::Path;

use crate::error::TtsError;

/// F5-TTS character-level tokenizer.
///
/// Construct via [`F5Tokenizer::from_vocab_path`] (file on disk) or
/// [`F5Tokenizer::from_vocab_str`] (already-read UTF-8 contents). The
/// type is `Send + Sync` so it can be stored in the shared pipeline
/// state and reused across synthesis requests.
#[derive(Debug, Clone)]
pub struct F5Tokenizer {
    char_to_idx: HashMap<String, u32>,
    idx_to_char: Vec<String>,
    pad_token: u32,
}

impl F5Tokenizer {
    /// Build the tokenizer by reading `vocab.txt` from disk.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when the file cannot be read or
    /// when the vocab fails the upstream invariant (`vocab[0] == " "`).
    pub fn from_vocab_path(path: &Path) -> Result<Self, TtsError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            TtsError::ModelLoad(format!(
                "f5-tts vocab read failed for {}: {e}",
                path.display()
            ))
        })?;
        Self::from_vocab_str(&contents)
    }

    /// Build the tokenizer from an already-read UTF-8 vocab string.
    ///
    /// Splits on `\n`, strips a single trailing `\r` per line (Windows
    /// line endings), and assigns sequential indices. The final
    /// trailing newline produces no extra empty token.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when the vocab is empty or when
    /// the upstream invariant `vocab[0] == " "` is violated. We treat
    /// the latter as a hard error rather than silently accepting a
    /// non-space token at index 0 because OOV characters would map to
    /// that token and silently corrupt synthesis.
    pub fn from_vocab_str(contents: &str) -> Result<Self, TtsError> {
        // Split on '\n', then trim a single trailing '\r' per line to
        // handle CRLF-checked-out vocab files. Mirrors upstream's
        // `char[:-1]` which only ever strips one trailing '\n'.
        let mut idx_to_char: Vec<String> = contents
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line).to_owned())
            .collect();
        // A trailing newline in the file (the common case) leaves an
        // empty string at the end of the split. Drop it so the vocab
        // size matches `wc -l < vocab.txt`. We only pop ONE trailing
        // empty entry — interior blank lines (which legitimately
        // encode the literal newline character as a vocab token) are
        // preserved.
        if idx_to_char.last().is_some_and(String::is_empty) {
            idx_to_char.pop();
        }

        if idx_to_char.is_empty() {
            return Err(TtsError::ModelLoad(
                "f5-tts vocab.txt is empty (expected at least the space token at index 0)"
                    .to_owned(),
            ));
        }

        // Upstream invariant: `assert vocab_char_map[" "] == 0`.
        // Treat violation as a fatal load error.
        if idx_to_char[0] != " " {
            return Err(TtsError::ModelLoad(format!(
                "f5-tts vocab.txt invariant violated: expected index 0 to be a single space, \
                 got {:?}",
                idx_to_char[0]
            )));
        }

        let mut char_to_idx: HashMap<String, u32> = HashMap::with_capacity(idx_to_char.len());
        for (i, tok) in idx_to_char.iter().enumerate() {
            // Last-write-wins on duplicates, matching Python dict
            // semantics — but we log via tracing so a malformed vocab
            // surfaces in the trace stream.
            #[allow(clippy::cast_possible_truncation)]
            let id = i as u32;
            if let Some(prev) = char_to_idx.insert(tok.clone(), id) {
                tracing::debug!(
                    token = %tok,
                    previous_id = prev,
                    new_id = id,
                    "f5-tts vocab.txt has a duplicate token; later index wins",
                );
            }
        }

        Ok(Self {
            char_to_idx,
            idx_to_char,
            pad_token: 0,
        })
    }

    /// Encode a string into a sequence of token ids.
    ///
    /// Iterates over the input's **Unicode scalar values** (one char
    /// per token), looks each up in the vocab, and falls back to the
    /// pad/space token (index 0) for any character not in the vocab.
    /// The returned vector has the same length as `text.chars().count()`.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| {
                // Allocate a 4-byte stack-friendly buffer for the
                // char-to-string conversion. We could instead key the
                // map on `char`, but the vocab's tokens are
                // string-shaped (some upstream vocabs include
                // multi-codepoint IPA-phoneme tokens, e.g.
                // "tˈ"), so the string-keyed map is the
                // safer-by-default representation.
                let mut buf = [0u8; 4];
                let key = c.encode_utf8(&mut buf);
                self.char_to_idx.get(key).copied().unwrap_or(self.pad_token)
            })
            .collect()
    }

    /// Encode a batch of strings, returning per-sequence token vectors
    /// and the maximum length (useful for caller-side padding).
    #[must_use]
    pub fn encode_batch(&self, texts: &[&str]) -> (Vec<Vec<u32>>, usize) {
        let encoded: Vec<Vec<u32>> = texts.iter().map(|t| self.encode(t)).collect();
        let max_len = encoded.iter().map(Vec::len).max().unwrap_or(0);
        (encoded, max_len)
    }

    /// Vocabulary size — the number of distinct tokens loaded from
    /// `vocab.txt`. For the canonical `F5TTS_Base` checkpoint this is
    /// 2545 (matches `text_num_embeds: 2545` in upstream config).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.idx_to_char.len()
    }

    /// The pad/OOV token id (always `0` — the space token, per the
    /// upstream invariant).
    #[must_use]
    pub fn pad_token(&self) -> u32 {
        self.pad_token
    }

    /// Look up a token by id (debugging / introspection). Returns
    /// `None` for ids beyond the vocab size.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.idx_to_char.get(id as usize).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, ASCII-only vocab fixture suitable for unit tests. The
    /// first entry MUST be a literal space to satisfy the upstream
    /// invariant. Layout: space, lowercase a-z (1..=26), digit 0
    /// (27), period (28).
    fn fixture_vocab() -> String {
        let mut s = String::from(" \n");
        for c in 'a'..='z' {
            s.push(c);
            s.push('\n');
        }
        s.push_str("0\n");
        s.push_str(".\n");
        s
    }

    #[test]
    fn tokenizer_loads_from_simple_vocab() {
        let tok = F5Tokenizer::from_vocab_str(&fixture_vocab()).expect("load vocab fixture");
        // 1 (space) + 26 (a-z) + 1 (0) + 1 (.) = 29.
        assert_eq!(tok.vocab_size(), 29);
        assert_eq!(tok.pad_token(), 0);
        assert_eq!(tok.id_to_token(0), Some(" "));
        assert_eq!(tok.id_to_token(1), Some("a"));
        assert_eq!(tok.id_to_token(26), Some("z"));
        assert_eq!(tok.id_to_token(27), Some("0"));
        assert_eq!(tok.id_to_token(28), Some("."));
        assert_eq!(tok.id_to_token(29), None);
    }

    #[test]
    fn encode_handles_unknown_chars() {
        let tok = F5Tokenizer::from_vocab_str(&fixture_vocab()).expect("load vocab fixture");
        // 'A' (uppercase — not in fixture, since upstream is
        // case-sensitive and the fixture only contains lowercase) and
        // '!' should both fall back to the pad/space token (0).
        let ids = tok.encode("Aa!.");
        assert_eq!(ids, vec![0, 1, 0, 28]);
    }

    #[test]
    fn encode_simple_sentence_correct_length() {
        let tok = F5Tokenizer::from_vocab_str(&fixture_vocab()).expect("load vocab fixture");
        let text = "hello world.";
        let ids = tok.encode(text);
        // Encoded length must equal `chars().count()` (Unicode scalar
        // values), NOT `len()` (bytes). For ASCII these match.
        assert_eq!(ids.len(), text.chars().count());
        // 'h'=8, 'e'=5, 'l'=12, 'l'=12, 'o'=15, ' '=0, 'w'=23, 'o'=15,
        // 'r'=18, 'l'=12, 'd'=4, '.'=28.
        assert_eq!(ids, vec![8, 5, 12, 12, 15, 0, 23, 15, 18, 12, 4, 28]);
    }

    #[test]
    fn from_vocab_str_rejects_non_space_at_index_zero() {
        // Upstream `assert vocab_char_map[" "] == 0` — index 0 MUST
        // be the space token. Anything else is a hard load error.
        let bad = "a\n \nb\n";
        let err =
            F5Tokenizer::from_vocab_str(bad).expect_err("non-space at index 0 must be rejected");
        match err {
            TtsError::ModelLoad(msg) => {
                assert!(msg.contains("index 0"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_vocab_str_rejects_empty_file() {
        let err = F5Tokenizer::from_vocab_str("").expect_err("empty vocab must be rejected");
        match err {
            TtsError::ModelLoad(msg) => assert!(msg.contains("empty"), "msg = {msg}"),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn encode_batch_returns_max_len() {
        let tok = F5Tokenizer::from_vocab_str(&fixture_vocab()).expect("load vocab fixture");
        let texts = ["hi", "hello", "h"];
        let (encoded, max_len) = tok.encode_batch(&texts);
        assert_eq!(encoded.len(), 3);
        assert_eq!(encoded[0].len(), 2);
        assert_eq!(encoded[1].len(), 5);
        assert_eq!(encoded[2].len(), 1);
        assert_eq!(max_len, 5);
    }

    #[test]
    fn crlf_line_endings_are_normalised() {
        // A vocab.txt checked out on Windows might contain CRLF
        // separators. Each line's trailing '\r' must be stripped so
        // tokens stay clean.
        let bad = " \r\na\r\nb\r\n";
        let tok = F5Tokenizer::from_vocab_str(bad).expect("CRLF vocab");
        assert_eq!(tok.vocab_size(), 3);
        assert_eq!(tok.id_to_token(0), Some(" "));
        assert_eq!(tok.id_to_token(1), Some("a"));
        assert_eq!(tok.id_to_token(2), Some("b"));
    }
}
