//! Bark text tokenizer — wraps [`tokenizers::Tokenizer`] configured for
//! `bert-base-multilingual-cased`. Bark uses an offset trick to fit the
//! BERT vocab into a larger combined vocab that also includes semantic
//! tokens: [`TEXT_ENCODING_OFFSET`] = `10_048`.
//!
//! Upstream: `suno-ai/bark/bark/generation.py` (`load_text_tokenizer`)
//! plus the constants block at the top of that file. The
//! `tokenizer.json` we load is the HF transformers `BarkProcessor`
//! mirror, shipped at `suno/bark` (and pointed to from `suno/bark-small`
//! via its tokenizer config).

#![cfg(feature = "bark")]

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::TtsError;

/// Offset added to every BERT token id to land it in Bark's combined
/// (semantic + text) vocabulary. From
/// `bark/generation.py::TEXT_ENCODING_OFFSET`.
pub const TEXT_ENCODING_OFFSET: u32 = 10_048;

/// End-of-sentence / pad token for the semantic stage; doubles as the
/// EOS marker upstream uses when `allow_early_stop=True`. From
/// `bark/generation.py::SEMANTIC_PAD_TOKEN`.
pub const SEMANTIC_PAD_TOKEN: u32 = 10_000;

/// Text-side pad token id (already offset). Used by the pipeline to pad
/// the BERT-encoded prompt up to the semantic stage's fixed input width.
/// From `bark/generation.py::TEXT_PAD_TOKEN`.
pub const TEXT_PAD_TOKEN: u32 = 129_595;

/// Inference-mode sentinel inserted between the text prompt and the
/// semantic-token suffix when seeding the semantic decoder. From
/// `bark/generation.py::SEMANTIC_INFER_TOKEN`.
pub const SEMANTIC_INFER_TOKEN: u32 = 129_599;

/// Bark text tokenizer.
///
/// Construct via [`Self::from_path`] (already-downloaded `tokenizer.json`)
/// or, in tests / live runs, by passing a path obtained from the
/// [`super::weights::BarkWeights`] HF download.
pub struct BarkTokenizer {
    inner: Tokenizer,
}

impl BarkTokenizer {
    /// Load a BERT-multilingual `tokenizer.json` from disk.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when the file is missing or
    /// malformed.
    pub fn from_path(path: &Path) -> Result<Self, TtsError> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| TtsError::ModelLoad(format!("bark tokenizer load: {e}")))?;
        Ok(Self { inner })
    }

    /// Tokenize `text` and apply the +[`TEXT_ENCODING_OFFSET`] trick so
    /// the resulting token ids land in the combined-vocab range that
    /// Bark's semantic stage expects.
    ///
    /// Upstream `bark/generation.py::_tokenize` does this offset addition
    /// after a `tokenizer.encode(text, add_special_tokens=False)` call;
    /// we match by passing `add_special_tokens=false` here too.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Synthesis`] if the underlying
    /// [`tokenizers::Tokenizer`] fails to encode the input.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::Synthesis(format!("bark tokenize: {e}")))?;
        Ok(encoding
            .get_ids()
            .iter()
            .map(|id| id + TEXT_ENCODING_OFFSET)
            .collect())
    }

    /// Read-only access to the underlying [`tokenizers::Tokenizer`] for
    /// callers (e.g. pipeline) that need to introspect special tokens or
    /// vocab size.
    #[must_use]
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check that the constants match upstream
    /// `bark/generation.py`. If any of these change, downstream
    /// `pipeline` plumbing must change in lockstep.
    #[test]
    fn upstream_constants_match_bark_generation_py() {
        assert_eq!(TEXT_ENCODING_OFFSET, 10_048);
        assert_eq!(SEMANTIC_PAD_TOKEN, 10_000);
        assert_eq!(TEXT_PAD_TOKEN, 129_595);
        assert_eq!(SEMANTIC_INFER_TOKEN, 129_599);
    }

    /// Live network test — downloads the canonical Bark `tokenizer.json`
    /// from the HF Hub and verifies the offset is applied to a known
    /// English string. Gated by `#[ignore]` so the default `cargo test`
    /// run stays hermetic.
    ///
    /// Run with:
    ///
    /// ```bash
    /// cargo test -p blazen-audio-tts --features bark \
    ///     bark::tokenizer::tests::encode_applies_text_encoding_offset_live -- --ignored
    /// ```
    #[test]
    #[ignore = "requires network — downloads suno/bark tokenizer.json from HF Hub"]
    fn encode_applies_text_encoding_offset_live() {
        let path = tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(super::super::weights::download_tokenizer_for_test(
                "suno/bark",
            ))
            .expect("hf-hub tokenizer download");
        let tokenizer = BarkTokenizer::from_path(&path).expect("from_path");
        let ids = tokenizer.encode("hello world").expect("encode");
        assert!(!ids.is_empty(), "encode produced no tokens");
        for id in &ids {
            assert!(
                *id >= TEXT_ENCODING_OFFSET,
                "id {id} should be >= TEXT_ENCODING_OFFSET ({TEXT_ENCODING_OFFSET})",
            );
        }
    }
}
