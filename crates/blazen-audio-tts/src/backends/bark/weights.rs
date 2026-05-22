//! Bark weight loading — downloads the canonical HF transformers Bark
//! checkpoint (`suno/bark-small` by default) via [`hf_hub`] and
//! constructs the three transformer stages from the resulting
//! [`VarBuilder`].
//!
//! # HF repo layout
//!
//! The HF transformers `BarkModel` ships **one** `pytorch_model.bin`
//! containing all three stages under the prefixes:
//!
//! - `semantic.*`         — semantic stage (`BarkSemanticModel`)
//! - `coarse_acoustics.*` — coarse stage (`BarkCoarseModel`)
//! - `fine_acoustics.*`   — fine stage (`BarkFineModel`)
//! - `codec_model.*`      — `EnCodec` (handled by `blazen-audio-codec`)
//!
//! The suno-ai mirror at `suno/bark` (full-size) additionally ships
//! per-stage pickle files (`text_2.pt`, `coarse_2.pt`, `fine_2.pt`) in
//! the original Suno format, but those use a different state-dict layout
//! and we do **not** load them. The `bark-small` repo ships only the
//! HF transformers-shaped `pytorch_model.bin`.
//!
//! No safetensors mirror exists at the time of writing; both `suno/bark`
//! and `suno/bark-small` ship `.bin` only. We use candle's
//! [`VarBuilder::from_pth`] which streams the pickle file directly.
//!
//! Upstream HF transformers config:
//! <https://github.com/huggingface/transformers/blob/main/src/transformers/models/bark/configuration_bark.py>.

#![cfg(feature = "bark")]

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::error::TtsError;

use super::coarse::{CoarseConfig, CoarseDecoder};
use super::fine::{FineConfig, FineDecoder};
use super::semantic::{SemanticConfig, SemanticDecoder};

/// Default model id; the small (~600 MB) Bark variant.
pub const DEFAULT_MODEL_ID: &str = "suno/bark-small";

/// Filename of the combined HF transformers Bark checkpoint inside the
/// HF repo.
pub const WEIGHTS_FILENAME: &str = "pytorch_model.bin";

/// Filename of the BERT-multilingual `tokenizer.json` shipped by HF
/// transformers' Bark repos.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// State-dict prefix for the semantic stage inside the combined
/// `pytorch_model.bin` (matches HF transformers `BarkModel.semantic`).
pub const SEMANTIC_PREFIX: &str = "semantic";

/// State-dict prefix for the coarse stage (matches
/// `BarkModel.coarse_acoustics`).
pub const COARSE_PREFIX: &str = "coarse_acoustics";

/// State-dict prefix for the fine stage (matches
/// `BarkModel.fine_acoustics`).
pub const FINE_PREFIX: &str = "fine_acoustics";

/// Bundle holding the three Bark stages plus the path to the BERT
/// tokenizer JSON. Returned by [`BarkWeights::from_hf`].
pub struct BarkWeights {
    /// Semantic stage (text → semantic-token GPT decoder).
    pub semantic: SemanticDecoder,
    /// Coarse acoustic stage (semantic → first two `EnCodec` codebooks).
    pub coarse: CoarseDecoder,
    /// Fine acoustic stage (coarse → remaining six `EnCodec` codebooks).
    pub fine: FineDecoder,
    /// Filesystem path to the BERT-multilingual `tokenizer.json` shipped
    /// alongside the HF transformers Bark checkpoint.
    pub tokenizer_path: PathBuf,
}

/// Per-stage configuration triple. Pass [`Self::bark_small`] for the
/// default small checkpoint.
#[derive(Debug, Clone)]
pub struct BarkConfigs {
    /// Semantic stage configuration.
    pub semantic: SemanticConfig,
    /// Coarse stage configuration.
    pub coarse: CoarseConfig,
    /// Fine stage configuration.
    pub fine: FineConfig,
}

impl BarkConfigs {
    /// Configs for `suno/bark-small`. The HF transformers checkpoint
    /// uses `bias=false` for the transformer body (see
    /// `suno/bark-small/config.json`); the published Suno-format weights
    /// use `bias=true`. The flag here mirrors the HF default so the
    /// `pytorch_model.bin` weight load succeeds out of the box.
    #[must_use]
    pub fn bark_small() -> Self {
        let mut semantic = SemanticConfig::bark_small();
        semantic.bias = false;
        let mut coarse = CoarseConfig::bark_small();
        coarse.bias = false;
        let mut fine = FineConfig::bark_small();
        fine.bias = false;
        Self {
            semantic,
            coarse,
            fine,
        }
    }
}

impl Default for BarkConfigs {
    fn default() -> Self {
        Self::bark_small()
    }
}

impl BarkWeights {
    /// Download and load all three Bark stages.
    ///
    /// Uses `hf-hub` (sync API on a `spawn_blocking` worker, mirroring
    /// `crates/blazen-audio-stt/src/backends/whisper_streaming/vad.rs`'s
    /// `from_hf` pattern). On cache hit no network is touched.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when the HF download fails (no
    /// network, repo gone, file moved) or when constructing one of the
    /// candle decoders fails (missing tensor name, dtype mismatch, …).
    pub async fn from_hf(
        model_id: &str,
        configs: BarkConfigs,
        device: &Device,
    ) -> Result<Self, TtsError> {
        let model_id_owned = model_id.to_owned();
        let paths = tokio::task::spawn_blocking(move || -> Result<(PathBuf, PathBuf), TtsError> {
            let api = hf_hub::api::sync::ApiBuilder::new()
                .build()
                .map_err(|e| TtsError::ModelLoad(format!("bark hf-hub api: {e}")))?;
            let repo = api.model(model_id_owned.clone());
            let weights = repo.get(WEIGHTS_FILENAME).map_err(|e| {
                TtsError::ModelLoad(format!(
                    "bark hf-hub fetch {model_id_owned}/{WEIGHTS_FILENAME}: {e}"
                ))
            })?;
            let tokenizer = repo.get(TOKENIZER_FILENAME).map_err(|e| {
                TtsError::ModelLoad(format!(
                    "bark hf-hub fetch {model_id_owned}/{TOKENIZER_FILENAME}: {e}"
                ))
            })?;
            Ok((weights, tokenizer))
        })
        .await
        .map_err(|e| TtsError::ModelLoad(format!("bark spawn_blocking: {e}")))??;
        let (weights_path, tokenizer_path) = paths;

        Self::from_paths(&weights_path, tokenizer_path, configs, device)
    }

    /// Construct from already-downloaded paths. Useful for tests with a
    /// pre-cached checkpoint, or for offline / air-gapped environments.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when the `pytorch_model.bin` cannot
    /// be parsed or when any stage's tensors are missing / mis-shaped.
    pub fn from_paths(
        weights_path: &std::path::Path,
        tokenizer_path: PathBuf,
        configs: BarkConfigs,
        device: &Device,
    ) -> Result<Self, TtsError> {
        let vb = VarBuilder::from_pth(weights_path, DType::F32, device)
            .map_err(|e| TtsError::ModelLoad(format!("bark from_pth: {e}")))?;

        let semantic = SemanticDecoder::load(vb.pp(SEMANTIC_PREFIX), configs.semantic)
            .map_err(|e| TtsError::ModelLoad(format!("bark semantic load: {e}")))?;
        let coarse = CoarseDecoder::from_vb(vb.pp(COARSE_PREFIX), configs.coarse)
            .map_err(|e| TtsError::ModelLoad(format!("bark coarse load: {e}")))?;
        let fine = FineDecoder::from_vb(vb.pp(FINE_PREFIX), configs.fine)
            .map_err(|e| TtsError::ModelLoad(format!("bark fine load: {e}")))?;

        Ok(Self {
            semantic,
            coarse,
            fine,
            tokenizer_path,
        })
    }
}

/// Internal helper for the tokenizer test — downloads just the
/// `tokenizer.json` from `model_id` and returns its cached path.
///
/// Crate-private; not part of the public API. Lives here (rather than in
/// `tokenizer.rs`) so the `hf-hub` glue stays in one place.
#[doc(hidden)]
pub(super) async fn download_tokenizer_for_test(model_id: &str) -> Result<PathBuf, TtsError> {
    let model_id_owned = model_id.to_owned();
    tokio::task::spawn_blocking(move || -> Result<PathBuf, TtsError> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .build()
            .map_err(|e| TtsError::ModelLoad(format!("bark hf-hub api: {e}")))?;
        let repo = api.model(model_id_owned.clone());
        repo.get(TOKENIZER_FILENAME).map_err(|e| {
            TtsError::ModelLoad(format!(
                "bark hf-hub fetch {model_id_owned}/{TOKENIZER_FILENAME}: {e}"
            ))
        })
    })
    .await
    .map_err(|e| TtsError::ModelLoad(format!("bark spawn_blocking: {e}")))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bark_small_configs_have_no_bias() {
        // HF transformers' `suno/bark-small/config.json` has
        // `bias: false` across all three stages — our default configs
        // must mirror that so weight loading from the published bin
        // file works without manual tweaking.
        let cfgs = BarkConfigs::bark_small();
        assert!(!cfgs.semantic.bias);
        assert!(!cfgs.coarse.bias);
        assert!(!cfgs.fine.bias);
    }

    /// Live network test — downloads `suno/bark-small` weights and
    /// tokenizer from HF Hub, constructs all three Bark stages, and
    /// verifies the bundle is non-empty. Gated by `#[ignore]`.
    ///
    /// Run with:
    ///
    /// ```bash
    /// cargo test -p blazen-audio-tts --features bark \
    ///     bark::weights::tests::from_hf_loads_bark_small_live -- --ignored
    /// ```
    ///
    /// First run downloads ~600 MB; subsequent runs hit the local
    /// HF cache and are fast.
    #[tokio::test]
    #[ignore = "requires network and ~600 MB download from HF Hub"]
    async fn from_hf_loads_bark_small_live() {
        let device = Device::Cpu;
        let weights = BarkWeights::from_hf(DEFAULT_MODEL_ID, BarkConfigs::bark_small(), &device)
            .await
            .expect("bark-small load");
        assert_eq!(weights.semantic.config().n_layer, 12);
        assert_eq!(weights.coarse.config().n_layer, 12);
        // tokenizer.json must be a real file on disk
        assert!(
            weights.tokenizer_path.exists(),
            "tokenizer path should exist: {}",
            weights.tokenizer_path.display()
        );
    }
}
