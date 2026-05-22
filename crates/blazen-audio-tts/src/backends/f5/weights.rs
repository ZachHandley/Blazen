//! F5-TTS weight loader — downloads the DiT checkpoint, the
//! character vocab, and the Vocos vocoder from Hugging Face and
//! materialises the two candle modules
//! ([`super::dit_wrapper::F5Dit`] + [`super::vocos::VocosBackbone`])
//! through key-remapped [`VarBuilder`]s.
//!
//! # Hugging Face repo layouts (verified 2026-05-22)
//!
//! `SWivid/F5-TTS`, directory `F5TTS_Base/` — files:
//! - `model_1200000.safetensors` (1.35 GB) — the F5-TTS DiT weights
//!   in EMA form. Top-level state-dict keys are all prefixed with
//!   `ema_model.transformer.` plus housekeeping entries `initted` and
//!   `step` (handled by [`load_state_dict`] in upstream
//!   `utils_infer.py`; we strip them here on the Rust side).
//! - `model_1200000.pt` (1.35 GB) — the same weights as a pickle.
//!   We prefer the safetensors variant for safety + parallelism.
//! - `vocab.txt` (13.8 kB) — UTF-8 char-per-line vocabulary consumed
//!   by [`super::tokenizer::F5Tokenizer`].
//!
//! `charactr/vocos-mel-24khz` — files:
//! - `pytorch_model.bin` (54.4 MB) — Vocos vocoder weights as a
//!   torch pickle. State-dict keys are split between `backbone.*`
//!   and `head.*`. The local [`super::vocos::VocosBackbone`] expects
//!   the backbone tensors at the **root** of its VarBuilder
//!   alongside the head tensors (see Wave 2's `VocosBackbone::load`
//!   for the layout it actually reads), so we apply a one-shot
//!   `backbone.` → `` strip during load.
//! - `config.yaml` (461 B) — not consumed (the local
//!   [`super::vocos::VocosConfig::vocos_24khz`] hardcodes the
//!   matching hyperparameters).
//!
//! # DiT state-dict prefix handling
//!
//! Upstream `load_checkpoint` in `f5_tts/infer/utils_infer.py` does
//! the equivalent of:
//!
//! ```python
//! # checkpoint is the raw safetensors dict.
//! checkpoint["model_state_dict"] = {
//!     k.replace("ema_model.", ""): v
//!     for k, v in checkpoint["ema_model_state_dict"].items()
//!     if k not in ["initted", "step"]
//! }
//! model.load_state_dict(checkpoint["model_state_dict"])
//! ```
//!
//! where `model` is a `CFM` instance whose `.transformer` submodule
//! is the F5 DiT. Stripping `ema_model.` leaves keys like
//! `transformer.time_embed.time_mlp.0.weight`. The local
//! [`super::dit_wrapper::F5Dit::new`] then expects its VarBuilder to
//! be rooted at the DiT — i.e. at `transformer.*`. We therefore
//! strip the full `ema_model.transformer.` prefix during load and
//! drop the `ema_model.mel_spec.*` keys, `initted`, and `step` (none
//! of which feed the DiT).

#![cfg(feature = "f5-tts")]
// The HF layout doc-comment and the per-function load docs cite
// upstream Python identifiers (DiT, CFM, VocosBackbone, ISTFTHead,
// EMA, …) and HF-side filenames (model_1200000.safetensors,
// pytorch_model.bin, vocab.txt). Backticking every naked
// PascalCase / filename reference would bury the prose. Mirror the
// allow from `dit_wrapper.rs`.
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::pickle::PthTensors;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::TtsError;

use super::dit_wrapper::{F5Dit, F5DitConfig};
use super::tokenizer::F5Tokenizer;
use super::vocos::{VocosBackbone, VocosConfig};

/// Default Hugging Face repo id for the F5-TTS DiT + vocab.
pub const DEFAULT_F5_MODEL_ID: &str = "SWivid/F5-TTS";

/// Default Hugging Face repo id for the Vocos mel-24-kHz vocoder.
pub const DEFAULT_VOCOS_MODEL_ID: &str = "charactr/vocos-mel-24khz";

/// Subdirectory inside [`DEFAULT_F5_MODEL_ID`] that holds the
/// flagship `F5TTS_Base` checkpoint.
pub const F5_BASE_SUBDIR: &str = "F5TTS_Base";

/// Filename of the F5-TTS DiT safetensors checkpoint inside
/// `F5TTS_Base/`. We always prefer this over the legacy `.pt`
/// variant — same weights, faster and safer load.
pub const F5_WEIGHTS_FILENAME: &str = "F5TTS_Base/model_1200000.safetensors";

/// Filename of the character vocab inside `F5TTS_Base/`.
pub const F5_VOCAB_FILENAME: &str = "F5TTS_Base/vocab.txt";

/// Filename of the Vocos vocoder pickle inside
/// [`DEFAULT_VOCOS_MODEL_ID`].
pub const VOCOS_WEIGHTS_FILENAME: &str = "pytorch_model.bin";

/// Prefix on the raw safetensors keys that wraps the EMA model
/// (`ema_model.transformer.<dit-subkey>`). The full prefix is
/// stripped during load so the resulting VarBuilder is rooted at
/// the DiT itself.
const F5_EMA_TRANSFORMER_PREFIX: &str = "ema_model.transformer.";

/// Top-level safetensors keys that do NOT live under
/// [`F5_EMA_TRANSFORMER_PREFIX`] and that the DiT doesn't consume
/// (mel-spec FFT buffers, EMA bookkeeping). Dropped during load.
const F5_DROPPED_KEY_PREFIXES: &[&str] = &["ema_model.mel_spec."];

/// Stand-alone bookkeeping keys (not prefixes) emitted by the
/// upstream EMA wrapper. Dropped during load.
const F5_DROPPED_KEYS: &[&str] = &["initted", "step", "ema_model.initted", "ema_model.step"];

/// Prefix on the Vocos backbone state-dict keys. Stripped during
/// load so [`VocosBackbone::load`] (which expects `embed`, `norm`,
/// `convnext`, `final_layer_norm` at the root alongside `head.*`)
/// finds its tensors.
const VOCOS_BACKBONE_PREFIX: &str = "backbone.";

/// Bundle holding the loaded F5-TTS DiT, the Vocos vocoder, and the
/// path to the cached `vocab.txt` (so the caller can build a
/// [`F5Tokenizer`] without re-downloading).
pub struct F5Weights {
    /// Loaded F5-TTS DiT.
    pub dit: F5Dit,
    /// Loaded Vocos vocoder.
    pub vocos: VocosBackbone,
    /// Filesystem path to the cached F5-TTS character vocab. Pass
    /// this to [`F5Tokenizer::from_vocab_path`].
    pub vocab_path: PathBuf,
}

impl std::fmt::Debug for F5Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F5Weights")
            .field("dit_depth", &self.dit.config().depth)
            .field("vocos_layers", &self.vocos.config().n_layers)
            .field("vocab_path", &self.vocab_path)
            .finish()
    }
}

impl F5Weights {
    /// Build a [`F5Tokenizer`] from the cached vocab file.
    ///
    /// # Errors
    ///
    /// Propagates I/O / parse errors from
    /// [`F5Tokenizer::from_vocab_path`].
    pub fn tokenizer(&self) -> Result<F5Tokenizer, TtsError> {
        F5Tokenizer::from_vocab_path(&self.vocab_path)
    }

    /// Download F5-TTS + Vocos weights from Hugging Face and
    /// construct both candle modules on `device`.
    ///
    /// Uses the **sync** `hf-hub` API on a `spawn_blocking` worker
    /// (same pattern as `crates/blazen-audio-stt/src/backends/
    /// whisper_streaming/vad.rs` and the sibling
    /// [`super::super::bark::weights::BarkWeights::from_hf`]).
    ///
    /// On cache hit no network is touched.
    ///
    /// # Arguments
    ///
    /// - `f5_model_id`: HF repo id for the F5-TTS DiT + vocab
    ///   (typically `"SWivid/F5-TTS"`).
    /// - `vocos_model_id`: HF repo id for the Vocos vocoder
    ///   (typically `"charactr/vocos-mel-24khz"`).
    /// - `dit_config`: DiT hyperparameters. Use
    ///   [`F5DitConfig::f5_base`] for the published `F5TTS_Base`.
    /// - `vocos_config`: Vocoder hyperparameters. Use
    ///   [`VocosConfig::vocos_24khz`] for the canonical 24 kHz
    ///   checkpoint.
    /// - `device`: target candle device.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when any of the HF downloads
    /// fail, when the safetensors / pickle parse fails, or when the
    /// candle module construction trips on a missing / mis-shaped
    /// tensor.
    pub async fn from_hf(
        f5_model_id: &str,
        vocos_model_id: &str,
        dit_config: F5DitConfig,
        vocos_config: VocosConfig,
        device: &Device,
    ) -> Result<Self, TtsError> {
        let paths = Self::download_all(f5_model_id, vocos_model_id).await?;
        Self::from_paths(
            &paths.f5_safetensors,
            paths.f5_vocab,
            &paths.vocos_pth,
            dit_config,
            vocos_config,
            device,
        )
    }

    /// Construct from already-downloaded paths. Useful for offline
    /// / air-gapped environments and for tests that pin a specific
    /// cached checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::ModelLoad`] when the safetensors /
    /// pickle parse fails or when the candle module construction
    /// fails.
    pub fn from_paths(
        f5_safetensors_path: &Path,
        vocab_path: PathBuf,
        vocos_pth_path: &Path,
        dit_config: F5DitConfig,
        vocos_config: VocosConfig,
        device: &Device,
    ) -> Result<Self, TtsError> {
        let dit_vb = load_f5_dit_var_builder(f5_safetensors_path, device)?;
        let dit = F5Dit::new(dit_vb, dit_config)
            .map_err(|e| TtsError::ModelLoad(format!("f5-tts dit construct: {e}")))?;

        let vocos_vb = load_vocos_var_builder(vocos_pth_path, device)?;
        let vocos = VocosBackbone::load(vocos_vb, vocos_config)
            .map_err(|e| TtsError::ModelLoad(format!("f5-tts vocos construct: {e}")))?;

        Ok(Self {
            dit,
            vocos,
            vocab_path,
        })
    }

    /// Internal: spawn-blocking-wrapped fetch of all three artefacts
    /// (F5 safetensors, F5 vocab, Vocos pth) via the sync `hf-hub`
    /// API. Returns the cached on-disk paths.
    async fn download_all(
        f5_model_id: &str,
        vocos_model_id: &str,
    ) -> Result<DownloadedPaths, TtsError> {
        let f5_id = f5_model_id.to_owned();
        let vocos_id = vocos_model_id.to_owned();
        tokio::task::spawn_blocking(move || -> Result<DownloadedPaths, TtsError> {
            let api = hf_hub::api::sync::ApiBuilder::new()
                .build()
                .map_err(|e| TtsError::ModelLoad(format!("f5-tts hf-hub api: {e}")))?;

            let f5_repo = api.model(f5_id.clone());
            let f5_safetensors = f5_repo.get(F5_WEIGHTS_FILENAME).map_err(|e| {
                TtsError::ModelLoad(format!(
                    "f5-tts hf-hub fetch {f5_id}/{F5_WEIGHTS_FILENAME}: {e}"
                ))
            })?;
            let f5_vocab = f5_repo.get(F5_VOCAB_FILENAME).map_err(|e| {
                TtsError::ModelLoad(format!(
                    "f5-tts hf-hub fetch {f5_id}/{F5_VOCAB_FILENAME}: {e}"
                ))
            })?;

            let vocos_repo = api.model(vocos_id.clone());
            let vocos_pth = vocos_repo.get(VOCOS_WEIGHTS_FILENAME).map_err(|e| {
                TtsError::ModelLoad(format!(
                    "f5-tts hf-hub fetch {vocos_id}/{VOCOS_WEIGHTS_FILENAME}: {e}"
                ))
            })?;

            Ok(DownloadedPaths {
                f5_safetensors,
                f5_vocab,
                vocos_pth,
            })
        })
        .await
        .map_err(|e| TtsError::ModelLoad(format!("f5-tts spawn_blocking: {e}")))?
    }
}

/// Cached HF paths returned by [`F5Weights::download_all`].
struct DownloadedPaths {
    f5_safetensors: PathBuf,
    f5_vocab: PathBuf,
    vocos_pth: PathBuf,
}

/// Decide what to do with a single raw safetensors key. Returns the
/// candle-side key if it should be loaded, `None` if it should be
/// dropped.
fn remap_f5_key(raw_key: &str) -> Option<String> {
    if F5_DROPPED_KEYS.contains(&raw_key) {
        return None;
    }
    if F5_DROPPED_KEY_PREFIXES
        .iter()
        .any(|p| raw_key.starts_with(p))
    {
        return None;
    }
    raw_key
        .strip_prefix(F5_EMA_TRANSFORMER_PREFIX)
        .map(ToOwned::to_owned)
}

/// Remap a single raw Vocos pickle key into the layout the local
/// [`VocosBackbone::load`] expects. Strips the `backbone.` prefix
/// and passes `head.*` (and any other top-level keys) through
/// unchanged. The function is total — no Vocos checkpoint keys are
/// dropped today.
fn remap_vocos_key(raw_key: &str) -> String {
    raw_key
        .strip_prefix(VOCOS_BACKBONE_PREFIX)
        .map_or_else(|| raw_key.to_owned(), ToOwned::to_owned)
}

/// Load the F5-TTS DiT safetensors file, remap keys, and return a
/// [`VarBuilder`] rooted at the DiT (i.e. with `time_embed`,
/// `input_embed`, `transformer_blocks`, … as direct children).
fn load_f5_dit_var_builder(
    safetensors_path: &Path,
    device: &Device,
) -> Result<VarBuilder<'static>, TtsError> {
    // SAFETY: candle's mmap loader requires `unsafe` because the
    // file must remain unchanged for the lifetime of the mapping.
    // We immediately materialise every tensor into device memory
    // and drop the mmap before returning — the safetensors file is
    // expected to live inside the immutable hf-hub cache anyway.
    #[allow(unsafe_code)]
    let mmap = unsafe {
        MmapedSafetensors::new(safetensors_path).map_err(|e| {
            TtsError::ModelLoad(format!(
                "f5-tts safetensors mmap {}: {e}",
                safetensors_path.display()
            ))
        })?
    };

    let mut remapped: HashMap<String, Tensor> = HashMap::new();
    let mut dropped: u32 = 0;
    let mut unmapped: u32 = 0;
    for (raw_key, _view) in mmap.tensors() {
        match remap_f5_key(&raw_key) {
            Some(candle_key) => {
                let tensor = mmap.load(&raw_key, device).map_err(|e| {
                    TtsError::ModelLoad(format!("f5-tts safetensors load {raw_key}: {e}"))
                })?;
                remapped.insert(candle_key, tensor);
            }
            None => {
                // Decide whether this is a known drop or an unknown
                // key we passed through. Anything that doesn't start
                // with `ema_model.transformer.` and isn't on the
                // drop list ends up here.
                if F5_DROPPED_KEYS.contains(&raw_key.as_str())
                    || F5_DROPPED_KEY_PREFIXES
                        .iter()
                        .any(|p| raw_key.starts_with(p))
                {
                    dropped = dropped.saturating_add(1);
                } else {
                    unmapped = unmapped.saturating_add(1);
                    tracing::debug!(
                        key = %raw_key,
                        "f5-tts safetensors key did not match any remap rule; ignoring",
                    );
                }
            }
        }
    }
    tracing::debug!(
        kept = remapped.len(),
        dropped,
        unmapped,
        "f5-tts dit weights remapped",
    );

    Ok(VarBuilder::from_tensors(remapped, DType::F32, device))
}

/// Load the Vocos pickle, remap `backbone.*` → root, and return a
/// [`VarBuilder`] rooted such that
/// [`super::vocos::VocosBackbone::load`] finds both its backbone
/// children (`embed`, `norm`, `convnext`, `final_layer_norm`) and
/// the head (`head.out`) under the same root.
fn load_vocos_var_builder(
    pth_path: &Path,
    device: &Device,
) -> Result<VarBuilder<'static>, TtsError> {
    // `PthTensors::new(path, None)` parses the entire torch pickle
    // and indexes its tensor table; downstream `.get(name)`
    // materialises each tensor on demand.
    let pth = PthTensors::new(pth_path, None).map_err(|e| {
        TtsError::ModelLoad(format!("f5-tts vocos pth open {}: {e}", pth_path.display()))
    })?;

    let mut remapped: HashMap<String, Tensor> = HashMap::new();
    let raw_keys: Vec<String> = pth.tensor_infos().keys().cloned().collect();
    for raw_key in raw_keys {
        let candle_key = remap_vocos_key(&raw_key);
        let tensor = pth
            .get(&raw_key)
            .map_err(|e| TtsError::ModelLoad(format!("f5-tts vocos pth load {raw_key}: {e}")))?
            .ok_or_else(|| {
                TtsError::ModelLoad(format!("f5-tts vocos pth missing tensor {raw_key}"))
            })?
            .to_device(device)
            .map_err(|e| {
                TtsError::ModelLoad(format!("f5-tts vocos pth move-to-device {raw_key}: {e}"))
            })?
            .to_dtype(DType::F32)
            .map_err(|e| TtsError::ModelLoad(format!("f5-tts vocos pth cast {raw_key}: {e}")))?;
        remapped.insert(candle_key, tensor);
    }
    tracing::debug!(kept = remapped.len(), "f5-tts vocos weights remapped");

    Ok(VarBuilder::from_tensors(remapped, DType::F32, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_f5_strips_ema_transformer_prefix() {
        let raw = "ema_model.transformer.time_embed.time_mlp.0.weight";
        let mapped = remap_f5_key(raw).expect("known key");
        assert_eq!(mapped, "time_embed.time_mlp.0.weight");
    }

    #[test]
    fn remap_f5_strips_transformer_blocks_path() {
        let raw = "ema_model.transformer.transformer_blocks.5.attn.to_qkv.weight";
        let mapped = remap_f5_key(raw).expect("known key");
        assert_eq!(mapped, "transformer_blocks.5.attn.to_qkv.weight");
    }

    #[test]
    fn remap_f5_drops_bookkeeping_keys() {
        assert!(remap_f5_key("initted").is_none());
        assert!(remap_f5_key("step").is_none());
        assert!(remap_f5_key("ema_model.initted").is_none());
        assert!(remap_f5_key("ema_model.step").is_none());
        assert!(remap_f5_key("ema_model.mel_spec.mel_stft.spectrogram.window").is_none());
    }

    #[test]
    fn remap_f5_passes_unknown_top_level_through_as_drop() {
        // Anything outside the ema_model.transformer namespace that
        // isn't on the drop list returns None (so the caller can log
        // it). The current upstream checkpoint only contains the
        // namespaces we already handle; this guards against future
        // additions.
        let raw = "some.future.key.weight";
        assert!(remap_f5_key(raw).is_none());
    }

    #[test]
    fn remap_vocos_strips_backbone_prefix() {
        let raw = "backbone.embed.weight";
        assert_eq!(remap_vocos_key(raw), "embed.weight");
    }

    #[test]
    fn remap_vocos_strips_backbone_convnext_path() {
        let raw = "backbone.convnext.3.dwconv.weight";
        assert_eq!(remap_vocos_key(raw), "convnext.3.dwconv.weight");
    }

    #[test]
    fn remap_vocos_keeps_head_prefix() {
        // `head.out.weight` must NOT be touched — vocos.rs reads it
        // verbatim via `vb.pp("head").pp("out")`.
        let raw = "head.out.weight";
        assert_eq!(remap_vocos_key(raw), "head.out.weight");
    }

    #[test]
    fn default_constants_match_documented_layout() {
        // Guard against accidental edits to the public constants. If
        // any of these change, the doc-comment HF-layout block at the
        // top of this file MUST be re-verified against the live HF
        // repo and updated alongside.
        assert_eq!(DEFAULT_F5_MODEL_ID, "SWivid/F5-TTS");
        assert_eq!(DEFAULT_VOCOS_MODEL_ID, "charactr/vocos-mel-24khz");
        assert_eq!(F5_WEIGHTS_FILENAME, "F5TTS_Base/model_1200000.safetensors");
        assert_eq!(F5_VOCAB_FILENAME, "F5TTS_Base/vocab.txt");
        assert_eq!(VOCOS_WEIGHTS_FILENAME, "pytorch_model.bin");
    }

    /// Live network test — downloads the full F5-TTS + Vocos bundle
    /// (~1.4 GB) from Hugging Face, constructs both candle modules,
    /// and verifies the in-memory shapes are sane.
    ///
    /// Gated by `#[ignore]`; first run downloads ~1.4 GB, subsequent
    /// runs hit the local HF cache.
    ///
    /// Run with:
    ///
    /// ```bash
    /// cargo test -p blazen-audio-tts --features f5-tts \
    ///     backends::f5::weights::tests::from_hf_loads_f5_base_live \
    ///     -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "requires network and ~1.4 GB download from HF Hub"]
    async fn from_hf_loads_f5_base_live() {
        let device = Device::Cpu;
        let weights = F5Weights::from_hf(
            DEFAULT_F5_MODEL_ID,
            DEFAULT_VOCOS_MODEL_ID,
            F5DitConfig::f5_base(),
            VocosConfig::vocos_24khz(),
            &device,
        )
        .await
        .expect("F5 + Vocos load");
        assert_eq!(weights.dit.config().depth, 22);
        assert_eq!(weights.dit.config().hidden_dim, 1024);
        assert_eq!(weights.vocos.config().n_layers, 8);
        assert!(
            weights.vocab_path.exists(),
            "vocab path should exist: {}",
            weights.vocab_path.display()
        );
        let tok = weights.tokenizer().expect("tokenizer from cached vocab");
        // F5TTS_Base ships a non-trivial vocab.
        assert!(
            tok.vocab_size() > 100,
            "expected vocab_size > 100, got {}",
            tok.vocab_size()
        );
        assert_eq!(tok.pad_token(), 0);
    }
}
