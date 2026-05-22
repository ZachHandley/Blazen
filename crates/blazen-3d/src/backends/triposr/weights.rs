//! `TripoSR` weight loader -- downloads the canonical HF checkpoint
//! (`stabilityai/TripoSR`) via [`hf_hub`] and carves the resulting
//! state-dict into three [`VarBuilder`] roots, one per architectural
//! component:
//!
//! 1. **`DINOv2` image encoder** -- prefix `image_tokenizer.` (the
//!    upstream `TSR` class wraps a `DINOv2`-base instance in a
//!    `DINOSingleImageTokenizer` submodule attribute named
//!    `image_tokenizer`).
//! 2. **Triplane transformer decoder** -- prefix `backbone.`
//!    (`Transformer1D` decoder that maps image tokens -> triplane
//!    tokens; named `backbone` in the upstream state-dict).
//! 3. **`NeRF`/SDF field MLP** -- prefix `decoder.` (`TriplaneNeRF`
//!    decoder MLP sampled from the triplane features; named `decoder`
//!    in the upstream state-dict).
//!
//! # HF repo layout (verified 2026-05-22)
//!
//! `stabilityai/TripoSR` ships only three files at the repo root:
//!
//! - `model.ckpt` (1.68 GB) -- the full `TSR` state-dict as a torch
//!   pickle, despite the `.ckpt` extension. No safetensors mirror is
//!   currently published.
//! - `config.yaml` (987 B) -- not consumed (hyperparameters live in
//!   the local component modules).
//! - `README.md` -- not consumed.
//!
//! We attempt `model.safetensors` first (in case upstream publishes a
//! safetensors mirror) and fall back to `model.ckpt`. The safetensors
//! path uses [`VarBuilder::from_mmaped_safetensors`]; the pickle path
//! uses [`candle_core::pickle::PthTensors`] + [`VarBuilder::from_tensors`].
//!
//! # Prefix verification
//!
//! The three prefixes above (`image_tokenizer`, `backbone`, `decoder`)
//! are taken from the upstream
//! <https://github.com/VAST-AI-Research/TripoSR/blob/main/tsr/system.py>
//! `TSR.configure()` submodule attribute names. Wave T.3's
//! `pipeline.rs` author should verify these against the actual
//! `model.ckpt` keys once the weights are first downloaded -- if the
//! upstream ever republishes with renamed attributes, only the three
//! `*_PREFIX` constants in this file need to change.

#![cfg(feature = "triposr")]
// The HF layout doc-comment and the prefix verification block cite
// upstream Python identifiers (TSR, DINOv2, DINOSingleImageTokenizer,
// Transformer1D, TriplaneNeRF, ...) and HF-side filenames. Backticking
// every naked PascalCase / filename reference would bury the prose;
// mirror the allow from `f5/weights.rs`.
#![allow(clippy::doc_markdown)]
// Wave T.2 scaffolding: this module's public surface is consumed by
// `pipeline.rs` in a follow-up wave. Until that lands, every item
// here looks "dead" to the compiler. The sibling `image_encoder` /
// `triplane_transformer` modules use the same allow for the same
// reason.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use thiserror::Error;

/// Default Hugging Face repo id for the upstream `TripoSR` weights.
pub const DEFAULT_MODEL_ID: &str = "stabilityai/TripoSR";

/// Preferred weights filename inside the HF repo. We probe this first
/// even though the canonical upstream only ships `model.ckpt` today --
/// if a safetensors mirror lands later, we pick it up for free.
pub const SAFETENSORS_FILENAME: &str = "model.safetensors";

/// Fallback weights filename inside the HF repo. The canonical
/// `stabilityai/TripoSR` repo ships this as a torch pickle despite the
/// `.ckpt` extension.
pub const CKPT_FILENAME: &str = "model.ckpt";

/// State-dict prefix for the `DINOv2` image encoder submodule (the
/// `TSR.image_tokenizer` attribute in upstream `tsr/system.py`).
pub const IMAGE_ENCODER_PREFIX: &str = "image_tokenizer";

/// State-dict prefix for the triplane transformer decoder submodule
/// (the `TSR.backbone` attribute in upstream `tsr/system.py`).
pub const TRANSFORMER_PREFIX: &str = "backbone";

/// State-dict prefix for the `NeRF`/SDF field MLP submodule (the
/// `TSR.decoder` attribute in upstream `tsr/system.py`).
pub const NERF_FIELD_PREFIX: &str = "decoder";

/// Errors surfaced by the `TripoSR` weight loader.
#[derive(Debug, Error)]
pub enum TripoSrWeightsError {
    /// Filesystem I/O failed (path missing, permission denied, ...).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// `HF` download failed (no network, repo gone, file moved, auth
    /// required, ...). Carries the upstream error message verbatim.
    #[error("hf-hub download: {0}")]
    Download(String),
    /// On-disk weight file could not be parsed as either safetensors
    /// or a torch pickle. Carries a human-readable explanation
    /// (filename + the underlying parse error).
    #[error("weight file format: {0}")]
    Format(String),
    /// Candle-side error during tensor materialisation, dtype cast, or
    /// `VarBuilder` construction.
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Bundle holding three carved [`VarBuilder`]s, one per architectural
/// component. Returned by [`load_weights`] / [`load_weights_from_path`].
///
/// Each `VarBuilder` is rooted at the corresponding submodule prefix,
/// so the component modules can index their tensors with their
/// natural attribute names (e.g. `transformer_vb.pp("layers.0.attn")`)
/// without re-encoding the `backbone.` prefix.
//
// The `*_vb` suffix on every field is deliberate — it reads as
// "image_encoder VarBuilder" at every call site and matches the
// CONTRACT in the Wave T.2 spec verbatim. Clippy's `struct_field_names`
// lint flags the shared suffix, but renaming would (a) make every
// caller less explicit about the field type and (b) diverge from the
// spec.
#[allow(clippy::struct_field_names)]
pub struct TripoSrWeights {
    /// `VarBuilder` rooted at `image_tokenizer.*` -- the `DINOv2`
    /// image encoder weights.
    pub image_encoder_vb: VarBuilder<'static>,
    /// `VarBuilder` rooted at `backbone.*` -- the triplane transformer
    /// decoder weights.
    pub transformer_vb: VarBuilder<'static>,
    /// `VarBuilder` rooted at `decoder.*` -- the `NeRF`/SDF field MLP
    /// weights sampled from the triplane features.
    pub nerf_field_vb: VarBuilder<'static>,
}

impl std::fmt::Debug for TripoSrWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TripoSrWeights")
            .field("image_encoder_prefix", &IMAGE_ENCODER_PREFIX)
            .field("transformer_prefix", &TRANSFORMER_PREFIX)
            .field("nerf_field_prefix", &NERF_FIELD_PREFIX)
            .finish()
    }
}

/// Download (if missing) and load `TripoSR` weights from the HF repo.
///
/// Uses the **sync** `hf-hub` API on a `spawn_blocking` worker (same
/// pattern as
/// `crates/blazen-audio-tts/src/backends/bark/weights.rs` and the
/// sibling `f5/weights.rs`).
///
/// On cache hit no network is touched.
///
/// We probe [`SAFETENSORS_FILENAME`] first and fall back to
/// [`CKPT_FILENAME`] if the safetensors mirror is missing. The current
/// canonical repo only ships `.ckpt`, so the fallback path is the
/// normal case.
///
/// # Arguments
///
/// - `hf_repo_id`: HF repo id (typically [`DEFAULT_MODEL_ID`]).
/// - `hf_revision`: optional git revision (branch, tag, or commit
///   SHA). `None` resolves to `main`.
/// - `device`: target candle device.
///
/// # Errors
///
/// - [`TripoSrWeightsError::Download`] if both filename probes fail
///   at the HF download stage (no network, repo gone, ...).
/// - [`TripoSrWeightsError::Format`] if the downloaded file cannot
///   be parsed as either safetensors or a torch pickle.
/// - [`TripoSrWeightsError::Candle`] if tensor materialisation or
///   `VarBuilder` construction fails on the candle side.
pub async fn load_weights(
    hf_repo_id: &str,
    hf_revision: Option<&str>,
    device: &Device,
) -> Result<TripoSrWeights, TripoSrWeightsError> {
    let repo_id = hf_repo_id.to_owned();
    let revision = hf_revision.map(ToOwned::to_owned);
    let path = tokio::task::spawn_blocking(move || -> Result<PathBuf, TripoSrWeightsError> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .build()
            .map_err(|e| TripoSrWeightsError::Download(format!("hf-hub api build: {e}")))?;
        let repo = match revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                repo_id.clone(),
                hf_hub::RepoType::Model,
                rev,
            )),
            None => api.model(repo_id.clone()),
        };
        // Probe safetensors first, fall back to .ckpt. The safetensors
        // probe is expected to 404 on the canonical upstream today; we
        // swallow that error and try the .ckpt path. Only when BOTH
        // probes fail do we surface a Download error.
        match repo.get(SAFETENSORS_FILENAME) {
            Ok(p) => Ok(p),
            Err(safetensors_err) => repo.get(CKPT_FILENAME).map_err(|ckpt_err| {
                TripoSrWeightsError::Download(format!(
                    "hf-hub fetch {repo_id}/{SAFETENSORS_FILENAME}: {safetensors_err}; \
                     fallback {repo_id}/{CKPT_FILENAME}: {ckpt_err}"
                ))
            }),
        }
    })
    .await
    .map_err(|e| TripoSrWeightsError::Download(format!("spawn_blocking: {e}")))??;

    load_weights_from_path(&path, device)
}

/// Load weights from a pre-downloaded file at `path`.
///
/// Picks the parser based on the file extension: `.safetensors` uses
/// [`VarBuilder::from_mmaped_safetensors`]; anything else (including
/// the canonical `.ckpt`) uses [`PthTensors`] + [`VarBuilder::from_tensors`].
///
/// # Errors
///
/// - [`TripoSrWeightsError::Io`] if `path` doesn't exist or can't be
///   read.
/// - [`TripoSrWeightsError::Format`] if the file can't be parsed as
///   safetensors / torch-pickle.
/// - [`TripoSrWeightsError::Candle`] if tensor materialisation or
///   `VarBuilder` construction fails.
pub fn load_weights_from_path(
    path: &Path,
    device: &Device,
) -> Result<TripoSrWeights, TripoSrWeightsError> {
    // Reject missing files up front with a clean io::Error so callers
    // see TripoSrWeightsError::Io rather than a deep candle / pickle
    // parse failure. `Path::metadata()` is the cheapest existence
    // probe that also surfaces permission errors.
    let _ = path.metadata()?;

    let vb = if has_extension(path, "safetensors") {
        load_safetensors_var_builder(path, device)?
    } else {
        load_pth_var_builder(path, device)?
    };

    Ok(TripoSrWeights {
        image_encoder_vb: vb.pp(IMAGE_ENCODER_PREFIX),
        transformer_vb: vb.pp(TRANSFORMER_PREFIX),
        nerf_field_vb: vb.pp(NERF_FIELD_PREFIX),
    })
}

/// Case-insensitive extension match. Returns `true` iff `path` has the
/// given extension (no leading dot).
fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Build a root [`VarBuilder`] from a `.safetensors` file. Uses
/// candle's mmap loader for zero-copy reads; the mapping outlives the
/// returned `VarBuilder` because [`VarBuilder::from_mmaped_safetensors`]
/// owns it internally.
fn load_safetensors_var_builder(
    path: &Path,
    device: &Device,
) -> Result<VarBuilder<'static>, TripoSrWeightsError> {
    // SAFETY: candle's mmap loader requires `unsafe` because the file
    // must remain unchanged for the lifetime of the mapping. The HF-
    // hub cache directory is treated as immutable by every other
    // backend in this workspace (`f5/weights.rs`, `bark/weights.rs`,
    // ...), so we follow the same convention here.
    #[allow(unsafe_code)]
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device).map_err(|e| {
            TripoSrWeightsError::Format(format!("safetensors mmap {}: {e}", path.display()))
        })?
    };
    Ok(vb)
}

/// Build a root [`VarBuilder`] from a torch pickle (`.ckpt` / `.pth` /
/// `.bin`). Materialises every tensor into device memory up front via
/// [`PthTensors`] + [`VarBuilder::from_tensors`].
fn load_pth_var_builder(
    path: &Path,
    device: &Device,
) -> Result<VarBuilder<'static>, TripoSrWeightsError> {
    let pickle = PthTensors::new(path, None)
        .map_err(|e| TripoSrWeightsError::Format(format!("pth open {}: {e}", path.display())))?;

    let raw_keys: Vec<String> = pickle.tensor_infos().keys().cloned().collect();
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(raw_keys.len());
    for raw_key in raw_keys {
        let tensor = pickle
            .get(&raw_key)
            .map_err(|e| TripoSrWeightsError::Format(format!("pth load tensor {raw_key}: {e}")))?
            .ok_or_else(|| TripoSrWeightsError::Format(format!("pth missing tensor {raw_key}")))?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        tensors.insert(raw_key, tensor);
    }

    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_constants_match_documented_layout() {
        // Guard against accidental edits. If any of these change, the
        // top-of-file HF-layout block and the prefix verification
        // note MUST be re-checked against the live upstream and
        // updated alongside.
        assert_eq!(DEFAULT_MODEL_ID, "stabilityai/TripoSR");
        assert_eq!(SAFETENSORS_FILENAME, "model.safetensors");
        assert_eq!(CKPT_FILENAME, "model.ckpt");
        assert_eq!(IMAGE_ENCODER_PREFIX, "image_tokenizer");
        assert_eq!(TRANSFORMER_PREFIX, "backbone");
        assert_eq!(NERF_FIELD_PREFIX, "decoder");
    }

    #[test]
    fn has_extension_matches_case_insensitively() {
        assert!(has_extension(Path::new("foo.safetensors"), "safetensors"));
        assert!(has_extension(Path::new("foo.SAFETENSORS"), "safetensors"));
        assert!(has_extension(Path::new("a/b/c.ckpt"), "ckpt"));
        assert!(!has_extension(Path::new("foo.ckpt"), "safetensors"));
        assert!(!has_extension(Path::new("foo"), "safetensors"));
    }

    #[test]
    fn load_weights_from_path_returns_error_for_missing_file() {
        // Use a tempdir-anchored path that is guaranteed not to exist
        // (the tempdir auto-cleans on drop, and we never create the
        // child file).
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing = tmp.path().join("nonexistent_blazen_triposr.ckpt");
        let device = Device::Cpu;
        let err = load_weights_from_path(&missing, &device).expect_err("missing file must error");
        assert!(
            matches!(err, TripoSrWeightsError::Io(_)),
            "expected Io variant for missing file, got {err:?}",
        );
    }

    #[test]
    fn load_weights_from_path_returns_error_for_corrupted_safetensors() {
        // Write a 100-byte buffer of 0xff into a temp .safetensors
        // file. Neither the safetensors header parser nor the pickle
        // parser should accept this. The exact variant depends on
        // which parser fires first; we accept any Format / Candle
        // surface, but Io is wrong because the file does exist.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("corrupt.safetensors");
        std::fs::write(&path, vec![0xffu8; 100]).expect("write corrupt file");
        let device = Device::Cpu;
        let err =
            load_weights_from_path(&path, &device).expect_err("corrupted safetensors must error");
        assert!(
            !matches!(err, TripoSrWeightsError::Io(_)),
            "expected parse error, not Io, for corrupt-but-present file: {err:?}",
        );
    }

    #[test]
    fn load_weights_from_path_returns_error_for_corrupted_ckpt() {
        // Same as the safetensors case, but with the `.ckpt` extension
        // so we exercise the pickle path. 100 bytes of 0xff is not a
        // valid torch pickle stream.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("corrupt.ckpt");
        std::fs::write(&path, vec![0xffu8; 100]).expect("write corrupt file");
        let device = Device::Cpu;
        let err = load_weights_from_path(&path, &device).expect_err("corrupted ckpt must error");
        assert!(
            !matches!(err, TripoSrWeightsError::Io(_)),
            "expected parse error, not Io, for corrupt-but-present file: {err:?}",
        );
    }
}
