//! RVC voice-profile loader.
//!
//! A voice profile is a directory under
//! `$BLAZEN_RVC_VOICE_DIR/<voice_id>/` (default
//! `~/.cache/blazen/rvc/voices/<voice_id>/`) holding:
//!
//! - `model.pth`  — pytorch state-dict for the NSF-HiFi-GAN generator.
//! - `index.bin`  — bincode-serialised [`FeatureIndex`] (built by
//!   [`super::retrieval::FeatureIndex::save`]).
//! - `meta.toml`  — optional `{ speaker_id, rvc_version, sample_rate_hz }`
//!   metadata. Defaults are `speaker_id = 0`, `rvc_version = "v2"`,
//!   `sample_rate_hz = 40000`.
//!
//! The `model.pth` is the canonical upstream RVC checkpoint that stores
//! weights under a few top-level submodules; this loader strips the
//! enclosing module prefixes (`dec.`, `enc_p.`) so the resulting
//! [`VarBuilder`] sees keys at the paths
//! [`super::generator::NsfHifiGan::load_from_var_builder`] expects.
//!
//! Pytorch -> candle key remap (the table is also documented inline in
//! [`remap_key`]):
//!
//! ```text
//! enc_p.emb_pitch.{weight,bias}        -> emb_pitch.{weight,bias}
//! enc_p.emb_g.{weight,bias}            -> emb_g.{weight,bias}        # multi-speaker
//! dec.emb_g.{weight,bias}              -> emb_g.{weight,bias}        # alt layout
//! dec.<rest>                           -> <rest>
//! ```
//!
//! Anything that doesn't match a known prefix is dropped with a
//! debug-level trace line so a future loader extension can pick up
//! checkpoint variants without breaking the existing path.

#![cfg(feature = "rvc")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};

use crate::error::VcError;

use super::content::RvcVersion;
use super::generator::{NsfHifiGan, NsfHifiGanConfig};
use super::retrieval::FeatureIndex;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Environment variable that overrides the per-voice profile directory.
/// Each voice lives at `$BLAZEN_RVC_VOICE_DIR/<voice_id>/`.
pub const VOICE_DIR_ENV: &str = "BLAZEN_RVC_VOICE_DIR";

/// File name of the pytorch generator state-dict inside a voice
/// directory.
pub const GENERATOR_FILENAME: &str = "model.pth";

/// File name of the bincode-serialised [`FeatureIndex`] inside a voice
/// directory.
pub const INDEX_FILENAME: &str = "index.bin";

/// File name of the optional metadata TOML inside a voice directory.
pub const META_FILENAME: &str = "meta.toml";

/// Default sample rate (Hz) used when `meta.toml` doesn't specify one.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 40_000;

// ---------------------------------------------------------------------------
// Voice profile
// ---------------------------------------------------------------------------

/// Fully-loaded RVC voice profile.
///
/// Returned by [`load_voice_profile`]. Holds the per-voice generator,
/// feature index, and metadata needed to drive a single conversion.
pub struct RvcVoiceProfile {
    /// NSF-HiFi-GAN generator with this voice's weights baked in.
    pub generator: NsfHifiGan,
    /// kNN retrieval index over this voice's training-time content
    /// features.
    pub index: FeatureIndex,
    /// Speaker-embedding index to use when invoking the generator.
    /// Single-speaker checkpoints use `0`.
    pub speaker_id: u32,
    /// RVC checkpoint family the profile was trained against.
    pub rvc_version: RvcVersion,
    /// Native sample rate of this voice (Hz).
    pub sample_rate_hz: u32,
}

impl std::fmt::Debug for RvcVoiceProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvcVoiceProfile")
            .field("speaker_id", &self.speaker_id)
            .field("rvc_version", &self.rvc_version)
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("index_len", &self.index.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Root directory holding per-voice profile sub-directories.
///
/// Resolution order:
///
/// 1. `$BLAZEN_RVC_VOICE_DIR` if set and non-empty.
/// 2. `$HOME/.cache/blazen/rvc/voices/` if `$HOME` is set.
/// 3. `./blazen-rvc-voices/` as a last-resort fallback (e.g. CI sandboxes
///    where `$HOME` is unset).
#[must_use]
pub fn voice_root_dir() -> PathBuf {
    if let Ok(env) = std::env::var(VOICE_DIR_ENV)
        && !env.is_empty()
    {
        return PathBuf::from(env);
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home)
            .join(".cache")
            .join("blazen")
            .join("rvc")
            .join("voices");
    }
    PathBuf::from("blazen-rvc-voices")
}

/// Directory for a single voice profile.
#[must_use]
pub fn voice_dir(voice_id: &str) -> PathBuf {
    voice_root_dir().join(voice_id)
}

/// Download an arbitrary file from a Hugging Face Hub repo to the
/// candle / hf-hub model cache and return the local path.
///
/// Mirrors the `spawn_blocking` pattern used by
/// `crates/blazen-audio-tts/src/backends/bark/weights.rs`.
///
/// # Errors
///
/// Returns [`VcError::ModelLoad`] when the hf-hub API can't be built,
/// when the file can't be fetched, or when `spawn_blocking` panics.
pub async fn hf_download(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<PathBuf, VcError> {
    let repo_owned = repo_id.to_owned();
    let file_owned = filename.to_owned();
    let revision_owned = revision.map(str::to_owned);

    tokio::task::spawn_blocking(move || -> Result<PathBuf, VcError> {
        let api = ApiBuilder::new()
            .build()
            .map_err(|e| VcError::ModelLoad(format!("rvc hf-hub api: {e}")))?;
        let repo = match revision_owned {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                repo_owned.clone(),
                hf_hub::RepoType::Model,
                rev,
            )),
            None => api.model(repo_owned.clone()),
        };
        repo.get(&file_owned).map_err(|e| {
            VcError::ModelLoad(format!("rvc hf-hub fetch {repo_owned}/{file_owned}: {e}"))
        })
    })
    .await
    .map_err(|e| VcError::ModelLoad(format!("rvc hf-hub spawn_blocking: {e}")))?
}

/// Load a voice profile from disk.
///
/// Looks up `$BLAZEN_RVC_VOICE_DIR/<voice_id>/` and reads the generator,
/// feature index, and (optional) metadata file. The generator is
/// instantiated with [`NsfHifiGanConfig::default_v2_40khz`] when no
/// metadata override is present.
///
/// # Errors
///
/// Returns [`VcError::VoiceNotFound`] when the voice directory or its
/// required files (`model.pth`, `index.bin`) are missing,
/// [`VcError::ModelLoad`] when the pytorch checkpoint or feature index
/// can't be parsed, and [`VcError::Io`] on filesystem failures while
/// reading `meta.toml`.
pub async fn load_voice_profile(
    voice_id: &str,
    device: &Device,
) -> Result<RvcVoiceProfile, VcError> {
    let dir = voice_dir(voice_id);
    if !dir.is_dir() {
        return Err(VcError::VoiceNotFound(format!(
            "{voice_id}: directory {} does not exist (see $BLAZEN_RVC_VOICE_DIR)",
            dir.display()
        )));
    }

    let model_path = dir.join(GENERATOR_FILENAME);
    let index_path = dir.join(INDEX_FILENAME);
    let meta_path = dir.join(META_FILENAME);

    if !model_path.is_file() {
        return Err(VcError::VoiceNotFound(format!(
            "{voice_id}: missing {GENERATOR_FILENAME} at {}",
            model_path.display()
        )));
    }
    if !index_path.is_file() {
        return Err(VcError::VoiceNotFound(format!(
            "{voice_id}: missing {INDEX_FILENAME} at {}",
            index_path.display()
        )));
    }

    // Read optional metadata first so the (heavy) weight load below
    // sees the correct config.
    let meta = if meta_path.is_file() {
        let raw = tokio::fs::read_to_string(&meta_path).await?;
        VoiceMeta::parse(&raw).map_err(|e| {
            VcError::ModelLoad(format!("{voice_id}: parse {}: {e}", meta_path.display()))
        })?
    } else {
        VoiceMeta::default()
    };

    // FeatureIndex::load is synchronous; do it on the blocking pool so
    // we don't stall the runtime on a large mmap.
    let index_path_owned = index_path.clone();
    let index = tokio::task::spawn_blocking(move || FeatureIndex::load(&index_path_owned))
        .await
        .map_err(|e| VcError::ModelLoad(format!("{voice_id}: index load join: {e}")))?
        .map_err(|e| {
            VcError::ModelLoad(format!("{voice_id}: load {}: {e}", index_path.display()))
        })?;

    // The pickle load + VarBuilder construction is CPU heavy; keep it
    // on `spawn_blocking` and clone the device handle into the closure.
    let device_clone = device.clone();
    let model_path_owned = model_path.clone();
    let meta_clone = meta.clone();
    let generator = tokio::task::spawn_blocking(move || -> Result<NsfHifiGan, VcError> {
        load_generator_from_pth(&model_path_owned, &device_clone, &meta_clone)
    })
    .await
    .map_err(|e| VcError::ModelLoad(format!("{voice_id}: generator load join: {e}")))??;

    Ok(RvcVoiceProfile {
        generator,
        index,
        speaker_id: meta.speaker_id,
        rvc_version: meta.rvc_version_enum(),
        sample_rate_hz: meta.sample_rate_hz,
    })
}

// ---------------------------------------------------------------------------
// Pytorch -> candle key remap + generator construction
// ---------------------------------------------------------------------------

/// Remap a single state-dict key from upstream RVC pytorch layout to
/// the [`VarBuilder`] paths that
/// [`super::generator::NsfHifiGan::load_from_var_builder`] reads.
///
/// Returns `None` for keys that don't belong to the generator (those are
/// silently dropped at load time -- e.g. encoder / discriminator tensors
/// that aren't needed for inference).
#[must_use]
pub fn remap_key(raw: &str) -> Option<String> {
    // Generator weights live under `dec.<rest>`.
    if let Some(rest) = raw.strip_prefix("dec.") {
        return Some(rest.to_owned());
    }
    // Conditioning embeddings can live either under `enc_p.` (the
    // standard RVC layout) or as siblings of `dec.` in some forks; in
    // both cases the generator's VarBuilder expects them at the root.
    if let Some(rest) = raw.strip_prefix("enc_p.") {
        match rest {
            // `emb_pitch` / `emb_g` are the only generator-side tensors
            // ever stored under `enc_p.`; everything else (`enc_p.fc`,
            // `enc_p.proj`, etc.) is unused at inference time.
            "emb_pitch.weight" | "emb_pitch.bias" | "emb_g.weight" | "emb_g.bias" => {
                Some(rest.to_owned())
            }
            _ => None,
        }
    } else {
        None
    }
}

/// Read `pth_path`, remap every tensor key through [`remap_key`], cast
/// to f32, move to `device`, and return a [`VarBuilder`] rooted at the
/// generator's namespace.
fn load_generator_from_pth(
    pth_path: &Path,
    device: &Device,
    meta: &VoiceMeta,
) -> Result<NsfHifiGan, VcError> {
    let pth = PthTensors::new(pth_path, None)
        .map_err(|e| VcError::ModelLoad(format!("rvc pth open {}: {e}", pth_path.display())))?;

    let mut remapped: HashMap<String, Tensor> = HashMap::new();
    let mut dropped = 0_usize;
    let raw_keys: Vec<String> = pth.tensor_infos().keys().cloned().collect();
    for raw_key in raw_keys {
        let Some(candle_key) = remap_key(&raw_key) else {
            dropped += 1;
            continue;
        };
        let tensor = pth
            .get(&raw_key)
            .map_err(|e| VcError::ModelLoad(format!("rvc pth load {raw_key}: {e}")))?
            .ok_or_else(|| VcError::ModelLoad(format!("rvc pth missing tensor {raw_key}")))?
            .to_device(device)
            .map_err(|e| VcError::ModelLoad(format!("rvc pth move-to-device {raw_key}: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| VcError::ModelLoad(format!("rvc pth cast {raw_key}: {e}")))?;
        remapped.insert(candle_key, tensor);
    }

    if remapped.is_empty() {
        return Err(VcError::ModelLoad(format!(
            "rvc pth {} contained no generator tensors after remap (dropped {dropped})",
            pth_path.display()
        )));
    }

    tracing::debug!(
        kept = remapped.len(),
        dropped,
        "rvc generator weights remapped"
    );

    let vb = VarBuilder::from_tensors(remapped, DType::F32, device);

    // Single-speaker checkpoints set sample_rate_hz=40000 by default;
    // we mirror that into the generator config so the source-side
    // SineGen runs at the right rate.
    let mut cfg = NsfHifiGanConfig::default_v2_40khz();
    cfg.output_sample_rate_hz = meta.sample_rate_hz;
    NsfHifiGan::load_from_var_builder(vb, device, cfg)
        .map_err(|e| VcError::ModelLoad(format!("rvc generator construct: {e}")))
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Parsed `meta.toml` contents.
///
/// `rvc_version` is stored as a string (`"v1"` or `"v2"`) on disk and
/// projected onto [`RvcVersion`] via [`Self::rvc_version_enum`]. The
/// string form keeps the on-disk surface stable while letting us evolve
/// the enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMeta {
    /// Speaker-embedding index to feed the generator (default `0`).
    #[serde(default)]
    pub speaker_id: u32,
    /// RVC checkpoint family (`"v1"` or `"v2"`, default `"v2"`).
    #[serde(default = "default_rvc_version")]
    pub rvc_version: String,
    /// Native output sample rate in Hz (default `40000`).
    #[serde(default = "default_sample_rate_hz")]
    pub sample_rate_hz: u32,
}

fn default_rvc_version() -> String {
    "v2".to_owned()
}

const fn default_sample_rate_hz() -> u32 {
    DEFAULT_SAMPLE_RATE_HZ
}

impl Default for VoiceMeta {
    fn default() -> Self {
        Self {
            speaker_id: 0,
            rvc_version: default_rvc_version(),
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
        }
    }
}

impl VoiceMeta {
    /// Project the on-disk version string onto the typed enum.
    /// Unknown / unparseable values fall back to [`RvcVersion::V2`].
    #[must_use]
    pub fn rvc_version_enum(&self) -> RvcVersion {
        match self.rvc_version.to_lowercase().as_str() {
            "v1" | "1" => RvcVersion::V1,
            _ => RvcVersion::V2,
        }
    }

    /// Parse a `meta.toml` body.
    ///
    /// We hand-roll a tiny key-value parser instead of pulling in the
    /// `toml` crate -- the schema is three known keys and lines look
    /// like `key = value` with optional quoting and `#` comments. This
    /// keeps the dep tree small and avoids a workspace addition.
    ///
    /// # Errors
    ///
    /// Returns a static error message on syntactic issues (e.g.
    /// non-integer where an integer is expected). Unknown keys are
    /// ignored so the file format can be extended without breaking
    /// older readers.
    pub fn parse(raw: &str) -> Result<Self, &'static str> {
        let mut meta = Self::default();
        for line in raw.lines() {
            let line = match line.find('#') {
                Some(ix) => &line[..ix],
                None => line,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let key = k.trim();
            let value = v.trim().trim_matches(|c| c == '"' || c == '\'');
            match key {
                "speaker_id" => {
                    meta.speaker_id = value.parse().map_err(|_| "speaker_id must be u32")?;
                }
                "rvc_version" => value.clone_into(&mut meta.rvc_version),
                "sample_rate_hz" => {
                    meta.sample_rate_hz =
                        value.parse().map_err(|_| "sample_rate_hz must be u32")?;
                }
                _ => {}
            }
        }
        Ok(meta)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(unsafe_code, reason = "tests mutate process env in a serialised block")]
mod tests {
    use super::*;

    #[test]
    fn remap_strips_dec_prefix() {
        assert_eq!(
            remap_key("dec.ups.0.weight").as_deref(),
            Some("ups.0.weight")
        );
        assert_eq!(
            remap_key("dec.resblocks.0.convs1.0.weight").as_deref(),
            Some("resblocks.0.convs1.0.weight")
        );
        assert_eq!(
            remap_key("dec.conv_pre.weight").as_deref(),
            Some("conv_pre.weight")
        );
        assert_eq!(
            remap_key("dec.conv_post.weight").as_deref(),
            Some("conv_post.weight")
        );
        assert_eq!(
            remap_key("dec.m_source.l_linear.weight").as_deref(),
            Some("m_source.l_linear.weight")
        );
        assert_eq!(
            remap_key("dec.noise_convs.0.weight").as_deref(),
            Some("noise_convs.0.weight")
        );
        assert_eq!(
            remap_key("dec.cond.0.weight").as_deref(),
            Some("cond.0.weight")
        );
    }

    #[test]
    fn remap_lifts_emb_pitch_and_emb_g_from_enc_p() {
        assert_eq!(
            remap_key("enc_p.emb_pitch.weight").as_deref(),
            Some("emb_pitch.weight")
        );
        assert_eq!(
            remap_key("enc_p.emb_g.weight").as_deref(),
            Some("emb_g.weight")
        );
    }

    #[test]
    fn remap_drops_non_generator_keys() {
        // Encoder body, discriminator, optimizer state -- all dropped.
        assert!(remap_key("enc_p.fc.weight").is_none());
        assert!(remap_key("enc_q.proj.weight").is_none());
        assert!(remap_key("disc.0.weight").is_none());
        assert!(remap_key("optimizer.state.0").is_none());
    }

    #[test]
    fn voice_meta_default_matches_v2_40khz() {
        let m = VoiceMeta::default();
        assert_eq!(m.speaker_id, 0);
        assert_eq!(m.rvc_version, "v2");
        assert_eq!(m.sample_rate_hz, DEFAULT_SAMPLE_RATE_HZ);
        assert_eq!(m.rvc_version_enum(), RvcVersion::V2);
    }

    #[test]
    fn voice_meta_parses_minimal_toml() {
        let raw = r#"
            # comment
            speaker_id = 3
            rvc_version = "v1"
            sample_rate_hz = 32000
        "#;
        let m = VoiceMeta::parse(raw).expect("parse");
        assert_eq!(m.speaker_id, 3);
        assert_eq!(m.rvc_version, "v1");
        assert_eq!(m.sample_rate_hz, 32_000);
        assert_eq!(m.rvc_version_enum(), RvcVersion::V1);
    }

    #[test]
    fn voice_meta_ignores_unknown_keys_and_blank_lines() {
        let raw = "\n\nspeaker_id = 7\nunknown = true\n";
        let m = VoiceMeta::parse(raw).expect("parse");
        assert_eq!(m.speaker_id, 7);
    }

    #[test]
    fn voice_meta_unknown_version_falls_back_to_v2() {
        let raw = "rvc_version = 'mystery'\n";
        let m = VoiceMeta::parse(raw).expect("parse");
        assert_eq!(m.rvc_version_enum(), RvcVersion::V2);
    }

    #[test]
    fn voice_meta_rejects_non_integer_speaker_id() {
        let raw = "speaker_id = \"three\"\n";
        let err = VoiceMeta::parse(raw).expect_err("should fail");
        assert!(err.contains("speaker_id"));
    }

    #[test]
    fn voice_root_dir_respects_env_override() {
        let prev = std::env::var(VOICE_DIR_ENV).ok();
        // SAFETY: tests touching process env are mutually exclusive
        // because we serialise on the env var via this single test.
        // SAFETY: setting / removing an environment variable is safe
        // here because we are not concurrently reading it from another
        // thread inside this test; cargo defaults to one-thread-per-test
        // for `#[test]` and we restore the previous value at the end.
        unsafe {
            std::env::set_var(VOICE_DIR_ENV, "/tmp/blazen-rvc-voice-dir-test-override");
        }
        let dir = voice_root_dir();
        assert_eq!(
            dir,
            PathBuf::from("/tmp/blazen-rvc-voice-dir-test-override")
        );
        // SAFETY: see above.
        unsafe {
            match prev {
                Some(v) => std::env::set_var(VOICE_DIR_ENV, v),
                None => std::env::remove_var(VOICE_DIR_ENV),
            }
        }
    }

    #[tokio::test]
    async fn load_voice_profile_missing_dir_returns_voice_not_found() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // SAFETY: per the env-override test above.
        unsafe {
            std::env::set_var(VOICE_DIR_ENV, tmp.path());
        }
        let device = Device::Cpu;
        let err = load_voice_profile("does-not-exist", &device)
            .await
            .expect_err("should fail");
        match err {
            VcError::VoiceNotFound(msg) => {
                assert!(msg.contains("does-not-exist"), "msg: {msg}");
            }
            other => panic!("expected VoiceNotFound, got {other:?}"),
        }
        // Best-effort cleanup. SAFETY as above.
        unsafe {
            std::env::remove_var(VOICE_DIR_ENV);
        }
    }
}
