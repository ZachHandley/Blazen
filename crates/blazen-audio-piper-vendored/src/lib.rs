//! Vendored fork of `piper-rs` v0.1.9 (upstream:
//! <https://github.com/thewh1teagle/piper-rs>), patched for Blazen.
//!
//! Two upstream design choices were incompatible with the Blazen workspace:
//!
//! 1. **ONNX runtime**: upstream depends on `ort = 2.0.0-rc.11`, which
//!    would ship a second ONNX runtime alongside the workspace standard
//!    `tract-onnx` 0.21 (also used by `blazen-embed-tract`). The fork
//!    swaps `ort::session::Session` for `tract_onnx::prelude::*` so the
//!    entire workspace converges on one inference backend.
//!
//! 2. **Phonemizer license**: upstream links `espeak-rs-sys`, which
//!    `cc`-compiles ~30 MB of eSpeak NG C sources into the resulting
//!    library. eSpeak NG is GPL-3.0+; Blazen ships under MPL-2.0, which
//!    is link-time incompatible with GPL. The fork drops the
//!    `espeak-rs` dependency entirely and instead invokes the system
//!    `espeak-ng` binary via `tokio::process::Command`. Subprocess
//!    invocation is process-boundary isolation — no GPL inheritance.
//!
//! Aside from those two surgical swaps the public API (`Piper::new`,
//! `Piper::create`, `ModelConfig`) is bit-for-bit compatible with
//! upstream. See `VENDORED.md` next to this crate for the full patch
//! list and upgrade procedure.
//!
//! # Runtime requirement
//!
//! Phonemization is delegated to the system `espeak-ng` binary. It MUST
//! be on `PATH` when `Piper::new` is called. If missing, construction
//! returns [`PiperError::EspeakNgMissing`] with an install hint.
//! Pre-phonemized input (`is_phonemes = true` on `Piper::create`) still
//! uses the binary only at load time.

#![deny(missing_docs)]
// pedantic-clippy nits that aren't worth churning the vendored layout for.
#![allow(
    clippy::module_name_repetitions,
    clippy::type_complexity,
    clippy::items_after_statements,
    clippy::unnecessary_wraps
)]

mod model;
mod phonemize;

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tract_onnx::prelude::{
    Framework, InferenceModelExt, SimplePlan, TypedFact, TypedModel, TypedOp,
};

pub use model::ModelConfig;
pub use model::TractPiperModel;

use model::infer;
use phonemize::text_to_phonemes_blocking;

/// Errors returned by the vendored piper backend.
#[derive(Debug, thiserror::Error)]
pub enum PiperError {
    /// Failed to load a voice model file (`.onnx`) or its sidecar config (`.onnx.json`).
    #[error("failed to load resource: {0}")]
    FailedToLoadResource(String),

    /// `espeak-ng` returned a non-zero exit code, failed to produce IPA
    /// output, or otherwise misbehaved.
    #[error("phonemization error: {0}")]
    PhonemizationError(String),

    /// Tract failed to parse, optimize, or run the ONNX graph.
    #[error("inference error: {0}")]
    InferenceError(String),

    /// The system `espeak-ng` binary is not on `PATH`. Install it:
    ///
    /// - Debian/Ubuntu: `apt install espeak-ng`
    /// - macOS:         `brew install espeak-ng`
    /// - Arch:          `pacman -S espeak-ng`
    #[error(
        "espeak-ng binary not found on PATH: {0}\n  install: apt install espeak-ng / brew install espeak-ng / pacman -S espeak-ng"
    )]
    EspeakNgMissing(String),
}

/// Convenience alias.
pub type PiperResult<T> = Result<T, PiperError>;

/// A loaded Piper voice. Holds a tract-optimized plan plus the parsed config.
pub struct Piper {
    config: ModelConfig,
    model: Arc<Mutex<SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>>>,
    espeak_ng_binary: PathBuf,
}

impl Piper {
    /// Load a Piper voice from on-disk ONNX + config files.
    ///
    /// Auto-locates the `espeak-ng` binary on `PATH`. Pre-flighting that
    /// here turns a synthesis-time error into a load-time error, which
    /// is friendlier to callers that want to validate setup eagerly.
    ///
    /// # Errors
    ///
    /// - [`PiperError::FailedToLoadResource`] if either file fails to open / parse.
    /// - [`PiperError::InferenceError`] if tract cannot parse / optimize the ONNX graph.
    /// - [`PiperError::EspeakNgMissing`] if `espeak-ng` is not on `PATH`.
    pub fn new(model_path: &Path, config_path: &Path) -> PiperResult<Self> {
        // Pre-flight: locate espeak-ng before we do any heavy work.
        let espeak_ng_binary = phonemize::locate_espeak_ng()?;

        let file = File::open(config_path).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "failed to open config `{}`: {e}",
                config_path.display()
            ))
        })?;
        let config: ModelConfig = serde_json::from_reader(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!("failed to parse config: {e}"))
        })?;

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "tract onnx parse failed for `{}`: {e}",
                    model_path.display()
                ))
            })?
            .into_optimized()
            .map_err(|e| PiperError::InferenceError(format!("tract optimize failed: {e}")))?
            .into_runnable()
            .map_err(|e| PiperError::InferenceError(format!("tract runnable failed: {e}")))?;

        Ok(Self {
            config,
            model: Arc::new(Mutex::new(model)),
            espeak_ng_binary,
        })
    }

    /// Construct a `Piper` from an already-loaded tract plan + parsed config.
    ///
    /// Mirrors upstream's `from_session` for callers that load the ONNX
    /// graph through their own code path (custom IO, model bundling, …).
    ///
    /// # Errors
    ///
    /// [`PiperError::EspeakNgMissing`] if `espeak-ng` is not on `PATH`.
    pub fn from_model(model: TractPiperModel, config: ModelConfig) -> PiperResult<Self> {
        let espeak_ng_binary = phonemize::locate_espeak_ng()?;
        Ok(Self {
            config,
            model: Arc::new(Mutex::new(model)),
            espeak_ng_binary,
        })
    }

    /// Synthesize speech from `text` (or pre-computed IPA phonemes).
    ///
    /// Returns `(samples, sample_rate)` where `samples` are f32 PCM in `[-1.0, 1.0]`.
    ///
    /// All overrides honor the upstream semantics: `speaker_id`,
    /// `length_scale`, `noise_scale`, and `noise_w` default to the values
    /// baked into the voice config when `None`.
    ///
    /// # Errors
    ///
    /// See [`PiperError`].
    pub fn create(
        &self,
        text: &str,
        is_phonemes: bool,
        speaker_id: Option<i64>,
        length_scale: Option<f32>,
        noise_scale: Option<f32>,
        noise_w: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes = if is_phonemes {
            text.to_string()
        } else {
            text_to_phonemes_blocking(&self.espeak_ng_binary, text, &self.config.espeak.voice)?
        };

        let inf = &self.config.inference;
        let samples = {
            let mut locked = self
                .model
                .lock()
                .map_err(|e| PiperError::InferenceError(format!("model mutex poisoned: {e}")))?;
            infer(
                &mut locked,
                &self.config,
                &phonemes,
                noise_scale.unwrap_or(inf.noise_scale),
                length_scale.unwrap_or(inf.length_scale),
                noise_w.unwrap_or(inf.noise_w),
                speaker_id.unwrap_or(0),
            )?
        };

        Ok((samples, self.config.audio.sample_rate))
    }

    /// Returns the speaker name→id map, or `None` for single-speaker voices.
    #[must_use]
    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        if self.config.speaker_id_map.is_empty() {
            None
        } else {
            Some(&self.config.speaker_id_map)
        }
    }

    /// Sample rate baked into the voice config (Hz).
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.config.audio.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_install_hint() {
        let err = PiperError::EspeakNgMissing("not found".into());
        let msg = err.to_string();
        assert!(msg.contains("apt install espeak-ng"));
        assert!(msg.contains("brew install espeak-ng"));
    }

    #[test]
    fn missing_model_file_is_resource_or_espeak_error() {
        let result = Piper::new(
            Path::new("/nonexistent/path/voice.onnx"),
            Path::new("/nonexistent/path/voice.onnx.json"),
        );
        // Either we get EspeakNgMissing (no espeak-ng on PATH) or
        // FailedToLoadResource (espeak-ng present, but model missing).
        // Both are acceptable as proof of pre-flight wiring.
        match result {
            Err(PiperError::EspeakNgMissing(_) | PiperError::FailedToLoadResource(_)) => {}
            Err(e) => panic!("expected EspeakNgMissing or FailedToLoadResource, got {e:?}"),
            Ok(_) => panic!("expected an error from Piper::new with missing model"),
        }
    }
}
