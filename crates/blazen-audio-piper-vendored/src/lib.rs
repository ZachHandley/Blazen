//! Vendored fork of `piper-rs` v0.1.9 (upstream:
//! <https://github.com/thewh1teagle/piper-rs>), patched for Blazen.
//!
//! One upstream design choice was incompatible with the Blazen workspace:
//!
//! **Phonemizer license**: upstream links `espeak-rs-sys`, which
//! `cc`-compiles ~30 MB of eSpeak NG C sources into the resulting
//! library. eSpeak NG is GPL-3.0+; Blazen ships under MPL-2.0, which
//! is link-time incompatible with GPL. The fork drops the
//! `espeak-rs` dependency entirely and instead invokes the system
//! `espeak-ng` binary via a subprocess. Subprocess invocation is
//! process-boundary isolation — no GPL inheritance.
//!
//! The inference engine is `ort` (ONNX Runtime), same as upstream. An
//! earlier revision of this fork swapped `ort` for the workspace's
//! `tract-onnx` to converge on one inference backend; that swap is
//! reverted because tract cannot statically analyse Piper VITS graphs
//! at all (see `model.rs` module docs for the exact failures on tract
//! 0.22 and 0.23). `ort` 2.0.0-rc.12 is already a workspace dependency
//! (`blazen-audio-stt`'s `vad-ort`), and is target-gated out of
//! `x86_64-apple-darwin` the same way (no Intel-mac prebuilt) — on that
//! triple [`Piper::new`] returns [`PiperError::Unsupported`].
//!
//! Aside from those swaps the public API (`Piper::new`, `Piper::create`,
//! `ModelConfig`) is bit-for-bit compatible with upstream. See
//! `VENDORED.md` next to this crate for the full patch list and upgrade
//! procedure.
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

pub use model::ModelConfig;

/// Re-export of the ONNX Runtime crate, so callers building their own
/// session for [`Piper::from_session`] can name its types at the exact
/// version this crate links.
#[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
pub use ort;

use model::Engine;
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

    /// ONNX Runtime failed to load or run the voice graph.
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

    /// Piper synthesis has no inference engine on this build target
    /// (`x86_64-apple-darwin`: ONNX Runtime ships no Intel-mac prebuilt).
    #[error("piper TTS unsupported on this target: {0}")]
    Unsupported(String),
}

/// Convenience alias.
pub type PiperResult<T> = Result<T, PiperError>;

/// A loaded Piper voice. Holds an ONNX Runtime session plus the parsed config.
pub struct Piper {
    config: ModelConfig,
    engine: Engine,
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
    /// - [`PiperError::InferenceError`] if ONNX Runtime cannot load the graph.
    /// - [`PiperError::EspeakNgMissing`] if `espeak-ng` is not on `PATH`.
    /// - [`PiperError::Unsupported`] on `x86_64-apple-darwin` (no engine).
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

        let engine = Engine::load(model_path)?;

        Ok(Self {
            config,
            engine,
            espeak_ng_binary,
        })
    }

    /// Construct a `Piper` from an already-built ONNX Runtime session +
    /// parsed config.
    ///
    /// Mirrors upstream's `from_session` for callers that load the ONNX
    /// graph through their own code path (custom IO, model bundling, …).
    /// Not available on `x86_64-apple-darwin` (no ONNX Runtime prebuilt).
    ///
    /// # Errors
    ///
    /// [`PiperError::EspeakNgMissing`] if `espeak-ng` is not on `PATH`.
    #[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
    pub fn from_session(session: ort::session::Session, config: ModelConfig) -> PiperResult<Self> {
        let espeak_ng_binary = phonemize::locate_espeak_ng()?;
        Ok(Self {
            config,
            engine: Engine::from_session(session),
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
        let samples = self.engine.infer(
            &self.config,
            &phonemes,
            noise_scale.unwrap_or(inf.noise_scale),
            length_scale.unwrap_or(inf.length_scale),
            noise_w.unwrap_or(inf.noise_w),
            speaker_id.unwrap_or(0),
        )?;

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
        // Either we get EspeakNgMissing (no espeak-ng on PATH),
        // FailedToLoadResource (espeak-ng present, but model missing), or
        // Unsupported (x86_64-apple-darwin: no engine on that triple).
        // All are acceptable as proof of pre-flight wiring.
        match result {
            Err(
                PiperError::EspeakNgMissing(_)
                | PiperError::FailedToLoadResource(_)
                | PiperError::Unsupported(_),
            ) => {}
            Err(e) => {
                panic!("expected EspeakNgMissing, FailedToLoadResource, or Unsupported, got {e:?}")
            }
            Ok(_) => panic!("expected an error from Piper::new with missing model"),
        }
    }
}
