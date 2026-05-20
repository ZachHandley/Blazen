//! Configuration options for the local TTS backend (any-tts).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Which underlying TTS model to load.
///
/// All variants map onto a `ModelType` understood by the `any-tts` crate
/// when the `engine` feature is enabled. The string form (`"kokoro82m"`,
/// `"vibevoice"`, …) is what gets serialised into JSON / IPC payloads so
/// that bindings can round-trip it without bringing the heavy any-tts
/// types onto the language boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsModel {
    /// Kokoro-82M (`StyleTTS2` + `ISTFTNet`). Small, CPU-friendly, default.
    #[default]
    Kokoro82m,
    /// `VibeVoice`-1.5B (Microsoft) — multi-speaker diffusion-based TTS.
    VibeVoice,
    /// Qwen3-TTS-12Hz-1.7B (`CustomVoice` variant) — multi-codebook LM.
    Qwen3Tts,
}

impl TtsModel {
    /// The default voice / speaker name for this model.
    ///
    /// For Kokoro-82M the upstream voice preset library is keyed by short
    /// codes like `"af_bella"` (American Female "Bella"); we ship that as
    /// the default because it is the most widely-cited example voice.
    /// `VibeVoice` and Qwen3-TTS default voices are TBD until we wire live
    /// smoke tests — fall through to `None` and let any-tts pick its own
    /// default in that case.
    #[must_use]
    pub fn default_voice(self) -> Option<&'static str> {
        match self {
            // TODO: verify exact voice string once `any-tts` Kokoro voice
            //       catalogue is queried at runtime; fall back to first
            //       available voice if `af_bella` is not present.
            Self::Kokoro82m => Some("af_bella"),
            Self::VibeVoice | Self::Qwen3Tts => None,
        }
    }

    /// Short string used for log / error contexts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kokoro82m => "kokoro",
            Self::VibeVoice => "vibevoice",
            Self::Qwen3Tts => "qwen3_tts",
        }
    }
}

/// Options for constructing a [`TtsProvider`](crate::TtsProvider).
///
/// All fields are optional and have sensible defaults (Kokoro-82M with
/// the `af_bella` preset voice and the model's native sample rate).
///
/// # Examples
///
/// ```
/// use blazen_audio_tts::{TtsModel, TtsOptions};
///
/// // Use defaults (Kokoro-82M, af_bella, native sample rate)
/// let opts = TtsOptions::default();
/// assert_eq!(opts.model, Some(TtsModel::Kokoro82m));
///
/// // Override specific fields
/// let opts = TtsOptions {
///     model: Some(TtsModel::VibeVoice),
///     voice: Some("alloy".into()),
///     ..TtsOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsOptions {
    /// Which underlying TTS model to load. Defaults to Kokoro-82M.
    pub model: Option<TtsModel>,

    /// Voice / speaker preset name (e.g. `"af_bella"` for Kokoro).
    ///
    /// When `None`, falls back to [`TtsModel::default_voice`] for the
    /// chosen model.
    pub voice: Option<String>,

    /// Language ISO 639-1 code (e.g. `"en"`, `"ja"`). When `None`,
    /// any-tts auto-detects from the input text.
    pub language: Option<String>,

    /// Override the output sample rate in Hz.
    ///
    /// When `None`, the model's native sample rate (24 kHz for Kokoro)
    /// is used.
    pub sample_rate: Option<u32>,

    /// Path to cache downloaded model weights.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            model: Some(TtsModel::default()),
            voice: None,
            language: None,
            sample_rate: None,
            cache_dir: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_pick_kokoro() {
        let opts = TtsOptions::default();
        assert_eq!(opts.model, Some(TtsModel::Kokoro82m));
        assert!(opts.voice.is_none());
        assert!(opts.language.is_none());
        assert!(opts.sample_rate.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = TtsOptions {
            model: Some(TtsModel::VibeVoice),
            voice: Some("alloy".into()),
            ..TtsOptions::default()
        };
        assert_eq!(opts.model, Some(TtsModel::VibeVoice));
        assert_eq!(opts.voice.as_deref(), Some("alloy"));
        assert!(opts.language.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = TtsOptions {
            model: Some(TtsModel::Kokoro82m),
            voice: Some("af_bella".into()),
            language: Some("en".into()),
            sample_rate: Some(24_000),
            cache_dir: Some(PathBuf::from("/var/cache/tts")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: TtsOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model, Some(TtsModel::Kokoro82m));
        assert_eq!(parsed.voice.as_deref(), Some("af_bella"));
        assert_eq!(parsed.language.as_deref(), Some("en"));
        assert_eq!(parsed.sample_rate, Some(24_000));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/var/cache/tts"))
        );
    }

    #[test]
    fn model_serializes_snake_case() {
        let s = serde_json::to_string(&TtsModel::Kokoro82m).unwrap();
        assert_eq!(s, "\"kokoro82m\"");
        let s = serde_json::to_string(&TtsModel::Qwen3Tts).unwrap();
        assert_eq!(s, "\"qwen3_tts\"");
    }

    #[test]
    fn kokoro_default_voice_is_af_bella() {
        assert_eq!(TtsModel::Kokoro82m.default_voice(), Some("af_bella"));
    }
}
