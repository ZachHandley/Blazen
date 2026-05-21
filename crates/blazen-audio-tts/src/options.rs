//! Shared configuration options for any TTS backend in `blazen-audio-tts`.
//!
//! Each backend treats unsupported fields as silently ignored — passing
//! [`TtsOptions::speed`] to an `AnyTtsBackend` configured for Kokoro is
//! a no-op rather than an error, mirroring the same fail-soft pattern
//! the `OpenAI` request DTO uses.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Which underlying local TTS model to load when using the `anytts` backend.
///
/// All variants map onto a `ModelType` understood by the `any-tts` crate
/// when the `anytts` feature is enabled. The string form (`"kokoro82m"`,
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
            Self::Kokoro82m => Some("af_bella"),
            Self::VibeVoice | Self::Qwen3Tts => None,
        }
    }

    /// Short string used for log / error contexts and backend ids.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kokoro82m => "kokoro",
            Self::VibeVoice => "vibevoice",
            Self::Qwen3Tts => "qwen3_tts",
        }
    }
}

/// Per-call options for [`TtsBackend::synthesize`](crate::TtsBackend::synthesize).
///
/// All fields are optional. Backends that don't understand a particular
/// override (e.g. the local `AnyTtsBackend` does not honor
/// [`speed`](Self::speed)) ignore it silently — this lets callers pass
/// the same `TtsOptions` to any backend without compile-time gating.
///
/// # Examples
///
/// ```
/// use blazen_audio_tts::{TtsModel, TtsOptions};
///
/// // Use defaults (Kokoro-82M, af_bella, native sample rate).
/// let opts = TtsOptions::default();
/// assert_eq!(opts.model, Some(TtsModel::Kokoro82m));
///
/// // Override fields with struct-update syntax.
/// let opts = TtsOptions {
///     model: Some(TtsModel::VibeVoice),
///     voice: Some("alloy".into()),
///     ..TtsOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsOptions {
    /// Which `any-tts` model to load (only honored by the `anytts` backend).
    ///
    /// Defaults to Kokoro-82M.
    pub model: Option<TtsModel>,

    /// Free-form remote model identifier — honored by the `openai`
    /// backend (e.g. `"tts-1"`, `"tts-1-hd"`). `None` falls back to the
    /// backend's configured default model.
    pub model_id: Option<String>,

    /// Voice / speaker preset name (e.g. `"af_bella"` for Kokoro,
    /// `"alloy"` for `OpenAI`).
    ///
    /// When `None`, each backend falls back to its own default.
    pub voice: Option<String>,

    /// Language ISO 639-1 code (e.g. `"en"`, `"ja"`). When `None`, the
    /// underlying engine auto-detects from the input text.
    pub language: Option<String>,

    /// Override the output sample rate in Hz.
    ///
    /// When `None`, each backend picks its own native rate (24 kHz for
    /// Kokoro, 24 kHz for `OpenAI` MP3).
    pub sample_rate: Option<u32>,

    /// Speech speed multiplier (`1.0` = normal). Honored by the
    /// `openai` backend.
    pub speed: Option<f32>,

    /// Output container — `"mp3"`, `"wav"`, `"flac"`, `"opus"`, `"pcm"`.
    /// Honored by the `openai` backend; the `anytts` backend always
    /// returns WAV.
    pub response_format: Option<String>,

    /// Path to cache downloaded model weights for the local
    /// `anytts` backend.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            model: Some(TtsModel::default()),
            model_id: None,
            voice: None,
            language: None,
            sample_rate: None,
            speed: None,
            response_format: None,
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
        assert!(opts.speed.is_none());
        assert!(opts.response_format.is_none());
        assert!(opts.cache_dir.is_none());
        assert!(opts.model_id.is_none());
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
            model_id: Some("tts-1-hd".into()),
            voice: Some("af_bella".into()),
            language: Some("en".into()),
            sample_rate: Some(24_000),
            speed: Some(1.25),
            response_format: Some("wav".into()),
            cache_dir: Some(PathBuf::from("/var/cache/tts")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: TtsOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model, Some(TtsModel::Kokoro82m));
        assert_eq!(parsed.model_id.as_deref(), Some("tts-1-hd"));
        assert_eq!(parsed.voice.as_deref(), Some("af_bella"));
        assert_eq!(parsed.language.as_deref(), Some("en"));
        assert_eq!(parsed.sample_rate, Some(24_000));
        assert_eq!(parsed.speed, Some(1.25));
        assert_eq!(parsed.response_format.as_deref(), Some("wav"));
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
