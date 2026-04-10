//! Configuration options for the Piper local TTS backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`PiperProvider`](crate::PiperProvider).
///
/// All fields are optional and have sensible defaults.
///
/// # Examples
///
/// ```
/// use blazen_audio_piper::PiperOptions;
///
/// // Use defaults
/// let opts = PiperOptions::default();
/// assert!(opts.model_id.is_none());
///
/// // Override specific fields
/// let opts = PiperOptions {
///     model_id: Some("en_US-amy-medium".into()),
///     sample_rate: Some(22050),
///     ..PiperOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PiperOptions {
    /// Piper voice model identifier (e.g. `"en_US-amy-medium"`).
    ///
    /// This maps to a model file in the Piper voice repository. When `None`,
    /// must be set before synthesis can run.
    pub model_id: Option<String>,

    /// Speaker ID for multi-speaker models.
    ///
    /// When `None`, the default speaker (0) is used.
    pub speaker_id: Option<u32>,

    /// Output audio sample rate in Hz (e.g. `22050`, `16000`).
    ///
    /// When `None`, the model's native sample rate is used.
    pub sample_rate: Option<u32>,

    /// Path to cache downloaded voice models.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_all_none() {
        let opts = PiperOptions::default();
        assert!(opts.model_id.is_none());
        assert!(opts.speaker_id.is_none());
        assert!(opts.sample_rate.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = PiperOptions {
            model_id: Some("en_US-amy-medium".into()),
            sample_rate: Some(22050),
            ..PiperOptions::default()
        };
        assert_eq!(opts.model_id.as_deref(), Some("en_US-amy-medium"));
        assert_eq!(opts.sample_rate, Some(22050));
        assert!(opts.speaker_id.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let opts = PiperOptions {
            model_id: Some("en_US-amy-medium".into()),
            speaker_id: Some(3),
            sample_rate: Some(22050),
            cache_dir: Some(PathBuf::from("/tmp/piper-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: PiperOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_id.as_deref(), Some("en_US-amy-medium"));
        assert_eq!(parsed.speaker_id, Some(3));
        assert_eq!(parsed.sample_rate, Some(22050));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/piper-cache"))
        );
    }
}
