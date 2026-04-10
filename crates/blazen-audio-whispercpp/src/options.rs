//! Configuration options for the whisper.cpp local speech-to-text backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`WhisperCppProvider`](crate::WhisperCppProvider).
///
/// All fields are optional and have sensible defaults. The model size
/// defaults to [`WhisperModel::Small`].
///
/// # Examples
///
/// ```
/// use blazen_audio_whispercpp::{WhisperOptions, WhisperModel};
///
/// // Use defaults (Small model, CPU, auto-detect language)
/// let opts = WhisperOptions::default();
/// assert_eq!(opts.model, WhisperModel::Small);
///
/// // Override specific fields
/// let opts = WhisperOptions {
///     model: WhisperModel::LargeV3,
///     language: Some("en".into()),
///     ..WhisperOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhisperOptions {
    /// Which Whisper model size to use. Defaults to [`WhisperModel::Small`].
    ///
    /// Larger models are more accurate but require more memory and are slower.
    #[serde(default)]
    pub model: WhisperModel,

    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Accepts the same format strings as `blazen_llm::Device::parse`:
    /// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`.
    ///
    /// When `None`, defaults to `"cpu"`.
    pub device: Option<String>,

    /// ISO 639-1 language code (e.g. `"en"`, `"es"`, `"ja"`).
    ///
    /// When `None`, Whisper will auto-detect the spoken language.
    pub language: Option<String>,

    /// Enable speaker diarization (who spoke when).
    ///
    /// When `None` or `false`, diarization is disabled.
    pub diarize: Option<bool>,

    /// Path to cache downloaded models.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

/// The Whisper model size variants available through `HuggingFace`.
///
/// Each variant maps to a specific `HuggingFace` repository containing the
/// GGML-format model weights. Larger models produce more accurate
/// transcriptions but require more memory and compute time.
///
/// | Model    | Parameters | English-only WER | Multilingual WER | RAM  |
/// |----------|-----------|-------------------|------------------|------|
/// | Tiny     | 39M       | ~8%               | ~12%             | ~1GB |
/// | Base     | 74M       | ~6%               | ~10%             | ~1GB |
/// | Small    | 244M      | ~4%               | ~7%              | ~2GB |
/// | Medium   | 769M      | ~3%               | ~5%              | ~5GB |
/// | LargeV3  | 1.5B      | ~2%               | ~3%              | ~10GB|
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum WhisperModel {
    /// Fastest, lowest accuracy (~39M parameters).
    Tiny,
    /// Fast, moderate accuracy (~74M parameters).
    Base,
    /// Good balance of speed and accuracy (~244M parameters).
    #[default]
    Small,
    /// High accuracy, slower (~769M parameters).
    Medium,
    /// Highest accuracy, most resource-intensive (~1.5B parameters).
    LargeV3,
}

impl WhisperModel {
    /// Returns the `HuggingFace` model repository ID for this model size.
    ///
    /// These point to the GGML-format model files hosted on `HuggingFace`,
    /// suitable for use with `whisper.cpp` / `whisper-rs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use blazen_audio_whispercpp::WhisperModel;
    ///
    /// assert_eq!(
    ///     WhisperModel::Small.as_model_id(),
    ///     "ggerganov/whisper.cpp"
    /// );
    /// ```
    #[must_use]
    pub const fn as_model_id(&self) -> &'static str {
        // All model sizes are hosted in the same HuggingFace repository;
        // the specific file (ggml-tiny.bin, ggml-base.bin, etc.) is selected
        // by `as_ggml_filename()`.
        "ggerganov/whisper.cpp"
    }

    /// Returns the GGML model filename for this model size.
    ///
    /// These filenames correspond to the files in the
    /// `ggerganov/whisper.cpp` `HuggingFace` repository.
    ///
    /// # Examples
    ///
    /// ```
    /// use blazen_audio_whispercpp::WhisperModel;
    ///
    /// assert_eq!(WhisperModel::Tiny.as_ggml_filename(), "ggml-tiny.bin");
    /// assert_eq!(WhisperModel::LargeV3.as_ggml_filename(), "ggml-large-v3.bin");
    /// ```
    #[must_use]
    pub const fn as_ggml_filename(&self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
            Self::LargeV3 => "ggml-large-v3.bin",
        }
    }
}

impl std::fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Tiny => "tiny",
            Self::Base => "base",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::LargeV3 => "large-v3",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_small() {
        assert_eq!(WhisperModel::default(), WhisperModel::Small);
    }

    #[test]
    fn default_options_uses_small_model() {
        let opts = WhisperOptions::default();
        assert_eq!(opts.model, WhisperModel::Small);
        assert!(opts.device.is_none());
        assert!(opts.language.is_none());
        assert!(opts.diarize.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = WhisperOptions {
            model: WhisperModel::LargeV3,
            language: Some("en".into()),
            ..WhisperOptions::default()
        };
        assert_eq!(opts.model, WhisperModel::LargeV3);
        assert_eq!(opts.language.as_deref(), Some("en"));
        assert!(opts.device.is_none());
    }

    #[test]
    fn model_id_is_consistent() {
        // All models come from the same HuggingFace repo
        let repo = "ggerganov/whisper.cpp";
        assert_eq!(WhisperModel::Tiny.as_model_id(), repo);
        assert_eq!(WhisperModel::Base.as_model_id(), repo);
        assert_eq!(WhisperModel::Small.as_model_id(), repo);
        assert_eq!(WhisperModel::Medium.as_model_id(), repo);
        assert_eq!(WhisperModel::LargeV3.as_model_id(), repo);
    }

    #[test]
    fn ggml_filenames_are_correct() {
        assert_eq!(WhisperModel::Tiny.as_ggml_filename(), "ggml-tiny.bin");
        assert_eq!(WhisperModel::Base.as_ggml_filename(), "ggml-base.bin");
        assert_eq!(WhisperModel::Small.as_ggml_filename(), "ggml-small.bin");
        assert_eq!(WhisperModel::Medium.as_ggml_filename(), "ggml-medium.bin");
        assert_eq!(
            WhisperModel::LargeV3.as_ggml_filename(),
            "ggml-large-v3.bin"
        );
    }

    #[test]
    fn display_impl() {
        assert_eq!(WhisperModel::Tiny.to_string(), "tiny");
        assert_eq!(WhisperModel::Base.to_string(), "base");
        assert_eq!(WhisperModel::Small.to_string(), "small");
        assert_eq!(WhisperModel::Medium.to_string(), "medium");
        assert_eq!(WhisperModel::LargeV3.to_string(), "large-v3");
    }

    #[test]
    fn serde_roundtrip_options() {
        let opts = WhisperOptions {
            model: WhisperModel::Medium,
            device: Some("cuda:0".into()),
            language: Some("ja".into()),
            diarize: Some(true),
            cache_dir: Some(PathBuf::from("/tmp/whisper-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: WhisperOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model, WhisperModel::Medium);
        assert_eq!(parsed.device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.language.as_deref(), Some("ja"));
        assert_eq!(parsed.diarize, Some(true));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/whisper-cache"))
        );
    }

    #[test]
    fn serde_roundtrip_model_enum() {
        for model in [
            WhisperModel::Tiny,
            WhisperModel::Base,
            WhisperModel::Small,
            WhisperModel::Medium,
            WhisperModel::LargeV3,
        ] {
            let json = serde_json::to_string(&model).expect("serialize");
            let parsed: WhisperModel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, model);
        }
    }
}
