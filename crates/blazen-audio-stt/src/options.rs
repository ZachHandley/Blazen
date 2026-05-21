//! Shared options for [`SttBackend`](crate::SttBackend) implementations.
//!
//! Backend-specific option structs (e.g. `WhisperCppOptions`) live alongside
//! their backends in [`crate::backends`] and typically embed an
//! [`SttOptions`] for the cross-backend fields.

use serde::{Deserialize, Serialize};

/// Shared transcription-time options that every STT backend honors when
/// applicable.
///
/// Backends that do not support a given option (e.g. a streaming-only
/// backend ignoring `sample_rate`) should document the deviation in their
/// own option type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SttOptions {
    /// Identifier of the model the backend should load.
    ///
    /// For local engines this is typically a `HuggingFace` repo ID or a path;
    /// for hosted engines it is the provider's model name (e.g.
    /// `"whisper-1"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,

    /// Optional ISO 639-1 language hint (e.g. `"en"`, `"ja"`).
    ///
    /// When `None`, the backend is free to auto-detect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Expected audio sample rate in hertz.
    ///
    /// Most modern STT engines (Whisper family) expect `16_000`. Backends
    /// that resample internally may ignore this hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,

    /// Hardware device specifier (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Honored by backends with platform-specific acceleration; ignored by
    /// hosted backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,

    /// Request speaker diarization output if the backend supports it.
    ///
    /// Backends without diarization should ignore the hint or return
    /// [`crate::SttError::Unsupported`] when set to `Some(true)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diarize: Option<bool>,
}

impl SttOptions {
    /// Convenience constructor for an empty / default options struct.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_all_none() {
        let opts = SttOptions::new();
        assert!(opts.model_id.is_none());
        assert!(opts.language.is_none());
        assert!(opts.sample_rate.is_none());
        assert!(opts.device.is_none());
        assert!(opts.diarize.is_none());
    }

    #[test]
    fn serde_roundtrip_minimal_omits_none_fields() {
        let opts = SttOptions::new();
        let json = serde_json::to_string(&opts).expect("serialize SttOptions");
        assert_eq!(json, "{}");
    }

    #[test]
    fn serde_roundtrip_populated() {
        let opts = SttOptions {
            model_id: Some("base".into()),
            language: Some("en".into()),
            sample_rate: Some(16_000),
            device: Some("cpu".into()),
            diarize: Some(false),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let back: SttOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_id.as_deref(), Some("base"));
        assert_eq!(back.language.as_deref(), Some("en"));
        assert_eq!(back.sample_rate, Some(16_000));
        assert_eq!(back.device.as_deref(), Some("cpu"));
        assert_eq!(back.diarize, Some(false));
    }
}
