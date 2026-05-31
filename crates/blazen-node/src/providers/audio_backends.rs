//! Node bindings for the audio backend handle + config surface re-exported
//! through `blazen-llm`.
//!
//! These wrap the concrete native inference backends and their typed
//! [`*BackendHandle`] wrappers as napi classes / objects so the same names
//! the Rust surface exposes (`FasterWhisperBackend`, `SttBackendHandle`,
//! `SparkTtsBackend`, `TtsBackendHandle`, `MusicBackendHandle`,
//! `CodecBackendHandle`, plus the `FasterWhisperConfig` / `SparkTtsConfig`
//! records) are reachable from JavaScript.
//!
//! Each surface is gated behind the Cargo feature that pulls in its concrete
//! backend, so a build without the backend feature simply omits the binding
//! rather than failing to compile. Weights are loaded lazily on first use —
//! constructing a backend or handle performs no I/O.

#[cfg(any(
    feature = "audio-stt-faster-whisper",
    feature = "audio-tts-spark",
    feature = "audio-music-musicgen",
    feature = "audio-codec-encodec",
))]
use napi_derive::napi;

// ===========================================================================
// STT — faster-whisper
// ===========================================================================

#[cfg(feature = "audio-stt-faster-whisper")]
mod stt {
    use super::napi;
    use blazen_llm::{FasterWhisperBackend, FasterWhisperConfig, SttBackendHandle};

    /// Configuration for the faster-whisper (`CTranslate2`) STT backend.
    ///
    /// Mirrors [`blazen_llm::FasterWhisperConfig`]. All fields are optional;
    /// unset fields fall back to the upstream defaults
    /// (`Systran/faster-whisper-tiny`, HF download on first use).
    #[napi(object, js_name = "FasterWhisperConfig")]
    pub struct JsFasterWhisperConfig {
        /// Hugging Face repo id for the `CTranslate2` Whisper bundle.
        pub model_id: Option<String>,
        /// Local path to a pre-downloaded bundle directory. When unset, the
        /// bundle is fetched from Hugging Face on first transcription.
        pub model_dir: Option<String>,
        /// Optional Hugging Face Hub revision pin (branch, tag, or commit
        /// SHA).
        pub revision: Option<String>,
    }

    impl From<JsFasterWhisperConfig> for FasterWhisperConfig {
        fn from(c: JsFasterWhisperConfig) -> Self {
            let mut cfg = FasterWhisperConfig::default();
            if let Some(model_id) = c.model_id {
                cfg.model_id = model_id;
            }
            cfg.model_dir = c.model_dir.map(std::path::PathBuf::from);
            cfg.revision = c.revision;
            cfg
        }
    }

    /// The faster-whisper STT backend.
    ///
    /// Mirrors [`blazen_llm::FasterWhisperBackend`]. Construct with an
    /// optional [`FasterWhisperConfig`](JsFasterWhisperConfig); weights load
    /// lazily on first transcription.
    #[napi(js_name = "FasterWhisperBackend")]
    pub struct JsFasterWhisperBackend {
        pub(crate) inner: FasterWhisperBackend,
    }

    #[napi]
    #[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
    impl JsFasterWhisperBackend {
        /// Build a faster-whisper backend. No weights are loaded until the
        /// first transcription call.
        #[napi(constructor)]
        pub fn new(config: Option<JsFasterWhisperConfig>) -> Self {
            let cfg = config.map(Into::into).unwrap_or_default();
            Self {
                inner: FasterWhisperBackend::new(cfg),
            }
        }

        /// The stable backend identifier (`faster-whisper:<model_id>`).
        #[napi(getter)]
        pub fn id(&self) -> String {
            use blazen_audio::AudioBackend;
            self.inner.id().to_owned()
        }

        /// Wrap this backend in a typed [`SttBackendHandle`].
        #[napi(js_name = "intoHandle")]
        pub fn into_handle(&self) -> JsSttBackendHandle {
            JsSttBackendHandle {
                inner: SttBackendHandle::new(self.inner.clone()),
            }
        }
    }

    /// Typed handle wrapping a faster-whisper STT backend.
    ///
    /// Mirrors [`blazen_llm::SttBackendHandle`]. Obtain one from
    /// [`FasterWhisperBackend.intoHandle`](JsFasterWhisperBackend::into_handle).
    #[napi(js_name = "SttBackendHandle")]
    pub struct JsSttBackendHandle {
        pub(crate) inner: SttBackendHandle<FasterWhisperBackend>,
    }

    #[napi]
    #[allow(clippy::must_use_candidate)]
    impl JsSttBackendHandle {
        /// The wrapped backend's stable identifier.
        #[napi(getter)]
        pub fn id(&self) -> String {
            self.inner.id().to_owned()
        }

        /// The wrapped backend's capability tag.
        #[napi(getter, js_name = "providerKind")]
        pub fn provider_kind(&self) -> String {
            self.inner.provider_kind().to_owned()
        }

        /// Load the wrapped backend's weights.
        #[napi(js_name = "load")]
        #[allow(clippy::missing_errors_doc)]
        pub async fn load(&self) -> napi::Result<()> {
            self.inner
                .load()
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))
        }
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
pub use stt::{JsFasterWhisperBackend, JsFasterWhisperConfig, JsSttBackendHandle};

// ===========================================================================
// TTS — Spark-TTS
// ===========================================================================

#[cfg(feature = "audio-tts-spark")]
mod tts {
    use super::napi;
    use blazen_llm::{SparkTtsBackend, SparkTtsConfig, TtsBackendHandle};

    /// Configuration for the Spark-TTS backend.
    ///
    /// Mirrors [`blazen_llm::SparkTtsConfig`]. All fields are optional; unset
    /// fields fall back to the upstream defaults
    /// (`SparkAudio/Spark-TTS-0.5B`, HF download on first use).
    #[napi(object, js_name = "SparkTtsConfig")]
    pub struct JsSparkTtsConfig {
        /// Hugging Face repo id for the Spark-TTS bundle.
        pub model_id: Option<String>,
        /// Pre-resolved bundle directory. When unset, the bundle is
        /// downloaded and cached on first synthesis.
        pub model_dir: Option<String>,
        /// Optional revision (branch / tag / commit SHA) to pin against.
        pub revision: Option<String>,
    }

    impl From<JsSparkTtsConfig> for SparkTtsConfig {
        fn from(c: JsSparkTtsConfig) -> Self {
            let mut cfg = SparkTtsConfig::default();
            if let Some(model_id) = c.model_id {
                cfg.model_id = model_id;
            }
            cfg.model_dir = c.model_dir.map(std::path::PathBuf::from);
            cfg.revision = c.revision;
            cfg
        }
    }

    /// The Spark-TTS backend.
    ///
    /// Mirrors [`blazen_llm::SparkTtsBackend`]. Construct with an optional
    /// [`SparkTtsConfig`](JsSparkTtsConfig); weights load lazily on first
    /// synthesis.
    #[napi(js_name = "SparkTtsBackend")]
    pub struct JsSparkTtsBackend {
        pub(crate) inner: SparkTtsBackend,
    }

    #[napi]
    #[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
    impl JsSparkTtsBackend {
        /// Build a Spark-TTS backend. No weights are loaded until the first
        /// synthesis call.
        #[napi(constructor)]
        pub fn new(config: Option<JsSparkTtsConfig>) -> Self {
            let cfg = config.map(Into::into).unwrap_or_default();
            Self {
                inner: SparkTtsBackend::new(cfg),
            }
        }

        /// The configured model id.
        #[napi(getter, js_name = "modelId")]
        pub fn model_id(&self) -> String {
            self.inner.model_id().to_owned()
        }

        /// Wrap this backend in a typed [`TtsBackendHandle`].
        #[napi(js_name = "intoHandle")]
        pub fn into_handle(&self) -> JsTtsBackendHandle {
            JsTtsBackendHandle {
                inner: TtsBackendHandle::new(self.inner.clone()),
            }
        }
    }

    /// Typed handle wrapping a Spark-TTS backend.
    ///
    /// Mirrors [`blazen_llm::TtsBackendHandle`]. Obtain one from
    /// [`SparkTtsBackend.intoHandle`](JsSparkTtsBackend::into_handle).
    #[napi(js_name = "TtsBackendHandle")]
    pub struct JsTtsBackendHandle {
        pub(crate) inner: TtsBackendHandle<SparkTtsBackend>,
    }

    #[napi]
    #[allow(clippy::must_use_candidate)]
    impl JsTtsBackendHandle {
        /// The wrapped backend's stable identifier.
        #[napi(getter)]
        pub fn id(&self) -> String {
            use blazen_audio::AudioBackend;
            self.inner.inner().id().to_owned()
        }
    }
}

#[cfg(feature = "audio-tts-spark")]
pub use tts::{JsSparkTtsBackend, JsSparkTtsConfig, JsTtsBackendHandle};

// ===========================================================================
// Music — MusicGen
// ===========================================================================

#[cfg(feature = "audio-music-musicgen")]
mod music {
    use super::napi;
    use blazen_llm::{MusicBackendHandle, MusicgenBackend, MusicgenConfig};

    /// Typed handle wrapping a `MusicGen` text-to-music backend.
    ///
    /// Mirrors [`blazen_llm::MusicBackendHandle`]. Construct it directly to
    /// get a default-configured `MusicGen` handle; weights load lazily on
    /// first generation.
    #[napi(js_name = "MusicBackendHandle")]
    pub struct JsMusicBackendHandle {
        pub(crate) inner: MusicBackendHandle<MusicgenBackend>,
    }

    impl Default for JsMusicBackendHandle {
        fn default() -> Self {
            Self::new()
        }
    }

    #[napi]
    #[allow(clippy::must_use_candidate)]
    impl JsMusicBackendHandle {
        /// Build a default-configured `MusicGen` music backend handle.
        #[napi(constructor)]
        pub fn new() -> Self {
            Self {
                inner: MusicBackendHandle::new(MusicgenBackend::new(MusicgenConfig::default())),
            }
        }

        /// The wrapped backend's stable identifier.
        #[napi(getter)]
        pub fn id(&self) -> String {
            use blazen_audio::AudioBackend;
            self.inner.backend().id().to_owned()
        }
    }
}

#[cfg(feature = "audio-music-musicgen")]
pub use music::JsMusicBackendHandle;

// ===========================================================================
// Codec — EnCodec
// ===========================================================================

#[cfg(feature = "audio-codec-encodec")]
mod codec {
    use super::napi;
    use blazen_audio_codec::backends::encodec::EncodecBackend;
    use blazen_llm::CodecBackendHandle;

    /// Typed handle wrapping an `EnCodec` neural audio codec backend.
    ///
    /// Mirrors [`blazen_llm::CodecBackendHandle`]. Construct it directly to
    /// get a default `facebook/encodec_24khz` handle; weights load lazily on
    /// first encode/decode.
    #[napi(js_name = "CodecBackendHandle")]
    pub struct JsCodecBackendHandle {
        pub(crate) inner: CodecBackendHandle<EncodecBackend>,
    }

    impl Default for JsCodecBackendHandle {
        fn default() -> Self {
            Self::new()
        }
    }

    #[napi]
    #[allow(clippy::must_use_candidate)]
    impl JsCodecBackendHandle {
        /// Build a default-configured `EnCodec` codec backend handle.
        #[napi(constructor)]
        pub fn new() -> Self {
            Self {
                inner: CodecBackendHandle::new(EncodecBackend::default_24khz()),
            }
        }

        /// The wrapped backend's stable identifier.
        #[napi(getter)]
        pub fn id(&self) -> String {
            use blazen_audio::AudioBackend;
            self.inner.backend().id().to_owned()
        }
    }
}

#[cfg(feature = "audio-codec-encodec")]
pub use codec::JsCodecBackendHandle;
