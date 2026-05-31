//! Python bindings for the audio backend handles and concrete backends
//! re-exported by `blazen-llm`.
//!
//! Each capability exposes a type-erased *handle* class
//! (`SttBackendHandle`, `TtsBackendHandle`, `MusicBackendHandle`,
//! `CodecBackendHandle`) plus, where a concrete native backend is feature-
//! gated in, a constructible *backend* + *config* pair
//! (`FasterWhisperBackend` / `FasterWhisperConfig`,
//! `SparkTtsBackend` / `SparkTtsConfig`).
//!
//! The handles hold the erased `Dyn*Provider` shape (`Box<dyn …>` /
//! `Arc<dyn …>`) so they can flow across the binding boundary without
//! carrying a generic backend type. Inference I/O is bridged through
//! `pythonize` / `serde_json` so request options and result payloads cross as
//! plain Python values.
//!
//! Everything here is gated behind the relevant audio feature so a build
//! without the native inference deps (the common case) simply omits these
//! classes.

#![allow(clippy::used_underscore_binding)]

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Pythonize a `Serialize` value into a `Py<PyAny>` (used for result types
/// like `GeneratedAudio` / `TranscriptionResult`).
#[cfg(any(
    feature = "whispercpp",
    feature = "audio-stt-faster-whisper",
    feature = "audio-stt-whisper-streaming",
    feature = "tts",
    feature = "audio-tts-spark",
    feature = "audio-tts-bark",
    feature = "audio-tts-f5",
    feature = "audio-music-musicgen",
    feature = "audio-music-audiogen",
    feature = "audio-music-stable-audio",
))]
fn pythonize_result<T: serde::Serialize>(
    py: pyo3::Python<'_>,
    value: &T,
) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    pythonize::pythonize(py, value)
        .map(pyo3::Bound::unbind)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to encode result: {e}"))
        })
}

// ---------------------------------------------------------------------------
// STT: SttBackendHandle + FasterWhisperBackend / FasterWhisperConfig
// ---------------------------------------------------------------------------

#[cfg(any(
    feature = "whispercpp",
    feature = "audio-stt-faster-whisper",
    feature = "audio-stt-whisper-streaming",
))]
mod stt {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::pythonize_result;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

    use blazen_llm::{DynSttProvider, SttError};

    use crate::error::blazen_error_to_pyerr;

    fn stt_err_to_py(e: SttError) -> PyErr {
        blazen_error_to_pyerr(blazen_llm::BlazenError::provider("stt", e.to_string()))
    }

    /// Type-erased speech-to-text backend handle.
    ///
    /// Wraps any concrete STT backend (e.g. :class:`FasterWhisperBackend`) and
    /// forwards the lifecycle (``load`` / ``unload`` / ``is_loaded``) and
    /// ``transcribe`` surface. Build one via a concrete backend's
    /// ``to_handle()`` method.
    #[gen_stub_pyclass]
    #[pyclass(name = "SttBackendHandle")]
    pub struct PySttBackendHandle {
        pub(crate) inner: Arc<DynSttProvider>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PySttBackendHandle {
        /// Stable identifier of the wrapped backend.
        #[getter]
        fn id(&self) -> String {
            self.inner.id().to_owned()
        }

        /// Capability tag of the wrapped backend (e.g. ``"stt"``).
        #[getter]
        fn provider_kind(&self) -> String {
            self.inner.provider_kind().to_owned()
        }

        /// Load the wrapped backend's weights.
        fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                inner.load().await.map_err(stt_err_to_py)
            })
        }

        /// Unload the wrapped backend, freeing its weights.
        fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                inner.unload().await.map_err(stt_err_to_py)
            })
        }

        /// Whether the wrapped backend is loaded and ready.
        fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(
                py,
                async move { Ok(inner.is_loaded().await) },
            )
        }

        /// Transcribe an audio file at ``audio_path``.
        ///
        /// Returns a dict with ``text``, ``segments``, and ``language``.
        #[pyo3(signature = (audio_path, language=None))]
        fn transcribe<'py>(
            &self,
            py: Python<'py>,
            audio_path: String,
            language: Option<String>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let path = PathBuf::from(audio_path);
                let result = inner
                    .transcribe(&path, language.as_deref())
                    .await
                    .map_err(stt_err_to_py)?;
                Python::attach(|py| pythonize_result(py, &result))
            })
        }
    }

    impl PySttBackendHandle {
        pub(crate) fn from_dyn(provider: DynSttProvider) -> Self {
            Self {
                inner: Arc::new(provider),
            }
        }
    }

    // -- FasterWhisper -------------------------------------------------------

    #[cfg(feature = "audio-stt-faster-whisper")]
    pub use faster_whisper::{PyFasterWhisperBackend, PyFasterWhisperConfig};

    #[cfg(feature = "audio-stt-faster-whisper")]
    mod faster_whisper {
        use std::path::PathBuf;

        use super::PySttBackendHandle;
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

        use blazen_llm::{FasterWhisperBackend, FasterWhisperConfig, SttBackendHandle};

        /// Configuration for :class:`FasterWhisperBackend`.
        ///
        /// Defaults target ``"Systran/faster-whisper-tiny"``; pass ``model_dir``
        /// to skip the Hugging Face download and load a pre-fetched
        /// ``CTranslate2`` bundle directory.
        #[gen_stub_pyclass]
        #[pyclass(name = "FasterWhisperConfig", from_py_object)]
        #[derive(Clone)]
        pub struct PyFasterWhisperConfig {
            pub(crate) inner: FasterWhisperConfig,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyFasterWhisperConfig {
            #[new]
            #[pyo3(signature = (model_id=None, model_dir=None, revision=None))]
            fn new(
                model_id: Option<String>,
                model_dir: Option<String>,
                revision: Option<String>,
            ) -> Self {
                let mut inner = FasterWhisperConfig::default();
                if let Some(model_id) = model_id {
                    inner.model_id = model_id;
                }
                inner.model_dir = model_dir.map(PathBuf::from);
                inner.revision = revision;
                Self { inner }
            }

            #[getter]
            fn model_id(&self) -> String {
                self.inner.model_id.clone()
            }

            #[getter]
            fn model_dir(&self) -> Option<String> {
                self.inner
                    .model_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
            }

            #[getter]
            fn revision(&self) -> Option<String> {
                self.inner.revision.clone()
            }

            fn __repr__(&self) -> String {
                format!("FasterWhisperConfig(model_id={:?})", self.inner.model_id)
            }
        }

        /// faster-whisper speech-to-text backend (``CTranslate2``).
        #[gen_stub_pyclass]
        #[pyclass(name = "FasterWhisperBackend")]
        pub struct PyFasterWhisperBackend {
            pub(crate) inner: FasterWhisperBackend,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyFasterWhisperBackend {
            /// Build a backend from the given config (default config when
            /// ``config`` is omitted). No weights are loaded until the first
            /// ``transcribe`` (or an explicit ``load`` on the handle).
            #[new]
            #[pyo3(signature = (config=None))]
            fn new(config: Option<PyFasterWhisperConfig>) -> Self {
                let cfg = config.map(|c| c.inner).unwrap_or_default();
                Self {
                    inner: FasterWhisperBackend::new(cfg),
                }
            }

            /// The backend's stable id (e.g. ``"faster-whisper:<model_id>"``).
            #[getter]
            fn id(&self) -> String {
                self.inner.model_id().to_owned()
            }

            /// Wrap this backend in a type-erased :class:`SttBackendHandle`.
            #[gen_stub(override_return_type(type_repr = "SttBackendHandle"))]
            fn to_handle(&self) -> PySttBackendHandle {
                let handle = SttBackendHandle::new(self.inner.clone());
                PySttBackendHandle::from_dyn(handle.into_dyn())
            }

            fn __repr__(&self) -> String {
                format!("FasterWhisperBackend(id={:?})", self.inner.model_id())
            }
        }
    }
}

#[cfg(any(
    feature = "whispercpp",
    feature = "audio-stt-faster-whisper",
    feature = "audio-stt-whisper-streaming",
))]
pub use stt::PySttBackendHandle;
#[cfg(feature = "audio-stt-faster-whisper")]
pub use stt::{PyFasterWhisperBackend, PyFasterWhisperConfig};

// ---------------------------------------------------------------------------
// TTS: TtsBackendHandle + SparkTtsBackend / SparkTtsConfig
// ---------------------------------------------------------------------------

#[cfg(any(
    feature = "tts",
    feature = "audio-tts-spark",
    feature = "audio-tts-bark",
    feature = "audio-tts-f5",
))]
mod tts {
    use std::sync::Arc;

    use super::pythonize_result;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

    use blazen_llm::{DynTtsProvider, TtsError, TtsOptions};

    use crate::error::blazen_error_to_pyerr;

    fn tts_err_to_py(e: TtsError) -> PyErr {
        blazen_error_to_pyerr(blazen_llm::BlazenError::provider("tts", e.to_string()))
    }

    /// Type-erased text-to-speech backend handle.
    ///
    /// Wraps any concrete TTS backend (e.g. :class:`SparkTtsBackend`) and
    /// forwards the lifecycle (``load`` / ``unload`` / ``is_loaded``) and
    /// ``synthesize`` surface. Build one via a concrete backend's
    /// ``to_handle()`` method.
    #[gen_stub_pyclass]
    #[pyclass(name = "TtsBackendHandle")]
    pub struct PyTtsBackendHandle {
        pub(crate) inner: Arc<DynTtsProvider>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyTtsBackendHandle {
        /// Stable identifier of the wrapped backend.
        #[getter]
        fn id(&self) -> String {
            self.inner.id().to_owned()
        }

        /// Capability tag of the wrapped backend (e.g. ``"tts"``).
        #[getter]
        fn provider_kind(&self) -> String {
            self.inner.provider_kind().to_owned()
        }

        /// Load the wrapped backend's weights.
        fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                inner.load().await.map_err(|e| {
                    blazen_error_to_pyerr(blazen_llm::BlazenError::provider("tts", e.to_string()))
                })
            })
        }

        /// Unload the wrapped backend, freeing its weights.
        fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                inner.unload().await.map_err(|e| {
                    blazen_error_to_pyerr(blazen_llm::BlazenError::provider("tts", e.to_string()))
                })
            })
        }

        /// Whether the wrapped backend is loaded and ready.
        fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(
                py,
                async move { Ok(inner.is_loaded().await) },
            )
        }

        /// Synthesize speech from ``text``.
        ///
        /// ``options`` is an optional dict matching the ``TtsOptions`` schema.
        /// Returns a dict with the generated audio (``bytes``, ``format``,
        /// ``sample_rate``, ``channels``, ``duration_seconds``).
        #[pyo3(signature = (text, options=None))]
        fn synthesize<'py>(
            &self,
            py: Python<'py>,
            text: String,
            options: Option<&Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let opts: TtsOptions = match options {
                Some(bound) if !bound.is_none() => pythonize::depythonize(bound).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid TtsOptions: {e}"))
                })?,
                _ => TtsOptions::default(),
            };
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let audio = inner
                    .synthesize(&text, &opts)
                    .await
                    .map_err(tts_err_to_py)?;
                Python::attach(|py| pythonize_result(py, &audio))
            })
        }
    }

    impl PyTtsBackendHandle {
        pub(crate) fn from_dyn(provider: DynTtsProvider) -> Self {
            Self {
                inner: Arc::new(provider),
            }
        }
    }

    // -- SparkTts ------------------------------------------------------------

    #[cfg(feature = "audio-tts-spark")]
    pub use spark::{PySparkTtsBackend, PySparkTtsConfig};

    #[cfg(feature = "audio-tts-spark")]
    mod spark {
        use std::path::PathBuf;

        use super::PyTtsBackendHandle;
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

        use blazen_llm::{DynTtsProvider, SparkTtsBackend, SparkTtsConfig};

        /// Configuration for :class:`SparkTtsBackend`.
        #[gen_stub_pyclass]
        #[pyclass(name = "SparkTtsConfig", from_py_object)]
        #[derive(Clone)]
        pub struct PySparkTtsConfig {
            pub(crate) inner: SparkTtsConfig,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PySparkTtsConfig {
            #[new]
            #[pyo3(signature = (model_id=None, model_dir=None, revision=None))]
            fn new(
                model_id: Option<String>,
                model_dir: Option<String>,
                revision: Option<String>,
            ) -> Self {
                let mut inner = SparkTtsConfig::default();
                if let Some(model_id) = model_id {
                    inner.model_id = model_id;
                }
                inner.model_dir = model_dir.map(PathBuf::from);
                inner.revision = revision;
                Self { inner }
            }

            #[getter]
            fn model_id(&self) -> String {
                self.inner.model_id.clone()
            }

            #[getter]
            fn model_dir(&self) -> Option<String> {
                self.inner
                    .model_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
            }

            #[getter]
            fn revision(&self) -> Option<String> {
                self.inner.revision.clone()
            }

            fn __repr__(&self) -> String {
                format!("SparkTtsConfig(model_id={:?})", self.inner.model_id)
            }
        }

        /// Spark-TTS text-to-speech backend.
        #[gen_stub_pyclass]
        #[pyclass(name = "SparkTtsBackend")]
        pub struct PySparkTtsBackend {
            pub(crate) inner: SparkTtsBackend,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PySparkTtsBackend {
            /// Build a backend from the given config (default config when
            /// ``config`` is omitted).
            #[new]
            #[pyo3(signature = (config=None))]
            fn new(config: Option<PySparkTtsConfig>) -> Self {
                let cfg = config.map(|c| c.inner).unwrap_or_default();
                Self {
                    inner: SparkTtsBackend::new(cfg),
                }
            }

            /// Wrap this backend in a type-erased :class:`TtsBackendHandle`.
            #[gen_stub(override_return_type(type_repr = "TtsBackendHandle"))]
            fn to_handle(&self) -> PyTtsBackendHandle {
                let provider = DynTtsProvider::erase(self.inner.clone());
                PyTtsBackendHandle::from_dyn(provider)
            }

            fn __repr__(&self) -> String {
                "SparkTtsBackend()".to_owned()
            }
        }
    }
}

#[cfg(any(
    feature = "tts",
    feature = "audio-tts-spark",
    feature = "audio-tts-bark",
    feature = "audio-tts-f5",
))]
pub use tts::PyTtsBackendHandle;
#[cfg(feature = "audio-tts-spark")]
pub use tts::{PySparkTtsBackend, PySparkTtsConfig};

// ---------------------------------------------------------------------------
// Music: MusicBackendHandle
// ---------------------------------------------------------------------------

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-audiogen",
    feature = "audio-music-stable-audio",
))]
mod music {
    use std::sync::Arc;

    use super::pythonize_result;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

    use blazen_llm::{DynMusicProvider, MusicError};

    use crate::error::blazen_error_to_pyerr;

    fn music_err_to_py(e: MusicError) -> PyErr {
        blazen_error_to_pyerr(blazen_llm::BlazenError::provider("music", e.to_string()))
    }

    /// Type-erased music / sound-effect backend handle.
    ///
    /// Wraps an ``Arc<dyn MusicBackend>`` and forwards ``generate_music`` /
    /// ``generate_sfx``. Built internally by :class:`MusicModel`; exposed here
    /// for parity and direct use.
    #[gen_stub_pyclass]
    #[pyclass(name = "MusicBackendHandle")]
    pub struct PyMusicBackendHandle {
        pub(crate) inner: DynMusicProvider,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyMusicBackendHandle {
        /// Stable identifier of the wrapped backend.
        #[getter]
        fn id(&self) -> String {
            <dyn blazen_audio_music::MusicBackend>::id(self.inner.as_ref()).to_owned()
        }

        /// Capability tag of the wrapped backend (e.g. ``"music"``).
        #[getter]
        fn provider_kind(&self) -> String {
            <dyn blazen_audio_music::MusicBackend>::provider_kind(self.inner.as_ref()).to_owned()
        }

        /// Generate music from ``prompt`` of the requested duration.
        ///
        /// Returns a dict with the generated audio (``bytes``, ``format``,
        /// ``sample_rate``, ``channels``, ``duration_seconds``).
        #[pyo3(signature = (prompt, duration_seconds))]
        fn generate_music<'py>(
            &self,
            py: Python<'py>,
            prompt: String,
            duration_seconds: f32,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let audio = blazen_audio_music::MusicBackend::generate_music(
                    inner.as_ref(),
                    &prompt,
                    duration_seconds,
                )
                .await
                .map_err(music_err_to_py)?;
                Python::attach(|py| pythonize_result(py, &audio))
            })
        }

        /// Generate a sound effect from ``prompt`` of the requested duration.
        #[pyo3(signature = (prompt, duration_seconds))]
        fn generate_sfx<'py>(
            &self,
            py: Python<'py>,
            prompt: String,
            duration_seconds: f32,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let audio = blazen_audio_music::MusicBackend::generate_sfx(
                    inner.as_ref(),
                    &prompt,
                    duration_seconds,
                )
                .await
                .map_err(music_err_to_py)?;
                Python::attach(|py| pythonize_result(py, &audio))
            })
        }
    }
}

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-audiogen",
    feature = "audio-music-stable-audio",
))]
pub use music::PyMusicBackendHandle;

// ---------------------------------------------------------------------------
// Codec: CodecBackendHandle
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-codec")]
mod codec {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

    use blazen_llm::{CodecError, DynCodecProvider};

    use crate::error::blazen_error_to_pyerr;

    fn codec_err_to_py(e: CodecError) -> PyErr {
        blazen_error_to_pyerr(blazen_llm::BlazenError::provider("codec", e.to_string()))
    }

    /// Type-erased neural-audio-codec backend handle.
    ///
    /// Wraps an ``Arc<dyn CodecBackend>`` and forwards ``encode_pcm`` /
    /// ``decode_tokens``. Construct one from a concrete codec backend (e.g.
    /// via the ``dac`` classmethod, available with the ``audio-codec-dac``
    /// feature).
    #[gen_stub_pyclass]
    #[pyclass(name = "CodecBackendHandle")]
    pub struct PyCodecBackendHandle {
        pub(crate) inner: DynCodecProvider,
    }

    impl PyCodecBackendHandle {
        pub(crate) fn from_provider(inner: DynCodecProvider) -> Self {
            Self { inner }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyCodecBackendHandle {
        /// Encode a mono PCM ``samples`` buffer at ``sample_rate`` into a flat
        /// row-major token stream.
        #[pyo3(signature = (samples, sample_rate))]
        fn encode_pcm<'py>(
            &self,
            py: Python<'py>,
            samples: Vec<f32>,
            sample_rate: u32,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                inner
                    .encode_pcm(&samples, sample_rate)
                    .await
                    .map_err(codec_err_to_py)
            })
        }

        /// Decode a flat row-major ``tokens`` stream (with ``num_codebooks``
        /// codebooks) back into a mono PCM buffer.
        #[pyo3(signature = (tokens, num_codebooks))]
        fn decode_tokens<'py>(
            &self,
            py: Python<'py>,
            tokens: Vec<u32>,
            num_codebooks: usize,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                inner
                    .decode_tokens(&tokens, num_codebooks)
                    .await
                    .map_err(codec_err_to_py)
            })
        }
    }

    // -- DAC concrete backend (feature-gated constructor) --------------------

    #[cfg(feature = "audio-codec-dac")]
    #[gen_stub_pymethods]
    #[pymethods]
    impl PyCodecBackendHandle {
        /// Build a handle backed by the DAC (Descript Audio Codec) backend at
        /// 44.1 kHz.
        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "CodecBackendHandle"))]
        fn dac(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
            use blazen_audio_codec::backends::dac::DacBackend;
            let backend: Arc<dyn blazen_audio_codec::CodecBackend> =
                Arc::new(DacBackend::default_44khz());
            Self::from_provider(DynCodecProvider::new(backend))
        }
    }
}

#[cfg(feature = "audio-codec")]
pub use codec::PyCodecBackendHandle;
