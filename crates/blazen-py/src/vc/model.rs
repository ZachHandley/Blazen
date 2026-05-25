//! `VcModel` Python wrapper.
//!
//! Exposes a unified Python class around any
//! [`VoiceConversionBackend`](blazen_audio_vc::VoiceConversionBackend) —
//! currently RVC — constructed via per-engine `@staticmethod` factories.
//! Each factory is individually feature-gated so a build with only one
//! backend feature compiles in only that constructor.

use std::path::PathBuf;
use std::sync::Arc;

use blazen_audio_vc::VoiceConversionBackend;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::vc_error_to_pyerr;
use crate::providers::options::PyDevice;
use crate::vc::stream::{LazyVcStreamState, PendingVcStream, PyVcStream};
use crate::vc::target_voice::PyTargetVoice;

/// Opaque handle around a `dyn VoiceConversionBackend`.
///
/// Constructed via one of the feature-gated `@staticmethod` factories
/// (currently `VcModel.rvc(...)`) — there is no public `__init__`.
///
/// The handle is cheap to clone (internally `Arc<dyn VoiceConversionBackend>`);
/// the underlying weights are lazily downloaded on the first `convert_voice`
/// or `stream_convert_pcm` call.
#[gen_stub_pyclass]
#[pyclass(name = "VcModel", from_py_object)]
#[derive(Clone)]
pub struct PyVcModel {
    pub(crate) inner: Arc<dyn VoiceConversionBackend>,
    pub(crate) id_str: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVcModel {
    // -----------------------------------------------------------------
    // RVC
    // -----------------------------------------------------------------

    /// Construct an RVC (Retrieval-based Voice Conversion) backend.
    ///
    /// Args:
    ///     voice_dir: Optional directory containing per-voice profile
    ///         subdirectories (`<voice_dir>/<voice_id>/model.pth`, etc).
    ///         When provided, the `BLAZEN_RVC_VOICE_DIR` environment
    ///         variable is set for the lifetime of this process so the
    ///         RVC backend can discover voice profiles from disk.
    ///     device: Optional [`Device`] override (default: CPU). RVC
    ///         backends today are CPU-bound; the parameter is accepted
    ///         for forward-compat and currently ignored.
    ///
    /// Voice profiles must already exist on disk; runtime registration
    /// via [`register_target_voice`](Self::register_target_voice) is not
    /// supported by the current RVC backend (raises `UnsupportedError`).
    #[staticmethod]
    #[pyo3(signature = (*, voice_dir=None, device=None))]
    fn rvc(voice_dir: Option<PathBuf>, device: Option<PyRef<'_, PyDevice>>) -> Self {
        use blazen_audio_vc::RvcBackend;

        let _ = device; // PyDevice -> candle::Device routing for the VC
        // backends will land alongside the same wiring for
        // the other capability surfaces. RVC defaults to CPU.

        if let Some(dir) = voice_dir {
            // SAFETY: `std::env::set_var` is `unsafe` in Rust 2024 because
            // mutating the process environment is not thread-safe on
            // POSIX. We accept that risk at module-init time — the
            // RVC backend reads the variable lazily on first voice
            // resolution, and callers constructing multiple `VcModel`s
            // concurrently must use a single shared voice_dir.
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var("BLAZEN_RVC_VOICE_DIR", dir);
            }
        }

        let backend = RvcBackend::new();
        let id_str = blazen_audio::AudioBackend::id(&backend).to_string();
        Self {
            inner: Arc::new(backend),
            id_str,
        }
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Stable backend identifier (e.g. `"rvc"`).
    #[getter]
    fn id(&self) -> &str {
        &self.id_str
    }

    fn __repr__(&self) -> String {
        format!("VcModel(id={:?})", self.id_str)
    }

    // -----------------------------------------------------------------
    // Voice management
    // -----------------------------------------------------------------

    /// List the target voices the active backend can currently render.
    ///
    /// Returns a coroutine that resolves to a list of
    /// [`TargetVoice`](super::PyTargetVoice) descriptors. The RVC
    /// backend reads voice profiles from `$BLAZEN_RVC_VOICE_DIR` lazily;
    /// passing an empty / non-existent directory returns an empty list.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.List[TargetVoice]]", imports = ("typing",)))]
    fn list_target_voices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            let voices = backend
                .list_target_voices()
                .await
                .map_err(vc_error_to_pyerr)?;
            Ok(voices
                .into_iter()
                .map(PyTargetVoice::from)
                .collect::<Vec<_>>())
        })
    }

    /// Register a new target voice with the active backend, using
    /// `reference_audio_path` as the speaker-embedding source.
    ///
    /// The current RVC backend does not support runtime voice
    /// registration — voice profiles must be placed on disk under
    /// `$BLAZEN_RVC_VOICE_DIR/<voice_id>/` ahead of time. Always
    /// raises [`UnsupportedError`](crate::error::UnsupportedError) on
    /// this build.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn register_target_voice<'py>(
        &self,
        py: Python<'py>,
        voice_id: String,
        reference_audio_path: PathBuf,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            backend
                .register_target_voice(&voice_id, &reference_audio_path)
                .await
                .map_err(vc_error_to_pyerr)?;
            Ok(())
        })
    }

    // -----------------------------------------------------------------
    // Non-streaming conversion
    // -----------------------------------------------------------------

    /// Convert a source utterance on disk into the voice of a
    /// previously-registered target speaker.
    ///
    /// Returns a coroutine that resolves to a self-describing WAV byte
    /// buffer (16-bit PCM at the target voice's native sample rate; see
    /// [`TargetVoice.sample_rate_hz`](super::PyTargetVoice)).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, bytes]", imports = ("typing",)))]
    fn convert_voice<'py>(
        &self,
        py: Python<'py>,
        input_audio_path: PathBuf,
        target_voice_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            let bytes = backend
                .convert_voice(&input_audio_path, &target_voice_id)
                .await
                .map_err(vc_error_to_pyerr)?;
            Python::attach(|py| Ok(PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -----------------------------------------------------------------
    // Streaming conversion
    // -----------------------------------------------------------------

    /// Stream-convert a single PCM utterance for low-latency progressive
    /// playback.
    ///
    /// Wraps the provided `input_pcm` f32 sample buffer in a single-item
    /// source stream, hands it to the backend's `stream_convert`, and
    /// returns a [`VcStream`](super::PyVcStream) async-iterator over the
    /// converted output chunks.
    ///
    /// Future revisions (per the binding-parity follow-up) will expand
    /// this to a true bidirectional stream over an async iterator of
    /// source chunks; the current single-utterance variant matches the
    /// surface exposed by every other binding.
    fn stream_convert_pcm<'py>(
        &self,
        py: Python<'py>,
        input_pcm: Vec<f32>,
        target_voice_id: String,
    ) -> PyResult<Bound<'py, PyVcStream>> {
        let stream = PyVcStream {
            state: Arc::new(Mutex::new(LazyVcStreamState::NotStarted(Box::new(
                PendingVcStream {
                    backend: self.inner.clone(),
                    input_pcm: Some(input_pcm),
                    target_voice_id,
                },
            )))),
        };
        Bound::new(py, stream)
    }
}
