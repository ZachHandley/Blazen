//! Python wrapper for the local TTS provider (`any-tts`).
//!
//! Exposes :class:`TtsProvider` (Kokoro-82M default) bound onto the
//! `blazen_audio_tts::TtsProvider` engine. When the `engine` feature is
//! not compiled in, :meth:`text_to_speech` raises :class:`TtsError` with
//! an ``"engine not available"`` message — same shape as the other local
//! backends.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::PySpeechRequest;
use crate::compute::result_types::PyAudioResult;
use crate::error::TtsError;
use crate::providers::options::PyTtsOptions;
use blazen_llm::compute::AudioGeneration;
use blazen_llm::{AnyTtsBackend, DynTtsProvider};

// ---------------------------------------------------------------------------
// PyTtsProvider
// ---------------------------------------------------------------------------

/// A local TTS provider backed by the `any-tts` crate (Kokoro-82M default).
///
/// Synthesis runs fully on-device. No API key is required.
///
/// Example:
///     >>> opts = TtsOptions(model="kokoro82m", voice="af_bella")
///     >>> provider = TtsProvider(options=opts)
#[gen_stub_pyclass]
#[pyclass(name = "TtsProvider", from_py_object)]
#[derive(Clone)]
pub struct PyTtsProvider {
    inner: Arc<DynTtsProvider>,
    model_str: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTtsProvider {
    /// Create a new TTS provider.
    ///
    /// Args:
    ///     options: Optional :class:`TtsOptions` for model, voice,
    ///         language, sample rate, and cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyTtsOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let model_str = opts.model.unwrap_or_default().as_str().to_owned();
        let backend =
            AnyTtsBackend::from_options(opts).map_err(|e| TtsError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(DynTtsProvider::erase(backend)),
            model_str,
        })
    }

    /// The configured model kind, as a string (`"kokoro"`, `"vibevoice"`, `"qwen3_tts"`).
    #[getter]
    fn model(&self) -> String {
        self.model_str.clone()
    }

    /// Whether the underlying any-tts engine is compiled into this build.
    /// When the `anytts` feature is on, this returns ``True`` — the
    /// provider can be constructed regardless of the runtime model-load
    /// outcome.
    #[getter]
    fn engine_available(&self) -> bool {
        true
    }

    /// Synthesize speech from text.
    ///
    /// Args:
    ///     request: A :class:`SpeechRequest` with text, voice, language,
    ///         and other parameters.
    ///
    /// Returns:
    ///     An :class:`AudioResult` with the synthesized audio.
    ///
    /// Raises:
    ///     TtsError: If the engine feature is not compiled in or
    ///         synthesis fails.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        request: PySpeechRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::text_to_speech(&*provider, req)
                .await
                .map_err(|e| TtsError::new_err(e.to_string()))?;
            Ok::<PyAudioResult, PyErr>(PyAudioResult { inner: result })
        })
    }

    fn __repr__(&self) -> String {
        format!("TtsProvider(model={:?})", self.model_str)
    }
}
