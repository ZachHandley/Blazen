//! Python wrapper for the audio transcription provider abstraction.
//!
//! Provides [`PyTranscription`] with factory methods for each supported
//! transcription backend (fal.ai, whisper.cpp, etc.). Mirrors the pattern
//! used for [`PyEmbeddingModel`](crate::types::embedding::PyEmbeddingModel)
//! and [`PyCompletionModel`](crate::providers::PyCompletionModel).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::PyTranscriptionRequest;
use crate::compute::result_types::PyTranscriptionResult;
use crate::error::blazen_error_to_pyerr;
use crate::providers::options::PyFalOptions;
#[cfg(feature = "whispercpp")]
use crate::providers::options::PyWhisperOptions;
use blazen_llm::compute::Transcription;

// ---------------------------------------------------------------------------
// PyTranscription
// ---------------------------------------------------------------------------

/// An audio transcription provider.
///
/// Use the static constructor methods to create a transcriber for a specific
/// provider, then call :meth:`transcribe` to convert audio to text.
///
/// Example:
///     >>> # Local, offline transcription via whisper.cpp
///     >>> opts = WhisperOptions(model=WhisperModel.Base)
///     >>> transcriber = Transcription.whispercpp(options=opts)
///     >>> result = await transcriber.transcribe(
///     ...     TranscriptionRequest.from_file("audio.wav")
///     ... )
///     >>> print(result.text)
///
///     >>> # Remote transcription via fal.ai (requires API key)
///     >>> transcriber = Transcription.fal()
///     >>> result = await transcriber.transcribe(
///     ...     TranscriptionRequest(audio_url="https://example.com/audio.mp3")
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "Transcription", from_py_object)]
#[derive(Clone)]
pub struct PyTranscription {
    pub(crate) inner: Arc<dyn Transcription>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTranscription {
    // -----------------------------------------------------------------
    // Provider constructors
    // -----------------------------------------------------------------

    /// Create a fal.ai transcription provider.
    ///
    /// Requires a fal.ai API key via ``options.api_key`` or the ``FAL_KEY``
    /// environment variable. Supports remote audio URLs.
    ///
    /// Args:
    ///     options: Optional typed ``FalOptions`` for endpoint, enterprise
    ///         tier, and modality auto-routing.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn fal(options: Option<PyRef<'_, PyFalOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::fal::FalProvider::from_options(opts)
                    .map_err(blazen_error_to_pyerr)?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Provider info
    // -----------------------------------------------------------------

    /// Get the provider identifier (e.g. "fal", "whispercpp").
    #[getter]
    fn provider_id(&self) -> String {
        blazen_llm::compute::ComputeProvider::provider_id(self.inner.as_ref()).to_owned()
    }

    // -----------------------------------------------------------------
    // Transcribe
    // -----------------------------------------------------------------

    /// Transcribe an audio clip to text.
    ///
    /// Args:
    ///     request: A :class:`TranscriptionRequest` (use
    ///         :meth:`TranscriptionRequest.from_file` for local backends
    ///         like whisper.cpp).
    ///
    /// Returns:
    ///     A :class:`TranscriptionResult` with ``text``, ``segments``,
    ///     ``language``, ``timing``, ``cost``, and ``metadata``.
    ///
    /// Example:
    ///     >>> req = TranscriptionRequest.from_file("/path/to/audio.wav")
    ///     >>> result = await transcriber.transcribe(req)
    ///     >>> print(result.text)
    fn transcribe<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PyTranscriptionRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner.clone();
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = Transcription::transcribe(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyTranscriptionResult { inner: result })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Transcription(provider_id='{}')",
            blazen_llm::compute::ComputeProvider::provider_id(self.inner.as_ref())
        )
    }
}

// ---------------------------------------------------------------------------
// Feature-gated whispercpp factory (separate impl block so pyo3-stub-gen
// does not try to resolve the option type when the feature is disabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "whispercpp")]
#[gen_stub_pymethods]
#[pymethods]
impl PyTranscription {
    /// Create a local whisper.cpp transcription provider.
    ///
    /// Runs transcription entirely on-device using whisper.cpp. The first
    /// call downloads the GGML model (tens to hundreds of MB depending on
    /// the chosen variant) and caches it for subsequent runs. No API key
    /// is required.
    ///
    /// whisper.cpp currently expects **16-bit PCM mono WAV at 16 kHz**.
    /// URL audio sources are not supported -- use
    /// :meth:`TranscriptionRequest.from_file` with a local path.
    ///
    /// Args:
    ///     options: Optional typed :class:`WhisperOptions` with model size,
    ///         device, language, and cache directory.
    ///
    /// Example:
    ///     >>> opts = WhisperOptions(model=WhisperModel.Base)
    ///     >>> transcriber = Transcription.whispercpp(options=opts)
    ///     >>> req = TranscriptionRequest.from_file("audio.wav")
    ///     >>> result = await transcriber.transcribe(req)
    ///     >>> print(result.text)
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn whispercpp(options: Option<PyRef<'_, PyWhisperOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider =
            crate::convert::block_on_context(blazen_llm::WhisperCppProvider::from_options(opts))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }
}
