//! Python wrapper for the local whisper.cpp transcription provider.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::PyTranscriptionRequest;
use crate::compute::result_types::PyTranscriptionResult;
use crate::error::WhisperError;
use crate::providers::options::PyWhisperOptions;
use blazen_llm::WhisperCppProvider;
use blazen_llm::compute::Transcription;
use blazen_llm::traits::LocalModel;

// ---------------------------------------------------------------------------
// PyWhisperCppProvider
// ---------------------------------------------------------------------------

/// A local whisper.cpp transcription provider.
///
/// Runs speech-to-text fully on-device using the whisper.cpp engine. The
/// provider expects 16-bit PCM mono WAV at 16 kHz; URL audio sources are
/// not supported. Use :meth:`TranscriptionRequest.from_file` with a local
/// path.
///
/// Example:
///     >>> opts = WhisperOptions(model=WhisperModel.Base)
///     >>> provider = WhisperCppProvider(options=opts)
///     >>> req = TranscriptionRequest.from_file("audio.wav")
///     >>> result = await provider.transcribe(req)
///     >>> print(result.text)
#[gen_stub_pyclass]
#[pyclass(name = "WhisperCppProvider", from_py_object)]
#[derive(Clone)]
pub struct PyWhisperCppProvider {
    inner: Arc<WhisperCppProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWhisperCppProvider {
    /// Create a new whisper.cpp provider.
    ///
    /// Args:
    ///     options: Optional :class:`WhisperOptions` for model size,
    ///         device, language, and cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyWhisperOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider = crate::convert::block_on_context(WhisperCppProvider::from_options(opts))
            .map_err(|e| WhisperError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Get the provider identifier (``"whispercpp"``).
    #[getter]
    fn provider_id(&self) -> String {
        blazen_llm::compute::ComputeProvider::provider_id(self.inner.as_ref()).to_owned()
    }

    /// Alias for :attr:`provider_id` to mirror :class:`CompletionModel`.
    #[getter]
    fn model_id(&self) -> String {
        self.provider_id()
    }

    /// Transcribe an audio clip to text.
    ///
    /// Args:
    ///     request: A :class:`TranscriptionRequest`. Use
    ///         :meth:`TranscriptionRequest.from_file` for local files
    ///         (URL sources are not supported by whisper.cpp).
    ///
    /// Returns:
    ///     A :class:`TranscriptionResult` with text, segments, language,
    ///     timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, TranscriptionResult]", imports = ("typing",)))]
    fn transcribe<'py>(
        &self,
        py: Python<'py>,
        request: PyTranscriptionRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = Transcription::transcribe(inner.as_ref(), rust_req)
                .await
                .map_err(|e| WhisperError::new_err(e.to_string()))?;
            Ok(PyTranscriptionResult { inner: result })
        })
    }

    /// Load the model weights into memory. Idempotent.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::load(inner.as_ref())
                .await
                .map_err(|e| WhisperError::new_err(e.to_string()))
        })
    }

    /// Drop the loaded model and free its memory.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::unload(inner.as_ref())
                .await
                .map_err(|e| WhisperError::new_err(e.to_string()))
        })
    }

    /// Whether the model is currently loaded.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.bool]", imports = ("typing", "builtins")))]
    fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(LocalModel::is_loaded(inner.as_ref()).await)
        })
    }

    fn __repr__(&self) -> String {
        format!("WhisperCppProvider(provider_id='{}')", self.provider_id())
    }
}
