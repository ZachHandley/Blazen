//! Python wrapper for the local Piper TTS provider.
//!
//! The Piper engine integration is in progress in the Rust crate (Phase 9
//! per the upstream roadmap); the Python class is exposed now so callers
//! can construct providers and surface engine-availability errors with
//! the same shape as the other local backends.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::PySpeechRequest;
use crate::compute::result_types::PyAudioResult;
use crate::error::PiperError;
use crate::providers::options::PyPiperOptions;
use blazen_llm::PiperProvider;

// ---------------------------------------------------------------------------
// PyPiperProvider
// ---------------------------------------------------------------------------

/// A local Piper text-to-speech provider.
///
/// Runs synthesis fully on-device via Piper voice models on ONNX Runtime.
/// No API key is required.
///
/// The underlying Rust integration is in progress; calls to
/// :meth:`text_to_speech` currently raise :class:`PiperError` with an
/// ``"engine not available"`` message until the engine wiring lands.
///
/// Example:
///     >>> opts = PiperOptions(model_id="en_US-amy-medium")
///     >>> provider = PiperProvider(options=opts)
#[gen_stub_pyclass]
#[pyclass(name = "PiperProvider", from_py_object)]
#[derive(Clone)]
pub struct PyPiperProvider {
    inner: Arc<PiperProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPiperProvider {
    /// Create a new Piper provider.
    ///
    /// Args:
    ///     options: Optional :class:`PiperOptions` for voice model id,
    ///         speaker id, sample rate, and cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyPiperOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider =
            PiperProvider::from_options(opts).map_err(|e| PiperError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Get the configured voice model id, if any.
    #[getter]
    fn model_id(&self) -> Option<String> {
        self.inner.model_id().map(str::to_owned)
    }

    /// Whether the underlying ONNX Runtime engine is compiled into this
    /// build. When ``False``, all synthesis calls raise :class:`PiperError`.
    #[getter]
    fn engine_available(&self) -> bool {
        self.inner.engine_available()
    }

    /// Synthesize speech from text.
    ///
    /// Args:
    ///     request: A :class:`SpeechRequest` with text, voice, and other
    ///         parameters.
    ///
    /// Returns:
    ///     An :class:`AudioResult` with the synthesized audio.
    ///
    /// Raises:
    ///     PiperError: While the upstream engine integration is in
    ///         progress, this call always raises with an
    ///         ``"engine not available"`` message.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        _request: PySpeechRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        // The upstream `PiperProvider` does not yet implement
        // `AudioGeneration`. Surface the engine-availability error
        // synchronously by short-circuiting before the future fires.
        // We deliberately consume `&self` (via `engine_available`) to keep
        // the method semantics consistent with the eventual wired version.
        let available = self.inner.engine_available();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if available {
                // When the engine lands, route to the real synthesizer.
                Err::<PyAudioResult, _>(PiperError::new_err(
                    "piper text-to-speech engine is compiled in but the AudioGeneration \
                     trait bridge has not been wired yet",
                ))
            } else {
                Err::<PyAudioResult, _>(PiperError::new_err(
                    "piper engine not available: build with the `piper/engine` feature",
                ))
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("PiperProvider(model_id={:?})", self.inner.model_id())
    }
}
