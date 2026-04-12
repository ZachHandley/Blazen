//! Python wrapper for the OpenAI provider.
//!
//! Exposes [`OpenAiProvider`](blazen_llm::providers::openai::OpenAiProvider)
//! to Python with text-to-speech support. For LLM completion, use
//! [`PyCompletionModel::openai`](crate::providers::completion_model::PyCompletionModel::openai)
//! instead -- this class is specifically for the non-completion capabilities
//! (TTS, and in the future other audio methods).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::PySpeechRequest;
use crate::compute::result_types::PyAudioResult;
use crate::error::blazen_error_to_pyerr;
use crate::providers::options::PyProviderOptions;
use blazen_llm::compute::AudioGeneration;
use blazen_llm::providers::openai::OpenAiProvider;

// ---------------------------------------------------------------------------
// PyOpenAiProvider
// ---------------------------------------------------------------------------

/// An OpenAI provider for text-to-speech and other compute capabilities.
///
/// For LLM chat completions, use [`CompletionModel.openai`] instead.
/// This class wraps [`blazen_llm::providers::openai::OpenAiProvider`]
/// directly and exposes its non-completion capabilities.
///
/// Example:
///     >>> from blazen import OpenAiProvider, ProviderOptions, SpeechRequest
///     >>> openai = OpenAiProvider(options=ProviderOptions(api_key="sk-..."))
///     >>> result = await openai.text_to_speech(SpeechRequest(text="Hello, world!"))
///
/// To target an OpenAI-compatible service (zvoice/VoxCPM2, etc.), set
/// ``base_url`` on the options. With an empty ``api_key`` the ``Authorization``
/// header is omitted:
///     >>> local = OpenAiProvider(options=ProviderOptions(
///     ...     api_key="",
///     ...     base_url="http://beastpc.lan:8900/v1",
///     ... ))
///     >>> result = await local.text_to_speech(SpeechRequest(text="Hello!"))
#[gen_stub_pyclass]
#[pyclass(name = "OpenAiProvider", from_py_object)]
#[derive(Clone)]
pub struct PyOpenAiProvider {
    inner: Arc<OpenAiProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAiProvider {
    /// Create a new OpenAI provider.
    ///
    /// Args:
    ///     options: Optional [`ProviderOptions`] with api_key, base_url, model.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(OpenAiProvider::from_options(opts).map_err(blazen_error_to_pyerr)?),
        })
    }

    /// Synthesize speech from text.
    ///
    /// Args:
    ///     request: A [`SpeechRequest`] with text and optional voice/model.
    ///
    /// Returns:
    ///     An [`AudioResult`] with audio clips, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        request: PySpeechRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::text_to_speech(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }
}
