//! Python wrapper for the OpenAI provider.
//!
//! Exposes [`OpenAiProvider`](blazen_llm::providers::openai::OpenAiProvider)
//! to Python with chat-completion (`complete`, `stream`) capabilities as well
//! as text-to-speech. This is the standalone class form of `Model.openai(...)`;
//! both surfaces wrap the same Rust provider.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::compute::request_types::PySpeechRequest;
use crate::compute::result_types::PyAudioResult;
use crate::error::blazen_error_to_pyerr;
use crate::providers::config::PyRetryConfig;
use crate::providers::model::{
    LazyStreamState, PendingStream, PyLazyCompletionStream, PyModelOptions, build_request,
};
use crate::providers::options::PyProviderOptions;
use crate::types::{PyChatMessage, PyHttpClientHandle, PyModelResponse};
use blazen_llm::ChatMessage;
use blazen_llm::compute::AudioGeneration;
use blazen_llm::providers::openai::OpenAiProvider;
use blazen_llm::traits::Model;

// ---------------------------------------------------------------------------
// PyOpenAiProvider
// ---------------------------------------------------------------------------

/// An OpenAI provider for chat completion, text-to-speech, and other compute
/// capabilities.
///
/// This is the standalone class form of `Model.openai(...)`. Both surfaces wrap
/// the same Rust provider; use whichever you prefer.
///
/// Example:
/// ```text
///  >>> from blazen import OpenAiProvider, ProviderOptions, ChatMessage
///  >>> openai = OpenAiProvider(options=ProviderOptions(api_key="sk-..."))
///  >>> resp = await openai.complete([ChatMessage.user("Hello!")])
/// ```
///
/// It also exposes text-to-speech:
/// ```text
///  >>> from blazen import SpeechRequest
///  >>> result = await openai.text_to_speech(SpeechRequest(text="Hello, world!"))
/// ```
///
/// To target an OpenAI-compatible service (zvoice/VoxCPM2, etc.), set
/// ``base_url`` on the options. With an empty ``api_key`` the ``Authorization``
/// header is omitted:
/// ```text
///  >>> local = OpenAiProvider(options=ProviderOptions(
///  ...     api_key="",
///  ...     base_url="http://beastpc.lan:8900/v1",
///  ... ))
///  >>> result = await local.text_to_speech(SpeechRequest(text="Hello!"))
/// ```
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

    /// Get the model ID.
    #[getter]
    fn model_id(&self) -> &str {
        Model::model_id(self.inner.as_ref())
    }

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     options: Optional [`ModelOptions`] for sampling parameters,
    ///         tools, and response format.
    ///
    /// Returns:
    ///     A ModelResponse with content, model, tool_calls, usage, etc.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ModelResponse]", imports = ("typing",)))]
    #[pyo3(signature = (messages, options=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyModelOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = Model::complete(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyModelResponse { inner: response })
        })
    }

    /// Stream a chat completion as an async iterator.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     options: Optional [`ModelOptions`] for sampling parameters,
    ///         tools, and response format.
    ///
    /// Returns:
    ///     A [`CompletionStream`] that yields chunks via ``async for``.
    #[pyo3(signature = (messages, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyModelOptions>>,
    ) -> PyResult<Bound<'py, PyLazyCompletionStream>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner: Arc<dyn Model> = self.inner.clone();
        let stream = PyLazyCompletionStream {
            state: Arc::new(Mutex::new(LazyStreamState::NotStarted(Box::new(
                PendingStream {
                    model: inner,
                    request: Some(request),
                },
            )))),
        };
        Bound::new(py, stream)
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

    /// Set the provider-level default retry config.
    pub fn with_retry_config(&self, config: PyRetryConfig) -> Self {
        let inner = (*self.inner).clone().with_retry_config(config.inner);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Return an opaque handle to the underlying HTTP client.
    pub fn http_client(&self) -> PyHttpClientHandle {
        PyHttpClientHandle {
            inner: self.inner.http_client(),
        }
    }
}
