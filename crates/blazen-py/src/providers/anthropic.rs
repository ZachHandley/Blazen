//! Python wrapper for the Anthropic Messages API provider.
//!
//! Exposes [`AnthropicProvider`](blazen_llm::providers::anthropic::AnthropicProvider)
//! to Python with chat-completion (`complete`, `stream`) capabilities.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::completion_model::{
    LazyStreamState, PendingStream, PyCompletionOptions, PyLazyCompletionStream, build_request,
};
use crate::providers::config::PyRetryConfig;
use crate::providers::options::PyProviderOptions;
use crate::types::{PyChatMessage, PyCompletionResponse, PyHttpClientHandle};
use blazen_llm::ChatMessage;
use blazen_llm::providers::anthropic::AnthropicProvider;
use blazen_llm::traits::CompletionModel;

/// An Anthropic Messages API provider.
///
/// This is the standalone class form of `CompletionModel.anthropic(...)`.
/// Both surfaces wrap the same Rust provider; use whichever you prefer.
///
/// Example:
///     >>> from blazen import AnthropicProvider, ProviderOptions, ChatMessage
///     >>> p = AnthropicProvider(options=ProviderOptions(api_key="sk-ant-..."))
///     >>> resp = await p.complete([ChatMessage.user("Hello!")])
#[gen_stub_pyclass]
#[pyclass(name = "AnthropicProvider", from_py_object)]
#[derive(Clone)]
pub struct PyAnthropicProvider {
    inner: Arc<AnthropicProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnthropicProvider {
    /// Create a new Anthropic provider.
    ///
    /// Args:
    ///     options: Optional [`ProviderOptions`] with api_key, base_url, model.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(AnthropicProvider::from_options(opts).map_err(blazen_error_to_pyerr)?),
        })
    }

    /// Get the model ID.
    #[getter]
    fn model_id(&self) -> &str {
        CompletionModel::model_id(self.inner.as_ref())
    }

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     options: Optional [`CompletionOptions`] for sampling parameters,
    ///         tools, and response format.
    ///
    /// Returns:
    ///     A CompletionResponse with content, model, tool_calls, usage, etc.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    #[pyo3(signature = (messages, options=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = CompletionModel::complete(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyCompletionResponse { inner: response })
        })
    }

    /// Stream a chat completion as an async iterator.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     options: Optional [`CompletionOptions`] for sampling parameters,
    ///         tools, and response format.
    ///
    /// Returns:
    ///     A [`CompletionStream`] that yields chunks via ``async for``.
    #[pyo3(signature = (messages, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyLazyCompletionStream>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner: Arc<dyn CompletionModel> = self.inner.clone();
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

    /// Set the provider-level default retry config.
    ///
    /// Returns a new provider sharing the same HTTP client and other
    /// settings, with the given retry config applied. Pipeline / workflow /
    /// step / call scopes can override this; if all are unset, this is the
    /// fallback.
    pub fn with_retry_config(&self, config: PyRetryConfig) -> Self {
        let inner = (*self.inner).clone().with_retry_config(config.inner);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Return an opaque handle to the underlying HTTP client.
    ///
    /// Escape hatch for power users who need to issue raw HTTP requests
    /// (custom headers, endpoints not yet covered by Blazen's typed
    /// surface, debugging) while reusing the same connection pool, TLS
    /// config, and timeouts as this provider.
    pub fn http_client(&self) -> PyHttpClientHandle {
        PyHttpClientHandle {
            inner: self.inner.http_client(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AnthropicProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}
