//! Python wrapper for the Azure OpenAI provider.
//!
//! Exposes [`AzureOpenAiProvider`](blazen_llm::providers::azure::AzureOpenAiProvider)
//! to Python with chat-completion capabilities.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::config::PyRetryConfig;
use crate::providers::model::{
    LazyStreamState, PendingStream, PyLazyCompletionStream, PyModelOptions, build_request,
};
use crate::providers::options::PyAzureOptions;
use crate::types::{PyChatMessage, PyHttpClientHandle, PyModelResponse};
use blazen_llm::ChatMessage;
use blazen_llm::providers::azure::AzureOpenAiProvider;
use blazen_llm::traits::Model;

/// An Azure OpenAI Service provider.
///
/// Standalone class form of `Model.azure(...)`. Requires
/// `resource_name` and `deployment_name` on [`AzureOptions`].
///
/// Example:
/// ```text
///  >>> from blazen import AzureOpenAiProvider, AzureOptions
///  >>> p = AzureOpenAiProvider(options=AzureOptions(
///  ...     resource_name="my-resource",
///  ...     deployment_name="gpt-4o",
///  ... ))
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "AzureOpenAiProvider", from_py_object)]
#[derive(Clone)]
pub struct PyAzureOpenAiProvider {
    inner: Arc<AzureOpenAiProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAzureOpenAiProvider {
    /// Create a new Azure OpenAI provider.
    ///
    /// Args:
    ///     options: [`AzureOptions`] with required ``resource_name`` and
    ///         ``deployment_name`` plus optional ``api_version``.
    #[new]
    #[pyo3(signature = (*, options))]
    fn new(options: PyRef<'_, PyAzureOptions>) -> PyResult<Self> {
        let opts = options.inner.clone();
        Ok(Self {
            inner: Arc::new(
                AzureOpenAiProvider::from_options(opts).map_err(blazen_error_to_pyerr)?,
            ),
        })
    }

    #[getter]
    fn model_id(&self) -> &str {
        Model::model_id(self.inner.as_ref())
    }

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

    fn __repr__(&self) -> String {
        format!(
            "AzureOpenAiProvider(model_id='{}')",
            Model::model_id(self.inner.as_ref())
        )
    }
}
