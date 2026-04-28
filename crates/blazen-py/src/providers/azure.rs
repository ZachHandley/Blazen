//! Python wrapper for the Azure OpenAI provider.
//!
//! Exposes [`AzureOpenAiProvider`](blazen_llm::providers::azure::AzureOpenAiProvider)
//! to Python with chat-completion capabilities.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::completion_model::{
    LazyStreamState, PendingStream, PyCompletionOptions, PyLazyCompletionStream, build_request,
};
use crate::providers::options::PyAzureOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::providers::azure::AzureOpenAiProvider;
use blazen_llm::traits::CompletionModel;

/// An Azure OpenAI Service provider.
///
/// Standalone class form of `CompletionModel.azure(...)`. Requires
/// `resource_name` and `deployment_name` on [`AzureOptions`].
///
/// Example:
///     >>> from blazen import AzureOpenAiProvider, AzureOptions
///     >>> p = AzureOpenAiProvider(options=AzureOptions(
///     ...     resource_name="my-resource",
///     ...     deployment_name="gpt-4o",
///     ... ))
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
        CompletionModel::model_id(self.inner.as_ref())
    }

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

    fn __repr__(&self) -> String {
        format!(
            "AzureOpenAiProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}
