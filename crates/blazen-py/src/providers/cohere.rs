//! Python wrapper for the Cohere provider.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::completion_model::{
    LazyStreamState, PendingStream, PyCompletionOptions, PyLazyCompletionStream, build_request,
};
use crate::providers::options::PyProviderOptions;
use crate::types::embedding::PyEmbeddingModel;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::providers::cohere::CohereProvider;
use blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel;
use blazen_llm::traits::CompletionModel;

/// A Cohere provider for chat completions and embeddings.
///
/// Standalone class form of `CompletionModel.cohere(...)`.
#[gen_stub_pyclass]
#[pyclass(name = "CohereProvider", from_py_object)]
#[derive(Clone)]
pub struct PyCohereProvider {
    inner: Arc<CohereProvider>,
    options: blazen_llm::types::provider_options::ProviderOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCohereProvider {
    /// Create a new Cohere provider.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                CohereProvider::from_options(opts.clone()).map_err(blazen_error_to_pyerr)?,
            ),
            options: opts,
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

    /// Build a Cohere [`EmbeddingModel`] sharing this provider's API key.
    ///
    /// Defaults to ``embed-v4.0`` (1024 dims).
    fn embedding_model(&self) -> PyResult<PyEmbeddingModel> {
        let model =
            OpenAiCompatEmbeddingModel::embedding_from_options("cohere", self.options.clone())
                .map_err(blazen_error_to_pyerr)?;
        Ok(PyEmbeddingModel::from_arc(Arc::new(model)))
    }

    fn __repr__(&self) -> String {
        format!(
            "CohereProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}
