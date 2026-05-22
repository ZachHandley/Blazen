//! Python wrapper for the Together AI provider.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::config::PyRetryConfig;
use crate::providers::model::{
    LazyStreamState, PendingStream, PyLazyCompletionStream, PyModelOptions, build_request,
};
use crate::providers::options::PyProviderOptions;
use crate::types::embedding::PyEmbeddingModel;
use crate::types::{PyChatMessage, PyHttpClientHandle, PyModelResponse};
use blazen_llm::ChatMessage;
use blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel;
use blazen_llm::providers::together::TogetherProvider;
use blazen_llm::traits::Model;

/// A Together AI provider for chat completions and embeddings.
///
/// Standalone class form of `Model.together(...)`.
#[gen_stub_pyclass]
#[pyclass(name = "TogetherProvider", from_py_object)]
#[derive(Clone)]
pub struct PyTogetherProvider {
    inner: Arc<TogetherProvider>,
    /// Cached options used to construct an embedding model on demand.
    options: blazen_llm::types::provider_options::ProviderOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTogetherProvider {
    /// Create a new Together AI provider.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                TogetherProvider::from_options(opts.clone()).map_err(blazen_error_to_pyerr)?,
            ),
            options: opts,
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

    /// Build a Together AI [`EmbeddingModel`] sharing this provider's API key.
    ///
    /// Defaults to ``togethercomputer/m2-bert-80M-8k-retrieval`` (768 dims).
    fn embedding_model(&self) -> PyResult<PyEmbeddingModel> {
        let model =
            OpenAiCompatEmbeddingModel::embedding_from_options("together", self.options.clone())
                .map_err(blazen_error_to_pyerr)?;
        Ok(PyEmbeddingModel::from_arc(Arc::new(model)))
    }

    /// Set the provider-level default retry config.
    pub fn with_retry_config(&self, config: PyRetryConfig) -> Self {
        let inner = (*self.inner).clone().with_retry_config(config.inner);
        Self {
            inner: Arc::new(inner),
            options: self.options.clone(),
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
            "TogetherProvider(model_id='{}')",
            Model::model_id(self.inner.as_ref())
        )
    }
}
