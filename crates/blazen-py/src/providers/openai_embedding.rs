//! Python wrapper for the OpenAI embedding model.
//!
//! Standalone class form of [`EmbeddingModel.openai`]. Wraps
//! [`OpenAiEmbeddingModel`](blazen_llm::providers::openai::OpenAiEmbeddingModel).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::{BlazenPyError, blazen_error_to_pyerr};
use crate::providers::config::PyRetryConfig;
use crate::providers::options::PyProviderOptions;
use crate::types::{PyEmbeddingResponse, PyHttpClientHandle};
use blazen_llm::providers::openai::OpenAiEmbeddingModel;
use blazen_llm::traits::EmbeddingModel;

/// An OpenAI embedding model.
///
/// Example:
///     >>> from blazen import OpenAiEmbeddingModel, ProviderOptions
///     >>> em = OpenAiEmbeddingModel(options=ProviderOptions(api_key="sk-..."))
///     >>> resp = await em.embed(["hello", "world"])
#[gen_stub_pyclass]
#[pyclass(name = "OpenAiEmbeddingModel", from_py_object)]
#[derive(Clone)]
pub struct PyOpenAiEmbeddingModel {
    inner: Arc<OpenAiEmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAiEmbeddingModel {
    /// Create a new OpenAI embedding model.
    ///
    /// Args:
    ///     options: Optional [`ProviderOptions`] with api_key, base_url, model.
    ///     model: Model id override (default: ``"text-embedding-3-small"``).
    ///     dimensions: Output dimensionality (default: 1536).
    #[new]
    #[pyo3(signature = (*, options=None, model=None, dimensions=None))]
    fn new(
        options: Option<PyRef<'_, PyProviderOptions>>,
        model: Option<String>,
        dimensions: Option<usize>,
    ) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let mut em = OpenAiEmbeddingModel::from_options(opts).map_err(blazen_error_to_pyerr)?;
        match (model, dimensions) {
            (Some(m), Some(d)) => em = em.with_model(m, d),
            (Some(m), None) => em = em.with_model(m, 1536),
            (None, Some(d)) => em = em.with_model("text-embedding-3-small", d),
            (None, None) => {}
        }
        Ok(Self {
            inner: Arc::new(em),
        })
    }

    #[getter]
    fn model_id(&self) -> &str {
        EmbeddingModel::model_id(self.inner.as_ref())
    }

    #[getter]
    fn dimensions(&self) -> usize {
        EmbeddingModel::dimensions(self.inner.as_ref())
    }

    /// Embed one or more texts.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = EmbeddingModel::embed(inner.as_ref(), &texts)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(PyEmbeddingResponse { inner: response })
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

    fn __repr__(&self) -> String {
        format!(
            "OpenAiEmbeddingModel(model_id='{}', dimensions={})",
            EmbeddingModel::model_id(self.inner.as_ref()),
            EmbeddingModel::dimensions(self.inner.as_ref()),
        )
    }
}
