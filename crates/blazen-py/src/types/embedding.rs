//! Python wrappers for embedding model and response types.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::{BlazenPyError, blazen_error_to_pyerr};
#[cfg(feature = "fastembed")]
use crate::providers::options::PyFastEmbedOptions;
use crate::providers::options::PyProviderOptions;
use blazen_llm::EmbeddingModel;
use blazen_llm::keys::resolve_api_key;

// ---------------------------------------------------------------------------
// PyEmbeddingModel
// ---------------------------------------------------------------------------

/// A text embedding model.
///
/// Use the static constructor methods to create a model for a specific
/// provider, then call `embed()` to generate embeddings.
///
/// Example:
///     >>> model = EmbeddingModel.openai(options=ProviderOptions(api_key="sk-..."))
///     >>> response = await model.embed(["Hello", "World"])
///     >>> print(response.embeddings)
#[gen_stub_pyclass]
#[pyclass(name = "EmbeddingModel", from_py_object)]
#[derive(Clone)]
pub struct PyEmbeddingModel {
    pub(crate) inner: Arc<dyn EmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingModel {
    // -----------------------------------------------------------------
    // Provider constructors
    // -----------------------------------------------------------------

    /// Create an OpenAI embedding model.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    ///     model: Optional model name (default: "text-embedding-3-small").
    ///     dimensions: Optional output dimensions (default: 1536).
    #[staticmethod]
    #[pyo3(signature = (*, options=None, model=None, dimensions=None))]
    fn openai(
        options: Option<PyRef<'_, PyProviderOptions>>,
        model: Option<&str>,
        dimensions: Option<usize>,
    ) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let api_key = resolve_api_key("openai", opts.api_key).map_err(blazen_error_to_pyerr)?;
        let mut provider = blazen_llm::providers::openai::OpenAiEmbeddingModel::new(&api_key);
        if let Some(m) = model {
            provider = provider.with_model(m, dimensions.unwrap_or(1536));
        } else if let Some(d) = dimensions {
            provider = provider.with_model("text-embedding-3-small", d);
        }
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Create a Together AI embedding model.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn together(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider = blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::embedding_from_options("together", opts)
            .map_err(blazen_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Create a Cohere embedding model.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn cohere(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider = blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::embedding_from_options("cohere", opts)
            .map_err(blazen_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Create a Fireworks AI embedding model.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn fireworks(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider = blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::embedding_from_options("fireworks", opts)
            .map_err(blazen_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    ///
    /// Returns:
    ///     The string identifier of the embedding model.
    #[getter]
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    /// Get the output dimensionality of the embedding model.
    ///
    /// Returns:
    ///     The number of dimensions in the output vectors.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    // -----------------------------------------------------------------
    // Embed
    // -----------------------------------------------------------------

    /// Embed one or more texts.
    ///
    /// Args:
    ///     texts: A list of strings to embed.
    ///
    /// Returns:
    ///     An EmbeddingResponse with embeddings, model, usage, and cost.
    ///
    /// Example:
    ///     >>> response = await model.embed(["Hello", "World"])
    ///     >>> print(len(response.embeddings))  # 2
    ///     >>> print(len(response.embeddings[0]))  # dimensionality
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = EmbeddingModel::embed(inner.as_ref(), &texts)
                .await
                .map_err(BlazenPyError::from)?;
            Python::attach(|py| -> PyResult<Py<PyAny>> {
                let obj = pythonize::pythonize(py, &response)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(obj.unbind())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingModel(model_id='{}', dimensions={})",
            self.inner.model_id(),
            self.inner.dimensions()
        )
    }
}

// ---------------------------------------------------------------------------
// Feature-gated fastembed factory (separate impl block so pyo3-stub-gen
// does not try to resolve the option type when the feature is disabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "fastembed")]
#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingModel {
    /// Create a local fastembed embedding model (ONNX Runtime, no API key required).
    ///
    /// Args:
    ///     options: Optional typed ``FastEmbedOptions`` object.
    ///
    /// Example:
    ///     >>> model = EmbeddingModel.fastembed()
    ///     >>> model = EmbeddingModel.fastembed(options=FastEmbedOptions(model_name="BGESmallENV15"))
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn fastembed(options: Option<PyRef<'_, PyFastEmbedOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let model = blazen_llm::FastEmbedModel::from_options(opts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }
}

pub use blazen_llm::types::EmbeddingResponse;
