//! Python wrappers for embedding model and response types.

use std::sync::Arc;

use pyo3::prelude::*;

use blazen_llm::EmbeddingModel;

use crate::error::BlazenPyError;
use crate::types::usage::{PyRequestTiming, PyTokenUsage};

// ---------------------------------------------------------------------------
// PyEmbeddingModel
// ---------------------------------------------------------------------------

/// A text embedding model.
///
/// Use the static constructor methods to create a model for a specific
/// provider, then call `embed()` to generate embeddings.
///
/// Example:
///     >>> model = EmbeddingModel.openai("sk-...")
///     >>> response = await model.embed(["Hello", "World"])
///     >>> print(response.embeddings)
#[pyclass(name = "EmbeddingModel", from_py_object)]
#[derive(Clone)]
pub struct PyEmbeddingModel {
    pub(crate) inner: Arc<dyn EmbeddingModel>,
}

#[pymethods]
impl PyEmbeddingModel {
    // -----------------------------------------------------------------
    // Provider constructors
    // -----------------------------------------------------------------

    /// Create an OpenAI embedding model.
    ///
    /// Args:
    ///     api_key: Your OpenAI API key.
    ///     model: Optional model name (default: "text-embedding-3-small").
    ///     dimensions: Optional output dimensions (default: 1536).
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None, dimensions=None))]
    fn openai(api_key: &str, model: Option<&str>, dimensions: Option<usize>) -> Self {
        let mut provider = blazen_llm::providers::openai::OpenAiEmbeddingModel::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m, dimensions.unwrap_or(1536));
        } else if let Some(d) = dimensions {
            provider = provider.with_model("text-embedding-3-small", d);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Together AI embedding model.
    ///
    /// Args:
    ///     api_key: Your Together API key.
    #[staticmethod]
    fn together(api_key: &str) -> Self {
        let provider =
            blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::together(api_key);
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Cohere embedding model.
    ///
    /// Args:
    ///     api_key: Your Cohere API key.
    #[staticmethod]
    fn cohere(api_key: &str) -> Self {
        let provider =
            blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::cohere(api_key);
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Fireworks AI embedding model.
    ///
    /// Args:
    ///     api_key: Your Fireworks API key.
    #[staticmethod]
    fn fireworks(api_key: &str) -> Self {
        let provider =
            blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::fireworks(api_key);
        Self {
            inner: Arc::new(provider),
        }
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
            let response = inner.embed(&texts).await.map_err(BlazenPyError::from)?;
            Ok(PyEmbeddingResponse { inner: response })
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
// PyEmbeddingResponse
// ---------------------------------------------------------------------------

/// The result of an embedding operation.
///
/// Example:
///     >>> response = await model.embed(["Hello", "World"])
///     >>> response.embeddings  # [[0.1, 0.2, ...], [0.3, 0.4, ...]]
///     >>> response.model       # "text-embedding-3-small"
///     >>> response.usage       # TokenUsage(...)
///     >>> response.cost        # 0.0001
#[pyclass(name = "EmbeddingResponse", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyEmbeddingResponse {
    pub(crate) inner: blazen_llm::EmbeddingResponse,
}

#[pymethods]
impl PyEmbeddingResponse {
    /// The embedding vectors -- one per input text.
    #[getter]
    fn embeddings(&self) -> Vec<Vec<f32>> {
        self.inner.embeddings.clone()
    }

    /// The model that produced this response.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Token usage statistics, if provided by the API.
    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.clone().map(|u| PyTokenUsage { inner: u })
    }

    /// Estimated cost for this request in USD, if available.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Request timing breakdown, if available.
    #[getter]
    fn timing(&self) -> Option<PyRequestTiming> {
        self.inner
            .timing
            .clone()
            .map(|t| PyRequestTiming { inner: t })
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingResponse(model='{}', num_embeddings={})",
            self.inner.model,
            self.inner.embeddings.len()
        )
    }
}
