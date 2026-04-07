//! Python wrappers for embedding model and response types.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::EmbeddingModel;
use blazen_llm::providers::openai_compat::{AuthMethod, OpenAiCompatConfig};

use crate::error::BlazenPyError;

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
        let provider = blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::new(
            OpenAiCompatConfig {
                provider_name: "together".into(),
                base_url: "https://api.together.xyz/v1".into(),
                api_key: api_key.into(),
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "togethercomputer/m2-bert-80M-8k-retrieval",
            768,
        );
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
        let provider = blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::new(
            OpenAiCompatConfig {
                provider_name: "cohere".into(),
                base_url: "https://api.cohere.ai/compatibility/v1".into(),
                api_key: api_key.into(),
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "embed-v4.0",
            1024,
        );
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
        let provider = blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::new(
            OpenAiCompatConfig {
                provider_name: "fireworks".into(),
                base_url: "https://api.fireworks.ai/inference/v1".into(),
                api_key: api_key.into(),
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "nomic-ai/nomic-embed-text-v1.5",
            768,
        );
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

pub use blazen_llm::types::EmbeddingResponse;
