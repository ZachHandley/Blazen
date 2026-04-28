//! Python wrappers for the generic OpenAI-compatible provider, its
//! configuration type, and the [`AuthMethod`] discriminator.
//!
//! Use this when targeting any OpenAI-compatible service that does not have
//! a dedicated provider class (e.g. private gateways, vLLM hosts, Azure-style
//! deployments). For supported public providers (Groq, OpenRouter, Together,
//! ...) prefer the dedicated `*Provider` class.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::completion_model::{
    LazyStreamState, PendingStream, PyCompletionOptions, PyLazyCompletionStream, build_request,
};
use crate::types::embedding::PyEmbeddingModel;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::providers::openai_compat::{
    AuthMethod, OpenAiCompatConfig, OpenAiCompatEmbeddingModel, OpenAiCompatProvider,
};
use blazen_llm::traits::CompletionModel;

// ---------------------------------------------------------------------------
// PyAuthMethod
// ---------------------------------------------------------------------------

/// How to send the API key for an OpenAI-compatible provider.
///
/// - ``Bearer`` → ``Authorization: Bearer <key>``
/// - ``ApiKeyHeader`` → custom header (set ``api_key_header`` on
///   [`OpenAiCompatConfig`])
/// - ``AzureApiKey`` → ``api-key: <key>``
/// - ``KeyPrefix`` → ``Authorization: Key <key>``
#[gen_stub_pyclass_enum]
#[pyclass(name = "AuthMethod", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyAuthMethod {
    Bearer,
    ApiKeyHeader,
    AzureApiKey,
    KeyPrefix,
}

// ---------------------------------------------------------------------------
// PyOpenAiCompatConfig
// ---------------------------------------------------------------------------

/// Configuration for a generic OpenAI-compatible provider.
///
/// Example:
///     >>> cfg = OpenAiCompatConfig(
///     ...     provider_name="my-gateway",
///     ...     base_url="https://gateway.example.com/v1",
///     ...     api_key="secret",
///     ...     default_model="gpt-4o",
///     ... )
///     >>> p = OpenAiCompatProvider(config=cfg)
#[gen_stub_pyclass]
#[pyclass(name = "OpenAiCompatConfig", from_py_object)]
#[derive(Clone)]
pub struct PyOpenAiCompatConfig {
    pub(crate) inner: OpenAiCompatConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAiCompatConfig {
    /// Create a new OpenAiCompatConfig.
    ///
    /// Args:
    ///     provider_name: Human-readable name (used in logs and model info).
    ///     base_url: API base URL (e.g. ``"https://api.openai.com/v1"``).
    ///     api_key: API key string.
    ///     default_model: Default model id when requests don't override it.
    ///     auth_method: One of [`AuthMethod`] (default ``Bearer``).
    ///     api_key_header: Header name when ``auth_method`` is ``ApiKeyHeader``.
    ///     extra_headers: Extra static headers to send on every request.
    ///     query_params: Extra static query parameters to attach.
    ///     supports_model_listing: Whether the provider supports ``GET /models``.
    #[new]
    #[pyo3(signature = (
        *,
        provider_name,
        base_url,
        api_key,
        default_model,
        auth_method=None,
        api_key_header=None,
        extra_headers=None,
        query_params=None,
        supports_model_listing=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        provider_name: String,
        base_url: String,
        api_key: String,
        default_model: String,
        auth_method: Option<PyAuthMethod>,
        api_key_header: Option<String>,
        extra_headers: Option<Vec<(String, String)>>,
        query_params: Option<Vec<(String, String)>>,
        supports_model_listing: bool,
    ) -> PyResult<Self> {
        let auth = match auth_method.unwrap_or(PyAuthMethod::Bearer) {
            PyAuthMethod::Bearer => AuthMethod::Bearer,
            PyAuthMethod::ApiKeyHeader => {
                let header = api_key_header.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "AuthMethod.ApiKeyHeader requires api_key_header to be set",
                    )
                })?;
                AuthMethod::ApiKeyHeader(header)
            }
            PyAuthMethod::AzureApiKey => AuthMethod::AzureApiKey,
            PyAuthMethod::KeyPrefix => AuthMethod::KeyPrefix,
        };
        Ok(Self {
            inner: OpenAiCompatConfig {
                provider_name,
                base_url,
                api_key,
                default_model,
                auth_method: auth,
                extra_headers: extra_headers.unwrap_or_default(),
                query_params: query_params.unwrap_or_default(),
                supports_model_listing,
            },
        })
    }

    #[getter]
    fn provider_name(&self) -> &str {
        &self.inner.provider_name
    }

    #[getter]
    fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    #[getter]
    fn default_model(&self) -> &str {
        &self.inner.default_model
    }

    fn __repr__(&self) -> String {
        format!(
            "OpenAiCompatConfig(provider_name={:?}, base_url={:?}, default_model={:?})",
            self.inner.provider_name, self.inner.base_url, self.inner.default_model
        )
    }
}

// ---------------------------------------------------------------------------
// PyOpenAiCompatProvider
// ---------------------------------------------------------------------------

/// A generic OpenAI-compatible chat completion provider.
///
/// Use this for any OpenAI-compatible service without a dedicated
/// provider class. For known services (Groq, OpenRouter, Together, ...)
/// prefer the dedicated wrapper which preconfigures the URL/auth.
///
/// Example:
///     >>> cfg = OpenAiCompatConfig(
///     ...     provider_name="vllm-host",
///     ...     base_url="http://localhost:8000/v1",
///     ...     api_key="",
///     ...     default_model="meta-llama/Llama-3.1-8B-Instruct",
///     ... )
///     >>> p = OpenAiCompatProvider(config=cfg)
///     >>> resp = await p.complete([ChatMessage.user("Hi!")])
#[gen_stub_pyclass]
#[pyclass(name = "OpenAiCompatProvider", from_py_object)]
#[derive(Clone)]
pub struct PyOpenAiCompatProvider {
    inner: Arc<OpenAiCompatProvider>,
    config: OpenAiCompatConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAiCompatProvider {
    /// Create a new OpenAI-compatible provider.
    ///
    /// Args:
    ///     config: A fully-specified [`OpenAiCompatConfig`].
    #[new]
    #[pyo3(signature = (*, config))]
    fn new(config: PyRef<'_, PyOpenAiCompatConfig>) -> Self {
        let cfg = config.inner.clone();
        Self {
            inner: Arc::new(OpenAiCompatProvider::new(cfg.clone())),
            config: cfg,
        }
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

    /// Build an embedding model sharing this provider's configuration.
    ///
    /// Args:
    ///     model: Embedding model id (e.g. ``"text-embedding-3-small"``).
    ///     dimensions: Output dimensionality of the embedding vectors.
    #[pyo3(signature = (*, model, dimensions))]
    fn embedding_model(&self, model: String, dimensions: usize) -> PyEmbeddingModel {
        let em = OpenAiCompatEmbeddingModel::new(self.config.clone(), model, dimensions);
        PyEmbeddingModel::from_arc(Arc::new(em))
    }

    fn __repr__(&self) -> String {
        format!(
            "OpenAiCompatProvider(provider_name={:?}, model_id='{}')",
            self.config.provider_name,
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}

// ---------------------------------------------------------------------------
// PyOpenAiCompatEmbeddingModel
// ---------------------------------------------------------------------------

/// A generic OpenAI-compatible embedding model.
///
/// Constructed via [`OpenAiCompatProvider.embedding_model`] in normal use.
/// Exposed as a free-standing class for parity with other providers.
///
/// Example:
///     >>> cfg = OpenAiCompatConfig(...)
///     >>> em = OpenAiCompatEmbeddingModel(
///     ...     config=cfg,
///     ...     model="text-embedding-3-small",
///     ...     dimensions=1536,
///     ... )
///     >>> resp = await em.embed(["hello"])
#[gen_stub_pyclass]
#[pyclass(name = "OpenAiCompatEmbeddingModel", from_py_object)]
#[derive(Clone)]
pub struct PyOpenAiCompatEmbeddingModel {
    inner: Arc<OpenAiCompatEmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAiCompatEmbeddingModel {
    #[new]
    #[pyo3(signature = (*, config, model, dimensions))]
    fn new(config: PyRef<'_, PyOpenAiCompatConfig>, model: String, dimensions: usize) -> Self {
        let em = OpenAiCompatEmbeddingModel::new(config.inner.clone(), model, dimensions);
        Self {
            inner: Arc::new(em),
        }
    }

    #[getter]
    fn model_id(&self) -> &str {
        blazen_llm::traits::EmbeddingModel::model_id(self.inner.as_ref())
    }

    #[getter]
    fn dimensions(&self) -> usize {
        blazen_llm::traits::EmbeddingModel::dimensions(self.inner.as_ref())
    }

    /// Embed one or more texts.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = blazen_llm::traits::EmbeddingModel::embed(inner.as_ref(), &texts)
                .await
                .map_err(crate::error::BlazenPyError::from)?;
            Ok(crate::types::PyEmbeddingResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "OpenAiCompatEmbeddingModel(model_id='{}', dimensions={})",
            blazen_llm::traits::EmbeddingModel::model_id(self.inner.as_ref()),
            blazen_llm::traits::EmbeddingModel::dimensions(self.inner.as_ref()),
        )
    }
}
