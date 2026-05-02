//! Python wrappers for embedding model and response types.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::{BlazenPyError, blazen_error_to_pyerr};
#[cfg(feature = "embed")]
use crate::providers::options::PyEmbedOptions;
use crate::providers::options::PyProviderOptions;
use crate::types::request_timing::PyRequestTiming;
use blazen_llm::EmbeddingModel;
use blazen_llm::keys::resolve_api_key;
use blazen_llm::types::EmbeddingResponse;

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
#[pyclass(name = "EmbeddingModel", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyEmbeddingModel {
    pub(crate) inner: Option<Arc<dyn EmbeddingModel>>,
    /// Configuration for subclassed models. `None` for built-in providers
    /// (whose config lives inside the `inner` trait object).
    pub(crate) config: Option<blazen_llm::ProviderConfig>,
    /// Stored separately because `ProviderConfig` has no `dimensions` field.
    pub(crate) dimensions_override: Option<usize>,
}

impl PyEmbeddingModel {
    /// Build a `PyEmbeddingModel` from a fully-constructed Rust model.
    ///
    /// Used by sibling provider wrappers (e.g. [`PyTogetherProvider`])
    /// that expose an `embedding_model()` factory.
    pub(crate) fn from_arc(inner: Arc<dyn EmbeddingModel>) -> Self {
        Self {
            inner: Some(inner),
            config: None,
            dimensions_override: None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingModel {
    // -----------------------------------------------------------------
    // Subclass constructor
    // -----------------------------------------------------------------

    /// Create a custom embedding model by subclassing.
    ///
    /// Override ``embed()`` in your subclass to implement a custom
    /// embedding provider.
    ///
    /// Args:
    ///     model_id: The model identifier.
    ///     dimensions: Output dimensionality of the embedding vectors.
    ///     base_url: Base URL for HTTP-based providers.
    ///     pricing: Optional pricing information.
    ///     vram_estimate_bytes: Estimated VRAM footprint in bytes.
    /// `__new__` for `EmbeddingModel`. Accepts arbitrary positional and
    /// keyword arguments so Python subclasses can use any `__init__`
    /// signature; the real configuration happens in `__init__` below.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        Self {
            inner: None,
            config: Some(blazen_llm::ProviderConfig::default()),
            dimensions_override: None,
        }
    }

    /// Subclass-friendly `__init__`. Mirrors the documented constructor
    /// keyword signature and re-populates ``self.config`` so a Python
    /// subclass that calls `super().__init__(model_id=..., dimensions=...)`
    /// sees the values it passed.
    #[pyo3(signature = (*, model_id=None, dimensions=None, base_url=None, pricing=None, vram_estimate_bytes=None))]
    fn __init__(
        &mut self,
        model_id: Option<String>,
        dimensions: Option<usize>,
        base_url: Option<String>,
        pricing: Option<PyRef<'_, crate::types::pricing::PyModelPricing>>,
        vram_estimate_bytes: Option<u64>,
    ) {
        if self.inner.is_none() {
            self.config = Some(blazen_llm::ProviderConfig {
                model_id,
                context_length: dimensions.map(|d| d as u64),
                base_url,
                vram_estimate_bytes,
                pricing: pricing.map(|p| p.inner.clone()),
                ..Default::default()
            });
            self.dimensions_override = dimensions;
        }
    }

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
            inner: Some(Arc::new(provider)),
            config: None,
            dimensions_override: None,
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
            inner: Some(Arc::new(provider)),
            config: None,
            dimensions_override: None,
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
            inner: Some(Arc::new(provider)),
            config: None,
            dimensions_override: None,
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
            inner: Some(Arc::new(provider)),
            config: None,
            dimensions_override: None,
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
    fn model_id(&self) -> String {
        if let Some(ref inner) = self.inner {
            inner.model_id().to_owned()
        } else {
            self.config
                .as_ref()
                .and_then(|c| c.model_id.clone())
                .unwrap_or_default()
        }
    }

    /// Get the output dimensionality of the embedding model.
    ///
    /// Returns:
    ///     The number of dimensions in the output vectors.
    #[getter]
    fn dimensions(&self) -> usize {
        if let Some(ref inner) = self.inner {
            inner.dimensions()
        } else {
            self.dimensions_override.unwrap_or(0)
        }
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
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(ref inner) = self.inner {
            let inner = inner.clone();
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let response = EmbeddingModel::embed(inner.as_ref(), &texts)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyEmbeddingResponse { inner: response })
            })
        } else {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override embed()",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingModel(model_id='{}', dimensions={})",
            self.model_id(),
            self.dimensions()
        )
    }
}

// ---------------------------------------------------------------------------
// Feature-gated local embedding factory (separate impl block so pyo3-stub-gen
// does not try to resolve the option type when the feature is disabled).
// The Rust fn is named `local` (not `embed`) to avoid collision with the
// instance `embed` method on the same class; pyo3 rejects same-name members.
// ---------------------------------------------------------------------------

#[cfg(feature = "embed")]
#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingModel {
    /// Create a local embedding model (ONNX Runtime, no API key required).
    ///
    /// Args:
    ///     options: Optional typed ``EmbedOptions`` object.
    ///
    /// Example:
    ///     >>> model = EmbeddingModel.local()
    ///     >>> model = EmbeddingModel.local(options=EmbedOptions(model_name="BGESmallENV15"))
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn local(options: Option<PyRef<'_, PyEmbedOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let model = blazen_llm::EmbedModel::from_options(opts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Some(Arc::new(model)),
            config: None,
            dimensions_override: None,
        })
    }
}

// ---------------------------------------------------------------------------
// PyEmbeddingResponse
// ---------------------------------------------------------------------------

/// Response from an embedding operation.
///
/// Contains the embedding vectors, model name, usage statistics, cost,
/// timing, and provider-specific metadata.
///
/// Example:
///     >>> response = await model.embed(["Hello", "World"])
///     >>> print(response.model)
///     >>> print(len(response.embeddings))  # 2
#[gen_stub_pyclass]
#[pyclass(name = "EmbeddingResponse")]
pub struct PyEmbeddingResponse {
    pub(crate) inner: EmbeddingResponse,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingResponse {
    /// The embedding vectors, one per input text.
    #[getter]
    fn embeddings(&self) -> Vec<Vec<f32>> {
        self.inner.embeddings.clone()
    }

    /// The model used to generate the embeddings.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Token usage statistics, if provided by the model.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[dict[str, typing.Any]]", imports = ("typing",)))]
    fn usage(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.usage {
            Some(u) => Ok(Some(pythonize::pythonize(py, u)?.unbind())),
            None => Ok(None),
        }
    }

    /// Estimated cost in USD, if available.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Request timing breakdown (queue, execution, total).
    #[getter]
    fn timing(&self) -> Option<PyRequestTiming> {
        self.inner
            .timing
            .clone()
            .map(|t| PyRequestTiming { inner: t })
    }

    /// Provider-specific metadata as a Python dict.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingResponse(model='{}', embeddings={})",
            self.inner.model,
            self.inner.embeddings.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Python-subclass adapter
// ---------------------------------------------------------------------------

/// Bridges a Python `EmbeddingModel` subclass into the Rust
/// [`blazen_llm::EmbeddingModel`] trait.
///
/// When a user subclasses `EmbeddingModel` in Python and overrides
/// `embed()`, the `PyEmbeddingModel.inner` field is `None` because there
/// is no Rust provider backing the instance. To make such subclasses usable
/// with Rust-side consumers like [`blazen_memory::Memory`], this adapter
/// holds the Python object itself and dispatches `embed()` calls back into
/// Python via `pyo3_async_runtimes`.
///
/// The dispatch follows the same three-phase pattern as
/// [`crate::providers::custom::PyHostDispatch`]:
///
/// 1. Under the GIL, call the Python subclass's `embed()` method to obtain
///    an awaitable coroutine, capture the active asyncio task locals, and
///    convert the coroutine into a Rust future.
/// 2. Outside the GIL (inside `pyo3_async_runtimes::tokio::scope`), drive
///    the future to completion so the Python coroutine runs on the correct
///    event loop.
/// 3. Under the GIL again, extract the returned `PyEmbeddingResponse`
///    (preferred path) or fall back to `depythonize` if the subclass
///    returns a compatible dict.
pub(crate) struct PySubclassEmbeddingModel {
    py_obj: Py<PyAny>,
    model_id: String,
    dimensions: usize,
}

impl PySubclassEmbeddingModel {
    pub(crate) fn new(py_obj: Py<PyAny>, model_id: String, dimensions: usize) -> Self {
        Self {
            py_obj,
            model_id,
            dimensions,
        }
    }
}

#[async_trait::async_trait]
impl blazen_llm::EmbeddingModel for PySubclassEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(
        &self,
        texts: &[String],
    ) -> Result<blazen_llm::EmbeddingResponse, blazen_llm::BlazenError> {
        let texts_vec = texts.to_vec();

        // Phase 1: under GIL, invoke the Python `embed` method to get a
        // coroutine and convert it into a Rust future.
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("embed", (texts_vec,)).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "subclass embed() raised before yielding a coroutine: {e}"
                    ))
                })?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut =
                    pyo3_async_runtimes::into_future_with_locals(&locals, coro).map_err(|e| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "subclass embed() must be an async def returning a coroutine: {e}"
                        ))
                    })?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            blazen_llm::BlazenError::provider("subclass", format!("dispatch setup failed: {e}"))
        })?;

        // Phase 2: drive the Python coroutine to completion.
        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| {
                blazen_llm::BlazenError::provider(
                    "subclass",
                    format!("subclass embed() raised: {e}"),
                )
            })?;

        // Phase 3: reacquire GIL and extract the EmbeddingResponse.
        tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<blazen_llm::EmbeddingResponse> {
                let bound = py_result.bind(py);
                // Preferred path: the subclass returned a PyEmbeddingResponse.
                if let Ok(resp) = bound.extract::<PyRef<'_, PyEmbeddingResponse>>() {
                    return Ok(resp.inner.clone());
                }
                // Fallback: allow a compatible dict shape.
                let response: blazen_llm::EmbeddingResponse = pythonize::depythonize(bound)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "subclass embed() must return EmbeddingResponse or a compatible dict: {e}"
                        ))
                    })?;
                Ok(response)
            })
        })
        .map_err(|e: PyErr| {
            blazen_llm::BlazenError::provider(
                "subclass",
                format!("failed to decode subclass embed() result: {e}"),
            )
        })
    }
}
