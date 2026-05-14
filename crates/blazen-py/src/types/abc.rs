//! Subclassable Python ABCs that mirror provider-extension Rust traits.
//!
//! These ABCs let Python code implement the role of a Rust trait without
//! the Python implementation actually being plugged into the Rust runtime.
//! They serve as type-checked subclassable bases so callers can express
//! "implements ProviderInfo / Tool / LocalModel / ..." in Python without
//! losing static help.
//!
//! Where a Python implementation does need to drive the Rust loop, dedicated
//! adapters exist already (see e.g. `PyToolWrapper` in `crate::agent` which
//! bridges a Python `ToolDef` callable to the `Tool` trait, or
//! `PyHostCheckpointStore` in `crate::persist`). These ABCs document the
//! *interface* that those adapters delegate to.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::types::{StructuredResponse, TokenUsage};

use super::{PyArtifact, PyCitation, PyReasoningTrace, PyRequestTiming, PyTokenUsage};

// ---------------------------------------------------------------------------
// PyTool (ABC)
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring the Rust [`blazen_llm::traits::Tool`] trait.
///
/// Implement to declare the shape of a tool ahead of registration; tools that
/// actually drive the agent loop should use [`ToolDef`](crate::agent::PyToolDef)
/// which carries a callable handler. Override every method -- the defaults
/// raise ``NotImplementedError``.
#[gen_stub_pyclass]
#[pyclass(name = "Tool", subclass)]
pub struct PyTool;

#[gen_stub_pymethods]
#[pymethods]
impl PyTool {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Return the tool's [`ToolDefinition`] (name, description, JSON schema).
    fn definition(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override definition()",
        ))
    }

    /// Execute the tool with a JSON-serializable arguments object. Should
    /// return a [`ToolOutput`] (or a value coercible into one).
    fn execute(&self, _arguments: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override execute()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyLocalModel (ABC)
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring [`blazen_llm::traits::LocalModel`].
///
/// Implement when authoring a Python-side wrapper for an in-process model
/// backend that supports explicit ``load`` / ``unload``. Override every
/// method -- the defaults raise ``NotImplementedError``.
#[gen_stub_pyclass]
#[pyclass(name = "LocalModel", subclass)]
pub struct PyLocalModel;

#[gen_stub_pymethods]
#[pymethods]
impl PyLocalModel {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Load the model into memory / VRAM. Idempotent.
    fn load(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override load()",
        ))
    }

    /// Drop the loaded model. Idempotent.
    fn unload(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override unload()",
        ))
    }

    /// Whether the model is currently loaded.
    fn is_loaded(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override is_loaded()",
        ))
    }

    /// Approximate VRAM footprint in bytes, if available.
    fn vram_bytes(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override vram_bytes()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyStructuredOutput (ABC)
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring [`blazen_llm::traits::StructuredOutput`].
///
/// The Rust trait has a blanket implementation for every `CompletionModel`,
/// so most callers just call ``model.complete(messages, options)`` with a
/// [`ResponseFormat.json_schema(...)`](crate::types::PyResponseFormat) hint.
/// This ABC lets a Python provider explicitly opt in to a typed
/// ``extract(messages)`` entry point.
#[gen_stub_pyclass]
#[pyclass(name = "StructuredOutput", subclass)]
pub struct PyStructuredOutput;

#[gen_stub_pymethods]
#[pymethods]
impl PyStructuredOutput {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Extract a structured value, returning a [`StructuredResponse`].
    fn extract(&self, _messages: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override extract()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyModelRegistry (ABC)
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring [`blazen_llm::traits::ModelRegistry`].
///
/// Implement to expose a custom model-listing capability from a Python
/// provider. Override both methods -- the defaults raise
/// ``NotImplementedError``.
#[gen_stub_pyclass]
#[pyclass(name = "ModelRegistry", subclass)]
pub struct PyModelRegistry;

#[gen_stub_pymethods]
#[pymethods]
impl PyModelRegistry {
    #[new]
    fn new() -> Self {
        Self
    }

    /// List all models. Should return a coroutine resolving to
    /// ``list[ModelInfo]``.
    fn list_models(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override list_models()",
        ))
    }

    /// Look up a model by id. Should return a coroutine resolving to an
    /// optional ``ModelInfo``.
    fn get_model(&self, _model_id: String) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override get_model()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyImageModel (ABC)
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring the Rust
/// [`blazen_llm::compute::traits::ImageModel`] trait.
///
/// ``ImageModel`` is the supertrait alias used by the compute layer for
/// any provider that supports both image generation and upscaling. Bind
/// this when authoring a custom image provider in pure Python that
/// wants to declare "implements ImageModel" without coupling to a
/// specific :class:`ImageProvider` subclass. Override both methods --
/// the defaults raise ``NotImplementedError``.
#[gen_stub_pyclass]
#[pyclass(name = "ImageModel", subclass)]
pub struct PyImageModel;

#[gen_stub_pymethods]
#[pymethods]
impl PyImageModel {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Generate one or more images from a text prompt. Should return a
    /// coroutine resolving to an :class:`ImageResult`.
    fn generate_image(&self, _request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override generate_image()",
        ))
    }

    /// Upscale an existing image. Should return a coroutine resolving
    /// to an :class:`ImageResult`.
    fn upscale_image(&self, _request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override upscale_image()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyStructuredResponse
// ---------------------------------------------------------------------------

/// Result of [`StructuredOutput.extract(...)`].
///
/// Mirrors [`blazen_llm::StructuredResponse`] with the type parameter
/// erased to a Python value (each provider deserializes into its preferred
/// shape -- typically a dict or a Pydantic model -- and stuffs it into
/// ``data``).
#[gen_stub_pyclass]
#[pyclass(name = "StructuredResponse", from_py_object)]
#[derive(Clone)]
pub struct PyStructuredResponse {
    pub(crate) inner: StructuredResponse<serde_json::Value>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStructuredResponse {
    /// Construct a structured response.
    #[new]
    #[pyo3(signature = (
        *,
        data,
        model,
        usage=None,
        cost=None,
        timing=None,
        metadata=None,
    ))]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        model: String,
        usage: Option<PyTokenUsage>,
        cost: Option<f64>,
        timing: Option<PyRequestTiming>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let data_value = crate::convert::py_to_json(py, data)?;
        let metadata_value = match metadata {
            Some(m) => crate::convert::py_to_json(py, m)?,
            None => serde_json::Value::Null,
        };
        let usage_inner: Option<TokenUsage> = usage.map(|u| u.inner);
        Ok(Self {
            inner: StructuredResponse {
                data: data_value,
                usage: usage_inner,
                model,
                cost,
                timing: timing.map(|t| t.inner),
                metadata: metadata_value,
                reasoning: None,
                citations: Vec::new(),
                artifacts: Vec::new(),
            },
        })
    }

    /// The extracted structured value.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.data)
    }

    /// The model that produced the response.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Token usage for the request.
    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.as_ref().map(PyTokenUsage::from)
    }

    /// Estimated cost in USD.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Request timing.
    #[getter]
    fn timing(&self) -> Option<PyRequestTiming> {
        self.inner
            .timing
            .as_ref()
            .map(|t| PyRequestTiming { inner: t.clone() })
    }

    /// Provider-specific metadata.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    /// Reasoning trace, if exposed by the provider.
    #[getter]
    fn reasoning(&self) -> Option<PyReasoningTrace> {
        self.inner.reasoning.as_ref().map(PyReasoningTrace::from)
    }

    /// Web/document citations backing the response.
    #[getter]
    fn citations(&self) -> Vec<PyCitation> {
        self.inner.citations.iter().map(PyCitation::from).collect()
    }

    /// Inline artifacts extracted from the response.
    #[getter]
    fn artifacts(&self) -> Vec<PyArtifact> {
        self.inner.artifacts.iter().map(PyArtifact::from).collect()
    }

    fn __repr__(&self) -> String {
        format!("StructuredResponse(model={:?})", self.inner.model)
    }
}

impl From<StructuredResponse<serde_json::Value>> for PyStructuredResponse {
    fn from(inner: StructuredResponse<serde_json::Value>) -> Self {
        Self { inner }
    }
}
