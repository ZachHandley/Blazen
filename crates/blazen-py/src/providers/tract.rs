//! Python wrapper for the local tract (pure-Rust ONNX) embedding model.
//!
//! Drop-in equivalent of :class:`FastEmbedModel` for environments where
//! the prebuilt ONNX Runtime binaries that back fastembed cannot be
//! linked (musl-libc Linux distributions, sandboxed/wasm-style targets,
//! ...). Loads the same fastembed model catalog via tract-onnx instead.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_embed_tract::{TractEmbedModel, TractResponse};

use crate::error::TractError;
use crate::providers::options::PyTractOptions;

// ---------------------------------------------------------------------------
// PyTractResponse
// ---------------------------------------------------------------------------

/// Result of a [`TractEmbedModel.embed`] call.
///
/// Mirrors [`blazen_embed_tract::TractResponse`] -- the embedding vectors
/// plus the model id that produced them. Provided as a typed class so
/// callers that gate on the ``tract`` feature can keep their type
/// annotations precise; for cross-backend code prefer the umbrella
/// :class:`EmbeddingResponse` returned by :class:`EmbeddingModel`.
#[gen_stub_pyclass]
#[pyclass(name = "TractResponse", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTractResponse {
    pub(crate) inner: TractResponse,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTractResponse {
    /// One L2-normalized embedding vector per input text.
    #[getter]
    fn embeddings(&self) -> Vec<Vec<f32>> {
        self.inner.embeddings.clone()
    }

    /// The model identifier (Hugging Face repo id) that produced these
    /// embeddings.
    #[getter]
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TractResponse(model={:?}, count={})",
            self.inner.model,
            self.inner.embeddings.len()
        )
    }
}

impl From<TractResponse> for PyTractResponse {
    fn from(inner: TractResponse) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyTractEmbedModel
// ---------------------------------------------------------------------------

/// A local tract (pure-Rust ONNX) embedding model.
///
/// Loads the same fastembed model catalog via tract-onnx instead of the
/// ONNX Runtime native library. Use this when the ``fastembed`` feature
/// is unavailable for your target -- typically musl-libc Linux builds.
///
/// Example:
///     >>> opts = TractOptions(model_name="BGESmallENV15")
///     >>> model = TractEmbedModel(options=opts)
///     >>> response = await model.embed(["hello", "world"])
#[gen_stub_pyclass]
#[pyclass(name = "TractEmbedModel", from_py_object)]
#[derive(Clone)]
pub struct PyTractEmbedModel {
    inner: Arc<TractEmbedModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTractEmbedModel {
    /// Create a new tract embedding model.
    ///
    /// Args:
    ///     options: Optional :class:`TractOptions` for model name, cache
    ///         directory, batch size, and download progress.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyTractOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let model =
            TractEmbedModel::from_options(opts).map_err(|e| TractError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Get the model identifier (Hugging Face repo id).
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Get the dimensionality of the produced embedding vectors.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Embed one or more texts.
    ///
    /// Args:
    ///     texts: A list of strings to embed.
    ///
    /// Returns:
    ///     A :class:`TractResponse` with embeddings and model id.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, TractResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = inner
                .embed(&texts)
                .await
                .map_err(|e| TractError::new_err(e.to_string()))?;
            Ok(PyTractResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TractEmbedModel(model_id='{}', dimensions={})",
            self.inner.model_id(),
            self.inner.dimensions(),
        )
    }
}
