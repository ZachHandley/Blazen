//! Python wrapper for the local fastembed (ONNX Runtime) embedding model.
//!
//! Only available on non-musl targets where Microsoft's prebuilt ONNX
//! Runtime binaries can link. On musl targets, use :class:`TractEmbedModel`.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::FastEmbedError;
use crate::providers::options::PyFastEmbedOptions;
use crate::types::embedding::PyEmbeddingResponse;
use blazen_llm::EmbedModel;
use blazen_llm::traits::EmbeddingModel;

// ---------------------------------------------------------------------------
// PyFastEmbedModel
// ---------------------------------------------------------------------------

/// A local fastembed (ONNX Runtime) embedding model.
///
/// Loads the same fastembed model catalog that backs
/// :class:`EmbeddingModel.local`, but exposes the typed standalone class
/// for callers that want explicit feature gating or per-instance options.
///
/// Example:
///     >>> opts = FastEmbedOptions(model_name="BGESmallENV15")
///     >>> model = FastEmbedModel(options=opts)
///     >>> response = await model.embed(["hello", "world"])
#[gen_stub_pyclass]
#[pyclass(name = "FastEmbedModel", from_py_object)]
#[derive(Clone)]
pub struct PyFastEmbedModel {
    inner: Arc<EmbedModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFastEmbedModel {
    /// Create a new fastembed model.
    ///
    /// Args:
    ///     options: Optional :class:`FastEmbedOptions` for model name,
    ///         cache directory, batch size, and download progress.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyFastEmbedOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let model =
            EmbedModel::from_options(opts).map_err(|e| FastEmbedError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Get the model identifier (HuggingFace repo id).
    #[getter]
    fn model_id(&self) -> String {
        EmbeddingModel::model_id(self.inner.as_ref()).to_owned()
    }

    /// Get the dimensionality of the produced embedding vectors.
    #[getter]
    fn dimensions(&self) -> usize {
        EmbeddingModel::dimensions(self.inner.as_ref())
    }

    /// Embed one or more texts.
    ///
    /// Args:
    ///     texts: A list of strings to embed.
    ///
    /// Returns:
    ///     An :class:`EmbeddingResponse` with embeddings, model, and usage.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = EmbeddingModel::embed(inner.as_ref(), &texts)
                .await
                .map_err(|e| FastEmbedError::new_err(e.to_string()))?;
            Ok(PyEmbeddingResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "FastEmbedModel(model_id='{}', dimensions={})",
            EmbeddingModel::model_id(self.inner.as_ref()),
            EmbeddingModel::dimensions(self.inner.as_ref()),
        )
    }
}
