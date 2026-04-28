//! Python wrapper for the local candle embedding model.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::CandleEmbedError;
use crate::providers::options::PyCandleEmbedOptions;
use crate::types::embedding::PyEmbeddingResponse;
use blazen_llm::CandleEmbedModel;
use blazen_llm::traits::EmbeddingModel;

// ---------------------------------------------------------------------------
// PyCandleEmbedModel
// ---------------------------------------------------------------------------

/// A local candle text embedding model.
///
/// Runs embedding inference fully on-device using the candle (HuggingFace)
/// engine. No API key is required.
///
/// Example:
///     >>> opts = CandleEmbedOptions(model_id="BAAI/bge-small-en-v1.5")
///     >>> model = CandleEmbedModel(options=opts)
///     >>> response = await model.embed(["Hello", "world"])
#[gen_stub_pyclass]
#[pyclass(name = "CandleEmbedModel", from_py_object)]
#[derive(Clone)]
pub struct PyCandleEmbedModel {
    inner: Arc<CandleEmbedModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCandleEmbedModel {
    /// Create a new candle embedding model.
    ///
    /// Args:
    ///     options: Optional :class:`CandleEmbedOptions` for model id,
    ///         device, revision, and cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyCandleEmbedOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        // `CandleEmbedModel::from_options` is async — bridge to the sync
        // PyO3 constructor via `block_on_context`. (Earlier sync-arm builds
        // existed when the upstream `engine` feature was off; the M3 wave
        // unified the signature to async across feature combinations.)
        let model = crate::convert::block_on_context(CandleEmbedModel::from_options(opts))
            .map_err(|e| CandleEmbedError::new_err(e.to_string()))?;
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
                .map_err(|e| CandleEmbedError::new_err(e.to_string()))?;
            Ok(PyEmbeddingResponse { inner: response })
        })
    }

    /// Load the model weights into memory / VRAM. Idempotent.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .load()
                .await
                .map_err(|e| CandleEmbedError::new_err(e.to_string()))
        })
    }

    /// Drop the loaded model and free its memory / VRAM.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .unload()
                .await
                .map_err(|e| CandleEmbedError::new_err(e.to_string()))
        })
    }

    /// Whether the model is currently loaded.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.bool]", imports = ("typing", "builtins")))]
    fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(inner.is_loaded().await) })
    }

    fn __repr__(&self) -> String {
        format!(
            "CandleEmbedModel(model_id='{}', dimensions={})",
            EmbeddingModel::model_id(self.inner.as_ref()),
            EmbeddingModel::dimensions(self.inner.as_ref()),
        )
    }
}
