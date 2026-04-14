//! Python binding for the VRAM-aware model manager.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;

use crate::error::BlazenPyError;
use blazen_manager::ModelManager;

/// Status of a registered model.
#[gen_stub_pyclass]
#[pyclass(name = "ModelStatus", frozen)]
pub struct PyModelStatus {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    loaded: bool,
    #[pyo3(get)]
    vram_estimate: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelStatus {
    fn __repr__(&self) -> String {
        format!(
            "ModelStatus(id={:?}, loaded={}, vram_estimate={})",
            self.id, self.loaded, self.vram_estimate
        )
    }
}

/// VRAM budget-aware model manager with LRU eviction.
///
/// Tracks registered local models and their estimated VRAM footprint.
/// When loading a model that would exceed the budget, the
/// least-recently-used loaded model is unloaded first.
///
/// Example:
///     >>> manager = ModelManager(budget_gb=24)
///     >>> manager.register("llm", my_local_model)
///     >>> await manager.load("llm")
///     >>> manager.is_loaded("llm")
///     True
#[gen_stub_pyclass]
#[pyclass(name = "ModelManager")]
pub struct PyModelManager {
    inner: Arc<ModelManager>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelManager {
    /// Create a new model manager.
    ///
    /// Args:
    ///     budget_gb: VRAM budget in gigabytes.
    ///     budget_bytes: VRAM budget in bytes (alternative to budget_gb).
    #[new]
    #[pyo3(signature = (*, budget_gb=None, budget_bytes=None))]
    fn new(budget_gb: Option<f64>, budget_bytes: Option<u64>) -> PyResult<Self> {
        let bytes = match (budget_gb, budget_bytes) {
            (Some(gb), _) => {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    (gb * 1_073_741_824.0) as u64
                }
            }
            (_, Some(b)) => b,
            (None, None) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "must provide either budget_gb or budget_bytes",
                ));
            }
        };
        Ok(Self {
            inner: Arc::new(ModelManager::new(bytes)),
        })
    }

    /// Register a model with its estimated VRAM footprint.
    ///
    /// The model must be a local provider (e.g. created via
    /// ``CompletionModel.mistralrs(...)`` or similar local factory).
    ///
    /// Args:
    ///     id: A unique identifier for this model.
    ///     model: A CompletionModel with local model support.
    ///     vram_estimate_bytes: Estimated VRAM footprint in bytes.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    #[pyo3(signature = (id, model, vram_estimate_bytes=None))]
    fn register<'py>(
        &self,
        py: Python<'py>,
        id: String,
        model: &Bound<'py, PyAny>,
        vram_estimate_bytes: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Try to extract a CompletionModel and get its local_model
        let local_model =
            if let Ok(cm) = model.extract::<PyRef<'_, crate::providers::PyCompletionModel>>() {
                cm.local_model.clone().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "model does not support local loading (no LocalModel implementation)",
                    )
                })?
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "expected a CompletionModel with local model support",
                ));
            };

        let vram = vram_estimate_bytes.unwrap_or(0);
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.register(&id, local_model, vram).await;
            Ok(())
        })
    }

    /// Load a model, evicting LRU models if needed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.load(&id).await.map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Unload a model, freeing its VRAM budget.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.unload(&id).await.map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Check if a model is currently loaded.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, bool]", imports = ("typing",)))]
    fn is_loaded<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move { Ok(inner.is_loaded(&id).await) },
        )
    }

    /// Ensure a model is loaded (load if not, update LRU if already loaded).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn ensure_loaded<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .ensure_loaded(&id)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Total VRAM currently used by loaded models.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, int]", imports = ("typing",)))]
    fn used_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(inner.used_bytes().await) })
    }

    /// Available VRAM within the budget.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, int]", imports = ("typing",)))]
    fn available_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move { Ok(inner.available_bytes().await) },
        )
    }

    /// Status of all registered models.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, list[ModelStatus]]", imports = ("typing",)))]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let statuses = inner.status().await;
            Ok(statuses
                .into_iter()
                .map(|s| PyModelStatus {
                    id: s.id,
                    loaded: s.loaded,
                    vram_estimate: s.vram_estimate,
                })
                .collect::<Vec<_>>())
        })
    }
}
