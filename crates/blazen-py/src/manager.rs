//! Python binding for the VRAM-aware model manager.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;

use crate::error::BlazenPyError;
use blazen_manager::ModelManager;

// ---------------------------------------------------------------------------
// PyLocalModelWrapper -- bridges a duck-typed Python object into LocalModel
// ---------------------------------------------------------------------------

/// A Python-implemented local model that satisfies the Rust [`LocalModel`]
/// trait via duck typing.
///
/// The wrapped object must have callable `load` and `unload` attributes
/// (either synchronous functions or async coroutine functions). It MAY also
/// have `is_loaded` and `vram_bytes` methods. Async methods are detected at
/// wrap construction time and awaited via the `pyo3-async-runtimes` bridge.
///
/// [`LocalModel`]: blazen_llm::LocalModel
#[allow(clippy::struct_excessive_bools)]
struct PyLocalModelWrapper {
    obj: Py<PyAny>,
    load_is_async: bool,
    unload_is_async: bool,
    is_loaded_is_async: bool,
    vram_bytes_is_async: bool,
    has_is_loaded: bool,
    has_vram_bytes: bool,
    vram_estimate: u64,
}

impl PyLocalModelWrapper {
    /// Call a no-arg method on the wrapped Python object, transparently
    /// awaiting the result if `is_async` is true.
    ///
    /// For async methods, [`pyo3_async_runtimes::tokio::get_current_locals`]
    /// is captured at call time (not at registration time) so the awaited
    /// coroutine is driven on the caller's currently-running asyncio event
    /// loop rather than whichever loop happened to be active when the model
    /// was registered.
    ///
    /// The returned `Py<PyAny>` is the resolved value of the (possibly async)
    /// method call.
    async fn call_method(&self, name: &'static str, is_async: bool) -> PyResult<Py<PyAny>> {
        if is_async {
            let (fut, locals) = tokio::task::block_in_place(|| {
                Python::attach(|py| -> PyResult<_> {
                    let coro = self.obj.bind(py).call_method0(name)?;
                    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                    let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                    Ok((fut, locals))
                })
            })?;

            let py_result = pyo3_async_runtimes::tokio::scope(locals, fut).await?;
            Ok(py_result)
        } else {
            let py_result = tokio::task::block_in_place(|| {
                Python::attach(|py| self.obj.call_method0(py, name))
            })?;
            Ok(py_result)
        }
    }
}

#[async_trait::async_trait]
impl blazen_llm::LocalModel for PyLocalModelWrapper {
    async fn load(&self) -> Result<(), blazen_llm::BlazenError> {
        self.call_method("load", self.load_is_async)
            .await
            .map(|_| ())
            .map_err(|e| blazen_llm::BlazenError::provider("python_local_model", e.to_string()))
    }

    async fn unload(&self) -> Result<(), blazen_llm::BlazenError> {
        self.call_method("unload", self.unload_is_async)
            .await
            .map(|_| ())
            .map_err(|e| blazen_llm::BlazenError::provider("python_local_model", e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        if !self.has_is_loaded {
            return false;
        }
        match self.call_method("is_loaded", self.is_loaded_is_async).await {
            Ok(result) => Python::attach(|py| result.extract::<bool>(py)).unwrap_or_else(|e| {
                tracing::warn!("PyLocalModelWrapper: is_loaded() returned non-bool or raised: {e}");
                false
            }),
            Err(e) => {
                // Treat NotImplementedError / AttributeError as "not overridden".
                let downgraded = Python::attach(|py| {
                    e.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py)
                        || e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py)
                });
                if !downgraded {
                    tracing::warn!("PyLocalModelWrapper: is_loaded() raised: {e}");
                }
                false
            }
        }
    }

    async fn vram_bytes(&self) -> Option<u64> {
        if !self.has_vram_bytes {
            return Some(self.vram_estimate);
        }
        match self
            .call_method("vram_bytes", self.vram_bytes_is_async)
            .await
        {
            Ok(result) => Python::attach(|py| result.extract::<Option<u64>>(py)).unwrap_or_else(
                |e| {
                    tracing::warn!(
                        "PyLocalModelWrapper: vram_bytes() returned non-int / non-None or raised: {e}"
                    );
                    Some(self.vram_estimate)
                },
            ),
            Err(e) => {
                let downgraded = Python::attach(|py| {
                    e.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py)
                        || e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py)
                });
                if !downgraded {
                    tracing::warn!("PyLocalModelWrapper: vram_bytes() raised: {e}");
                }
                Some(self.vram_estimate)
            }
        }
    }
}

/// Build a [`PyLocalModelWrapper`] from a Python object, validating that
/// `load` and `unload` are present and callable, and detecting async-ness for
/// each method up front.
///
/// `is_loaded` and `vram_bytes` are optional; their presence is detected by
/// trying `getattr` *and* (for ABC subclasses) checking that the method has
/// been overridden relative to `blazen.LocalModel`'s stubs. Final fallback is
/// catching `NotImplementedError` / `AttributeError` at call time.
fn build_local_model_wrapper(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    vram_estimate: u64,
) -> PyResult<PyLocalModelWrapper> {
    let inspect = py.import("inspect")?;

    // Validate `load` and `unload` exist and are callable.
    let load_attr = obj.getattr("load").map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "model object must have a callable `load` method (or be a CompletionModel with local model support)",
        )
    })?;
    if !load_attr.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "model.load must be callable",
        ));
    }

    let unload_attr = obj.getattr("unload").map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("model object must have a callable `unload` method")
    })?;
    if !unload_attr.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "model.unload must be callable",
        ));
    }

    let detect_async = |attr: &Bound<'_, PyAny>| -> bool {
        inspect
            .call_method1("iscoroutinefunction", (attr,))
            .and_then(|r| r.extract())
            .unwrap_or(false)
    };

    let load_is_async = detect_async(&load_attr);
    let unload_is_async = detect_async(&unload_attr);

    // Determine whether the user overrode `is_loaded` / `vram_bytes`.
    //
    // For plain duck-typed objects (no inheritance from `blazen.LocalModel`),
    // we treat the attribute's presence as "overridden". For ABC subclasses,
    // compare the bound method's underlying function against the ABC default.
    let abc_class = py
        .import("blazen")
        .ok()
        .and_then(|m| m.getattr("LocalModel").ok());

    let is_user_override = |name: &str| -> (bool, bool) {
        // Returns (has_method, is_async)
        let Ok(attr) = obj.getattr(name) else {
            return (false, false);
        };
        if !attr.is_callable() {
            return (false, false);
        }

        if let Some(ref abc) = abc_class {
            // If this object is an instance of the ABC, compare the attribute
            // identity against the ABC's same-named method to detect override.
            let is_instance = obj.is_instance(abc).unwrap_or(false);
            if is_instance {
                let abc_method = abc.getattr(name).ok();
                let obj_func = attr.getattr("__func__").ok();
                if let (Some(abc_m), Some(obj_f)) = (abc_method, obj_func)
                    && abc_m.is(&obj_f)
                {
                    // Inherited the ABC default -- not overridden.
                    return (false, false);
                }
            }
        }

        let is_async = detect_async(&attr);
        (true, is_async)
    };

    let (has_is_loaded, is_loaded_is_async) = is_user_override("is_loaded");
    let (has_vram_bytes, vram_bytes_is_async) = is_user_override("vram_bytes");

    Ok(PyLocalModelWrapper {
        obj: obj.clone().unbind(),
        load_is_async,
        unload_is_async,
        is_loaded_is_async,
        vram_bytes_is_async,
        has_is_loaded,
        has_vram_bytes,
        vram_estimate,
    })
}

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
    ///
    /// When neither is provided, the budget is unlimited (`u64::MAX` bytes),
    /// useful for tests and runtime-unconstrained environments.
    #[new]
    #[pyo3(signature = (*, budget_gb=None, budget_bytes=None))]
    fn new(budget_gb: Option<f64>, budget_bytes: Option<u64>) -> Self {
        let bytes = match (budget_gb, budget_bytes) {
            (Some(gb), _) => {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    (gb * 1_073_741_824.0) as u64
                }
            }
            (_, Some(b)) => b,
            (None, None) => u64::MAX,
        };
        Self {
            inner: Arc::new(ModelManager::new(bytes)),
        }
    }

    /// Register a model with its estimated VRAM footprint.
    ///
    /// The `model` argument can be either:
    ///
    /// * A :class:`CompletionModel` produced by a local factory (e.g.
    ///   ``CompletionModel.mistralrs(...)``), in which case its built-in
    ///   ``LocalModel`` implementation is used.
    /// * Any Python object exposing callable ``load`` and ``unload``
    ///   attributes (sync or async). It does *not* need to subclass
    ///   :class:`blazen.LocalModel`. ``is_loaded`` and ``vram_bytes`` are
    ///   optional; if absent or unimplemented the manager falls back to
    ///   ``False`` and ``vram_estimate_bytes`` respectively.
    ///
    /// Args:
    ///     id: A unique identifier for this model.
    ///     model: A CompletionModel with local model support, or a duck-typed
    ///         object with ``load`` and ``unload`` methods.
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
        let vram = vram_estimate_bytes.unwrap_or(0);

        // First, see if the user passed a CompletionModel that already carries
        // a LocalModel impl. If so, use it directly.
        let local_model: Arc<dyn blazen_llm::LocalModel> =
            if let Ok(cm) = model.extract::<PyRef<'_, crate::providers::PyCompletionModel>>() {
                if let Some(lm) = cm.local_model.clone() {
                    lm
                } else {
                    // CompletionModel without local support -- fall back to duck
                    // typing on the same object (rare, but harmless).
                    drop(cm);
                    Arc::new(build_local_model_wrapper(py, model, vram)?)
                }
            } else {
                // Arbitrary Python object: duck-type via load/unload.
                Arc::new(build_local_model_wrapper(py, model, vram)?)
            };

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
