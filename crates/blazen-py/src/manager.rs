//! Python binding for the memory-budget-aware model manager.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::BlazenPyError;
use blazen_llm::Pool;
use blazen_manager::ModelManager;

// ---------------------------------------------------------------------------
// Pool label parsing
// ---------------------------------------------------------------------------

/// Parse a pool label string into a [`Pool`].
///
/// Accepted forms (case-insensitive):
///
/// | Input        | Result          |
/// |--------------|-----------------|
/// | `"cpu"`      | `Pool::Cpu`     |
/// | `"gpu"`      | `Pool::Gpu(0)`  |
/// | `"gpu:0"`    | `Pool::Gpu(0)`  |
/// | `"gpu:3"`    | `Pool::Gpu(3)`  |
///
/// Anything else raises ``ValueError``.
fn parse_pool_label(s: &str) -> PyResult<Pool> {
    let trimmed = s.trim();
    let lower = trimmed.to_ascii_lowercase();

    if let Some((name, idx)) = lower.split_once(':') {
        match name {
            "gpu" => {
                let index = idx.parse::<usize>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
                    ))
                })?;
                Ok(Pool::Gpu(index))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
            ))),
        }
    } else {
        match lower.as_str() {
            "cpu" => Ok(Pool::Cpu),
            "gpu" => Ok(Pool::Gpu(0)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// PyLocalModelWrapper -- bridges a duck-typed Python object into LocalModel
// ---------------------------------------------------------------------------

/// A Python-implemented local model that satisfies the Rust [`LocalModel`]
/// trait via duck typing.
///
/// The wrapped object must have callable `load` and `unload` attributes
/// (either synchronous functions or async coroutine functions). It MAY also
/// have `is_loaded` and `memory_bytes` methods, and a (sync) `device` method
/// returning a string like `"cpu"` or `"cuda:0"`. Async methods are detected
/// at wrap construction time and awaited via the `pyo3-async-runtimes` bridge.
///
/// [`LocalModel`]: blazen_llm::LocalModel
#[allow(clippy::struct_excessive_bools)]
struct PyLocalModelWrapper {
    obj: Py<PyAny>,
    load_is_async: bool,
    unload_is_async: bool,
    is_loaded_is_async: bool,
    memory_bytes_is_async: bool,
    has_is_loaded: bool,
    has_memory_bytes: bool,
    memory_estimate_bytes: u64,
    device: blazen_llm::Device,
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

    fn device(&self) -> blazen_llm::Device {
        self.device
    }

    async fn memory_bytes(&self) -> Option<u64> {
        if !self.has_memory_bytes {
            return Some(self.memory_estimate_bytes);
        }
        match self
            .call_method("memory_bytes", self.memory_bytes_is_async)
            .await
        {
            Ok(result) => Python::attach(|py| result.extract::<Option<u64>>(py)).unwrap_or_else(
                |e| {
                    tracing::warn!(
                        "PyLocalModelWrapper: memory_bytes() returned non-int / non-None or raised: {e}"
                    );
                    Some(self.memory_estimate_bytes)
                },
            ),
            Err(e) => {
                let downgraded = Python::attach(|py| {
                    e.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py)
                        || e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py)
                });
                if !downgraded {
                    tracing::warn!("PyLocalModelWrapper: memory_bytes() raised: {e}");
                }
                Some(self.memory_estimate_bytes)
            }
        }
    }
}

/// Build a [`PyLocalModelWrapper`] from a Python object, validating that
/// `load` and `unload` are present and callable, and detecting async-ness for
/// each method up front.
///
/// `is_loaded`, `memory_bytes`, and `device` are optional; their presence is
/// detected by trying `getattr` *and* (for ABC subclasses) checking that the
/// method has been overridden relative to `blazen.LocalModel`'s stubs. Final
/// fallback is catching `NotImplementedError` / `AttributeError` at call time.
///
/// `device` is probed synchronously at construction time: if the Python object
/// exposes a callable `device()` returning a parseable device string, that
/// device is stored; otherwise we fall back to the explicit `device` argument
/// (typically `Device::Cpu`).
fn build_local_model_wrapper(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    memory_estimate_bytes: u64,
    device: blazen_llm::Device,
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

    // Determine whether the user overrode `is_loaded` / `memory_bytes` /
    // `device`.
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
    let (has_memory_bytes, memory_bytes_is_async) = is_user_override("memory_bytes");

    // Probe the Python object for a callable `device()` returning a string.
    // Sync-only call by design -- keeps the LocalModel::device() impl sync.
    // Any failure (missing, raised NotImplementedError/AttributeError, junk
    // return value) falls back to the explicit `device` argument.
    let (has_device, _device_is_async) = is_user_override("device");
    let device = if has_device {
        match obj.call_method0("device") {
            Ok(result) => match result.extract::<String>() {
                Ok(s) => blazen_llm::Device::parse(&s).unwrap_or(device),
                Err(_) => device,
            },
            Err(_) => device,
        }
    } else {
        device
    };

    Ok(PyLocalModelWrapper {
        obj: obj.clone().unbind(),
        load_is_async,
        unload_is_async,
        is_loaded_is_async,
        memory_bytes_is_async,
        has_is_loaded,
        has_memory_bytes,
        memory_estimate_bytes,
        device,
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
    memory_estimate_bytes: u64,
    pool: Pool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelStatus {
    /// The memory pool this model targets.
    ///
    /// Returned as a string label: ``"cpu"`` for host RAM, ``"gpu:N"`` for
    /// the GPU at device index ``N``.
    #[getter]
    fn pool(&self) -> String {
        format!("{}", self.pool)
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelStatus(id={:?}, loaded={}, memory_estimate_bytes={}, pool={:?})",
            self.id,
            self.loaded,
            self.memory_estimate_bytes,
            format!("{}", self.pool)
        )
    }
}

/// Memory-budget-aware model manager with LRU eviction.
///
/// Tracks registered local models and their estimated memory footprint per
/// pool (host RAM or per-GPU VRAM). When loading a model that would exceed
/// the pool's budget, the least-recently-used loaded model in the same pool
/// is unloaded first.
///
/// Example:
///     >>> manager = ModelManager(cpu_ram_gb=64, gpu_vram_gb=24)
///     >>> await manager.register("llm", my_local_model)
///     >>> await manager.load("llm")
///     >>> await manager.is_loaded("llm")
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
    ///     cpu_ram_gb: Host RAM budget for the CPU pool in gigabytes.
    ///     gpu_vram_gb: VRAM budget for the GPU pool (device 0) in gigabytes.
    ///     pool_budgets: Explicit per-pool budgets as a dict mapping pool
    ///         labels (``"cpu"``, ``"gpu"``, ``"gpu:0"``, ``"gpu:1"``, ...)
    ///         to budget sizes in bytes. Takes precedence over
    ///         ``cpu_ram_gb`` / ``gpu_vram_gb`` when provided.
    ///
    /// When all three arguments are ``None``, the manager is seeded with
    /// ``Pool::Cpu`` and ``Pool::Gpu(0)`` BOTH set to ``u64::MAX``, which
    /// matches the "no budget enforcement" intent used by tests and
    /// runtime-unconstrained environments.
    #[new]
    #[pyo3(signature = (*, cpu_ram_gb=None, gpu_vram_gb=None, pool_budgets=None))]
    fn new(
        cpu_ram_gb: Option<f64>,
        gpu_vram_gb: Option<f64>,
        pool_budgets: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let inner = if let Some(budgets) = pool_budgets {
            let mut map: HashMap<Pool, u64> = HashMap::new();
            for (key, value) in budgets.iter() {
                let label: String = key.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "pool_budgets keys must be strings (pool labels like 'cpu', 'gpu', 'gpu:0')",
                    )
                })?;
                let bytes: u64 = value.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "pool_budgets values must be non-negative integers (budgets in bytes)",
                    )
                })?;
                let pool = parse_pool_label(&label)?;
                map.insert(pool, bytes);
            }
            ModelManager::new(map)
        } else if cpu_ram_gb.is_none() && gpu_vram_gb.is_none() {
            // Sentinel "unlimited" mode: seed both Cpu and Gpu(0) with u64::MAX
            // so tests that don't care about budget enforcement can construct
            // `ModelManager()` and freely register/load on either pool.
            let mut map: HashMap<Pool, u64> = HashMap::new();
            map.insert(Pool::Cpu, u64::MAX);
            map.insert(Pool::Gpu(0), u64::MAX);
            ModelManager::new(map)
        } else {
            ModelManager::with_budgets_gb(cpu_ram_gb.unwrap_or(0.0), gpu_vram_gb.unwrap_or(0.0))
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Register a model with its estimated memory footprint.
    ///
    /// The `model` argument can be either:
    ///
    /// * A :class:`CompletionModel` produced by a local factory (e.g.
    ///   ``CompletionModel.mistralrs(...)``), in which case its built-in
    ///   ``LocalModel`` implementation is used.
    /// * Any Python object exposing callable ``load`` and ``unload``
    ///   attributes (sync or async). It does *not* need to subclass
    ///   :class:`blazen.LocalModel`. ``is_loaded``, ``memory_bytes``, and
    ///   ``device`` are optional; if absent or unimplemented the manager
    ///   falls back to ``False``, ``memory_estimate_bytes``, and ``Cpu``
    ///   respectively.
    ///
    /// Args:
    ///     id: A unique identifier for this model.
    ///     model: A CompletionModel with local model support, or a duck-typed
    ///         object with ``load`` and ``unload`` methods.
    ///     memory_estimate_bytes: Estimated memory footprint in bytes (host
    ///         RAM if the model targets CPU, GPU VRAM otherwise).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    #[pyo3(signature = (id, model, *, memory_estimate_bytes=None))]
    fn register<'py>(
        &self,
        py: Python<'py>,
        id: String,
        model: &Bound<'py, PyAny>,
        memory_estimate_bytes: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let memory_estimate = memory_estimate_bytes.unwrap_or(0);

        // First, see if the user passed a CompletionModel that already carries
        // a LocalModel impl. If so, use it directly (its device() is already
        // implemented in Rust).
        let local_model: Arc<dyn blazen_llm::LocalModel> =
            if let Ok(cm) = model.extract::<PyRef<'_, crate::providers::PyCompletionModel>>() {
                if let Some(lm) = cm.local_model.clone() {
                    lm
                } else {
                    // CompletionModel without local support -- fall back to duck
                    // typing on the same object (rare, but harmless).
                    drop(cm);
                    Arc::new(build_local_model_wrapper(
                        py,
                        model,
                        memory_estimate,
                        blazen_llm::Device::Cpu,
                    )?)
                }
            } else {
                // Arbitrary Python object: duck-type via load/unload, probe
                // device() for an explicit target.
                Arc::new(build_local_model_wrapper(
                    py,
                    model,
                    memory_estimate,
                    blazen_llm::Device::Cpu,
                )?)
            };

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.register(&id, local_model, memory_estimate).await;
            Ok(())
        })
    }

    /// Load a model, evicting LRU models in the same pool if needed.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.load(&id).await.map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// Unload a model, freeing its memory budget.
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

    /// Total memory currently used by loaded models in the given pool.
    ///
    /// Args:
    ///     pool: Pool label (``"cpu"``, ``"gpu"``, or ``"gpu:N"``).
    ///         Defaults to ``"cpu"``.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, int]", imports = ("typing",)))]
    #[pyo3(signature = (pool=None))]
    fn used_bytes<'py>(
        &self,
        py: Python<'py>,
        pool: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pool = parse_pool_label(pool.as_deref().unwrap_or("cpu"))?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move { Ok(inner.used_bytes(pool).await) },
        )
    }

    /// Available memory within the given pool's budget.
    ///
    /// Args:
    ///     pool: Pool label (``"cpu"``, ``"gpu"``, or ``"gpu:N"``).
    ///         Defaults to ``"cpu"``.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, int]", imports = ("typing",)))]
    #[pyo3(signature = (pool=None))]
    fn available_bytes<'py>(
        &self,
        py: Python<'py>,
        pool: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pool = parse_pool_label(pool.as_deref().unwrap_or("cpu"))?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(inner.available_bytes(pool).await)
        })
    }

    /// List configured pools and their budgets.
    ///
    /// Returns:
    ///     A list of ``(pool_label, budget_bytes)`` tuples, one per
    ///     configured pool.
    fn pools(&self) -> Vec<(String, u64)> {
        self.inner
            .pools()
            .into_iter()
            .map(|(pool, budget)| (format!("{pool}"), budget))
            .collect()
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
                    memory_estimate_bytes: s.memory_estimate_bytes,
                    pool: s.pool,
                })
                .collect::<Vec<_>>())
        })
    }
}
