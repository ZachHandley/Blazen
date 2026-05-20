//! Python binding for the memory-budget-aware model manager.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::BlazenPyError;
use blazen_llm::Pool;
use blazen_manager::ModelManager;
use blazen_manager::hf_loader::{BackendHint, HfLoadOptions};

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

    async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        options: blazen_llm::AdapterOptions,
    ) -> Result<blazen_llm::AdapterHandle, blazen_llm::BlazenError> {
        // Probe the Python object for a callable `load_adapter`. Missing /
        // non-callable falls back to `Unsupported`, mirroring the default
        // `LocalModel::load_adapter` impl but with a wrapper-specific
        // diagnostic so callers know the bridge layer (not the underlying
        // backend) rejected the verb.
        //
        // We invoke the Python method under `Python::attach`; if it returned
        // a coroutine we drive it on the caller's asyncio loop via the
        // pyo3-async bridge. Synchronous returns are extracted in-place.
        let prepared = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<Option<PreparedPyCall>> {
                let bound = self.obj.bind(py);
                let Ok(attr) = bound.getattr("load_adapter") else {
                    return Ok(None);
                };
                if !attr.is_callable() {
                    return Ok(None);
                }
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("adapter_id", &options.adapter_id)?;
                kwargs.set_item("scale", options.scale)?;
                let path_str = adapter_dir.display().to_string();
                let result = attr.call((path_str,), Some(&kwargs))?;
                prepare_call(py, result).map(Some)
            })
        })
        .map_err(|e| blazen_llm::BlazenError::provider("python_local_model", e.to_string()))?;

        let Some(prepared) = prepared else {
            return Err(blazen_llm::BlazenError::unsupported(
                "Python LocalModel does not implement load_adapter",
            ));
        };

        let py_result = await_prepared(prepared)
            .await
            .map_err(|e| blazen_llm::BlazenError::provider("python_local_model", e.to_string()))?;

        // Expected return: a string adapter_id, OR a dict-like with
        // `adapter_id` / `memory_bytes` keys. We accept either shape and
        // synthesize an `AdapterHandle` either way. The strategy is always
        // reported as `Attached` because Python wrappers don't surface
        // mount-strategy distinctions today.
        Python::attach(
            |py| -> Result<blazen_llm::AdapterHandle, blazen_llm::BlazenError> {
                let bound = py_result.bind(py);
                if let Ok(adapter_id) = bound.extract::<String>() {
                    return Ok(blazen_llm::AdapterHandle {
                        adapter_id,
                        memory_bytes: 0,
                        mount_strategy: blazen_llm::AdapterMountStrategy::Attached,
                    });
                }
                let adapter_id: String = bound
                .get_item("adapter_id")
                .map_err(|e| {
                    blazen_llm::BlazenError::provider(
                        "python_local_model",
                        format!(
                            "load_adapter return value must be a str or have an 'adapter_id' key: {e}"
                        ),
                    )
                })?
                .extract()
                .map_err(|e| {
                    blazen_llm::BlazenError::provider(
                        "python_local_model",
                        format!("load_adapter 'adapter_id' must be str: {e}"),
                    )
                })?;
                let memory_bytes: u64 = bound
                    .get_item("memory_bytes")
                    .ok()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0);
                Ok(blazen_llm::AdapterHandle {
                    adapter_id,
                    memory_bytes,
                    mount_strategy: blazen_llm::AdapterMountStrategy::Attached,
                })
            },
        )
    }

    async fn unload_adapter(
        &self,
        handle: &blazen_llm::AdapterHandle,
    ) -> Result<(), blazen_llm::BlazenError> {
        let prepared = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<Option<PreparedPyCall>> {
                let bound = self.obj.bind(py);
                let Ok(attr) = bound.getattr("unload_adapter") else {
                    return Ok(None);
                };
                if !attr.is_callable() {
                    return Ok(None);
                }
                let result = attr.call1((handle.adapter_id.clone(),))?;
                prepare_call(py, result).map(Some)
            })
        })
        .map_err(|e| blazen_llm::BlazenError::provider("python_local_model", e.to_string()))?;

        let Some(prepared) = prepared else {
            return Err(blazen_llm::BlazenError::unsupported(
                "Python LocalModel does not implement unload_adapter",
            ));
        };

        await_prepared(prepared)
            .await
            .map_err(|e| blazen_llm::BlazenError::provider("python_local_model", e.to_string()))?;
        Ok(())
    }

    async fn list_adapters(&self) -> Vec<blazen_llm::AdapterStatus> {
        let prepared = match tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<Option<PreparedPyCall>> {
                let bound = self.obj.bind(py);
                let Ok(attr) = bound.getattr("list_adapters") else {
                    return Ok(None);
                };
                if !attr.is_callable() {
                    return Ok(None);
                }
                let result = attr.call0()?;
                prepare_call(py, result).map(Some)
            })
        }) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("PyLocalModelWrapper: list_adapters() raised: {e}");
                return Vec::new();
            }
        };

        let Some(prepared) = prepared else {
            return Vec::new();
        };

        let py_result = match await_prepared(prepared).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("PyLocalModelWrapper: list_adapters() raised: {e}");
                return Vec::new();
            }
        };

        Python::attach(|py| -> Vec<blazen_llm::AdapterStatus> {
            let bound = py_result.bind(py);
            let Ok(iter) = bound.try_iter() else {
                tracing::warn!("PyLocalModelWrapper: list_adapters() did not return an iterable");
                return Vec::new();
            };
            let mut out = Vec::new();
            for item in iter {
                let Ok(item) = item else { continue };
                let adapter_id: String = match item.get_item("adapter_id").and_then(|v| v.extract())
                {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(
                            "PyLocalModelWrapper: list_adapters item missing 'adapter_id': {e}"
                        );
                        continue;
                    }
                };
                let scale: f32 = item
                    .get_item("scale")
                    .ok()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(1.0);
                let source_dir: String = item
                    .get_item("source_dir")
                    .ok()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_default();
                let memory_bytes: u64 = item
                    .get_item("memory_bytes")
                    .ok()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0);
                out.push(blazen_llm::AdapterStatus {
                    adapter_id,
                    scale,
                    source_dir: std::path::PathBuf::from(source_dir),
                    memory_bytes,
                });
            }
            out
        })
    }
}

/// Either a resolved synchronous Python return value, or a coroutine-driven
/// future paired with the asyncio task-local context it needs to run under.
enum PreparedPyCall {
    Sync(Py<PyAny>),
    Async {
        fut: std::pin::Pin<Box<dyn std::future::Future<Output = PyResult<Py<PyAny>>> + Send>>,
        locals: pyo3_async_runtimes::TaskLocals,
    },
}

/// Inspect a Python call result: if it's a coroutine, wire it into a Rust
/// future driven by the caller's asyncio loop; otherwise capture the value
/// for synchronous return.
fn prepare_call(py: Python<'_>, result: Bound<'_, PyAny>) -> PyResult<PreparedPyCall> {
    let inspect = py.import("inspect")?;
    let is_coro: bool = inspect
        .call_method1("iscoroutine", (&result,))
        .and_then(|r| r.extract())
        .unwrap_or(false);
    if is_coro {
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let fut = pyo3_async_runtimes::into_future_with_locals(&locals, result)?;
        Ok(PreparedPyCall::Async {
            fut: Box::pin(fut),
            locals,
        })
    } else {
        Ok(PreparedPyCall::Sync(result.unbind()))
    }
}

/// Resolve a [`PreparedPyCall`] uniformly, scoping the async branch onto its
/// captured task-locals so the awaited coroutine runs on the right loop.
async fn await_prepared(prepared: PreparedPyCall) -> PyResult<Py<PyAny>> {
    match prepared {
        PreparedPyCall::Sync(v) => Ok(v),
        PreparedPyCall::Async { fut, locals } => {
            pyo3_async_runtimes::tokio::scope(locals, fut).await
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

/// Snapshot of one mounted adapter.
#[gen_stub_pyclass]
#[pyclass(name = "AdapterStatus", frozen, module = "blazen", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAdapterStatus {
    #[pyo3(get)]
    pub adapter_id: String,
    #[pyo3(get)]
    pub scale: f32,
    #[pyo3(get)]
    pub source_dir: String,
    #[pyo3(get)]
    pub memory_bytes: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAdapterStatus {
    fn __repr__(&self) -> String {
        format!(
            "AdapterStatus(adapter_id={:?}, scale={}, source_dir={:?}, memory_bytes={})",
            self.adapter_id, self.scale, self.source_dir, self.memory_bytes
        )
    }
}

impl From<blazen_llm::AdapterStatus> for PyAdapterStatus {
    fn from(s: blazen_llm::AdapterStatus) -> Self {
        Self {
            adapter_id: s.adapter_id,
            scale: s.scale,
            source_dir: s.source_dir.display().to_string(),
            memory_bytes: s.memory_bytes,
        }
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

    /// Mount a PEFT-format LoRA adapter onto a registered model.
    ///
    /// The base model is loaded if necessary (via the same single-model
    /// `ensure_loaded` path as `load`). The adapter directory must contain
    /// `adapter_model.safetensors` and `adapter_config.json`; the on-disk
    /// size of those files is charged against the model's pool budget.
    ///
    /// Args:
    ///     model_id: The id of a previously-registered model.
    ///     adapter_dir: Filesystem path containing the PEFT adapter files.
    ///     adapter_id: Caller-chosen unique id for this adapter (passed back
    ///         to ``unload_adapter`` and surfaced in ``list_adapters``).
    ///     scale: Strength multiplier for the adapter delta-weights.
    ///         Defaults to ``1.0`` (full PEFT strength).
    ///
    /// Returns:
    ///     The ``adapter_id`` echoed by the backend, as a string.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, str]",
        imports = ("typing",)
    ))]
    #[pyo3(signature = (model_id, adapter_dir, *, adapter_id, scale=None))]
    fn load_adapter<'py>(
        &self,
        py: Python<'py>,
        model_id: String,
        adapter_dir: String,
        adapter_id: String,
        scale: Option<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = blazen_llm::AdapterOptions {
            adapter_id,
            scale: scale.unwrap_or(1.0),
        };
        let path = std::path::PathBuf::from(adapter_dir);
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handle = inner
                .load_adapter(&model_id, &path, options)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(handle.adapter_id)
        })
    }

    /// Unmount a previously-loaded adapter, freeing its memory budget.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, None]",
        imports = ("typing",)
    ))]
    fn unload_adapter<'py>(
        &self,
        py: Python<'py>,
        model_id: String,
        adapter_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .unload_adapter(&model_id, &adapter_id)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(())
        })
    }

    /// List adapters currently mounted on a registered model.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, list[AdapterStatus]]",
        imports = ("typing",)
    ))]
    fn list_adapters<'py>(&self, py: Python<'py>, model_id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let statuses = inner
                .list_adapters(&model_id)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(statuses
                .into_iter()
                .map(PyAdapterStatus::from)
                .collect::<Vec<_>>())
        })
    }

    /// Auto-detect the right local-inference backend for a Hugging Face repo
    /// and register the resulting model under ``id``.
    ///
    /// Probes ``GET /api/models/{repo}`` once for metadata, picks a backend
    /// per the rules documented on
    /// :func:`blazen_manager.hf_loader.choose_backend`, and computes a memory
    /// estimate by summing the chosen backend's weight files. Pass an explicit
    /// ``options.memory_estimate_bytes`` to override the estimate.
    ///
    /// Args:
    ///     id: A unique identifier for the registered model.
    ///     repo: Hugging Face repo id (``"meta-llama/Llama-3.2-1B-Instruct"``).
    ///     options: Optional :class:`HfLoadOptions` controlling backend
    ///         selection, revision, token, cache dir, device, GGUF override,
    ///         memory estimate, and target pool.
    ///
    /// Returns:
    ///     The chosen backend as a string (``"mistralrs"`` / ``"candle"`` /
    ///     ``"llamacpp"``).
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, str]",
        imports = ("typing",)
    ))]
    #[pyo3(signature = (id, repo, options=None))]
    fn load_from_hf<'py>(
        &self,
        py: Python<'py>,
        id: String,
        repo: String,
        options: Option<PyHfLoadOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let opts: HfLoadOptions = options.map(HfLoadOptions::from).unwrap_or_default();
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let backend = inner
                .load_from_hf(id, &repo, opts)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(backend.as_str().to_string())
        })
    }
}

// ---------------------------------------------------------------------------
// PyBackendHint -- frozen enum mirror of blazen_manager::hf_loader::BackendHint
// ---------------------------------------------------------------------------

/// Local inference backend identifier returned by
/// :meth:`ModelManager.load_from_hf` and accepted as
/// :attr:`HfLoadOptions.backend_hint`.
///
/// Variants:
///     * ``BackendHint.MISTRALRS`` -- mistral.rs (broad arch coverage,
///       safetensors + GGUF, vision support).
///     * ``BackendHint.CANDLE`` -- pure-Rust candle.
///     * ``BackendHint.LLAMACPP`` -- llama.cpp (GGUF only, best CPU perf).
#[gen_stub_pyclass_enum]
#[pyclass(name = "BackendHint", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyBackendHint {
    MISTRALRS,
    CANDLE,
    LLAMACPP,
}

impl From<PyBackendHint> for BackendHint {
    fn from(h: PyBackendHint) -> Self {
        match h {
            PyBackendHint::MISTRALRS => Self::Mistralrs,
            PyBackendHint::CANDLE => Self::Candle,
            PyBackendHint::LLAMACPP => Self::Llamacpp,
        }
    }
}

// ---------------------------------------------------------------------------
// PyHfLoadOptions -- mirror of blazen_manager::hf_loader::HfLoadOptions
// ---------------------------------------------------------------------------

/// Options for :meth:`ModelManager.load_from_hf`.
#[gen_stub_pyclass]
#[pyclass(name = "HfLoadOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyHfLoadOptions {
    backend_hint: Option<PyBackendHint>,
    revision: Option<String>,
    hf_token: Option<String>,
    cache_dir: Option<String>,
    device: Option<String>,
    gguf_file: Option<String>,
    memory_estimate_bytes: Option<u64>,
    pool: Option<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHfLoadOptions {
    /// Build an options bag for :meth:`ModelManager.load_from_hf`.
    ///
    /// Args:
    ///     backend_hint: Force a specific backend, skipping auto-detection.
    ///     revision: Git revision (branch / tag / commit sha).
    ///     hf_token: HF access token; falls back to ``$HF_TOKEN`` then
    ///         anonymous access when ``None``.
    ///     cache_dir: On-disk cache directory; defaults to ``$HF_HOME`` /
    ///         ``~/.cache/huggingface/``.
    ///     device: Device string (``"cpu"``, ``"cuda:0"``, ``"metal"``, ...).
    ///     gguf_file: Explicit GGUF filename for repos with multiple
    ///         quantizations.
    ///     memory_estimate_bytes: Override the auto-summed memory estimate.
    ///     pool: Target pool label (``"cpu"``, ``"gpu"``, or ``"gpu:N"``).
    #[new]
    #[pyo3(signature = (
        *,
        backend_hint = None,
        revision = None,
        hf_token = None,
        cache_dir = None,
        device = None,
        gguf_file = None,
        memory_estimate_bytes = None,
        pool = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend_hint: Option<PyBackendHint>,
        revision: Option<String>,
        hf_token: Option<String>,
        cache_dir: Option<String>,
        device: Option<String>,
        gguf_file: Option<String>,
        memory_estimate_bytes: Option<u64>,
        pool: Option<String>,
    ) -> PyResult<Self> {
        if let Some(ref label) = pool {
            parse_pool_label(label)?;
        }
        Ok(Self {
            backend_hint,
            revision,
            hf_token,
            cache_dir,
            device,
            gguf_file,
            memory_estimate_bytes,
            pool,
        })
    }
}

impl From<PyHfLoadOptions> for HfLoadOptions {
    fn from(o: PyHfLoadOptions) -> Self {
        Self {
            backend_hint: o.backend_hint.map(BackendHint::from),
            revision: o.revision,
            hf_token: o.hf_token,
            cache_dir: o.cache_dir.map(std::path::PathBuf::from),
            device: o.device,
            gguf_file: o.gguf_file,
            memory_estimate_bytes: o.memory_estimate_bytes,
            // Why: pool labels are validated in `PyHfLoadOptions::new`, so any
            // string surviving to here parses cleanly; default to CPU as a
            // belt-and-suspenders fallback if the option somehow re-enters
            // From with an invalid label (cannot happen via the pyclass ctor).
            pool: o.pool.as_deref().and_then(|s| parse_pool_label(s).ok()),
        }
    }
}
