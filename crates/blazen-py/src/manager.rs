//! Python binding for the memory-budget-aware model manager.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::BlazenPyError;
use crate::providers::model::{PyModel, PyModelOptions, build_request};
use crate::types::{PyChatMessage, PyModelResponse};
use blazen_llm::Pool;
use blazen_llm::types::ChatMessage;
use blazen_manager::ModelManager;
use blazen_manager::hf_loader::{BackendHint, HfLoadOptions};
use tokio_stream::StreamExt;

/// The chat (`Model`) and local-lifecycle (`LocalModel`) trait objects pulled
/// out of a Python [`PyModel`] at registration time. Either may be `None`.
type ModelArcs = (
    Option<Arc<dyn blazen_llm::Model>>,
    Option<Arc<dyn blazen_llm::LocalModel>>,
);

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
        self.device.clone()
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
            "model object must have a callable `load` method (or be a Model with local model support)",
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
/// ```text
///  >>> manager = ModelManager(cpu_ram_gb=64, gpu_vram_gb=24)
///  >>> await manager.register("llm", my_local_model)
///  >>> await manager.load("llm")
///  >>> await manager.is_loaded("llm")
///  True
/// ```
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

    /// Register a provider under ``id`` so it can be dispatched by name with
    /// :meth:`complete` / :meth:`stream`, or fetched back with :meth:`get`.
    ///
    /// This is the unified registry — local models *and* remote providers in
    /// one place. The ``model`` argument can be:
    ///
    /// * A remote provider (``Model.openai(...)``, ``FalProvider(...)``, …):
    ///   registered as a dispatch-only entry. It owns no local weights, so it
    ///   never counts against a memory budget.
    /// * A local backend (``Model.mistralrs(...)``, ``Model.llamacpp(...)``,
    ///   …): registered for both by-name dispatch *and* load/unload lifecycle
    ///   with per-pool LRU eviction; ``memory_estimate_bytes`` reports its
    ///   footprint.
    /// * Any Python object exposing callable ``load`` and ``unload``
    ///   attributes (sync or async). It does *not* need to subclass
    ///   :class:`blazen.LocalModel`. ``is_loaded``, ``memory_bytes``, and
    ///   ``device`` are optional; such duck-typed objects are registered for
    ///   lifecycle/budget bookkeeping only (not dispatchable via
    ///   :meth:`complete`).
    ///
    /// Args:
    ///     id: A unique identifier for this model.
    ///     model: A Model / provider, or a duck-typed object with ``load``
    ///         and ``unload`` methods.
    ///     memory_estimate_bytes: Estimated memory footprint in bytes (host
    ///         RAM if the model targets CPU, GPU VRAM otherwise). Pass ``0``
    ///         (the default) for remote providers.
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

        // If the user passed a Blazen Model, pull its chat (`Model`) and/or
        // local (`LocalModel`) trait objects straight out — they carry their
        // own device()/dispatch impls. Cloning the Arcs lets the PyRef drop
        // before we may re-borrow `model` for the duck-typed fallback.
        let model_arcs: Option<ModelArcs> = model
            .extract::<PyRef<'_, crate::providers::PyModel>>()
            .ok()
            .map(|cm| (cm.inner.clone(), cm.local_model.clone()));

        if let Some((chat, local)) = model_arcs {
            if let Some(chat) = chat {
                // Dispatchable provider (remote, or a local backend that is
                // also a chat Model): register for by-name completion plus
                // lifecycle/budget when local weights are present.
                let inner = self.inner.clone();
                return pyo3_async_runtimes::tokio::future_into_py(py, async move {
                    inner
                        .register_provider(&id, chat, local, memory_estimate)
                        .await;
                    Ok(())
                });
            }
            if let Some(local) = local {
                // Lifecycle-only local backend (no chat surface).
                let inner = self.inner.clone();
                return pyo3_async_runtimes::tokio::future_into_py(py, async move {
                    inner.register(&id, local, memory_estimate).await;
                    Ok(())
                });
            }
            // PyModel subclass with neither a Rust chat impl nor a local
            // impl: fall through to duck-typing on the same object.
        }

        // Arbitrary Python object: duck-type via load/unload, probe device()
        // for an explicit target. Registered for lifecycle/budget only.
        let local_model: Arc<dyn blazen_llm::LocalModel> = Arc::new(build_local_model_wrapper(
            py,
            model,
            memory_estimate,
            blazen_llm::Device::Cpu,
        )?);
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.register(&id, local_model, memory_estimate).await;
            Ok(())
        })
    }

    /// Run a chat completion against the provider registered under ``id``.
    ///
    /// Local entries are auto-loaded on first use (subject to the pool
    /// budget); remote entries dispatch straight through.
    ///
    /// Args:
    ///     id: The identifier the provider was registered under.
    ///     messages: A list of :class:`ChatMessage` objects.
    ///     options: Optional :class:`ModelOptions` for sampling parameters,
    ///         tools, and response format.
    ///
    /// Raises:
    ///     ValueError: if ``id`` is not registered.
    ///     NotImplementedError-equivalent: if ``id`` was registered for
    ///         lifecycle only (no chat ``Model``).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ModelResponse]", imports = ("typing",)))]
    #[pyo3(signature = (id, messages, *, options=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        id: String,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyModelOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = inner
                .complete(&id, request)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(PyModelResponse { inner: response })
        })
    }

    /// Stream a chat completion against the provider registered under ``id``,
    /// invoking ``on_chunk`` once per :class:`StreamChunk`.
    ///
    /// For async-iterator streaming (``async for``) or to pass the instance
    /// around, fetch the provider with :meth:`get` and stream on it directly.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    #[pyo3(signature = (id, messages, on_chunk, *, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        id: String,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Py<PyAny>,
        options: Option<PyRef<'py, PyModelOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = inner
                .stream(&id, request)
                .await
                .map_err(BlazenPyError::from)?;
            let mut stream = std::pin::pin!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        tokio::task::block_in_place(|| {
                            Python::attach(|py| {
                                let py_chunk = pythonize::pythonize(py, &chunk).map_err(|e| {
                                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                                })?;
                                on_chunk.call1(py, (py_chunk,))?;
                                Ok::<_, PyErr>(())
                            })
                        })?;
                    }
                    Err(e) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()));
                    }
                }
            }
            Ok(())
        })
    }

    /// Fetch the provider registered under ``id`` to use or compose directly.
    ///
    /// Returns ``None`` if ``id`` is unknown or was registered for lifecycle
    /// only (no chat ``Model``).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.Optional[Model]]", imports = ("typing",)))]
    fn get<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(inner.get(&id).await.map(PyModel::from_inner))
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

// ---------------------------------------------------------------------------
// Training surface (feature = "training")
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
pub use training::{
    PyDistributedConfig, PyDpoConfig, PyFullFineTuneConfig, PyFullFineTuneResult, PyJsonlDataset,
    PyKtoConfig, PyLoraConfig, PyMixedPrecision, PyOptimConfig, PyOrpoConfig,
    PyPreferenceJsonlDataset, PyRatedJsonlDataset, PySchedulerConfig, PySchedulerKind,
    PySimpoConfig, PyTrainConfig, PyTrainCoreConfig, PyTrainedAdapter, PyTrainingEvent,
};

#[cfg(feature = "training")]
mod training {
    use std::path::PathBuf;
    use std::sync::Arc;

    use async_trait::async_trait;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

    use blazen_train::dataset::{JsonlDataset, PreferenceJsonlDataset, RatedJsonlDataset};
    use blazen_train::{
        BlazenTrainError, DpoConfig, FullFineTuneConfig, FullFineTuneResult, KtoConfig, LoraConfig,
        MixedPrecision, OptimConfig, OrpoConfig, PreferenceDataset, RatedDataset, SchedulerConfig,
        SchedulerKind, SimpoConfig, TrainConfig, TrainCoreConfig, TrainedAdapter, TrainingBatch,
        TrainingDataset, TrainingEvent, TrainingProgress,
    };
    use tokenizers::Tokenizer;

    use crate::error::BlazenPyError;

    use super::PyModelManager;

    // -----------------------------------------------------------------------
    // PySchedulerKind / PyMixedPrecision
    // -----------------------------------------------------------------------

    /// Learning-rate schedule shape passed to :class:`SchedulerConfig`.
    #[gen_stub_pyclass_enum]
    #[pyclass(name = "SchedulerKind", eq, eq_int, frozen, from_py_object)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum PySchedulerKind {
        CONSTANT,
        LINEAR,
        COSINE,
    }

    impl From<PySchedulerKind> for SchedulerKind {
        fn from(k: PySchedulerKind) -> Self {
            match k {
                PySchedulerKind::CONSTANT => Self::Constant,
                PySchedulerKind::LINEAR => Self::Linear,
                PySchedulerKind::COSINE => Self::Cosine,
            }
        }
    }

    /// Mixed-precision mode passed to :class:`TrainConfig`.
    #[gen_stub_pyclass_enum]
    #[pyclass(name = "MixedPrecision", eq, eq_int, frozen, from_py_object)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum PyMixedPrecision {
        NONE,
        BF16,
    }

    impl From<PyMixedPrecision> for MixedPrecision {
        fn from(m: PyMixedPrecision) -> Self {
            match m {
                PyMixedPrecision::NONE => Self::None,
                PyMixedPrecision::BF16 => Self::Bf16,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyLoraConfig
    // -----------------------------------------------------------------------

    /// LoRA hyperparameters.
    #[gen_stub_pyclass]
    #[pyclass(name = "LoraConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyLoraConfig {
        #[pyo3(get, set)]
        pub rank: usize,
        #[pyo3(get, set)]
        pub alpha: f32,
        #[pyo3(get, set)]
        pub dropout: f32,
        #[pyo3(get, set)]
        pub target_modules: Vec<String>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyLoraConfig {
        /// Build a LoRA hyperparameter bag.
        #[new]
        #[pyo3(signature = (*, rank = 16, alpha = 32.0, dropout = 0.0, target_modules = None))]
        fn new(
            rank: usize,
            alpha: f32,
            dropout: f32,
            target_modules: Option<Vec<String>>,
        ) -> PyResult<Self> {
            if rank == 0 {
                return Err(PyValueError::new_err("LoraConfig.rank must be > 0"));
            }
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(PyValueError::new_err("LoraConfig.alpha must be > 0"));
            }
            if !(0.0..1.0).contains(&dropout) {
                return Err(PyValueError::new_err(
                    "LoraConfig.dropout must be in [0.0, 1.0)",
                ));
            }
            let target_modules = target_modules.unwrap_or_else(|| {
                vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                ]
            });
            if target_modules.is_empty() {
                return Err(PyValueError::new_err(
                    "LoraConfig.target_modules must be non-empty",
                ));
            }
            Ok(Self {
                rank,
                alpha,
                dropout,
                target_modules,
            })
        }
    }

    impl From<PyLoraConfig> for LoraConfig {
        fn from(c: PyLoraConfig) -> Self {
            Self {
                rank: c.rank,
                alpha: c.alpha,
                dropout: c.dropout,
                target_modules: c.target_modules,
            }
        }
    }

    impl Default for PyLoraConfig {
        fn default() -> Self {
            Self {
                rank: 16,
                alpha: 32.0,
                dropout: 0.0,
                target_modules: vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                ],
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyOptimConfig
    // -----------------------------------------------------------------------

    /// AdamW optimizer hyperparameters.
    #[gen_stub_pyclass]
    #[pyclass(name = "OptimConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyOptimConfig {
        #[pyo3(get, set)]
        pub learning_rate: f64,
        #[pyo3(get, set)]
        pub beta1: f64,
        #[pyo3(get, set)]
        pub beta2: f64,
        #[pyo3(get, set)]
        pub epsilon: f64,
        #[pyo3(get, set)]
        pub weight_decay: f64,
        #[pyo3(get, set)]
        pub gradient_clip: Option<f32>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyOptimConfig {
        /// Build an AdamW hyperparameter bag.
        #[new]
        #[pyo3(signature = (
            *,
            learning_rate = 2e-4,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8,
            weight_decay = 0.0,
            gradient_clip = Some(1.0),
        ))]
        fn new(
            learning_rate: f64,
            beta1: f64,
            beta2: f64,
            epsilon: f64,
            weight_decay: f64,
            gradient_clip: Option<f32>,
        ) -> PyResult<Self> {
            if !learning_rate.is_finite() || learning_rate <= 0.0 {
                return Err(PyValueError::new_err(
                    "OptimConfig.learning_rate must be > 0",
                ));
            }
            if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
                return Err(PyValueError::new_err(
                    "OptimConfig.beta1 / beta2 must be in [0.0, 1.0)",
                ));
            }
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(PyValueError::new_err("OptimConfig.epsilon must be > 0"));
            }
            if !weight_decay.is_finite() || weight_decay < 0.0 {
                return Err(PyValueError::new_err(
                    "OptimConfig.weight_decay must be >= 0",
                ));
            }
            if let Some(g) = gradient_clip
                && (!g.is_finite() || g <= 0.0)
            {
                return Err(PyValueError::new_err(
                    "OptimConfig.gradient_clip, when set, must be > 0",
                ));
            }
            Ok(Self {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                gradient_clip,
            })
        }
    }

    impl From<PyOptimConfig> for OptimConfig {
        fn from(c: PyOptimConfig) -> Self {
            Self {
                learning_rate: c.learning_rate,
                beta1: c.beta1,
                beta2: c.beta2,
                epsilon: c.epsilon,
                weight_decay: c.weight_decay,
                gradient_clip: c.gradient_clip,
            }
        }
    }

    impl Default for PyOptimConfig {
        fn default() -> Self {
            Self {
                learning_rate: 2e-4,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                gradient_clip: Some(1.0),
            }
        }
    }

    // -----------------------------------------------------------------------
    // PySchedulerConfig
    // -----------------------------------------------------------------------

    /// Learning-rate scheduler configuration.
    #[gen_stub_pyclass]
    #[pyclass(name = "SchedulerConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PySchedulerConfig {
        #[pyo3(get, set)]
        pub kind: PySchedulerKind,
        #[pyo3(get, set)]
        pub warmup_steps: usize,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PySchedulerConfig {
        /// Build a scheduler configuration.
        #[new]
        #[pyo3(signature = (*, kind = PySchedulerKind::COSINE, warmup_steps = 0))]
        fn new(kind: PySchedulerKind, warmup_steps: usize) -> Self {
            Self { kind, warmup_steps }
        }
    }

    impl From<PySchedulerConfig> for SchedulerConfig {
        fn from(c: PySchedulerConfig) -> Self {
            Self {
                kind: c.kind.into(),
                warmup_steps: c.warmup_steps,
            }
        }
    }

    impl Default for PySchedulerConfig {
        fn default() -> Self {
            Self {
                kind: PySchedulerKind::COSINE,
                warmup_steps: 0,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyTrainConfig
    // -----------------------------------------------------------------------

    /// Full configuration for one training run.
    #[gen_stub_pyclass]
    #[pyclass(name = "TrainConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyTrainConfig {
        #[pyo3(get, set)]
        pub base_model_repo: String,
        #[pyo3(get, set)]
        pub output_dir: String,
        #[pyo3(get, set)]
        pub lora: PyLoraConfig,
        #[pyo3(get, set)]
        pub optim: PyOptimConfig,
        #[pyo3(get, set)]
        pub scheduler: PySchedulerConfig,
        #[pyo3(get, set)]
        pub max_steps: usize,
        #[pyo3(get, set)]
        pub batch_size: usize,
        #[pyo3(get, set)]
        pub gradient_accumulation_steps: usize,
        #[pyo3(get, set)]
        pub max_seq_len: usize,
        #[pyo3(get, set)]
        pub eval_steps: Option<usize>,
        #[pyo3(get, set)]
        pub save_steps: Option<usize>,
        #[pyo3(get, set)]
        pub seed: u64,
        #[pyo3(get, set)]
        pub mixed_precision: PyMixedPrecision,
        #[pyo3(get, set)]
        pub device: Option<String>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyTrainConfig {
        /// Build a TrainConfig.
        ///
        /// Raises ``ValueError`` if ``max_steps == 0``, ``batch_size == 0``,
        /// ``gradient_accumulation_steps == 0``, or ``max_seq_len == 0``.
        #[new]
        #[pyo3(signature = (
            *,
            base_model_repo,
            output_dir,
            lora = None,
            optim = None,
            scheduler = None,
            max_steps = 1000,
            batch_size = 4,
            gradient_accumulation_steps = 1,
            max_seq_len = 2048,
            eval_steps = None,
            save_steps = None,
            seed = 42,
            mixed_precision = PyMixedPrecision::BF16,
            device = None,
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            base_model_repo: String,
            output_dir: String,
            lora: Option<PyLoraConfig>,
            optim: Option<PyOptimConfig>,
            scheduler: Option<PySchedulerConfig>,
            max_steps: usize,
            batch_size: usize,
            gradient_accumulation_steps: usize,
            max_seq_len: usize,
            eval_steps: Option<usize>,
            save_steps: Option<usize>,
            seed: u64,
            mixed_precision: PyMixedPrecision,
            device: Option<String>,
        ) -> PyResult<Self> {
            if base_model_repo.trim().is_empty() {
                return Err(PyValueError::new_err(
                    "TrainConfig.base_model_repo must be non-empty",
                ));
            }
            if output_dir.trim().is_empty() {
                return Err(PyValueError::new_err(
                    "TrainConfig.output_dir must be non-empty",
                ));
            }
            if max_steps == 0 {
                return Err(PyValueError::new_err("TrainConfig.max_steps must be > 0"));
            }
            if batch_size == 0 {
                return Err(PyValueError::new_err("TrainConfig.batch_size must be > 0"));
            }
            if gradient_accumulation_steps == 0 {
                return Err(PyValueError::new_err(
                    "TrainConfig.gradient_accumulation_steps must be > 0",
                ));
            }
            if max_seq_len == 0 {
                return Err(PyValueError::new_err("TrainConfig.max_seq_len must be > 0"));
            }
            Ok(Self {
                base_model_repo,
                output_dir,
                lora: lora.unwrap_or_default(),
                optim: optim.unwrap_or_default(),
                scheduler: scheduler.unwrap_or_default(),
                max_steps,
                batch_size,
                gradient_accumulation_steps,
                max_seq_len,
                eval_steps,
                save_steps,
                seed,
                mixed_precision,
                device,
            })
        }
    }

    impl From<PyTrainConfig> for TrainConfig {
        fn from(c: PyTrainConfig) -> Self {
            Self {
                base_model_repo: c.base_model_repo,
                output_dir: PathBuf::from(c.output_dir),
                lora: c.lora.into(),
                optim: c.optim.into(),
                scheduler: c.scheduler.into(),
                max_steps: c.max_steps,
                batch_size: c.batch_size,
                gradient_accumulation_steps: c.gradient_accumulation_steps,
                max_seq_len: c.max_seq_len,
                eval_steps: c.eval_steps,
                save_steps: c.save_steps,
                seed: c.seed,
                mixed_precision: c.mixed_precision.into(),
                device: c.device,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyTrainedAdapter
    // -----------------------------------------------------------------------

    /// Result of a completed training run.
    #[gen_stub_pyclass]
    #[pyclass(name = "TrainedAdapter", frozen)]
    pub struct PyTrainedAdapter {
        #[pyo3(get)]
        pub adapter_dir: String,
        #[pyo3(get)]
        pub final_loss: f32,
        #[pyo3(get)]
        pub total_steps: usize,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyTrainedAdapter {
        fn __repr__(&self) -> String {
            format!(
                "TrainedAdapter(adapter_dir={:?}, final_loss={}, total_steps={})",
                self.adapter_dir, self.final_loss, self.total_steps
            )
        }
    }

    impl From<TrainedAdapter> for PyTrainedAdapter {
        fn from(a: TrainedAdapter) -> Self {
            Self {
                adapter_dir: a.adapter_dir.display().to_string(),
                final_loss: a.final_loss,
                total_steps: a.total_steps,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyTrainingEvent (flat discriminated record)
    // -----------------------------------------------------------------------

    /// One observable event emitted during a training run.
    ///
    /// Switch on :attr:`kind` (one of ``"started"`` / ``"step_completed"`` /
    /// ``"evaluating"`` / ``"eval_completed"`` / ``"checkpoint_saved"`` /
    /// ``"finished"``); the remaining attributes carry the per-variant payload
    /// and are ``None`` for variants that do not populate them.
    #[gen_stub_pyclass]
    #[pyclass(name = "TrainingEvent", frozen)]
    pub struct PyTrainingEvent {
        #[pyo3(get)]
        pub kind: String,
        #[pyo3(get)]
        pub step: Option<usize>,
        #[pyo3(get)]
        pub loss: Option<f32>,
        #[pyo3(get)]
        pub learning_rate: Option<f64>,
        #[pyo3(get)]
        pub elapsed_ms: Option<f64>,
        #[pyo3(get)]
        pub total_steps: Option<usize>,
        #[pyo3(get)]
        pub eval_loss: Option<f32>,
        #[pyo3(get)]
        pub checkpoint_path: Option<String>,
        #[pyo3(get)]
        pub adapter_dir: Option<String>,
        #[pyo3(get)]
        pub final_loss: Option<f32>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyTrainingEvent {
        fn __repr__(&self) -> String {
            format!(
                "TrainingEvent(kind={:?}, step={:?}, loss={:?}, learning_rate={:?}, \
                 elapsed_ms={:?}, total_steps={:?}, eval_loss={:?}, checkpoint_path={:?}, \
                 adapter_dir={:?}, final_loss={:?})",
                self.kind,
                self.step,
                self.loss,
                self.learning_rate,
                self.elapsed_ms,
                self.total_steps,
                self.eval_loss,
                self.checkpoint_path,
                self.adapter_dir,
                self.final_loss,
            )
        }
    }

    impl PyTrainingEvent {
        fn empty(kind: &'static str) -> Self {
            Self {
                kind: kind.to_string(),
                step: None,
                loss: None,
                learning_rate: None,
                elapsed_ms: None,
                total_steps: None,
                eval_loss: None,
                checkpoint_path: None,
                adapter_dir: None,
                final_loss: None,
            }
        }

        fn from_event(ev: TrainingEvent) -> Self {
            match ev {
                TrainingEvent::Started { total_steps } => Self {
                    total_steps: Some(total_steps),
                    ..Self::empty("started")
                },
                TrainingEvent::StepCompleted {
                    step,
                    loss,
                    learning_rate,
                    elapsed,
                } => Self {
                    step: Some(step),
                    loss: Some(loss),
                    learning_rate: Some(learning_rate),
                    elapsed_ms: Some(elapsed.as_secs_f64() * 1000.0),
                    ..Self::empty("step_completed")
                },
                TrainingEvent::Evaluating { step } => Self {
                    step: Some(step),
                    ..Self::empty("evaluating")
                },
                TrainingEvent::EvalCompleted { step, eval_loss } => Self {
                    step: Some(step),
                    eval_loss: Some(eval_loss),
                    ..Self::empty("eval_completed")
                },
                TrainingEvent::CheckpointSaved { step, path } => Self {
                    step: Some(step),
                    checkpoint_path: Some(path.display().to_string()),
                    ..Self::empty("checkpoint_saved")
                },
                TrainingEvent::Finished {
                    final_loss,
                    total_steps,
                    adapter_dir,
                } => Self {
                    total_steps: Some(total_steps),
                    final_loss: Some(final_loss),
                    adapter_dir: Some(adapter_dir.display().to_string()),
                    ..Self::empty("finished")
                },
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyJsonlDataset
    // -----------------------------------------------------------------------

    /// JSONL-backed training dataset.
    ///
    /// Each line of the input file must deserialize to either
    /// ``{"messages": [{"role": ..., "content": ...}, ...]}`` (OpenAI shape)
    /// or ``{"prompt": "...", "completion": "..."}`` (legacy SFT).
    #[gen_stub_pyclass]
    #[pyclass(name = "JsonlDataset", frozen)]
    pub struct PyJsonlDataset {
        inner: Arc<JsonlDataset>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyJsonlDataset {
        /// Load a JSONL training file using the tokenizer at ``tokenizer_path``.
        ///
        /// Args:
        ///     path: Filesystem path to the JSONL file.
        ///     tokenizer_path: Path to a HuggingFace ``tokenizer.json`` file.
        ///     chat_template: Optional Jinja2 chat template from the model's
        ///         ``tokenizer_config.json``. Required if any row uses the
        ///         ``messages`` shape.
        ///     max_seq_len: Maximum tokenized sequence length per example.
        ///     device: Candle device string (``"cpu"``, ``"cuda:0"``, ``"metal"``).
        ///     pad_token_id: Token id to write into padded positions.
        #[staticmethod]
        #[pyo3(signature = (
            path,
            tokenizer_path,
            chat_template = None,
            max_seq_len = 2048,
            device = "cpu".to_string(),
            pad_token_id = 0,
        ))]
        fn from_path(
            path: String,
            tokenizer_path: String,
            chat_template: Option<String>,
            max_seq_len: usize,
            device: String,
            pad_token_id: u32,
        ) -> PyResult<Self> {
            if max_seq_len == 0 {
                return Err(PyValueError::new_err(
                    "JsonlDataset.max_seq_len must be > 0",
                ));
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                PyValueError::new_err(format!(
                    "failed to load tokenizer from {tokenizer_path:?}: {e}"
                ))
            })?;
            let cdev = super::parse_train_device_py(&device)?;
            let ds = JsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                chat_template.as_deref(),
                max_seq_len,
                cdev,
                pad_token_id,
            )
            .map_err(|e| PyValueError::new_err(format!("JsonlDataset load failed: {e}")))?;
            Ok(Self {
                inner: Arc::new(ds),
            })
        }

        /// Number of examples in the dataset.
        fn __len__(&self) -> usize {
            self.inner.len()
        }
    }

    // -----------------------------------------------------------------------
    // Bridges: dataset (Arc-wrapped) + progress callback
    // -----------------------------------------------------------------------

    /// Adapter so `Arc<JsonlDataset>` satisfies `Box<dyn TrainingDataset>`.
    struct ArcDataset(Arc<JsonlDataset>);

    #[async_trait]
    impl TrainingDataset for ArcDataset {
        fn len(&self) -> usize {
            self.0.len()
        }

        async fn batch(
            &self,
            batch_size: usize,
            idx: usize,
        ) -> Result<TrainingBatch, BlazenTrainError> {
            self.0.batch(batch_size, idx).await
        }
    }

    /// Bridge between the Rust [`TrainingProgress`] trait and a Python
    /// callable.
    ///
    /// Each event is constructed as a [`PyTrainingEvent`] pyclass and passed
    /// to the user callable under the GIL. Any exception raised by the
    /// callable is logged and converted into
    /// [`BlazenTrainError::Cancelled`], which the trainer surfaces to the
    /// caller as a `BlazenError::cancelled()`.
    struct PyTrainingProgressBridge {
        callback: Arc<Py<PyAny>>,
    }

    impl TrainingProgress for PyTrainingProgressBridge {
        fn on_event(&self, event: TrainingEvent) -> Result<(), BlazenTrainError> {
            let py_event = PyTrainingEvent::from_event(event);
            Python::attach(|py| {
                let cb = self.callback.bind(py);
                match cb.call1((py_event,)) {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "training progress callback raised; aborting run",
                        );
                        Err(BlazenTrainError::Cancelled)
                    }
                }
            })
        }
    }

    // -----------------------------------------------------------------------
    // ModelManager.train_lora
    // -----------------------------------------------------------------------

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyModelManager {
        /// Train a LoRA adapter end-to-end on the configured base model.
        ///
        /// Downloads the base model from HuggingFace (cached), builds a
        /// `VarMap`, runs the AdamW + LoRA training loop driven by
        /// ``dataset``, and writes the resulting PEFT-format adapter to
        /// ``config.output_dir``. The returned :class:`TrainedAdapter`'s
        /// ``adapter_dir`` is immediately mountable via
        /// :meth:`ModelManager.load_adapter` on a compatible backend.
        ///
        /// Args:
        ///     config: Full :class:`TrainConfig` for the run.
        ///     dataset: A :class:`JsonlDataset` providing training batches.
        ///     progress: Optional callable invoked with one
        ///         :class:`TrainingEvent` per Started/StepCompleted/
        ///         CheckpointSaved/Finished transition. Raising from the
        ///         callable cancels the run.
        ///
        /// Returns:
        ///     A :class:`TrainedAdapter` describing the on-disk adapter.
        #[gen_stub(override_return_type(
            type_repr = "typing.Coroutine[typing.Any, typing.Any, TrainedAdapter]",
            imports = ("typing",)
        ))]
        #[pyo3(signature = (config, dataset, progress = None))]
        fn train_lora<'py>(
            &self,
            py: Python<'py>,
            config: PyTrainConfig,
            dataset: Py<PyJsonlDataset>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            let rust_cfg: TrainConfig = config.into();
            let ds_arc = {
                let borrowed = dataset.bind(py).borrow();
                borrowed.inner.clone()
            };
            let sink: Option<Arc<dyn TrainingProgress>> = progress.map(|cb| {
                let bridge = PyTrainingProgressBridge {
                    callback: Arc::new(cb),
                };
                Arc::new(bridge) as Arc<dyn TrainingProgress>
            });

            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let dataset_box: Box<dyn TrainingDataset> = Box::new(ArcDataset(ds_arc));
                let adapter = inner
                    .train_lora(rust_cfg, dataset_box, sink)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyTrainedAdapter::from(adapter))
            })
        }
    }

    // -----------------------------------------------------------------------
    // PyDistributedConfig — ring-AllReduce config for multi-GPU / multi-node
    // training. Lifted to/from blazen_train::config::DistributedConfig.
    // -----------------------------------------------------------------------

    /// Configuration for distributed (ring-AllReduce) training. Pass to
    /// :meth:`ModelManager.train_grpo` / :meth:`train_ppo` to enable
    /// gradient averaging across ``world_size`` workers connected via
    /// gRPC.
    ///
    /// ``rank`` is the 0-indexed rank of this worker; ``world_size`` is
    /// the total number of workers. ``peers`` is the ordered list of
    /// ``"host:port"`` gRPC endpoints — one entry per rank. ``master_addr``
    /// and ``master_port`` identify the bootstrap node (typically the
    /// host part of ``peers[0]``).
    #[gen_stub_pyclass]
    #[pyclass(name = "DistributedConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyDistributedConfig {
        #[pyo3(get, set)]
        pub rank: usize,
        #[pyo3(get, set)]
        pub world_size: usize,
        #[pyo3(get, set)]
        pub peers: Vec<String>,
        #[pyo3(get, set)]
        pub master_addr: String,
        #[pyo3(get, set)]
        pub master_port: u16,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyDistributedConfig {
        #[new]
        #[pyo3(signature = (*, rank, world_size, peers, master_addr, master_port))]
        #[must_use]
        pub fn new(
            rank: usize,
            world_size: usize,
            peers: Vec<String>,
            master_addr: String,
            master_port: u16,
        ) -> Self {
            Self {
                rank,
                world_size,
                peers,
                master_addr,
                master_port,
            }
        }
    }

    #[cfg(feature = "distributed")]
    impl From<PyDistributedConfig> for blazen_train::config::DistributedConfig {
        fn from(v: PyDistributedConfig) -> Self {
            Self {
                rank: v.rank,
                world_size: v.world_size,
                peers: v.peers,
                master_addr: v.master_addr,
                master_port: v.master_port,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyTrainCoreConfig — shared hyperparameters for the preference / full
    // fine-tune verbs introduced in PR8. Mirrors `TrainCoreConfig` from
    // `blazen_train::config`.
    // -----------------------------------------------------------------------

    /// Shared training hyperparameters used by DPO / ORPO / SimPO / KTO /
    /// full fine-tune. Mirrors the legacy :class:`TrainConfig` minus the
    /// PEFT-specific :class:`LoraConfig`.
    #[gen_stub_pyclass]
    #[pyclass(name = "TrainCoreConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyTrainCoreConfig {
        #[pyo3(get, set)]
        pub base_model_repo: String,
        #[pyo3(get, set)]
        pub base_model_revision: Option<String>,
        #[pyo3(get, set)]
        pub output_dir: String,
        #[pyo3(get, set)]
        pub max_steps: usize,
        #[pyo3(get, set)]
        pub batch_size: usize,
        #[pyo3(get, set)]
        pub gradient_accumulation_steps: usize,
        #[pyo3(get, set)]
        pub max_seq_len: usize,
        #[pyo3(get, set)]
        pub eval_steps: Option<usize>,
        #[pyo3(get, set)]
        pub save_steps: Option<usize>,
        #[pyo3(get, set)]
        pub seed: u64,
        #[pyo3(get, set)]
        pub mixed_precision: PyMixedPrecision,
        #[pyo3(get, set)]
        pub device: Option<String>,
        #[pyo3(get, set)]
        pub optim: PyOptimConfig,
        #[pyo3(get, set)]
        pub scheduler: PySchedulerConfig,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyTrainCoreConfig {
        /// Build a TrainCoreConfig.
        ///
        /// Raises ``ValueError`` if ``max_steps == 0``, ``batch_size == 0``,
        /// ``gradient_accumulation_steps == 0``, or ``max_seq_len == 0``.
        #[new]
        #[pyo3(signature = (
            *,
            base_model_repo,
            output_dir,
            base_model_revision = None,
            max_steps = 1000,
            batch_size = 1,
            gradient_accumulation_steps = 8,
            max_seq_len = 1024,
            eval_steps = None,
            save_steps = None,
            seed = 42,
            mixed_precision = PyMixedPrecision::BF16,
            device = None,
            optim = None,
            scheduler = None,
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            base_model_repo: String,
            output_dir: String,
            base_model_revision: Option<String>,
            max_steps: usize,
            batch_size: usize,
            gradient_accumulation_steps: usize,
            max_seq_len: usize,
            eval_steps: Option<usize>,
            save_steps: Option<usize>,
            seed: u64,
            mixed_precision: PyMixedPrecision,
            device: Option<String>,
            optim: Option<PyOptimConfig>,
            scheduler: Option<PySchedulerConfig>,
        ) -> PyResult<Self> {
            if base_model_repo.trim().is_empty() {
                return Err(PyValueError::new_err(
                    "TrainCoreConfig.base_model_repo must be non-empty",
                ));
            }
            if output_dir.trim().is_empty() {
                return Err(PyValueError::new_err(
                    "TrainCoreConfig.output_dir must be non-empty",
                ));
            }
            if max_steps == 0 {
                return Err(PyValueError::new_err(
                    "TrainCoreConfig.max_steps must be > 0",
                ));
            }
            if batch_size == 0 {
                return Err(PyValueError::new_err(
                    "TrainCoreConfig.batch_size must be > 0",
                ));
            }
            if gradient_accumulation_steps == 0 {
                return Err(PyValueError::new_err(
                    "TrainCoreConfig.gradient_accumulation_steps must be > 0",
                ));
            }
            if max_seq_len == 0 {
                return Err(PyValueError::new_err(
                    "TrainCoreConfig.max_seq_len must be > 0",
                ));
            }
            Ok(Self {
                base_model_repo,
                base_model_revision,
                output_dir,
                max_steps,
                batch_size,
                gradient_accumulation_steps,
                max_seq_len,
                eval_steps,
                save_steps,
                seed,
                mixed_precision,
                device,
                optim: optim.unwrap_or_default(),
                scheduler: scheduler.unwrap_or_default(),
            })
        }
    }

    impl From<PyTrainCoreConfig> for TrainCoreConfig {
        fn from(c: PyTrainCoreConfig) -> Self {
            Self {
                base_model_repo: c.base_model_repo,
                base_model_revision: c.base_model_revision,
                output_dir: PathBuf::from(c.output_dir),
                max_steps: c.max_steps,
                batch_size: c.batch_size,
                gradient_accumulation_steps: c.gradient_accumulation_steps,
                max_seq_len: c.max_seq_len,
                eval_steps: c.eval_steps,
                save_steps: c.save_steps,
                seed: c.seed,
                mixed_precision: c.mixed_precision.into(),
                device: c.device,
                optim: c.optim.into(),
                scheduler: c.scheduler.into(),
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyDpoConfig
    // -----------------------------------------------------------------------

    /// Direct Preference Optimization (DPO) configuration.
    #[gen_stub_pyclass]
    #[pyclass(name = "DpoConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyDpoConfig {
        #[pyo3(get, set)]
        pub core: PyTrainCoreConfig,
        #[pyo3(get, set)]
        pub lora: PyLoraConfig,
        #[pyo3(get, set)]
        pub beta: f32,
        #[pyo3(get, set)]
        pub label_smoothing: f32,
        #[pyo3(get, set)]
        pub reference_model_repo: Option<String>,
        #[pyo3(get, set)]
        pub reference_model_revision: Option<String>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyDpoConfig {
        /// Build a DpoConfig.
        #[new]
        #[pyo3(signature = (
            *,
            core,
            lora = None,
            beta = 0.1,
            label_smoothing = 0.0,
            reference_model_repo = None,
            reference_model_revision = None,
        ))]
        fn new(
            core: PyTrainCoreConfig,
            lora: Option<PyLoraConfig>,
            beta: f32,
            label_smoothing: f32,
            reference_model_repo: Option<String>,
            reference_model_revision: Option<String>,
        ) -> PyResult<Self> {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err("DpoConfig.beta must be > 0"));
            }
            if !label_smoothing.is_finite() || !(0.0..=0.5).contains(&label_smoothing) {
                return Err(PyValueError::new_err(
                    "DpoConfig.label_smoothing must be in [0.0, 0.5]",
                ));
            }
            Ok(Self {
                core,
                lora: lora.unwrap_or_default(),
                beta,
                label_smoothing,
                reference_model_repo,
                reference_model_revision,
            })
        }
    }

    impl From<PyDpoConfig> for DpoConfig {
        fn from(c: PyDpoConfig) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                beta: c.beta,
                reference_model_repo: c.reference_model_repo,
                reference_model_revision: c.reference_model_revision,
                label_smoothing: c.label_smoothing,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyOrpoConfig
    // -----------------------------------------------------------------------

    /// Odds Ratio Preference Optimization (ORPO) configuration.
    #[gen_stub_pyclass]
    #[pyclass(name = "OrpoConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyOrpoConfig {
        #[pyo3(get, set)]
        pub core: PyTrainCoreConfig,
        #[pyo3(get, set)]
        pub lora: PyLoraConfig,
        #[pyo3(get, set)]
        pub lambda_weight: f32,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyOrpoConfig {
        /// Build an OrpoConfig.
        #[new]
        #[pyo3(signature = (*, core, lora = None, lambda_weight = 0.1))]
        fn new(
            core: PyTrainCoreConfig,
            lora: Option<PyLoraConfig>,
            lambda_weight: f32,
        ) -> PyResult<Self> {
            if !lambda_weight.is_finite() || lambda_weight < 0.0 {
                return Err(PyValueError::new_err(
                    "OrpoConfig.lambda_weight must be >= 0",
                ));
            }
            Ok(Self {
                core,
                lora: lora.unwrap_or_default(),
                lambda_weight,
            })
        }
    }

    impl From<PyOrpoConfig> for OrpoConfig {
        fn from(c: PyOrpoConfig) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                lambda: c.lambda_weight,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PySimpoConfig
    // -----------------------------------------------------------------------

    /// Simple Preference Optimization (SimPO) configuration.
    #[gen_stub_pyclass]
    #[pyclass(name = "SimpoConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PySimpoConfig {
        #[pyo3(get, set)]
        pub core: PyTrainCoreConfig,
        #[pyo3(get, set)]
        pub lora: PyLoraConfig,
        #[pyo3(get, set)]
        pub beta: f32,
        #[pyo3(get, set)]
        pub gamma: f32,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PySimpoConfig {
        /// Build a SimpoConfig.
        #[new]
        #[pyo3(signature = (*, core, lora = None, beta = 2.0, gamma = 1.0))]
        fn new(
            core: PyTrainCoreConfig,
            lora: Option<PyLoraConfig>,
            beta: f32,
            gamma: f32,
        ) -> PyResult<Self> {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err("SimpoConfig.beta must be > 0"));
            }
            if !gamma.is_finite() || gamma < 0.0 {
                return Err(PyValueError::new_err("SimpoConfig.gamma must be >= 0"));
            }
            Ok(Self {
                core,
                lora: lora.unwrap_or_default(),
                beta,
                gamma,
            })
        }
    }

    impl From<PySimpoConfig> for SimpoConfig {
        fn from(c: PySimpoConfig) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                beta: c.beta,
                gamma: c.gamma,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyKtoConfig
    // -----------------------------------------------------------------------

    /// Kahneman-Tversky Optimization (KTO) configuration.
    #[gen_stub_pyclass]
    #[pyclass(name = "KtoConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyKtoConfig {
        #[pyo3(get, set)]
        pub core: PyTrainCoreConfig,
        #[pyo3(get, set)]
        pub lora: PyLoraConfig,
        #[pyo3(get, set)]
        pub beta: f32,
        #[pyo3(get, set)]
        pub lambda_d: f32,
        #[pyo3(get, set)]
        pub lambda_u: f32,
        #[pyo3(get, set)]
        pub reference_model_repo: Option<String>,
        #[pyo3(get, set)]
        pub reference_model_revision: Option<String>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyKtoConfig {
        /// Build a KtoConfig.
        #[new]
        #[pyo3(signature = (
            *,
            core,
            lora = None,
            beta = 0.1,
            lambda_d = 1.0,
            lambda_u = 1.0,
            reference_model_repo = None,
            reference_model_revision = None,
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            core: PyTrainCoreConfig,
            lora: Option<PyLoraConfig>,
            beta: f32,
            lambda_d: f32,
            lambda_u: f32,
            reference_model_repo: Option<String>,
            reference_model_revision: Option<String>,
        ) -> PyResult<Self> {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err("KtoConfig.beta must be > 0"));
            }
            if !lambda_d.is_finite() || lambda_d < 0.0 {
                return Err(PyValueError::new_err("KtoConfig.lambda_d must be >= 0"));
            }
            if !lambda_u.is_finite() || lambda_u < 0.0 {
                return Err(PyValueError::new_err("KtoConfig.lambda_u must be >= 0"));
            }
            Ok(Self {
                core,
                lora: lora.unwrap_or_default(),
                beta,
                lambda_d,
                lambda_u,
                reference_model_repo,
                reference_model_revision,
            })
        }
    }

    impl From<PyKtoConfig> for KtoConfig {
        fn from(c: PyKtoConfig) -> Self {
            Self {
                core: c.core.into(),
                lora: c.lora.into(),
                beta: c.beta,
                lambda_d: c.lambda_d,
                lambda_u: c.lambda_u,
                reference_model_repo: c.reference_model_repo,
                reference_model_revision: c.reference_model_revision,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyFullFineTuneConfig
    // -----------------------------------------------------------------------

    /// Full fine-tune configuration (every parameter trains).
    ///
    /// ``gradient_checkpointing = True`` is accepted for forward
    /// compatibility but the trainer currently rejects it with
    /// ``ValueError`` at init time because candle 0.10.2 has no
    /// activation-checkpointing primitive.
    #[gen_stub_pyclass]
    #[pyclass(name = "FullFineTuneConfig", from_py_object)]
    #[derive(Clone)]
    pub struct PyFullFineTuneConfig {
        #[pyo3(get, set)]
        pub core: PyTrainCoreConfig,
        #[pyo3(get, set)]
        pub gradient_checkpointing: bool,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyFullFineTuneConfig {
        /// Build a FullFineTuneConfig.
        #[new]
        #[pyo3(signature = (*, core, gradient_checkpointing = false))]
        fn new(core: PyTrainCoreConfig, gradient_checkpointing: bool) -> Self {
            Self {
                core,
                gradient_checkpointing,
            }
        }
    }

    impl From<PyFullFineTuneConfig> for FullFineTuneConfig {
        fn from(c: PyFullFineTuneConfig) -> Self {
            Self {
                core: c.core.into(),
                gradient_checkpointing: c.gradient_checkpointing,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyFullFineTuneResult
    // -----------------------------------------------------------------------

    /// Result of a completed full fine-tune run.
    #[gen_stub_pyclass]
    #[pyclass(name = "FullFineTuneResult", frozen)]
    pub struct PyFullFineTuneResult {
        #[pyo3(get)]
        pub output_dir: String,
        #[pyo3(get)]
        pub final_loss: f32,
        #[pyo3(get)]
        pub steps_completed: usize,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyFullFineTuneResult {
        fn __repr__(&self) -> String {
            format!(
                "FullFineTuneResult(output_dir={:?}, final_loss={}, steps_completed={})",
                self.output_dir, self.final_loss, self.steps_completed
            )
        }
    }

    impl From<FullFineTuneResult> for PyFullFineTuneResult {
        fn from(r: FullFineTuneResult) -> Self {
            Self {
                output_dir: r.output_dir.display().to_string(),
                final_loss: r.final_loss,
                steps_completed: r.steps_completed,
            }
        }
    }

    // -----------------------------------------------------------------------
    // PyPreferenceJsonlDataset
    // -----------------------------------------------------------------------

    /// JSONL-backed preference-pair dataset for DPO / ORPO / SimPO.
    ///
    /// Each line of the input file must deserialize to either
    /// ``{"prompt": "...", "chosen": "...", "rejected": "..."}`` or
    /// ``{"messages": [{"role": ..., "content": ...}, ...], "chosen": "...",
    /// "rejected": "..."}`` (the latter requires ``chat_template``).
    #[gen_stub_pyclass]
    #[pyclass(name = "PreferenceJsonlDataset", frozen)]
    pub struct PyPreferenceJsonlDataset {
        inner: Arc<PreferenceJsonlDataset>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyPreferenceJsonlDataset {
        /// Load a preference-pair JSONL file using the tokenizer at
        /// ``tokenizer_path``.
        ///
        /// Args mirror :meth:`JsonlDataset.from_path`.
        #[staticmethod]
        #[pyo3(signature = (
            path,
            tokenizer_path,
            chat_template = None,
            max_seq_len = 2048,
            device = "cpu".to_string(),
            pad_token_id = 0,
        ))]
        fn from_path(
            path: String,
            tokenizer_path: String,
            chat_template: Option<String>,
            max_seq_len: usize,
            device: String,
            pad_token_id: u32,
        ) -> PyResult<Self> {
            if max_seq_len == 0 {
                return Err(PyValueError::new_err(
                    "PreferenceJsonlDataset.max_seq_len must be > 0",
                ));
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                PyValueError::new_err(format!(
                    "failed to load tokenizer from {tokenizer_path:?}: {e}"
                ))
            })?;
            let cdev = super::parse_train_device_py(&device)?;
            let ds = PreferenceJsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                chat_template.as_deref(),
                max_seq_len,
                cdev,
                pad_token_id,
            )
            .map_err(|e| {
                PyValueError::new_err(format!("PreferenceJsonlDataset load failed: {e}"))
            })?;
            Ok(Self {
                inner: Arc::new(ds),
            })
        }

        /// Number of preference examples in the dataset.
        fn __len__(&self) -> usize {
            self.inner.len()
        }
    }

    // -----------------------------------------------------------------------
    // PyRatedJsonlDataset
    // -----------------------------------------------------------------------

    /// JSONL-backed rated single-completion dataset for KTO.
    ///
    /// Each line of the input file must deserialize to either
    /// ``{"prompt": "...", "completion": "...", "label": true|false}`` or
    /// ``{"messages": [...], "completion": "...", "label": ...}`` (the
    /// latter requires ``chat_template``).
    #[gen_stub_pyclass]
    #[pyclass(name = "RatedJsonlDataset", frozen)]
    pub struct PyRatedJsonlDataset {
        inner: Arc<RatedJsonlDataset>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyRatedJsonlDataset {
        /// Load a rated JSONL file using the tokenizer at ``tokenizer_path``.
        ///
        /// Args mirror :meth:`JsonlDataset.from_path`.
        #[staticmethod]
        #[pyo3(signature = (
            path,
            tokenizer_path,
            chat_template = None,
            max_seq_len = 2048,
            device = "cpu".to_string(),
            pad_token_id = 0,
        ))]
        fn from_path(
            path: String,
            tokenizer_path: String,
            chat_template: Option<String>,
            max_seq_len: usize,
            device: String,
            pad_token_id: u32,
        ) -> PyResult<Self> {
            if max_seq_len == 0 {
                return Err(PyValueError::new_err(
                    "RatedJsonlDataset.max_seq_len must be > 0",
                ));
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                PyValueError::new_err(format!(
                    "failed to load tokenizer from {tokenizer_path:?}: {e}"
                ))
            })?;
            let cdev = super::parse_train_device_py(&device)?;
            let ds = RatedJsonlDataset::from_path(
                std::path::Path::new(&path),
                Arc::new(tokenizer),
                chat_template.as_deref(),
                max_seq_len,
                cdev,
                pad_token_id,
            )
            .map_err(|e| PyValueError::new_err(format!("RatedJsonlDataset load failed: {e}")))?;
            Ok(Self {
                inner: Arc::new(ds),
            })
        }

        /// Number of rated examples in the dataset.
        fn __len__(&self) -> usize {
            self.inner.len()
        }
    }

    // -----------------------------------------------------------------------
    // ModelManager.train_dpo / train_orpo / train_simpo / train_kto / fine_tune
    // -----------------------------------------------------------------------

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyModelManager {
        /// Train a LoRA adapter via Direct Preference Optimization (DPO).
        ///
        /// Args:
        ///     config: A :class:`DpoConfig`.
        ///     dataset: A :class:`PreferenceJsonlDataset` providing
        ///         ``(prompt, chosen, rejected)`` triples.
        ///     progress: Optional callable invoked with one
        ///         :class:`TrainingEvent` per transition.
        ///
        /// Returns:
        ///     A :class:`TrainedAdapter` describing the on-disk PEFT
        ///     adapter.
        #[gen_stub(override_return_type(
            type_repr = "typing.Coroutine[typing.Any, typing.Any, TrainedAdapter]",
            imports = ("typing",)
        ))]
        #[pyo3(signature = (config, dataset, progress = None))]
        fn train_dpo<'py>(
            &self,
            py: Python<'py>,
            config: PyDpoConfig,
            dataset: Py<PyPreferenceJsonlDataset>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            let rust_cfg: DpoConfig = config.into();
            let ds_arc: Arc<dyn PreferenceDataset> = {
                let borrowed = dataset.bind(py).borrow();
                borrowed.inner.clone()
            };
            let sink = py_progress_sink(progress);

            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let adapter = inner
                    .train_dpo(rust_cfg, ds_arc, sink)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyTrainedAdapter::from(adapter))
            })
        }

        /// Train a LoRA adapter via Odds Ratio Preference Optimization
        /// (ORPO).
        ///
        /// Reference-free. Combines an SFT loss on chosen completions with
        /// an odds-ratio preference term weighted by ``config.lambda_weight``.
        #[gen_stub(override_return_type(
            type_repr = "typing.Coroutine[typing.Any, typing.Any, TrainedAdapter]",
            imports = ("typing",)
        ))]
        #[pyo3(signature = (config, dataset, progress = None))]
        fn train_orpo<'py>(
            &self,
            py: Python<'py>,
            config: PyOrpoConfig,
            dataset: Py<PyPreferenceJsonlDataset>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            let rust_cfg: OrpoConfig = config.into();
            let ds_arc: Arc<dyn PreferenceDataset> = {
                let borrowed = dataset.bind(py).borrow();
                borrowed.inner.clone()
            };
            let sink = py_progress_sink(progress);

            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let adapter = inner
                    .train_orpo(rust_cfg, ds_arc, sink)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyTrainedAdapter::from(adapter))
            })
        }

        /// Train a LoRA adapter via Simple Preference Optimization
        /// (SimPO).
        ///
        /// Reference-free and length-normalized.
        #[gen_stub(override_return_type(
            type_repr = "typing.Coroutine[typing.Any, typing.Any, TrainedAdapter]",
            imports = ("typing",)
        ))]
        #[pyo3(signature = (config, dataset, progress = None))]
        fn train_simpo<'py>(
            &self,
            py: Python<'py>,
            config: PySimpoConfig,
            dataset: Py<PyPreferenceJsonlDataset>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            let rust_cfg: SimpoConfig = config.into();
            let ds_arc: Arc<dyn PreferenceDataset> = {
                let borrowed = dataset.bind(py).borrow();
                borrowed.inner.clone()
            };
            let sink = py_progress_sink(progress);

            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let adapter = inner
                    .train_simpo(rust_cfg, ds_arc, sink)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyTrainedAdapter::from(adapter))
            })
        }

        /// Train a LoRA adapter via Kahneman-Tversky Optimization (KTO).
        ///
        /// Args:
        ///     config: A :class:`KtoConfig`.
        ///     dataset: A :class:`RatedJsonlDataset` providing
        ///         ``(prompt, completion, desirable)`` triples.
        #[gen_stub(override_return_type(
            type_repr = "typing.Coroutine[typing.Any, typing.Any, TrainedAdapter]",
            imports = ("typing",)
        ))]
        #[pyo3(signature = (config, dataset, progress = None))]
        fn train_kto<'py>(
            &self,
            py: Python<'py>,
            config: PyKtoConfig,
            dataset: Py<PyRatedJsonlDataset>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            let rust_cfg: KtoConfig = config.into();
            let ds_arc: Arc<dyn RatedDataset> = {
                let borrowed = dataset.bind(py).borrow();
                borrowed.inner.clone()
            };
            let sink = py_progress_sink(progress);

            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let adapter = inner
                    .train_kto(rust_cfg, ds_arc, sink)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyTrainedAdapter::from(adapter))
            })
        }

        /// Run a full fine-tune (every parameter trainable; no LoRA
        /// adapter).
        ///
        /// Returns :class:`FullFineTuneResult` — not :class:`TrainedAdapter`
        /// — because the output is a complete set of model weights in
        /// ``config.core.output_dir`` rather than a mountable PEFT delta.
        ///
        /// Setting ``config.gradient_checkpointing = True`` raises
        /// ``ValueError`` at init time because candle 0.10.2 has no
        /// activation-checkpointing primitive.
        #[gen_stub(override_return_type(
            type_repr = "typing.Coroutine[typing.Any, typing.Any, FullFineTuneResult]",
            imports = ("typing",)
        ))]
        #[pyo3(signature = (config, dataset, progress = None))]
        fn fine_tune<'py>(
            &self,
            py: Python<'py>,
            config: PyFullFineTuneConfig,
            dataset: Py<PyJsonlDataset>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            let rust_cfg: FullFineTuneConfig = config.into();
            let ds_arc: Arc<dyn TrainingDataset> = {
                let borrowed = dataset.bind(py).borrow();
                Arc::new(ArcDataset(borrowed.inner.clone()))
            };
            let sink = py_progress_sink(progress);

            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let result = inner
                    .fine_tune(rust_cfg, ds_arc, sink)
                    .await
                    .map_err(BlazenPyError::from)?;
                Ok(PyFullFineTuneResult::from(result))
            })
        }
    }

    /// Shared helper: wrap a Python callable in a [`TrainingProgress`] sink.
    fn py_progress_sink(progress: Option<Py<PyAny>>) -> Option<Arc<dyn TrainingProgress>> {
        progress.map(|cb| {
            let bridge = PyTrainingProgressBridge {
                callback: Arc::new(cb),
            };
            Arc::new(bridge) as Arc<dyn TrainingProgress>
        })
    }
}

#[cfg(feature = "training")]
fn parse_train_device_py(device: &str) -> PyResult<candle_core::Device> {
    let trimmed = device.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "cpu" {
        return Ok(candle_core::Device::Cpu);
    }
    if let Some(idx_str) = lower.strip_prefix("cuda:") {
        let idx: usize = idx_str.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid CUDA device {trimmed:?}: expected 'cuda:N'"
            ))
        })?;
        return candle_core::Device::new_cuda(idx).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to open CUDA device {trimmed:?}: {e}"
            ))
        });
    }
    if lower == "cuda" {
        return candle_core::Device::new_cuda(0).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to open cuda:0: {e}"))
        });
    }
    if lower == "metal" {
        return candle_core::Device::new_metal(0).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to open metal:0: {e}"))
        });
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "unrecognized device {trimmed:?}: expected 'cpu', 'cuda', 'cuda:N', or 'metal'"
    )))
}
