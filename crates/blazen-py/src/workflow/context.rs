//! Python wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! Accepts any Python value — JSON-serializable types are stored efficiently
//! as JSON, `bytes`/`bytearray` as raw binary, picklable objects via pickle,
//! and unpicklable objects (DB connections, file handles) as live references.
//!
//! ## Namespaces
//!
//! Two explicit namespaces are exposed alongside the smart-routing
//! shortcuts (`ctx.set` / `ctx.get`):
//!
//! - **`ctx.state`** — persistable values: JSON, bytes, picklable
//!   objects. Survives `pause()` / `resume()` and checkpoints.
//! - **`ctx.session`** — live in-process references (DB connections,
//!   file handles, sockets, sqlite cursors, …). Identity is preserved
//!   within a single workflow run; subject to the workflow's
//!   [`SessionPausePolicy`](blazen_core::session_ref::SessionPausePolicy)
//!   on snapshot.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::event::PyEvent;
use crate::convert::block_on_context;

/// Shared workflow context accessible by all steps.
///
/// Provides typed key/value storage, event emission, and stream publishing.
/// Accepts any Python value — JSON-serializable types, raw bytes, picklable
/// objects, or live references for unpicklable objects (DB connections, etc.).
///
/// Example:
///     >>> def my_step(ctx: Context, ev: Event) -> Event:
///     ...     ctx.set("counter", 42)
///     ...     ctx.set("model", MyPydanticModel(name="foo"))
///     ...     ctx.set("db", sqlite3.connect(":memory:"))
///     ...     val = ctx.get("counter")  # returns 42
///     ...     db = ctx.get("db")        # returns the same connection
#[gen_stub_pyclass]
#[pyclass(name = "Context", from_py_object)]
#[derive(Clone)]
pub struct PyContext {
    pub(crate) inner: blazen_core::Context,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyContext {
    /// Store any value under the given key.
    ///
    /// Storage tiers (tried in order):
    /// 1. `bytes`/`bytearray` → raw binary (survives snapshots)
    /// 2. JSON-serializable (dict, list, str, int, float, bool, None) → JSON (survives snapshots)
    /// 3. Picklable objects (Pydantic, dataclasses, etc.) → pickled bytes (survives snapshots)
    /// 4. Unpicklable objects (DB connections, file handles) → live reference (same-process only)
    ///
    /// Args:
    ///     key: The storage key.
    ///     value: Any Python value.
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // BlazenState → per-field storage
        if value.is_instance_of::<super::state::PyBlazenState>() {
            return self.set_blazen_state(py, key, value);
        }

        let inner = self.inner.clone();
        let key = key.to_string();

        if value.is_instance_of::<pyo3::types::PyBytes>()
            || value.is_instance_of::<pyo3::types::PyByteArray>()
        {
            // Tier 1: bytes/bytearray → Bytes variant
            let bytes: Vec<u8> = value.extract()?;
            block_on_context(async { inner.set_bytes(&key, bytes).await });
        } else if let Ok(json_val) = crate::convert::try_py_to_json(py, value) {
            // Tier 2: JSON-serializable → Json variant
            block_on_context(async { inner.set(&key, json_val).await });
        } else {
            // Tier 3: try pickle → Native variant
            let pickle = py.import("pickle")?;
            if let Ok(pickled_obj) = pickle.call_method1("dumps", (value,)) {
                let pickled: Vec<u8> = pickled_obj.extract()?;
                let sv = blazen_core::StateValue::Native(blazen_core::BytesWrapper(pickled));
                block_on_context(async { inner.set_value(&key, sv).await });
            } else {
                // Tier 4: unpicklable → store as live object in core.
                // Wrap in Arc because Py<PyAny> isn't Clone (needs GIL),
                // but Arc<Py<PyAny>> is Clone + Send + Sync + 'static.
                let obj: Arc<Py<PyAny>> = Arc::new(value.clone().unbind());
                block_on_context(async { inner.set_object(&key, obj).await });
            }
        }
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    ///
    /// Returns the original Python type: JSON values as their Python
    /// equivalents, binary data as `bytes`, pickled objects as their
    /// original type, live references as-is. If the key is missing,
    /// or the stored value would resolve to Python `None`, the
    /// `default` argument is returned instead (default: `None`).
    ///
    /// Args:
    ///     key: The storage key.
    ///     default: Value to return when the key is missing. Defaults
    ///         to `None`.
    ///
    /// Returns:
    ///     The stored value in its original type, or `default`.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let val = self.get_inner(py, key)?;
        if val.bind(py).is_none() {
            return Ok(default.unwrap_or_else(|| py.None()));
        }
        Ok(val)
    }

    /// Emit an event into the internal routing queue.
    ///
    /// The event will be routed to any step whose `accepts` list includes
    /// its event type.
    ///
    /// Args:
    ///     event: The event to send.
    fn send_event(&self, _py: Python<'_>, event: PyRef<'_, PyEvent>) {
        let dynamic = blazen_events::DynamicEvent {
            event_type: event.event_type.clone(),
            data: event.data.clone(),
        };
        drop(event);
        let inner = self.inner.clone();
        block_on_context(async {
            inner.send_event(dynamic).await;
        });
    }

    /// Publish an event to the external broadcast stream.
    ///
    /// Consumers that subscribed via `WorkflowHandler.stream_events()` will
    /// receive this event. Unlike `send_event`, this does NOT route the
    /// event through the internal step registry.
    ///
    /// Args:
    ///     event: The event to publish to the stream.
    fn write_event_to_stream(&self, _py: Python<'_>, event: PyRef<'_, PyEvent>) {
        let dynamic = blazen_events::DynamicEvent {
            event_type: event.event_type.clone(),
            data: event.data.clone(),
        };
        drop(event);
        let inner = self.inner.clone();
        block_on_context(async {
            inner.write_event_to_stream(dynamic).await;
        });
    }

    /// Store raw binary data under the given key.
    ///
    /// Convenience method — equivalent to `ctx.set(key, data)` when `data`
    /// is `bytes`. Useful when you want to be explicit about binary storage.
    ///
    /// Args:
    ///     key: The storage key.
    ///     data: Raw bytes to store.
    #[gen_stub(skip)]
    fn set_bytes(&self, _py: Python<'_>, key: &str, data: &[u8]) {
        let inner = self.inner.clone();
        let key = key.to_string();
        let data = data.to_vec();
        block_on_context(async { inner.set_bytes(&key, data).await });
    }

    /// Retrieve raw binary data previously stored under the given key.
    ///
    /// Returns `default` (which itself defaults to `None`) if the key
    /// does not exist or the stored value is not binary data.
    ///
    /// Args:
    ///     key: The storage key.
    ///     default: Bytes to return when the key is missing. Defaults
    ///         to `None`.
    ///
    /// Returns:
    ///     The stored bytes, or `default`.
    #[pyo3(signature = (key, default=None))]
    fn get_bytes(&self, _py: Python<'_>, key: &str, default: Option<Vec<u8>>) -> Option<Vec<u8>> {
        let inner = self.inner.clone();
        let key = key.to_string();
        block_on_context(async { inner.get_bytes(&key).await }).or(default)
    }

    /// Get the workflow run ID.
    ///
    /// Returns:
    ///     The UUID string for this workflow run.
    fn run_id(&self, _py: Python<'_>) -> String {
        let inner = self.inner.clone();
        let id = block_on_context(async { inner.run_id().await });
        id.to_string()
    }

    /// Persistable workflow state. Survives `pause()` / `resume()`,
    /// checkpoints, and durable storage.
    ///
    /// ```python
    /// ctx.state.set("counter", 5)
    /// count = ctx.state.get("counter")
    /// ```
    #[getter]
    fn state(&self) -> PyStateNamespace {
        PyStateNamespace {
            inner: self.inner.clone(),
        }
    }

    /// Live in-process references. Identity is preserved within a
    /// single workflow run; subject to the workflow's
    /// `session_pause_policy` on snapshot.
    ///
    /// ```python
    /// ctx.session.set("conn", sqlite3.connect(":memory:"))
    /// conn = ctx.session.get("conn")  # same object as above
    /// ```
    #[getter]
    fn session(&self) -> PySessionNamespace {
        PySessionNamespace {
            inner: self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        "Context()".to_owned()
    }
}

// ---------------------------------------------------------------------------
// PyStateNamespace — persistable workflow state
// ---------------------------------------------------------------------------

/// Namespace for persistable workflow state.
///
/// Routes through the same 4-tier dispatch as the legacy `ctx.set` /
/// `ctx.get`: bytes/JSON/pickle/live-object. The first three tiers
/// are durable across `pause()` / `resume()` and checkpoint stores;
/// the live-object tier (used as a last resort for unpicklable values)
/// is in-process only.
#[gen_stub_pyclass]
#[pyclass(name = "StateNamespace", from_py_object)]
#[derive(Clone)]
pub struct PyStateNamespace {
    inner: blazen_core::Context,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStateNamespace {
    /// Store a value under the given key. See `Context.set` for the
    /// 4-tier dispatch order.
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Delegate through PyContext::set for the existing tier dispatch.
        let ctx = PyContext::new(self.inner.clone());
        ctx.set(py, key, value)
    }

    /// Retrieve a value previously stored under the given key.
    /// If the key is missing, returns `default` (defaults to `None`).
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let ctx = PyContext::new(self.inner.clone());
        ctx.get(py, key, default)
    }

    /// Store raw binary data under the given key.
    #[gen_stub(skip)]
    fn set_bytes(&self, py: Python<'_>, key: &str, data: &[u8]) {
        let ctx = PyContext::new(self.inner.clone());
        ctx.set_bytes(py, key, data);
    }

    /// Retrieve raw binary data previously stored under the given key.
    /// If the key is missing, returns `default` (defaults to `None`).
    #[pyo3(signature = (key, default=None))]
    fn get_bytes(&self, py: Python<'_>, key: &str, default: Option<Vec<u8>>) -> Option<Vec<u8>> {
        let ctx = PyContext::new(self.inner.clone());
        ctx.get_bytes(py, key, default)
    }

    fn __setitem__(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set(py, key, value)
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let val = self.get(py, key, None)?;
        if val.bind(py).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned()));
        }
        Ok(val)
    }

    fn __contains__(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        let val = self.get(py, key, None)?;
        Ok(!val.bind(py).is_none())
    }

    fn __repr__(&self) -> String {
        "StateNamespace()".to_owned()
    }
}

// ---------------------------------------------------------------------------
// PySessionNamespace — live in-process references
// ---------------------------------------------------------------------------

/// Namespace for live in-process references.
///
/// Values are stored as `Arc<Py<PyAny>>` in the underlying
/// `ContextInner.objects` map. Identity is preserved within a single
/// workflow run; the entries are *not* serialised into snapshots
/// (subject to the workflow's `session_pause_policy` for what happens
/// at pause time).
#[gen_stub_pyclass]
#[pyclass(name = "SessionNamespace", from_py_object)]
#[derive(Clone)]
pub struct PySessionNamespace {
    inner: blazen_core::Context,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySessionNamespace {
    /// Store a live reference under the given key. The value is *not*
    /// serialised; identity is preserved within this run.
    fn set(&self, _py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) {
        let inner = self.inner.clone();
        let key = key.to_string();
        let obj: Arc<Py<PyAny>> = Arc::new(value.clone().unbind());
        block_on_context(async { inner.set_object(&key, obj).await });
    }

    /// Retrieve a live reference previously stored under the given key.
    /// If the key is missing, returns `default` (defaults to `None`).
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> Option<Py<PyAny>> {
        let inner = self.inner.clone();
        let key_owned = key.to_string();
        let obj: Option<Arc<Py<PyAny>>> =
            block_on_context(async { inner.get_object::<Arc<Py<PyAny>>>(&key_owned).await });
        obj.map(|o| o.clone_ref(py)).or(default)
    }

    /// Remove a live reference stored under the given key.
    fn remove(&self, _py: Python<'_>, key: &str) {
        let inner = self.inner.clone();
        let key = key.to_string();
        block_on_context(async { inner.remove_object(&key).await });
    }

    /// Check whether a live reference exists under the given key.
    fn has(&self, _py: Python<'_>, key: &str) -> bool {
        let inner = self.inner.clone();
        let key = key.to_string();
        block_on_context(async { inner.has_object(&key).await })
    }

    fn __setitem__(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) {
        self.set(py, key, value);
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.get(py, key, None)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))
    }

    fn __contains__(&self, _py: Python<'_>, key: &str) -> bool {
        self.has(_py, key)
    }

    fn __repr__(&self) -> String {
        "SessionNamespace()".to_owned()
    }
}

impl PyContext {
    /// Create a new `PyContext` wrapping a Rust `Context`.
    pub fn new(inner: blazen_core::Context) -> Self {
        Self { inner }
    }

    // 4-tier dispatch helper. Returns `py.None()` on a complete miss;
    // the public `get` substitutes the user-supplied default.
    fn get_inner(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        // Check for BlazenState per-field metadata.
        {
            let meta_key = format!("{key}.__blazen_meta__");
            let inner = self.inner.clone();
            if let Some(blazen_core::StateValue::Json(meta)) =
                block_on_context(async { inner.get_value(&meta_key).await })
            {
                return self.get_blazen_state(py, key, &meta);
            }
        }

        // Fall through to Rust core (tiers 1-3).
        let inner = self.inner.clone();
        let key_owned = key.to_string();
        let val = block_on_context(async { inner.get_value(&key_owned).await });
        match val {
            Some(blazen_core::StateValue::Json(v)) => crate::convert::json_to_py(py, &v),
            Some(blazen_core::StateValue::Bytes(b)) => {
                Ok(pyo3::types::PyBytes::new(py, &b.0).into_any().unbind())
            }
            Some(blazen_core::StateValue::Native(b)) => {
                let pickle = py.import("pickle")?;
                let obj = pickle.call_method1("loads", (&b.0[..],))?;
                Ok(obj.unbind())
            }
            None => {
                // Check opaque object store (tier 4 values).
                // Stored as Arc<Py<PyAny>> because Py<PyAny> isn't Clone.
                let inner = self.inner.clone();
                let key_owned = key.to_string();
                if let Some(obj) =
                    block_on_context(async { inner.get_object::<Arc<Py<PyAny>>>(&key_owned).await })
                {
                    return Ok(obj.clone_ref(py));
                }
                Ok(py.None())
            }
        }
    }

    /// Store a [`BlazenState`](super::state::PyBlazenState) per-field.
    fn set_blazen_state(
        &self,
        py: Python<'_>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let cls = value.get_type();

        // Read Meta.transient (set of field names to skip serialization for).
        let transient: std::collections::HashSet<String> = cls
            .getattr("Meta")
            .ok()
            .and_then(|meta| meta.getattr("transient").ok())
            .and_then(|t| t.extract().ok())
            .unwrap_or_default();

        // Read Meta.store_by (dict of field name → FieldStore).
        let store_by: Option<Bound<'_, pyo3::types::PyDict>> = cls
            .getattr("Meta")
            .ok()
            .and_then(|meta| meta.getattr("store_by").ok())
            .and_then(|sb| sb.cast_into().ok());

        // Iterate __dict__ and store each field.
        let dict = value
            .getattr("__dict__")?
            .cast_into::<pyo3::types::PyDict>()?;
        let mut field_names: Vec<String> = Vec::new();

        for (k, v) in dict.iter() {
            let field_name: String = k.extract()?;
            field_names.push(field_name.clone());
            let field_key = format!("{key}.{field_name}");

            // Check store_by first.
            if let Some(ref sb) = store_by
                && let Ok(Some(store)) = sb.get_item(&field_name)
            {
                // Call FieldStore.save(key, value, ctx)
                let self_py = Py::new(py, self.clone())?;
                store.call_method1("save", (&field_key, &v, self_py))?;
                continue;
            }

            // Normal per-field storage via the 4-tier dispatch.
            self.set(py, &field_key, &v)?;
        }

        // Store metadata so get() can reconstruct.
        let meta = serde_json::json!({
            "class_module": cls.getattr("__module__")?.extract::<String>()?,
            "class_name": cls.getattr("__qualname__")?.extract::<String>()?,
            "fields": field_names,
            "transient": transient.into_iter().collect::<Vec<_>>(),
        });
        let inner = self.inner.clone();
        let meta_key = format!("{key}.__blazen_meta__");
        block_on_context(async { inner.set(&meta_key, meta).await });
        Ok(())
    }

    /// Reconstruct a [`BlazenState`](super::state::PyBlazenState) from per-field storage.
    fn get_blazen_state(
        &self,
        py: Python<'_>,
        key: &str,
        meta: &serde_json::Value,
    ) -> PyResult<Py<PyAny>> {
        let class_module = meta["class_module"]
            .as_str()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing class_module"))?;
        let class_name = meta["class_name"]
            .as_str()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing class_name"))?;
        let fields = meta["fields"]
            .as_array()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing fields"))?;

        // Import the class.
        let module = py.import(class_module)?;
        // Handle nested qualnames like "Outer.Inner" by traversing attrs.
        let mut cls: Bound<'_, PyAny> = module.into_any();
        for part in class_name.split('.') {
            cls = cls.getattr(part)?;
        }

        // Create instance via __new__ (bypasses __init__).
        let obj = cls.call_method1("__new__", (&cls,))?;

        // Read Meta.store_by for custom loaders.
        let store_by: Option<Bound<'_, pyo3::types::PyDict>> = cls
            .getattr("Meta")
            .ok()
            .and_then(|meta| meta.getattr("store_by").ok())
            .and_then(|sb| sb.cast_into().ok());

        // Restore each field.
        for field_val in fields {
            let field_name = field_val
                .as_str()
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad field name"))?;
            let field_key = format!("{key}.{field_name}");

            // Check store_by first.
            if let Some(ref sb) = store_by
                && let Ok(Some(store)) = sb.get_item(field_name)
            {
                let self_py = Py::new(py, self.clone())?;
                let value = store.call_method1("load", (&field_key, self_py))?;
                obj.setattr(field_name, value)?;
                continue;
            }

            // Normal get.
            let value = self.get(py, &field_key, None)?;
            obj.setattr(field_name, value)?;
        }

        // Call restore() if defined.
        if obj.hasattr("restore")? {
            obj.call_method0("restore")?;
        }

        Ok(obj.unbind())
    }
}
