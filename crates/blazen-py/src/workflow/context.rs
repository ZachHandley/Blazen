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
//!
//! ## Sync vs async
//!
//! Every storage-touching method is exposed in **two** forms via the
//! [`blazen_macros::py_async`] codegen attribute on each `#[pymethods]`
//! `impl` block:
//!
//! - **Sync** (`set`, `get`, …) — drives the future to completion via
//!   [`crate::convert::block_on_context`], which releases the GIL for
//!   the duration of the wait so the await body is free to reattach.
//! - **Async** (`aset`, `aget`, …) — returns an asyncio awaitable via
//!   `pyo3_async_runtimes::tokio::future_into_py`, for use inside an
//!   `asyncio.run(...)` event loop.

use std::sync::Arc;

use blazen_macros::py_async;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::event::PyEvent;

/// Tier-tagged result of `Context.set`'s value classification. Built
/// while the GIL is held; the async store path consumes it without
/// touching Python.
enum PreparedSet {
    BlazenState(Py<PyAny>),
    Bytes(Vec<u8>),
    Json(serde_json::Value),
    Pickle(Vec<u8>),
    Live(Arc<Py<PyAny>>),
}

/// Classify a Python value into a [`PreparedSet`] under the GIL. The
/// returned `Prepared*` variants are `Send + 'static` so the async store
/// path can move them across `await` points.
fn classify_for_set(py: Python<'_>, value: &Py<PyAny>) -> PyResult<PreparedSet> {
    let bound = value.bind(py);
    if bound.is_instance_of::<super::state::PyBlazenState>() {
        return Ok(PreparedSet::BlazenState(value.clone_ref(py)));
    }
    if bound.is_instance_of::<pyo3::types::PyBytes>()
        || bound.is_instance_of::<pyo3::types::PyByteArray>()
    {
        let b: Vec<u8> = bound.extract()?;
        return Ok(PreparedSet::Bytes(b));
    }
    if let Ok(json) = crate::convert::try_py_to_json(py, bound) {
        return Ok(PreparedSet::Json(json));
    }
    let pickle = py.import("pickle")?;
    if let Ok(pickled) = pickle.call_method1("dumps", (bound,)) {
        let bytes: Vec<u8> = pickled.extract()?;
        return Ok(PreparedSet::Pickle(bytes));
    }
    Ok(PreparedSet::Live(Arc::new(value.clone_ref(py))))
}

/// Shared workflow context accessible by all steps.
///
/// Provides typed key/value storage, event emission, and stream publishing.
/// Accepts any Python value — JSON-serializable types, raw bytes, picklable
/// objects, or live references for unpicklable objects (DB connections, etc.).
///
/// Example:
/// ```text
///  >>> def my_step(ctx: Context, ev: Event) -> Event:
///  ...     ctx.set("counter", 42)
///  ...     ctx.set("model", MyPydanticModel(name="foo"))
///  ...     ctx.set("db", sqlite3.connect(":memory:"))
///  ...     val = ctx.get("counter")  # returns 42
///  ...     db = ctx.get("db")        # returns the same connection
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "Context", from_py_object)]
#[derive(Clone)]
pub struct PyContext {
    pub(crate) inner: blazen_core::Context,
}

#[py_async]
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
    async fn set(&self, key: String, value: Py<PyAny>) -> PyResult<()> {
        let prepared = Python::attach(|py| classify_for_set(py, &value))?;
        match prepared {
            PreparedSet::BlazenState(val) => {
                self.clone().set_blazen_state_async(key, val).await?;
            }
            PreparedSet::Bytes(b) => {
                self.inner.clone().set_bytes(&key, b).await;
            }
            PreparedSet::Json(j) => {
                self.inner.clone().set(&key, j).await;
            }
            PreparedSet::Pickle(b) => {
                let sv = blazen_core::StateValue::Native(blazen_core::BytesWrapper(b));
                self.inner.clone().set_value(&key, sv).await;
            }
            PreparedSet::Live(arc) => {
                self.inner.clone().set_object(&key, arc).await;
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
    async fn get(&self, key: String, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let value = self.clone().get_inner_async(key).await?;
        Ok(Python::attach(|py| {
            if value.bind(py).is_none() {
                default.unwrap_or_else(|| py.None())
            } else {
                value
            }
        }))
    }

    /// Emit an event into the internal routing queue.
    ///
    /// The event will be routed to any step whose `accepts` list includes
    /// its event type.
    ///
    /// Args:
    ///     event: The event to send.
    async fn send_event(&self, event: Py<PyEvent>) -> PyResult<()> {
        let dynamic = Python::attach(|py| {
            let ev = event.borrow(py);
            blazen_events::DynamicEvent::from_json(ev.event_type.clone(), ev.data.clone())
        });
        self.inner.clone().send_event(dynamic).await;
        Ok(())
    }

    /// Publish an event to the external broadcast stream.
    ///
    /// Consumers that subscribed via `WorkflowHandler.stream_events()` will
    /// receive this event. Unlike `send_event`, this does NOT route the
    /// event through the internal step registry.
    ///
    /// Args:
    ///     event: The event to publish to the stream.
    async fn write_event_to_stream(&self, event: Py<PyEvent>) -> PyResult<()> {
        let dynamic = Python::attach(|py| {
            let ev = event.borrow(py);
            blazen_events::DynamicEvent::from_json(ev.event_type.clone(), ev.data.clone())
        });
        self.inner.clone().write_event_to_stream(dynamic).await;
        Ok(())
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
    async fn set_bytes(&self, key: String, data: Vec<u8>) -> PyResult<()> {
        self.inner.clone().set_bytes(&key, data).await;
        Ok(())
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
    async fn get_bytes(&self, key: String, default: Option<Vec<u8>>) -> PyResult<Option<Vec<u8>>> {
        Ok(self.inner.clone().get_bytes(&key).await.or(default))
    }

    /// Get the workflow run ID.
    ///
    /// Returns:
    ///     The UUID string for this workflow run.
    async fn run_id(&self) -> PyResult<String> {
        Ok(self.inner.clone().run_id().await.to_string())
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

#[py_async]
#[gen_stub_pymethods]
#[pymethods]
impl PyStateNamespace {
    /// Store a value under the given key. See `Context.set` for the
    /// 4-tier dispatch order.
    async fn set(&self, key: String, value: Py<PyAny>) -> PyResult<()> {
        PyContext::new(self.inner.clone())
            .set_body(key, value)
            .await
    }

    /// Retrieve a value previously stored under the given key.
    /// If the key is missing, returns `default` (defaults to `None`).
    #[pyo3(signature = (key, default=None))]
    async fn get(&self, key: String, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        PyContext::new(self.inner.clone())
            .get_body(key, default)
            .await
    }

    /// Store raw binary data under the given key.
    #[gen_stub(skip)]
    async fn set_bytes(&self, key: String, data: Vec<u8>) -> PyResult<()> {
        self.inner.clone().set_bytes(&key, data).await;
        Ok(())
    }

    /// Retrieve raw binary data previously stored under the given key.
    /// If the key is missing, returns `default` (defaults to `None`).
    #[pyo3(signature = (key, default=None))]
    async fn get_bytes(&self, key: String, default: Option<Vec<u8>>) -> PyResult<Option<Vec<u8>>> {
        Ok(self.inner.clone().get_bytes(&key).await.or(default))
    }

    fn __setitem__(&self, py: Python<'_>, key: String, value: Py<PyAny>) -> PyResult<()> {
        self.set(py, key, value)
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let val = self.get(py, key.clone(), None)?;
        if val.bind(py).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(key));
        }
        Ok(val)
    }

    fn __contains__(&self, py: Python<'_>, key: String) -> PyResult<bool> {
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

#[py_async]
#[gen_stub_pymethods]
#[pymethods]
impl PySessionNamespace {
    /// Store a live reference under the given key. The value is *not*
    /// serialised; identity is preserved within this run.
    async fn set(&self, key: String, value: Py<PyAny>) -> PyResult<()> {
        let obj: Arc<Py<PyAny>> = Arc::new(value);
        self.inner.clone().set_object(&key, obj).await;
        Ok(())
    }

    /// Retrieve a live reference previously stored under the given key.
    /// If the key is missing, returns `default` (defaults to `None`).
    #[pyo3(signature = (key, default=None))]
    async fn get(&self, key: String, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let obj: Option<Arc<Py<PyAny>>> =
            self.inner.clone().get_object::<Arc<Py<PyAny>>>(&key).await;
        Ok(Python::attach(|py| match obj {
            Some(o) => o.clone_ref(py),
            None => default.unwrap_or_else(|| py.None()),
        }))
    }

    /// Remove a live reference stored under the given key.
    async fn remove(&self, key: String) -> PyResult<()> {
        self.inner.clone().remove_object(&key).await;
        Ok(())
    }

    /// Check whether a live reference exists under the given key.
    async fn has(&self, key: String) -> PyResult<bool> {
        Ok(self.inner.clone().has_object(&key).await)
    }

    fn __setitem__(&self, py: Python<'_>, key: String, value: Py<PyAny>) -> PyResult<()> {
        self.set(py, key, value)
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let val = self.get(py, key.clone(), None)?;
        if val.bind(py).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(key));
        }
        Ok(val)
    }

    fn __contains__(&self, py: Python<'_>, key: String) -> PyResult<bool> {
        self.has(py, key)
    }

    fn __repr__(&self) -> String {
        "SessionNamespace()".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Internal async helpers (not exposed to Python). Re-used by both the
// sync and async wrappers via the `#[py_async]` macro expansion, and by
// the BlazenState recursive store/load paths.
// ---------------------------------------------------------------------------

impl PyContext {
    /// Create a new `PyContext` wrapping a Rust `Context`.
    pub fn new(inner: blazen_core::Context) -> Self {
        Self { inner }
    }

    /// The async-only body of `set`. Public to this crate so namespaces
    /// can delegate without going back through the `#[pymethods]`
    /// wrapper.
    pub(crate) async fn set_body(self, key: String, value: Py<PyAny>) -> PyResult<()> {
        let prepared = Python::attach(|py| classify_for_set(py, &value))?;
        match prepared {
            PreparedSet::BlazenState(val) => {
                Box::pin(self.set_blazen_state_async(key, val)).await?;
            }
            PreparedSet::Bytes(b) => {
                self.inner.set_bytes(&key, b).await;
            }
            PreparedSet::Json(j) => {
                self.inner.set(&key, j).await;
            }
            PreparedSet::Pickle(b) => {
                let sv = blazen_core::StateValue::Native(blazen_core::BytesWrapper(b));
                self.inner.set_value(&key, sv).await;
            }
            PreparedSet::Live(arc) => {
                self.inner.set_object(&key, arc).await;
            }
        }
        Ok(())
    }

    /// The async-only body of `get`. Returns `py.None()` if the key is
    /// missing (callers substitute the user-supplied default).
    pub(crate) async fn get_body(
        self,
        key: String,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let value = self.get_inner_async(key).await?;
        Ok(Python::attach(|py| {
            if value.bind(py).is_none() {
                default.unwrap_or_else(|| py.None())
            } else {
                value
            }
        }))
    }

    /// 4-tier dispatch helper. Returns `py.None()` on a complete miss.
    pub(crate) async fn get_inner_async(self, key: String) -> PyResult<Py<PyAny>> {
        // Check for BlazenState per-field metadata first.
        let meta_key = format!("{key}.__blazen_meta__");
        if let Some(blazen_core::StateValue::Json(meta)) = self.inner.get_value(&meta_key).await {
            return Box::pin(self.get_blazen_state_async(key, meta)).await;
        }

        // Fall through to Rust core (tiers 1-3).
        let val = self.inner.get_value(&key).await;
        match val {
            Some(blazen_core::StateValue::Json(v)) => {
                Python::attach(|py| crate::convert::json_to_py(py, &v))
            }
            Some(blazen_core::StateValue::Bytes(b)) => Ok(Python::attach(|py| {
                pyo3::types::PyBytes::new(py, &b.0).into_any().unbind()
            })),
            Some(blazen_core::StateValue::Native(b)) => Python::attach(|py| {
                let pickle = py.import("pickle")?;
                let obj = pickle.call_method1("loads", (&b.0[..],))?;
                Ok(obj.unbind())
            }),
            None => {
                // Tier 4: check the opaque object store. Single
                // `Python::attach` handles both the hit (clone the
                // existing handle) and the miss (`py.None()`); folding
                // them into one match arm keeps the closure body
                // non-trivial so clippy's `redundant_closure_for_method_calls`
                // doesn't bait `Python::None` as a replacement (which
                // fails to unify against the higher-ranked `Python<'_>`
                // bound on `Python::attach`).
                let obj = self.inner.get_object::<Arc<Py<PyAny>>>(&key).await;
                Ok(Python::attach(|py| match obj {
                    Some(o) => o.clone_ref(py),
                    None => py.None(),
                }))
            }
        }
    }

    /// Store a [`BlazenState`](super::state::PyBlazenState) per-field.
    /// Recursive via `set_body` on each field.
    pub(crate) async fn set_blazen_state_async(
        self,
        key: String,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        // Phase 1: classify each field under the GIL.
        let (plans, meta_key, meta) =
            Python::attach(|py| classify_blazen_state_fields(py, &key, &value))?;

        // Phase 2: execute each plan. Custom stores run synchronously
        // under the GIL (FieldStore.save is a Python callable); normal
        // fields recurse through `set_body`.
        for plan in plans {
            match plan.kind {
                FieldKind::CustomStore { store, value } => {
                    let self_clone = self.clone();
                    Python::attach(|py| -> PyResult<()> {
                        let self_py = Py::new(py, self_clone)?;
                        let store_bound = store.bind(py);
                        let value_bound = value.bind(py);
                        store_bound.call_method1(
                            "save",
                            (plan.field_key.as_str(), value_bound, self_py),
                        )?;
                        Ok(())
                    })?;
                }
                FieldKind::Normal(v) => {
                    Box::pin(self.clone().set_body(plan.field_key, v)).await?;
                }
            }
        }

        // Phase 3: write the metadata so `get_body` can reconstruct.
        self.inner.set(&meta_key, meta).await;
        Ok(())
    }

    /// Reconstruct a [`BlazenState`](super::state::PyBlazenState) from per-field storage.
    pub(crate) async fn get_blazen_state_async(
        self,
        key: String,
        meta: serde_json::Value,
    ) -> PyResult<Py<PyAny>> {
        let fields = parse_blazen_state_fields(&meta)?;
        let (obj, plans, has_restore) =
            Python::attach(|py| plan_blazen_state_load(py, &key, &fields, &meta))?;

        for plan in plans {
            match plan.kind {
                LoadKind::CustomStore(store) => {
                    let self_clone = self.clone();
                    let field_name = plan.field_name.clone();
                    let field_key = plan.field_key.clone();
                    Python::attach(|py| -> PyResult<()> {
                        let self_py = Py::new(py, self_clone)?;
                        let store_bound = store.bind(py);
                        let value =
                            store_bound.call_method1("load", (field_key.as_str(), self_py))?;
                        obj.bind(py).setattr(field_name.as_str(), value)?;
                        Ok(())
                    })?;
                }
                LoadKind::Normal => {
                    let value =
                        Box::pin(self.clone().get_inner_async(plan.field_key.clone())).await?;
                    Python::attach(|py| obj.bind(py).setattr(plan.field_name.as_str(), value))?;
                }
            }
        }

        if has_restore {
            Python::attach(|py| -> PyResult<()> {
                obj.bind(py).call_method0("restore")?;
                Ok(())
            })?;
        }

        Ok(obj)
    }
}

// ---------------------------------------------------------------------------
// BlazenState helper types (private to this module)
// ---------------------------------------------------------------------------

struct FieldPlan {
    field_key: String,
    kind: FieldKind,
}

enum FieldKind {
    /// Field has a custom `FieldStore` in `Meta.store_by`.
    CustomStore { store: Py<PyAny>, value: Py<PyAny> },
    /// Normal per-field storage (recurses through `set_body`).
    Normal(Py<PyAny>),
}

struct LoadPlan {
    field_name: String,
    field_key: String,
    kind: LoadKind,
}

enum LoadKind {
    CustomStore(Py<PyAny>),
    Normal,
}

/// GIL-bound classification step for `set_blazen_state_async`. Returns
/// the per-field plans plus the meta blob to persist after the field
/// writes complete.
fn classify_blazen_state_fields(
    py: Python<'_>,
    key: &str,
    value: &Py<PyAny>,
) -> PyResult<(Vec<FieldPlan>, String, serde_json::Value)> {
    let bound = value.bind(py);
    let cls = bound.get_type();

    let transient: std::collections::HashSet<String> = cls
        .getattr("Meta")
        .ok()
        .and_then(|meta| meta.getattr("transient").ok())
        .and_then(|t| t.extract().ok())
        .unwrap_or_default();

    let store_by: Option<Bound<'_, pyo3::types::PyDict>> = cls
        .getattr("Meta")
        .ok()
        .and_then(|meta| meta.getattr("store_by").ok())
        .and_then(|sb| sb.cast_into().ok());

    let dict = bound
        .getattr("__dict__")?
        .cast_into::<pyo3::types::PyDict>()?;
    let mut field_names: Vec<String> = Vec::new();
    let mut plans: Vec<FieldPlan> = Vec::new();

    for (k, v) in dict.iter() {
        let field_name: String = k.extract()?;
        field_names.push(field_name.clone());
        let field_key = format!("{key}.{field_name}");

        let kind = if let Some(ref sb) = store_by
            && let Ok(Some(store)) = sb.get_item(&field_name)
        {
            FieldKind::CustomStore {
                store: store.unbind(),
                value: v.unbind(),
            }
        } else {
            FieldKind::Normal(v.unbind())
        };

        plans.push(FieldPlan { field_key, kind });
    }

    let meta = serde_json::json!({
        "class_module": cls.getattr("__module__")?.extract::<String>()?,
        "class_name": cls.getattr("__qualname__")?.extract::<String>()?,
        "fields": field_names,
        "transient": transient.into_iter().collect::<Vec<_>>(),
    });
    let meta_key = format!("{key}.__blazen_meta__");

    Ok((plans, meta_key, meta))
}

/// Extract the ordered `fields` list from a stored meta blob.
fn parse_blazen_state_fields(meta: &serde_json::Value) -> PyResult<Vec<String>> {
    meta["fields"]
        .as_array()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing fields"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad field name"))
        })
        .collect()
}

/// GIL-bound class import + per-field load planning for
/// `get_blazen_state_async`. Returns the fresh instance, the plans, and
/// whether the class has a `restore()` method to call at the end.
fn plan_blazen_state_load(
    py: Python<'_>,
    key: &str,
    fields: &[String],
    meta: &serde_json::Value,
) -> PyResult<(Py<PyAny>, Vec<LoadPlan>, bool)> {
    let class_module = meta["class_module"]
        .as_str()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing class_module"))?;
    let class_name = meta["class_name"]
        .as_str()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing class_name"))?;

    let module = py.import(class_module)?;
    let mut cls: Bound<'_, PyAny> = module.into_any();
    for part in class_name.split('.') {
        cls = cls.getattr(part)?;
    }
    let obj = cls.call_method1("__new__", (&cls,))?;

    let store_by: Option<Bound<'_, pyo3::types::PyDict>> = cls
        .getattr("Meta")
        .ok()
        .and_then(|m| m.getattr("store_by").ok())
        .and_then(|sb| sb.cast_into().ok());

    let mut plans: Vec<LoadPlan> = Vec::new();
    for field_name in fields {
        let field_key = format!("{key}.{field_name}");
        let kind = if let Some(ref sb) = store_by
            && let Ok(Some(store)) = sb.get_item(field_name)
        {
            LoadKind::CustomStore(store.unbind())
        } else {
            LoadKind::Normal
        };
        plans.push(LoadPlan {
            field_name: field_name.clone(),
            field_key,
            kind,
        });
    }

    let has_restore = obj.hasattr("restore")?;
    Ok((obj.unbind(), plans, has_restore))
}
