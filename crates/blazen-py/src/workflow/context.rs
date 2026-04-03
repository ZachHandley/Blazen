//! Python wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! Accepts any Python value — JSON-serializable types are stored efficiently
//! as JSON, `bytes`/`bytearray` as raw binary, picklable objects via pickle,
//! and unpicklable objects (DB connections, file handles) as live references.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use super::event::PyEvent;

/// Run a future to completion, handling both inside-tokio and outside-tokio
/// contexts. Uses `block_in_place` when called from a tokio worker thread
/// (e.g. from within a step handler), and falls back to the pyo3-async-runtimes
/// runtime otherwise.
fn block_on_context<F: std::future::Future>(fut: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // We're inside a tokio runtime -- use block_in_place to avoid panics
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        // No tokio runtime on this thread -- use the pyo3 runtime
        pyo3_async_runtimes::tokio::get_runtime().block_on(fut)
    }
}

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
#[pyclass(name = "Context", from_py_object)]
#[derive(Clone)]
pub struct PyContext {
    pub(crate) inner: blazen_core::Context,
    /// Live Python object references for unpicklable values.
    /// Shared across steps via Arc. NOT persisted through snapshots.
    locals: Arc<Mutex<HashMap<String, Py<PyAny>>>>,
}

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
        let inner = self.inner.clone();
        let key = key.to_string();

        if let Ok(bytes) = value.extract::<Vec<u8>>() {
            // Tier 1: bytes/bytearray → Bytes variant
            self.locals.lock().unwrap().remove(&key);
            block_on_context(async { inner.set_bytes(&key, bytes).await });
        } else if let Ok(json_val) = super::event::try_py_to_json(py, value) {
            // Tier 2: JSON-serializable → Json variant
            self.locals.lock().unwrap().remove(&key);
            block_on_context(async { inner.set(&key, json_val).await });
        } else {
            // Tier 3: try pickle → Native variant
            let pickle = py.import("pickle")?;
            match pickle.call_method1("dumps", (value,)) {
                Ok(pickled_obj) => {
                    let pickled: Vec<u8> = pickled_obj.extract()?;
                    let sv = blazen_core::StateValue::Native(blazen_core::BytesWrapper(pickled));
                    self.locals.lock().unwrap().remove(&key);
                    block_on_context(async { inner.set_value(&key, sv).await });
                }
                Err(_) => {
                    // Tier 4: unpicklable → store live Python reference
                    self.locals
                        .lock()
                        .unwrap()
                        .insert(key, value.clone().unbind());
                }
            }
        }
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    ///
    /// Returns the original Python type: JSON values as their Python
    /// equivalents, binary data as `bytes`, pickled objects as their
    /// original type, live references as-is. Returns `None` if the key
    /// does not exist.
    ///
    /// Args:
    ///     key: The storage key.
    ///
    /// Returns:
    ///     The stored value in its original type, or None.
    fn get(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        // Check live object store first (tier 4 values).
        if let Some(obj) = self.locals.lock().unwrap().get(key) {
            return Ok(obj.clone_ref(py));
        }

        // Fall through to Rust core (tiers 1-3).
        let inner = self.inner.clone();
        let key = key.to_string();
        let val = block_on_context(async { inner.get_value(&key).await });
        match val {
            Some(blazen_core::StateValue::Json(v)) => super::event::json_to_py(py, &v),
            Some(blazen_core::StateValue::Bytes(b)) => {
                Ok(pyo3::types::PyBytes::new(py, &b.0).into_any().unbind())
            }
            Some(blazen_core::StateValue::Native(b)) => {
                let pickle = py.import("pickle")?;
                let obj = pickle.call_method1("loads", (&b.0[..],))?;
                Ok(obj.unbind())
            }
            None => Ok(py.None()),
        }
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
    fn set_bytes(&self, _py: Python<'_>, key: &str, data: &[u8]) {
        let inner = self.inner.clone();
        let key = key.to_string();
        let data = data.to_vec();
        self.locals.lock().unwrap().remove(&key);
        block_on_context(async { inner.set_bytes(&key, data).await });
    }

    /// Retrieve raw binary data previously stored under the given key.
    ///
    /// Returns None if the key does not exist or the stored value is
    /// not binary data.
    ///
    /// Args:
    ///     key: The storage key.
    ///
    /// Returns:
    ///     The stored bytes, or None.
    fn get_bytes(&self, _py: Python<'_>, key: &str) -> Option<Vec<u8>> {
        let inner = self.inner.clone();
        let key = key.to_string();
        block_on_context(async { inner.get_bytes(&key).await })
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

    fn __repr__(&self) -> String {
        "Context()".to_owned()
    }
}

impl PyContext {
    /// Create a new `PyContext` wrapping a Rust `Context`.
    pub fn new(inner: blazen_core::Context) -> Self {
        Self {
            inner,
            locals: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
