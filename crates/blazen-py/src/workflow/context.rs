//! Python wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! Accepts any Python value — JSON-serializable types are stored efficiently
//! as JSON, `bytes`/`bytearray` as raw binary, and everything else (Pydantic
//! models, custom classes, etc.) is pickled automatically.

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
/// Accepts any Python value — JSON-serializable types, raw bytes, or
/// arbitrary objects (pickled automatically).
///
/// Example:
///     >>> def my_step(ctx: Context, ev: Event) -> Event:
///     ...     ctx.set("counter", 42)
///     ...     ctx.set("model", MyPydanticModel(name="foo"))
///     ...     val = ctx.get("counter")  # returns 42
///     ...     model = ctx.get("model")  # returns MyPydanticModel
#[pyclass(name = "Context", from_py_object)]
#[derive(Clone)]
pub struct PyContext {
    pub(crate) inner: blazen_core::Context,
}

#[pymethods]
impl PyContext {
    /// Store any value under the given key.
    ///
    /// - `bytes`/`bytearray` → stored as raw binary
    /// - JSON-serializable (dict, list, str, int, float, bool, None) → stored as JSON
    /// - Everything else (Pydantic models, custom classes) → pickled automatically
    ///
    /// Args:
    ///     key: The storage key.
    ///     value: Any Python value.
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let inner = self.inner.clone();
        let key = key.to_string();

        if let Ok(bytes) = value.extract::<Vec<u8>>() {
            // bytes/bytearray → Bytes variant (raw binary, returned as bytes on get)
            block_on_context(async { inner.set_bytes(&key, bytes).await });
        } else if let Ok(json_val) = super::event::try_py_to_json(py, value) {
            // JSON-serializable primitive/container → Json variant
            block_on_context(async { inner.set(&key, json_val).await });
        } else {
            // Anything else (Pydantic, custom class, etc.) → pickle → Native variant
            let pickle = py.import("pickle")?;
            let pickled: Vec<u8> = pickle.call_method1("dumps", (value,))?.extract()?;
            let sv = blazen_core::StateValue::Native(blazen_core::BytesWrapper(pickled));
            block_on_context(async { inner.set_value(&key, sv).await });
        }
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    ///
    /// Returns the original Python type: JSON values as their Python
    /// equivalents, binary data as `bytes`, pickled objects as their
    /// original type. Returns `None` if the key does not exist.
    ///
    /// Args:
    ///     key: The storage key.
    ///
    /// Returns:
    ///     The stored value in its original type, or None.
    fn get(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
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
        Self { inner }
    }
}
