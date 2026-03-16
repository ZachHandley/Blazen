//! Python wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! Python values are serialized to JSON before storage in the Rust context,
//! and deserialized back when retrieved. This means all Python context values
//! must be JSON-serializable.

use pyo3::prelude::*;

use crate::event::PyEvent;

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
/// All values are stored as JSON internally, so they must be JSON-serializable.
///
/// Example:
///     >>> def my_step(ctx: Context, ev: Event) -> Event:
///     ...     ctx.set("counter", 42)
///     ...     val = ctx.get("counter")  # returns 42
///     ...     ctx.send_event(Event("NextStep", data="hello"))
#[pyclass(name = "Context", from_py_object)]
#[derive(Clone)]
pub struct PyContext {
    pub(crate) inner: blazen_core::Context,
}

#[pymethods]
impl PyContext {
    /// Store a value under the given key.
    ///
    /// The value is serialized to JSON before storage.
    ///
    /// Args:
    ///     key: The storage key.
    ///     value: Any JSON-serializable Python value.
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let json_val = crate::event::py_to_json(py, value)?;
        let inner = self.inner.clone();
        let key = key.to_string();
        block_on_context(async {
            inner.set(&key, json_val).await;
        });
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    ///
    /// Returns `None` if the key does not exist.
    ///
    /// Args:
    ///     key: The storage key.
    ///
    /// Returns:
    ///     The stored value, or None.
    fn get(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let key = key.to_string();
        let val: Option<serde_json::Value> = block_on_context(async { inner.get(&key).await });
        match val {
            Some(v) => crate::event::json_to_py(py, &v),
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
    fn send_event(&self, _py: Python<'_>, event: PyRef<'_, PyEvent>) -> PyResult<()> {
        let dynamic = blazen_events::DynamicEvent {
            event_type: event.event_type.clone(),
            data: event.data.clone(),
        };
        drop(event);
        let inner = self.inner.clone();
        block_on_context(async {
            inner.send_event(dynamic).await;
        });
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
    fn write_event_to_stream(&self, _py: Python<'_>, event: PyRef<'_, PyEvent>) -> PyResult<()> {
        let dynamic = blazen_events::DynamicEvent {
            event_type: event.event_type.clone(),
            data: event.data.clone(),
        };
        drop(event);
        let inner = self.inner.clone();
        block_on_context(async {
            inner.write_event_to_stream(dynamic).await;
        });
        Ok(())
    }

    /// Get the workflow run ID.
    ///
    /// Returns:
    ///     The UUID string for this workflow run.
    fn run_id(&self, _py: Python<'_>) -> PyResult<String> {
        let inner = self.inner.clone();
        let id = block_on_context(async { inner.run_id().await });
        Ok(id.to_string())
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
