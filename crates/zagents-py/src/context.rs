//! Python wrapper for the workflow [`Context`](zagents_core::Context).
//!
//! Python values are serialized to JSON before storage in the Rust context,
//! and deserialized back when retrieved. This means all Python context values
//! must be JSON-serializable.

use pyo3::prelude::*;

use crate::event::{JsonValue, PyEvent};

/// Shared workflow context accessible by all steps.
///
/// Provides typed key/value storage, event emission, and stream publishing.
/// All values are stored as JSON internally, so they must be JSON-serializable.
///
/// Example:
///     >>> async def my_step(ctx: Context, ev: Event) -> Event:
///     ...     await ctx.set("counter", 42)
///     ...     val = await ctx.get("counter")  # returns 42
///     ...     await ctx.send_event(Event("NextStep", data="hello"))
#[pyclass(name = "Context", from_py_object)]
#[derive(Clone)]
pub struct PyContext {
    pub(crate) inner: zagents_core::Context,
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
    fn set<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let json_val = crate::event::py_to_json(py, value)?;
        let inner = self.inner.clone();
        let key = key.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.set(&key, json_val).await;
            Ok(())
        })
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
    fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let key = key.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let val: Option<serde_json::Value> = inner.get(&key).await;
            match val {
                Some(v) => {
                    // Convert JSON to a Python-compatible wrapper
                    Ok(JsonValue(v))
                }
                None => Ok(JsonValue(serde_json::Value::Null)),
            }
        })
    }

    /// Emit an event into the internal routing queue.
    ///
    /// The event will be routed to any step whose `accepts` list includes
    /// its event type.
    ///
    /// Args:
    ///     event: The event to send.
    fn send_event<'py>(
        &self,
        py: Python<'py>,
        event: PyRef<'py, PyEvent>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dynamic = zagents_events::DynamicEvent {
            event_type: event.event_type.clone(),
            data: event.data.clone(),
        };
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_event(dynamic).await;
            Ok(())
        })
    }

    /// Publish an event to the external broadcast stream.
    ///
    /// Consumers that subscribed via `WorkflowHandler.stream_events()` will
    /// receive this event. Unlike `send_event`, this does NOT route the
    /// event through the internal step registry.
    ///
    /// Args:
    ///     event: The event to publish to the stream.
    fn write_event_to_stream<'py>(
        &self,
        py: Python<'py>,
        event: PyRef<'py, PyEvent>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dynamic = zagents_events::DynamicEvent {
            event_type: event.event_type.clone(),
            data: event.data.clone(),
        };
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.write_event_to_stream(dynamic).await;
            Ok(())
        })
    }

    /// Get the workflow run ID.
    ///
    /// Returns:
    ///     The UUID string for this workflow run.
    fn run_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = inner.run_id().await;
            Ok(id.to_string())
        })
    }

    fn __repr__(&self) -> String {
        "Context()".to_owned()
    }
}

impl PyContext {
    /// Create a new `PyContext` wrapping a Rust `Context`.
    pub fn new(inner: zagents_core::Context) -> Self {
        Self { inner }
    }
}
