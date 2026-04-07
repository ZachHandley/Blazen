//! Python event types and Rust <-> Python event bridging.
//!
//! Python events are dict-like objects backed by JSON. The [`DynamicEvent`]
//! type bridges them into the Rust [`Event`](blazen_events::Event) trait so
//! the workflow engine can route them.

use std::sync::Arc;

use blazen_core::session_ref::SessionRefRegistry;
use blazen_events::{AnyEvent, DynamicEvent};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{dict_to_json, json_to_py};
use crate::workflow::session_ref::{CURRENT_SESSION_REGISTRY, current_session_registry};

/// Run `f` with the given session registry installed as the current
/// `tokio::task_local!` (synchronously). Used by [`PyEvent::__getattr__`]
/// and [`PyEvent::to_dict`] so a [`json_to_py`] call inside `f` can
/// resolve `__blazen_session_ref__` markers carried by event payloads.
fn with_event_registry<R>(reg: Option<&Arc<SessionRefRegistry>>, f: impl FnOnce() -> R) -> R {
    if let Some(reg) = reg {
        CURRENT_SESSION_REGISTRY.sync_scope(Arc::clone(reg), f)
    } else {
        f()
    }
}

// ---------------------------------------------------------------------------
// PyEvent -- Python side
// ---------------------------------------------------------------------------

/// A dict-like event object exposed to Python.
///
/// Python usage:
/// ```python
/// ev = Event("AnalyzeEvent", text="hello", score=0.9)
/// print(ev.text)       # "hello"
/// print(ev.to_dict())  # {"text": "hello", "score": 0.9}
/// ```
#[pyclass(name = "Event", subclass, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyEvent {
    /// The event type name (e.g. `"AnalyzeEvent"`).
    #[pyo3(get, set)]
    pub event_type: String,
    /// The event data as a JSON object.
    pub data: serde_json::Value,
    /// Active session-ref registry, captured at construction time so
    /// that attribute access can resolve `__blazen_session_ref__`
    /// markers carried by `data` even after the workflow's `Context`
    /// has been dropped (e.g. when the user reads `result.result`
    /// after `await handler.result()` returns).
    pub(crate) session_refs: Option<Arc<SessionRefRegistry>>,
}

#[pymethods]
impl PyEvent {
    /// Create a new event.
    ///
    /// When called on the base `Event` class, `event_type` is required:
    ///     `Event("MyEvent", key=value)`
    ///
    /// When called on a subclass, `event_type` is auto-inferred from the
    /// class name:
    ///     `class MyEvent(Event): ...`
    ///     `MyEvent(key=value)`  # event_type == "MyEvent"
    /// Create a new event.
    ///
    /// - Base `Event` class: `Event("MyEvent", key=value)`
    /// - Subclasses: `MyEvent(key=value)` — event_type auto-set by `__init__`
    #[new]
    #[pyo3(signature = (event_type=None, **kwargs))]
    fn new(event_type: Option<String>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // When event_type is None, use a placeholder. The __init__ generated
        // by __init_subclass__ will set the real event_type on the instance.
        // If no subclass __init__ runs (bare Event()), we reject it below.
        let event_type = event_type.unwrap_or_default();

        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        // Capture the active registry (if any) so subsequent attribute
        // reads can resolve session-ref markers we just inserted.
        Ok(Self {
            event_type,
            data,
            session_refs: current_session_registry(),
        })
    }

    /// Attribute access delegates to the underlying JSON data.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(val) = self.data.get(name) {
            with_event_registry(self.session_refs.as_ref(), || json_to_py(py, val))
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'Event' object has no attribute '{name}'"
            )))
        }
    }

    /// Convert the event data to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        with_event_registry(self.session_refs.as_ref(), || json_to_py(py, &self.data))
    }

    fn __repr__(&self) -> String {
        format!("Event('{}', {})", self.event_type, self.data)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Called when a Python subclass of Event is defined.
    ///
    /// Auto-generates an `__init__` that sets `event_type` to the class name,
    /// enabling:
    /// ```python
    /// class GreetEvent(Event):
    ///     name: str
    ///     style: str
    ///
    /// ev = GreetEvent(name="Alice", style="formal")
    /// # ev.event_type == "GreetEvent"
    /// ```
    #[classmethod]
    #[pyo3(signature = (**_kwargs))]
    fn __init_subclass__(
        cls: &Bound<'_, pyo3::types::PyType>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = cls.py();
        let class_name: String = cls.getattr("__name__")?.extract()?;

        // Skip StartEvent and StopEvent — they have their own Rust-side __new__
        if class_name == "StartEvent" || class_name == "StopEvent" {
            return Ok(());
        }

        // Generate an __init__ that sets event_type to the class name.
        // Flow: __new__ creates PyEvent with event_type="", then __init__
        // patches it to the real class name.
        let code = format!("def __init__(self, **kwargs):\n    self.event_type = '{class_name}'\n");
        let globals = PyDict::new(py);
        let locals = PyDict::new(py);
        py.run(
            &std::ffi::CString::new(code).unwrap(),
            Some(&globals),
            Some(&locals),
        )?;
        let init_fn = locals.get_item("__init__")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to generate __init__ for Event subclass",
            )
        })?;
        cls.setattr("__init__", init_fn)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StartEvent / StopEvent convenience classes
// ---------------------------------------------------------------------------

/// A start event that kicks off a workflow.
///
/// Python usage:
/// ```python
/// ev = StartEvent(text="hello", count=5)
/// ```
#[pyclass(name = "StartEvent", extends = PyEvent, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyStartEvent;

#[pymethods]
impl PyStartEvent {
    /// Create a new `StartEvent` with keyword data.
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyEvent)> {
        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        Ok((
            Self,
            PyEvent {
                event_type: "blazen::StartEvent".to_owned(),
                data,
                session_refs: current_session_registry(),
            },
        ))
    }
}

/// A stop event that terminates a workflow with a result.
///
/// Python usage:
/// ```python
/// ev = StopEvent(result={"answer": 42})
/// ```
#[pyclass(name = "StopEvent", extends = PyEvent, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyStopEvent;

#[pymethods]
impl PyStopEvent {
    /// Create a new `StopEvent` with a result value.
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyEvent)> {
        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        Ok((
            Self,
            PyEvent {
                event_type: "blazen::StopEvent".to_owned(),
                data,
                session_refs: current_session_registry(),
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a `Box<dyn AnyEvent>` to a [`PyEvent`].
///
/// The returned `PyEvent` captures whichever session-ref registry is
/// active at call time (via [`current_session_registry`]) so that
/// downstream attribute access can resolve `__blazen_session_ref__`
/// markers carried by the event payload.
pub fn any_event_to_py_event(event: &dyn AnyEvent) -> PyEvent {
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();
    let session_refs = current_session_registry();

    // If the event is a StartEvent, extract the "data" field.
    if event_type == "blazen::StartEvent" {
        let data = json.get("data").cloned().unwrap_or(serde_json::Value::Null);
        return PyEvent {
            event_type,
            data,
            session_refs,
        };
    }

    // If the event is a StopEvent, wrap "result" in the data.
    if event_type == "blazen::StopEvent" {
        let result = json
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let mut map = serde_json::Map::new();
        map.insert("result".to_owned(), result);
        return PyEvent {
            event_type,
            data: serde_json::Value::Object(map),
            session_refs,
        };
    }

    // If it's a DynamicEvent, extract event_type and data directly.
    if let Some(dynamic) = event.as_any().downcast_ref::<DynamicEvent>() {
        return PyEvent {
            event_type: dynamic.event_type.clone(),
            data: dynamic.data.clone(),
            session_refs,
        };
    }

    // Fallback: use the full JSON as data.
    PyEvent {
        event_type,
        data: json,
        session_refs,
    }
}

/// Convert a [`PyEvent`] to a `Box<dyn AnyEvent>`.
pub fn py_event_to_any_event(event: &PyEvent) -> Box<dyn AnyEvent> {
    // Check for built-in event types and convert to their concrete Rust types.
    if event.event_type == "blazen::StartEvent" {
        return Box::new(blazen_events::StartEvent {
            data: event.data.clone(),
        });
    }

    if event.event_type == "blazen::StopEvent" {
        let result = event
            .data
            .get("result")
            .cloned()
            .unwrap_or_else(|| event.data.clone());
        return Box::new(blazen_events::StopEvent { result });
    }

    // For all other event types, use DynamicEvent.
    Box::new(DynamicEvent {
        event_type: event.event_type.clone(),
        data: event.data.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_events::{Event, intern_event_type};

    #[test]
    fn intern_event_type_returns_same_pointer() {
        let a = intern_event_type("TestEventPy");
        let b = intern_event_type("TestEventPy");
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn dynamic_event_roundtrip() {
        let evt = DynamicEvent {
            event_type: "MyEvent".to_owned(),
            data: serde_json::json!({"key": "value"}),
        };
        let json = Event::to_json(&evt);
        // DynamicEvent::to_json() now returns the flat data directly.
        assert_eq!(json["key"], "value");
    }
}
