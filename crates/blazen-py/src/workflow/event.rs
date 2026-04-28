//! Python event types and Rust <-> Python event bridging.
//!
//! Python events are dict-like objects backed by JSON. The [`DynamicEvent`]
//! type bridges them into the Rust [`Event`](blazen_events::Event) trait so
//! the workflow engine can route them.

use std::sync::Arc;

use blazen_core::session_ref::SessionRefRegistry;
use blazen_events::{AnyEvent, DynamicEvent, EventEnvelope};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

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
#[gen_stub_pyclass]
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

#[gen_stub_pymethods]
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
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
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
#[gen_stub_pyclass]
#[pyclass(name = "StartEvent", extends = PyEvent, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyStartEvent;

#[gen_stub_pymethods]
#[pymethods]
impl PyStartEvent {
    /// Create a new `StartEvent` with keyword data.
    #[gen_stub(skip)]
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
#[gen_stub_pyclass]
#[pyclass(name = "StopEvent", extends = PyEvent, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyStopEvent;

#[gen_stub_pymethods]
#[pymethods]
impl PyStopEvent {
    /// Create a new `StopEvent` with a result value.
    #[gen_stub(skip)]
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

// ---------------------------------------------------------------------------
// PyDynamicEvent
// ---------------------------------------------------------------------------

/// A type-erased event carrying its type name and JSON payload.
///
/// Mirrors [`blazen_events::DynamicEvent`]. Used to transport events
/// defined in foreign language bindings through the Rust workflow engine.
///
/// Python usage:
/// ```python
/// ev = DynamicEvent("MyEvent", key="value", count=3)
/// print(ev.event_type)   # "MyEvent"
/// print(ev.to_dict())    # {"key": "value", "count": 3}
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "DynamicEvent", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDynamicEvent {
    pub(crate) inner: DynamicEvent,
    pub(crate) session_refs: Option<Arc<SessionRefRegistry>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDynamicEvent {
    #[gen_stub(skip)]
    #[new]
    #[pyo3(signature = (event_type, **kwargs))]
    fn new(event_type: String, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        Ok(Self {
            inner: DynamicEvent { event_type, data },
            session_refs: current_session_registry(),
        })
    }

    #[getter]
    fn event_type(&self) -> &str {
        &self.inner.event_type
    }

    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        with_event_registry(self.session_refs.as_ref(), || {
            json_to_py(py, &self.inner.data)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicEvent('{}', {})",
            self.inner.event_type, self.inner.data
        )
    }
}

// ---------------------------------------------------------------------------
// PyEventEnvelope
// ---------------------------------------------------------------------------

/// Wraps an event with metadata for the internal queue.
///
/// Mirrors [`blazen_events::EventEnvelope`]. Carries the typed event plus
/// the optional source step name that produced it.
///
/// Python usage:
/// ```python
/// envelope = EventEnvelope(my_event, source_step="step_a")
/// print(envelope.event_type)   # the inner event's type
/// print(envelope.source_step)  # "step_a"
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "EventEnvelope", frozen)]
pub struct PyEventEnvelope {
    pub(crate) event: PyEvent,
    pub(crate) source_step: Option<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEventEnvelope {
    #[new]
    #[pyo3(signature = (event, source_step=None))]
    fn new(event: PyEvent, source_step: Option<String>) -> Self {
        Self { event, source_step }
    }

    #[getter]
    fn event_type(&self) -> &str {
        &self.event.event_type
    }

    #[getter]
    fn source_step(&self) -> Option<String> {
        self.source_step.clone()
    }

    #[getter]
    fn event(&self) -> PyEvent {
        self.event.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EventEnvelope(event_type='{}', source_step={:?})",
            self.event.event_type, self.source_step
        )
    }
}

/// Build a [`blazen_events::EventEnvelope`] from a [`PyEventEnvelope`].
pub fn py_envelope_to_envelope(env: &PyEventEnvelope) -> EventEnvelope {
    EventEnvelope::new(py_event_to_any_event(&env.event), env.source_step.clone())
}

// ---------------------------------------------------------------------------
// PyInputRequestEvent / PyInputResponseEvent
// ---------------------------------------------------------------------------

/// Emitted by a step to request human input. Triggers auto-pause.
///
/// Python usage:
/// ```python
/// ev = InputRequestEvent("req-1", prompt="What is your name?")
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "InputRequestEvent", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyInputRequestEvent {
    #[pyo3(get, set)]
    pub request_id: String,
    #[pyo3(get, set)]
    pub prompt: String,
    pub metadata: serde_json::Value,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInputRequestEvent {
    #[new]
    #[pyo3(signature = (request_id, prompt=None, metadata=None))]
    fn new(
        request_id: String,
        prompt: Option<String>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let metadata_json = if let Some(m) = metadata {
            dict_to_json(m)?
        } else {
            serde_json::Value::Null
        };
        Ok(Self {
            request_id,
            prompt: prompt.unwrap_or_default(),
            metadata: metadata_json,
        })
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "InputRequestEvent(request_id='{}', prompt='{}', metadata={})",
            self.request_id, self.prompt, self.metadata
        )
    }
}

/// The human's response, injected on resume.
///
/// Python usage:
/// ```python
/// ev = InputResponseEvent("req-1", {"answer": "Alice"})
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "InputResponseEvent", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyInputResponseEvent {
    #[pyo3(get, set)]
    pub request_id: String,
    pub response: serde_json::Value,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInputResponseEvent {
    #[new]
    #[pyo3(signature = (request_id, response))]
    fn new(request_id: String, response: &Bound<'_, PyAny>) -> PyResult<Self> {
        let response_json = crate::convert::py_to_json(response.py(), response)?;
        Ok(Self {
            request_id,
            response: response_json,
        })
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn response(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.response)
    }

    fn __repr__(&self) -> String {
        format!(
            "InputResponseEvent(request_id='{}', response={})",
            self.request_id, self.response
        )
    }
}

impl PyInputResponseEvent {
    /// Build a [`blazen_events::InputResponseEvent`] from this Python wrapper.
    pub fn to_rust(&self) -> blazen_events::InputResponseEvent {
        blazen_events::InputResponseEvent {
            request_id: self.request_id.clone(),
            response: self.response.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level free functions: deserializer registry / type interning
// ---------------------------------------------------------------------------

/// Global registry of Python deserializer callables, keyed by event type.
///
/// `register_event_deserializer` (the Rust API) takes a `fn` pointer that
/// cannot capture state, so we hold the user-supplied Python callable here
/// and dispatch through a single fixed `fn` that looks it up by event type.
static PY_DESERIALIZER_REGISTRY: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, Py<PyAny>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Fixed Rust function pointer registered with `blazen_events`.
///
/// Looks up the originating event type via a thread-local, calls the matching
/// Python callable with the JSON payload, expects a [`PyEvent`] back, and
/// boxes it as a [`DynamicEvent`].
fn dispatch_python_deserializer(value: serde_json::Value) -> Option<Box<dyn AnyEvent>> {
    let event_type = PY_DISPATCH_EVENT_TYPE.with(|cell| cell.borrow().clone())?;
    Python::attach(|py| {
        let callable = {
            let map = PY_DESERIALIZER_REGISTRY.lock().ok()?;
            map.get(&event_type)?.clone_ref(py)
        };
        let json_obj = json_to_py(py, &value).ok()?;
        let bound = json_obj.bind(py);
        let result = callable.call1(py, (bound,)).ok()?;
        let py_event: PyEvent = result.extract(py).ok()?;
        Some(Box::new(DynamicEvent {
            event_type: py_event.event_type,
            data: py_event.data,
        }) as Box<dyn AnyEvent>)
    })
}

thread_local! {
    /// Active event-type during a `try_deserialize_event` call so the fixed
    /// `fn` dispatcher can look up the right Python callable.
    static PY_DISPATCH_EVENT_TYPE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Register a Python callable as the deserializer for the given event type.
///
/// The callable receives a Python dict (the JSON payload) and must return
/// a [`PyEvent`]. The resulting event is wrapped as a [`DynamicEvent`] for
/// transport through the workflow engine.
///
/// Python usage:
/// ```python
/// def make_my_event(data: dict) -> Event:
///     return Event("MyEvent", **data)
///
/// register_event_deserializer("MyEvent", make_my_event)
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, deserializer))]
pub fn register_event_deserializer(name: String, deserializer: Py<PyAny>) -> PyResult<()> {
    let interned: &'static str = blazen_events::intern_event_type(&name);
    {
        let mut map = PY_DESERIALIZER_REGISTRY
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("registry poisoned"))?;
        map.insert(name, deserializer);
    }
    blazen_events::register_event_deserializer(interned, dispatch_python_deserializer);
    Ok(())
}

/// Attempt to deserialize a JSON string into a [`PyEvent`] using the registry.
///
/// Returns `None` if no deserializer is registered for `name` or if the JSON
/// cannot be parsed.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, json_str))]
pub fn try_deserialize_event(name: String, json_str: String) -> PyResult<Option<PyEvent>> {
    let value: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid JSON: {e}")))?;
    let result = PY_DISPATCH_EVENT_TYPE.with(|cell| {
        *cell.borrow_mut() = Some(name.clone());
        let out = blazen_events::try_deserialize_event(&name, &value);
        *cell.borrow_mut() = None;
        out
    });
    Ok(result.map(|boxed| any_event_to_py_event(&*boxed)))
}

/// Intern a dynamic event-type name so it can be referenced as a stable
/// identifier. Returns the interned string (always equal to `name`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name))]
pub fn intern_event_type(name: String) -> String {
    blazen_events::intern_event_type(&name).to_owned()
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
