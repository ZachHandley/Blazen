//! Python event types and Rust <-> Python event bridging.
//!
//! Python events are dict-like objects backed by JSON. The [`DynamicEvent`]
//! type bridges them into the Rust [`Event`](zagents_events::Event) trait so
//! the workflow engine can route them.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use zagents_events::{AnyEvent, DynamicEvent};

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
    #[pyo3(get)]
    pub event_type: String,
    /// The event data as a JSON object.
    pub data: serde_json::Value,
}

#[pymethods]
impl PyEvent {
    /// Create a new event with the given type and keyword arguments.
    #[new]
    #[pyo3(signature = (event_type, **kwargs))]
    fn new(event_type: String, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        Ok(Self { event_type, data })
    }

    /// Attribute access delegates to the underlying JSON data.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(val) = self.data.get(name) {
            json_to_py(py, val)
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'Event' object has no attribute '{name}'"
            )))
        }
    }

    /// Convert the event data to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.data)
    }

    fn __repr__(&self) -> String {
        format!("Event('{}', {})", self.event_type, self.data)
    }

    fn __str__(&self) -> String {
        self.__repr__()
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
                event_type: "zagents::StartEvent".to_owned(),
                data,
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
                event_type: "zagents::StopEvent".to_owned(),
                data,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a `Box<dyn AnyEvent>` to a [`PyEvent`].
pub fn any_event_to_py_event(event: &dyn AnyEvent) -> PyEvent {
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();

    // If the event is a StartEvent, extract the "data" field.
    if event_type == "zagents::StartEvent" {
        let data = json.get("data").cloned().unwrap_or(serde_json::Value::Null);
        return PyEvent { event_type, data };
    }

    // If the event is a StopEvent, wrap "result" in the data.
    if event_type == "zagents::StopEvent" {
        let result = json
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let mut map = serde_json::Map::new();
        map.insert("result".to_owned(), result);
        return PyEvent {
            event_type,
            data: serde_json::Value::Object(map),
        };
    }

    // If it's a DynamicEvent, extract event_type and data directly.
    if let Some(dynamic) = event.as_any().downcast_ref::<DynamicEvent>() {
        return PyEvent {
            event_type: dynamic.event_type.clone(),
            data: dynamic.data.clone(),
        };
    }

    // Fallback: use the full JSON as data.
    PyEvent {
        event_type,
        data: json,
    }
}

/// Convert a [`PyEvent`] to a `Box<dyn AnyEvent>`.
pub fn py_event_to_any_event(event: &PyEvent) -> Box<dyn AnyEvent> {
    // Check for built-in event types and convert to their concrete Rust types.
    if event.event_type == "zagents::StartEvent" {
        return Box::new(zagents_events::StartEvent {
            data: event.data.clone(),
        });
    }

    if event.event_type == "zagents::StopEvent" {
        let result = event
            .data
            .get("result")
            .cloned()
            .unwrap_or_else(|| event.data.clone());
        return Box::new(zagents_events::StopEvent { result });
    }

    // For all other event types, use DynamicEvent.
    Box::new(DynamicEvent {
        event_type: event.event_type.clone(),
        data: event.data.clone(),
    })
}

// ---------------------------------------------------------------------------
// JSON <-> Python conversion helpers
// ---------------------------------------------------------------------------

/// Convert a Python dict to a `serde_json::Value`.
pub fn dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let py = dict.py();
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let json_val = py_to_json(py, &value)?;
        map.insert(key_str, json_val);
    }
    Ok(serde_json::Value::Object(map))
}

/// Convert a Python object to a `serde_json::Value`.
pub fn py_to_json(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    // None
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }

    // Bool (must check before int, since bool is a subclass of int in Python)
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }

    // Integer
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }

    // Float
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number));
    }

    // String
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }

    // List
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(py_to_json(_py, &item)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    // Dict
    if let Ok(dict) = obj.cast::<PyDict>() {
        return dict_to_json(dict);
    }

    // Fallback: convert to string representation
    let repr = obj.str()?.to_string();
    Ok(serde_json::Value::String(repr))
}

/// Convert a `serde_json::Value` to a Python object.
pub fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            // `bool.into_pyobject` returns a borrowed reference; clone to owned.
            Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = pyo3::types::PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// JsonValue wrapper for IntoPyObject
// ---------------------------------------------------------------------------

/// Newtype wrapper around `serde_json::Value` that implements
/// [`IntoPyObject`] so it can be returned from `future_into_py` closures.
pub struct JsonValue(pub serde_json::Value);

impl<'py> IntoPyObject<'py> for JsonValue {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        let obj = json_to_py(py, &self.0)?;
        Ok(obj.into_bound(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zagents_events::{Event, intern_event_type};

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
        assert_eq!(json["event_type"], "MyEvent");
        assert_eq!(json["data"]["key"], "value");
    }
}
