//! Unified Python <-> JSON conversion helpers.
//!
//! This module consolidates all `serde_json::Value` <-> Python object
//! conversion logic into a single place, replacing the duplicate
//! implementations that previously lived in `workflow::event` and
//! `types::memory`.
//!
//! ## Pickle round-trip
//!
//! When [`py_to_json`] encounters a Python object that cannot be
//! represented as JSON (e.g., a dataclass, a set, a custom class), it
//! pickles the object and stores it as a tagged JSON object:
//!
//! ```json
//! {"__blazen_pickled__": "<base64-encoded pickle bytes>"}
//! ```
//!
//! [`json_to_py`] detects this tag and unpickles the object back,
//! providing lossless round-trip for arbitrary Python values.

use base64::Engine as _;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::Serialize;

/// The JSON key used to tag pickle-serialized Python objects.
const PICKLE_TAG: &str = "__blazen_pickled__";

// ---------------------------------------------------------------------------
// Python -> JSON
// ---------------------------------------------------------------------------

/// Convert a Python dict to a [`serde_json::Value::Object`].
///
/// Each value is converted via [`py_to_json`], which falls back to
/// pickle for non-JSON-serializable types.
pub(crate) fn dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let py = dict.py();
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.str()?.extract()?;
        let json_val = py_to_json(py, &value)?;
        map.insert(key_str, json_val);
    }
    Ok(serde_json::Value::Object(map))
}

/// Strict dict conversion: returns `Err` if any value is not
/// JSON-serializable (no pickle/`__str__` fallback).
pub(crate) fn try_dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let py = dict.py();
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.str()?.extract()?;
        let json_val = try_py_to_json(py, &value)?;
        map.insert(key_str, json_val);
    }
    Ok(serde_json::Value::Object(map))
}

/// Strict conversion: returns `Err` for types that cannot be directly
/// represented as JSON (no pickle/`__str__` fallback). Used to detect
/// when pickle is needed.
pub(crate) fn try_py_to_json(
    _py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let arr: PyResult<Vec<_>> = list.iter().map(|item| try_py_to_json(_py, &item)).collect();
        return Ok(serde_json::Value::Array(arr?));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return try_dict_to_json(dict);
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let arr: PyResult<Vec<_>> = tuple
            .iter()
            .map(|item| try_py_to_json(_py, &item))
            .collect();
        return Ok(serde_json::Value::Array(arr?));
    }
    let type_name = obj
        .get_type()
        .name()
        .map_or_else(|_| "<unknown>".to_owned(), |n| n.to_string());
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "not JSON-serializable: {type_name}"
    )))
}

/// Convert a Python object to a [`serde_json::Value`].
///
/// Handles `None`, `bool`, `int`, `float`, `str`, `list`, `dict`, and
/// `tuple` natively. For any other type, falls back to:
///
/// 1. **Pickle + base64** -- produces `{"__blazen_pickled__": "<b64>"}`.
///    This is lossless and allows [`json_to_py`] to reconstruct the
///    original object.
/// 2. **`__str__`** -- absolute last resort if pickle itself fails
///    (should be extremely rare).
pub(crate) fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
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
    if let Ok(list) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(py_to_json(py, &item)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    // Dict
    if let Ok(dict) = obj.cast::<PyDict>() {
        return dict_to_json(dict);
    }

    // Tuple
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut arr = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            arr.push(py_to_json(py, &item)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    // Fallback: pickle -> base64 tagged JSON object
    let pickle = py.import("pickle")?;
    if let Ok(pickled) = pickle.call_method1("dumps", (obj,)) {
        let bytes: Vec<u8> = pickled.extract()?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        Ok(serde_json::json!({PICKLE_TAG: b64}))
    } else {
        // Truly unserializable -- still use __str__ as absolute last resort
        let repr = obj.str()?.to_string();
        Ok(serde_json::Value::String(repr))
    }
}

// ---------------------------------------------------------------------------
// JSON -> Python
// ---------------------------------------------------------------------------

/// Convert a [`serde_json::Value`] to a Python object.
///
/// Recognizes the `{"__blazen_pickled__": "<b64>"}` tag produced by
/// [`py_to_json`] and unpickles the original Python object.
pub(crate) fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
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
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            // Check for pickled object tag BEFORE building a dict
            if map.len() == 1 {
                if let Some(serde_json::Value::String(b64)) = map.get(PICKLE_TAG) {
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                        let pickle = py.import("pickle")?;
                        let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                        let obj = pickle.call_method1("loads", (py_bytes,))?;
                        return Ok(obj.unbind());
                    }
                }
            }
            // Normal dict conversion
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// JsonValue newtype (for IntoPyObject)
// ---------------------------------------------------------------------------

/// Newtype wrapper around [`serde_json::Value`] that implements
/// [`IntoPyObject`] so it can be returned from `future_into_py` closures.
pub(crate) struct JsonValue(pub serde_json::Value);

impl<'py> IntoPyObject<'py> for JsonValue {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        let obj = json_to_py(py, &self.0)?;
        Ok(obj.into_bound(py))
    }
}

// ---------------------------------------------------------------------------
// Convenience: Rust Serialize -> Python
// ---------------------------------------------------------------------------

/// Serialize any `T: Serialize` to a `serde_json::Value` and then convert
/// to a Python object. Useful for returning Rust structs to Python without
/// a dedicated `#[pyclass]`.
#[allow(dead_code)]
pub(crate) fn rust_to_py<T: Serialize>(py: Python<'_>, val: &T) -> PyResult<Py<PyAny>> {
    let json_val = serde_json::to_value(val).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("JSON serialization failed: {e}"))
    })?;
    json_to_py(py, &json_val)
}
