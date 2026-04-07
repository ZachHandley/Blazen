//! Unified Python <-> JSON conversion helpers.
//!
//! This module consolidates all `serde_json::Value` <-> Python object
//! conversion logic into a single place, replacing the duplicate
//! implementations that previously lived in `workflow::event` and
//! `types::memory`.
//!
//! ## Auto-routing through the session-ref registry
//!
//! When [`py_to_json`] encounters a Python object that cannot be
//! represented as JSON (e.g., a dataclass, a Pydantic model, a DB
//! connection, a lambda) **and** the call originates from inside a
//! workflow step, the value is auto-routed into the per-`Context`
//! [`SessionRefRegistry`](blazen_core::session_ref::SessionRefRegistry).
//! The JSON payload carries only a marker:
//!
//! ```json
//! {"__blazen_session_ref__": "<uuid>"}
//! ```
//!
//! [`json_to_py`] detects this marker, looks the UUID up in the
//! currently-active registry (installed via
//! [`crate::workflow::session_ref::CURRENT_SESSION_REGISTRY`]), and returns
//! the **same** Python object that was originally inserted. Identity
//! (`is`) is preserved within a single workflow run.
//!
//! Outside an active step, [`py_to_json`] raises a clear `PyTypeError`
//! instead of silently stringifying the value.
//!
//! ## Pickle round-trip (legacy reader)
//!
//! Older snapshots may contain pickle-tagged objects of the form
//! `{"__blazen_pickled__": "<b64>"}`. [`json_to_py`] still recognises
//! this legacy tag and unpickles transparently for backward compat.
//! Nothing in the current code path *writes* this tag through
//! [`py_to_json`] anymore — the writer side is exercised only by
//! [`crate::workflow::context::PyContext::set`]'s Tier-3 fallback,
//! which stores pickled bytes as
//! [`StateValue::Native`](blazen_core::StateValue::Native) (not as a
//! tagged JSON object).

use std::any::Any;
use std::sync::Arc;

use base64::Engine as _;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::Serialize;

use blazen_core::RegistryKey;
use blazen_core::session_ref::SESSION_REF_TAG;

use crate::workflow::session_ref::current_session_registry;

/// The JSON key used to tag legacy pickle-serialized Python objects.
/// Only the *reader* side is exercised today; nothing writes this tag
/// through [`py_to_json`] anymore.
const PICKLE_TAG: &str = "__blazen_pickled__";

/// Run a future to completion, handling both inside-tokio and outside-tokio
/// contexts. Uses `block_in_place` when called from a tokio worker thread
/// (e.g. from within a step handler), and falls back to the
/// `pyo3-async-runtimes` runtime otherwise.
pub(crate) fn block_on_context<F: std::future::Future>(fut: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // We're inside a tokio runtime -- use block_in_place to avoid panics
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        // No tokio runtime on this thread -- use the pyo3 runtime
        pyo3_async_runtimes::tokio::get_runtime().block_on(fut)
    }
}

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
/// `tuple` natively. For any other type, the value is **auto-routed**
/// into the active session-ref registry (when called from inside a
/// workflow step) and returned as a marker JSON object:
///
/// ```json
/// {"__blazen_session_ref__": "<uuid>"}
/// ```
///
/// Identity is preserved: a subsequent [`json_to_py`] call resolving the
/// same marker returns the *same* Python object via
/// `Arc<Py<PyAny>>::clone_ref`. There is no longer any silent pickling
/// or stringification — values that escape this function intact are
/// either pure JSON or live registry handles.
///
/// # Errors
///
/// Returns `PyTypeError` if a non-JSON value is encountered while no
/// session registry is installed (i.e. the call originates outside an
/// active workflow step). The fix in that case is to construct the
/// event inside a `@step`-decorated function or convert the value to a
/// JSON-serializable form first.
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

    // Non-JSON value. Try the active session-ref registry first.
    if let Some(reg) = current_session_registry() {
        // Wrap the live Py<PyAny> in an Arc and store in the registry.
        // Py<PyAny> is Send + Sync + 'static so the trait coercion to
        // Arc<dyn Any + Send + Sync> is direct.
        let live: Arc<dyn Any + Send + Sync> = Arc::new(obj.clone().unbind());
        let key = block_on_context(async move { reg.insert_arc(live).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        return Ok(serde_json::json!({ SESSION_REF_TAG: key.to_string() }));
    }

    // Outside a workflow step → loud error. Never silently stringify.
    let type_name = obj
        .get_type()
        .name()
        .map_or_else(|_| "<unknown>".to_owned(), |n| n.to_string());
    let _ = py;
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "cannot serialize `{type_name}` to JSON outside an active workflow step. \
         Construct the event inside a `@step`-decorated function so non-JSON values \
         can be auto-routed to the session registry, or convert the value to a \
         JSON-serializable form first."
    )))
}

// ---------------------------------------------------------------------------
// JSON -> Python
// ---------------------------------------------------------------------------

/// Convert a [`serde_json::Value`] to a Python object.
///
/// Recognises two tagged envelopes inside single-key JSON objects:
///
/// 1. `{"__blazen_session_ref__": "<uuid>"}` — looks the UUID up in the
///    currently-active [`SessionRefRegistry`](blazen_core::session_ref::SessionRefRegistry)
///    and returns the original `Py<PyAny>` (identity-preserving). Raises
///    `RuntimeError` if no registry is installed or the entry has been
///    dropped (e.g. after cross-process resume).
/// 2. `{"__blazen_pickled__": "<b64>"}` — legacy reader for older
///    snapshots. Decodes the base64 payload and calls `pickle.loads`.
///
/// All other JSON shapes are converted to their natural Python
/// equivalents.
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
            // Check for tagged envelopes BEFORE building a dict.
            if map.len() == 1 {
                // Session-ref marker (preferred path).
                if let Some(serde_json::Value::String(uuid_str)) = map.get(SESSION_REF_TAG) {
                    let key = RegistryKey::parse(uuid_str).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "malformed session ref uuid `{uuid_str}`: {e}"
                        ))
                    })?;
                    let Some(reg) = current_session_registry() else {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "session ref `{uuid_str}` cannot be resolved: no registry is \
                             installed (this value can only be read from inside an active \
                             workflow step or the workflow handler that produced it)"
                        )));
                    };
                    let arc_any = block_on_context(async move { reg.get_any(key).await });
                    let Some(arc_any) = arc_any else {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "session ref `{uuid_str}` no longer available (the workflow may \
                             have been resumed in a different process, or the registry was \
                             cleared)"
                        )));
                    };
                    let py_arc: Arc<Py<PyAny>> =
                        Arc::downcast::<Py<PyAny>>(arc_any).map_err(|_| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "session ref `{uuid_str}` has unexpected runtime type \
                                 (expected Python object, got cross-runtime entry)"
                            ))
                        })?;
                    return Ok(py_arc.clone_ref(py));
                }
                // Legacy pickle tag (back-compat reader only).
                if let Some(serde_json::Value::String(b64)) = map.get(PICKLE_TAG)
                    && let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64)
                {
                    let pickle = py.import("pickle")?;
                    let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                    let obj = pickle.call_method1("loads", (py_bytes,))?;
                    return Ok(obj.unbind());
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
