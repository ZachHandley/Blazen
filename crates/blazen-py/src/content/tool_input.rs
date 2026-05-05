//! Module-level Python helpers wrapping the content tool-input schema
//! builders from [`blazen_llm::content::tool_input`].
//!
//! Each builder returns a Python dict (the JSON Schema fragment) so it can
//! be plugged directly into a `ToolDefinition`'s parameter schema.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use blazen_llm::content::tool_input as rust_tool_input;

use crate::convert::{dict_to_json, json_to_py};

use super::kind::PyContentKind;

/// Convert a `serde_json::Value` to a Python object, surfacing any
/// PyResult error.
fn json_to_py_any(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    json_to_py(py, val)
}

/// Build a JSON Schema property fragment for a content-reference input.
///
/// Returns a dict of the form
/// ``{"type": "string", "description": ..., "x-blazen-content-ref": {"kind": ...}}``
/// ready to be embedded inside an object-typed schema's `properties` map.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (kind, description))]
pub fn content_ref_property(
    py: Python<'_>,
    kind: PyContentKind,
    description: String,
) -> PyResult<Py<PyAny>> {
    let value = rust_tool_input::content_ref_property(kind.into(), description);
    json_to_py_any(py, &value)
}

/// Build a complete object-typed JSON Schema declaring a single required
/// content-reference input plus optional companion fields.
///
/// `extra_properties` is an optional dict of additional schema properties
/// (already JSON-Schema-shaped) to merge alongside the content reference.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, kind, description, *, extra_properties=None))]
pub fn content_ref_required_object(
    py: Python<'_>,
    name: String,
    kind: PyContentKind,
    description: String,
    extra_properties: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let extras = if let Some(d) = extra_properties {
        let json = dict_to_json(&d)?;
        match json {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(PyValueError::new_err("extra_properties must be a dict"));
            }
        }
    } else {
        serde_json::Map::new()
    };
    let value =
        rust_tool_input::content_ref_required_object(name, kind.into(), description, extras);
    json_to_py_any(py, &value)
}

/// Schema declaring a single required image input. Sugar over
/// [`content_ref_required_object`] with `kind=ContentKind.Image`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn image_input(py: Python<'_>, name: String, description: String) -> PyResult<Py<PyAny>> {
    json_to_py_any(py, &rust_tool_input::image_input(name, description))
}

/// Schema declaring a single required audio input.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn audio_input(py: Python<'_>, name: String, description: String) -> PyResult<Py<PyAny>> {
    json_to_py_any(py, &rust_tool_input::audio_input(name, description))
}

/// Schema declaring a single required video input.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn video_input(py: Python<'_>, name: String, description: String) -> PyResult<Py<PyAny>> {
    json_to_py_any(py, &rust_tool_input::video_input(name, description))
}

/// Schema declaring a single required generic-file / document input.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn file_input(py: Python<'_>, name: String, description: String) -> PyResult<Py<PyAny>> {
    json_to_py_any(py, &rust_tool_input::file_input(name, description))
}

/// Schema declaring a single required 3D-model input.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn three_d_input(py: Python<'_>, name: String, description: String) -> PyResult<Py<PyAny>> {
    json_to_py_any(py, &rust_tool_input::three_d_input(name, description))
}

/// Schema declaring a single required CAD-file input.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn cad_input(py: Python<'_>, name: String, description: String) -> PyResult<Py<PyAny>> {
    json_to_py_any(py, &rust_tool_input::cad_input(name, description))
}
