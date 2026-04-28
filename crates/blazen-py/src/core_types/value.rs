//! Python wrappers for [`blazen_core::value::BytesWrapper`] and
//! [`blazen_core::value::StateValue`].
//!
//! `PyStateValue` is exposed as a complex pyclass enum so the variants
//! map cleanly to Python pattern-matching:
//!
//! ```python
//! match sv:
//!     case StateValue.Json(value=v):
//!         ...
//!     case StateValue.Bytes(data=b):
//!         ...
//!     case StateValue.Native(data=b):
//!         ...
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::value::{BytesWrapper, StateValue};

use crate::convert::{json_to_py, py_to_json};

// ---------------------------------------------------------------------------
// PyBytesWrapper
// ---------------------------------------------------------------------------

/// Wrapper around a `bytes` payload that mirrors
/// [`BytesWrapper`](blazen_core::value::BytesWrapper).
///
/// On the Rust side the wrapper exists to opt into `serde_bytes` for
/// efficient binary-format encoding; on the Python side it is mostly
/// useful as a typed container so callers can disambiguate
/// "JSON-encoded bytes" from "raw bytes" at a class level.
#[gen_stub_pyclass]
#[pyclass(name = "BytesWrapper", from_py_object)]
#[derive(Clone)]
pub struct PyBytesWrapper {
    pub(crate) inner: BytesWrapper,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBytesWrapper {
    /// Construct from a `bytes` value.
    #[new]
    fn new(data: Vec<u8>) -> Self {
        Self {
            inner: BytesWrapper(data),
        }
    }

    /// Return the wrapped payload as `bytes`.
    #[getter]
    fn data(&self) -> Vec<u8> {
        self.inner.0.clone()
    }

    fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    fn __bytes__(&self) -> Vec<u8> {
        self.inner.0.clone()
    }

    fn __repr__(&self) -> String {
        format!("BytesWrapper(len={})", self.inner.0.len())
    }
}

impl From<BytesWrapper> for PyBytesWrapper {
    fn from(inner: BytesWrapper) -> Self {
        Self { inner }
    }
}

impl From<PyBytesWrapper> for BytesWrapper {
    fn from(p: PyBytesWrapper) -> Self {
        p.inner
    }
}

// ---------------------------------------------------------------------------
// PyStateValue
// ---------------------------------------------------------------------------

/// Tagged union mirroring [`StateValue`].
///
/// The Python class exposes a `kind` discriminator (`"json"`, `"bytes"`,
/// `"native"`) plus accessor methods for each variant. Construct
/// instances through the static factories `StateValue.json(value)`,
/// `StateValue.bytes(data)`, and `StateValue.native(data)`.
#[gen_stub_pyclass]
#[pyclass(name = "StateValue", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStateValue {
    pub(crate) inner: StateValue,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStateValue {
    /// Build a JSON-typed `StateValue` from any JSON-compatible Python
    /// value.
    #[staticmethod]
    fn json(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Self> {
        let json = py_to_json(py, &value)?;
        Ok(Self {
            inner: StateValue::Json(json),
        })
    }

    /// Build a `Bytes`-typed `StateValue` from a `bytes` payload.
    #[staticmethod]
    fn bytes(data: Vec<u8>) -> Self {
        Self {
            inner: StateValue::Bytes(BytesWrapper(data)),
        }
    }

    /// Build a `Native`-typed `StateValue` from a binding-serialized
    /// `bytes` payload (Python pickle, etc.).
    #[staticmethod]
    fn native(data: Vec<u8>) -> Self {
        Self {
            inner: StateValue::native(data),
        }
    }

    /// Discriminator: `"json"`, `"bytes"`, or `"native"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            StateValue::Json(_) => "json",
            StateValue::Bytes(_) => "bytes",
            StateValue::Native(_) => "native",
        }
    }

    fn is_json(&self) -> bool {
        self.inner.is_json()
    }

    fn is_bytes(&self) -> bool {
        self.inner.is_bytes()
    }

    fn is_native(&self) -> bool {
        self.inner.is_native()
    }

    /// Return the inner JSON value as a Python object, or raise
    /// `ValueError` if this is not a JSON variant.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn as_json(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            StateValue::Json(v) => json_to_py(py, v),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "StateValue is not a Json variant",
            )),
        }
    }

    /// Return the inner byte payload, or raise `ValueError` if this is
    /// not a `Bytes` variant.
    fn as_bytes(&self) -> PyResult<Vec<u8>> {
        self.inner.as_bytes().map(<[u8]>::to_vec).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("StateValue is not a Bytes variant")
        })
    }

    /// Return the inner native-serialized payload, or raise `ValueError`
    /// if this is not a `Native` variant.
    fn as_native(&self) -> PyResult<Vec<u8>> {
        self.inner.as_native().map(<[u8]>::to_vec).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("StateValue is not a Native variant")
        })
    }

    /// Encode the value as a `dict` matching the serde wire format used
    /// by the core crate (`{"Json": ...}`, `{"Bytes": [...]}`,
    /// `{"Native": [...]}`).
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = serde_json::to_value(&self.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to serialize StateValue: {e}"
            ))
        })?;
        json_to_py(py, &json)
    }

    /// Build a `StateValue` from a serde-shaped `dict`. Inverse of
    /// `to_dict`.
    #[staticmethod]
    fn from_dict(value: &Bound<'_, PyDict>) -> PyResult<Self> {
        let json = crate::convert::dict_to_json(value)?;
        let sv: StateValue = serde_json::from_value(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid StateValue dict: {e}"))
        })?;
        Ok(Self { inner: sv })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            StateValue::Json(_) => "StateValue.Json(...)".to_owned(),
            StateValue::Bytes(b) => format!("StateValue.Bytes(len={})", b.0.len()),
            StateValue::Native(b) => format!("StateValue.Native(len={})", b.0.len()),
        }
    }
}

impl From<StateValue> for PyStateValue {
    fn from(inner: StateValue) -> Self {
        Self { inner }
    }
}

impl From<PyStateValue> for StateValue {
    fn from(p: PyStateValue) -> Self {
        p.inner
    }
}
