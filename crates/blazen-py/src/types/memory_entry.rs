//! Python wrapper for [`blazen_memory::types::MemoryEntry`].
//!
//! Provides a typed input struct for `Memory.add_many(...)` so callers can
//! pass `MemoryEntry` objects in addition to bare dicts.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_memory::types::MemoryEntry;

/// A lightweight input record for adding entries to a [`Memory`] store.
///
/// Args:
///     text: The text content to store.
///     id: Optional identifier. If omitted, one is generated when added.
///     metadata: Optional arbitrary user metadata (any JSON-serializable
///         Python value).
///
/// Example:
///     >>> entry = MemoryEntry(text="hello", id="my-id", metadata={"k": "v"})
///     >>> await memory.add_many([entry])
#[gen_stub_pyclass]
#[pyclass(name = "MemoryEntry", from_py_object)]
#[derive(Clone)]
pub struct PyMemoryEntry {
    pub(crate) inner: MemoryEntry,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMemoryEntry {
    /// Construct a memory entry.
    #[new]
    #[pyo3(signature = (*, text, id=None, metadata=None))]
    fn new(
        py: Python<'_>,
        text: String,
        id: Option<String>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let metadata_value = match metadata {
            Some(m) => crate::convert::py_to_json(py, m)?,
            None => serde_json::Value::Null,
        };
        Ok(Self {
            inner: MemoryEntry {
                id: id.unwrap_or_default(),
                text,
                metadata: metadata_value,
            },
        })
    }

    /// The text content stored under this entry.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The entry id. Empty string when one will be generated server-side.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// User metadata associated with this entry.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        if self.inner.id.is_empty() {
            format!("MemoryEntry(text={:?})", self.inner.text)
        } else {
            format!(
                "MemoryEntry(id={:?}, text={:?})",
                self.inner.id, self.inner.text
            )
        }
    }
}

impl From<MemoryEntry> for PyMemoryEntry {
    fn from(inner: MemoryEntry) -> Self {
        Self { inner }
    }
}

impl From<PyMemoryEntry> for MemoryEntry {
    fn from(p: PyMemoryEntry) -> Self {
        p.inner
    }
}
