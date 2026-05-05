//! Python wrapper for [`blazen_llm::content::ContentHandle`].

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::content::ContentHandle;

use super::kind::PyContentKind;

/// Stable reference to content registered with a
/// [`ContentStore`](super::store::PyContentStore).
///
/// `id` is opaque (store-defined). The other fields are metadata captured
/// at `put` time and can be used for routing without dereferencing.
///
/// Example:
///     >>> handle = await store.put(b"...", kind=ContentKind.Image)
///     >>> handle.id
///     'blazen_a1b2c3d4...'
///     >>> handle.kind
///     ContentKind.Image
#[gen_stub_pyclass]
#[pyclass(name = "ContentHandle", from_py_object)]
#[derive(Clone)]
pub struct PyContentHandle {
    pub(crate) inner: ContentHandle,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyContentHandle {
    /// Construct a handle directly. Most callers obtain handles from
    /// [`ContentStore.put`](super::store::PyContentStore::put) instead.
    #[new]
    #[pyo3(signature = (id, kind, *, mime_type=None, byte_size=None, display_name=None))]
    fn new(
        id: String,
        kind: PyContentKind,
        mime_type: Option<String>,
        byte_size: Option<u64>,
        display_name: Option<String>,
    ) -> Self {
        Self {
            inner: ContentHandle {
                id,
                kind: kind.into(),
                mime_type,
                byte_size,
                display_name,
            },
        }
    }

    /// Opaque, stable identifier. Format is store-defined; treat as a
    /// black box.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// What kind of content this handle refers to.
    #[getter]
    fn kind(&self) -> PyContentKind {
        self.inner.kind.into()
    }

    /// MIME type if known.
    #[getter]
    fn mime_type(&self) -> Option<&str> {
        self.inner.mime_type.as_deref()
    }

    /// Byte size if known.
    #[getter]
    fn byte_size(&self) -> Option<u64> {
        self.inner.byte_size
    }

    /// Human-readable display name (e.g. original filename) if known.
    #[getter]
    fn display_name(&self) -> Option<&str> {
        self.inner.display_name.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ContentHandle(id='{}', kind={}, mime_type={:?}, byte_size={:?}, display_name={:?})",
            self.inner.id,
            self.inner.kind.as_str(),
            self.inner.mime_type,
            self.inner.byte_size,
            self.inner.display_name,
        )
    }
}
