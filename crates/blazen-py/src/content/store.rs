//! Python wrapper for [`blazen_llm::content::ContentStore`] and the
//! built-in store implementations.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::content::{
    AnthropicFilesStore, ContentBody, ContentHint, DynContentStore, FalStorageStore,
    GeminiFilesStore, InMemoryContentStore, LocalFileContentStore, OpenAiFilesStore,
};

use crate::convert::json_to_py;
use crate::error::blazen_error_to_pyerr;

use super::handle::PyContentHandle;
use super::kind::PyContentKind;

// ---------------------------------------------------------------------------
// Body extraction
// ---------------------------------------------------------------------------

/// Convert a Python value to a [`ContentBody`].
///
/// Accepts `bytes` / `bytearray` (inline payload), `str` (URL), and
/// `pathlib.Path` or any object with a `__fspath__` (local file path).
fn extract_body(obj: &Bound<'_, PyAny>) -> PyResult<ContentBody> {
    // Bytes / bytearray.
    if let Ok(bytes) = obj.cast::<PyBytes>() {
        return Ok(ContentBody::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(buf) = obj.extract::<Vec<u8>>() {
        // Covers bytearray and any buffer-protocol type that decodes to bytes.
        // Note: this branch is reached only if the PyBytes downcast above
        // failed, so we don't double-copy regular `bytes` objects.
        return Ok(ContentBody::Bytes(buf));
    }

    // pathlib.Path / os.PathLike — duck-typed via __fspath__.
    if obj.hasattr("__fspath__").unwrap_or(false) {
        let fspath = obj.call_method0("__fspath__")?;
        if let Ok(path_str) = fspath.extract::<String>() {
            return Ok(ContentBody::LocalPath(PathBuf::from(path_str)));
        }
        if let Ok(path_bytes) = fspath.extract::<Vec<u8>>() {
            // PEP 519 allows bytes paths; only valid on POSIX.
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                let os = std::ffi::OsString::from_vec(path_bytes);
                return Ok(ContentBody::LocalPath(PathBuf::from(os)));
            }
            #[cfg(not(unix))]
            {
                let _ = path_bytes;
                return Err(PyValueError::new_err(
                    "bytes paths from __fspath__ are only supported on POSIX",
                ));
            }
        }
    }

    // String — interpret as URL.
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(ContentBody::Url(s.to_str()?.to_owned()));
    }

    Err(PyValueError::new_err(
        "ContentStore body must be bytes, str (URL), or pathlib.Path",
    ))
}

// ---------------------------------------------------------------------------
// PyContentStore
// ---------------------------------------------------------------------------

/// A pluggable content registry that backs handle resolution.
///
/// Wraps any backend implementing `blazen_llm::content::ContentStore`.
/// Construct with one of the static factory methods (`in_memory`,
/// `local_file`, `openai_files`, `anthropic_files`, `gemini_files`,
/// `fal_storage`).
///
/// All I/O methods return awaitables.
///
/// Example:
///     >>> store = ContentStore.in_memory()
///     >>> handle = await store.put(b"hello", kind=ContentKind.Other)
///     >>> source = await store.resolve(handle)
///     >>> source["type"]
///     'base64'
#[gen_stub_pyclass]
#[pyclass(name = "ContentStore", from_py_object)]
#[derive(Clone)]
pub struct PyContentStore {
    pub(crate) inner: DynContentStore,
}

impl PyContentStore {
    /// Construct from an `Arc<dyn ContentStore>` produced elsewhere in the
    /// codebase (used by future agent integrations).
    #[must_use]
    pub fn from_arc(inner: DynContentStore) -> Self {
        Self { inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyContentStore {
    // ------------------------------------------------------------------
    // Factories
    // ------------------------------------------------------------------

    /// Create the default in-memory store. Suitable for ephemeral / test
    /// workloads; switch to `local_file` or a provider store for
    /// persistence.
    #[staticmethod]
    fn in_memory() -> Self {
        Self {
            inner: Arc::new(InMemoryContentStore::new()),
        }
    }

    /// Create a filesystem-backed store rooted at `path`. The directory
    /// is created (recursively) if it does not already exist.
    #[staticmethod]
    fn local_file(path: PathBuf) -> PyResult<Self> {
        let store = LocalFileContentStore::new(path).map_err(blazen_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Create a store backed by the OpenAI Files API.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, base_url=None))]
    fn openai_files(api_key: String, base_url: Option<String>) -> Self {
        let mut store = OpenAiFilesStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self {
            inner: Arc::new(store),
        }
    }

    /// Create a store backed by the Anthropic Files API (beta).
    #[staticmethod]
    #[pyo3(signature = (api_key, *, base_url=None))]
    fn anthropic_files(api_key: String, base_url: Option<String>) -> Self {
        let mut store = AnthropicFilesStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self {
            inner: Arc::new(store),
        }
    }

    /// Create a store backed by the Google Gemini Files API.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, base_url=None))]
    fn gemini_files(api_key: String, base_url: Option<String>) -> Self {
        let mut store = GeminiFilesStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self {
            inner: Arc::new(store),
        }
    }

    /// Create a store backed by fal.ai object storage.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, base_url=None))]
    fn fal_storage(api_key: String, base_url: Option<String>) -> Self {
        let mut store = FalStorageStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self {
            inner: Arc::new(store),
        }
    }

    // ------------------------------------------------------------------
    // Async I/O
    // ------------------------------------------------------------------

    /// Persist content and return a freshly-issued `ContentHandle`.
    ///
    /// `body` may be `bytes`, a URL `str`, or a `pathlib.Path`. All
    /// keyword arguments are optional hints; the store may auto-detect
    /// the kind / MIME from the bytes when not provided.
    #[pyo3(signature = (body, *, kind=None, mime_type=None, display_name=None, byte_size=None))]
    fn put<'py>(
        &self,
        py: Python<'py>,
        body: Bound<'py, PyAny>,
        kind: Option<PyContentKind>,
        mime_type: Option<String>,
        display_name: Option<String>,
        byte_size: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_body = extract_body(&body)?;
        let mut hint = ContentHint::default();
        if let Some(k) = kind {
            hint.kind_hint = Some(k.into());
        }
        hint.mime_type = mime_type;
        hint.display_name = display_name;
        hint.byte_size = byte_size;

        let store = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handle = store
                .put(rust_body, hint)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyContentHandle { inner: handle })
        })
    }

    /// Resolve a handle to a wire-renderable media source. Returns a
    /// dict with the serialized `MediaSource` shape (e.g.
    /// `{"type": "url", "url": "..."}` or
    /// `{"type": "base64", "data": "..."}`).
    fn resolve<'py>(
        &self,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let source = store
                .resolve(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&source).map_err(|e| {
                PyValueError::new_err(format!("MediaSource serialization failed: {e}"))
            })?;
            Python::attach(|py| json_to_py(py, &json))
        })
    }

    /// Fetch the raw bytes for a handle. Stores that hold only references
    /// (URL / provider-file / local-path) may raise `UnsupportedError`.
    fn fetch_bytes<'py>(
        &self,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let bytes = store
                .fetch_bytes(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Python::attach(|py| Ok(PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Cheap metadata lookup (no byte materialization). Returns a dict
    /// with `kind`, `mime_type`, `byte_size`, `display_name`.
    fn metadata<'py>(
        &self,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let meta = store
                .metadata(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Python::attach(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("kind", meta.kind.as_str())?;
                dict.set_item("mime_type", meta.mime_type)?;
                dict.set_item("byte_size", meta.byte_size)?;
                dict.set_item("display_name", meta.display_name)?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    /// Optional cleanup hook. Default backends drop the entry; provider
    /// backends issue a delete call to the upstream API.
    fn delete<'py>(&self, py: Python<'py>, handle: PyContentHandle) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store
                .delete(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        format!("ContentStore({:?})", self.inner)
    }
}
