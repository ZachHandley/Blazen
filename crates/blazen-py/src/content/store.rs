//! Python wrapper for [`blazen_llm::content::ContentStore`] and the
//! built-in store implementations.
//!
//! `PyContentStore` is `#[pyclass(subclass)]`, so Python users can plug in
//! their own backend in two equivalent ways:
//!
//! 1. **Subclass** [`PyContentStore`] and override the async methods.
//!    `extract_store` detects the subclass and wraps it in a
//!    [`PyHostContentStore`] that implements the Rust [`ContentStore`]
//!    trait by calling back into Python.
//! 2. **Callbacks** via [`PyContentStore::custom`], which mirrors the
//!    Rust [`CustomContentStore::builder`] API. Each callable is bridged
//!    into a Rust closure using the same `block_in_place + Python::attach
//!    + into_future_with_locals + tokio::scope` dance used by
//!    `PyHostDispatch` in [`crate::providers::custom`].
//!
//! Both paths produce an `Arc<dyn ContentStore>` that any other Blazen
//! API accepting a content store can consume.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use pyo3::exceptions::{PyStopAsyncIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc;

use blazen_llm::content::store::ByteStream;
use blazen_llm::content::{
    AnthropicFilesStore, ContentBody, ContentHandle, ContentHint, ContentMetadata, ContentStore,
    CustomContentStore, DynContentStore, FalStorageStore, GeminiFilesStore, InMemoryContentStore,
    LocalFileContentStore, OpenAiFilesStore,
};
use blazen_llm::error::BlazenError;
use blazen_llm::types::MediaSource;

use crate::convert::json_to_py;
use crate::error::blazen_error_to_pyerr;

use super::handle::PyContentHandle;
use super::kind::PyContentKind;

// ---------------------------------------------------------------------------
// Kind parser (inverse of ContentKind::as_str)
// ---------------------------------------------------------------------------

/// Parse the canonical wire-name of a [`blazen_llm::content::ContentKind`].
/// Mirrors the names emitted by `ContentKind::as_str`.
fn parse_content_kind(s: &str) -> Option<blazen_llm::content::ContentKind> {
    use blazen_llm::content::ContentKind;
    Some(match s {
        "image" => ContentKind::Image,
        "audio" => ContentKind::Audio,
        "video" => ContentKind::Video,
        "document" => ContentKind::Document,
        "three_d_model" => ContentKind::ThreeDModel,
        "cad" => ContentKind::Cad,
        "archive" => ContentKind::Archive,
        "font" => ContentKind::Font,
        "code" => ContentKind::Code,
        "data" => ContentKind::Data,
        "other" => ContentKind::Other,
        _ => return None,
    })
}

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
        return Ok(ContentBody::Bytes {
            data: bytes.as_bytes().to_vec(),
        });
    }
    if let Ok(buf) = obj.extract::<Vec<u8>>() {
        // Covers bytearray and any buffer-protocol type that decodes to bytes.
        // Note: this branch is reached only if the PyBytes downcast above
        // failed, so we don't double-copy regular `bytes` objects.
        return Ok(ContentBody::Bytes { data: buf });
    }

    // pathlib.Path / os.PathLike — duck-typed via __fspath__.
    if obj.hasattr("__fspath__").unwrap_or(false) {
        let fspath = obj.call_method0("__fspath__")?;
        if let Ok(path_str) = fspath.extract::<String>() {
            return Ok(ContentBody::LocalPath {
                path: PathBuf::from(path_str),
            });
        }
        if let Ok(path_bytes) = fspath.extract::<Vec<u8>>() {
            // PEP 519 allows bytes paths; only valid on POSIX.
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                let os = std::ffi::OsString::from_vec(path_bytes);
                return Ok(ContentBody::LocalPath {
                    path: PathBuf::from(os),
                });
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
        return Ok(ContentBody::Url {
            url: s.to_str()?.to_owned(),
        });
    }

    Err(PyValueError::new_err(
        "ContentStore body must be bytes, str (URL), or pathlib.Path",
    ))
}

// ---------------------------------------------------------------------------
// Inner — built-in (Rust-side) vs subclass marker
// ---------------------------------------------------------------------------

/// Backing storage for [`PyContentStore`].
///
/// Built-in stores hold a real `Arc<dyn ContentStore>`. The `Subclass`
/// marker is what `PyContentStore::__init__` produces when a Python user
/// instantiates a subclass — calls to the base-class methods on this
/// variant raise `NotImplementedError` because the subclass is expected
/// to override them. The Rust side never asks the marker variant to do
/// anything useful; instead, [`extract_store`] detects subclasses and
/// hands out a [`PyHostContentStore`] adapter that dispatches into
/// Python.
enum Inner {
    /// A concrete Rust [`ContentStore`] (one of the built-in factories,
    /// or a [`CustomContentStore`] built from Python callbacks).
    BuiltIn(DynContentStore),
    /// Sentinel: this `PyContentStore` was instantiated as the base
    /// class of a Python subclass. Calls to the base-class default
    /// methods on this variant raise `NotImplementedError`.
    Subclass,
}

impl Clone for Inner {
    fn clone(&self) -> Self {
        match self {
            Self::BuiltIn(inner) => Self::BuiltIn(Arc::clone(inner)),
            Self::Subclass => Self::Subclass,
        }
    }
}

// ---------------------------------------------------------------------------
// PyContentStore
// ---------------------------------------------------------------------------

/// A pluggable content registry that backs handle resolution.
///
/// Wraps any backend implementing
/// [`blazen_llm::content::ContentStore`]. Construct with one of the
/// static factory methods (``in_memory``, ``local_file``,
/// ``openai_files``, ``anthropic_files``, ``gemini_files``,
/// ``fal_storage``, ``custom``) or by subclassing and overriding the
/// async methods (``put``, ``resolve``, ``fetch_bytes``,
/// ``fetch_stream``, ``delete``).
///
/// All I/O methods return awaitables.
///
/// Example (built-in factory):
///     >>> store = ContentStore.in_memory()
///     >>> handle = await store.put(b"hello", kind=ContentKind.Other)
///     >>> source = await store.resolve(handle)
///     >>> source["type"]
///     'base64'
///
/// Example (subclass):
///     >>> class S3ContentStore(ContentStore):
///     ...     def __init__(self, bucket: str):
///     ...         super().__init__()
///     ...         self.bucket = bucket
///     ...     async def put(self, body, hint): ...
///     ...     async def resolve(self, handle): ...
///     ...     async def fetch_bytes(self, handle): ...
///
/// Example (streaming subclass):
///     >>> class StreamingStore(ContentStore):
///     ...     async def put(self, body, hint):
///     ...         # `body` is a serde-tagged dict; the streaming variant
///     ...         # exposes an AsyncByteIter under body["stream"].
///     ...         if body["type"] == "stream":
///     ...             async for chunk in body["stream"]:
///     ...                 await self._sink.write(chunk)
///     ...         else:
///     ...             ...  # handle "bytes" / "url" / "local_path" / "provider_file"
///     ...         return ContentHandle(...)
///     ...     async def fetch_stream(self, handle):
///     ...         # Yield chunks from an async generator; the runtime
///     ...         # drives __anext__() and bridges into a Rust ByteStream.
///     ...         async def gen():
///     ...             async for chunk in self._source.iter(handle):
///     ...                 yield chunk
///     ...         return gen()
///
/// Example (callback factory):
///     >>> store = ContentStore.custom(
///     ...     put=async_put_fn,
///     ...     resolve=async_resolve_fn,
///     ...     fetch_bytes=async_fetch_fn,
///     ...     fetch_stream=async_fetch_stream_fn,  # may return bytes or AsyncIterator[bytes]
///     ...     name="my_s3_store",
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "ContentStore", subclass, skip_from_py_object)]
#[derive(Clone)]
pub struct PyContentStore {
    inner: Inner,
}

impl PyContentStore {
    /// Construct from an `Arc<dyn ContentStore>` produced elsewhere in the
    /// codebase (used by future agent integrations).
    #[must_use]
    pub fn from_arc(inner: DynContentStore) -> Self {
        Self {
            inner: Inner::BuiltIn(inner),
        }
    }
}

/// Attempt to extract a [`DynContentStore`] from any Python object.
///
/// - If `obj` is exactly a [`PyContentStore`] (not a subclass) wrapping a
///   [`Inner::BuiltIn`], unwrap and clone the inner `Arc`.
/// - If `obj` is a subclass of [`PyContentStore`] (or the base class
///   instantiated directly), wrap it in a [`PyHostContentStore`] adapter.
/// - Otherwise, raise `TypeError`.
///
/// Callers throughout `blazen-py` that need to feed a content store into
/// Rust code should go through this helper rather than poking at
/// `PyContentStore::inner` directly, so subclasses are handled
/// transparently.
#[allow(dead_code)]
pub(crate) fn extract_store(obj: &Bound<'_, PyAny>) -> PyResult<DynContentStore> {
    if obj.is_exact_instance_of::<PyContentStore>() {
        let store: PyRef<'_, PyContentStore> = obj.extract()?;
        return match &store.inner {
            Inner::BuiltIn(inner) => Ok(Arc::clone(inner)),
            // Direct instantiation of the base class with no overrides:
            // every method would raise NotImplementedError. Wrap in the
            // host adapter anyway so the standard error surface kicks in
            // when someone actually calls a method.
            Inner::Subclass => Ok(Arc::new(PyHostContentStore::new(obj.clone().unbind()))),
        };
    }
    if obj.is_instance_of::<PyContentStore>() {
        return Ok(Arc::new(PyHostContentStore::new(obj.clone().unbind())));
    }
    Err(PyTypeError::new_err(
        "expected a ContentStore instance or subclass",
    ))
}

#[gen_stub_pymethods]
#[pymethods]
impl PyContentStore {
    /// Base-class constructor. Call from your subclass's ``__init__`` via
    /// ``super().__init__()``. The base class on its own is not useful —
    /// the default method implementations raise ``NotImplementedError``.
    #[new]
    fn new() -> Self {
        Self {
            inner: Inner::Subclass,
        }
    }

    // ------------------------------------------------------------------
    // Factories
    // ------------------------------------------------------------------

    /// Create the default in-memory store. Suitable for ephemeral / test
    /// workloads; switch to `local_file` or a provider store for
    /// persistence.
    #[staticmethod]
    fn in_memory() -> Self {
        Self {
            inner: Inner::BuiltIn(Arc::new(InMemoryContentStore::new())),
        }
    }

    /// Create a filesystem-backed store rooted at `path`. The directory
    /// is created (recursively) if it does not already exist.
    #[staticmethod]
    fn local_file(path: PathBuf) -> PyResult<Self> {
        let store = LocalFileContentStore::new(path).map_err(blazen_error_to_pyerr)?;
        Ok(Self {
            inner: Inner::BuiltIn(Arc::new(store)),
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
            inner: Inner::BuiltIn(Arc::new(store)),
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
            inner: Inner::BuiltIn(Arc::new(store)),
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
            inner: Inner::BuiltIn(Arc::new(store)),
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
            inner: Inner::BuiltIn(Arc::new(store)),
        }
    }

    /// Create a store backed by user-supplied async callables.
    ///
    /// Mirrors the Rust [`CustomContentStore::builder`] API. ``put``,
    /// ``resolve``, and ``fetch_bytes`` are required; ``fetch_stream``
    /// and ``delete`` are optional. All callables must be ``async def``;
    /// they receive plain dicts for ``ContentBody`` / ``ContentHint``
    /// and a [`ContentHandle`] object for resolve / fetch / delete.
    ///
    /// ``put(body, hint)`` receives ``body`` as a serde-tagged dict that
    /// is one of:
    ///
    /// - ``{"type": "bytes", "data": [...]}`` — fully buffered payload.
    /// - ``{"type": "url", "url": "..."}`` — remote URL reference.
    /// - ``{"type": "local_path", "path": "..."}`` — local filesystem
    ///   reference.
    /// - ``{"type": "provider_file", ...}`` — opaque provider-file
    ///   reference (shape mirrors the Rust ``ProviderFile`` variant).
    /// - ``{"type": "stream", "stream": <AsyncByteIter>,
    ///   "size_hint": int | None}`` — chunk-by-chunk streaming. Iterate
    ///   the ``stream`` field with ``async for chunk in body["stream"]``
    ///   to consume the upload without buffering the full payload in
    ///   memory.
    ///
    /// ``put`` must return a [`ContentHandle`].
    /// ``resolve`` must return a dict shaped like the serialized
    /// [`MediaSource`] (e.g. ``{"type": "url", "url": "..."}`` or
    /// ``{"type": "base64", "data": "..."}``).
    /// ``fetch_bytes`` must return ``bytes``.
    /// ``fetch_stream`` may return either ``bytes`` (drained) or an
    /// ``AsyncIterator[bytes]`` for true chunk-by-chunk streaming.
    /// ``delete`` returns ``None``.
    #[staticmethod]
    #[pyo3(signature = (
        *,
        put,
        resolve,
        fetch_bytes,
        fetch_stream=None,
        delete=None,
        name=None,
    ))]
    fn custom(
        put: Py<PyAny>,
        resolve: Py<PyAny>,
        fetch_bytes: Py<PyAny>,
        fetch_stream: Option<Py<PyAny>>,
        delete: Option<Py<PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let name = name.unwrap_or_else(|| "custom".to_owned());
        let mut builder = CustomContentStore::builder(name);

        // put(body, hint) -> ContentHandle
        let put_cb = Arc::new(put);
        builder = builder.put(move |body, hint| {
            let put_cb = Arc::clone(&put_cb);
            Box::pin(async move { call_put(put_cb.as_ref(), body, hint).await })
        });

        // resolve(handle) -> MediaSource (dict)
        let resolve_cb = Arc::new(resolve);
        builder = builder.resolve(move |handle| {
            let resolve_cb = Arc::clone(&resolve_cb);
            Box::pin(async move { call_resolve(resolve_cb.as_ref(), handle).await })
        });

        // fetch_bytes(handle) -> bytes
        let fetch_cb = Arc::new(fetch_bytes);
        builder = builder.fetch_bytes(move |handle| {
            let fetch_cb = Arc::clone(&fetch_cb);
            Box::pin(async move { call_fetch_bytes(fetch_cb.as_ref(), handle).await })
        });

        // fetch_stream(handle) -> bytes | AsyncIterator[bytes]
        if let Some(fs) = fetch_stream {
            let fs_cb = Arc::new(fs);
            builder = builder.fetch_stream(move |handle| {
                let fs_cb = Arc::clone(&fs_cb);
                Box::pin(async move { call_fetch_stream_dispatch(fs_cb.as_ref(), handle).await })
            });
        }

        // delete(handle) -> None
        if let Some(del) = delete {
            let del_cb = Arc::new(del);
            builder = builder.delete(move |handle| {
                let del_cb = Arc::clone(&del_cb);
                Box::pin(async move { call_delete(del_cb.as_ref(), handle).await })
            });
        }

        let store = builder.build().map_err(blazen_error_to_pyerr)?;
        Ok(Self {
            inner: Inner::BuiltIn(Arc::new(store)),
        })
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
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        body: Bound<'py, PyAny>,
        kind: Option<PyContentKind>,
        mime_type: Option<String>,
        display_name: Option<String>,
        byte_size: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = require_built_in(&slf, "put")?;
        let rust_body = extract_body(&body)?;
        let mut hint = ContentHint::default();
        if let Some(k) = kind {
            hint.kind_hint = Some(k.into());
        }
        hint.mime_type = mime_type;
        hint.display_name = display_name;
        hint.byte_size = byte_size;

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
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = require_built_in(&slf, "resolve")?;
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
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = require_built_in(&slf, "fetch_bytes")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let bytes = store
                .fetch_bytes(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Python::attach(|py| Ok(PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Fetch the content as a chunk-by-chunk async iterator. Returns an
    /// awaitable that resolves to an [`AsyncByteIter`]; iterate it with
    /// ``async for chunk in await store.fetch_stream(handle)`` to consume
    /// the payload without buffering it in memory.
    ///
    /// Stores that hold only references (URL / provider-file) may raise
    /// `UnsupportedError`. Built-in stores fall back to a single-chunk
    /// iterator over `fetch_bytes` when no native streaming path exists.
    fn fetch_stream<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = require_built_in(&slf, "fetch_stream")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = store
                .fetch_stream(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let iter = PyAsyncByteIter::from_byte_stream(stream);
            Python::attach(|py| Py::new(py, iter).map(pyo3::Py::into_any))
        })
    }

    /// Cheap metadata lookup (no byte materialization). Returns a dict
    /// with `kind`, `mime_type`, `byte_size`, `display_name`.
    fn metadata<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = require_built_in(&slf, "metadata")?;
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
    fn delete<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handle: PyContentHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = require_built_in(&slf, "delete")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store
                .delete(&handle.inner)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Inner::BuiltIn(inner) => format!("ContentStore({inner:?})"),
            Inner::Subclass => "ContentStore(<subclass>)".to_owned(),
        }
    }
}

/// Default-method guard: the base-class `put`/`resolve`/etc. methods only
/// dispatch on built-in stores. When a subclass forgets to override a
/// method and `super().method(...)` falls through to the base, we raise
/// `NotImplementedError` instead of silently looping back into the
/// host-dispatch adapter.
fn require_built_in(slf: &PyRef<'_, PyContentStore>, method: &str) -> PyResult<DynContentStore> {
    match &slf.inner {
        Inner::BuiltIn(inner) => Ok(Arc::clone(inner)),
        Inner::Subclass => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "ContentStore subclass must override `{method}()`"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Callback bridges (used by ContentStore.custom)
// ---------------------------------------------------------------------------
//
// Each helper:
//   1. Acquires the GIL (under `block_in_place` so we don't pin a tokio
//      worker), pythonizes its argument(s), invokes the Python callable
//      to obtain a coroutine, and converts the coroutine to a Rust
//      future via `into_future_with_locals`.
//   2. Drives the future to completion outside the GIL inside
//      `tokio::scope`, so any nested asyncio operations the host
//      callable performs see the right task locals.
//   3. Reacquires the GIL to extract the result back into Rust.

/// Build the Python-side `body` argument for a `put` dispatch.
///
/// For [`ContentBody::Stream`] the inner stream is wrapped in a
/// [`PyAsyncByteIter`] and exposed as `{"type": "stream", "stream":
/// <AsyncByteIter>, "size_hint": int | None}` so the Python callback
/// can iterate chunks without a Rust-side drain. All other variants go
/// through `pythonize` and keep their existing serde-tagged shape.
fn pythonize_put_body(py: Python<'_>, body: ContentBody) -> PyResult<Bound<'_, PyAny>> {
    match body {
        ContentBody::Stream { stream, size_hint } => {
            let py_iter = PyAsyncByteIter::from_byte_stream(stream);
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("type", "stream")?;
            dict.set_item("stream", Py::new(py, py_iter)?)?;
            dict.set_item("size_hint", size_hint)?;
            Ok(dict.into_any())
        }
        other => pythonize::pythonize(py, &other)
            .map_err(|e| PyValueError::new_err(format!("failed to pythonize ContentBody: {e}"))),
    }
}

async fn call_put(
    cb: &Py<PyAny>,
    body: ContentBody,
    hint: ContentHint,
) -> Result<ContentHandle, BlazenError> {
    let (fut, locals) = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<_> {
            let py_body = pythonize_put_body(py, body)?;
            let py_hint = pythonize::pythonize(py, &hint).map_err(|e| {
                PyValueError::new_err(format!("failed to pythonize ContentHint: {e}"))
            })?;
            let cb_bound = cb.bind(py);
            let coro = cb_bound.call1((py_body, py_hint))?;
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
            Ok((fut, locals))
        })
    })
    .map_err(|e: PyErr| BlazenError::provider("custom", format!("put dispatch failed: {e}")))?;

    let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
        .await
        .map_err(|e: PyErr| BlazenError::provider("custom", format!("put raised: {e}")))?;

    tokio::task::block_in_place(|| {
        Python::attach(|py| -> Result<ContentHandle, BlazenError> {
            let bound = py_result.bind(py);
            let handle: PyRef<'_, PyContentHandle> = bound.extract().map_err(|e| {
                BlazenError::provider("custom", format!("put() must return a ContentHandle: {e}"))
            })?;
            Ok(handle.inner.clone())
        })
    })
}

async fn call_resolve(cb: &Py<PyAny>, handle: ContentHandle) -> Result<MediaSource, BlazenError> {
    let (fut, locals) = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<_> {
            let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
            let cb_bound = cb.bind(py);
            let coro = cb_bound.call1((py_handle,))?;
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
            Ok((fut, locals))
        })
    })
    .map_err(|e: PyErr| BlazenError::provider("custom", format!("resolve dispatch failed: {e}")))?;

    let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
        .await
        .map_err(|e: PyErr| BlazenError::provider("custom", format!("resolve raised: {e}")))?;

    tokio::task::block_in_place(|| {
        Python::attach(|py| -> Result<MediaSource, BlazenError> {
            let bound = py_result.bind(py);
            let value: serde_json::Value = pythonize::depythonize(bound).map_err(|e| {
                BlazenError::provider(
                    "custom",
                    format!("failed to depythonize resolve() result: {e}"),
                )
            })?;
            serde_json::from_value(value).map_err(|e| {
                BlazenError::provider(
                    "custom",
                    format!("resolve() result is not a valid MediaSource: {e}"),
                )
            })
        })
    })
}

async fn call_fetch_bytes(cb: &Py<PyAny>, handle: ContentHandle) -> Result<Vec<u8>, BlazenError> {
    let (fut, locals) = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<_> {
            let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
            let cb_bound = cb.bind(py);
            let coro = cb_bound.call1((py_handle,))?;
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
            Ok((fut, locals))
        })
    })
    .map_err(|e: PyErr| {
        BlazenError::provider("custom", format!("fetch_bytes dispatch failed: {e}"))
    })?;

    let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
        .await
        .map_err(|e: PyErr| BlazenError::provider("custom", format!("fetch_bytes raised: {e}")))?;

    tokio::task::block_in_place(|| {
        Python::attach(|py| -> Result<Vec<u8>, BlazenError> {
            let bound = py_result.bind(py);
            bound.extract::<Vec<u8>>().map_err(|e| {
                BlazenError::provider("custom", format!("fetch_bytes() must return bytes: {e}"))
            })
        })
    })
}

async fn call_fetch_stream_dispatch(
    cb: &Py<PyAny>,
    handle: ContentHandle,
) -> Result<ByteStream, BlazenError> {
    let (fut, locals) = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<_> {
            let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
            let cb_bound = cb.bind(py);
            let coro = cb_bound.call1((py_handle,))?;
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
            Ok((fut, locals))
        })
    })
    .map_err(|e: PyErr| {
        BlazenError::provider("custom", format!("fetch_stream dispatch failed: {e}"))
    })?;

    let py_result = pyo3_async_runtimes::tokio::scope(locals.clone(), fut)
        .await
        .map_err(|e: PyErr| BlazenError::provider("custom", format!("fetch_stream raised: {e}")))?;

    dispatch_fetch_stream_result(py_result, &locals)
}

/// Dispatch a resolved Python ``fetch_stream`` result into a [`ByteStream`].
///
/// Order matters: we test ``__aiter__`` BEFORE ``extract::<Vec<u8>>()``
/// because some buffer-protocol objects could satisfy both, and we don't
/// want to drain an iterator we should be streaming chunk-by-chunk.
///
/// `locals` is captured from the calling asyncio context — the spawned
/// pump task in `pystream_into_byte_stream` needs it to drive nested
/// `__anext__()` coroutines after the parent loop has unwound.
fn dispatch_fetch_stream_result(
    py_result: Py<PyAny>,
    locals: &pyo3_async_runtimes::TaskLocals,
) -> Result<ByteStream, BlazenError> {
    tokio::task::block_in_place(|| {
        Python::attach(|py| -> Result<ByteStream, BlazenError> {
            let bound = py_result.bind(py);
            let is_async_iter = bound.hasattr("__aiter__").map_err(|e| {
                BlazenError::provider(
                    "custom",
                    format!("fetch_stream() result hasattr probe failed: {e}"),
                )
            })?;
            if is_async_iter {
                let iter_obj = bound
                    .call_method0("__aiter__")
                    .map_err(|e| {
                        BlazenError::provider(
                            "custom",
                            format!("fetch_stream() __aiter__() failed: {e}"),
                        )
                    })?
                    .unbind();
                return Ok(pystream_into_byte_stream(iter_obj, locals.clone()));
            }
            if let Ok(buf) = bound.extract::<Vec<u8>>() {
                return Ok(bytes_into_byte_stream(buf));
            }
            Err(BlazenError::provider(
                "custom",
                "fetch_stream() must return bytes or AsyncIterator[bytes]",
            ))
        })
    })
}

async fn call_delete(cb: &Py<PyAny>, handle: ContentHandle) -> Result<(), BlazenError> {
    let (fut, locals) = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<_> {
            let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
            let cb_bound = cb.bind(py);
            let coro = cb_bound.call1((py_handle,))?;
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
            Ok((fut, locals))
        })
    })
    .map_err(|e: PyErr| BlazenError::provider("custom", format!("delete dispatch failed: {e}")))?;

    pyo3_async_runtimes::tokio::scope(locals, fut)
        .await
        .map_err(|e: PyErr| BlazenError::provider("custom", format!("delete raised: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// PyHostContentStore — adapter for Python subclasses
// ---------------------------------------------------------------------------

/// Implements [`ContentStore`] by dispatching back into a Python object
/// (typically a subclass of [`PyContentStore`]).
///
/// Mirrors [`crate::persist::store::PyHostCheckpointStore`] in style, and
/// uses the same Python-coroutine bridging pattern as
/// [`crate::providers::custom::PyHostDispatch`]. Optional methods
/// (`fetch_stream`, `delete`) are detected via a cached `hasattr` check
/// so we don't pay the GIL cost on every call.
pub struct PyHostContentStore {
    py_obj: Py<PyAny>,
    /// Cache of `hasattr` results for optional methods, keyed by method
    /// name. Avoids repeated GIL acquisitions on the hot path.
    has_method_cache: Mutex<std::collections::HashMap<String, bool>>,
}

impl PyHostContentStore {
    pub fn new(py_obj: Py<PyAny>) -> Self {
        Self {
            py_obj,
            has_method_cache: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Cached `hasattr(self.py_obj, method) and callable(...)`. False on
    /// any GIL error so callers fall through to the default behavior.
    fn has_method(&self, method: &str) -> bool {
        if let Ok(cache) = self.has_method_cache.lock()
            && let Some(&cached) = cache.get(method)
        {
            return cached;
        }
        let has = Python::attach(|py| -> PyResult<bool> {
            let host = self.py_obj.bind(py);
            if !host.hasattr(method)? {
                return Ok(false);
            }
            Ok(host.getattr(method)?.is_callable())
        })
        .unwrap_or(false);
        if let Ok(mut cache) = self.has_method_cache.lock() {
            cache.insert(method.to_owned(), has);
        }
        has
    }
}

impl std::fmt::Debug for PyHostContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyHostContentStore").finish()
    }
}

#[async_trait]
impl ContentStore for PyHostContentStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_body = pythonize_put_body(py, body)?;
                let py_hint = pythonize::pythonize(py, &hint).map_err(|e| {
                    PyValueError::new_err(format!("failed to pythonize ContentHint: {e}"))
                })?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("put", (py_body, py_hint))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| BlazenError::provider("custom", format!("put dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| BlazenError::provider("custom", format!("put raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<ContentHandle, BlazenError> {
                let bound = py_result.bind(py);
                let handle: PyRef<'_, PyContentHandle> = bound.extract().map_err(|e| {
                    BlazenError::provider(
                        "custom",
                        format!("put() must return a ContentHandle: {e}"),
                    )
                })?;
                Ok(handle.inner.clone())
            })
        })
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        let handle = handle.clone();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("resolve", (py_handle,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider("custom", format!("resolve dispatch failed: {e}"))
        })?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| BlazenError::provider("custom", format!("resolve raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<MediaSource, BlazenError> {
                let bound = py_result.bind(py);
                let value: serde_json::Value = pythonize::depythonize(bound).map_err(|e| {
                    BlazenError::provider(
                        "custom",
                        format!("failed to depythonize resolve() result: {e}"),
                    )
                })?;
                serde_json::from_value(value).map_err(|e| {
                    BlazenError::provider(
                        "custom",
                        format!("resolve() result is not a valid MediaSource: {e}"),
                    )
                })
            })
        })
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let handle = handle.clone();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("fetch_bytes", (py_handle,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider("custom", format!("fetch_bytes dispatch failed: {e}"))
        })?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| {
                BlazenError::provider("custom", format!("fetch_bytes raised: {e}"))
            })?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<Vec<u8>, BlazenError> {
                let bound = py_result.bind(py);
                bound.extract::<Vec<u8>>().map_err(|e| {
                    BlazenError::provider("custom", format!("fetch_bytes() must return bytes: {e}"))
                })
            })
        })
    }

    async fn fetch_stream(&self, handle: &ContentHandle) -> Result<ByteStream, BlazenError> {
        let handle = handle.clone();
        if self.has_method("fetch_stream") {
            let h = handle.clone();
            let (fut, locals) = tokio::task::block_in_place(|| {
                Python::attach(|py| -> PyResult<_> {
                    let py_handle = Py::new(py, PyContentHandle { inner: h })?;
                    let host = self.py_obj.bind(py);
                    let coro = host.call_method1("fetch_stream", (py_handle,))?;
                    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                    let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                    Ok((fut, locals))
                })
            })
            .map_err(|e: PyErr| {
                BlazenError::provider("custom", format!("fetch_stream dispatch failed: {e}"))
            })?;

            let py_result = pyo3_async_runtimes::tokio::scope(locals.clone(), fut)
                .await
                .map_err(|e: PyErr| {
                    BlazenError::provider("custom", format!("fetch_stream raised: {e}"))
                })?;

            dispatch_fetch_stream_result(py_result, &locals)
        } else {
            let bytes = self.fetch_bytes(&handle).await?;
            Ok(bytes_into_byte_stream(bytes))
        }
    }

    async fn metadata(&self, handle: &ContentHandle) -> Result<ContentMetadata, BlazenError> {
        // If the host overrides `metadata`, dispatch to it; otherwise fall
        // through to the trait's default impl (which calls `resolve` and
        // re-uses handle metadata). Python-side return shape mirrors what
        // the built-in `metadata` getter exposes:
        // ``{"kind": str, "mime_type": str | None,
        //    "byte_size": int | None, "display_name": str | None}``.
        if !self.has_method("metadata") {
            // Default impl from the trait: resolve and report what the
            // handle already knows. We can't call the trait's default
            // method directly from here, so inline it.
            let _ = self.resolve(handle).await?;
            return Ok(ContentMetadata {
                kind: handle.kind,
                mime_type: handle.mime_type.clone(),
                byte_size: handle.byte_size,
                display_name: handle.display_name.clone(),
            });
        }

        let handle = handle.clone();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("metadata", (py_handle,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider("custom", format!("metadata dispatch failed: {e}"))
        })?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| BlazenError::provider("custom", format!("metadata raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<ContentMetadata, BlazenError> {
                let bound = py_result.bind(py);
                let dict = bound.cast::<pyo3::types::PyDict>().map_err(|e| {
                    BlazenError::provider("custom", format!("metadata() must return a dict: {e}"))
                })?;
                let kind_str: String = dict
                    .get_item("kind")
                    .map_err(|e| {
                        BlazenError::provider(
                            "custom",
                            format!("metadata() dict missing `kind`: {e}"),
                        )
                    })?
                    .ok_or_else(|| {
                        BlazenError::provider("custom", "metadata() dict missing `kind`")
                    })?
                    .extract()
                    .map_err(|e| {
                        BlazenError::provider(
                            "custom",
                            format!("metadata() `kind` must be a string: {e}"),
                        )
                    })?;
                let kind = parse_content_kind(&kind_str).ok_or_else(|| {
                    BlazenError::provider(
                        "custom",
                        format!("metadata() `kind` is not a known ContentKind: {kind_str}"),
                    )
                })?;
                let mime_type: Option<String> = dict
                    .get_item("mime_type")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok());
                let byte_size: Option<u64> = dict
                    .get_item("byte_size")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok());
                let display_name: Option<String> = dict
                    .get_item("display_name")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok());
                Ok(ContentMetadata {
                    kind,
                    mime_type,
                    byte_size,
                    display_name,
                })
            })
        })
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        if !self.has_method("delete") {
            return Ok(());
        }
        let handle = handle.clone();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_handle = Py::new(py, PyContentHandle { inner: handle })?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("delete", (py_handle,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider("custom", format!("delete dispatch failed: {e}"))
        })?;

        pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| BlazenError::provider("custom", format!("delete raised: {e}")))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Streaming bridges (Rust ByteStream <-> Python async iterator)
// ---------------------------------------------------------------------------
//
// These helpers are intentionally unwired in Wave 1 — Wave 2/3 will
// integrate them into the `put`/`fetch_stream` paths. The `dead_code`
// allows are scoped to each item and should be removed at integration.

/// Channel buffer size for the chunk-pump. Four chunks gives modest
/// readahead while keeping memory bounded under backpressure.
const STREAM_CHANNEL_CAPACITY: usize = 4;

/// Async iterator over a Rust [`ByteStream`], exposed to Python.
///
/// Implements the ``__aiter__`` / ``__anext__`` protocol so Python code
/// can do ``async for chunk in stream: ...`` with each chunk arriving as
/// a ``bytes`` object. Chunks are pulled from a bounded
/// `mpsc::channel(STREAM_CHANNEL_CAPACITY)` fed by a background tokio
/// task that drains the underlying stream.
///
/// Instances are produced by the runtime (e.g. as ``body["stream"]`` in
/// a streaming ``put`` callback); they are not constructible from Python.
#[gen_stub_pyclass]
#[pyclass(name = "AsyncByteIter")]
pub struct PyAsyncByteIter {
    rx: Arc<AsyncMutex<mpsc::Receiver<Result<Bytes, BlazenError>>>>,
}

impl PyAsyncByteIter {
    /// Wrap a Rust [`ByteStream`] as a Python async iterator. Spawns a
    /// background tokio task that pumps chunks into a bounded channel
    /// and exits cleanly when the source stream ends or errors.
    pub(crate) fn from_byte_stream(stream: ByteStream) -> Self {
        use futures_util::StreamExt;
        let (tx, rx) = mpsc::channel::<Result<Bytes, BlazenError>>(STREAM_CHANNEL_CAPACITY);
        tokio::spawn(async move {
            let mut stream = stream;
            while let Some(item) = stream.next().await {
                let is_err = item.is_err();
                if tx.send(item).await.is_err() {
                    // Receiver dropped — consumer gave up; stop pulling.
                    return;
                }
                if is_err {
                    return;
                }
            }
        });
        Self {
            rx: Arc::new(AsyncMutex::new(rx)),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAsyncByteIter {
    /// Return ``self``; the iterator is its own async iterator.
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Pull the next chunk. Returns an awaitable that resolves to
    /// ``bytes`` (the next chunk) or raises ``StopAsyncIteration`` when
    /// the underlying stream is exhausted. Errors from the source stream
    /// surface here as the Blazen error type.
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = Arc::clone(&self.rx);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                Some(Ok(chunk)) => {
                    Python::attach(|py| Ok(PyBytes::new(py, &chunk).unbind().into_any()))
                }
                Some(Err(e)) => Err(blazen_error_to_pyerr(e)),
                None => Err(PyStopAsyncIteration::new_err("stream exhausted")),
            }
        })
    }
}

/// Drive a Python async-iterator object into a Rust [`ByteStream`].
///
/// Spawns a tokio task that repeatedly calls ``__anext__`` on the
/// supplied iterator, awaits each coroutine via
/// `pyo3_async_runtimes::tokio::scope` (so nested asyncio sees the
/// right task locals), and feeds the resulting chunks into a bounded
/// `mpsc::channel(STREAM_CHANNEL_CAPACITY)`. Accepts ``bytes``,
/// ``bytearray``, or any buffer-protocol object as chunk types.
///
/// `py_iter` must already be the result of `__aiter__` (or an object
/// that is already an async iterator), and `locals` must be the
/// asyncio task locals captured from the calling Python context.
/// Both are required up front because the spawned tokio task runs
/// after the calling asyncio loop has unwound — `get_current_locals`
/// would fail there with "no running event loop".
///
/// We use `futures_util::stream::unfold` to bridge the receiver back
/// into a `Stream` because the workspace's `tokio-stream` build only
/// enables the `sync` feature and does not expose `wrappers`.
pub(crate) fn pystream_into_byte_stream(
    iter_obj: Py<PyAny>,
    locals: pyo3_async_runtimes::TaskLocals,
) -> ByteStream {
    let (tx, rx) = mpsc::channel::<Result<Bytes, BlazenError>>(STREAM_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        loop {
            let next_fut = tokio::task::block_in_place(|| {
                Python::attach(|py| -> PyResult<_> {
                    let bound = iter_obj.bind(py);
                    let coro = bound.call_method0("__anext__")?;
                    pyo3_async_runtimes::into_future_with_locals(&locals, coro)
                })
            });

            let next_fut = match next_fut {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx
                        .send(Err(BlazenError::request(format!(
                            "__anext__ dispatch failed: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            let py_result = pyo3_async_runtimes::tokio::scope(locals.clone(), next_fut).await;

            let py_obj = match py_result {
                Ok(obj) => obj,
                Err(e) => {
                    let stop = Python::attach(|py| e.is_instance_of::<PyStopAsyncIteration>(py));
                    if stop {
                        return;
                    }
                    let _ = tx
                        .send(Err(BlazenError::request(format!(
                            "Python async iterator raised: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            // Extract chunk: PyBytes via fast-path (no allocation in the
            // bytes->Vec conversion is mooted because Bytes::copy_from_slice
            // copies anyway), otherwise fall back to extract::<Vec<u8>>()
            // which handles bytearray and buffer-protocol objects (matches
            // the existing `extract_body` policy in this file).
            let chunk = tokio::task::block_in_place(|| {
                Python::attach(|py| -> PyResult<Bytes> {
                    let bound = py_obj.bind(py);
                    if let Ok(b) = bound.cast::<PyBytes>() {
                        return Ok(Bytes::copy_from_slice(b.as_bytes()));
                    }
                    if let Ok(buf) = bound.extract::<Vec<u8>>() {
                        return Ok(Bytes::from(buf));
                    }
                    Err(PyTypeError::new_err(
                        "async iterator must yield bytes, bytearray, or a buffer-protocol object",
                    ))
                })
            });

            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx
                        .send(Err(BlazenError::request(format!(
                            "failed to extract chunk: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            if tx.send(Ok(chunk)).await.is_err() {
                // Consumer dropped.
                return;
            }
        }
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    Box::pin(stream)
}

/// Wrap a fully-materialized byte buffer as a single-chunk
/// [`ByteStream`]. Provided so callers can branch
/// ``if is_async_iter { pystream_into_byte_stream(...) }
/// else { bytes_into_byte_stream(...) }`` without re-stating the
/// `futures_util::stream::once` boilerplate.
pub(crate) fn bytes_into_byte_stream(data: Vec<u8>) -> ByteStream {
    Box::pin(futures_util::stream::once(
        async move { Ok(Bytes::from(data)) },
    ))
}
