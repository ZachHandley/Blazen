//! Python wrappers for `blazen-memory` types: backends, `Memory`, and search results.

use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_llm::retry::RetryConfig;
use blazen_memory::search;
use blazen_memory::store::MemoryBackend;
use blazen_memory::types::{MemoryEntry, MemoryResult, StoredEntry};
use blazen_memory::{
    InMemoryBackend, JsonlBackend, Memory, MemoryError, MemoryStore, RetryMemoryBackend,
};
use blazen_memory_valkey::{UpstashBackend, ValkeyBackend};

/// Local alias for the memory-crate result type.
type MemResult<T> = std::result::Result<T, MemoryError>;

use crate::error::{BlazenException, BlazenPyError};
use crate::types::embedding::PyEmbeddingModel;

// ---------------------------------------------------------------------------
// MemoryException -- Python exception subclass of BlazenException
// ---------------------------------------------------------------------------

pyo3::create_exception!(blazen, MemoryException, BlazenException);

/// Register the memory exception type on the module.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MemoryError", m.py().get_type::<MemoryException>())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: extract a backend from a Python object
// ---------------------------------------------------------------------------

/// Attempt to extract one of the supported backend types from a Python object.
fn extract_backend(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn MemoryBackend>> {
    if let Ok(b) = obj.extract::<PyRef<'_, PyInMemoryBackend>>() {
        return Ok(b.inner.clone());
    }
    if let Ok(b) = obj.extract::<PyRef<'_, PyJsonlBackend>>() {
        return Ok(b.inner.clone());
    }
    if let Ok(b) = obj.extract::<PyRef<'_, PyValkeyBackend>>() {
        return Ok(b.inner.clone());
    }
    if let Ok(b) = obj.extract::<PyRef<'_, PyUpstashBackend>>() {
        return Ok(b.inner.clone());
    }
    if let Ok(b) = obj.extract::<PyRef<'_, PyRetryMemoryBackend>>() {
        return Ok(b.inner.clone());
    }
    // Try MemoryBackend subclass
    if obj.is_instance_of::<PyMemoryBackend>() {
        return Ok(Arc::new(PyHostMemoryBackend::new(obj.clone().unbind())));
    }
    Err(PyTypeError::new_err(
        "expected InMemoryBackend, JsonlBackend, ValkeyBackend, UpstashBackend, RetryMemoryBackend, or MemoryBackend subclass",
    ))
}

// ---------------------------------------------------------------------------
// Helper: convert MemoryError -> PyErr
// ---------------------------------------------------------------------------

pub(crate) fn memory_err(e: blazen_memory::MemoryError) -> PyErr {
    MemoryException::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// PyMemoryStore (subclassable base mirroring the high-level trait)
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring the Rust [`blazen_memory::MemoryStore`] trait.
///
/// ``MemoryStore`` is the user-facing interface a fully-built memory system
/// satisfies (handles embedding + ELID encoding + search + persistence).
/// The shipped concrete implementation lives behind :class:`Memory`; this
/// ABC exists so callers can document the surface that custom orchestrators
/// must satisfy when composing memory subsystems together (e.g. a router
/// that fans out across several stores, or a wrapper that adds caching).
///
/// Override every coroutine method -- the defaults raise
/// ``NotImplementedError``.
///
/// Example:
///     >>> class CachingStore(MemoryStore):
///     ...     def __init__(self, inner: Memory, cache: dict): ...
///     ...     async def add(self, entries): ...
///     ...     async def search(self, query, limit, metadata_filter=None): ...
///     ...     async def search_local(self, query, limit, metadata_filter=None): ...
///     ...     async def get(self, id): ...
///     ...     async def delete(self, id): ...
///     ...     async def len(self): ...
#[gen_stub_pyclass]
#[pyclass(name = "MemoryStore", subclass)]
pub struct PyMemoryStore;

#[gen_stub_pymethods]
#[pymethods]
impl PyMemoryStore {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Add one or more entries to the store. Should return a coroutine
    /// resolving to ``list[str]`` of stored ids.
    fn add(&self, entries: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let _ = entries;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override add()",
        ))
    }

    /// Semantic search using the configured embedding model.
    ///
    /// Should return a coroutine resolving to ``list[MemoryResult]``.
    /// Implementations that do not have an embedding model should
    /// raise an error.
    #[pyo3(signature = (query, limit=5, metadata_filter=None))]
    fn search(
        &self,
        query: String,
        limit: usize,
        metadata_filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (query, limit, metadata_filter);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override search()",
        ))
    }

    /// Local SimHash-based search (no embedding model required).
    ///
    /// Should return a coroutine resolving to ``list[MemoryResult]``.
    #[pyo3(signature = (query, limit=5, metadata_filter=None))]
    fn search_local(
        &self,
        query: String,
        limit: usize,
        metadata_filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (query, limit, metadata_filter);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override search_local()",
        ))
    }

    /// Retrieve a single entry by id. Should return a coroutine
    /// resolving to an optional dict.
    fn get(&self, id: String) -> PyResult<Py<PyAny>> {
        let _ = id;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override get()",
        ))
    }

    /// Delete an entry by id. Should return a coroutine resolving to
    /// ``bool``.
    fn delete(&self, id: String) -> PyResult<Py<PyAny>> {
        let _ = id;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override delete()",
        ))
    }

    /// Return the number of entries. Should return a coroutine
    /// resolving to ``int``.
    fn len(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override len()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyMemoryBackend (subclassable base)
// ---------------------------------------------------------------------------

/// Base class for custom memory storage backends.
///
/// Subclass and override all methods to implement a custom backend
/// (e.g. PostgreSQL, DynamoDB, SQLite).
///
/// Example:
///     >>> class PostgresBackend(MemoryBackend):
///     ...     async def put(self, entry): ...
///     ...     async def get(self, id): ...
///     ...     async def delete(self, id): ...
///     ...     async def list(self): ...
///     ...     async def len(self): ...
///     ...     async def search_by_bands(self, bands, limit): ...
#[gen_stub_pyclass]
#[pyclass(name = "MemoryBackend", subclass)]
pub struct PyMemoryBackend {}

#[gen_stub_pymethods]
#[pymethods]
impl PyMemoryBackend {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Subclass-friendly `__init__` no-op. PyO3's `#[new]` only implements
    /// `__new__`; without an explicit `__init__` here, a Python subclass that
    /// chains `super().__init__()` falls through to `object.__init__` and
    /// raises ``TypeError``. The base `MemoryBackend` carries no state, so
    /// this is a true no-op.
    fn __init__(&self) {}

    /// Store an entry.
    fn put(&self, _entry: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override put()",
        ))
    }

    /// Retrieve an entry by ID.
    fn get(&self, _id: String) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override get()",
        ))
    }

    /// Delete an entry by ID.
    fn delete(&self, _id: String) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override delete()",
        ))
    }

    /// List all entries.
    fn list(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override list()",
        ))
    }

    /// Return the number of entries.
    fn len(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override len()",
        ))
    }

    /// Search by SimHash bands.
    fn search_by_bands(&self, _bands: Vec<String>, _limit: usize) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override search_by_bands()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyHostMemoryBackend (Rust trait adapter over Python subclass)
// ---------------------------------------------------------------------------

/// Adapter that implements the Rust [`MemoryBackend`] trait by calling
/// back into a Python subclass instance.
pub(crate) struct PyHostMemoryBackend {
    py_obj: Py<PyAny>,
}

impl PyHostMemoryBackend {
    pub fn new(py_obj: Py<PyAny>) -> Self {
        Self { py_obj }
    }
}

#[async_trait]
impl MemoryBackend for PyHostMemoryBackend {
    async fn put(&self, entry: StoredEntry) -> MemResult<()> {
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_entry = pythonize::pythonize(py, &entry).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "failed to pythonize StoredEntry: {e}"
                    ))
                })?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("put", (py_entry,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| MemoryError::Backend(format!("put dispatch failed: {e}")))?;

        pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| MemoryError::Backend(format!("put raised: {e}")))?;

        Ok(())
    }

    async fn get(&self, id: &str) -> MemResult<Option<StoredEntry>> {
        let id = id.to_owned();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("get", (id,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| MemoryError::Backend(format!("get dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| MemoryError::Backend(format!("get raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> MemResult<Option<StoredEntry>> {
                let bound = py_result.bind(py);
                if bound.is_none() {
                    return Ok(None);
                }
                let entry: StoredEntry = pythonize::depythonize(bound).map_err(|e| {
                    MemoryError::Backend(format!("failed to depythonize get result: {e}"))
                })?;
                Ok(Some(entry))
            })
        })
    }

    async fn delete(&self, id: &str) -> MemResult<bool> {
        let id = id.to_owned();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("delete", (id,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| MemoryError::Backend(format!("delete dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| MemoryError::Backend(format!("delete raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> MemResult<bool> {
                let bound = py_result.bind(py);
                let val: bool = pythonize::depythonize(bound).map_err(|e| {
                    MemoryError::Backend(format!("failed to depythonize delete result: {e}"))
                })?;
                Ok(val)
            })
        })
    }

    async fn list(&self) -> MemResult<Vec<StoredEntry>> {
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method0("list")?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| MemoryError::Backend(format!("list dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| MemoryError::Backend(format!("list raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> MemResult<Vec<StoredEntry>> {
                let bound = py_result.bind(py);
                let entries: Vec<StoredEntry> = pythonize::depythonize(bound).map_err(|e| {
                    MemoryError::Backend(format!("failed to depythonize list result: {e}"))
                })?;
                Ok(entries)
            })
        })
    }

    async fn len(&self) -> MemResult<usize> {
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method0("len")?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| MemoryError::Backend(format!("len dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| MemoryError::Backend(format!("len raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> MemResult<usize> {
                let bound = py_result.bind(py);
                let val: usize = pythonize::depythonize(bound).map_err(|e| {
                    MemoryError::Backend(format!("failed to depythonize len result: {e}"))
                })?;
                Ok(val)
            })
        })
    }

    async fn search_by_bands(&self, bands: &[String], limit: usize) -> MemResult<Vec<StoredEntry>> {
        let bands = bands.to_vec();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("search_by_bands", (bands, limit))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            MemoryError::Backend(format!("search_by_bands dispatch failed: {e}"))
        })?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| MemoryError::Backend(format!("search_by_bands raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> MemResult<Vec<StoredEntry>> {
                let bound = py_result.bind(py);
                let entries: Vec<StoredEntry> = pythonize::depythonize(bound).map_err(|e| {
                    MemoryError::Backend(format!(
                        "failed to depythonize search_by_bands result: {e}"
                    ))
                })?;
                Ok(entries)
            })
        })
    }
}

// ---------------------------------------------------------------------------
// PyInMemoryBackend
// ---------------------------------------------------------------------------

/// An in-memory storage backend for Memory.
///
/// Data lives only as long as the process -- nothing is persisted to disk.
/// Suitable for testing, prototyping, and short-lived processes.
///
/// Example:
///     >>> backend = InMemoryBackend()
///     >>> memory = Memory(embedder, backend)
#[gen_stub_pyclass]
#[pyclass(name = "InMemoryBackend", from_py_object)]
#[derive(Clone)]
pub struct PyInMemoryBackend {
    inner: Arc<InMemoryBackend>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryBackend::new()),
        }
    }

    fn __repr__(&self) -> &'static str {
        "InMemoryBackend()"
    }
}

// ---------------------------------------------------------------------------
// PyJsonlBackend
// ---------------------------------------------------------------------------

/// A JSONL file-backed storage backend for Memory.
///
/// Loads entries from the file on construction and appends new entries to
/// the file. Deletions trigger a full rewrite.
///
/// Example:
///     >>> backend = await JsonlBackend.create("memories.jsonl")
///     >>> memory = Memory(embedder, backend)
#[gen_stub_pyclass]
#[pyclass(name = "JsonlBackend", from_py_object)]
#[derive(Clone)]
pub struct PyJsonlBackend {
    inner: Arc<JsonlBackend>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyJsonlBackend {
    /// Create a new JSONL backend at the given file path.
    ///
    /// This is an async constructor because the file is loaded on creation.
    ///
    /// Args:
    ///     path: Path to the JSONL file. Created if it does not exist.
    ///
    /// Returns:
    ///     A new JsonlBackend instance.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, JsonlBackend]", imports = ("typing",)))]
    #[staticmethod]
    fn create(py: Python<'_>, path: String) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let backend = JsonlBackend::new(&path).await.map_err(memory_err)?;
            Ok(PyJsonlBackend {
                inner: Arc::new(backend),
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "JsonlBackend(...)"
    }
}

// ---------------------------------------------------------------------------
// PyValkeyBackend
// ---------------------------------------------------------------------------

/// A Valkey/Redis-backed storage backend for Memory.
///
/// Uses Redis-compatible commands for persistence and band-based search.
///
/// Example:
///     >>> backend = ValkeyBackend("redis://localhost:6379")
///     >>> memory = Memory(embedder, backend)
#[gen_stub_pyclass]
#[pyclass(name = "ValkeyBackend", from_py_object)]
#[derive(Clone)]
pub struct PyValkeyBackend {
    inner: Arc<ValkeyBackend>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyValkeyBackend {
    /// Create a new Valkey backend connected to the given URL.
    ///
    /// Args:
    ///     url: A Redis/Valkey connection URL, e.g. "redis://localhost:6379".
    ///     prefix: Optional key prefix for namespacing (default: "blazen:memory:").
    #[new]
    #[pyo3(signature = (url, prefix=None))]
    fn new(url: &str, prefix: Option<&str>) -> PyResult<Self> {
        let mut backend =
            ValkeyBackend::new(url).map_err(|e| PyErr::from(BlazenPyError::Llm(e.to_string())))?;
        if let Some(p) = prefix {
            backend = backend.with_prefix(p);
        }
        Ok(Self {
            inner: Arc::new(backend),
        })
    }

    fn __repr__(&self) -> &'static str {
        "ValkeyBackend(...)"
    }
}

// ---------------------------------------------------------------------------
// PyUpstashBackend
// ---------------------------------------------------------------------------

/// Upstash REST-backed memory backend (HTTPS over `Arc<dyn HttpClient>`).
///
/// Wasi-compatible alternative to ``ValkeyBackend`` for hosts that cannot
/// open raw TCP connections (Cloudflare Workers, Deno, etc.). On native
/// targets this still works — it just uses HTTPS instead of raw TCP.
///
/// Example:
///     >>> backend = UpstashBackend("https://us1-merry-cat-32242.upstash.io", "AYAg...")
#[gen_stub_pyclass]
#[pyclass(name = "UpstashBackend", from_py_object)]
#[derive(Clone)]
pub struct PyUpstashBackend {
    pub(crate) inner: Arc<UpstashBackend>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUpstashBackend {
    /// Create a new Upstash backend.
    ///
    /// Args:
    ///     rest_url: The Upstash REST endpoint, e.g.
    ///         "https://us1-merry-cat-32242.upstash.io".
    ///     rest_token: The Upstash REST token (sent as a Bearer token).
    ///     prefix: Optional key prefix for namespacing
    ///         (default: "blazen:memory:").
    #[new]
    #[pyo3(signature = (rest_url, rest_token, prefix=None))]
    fn new(rest_url: String, rest_token: String, prefix: Option<String>) -> Self {
        let http = default_http_client();
        let mut backend = UpstashBackend::new(rest_url, rest_token, http);
        if let Some(p) = prefix {
            backend = backend.with_prefix(p);
        }
        Self {
            inner: Arc::new(backend),
        }
    }

    fn __repr__(&self) -> &'static str {
        "UpstashBackend(...)"
    }
}

// ---------------------------------------------------------------------------
// PyRetryMemoryBackend
// ---------------------------------------------------------------------------

/// A `MemoryBackend` decorator that retries transient errors with exponential
/// backoff.
///
/// Mirrors :class:`RetryCompletionModel` for `CompletionModel`. Wraps any
/// `MemoryBackend` (subclass or built-in) and reissues each `put`/`get`/
/// `delete`/`list`/`len`/`search_by_bands` call up to `config.max_retries`
/// times when the underlying backend raises.
///
/// Example:
///     >>> backend = InMemoryBackend()
///     >>> retried = RetryMemoryBackend(backend, RetryConfig(max_retries=5))
///     >>> memory = Memory(embedder, retried)
#[gen_stub_pyclass]
#[pyclass(name = "RetryMemoryBackend", from_py_object)]
#[derive(Clone)]
pub struct PyRetryMemoryBackend {
    pub(crate) inner: Arc<dyn MemoryBackend>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryMemoryBackend {
    /// Wrap a `MemoryBackend` with automatic retry on transient failures.
    ///
    /// Args:
    ///     backend: The `MemoryBackend` to wrap (any built-in or subclass).
    ///     config: Optional typed `RetryConfig`. Defaults to
    ///         `RetryConfig()` (3 retries, 1s initial, 30s max).
    #[new]
    #[pyo3(signature = (backend, config=None))]
    fn new(
        backend: &Bound<'_, PyAny>,
        config: Option<PyRef<'_, crate::providers::config::PyRetryConfig>>,
    ) -> PyResult<Self> {
        let inner_backend = extract_backend(backend)?;
        let retry_config: RetryConfig = config.map(|c| c.inner.clone()).unwrap_or_default();
        let wrapped = RetryMemoryBackend::new(inner_backend, retry_config);
        Ok(Self {
            inner: wrapped.into_arc(),
        })
    }

    fn __repr__(&self) -> &'static str {
        "RetryMemoryBackend(...)"
    }
}

// ---------------------------------------------------------------------------
// PyStoredEntry
// ---------------------------------------------------------------------------

/// An entry as persisted in a backend, including ELID and LSH band data.
///
/// `StoredEntry` is the on-disk / on-the-wire shape used by all
/// `MemoryBackend` implementations. Custom backends written in Python
/// receive and return these objects.
///
/// Example:
///     >>> entry = StoredEntry(
///     ...     id="doc1",
///     ...     text="Paris is the capital of France",
///     ...     text_simhash=0xDEADBEEF,
///     ...     bands=["b0:0001", "b1:002a"],
///     ...     metadata={"category": "geo"},
///     ... )
///     >>> entry.id
///     'doc1'
#[gen_stub_pyclass]
#[pyclass(name = "StoredEntry", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStoredEntry {
    pub(crate) inner: StoredEntry,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStoredEntry {
    /// Construct a `StoredEntry`.
    ///
    /// Args:
    ///     id: Unique identifier for this entry.
    ///     text: The original text content.
    ///     text_simhash: 64-bit `SimHash` of the raw text (used by local search).
    ///     elid: Optional ELID-encoded embedding string.
    ///     simhash_hex: Optional 128-bit embedding `SimHash` as a 32-char hex string.
    ///     bands: LSH band strings derived from the embedding `SimHash`.
    ///     metadata: Arbitrary user metadata.
    #[new]
    #[pyo3(signature = (
        *,
        id,
        text,
        text_simhash = 0,
        elid = None,
        simhash_hex = None,
        bands = Vec::new(),
        metadata = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        id: String,
        text: String,
        text_simhash: u64,
        elid: Option<String>,
        simhash_hex: Option<String>,
        bands: Vec<String>,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let metadata_value = match metadata {
            Some(obj) => crate::convert::py_to_json(py, obj.bind(py))?,
            None => serde_json::Value::Null,
        };
        Ok(Self {
            inner: StoredEntry {
                id,
                text,
                elid,
                simhash_hex,
                text_simhash,
                bands,
                metadata: metadata_value,
            },
        })
    }

    /// Unique identifier for this entry.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// The original text content.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// ELID-encoded embedding string (None when running in local-only mode).
    #[getter]
    fn elid(&self) -> Option<&str> {
        self.inner.elid.as_deref()
    }

    /// 128-bit `SimHash` of the embedding, as a 32-char hex string.
    #[getter]
    fn simhash_hex(&self) -> Option<&str> {
        self.inner.simhash_hex.as_deref()
    }

    /// String-level `SimHash` of the raw text (always available, for local search).
    #[getter]
    fn text_simhash(&self) -> u64 {
        self.inner.text_simhash
    }

    /// LSH band strings derived from the embedding `SimHash`.
    #[getter]
    fn bands(&self) -> Vec<String> {
        self.inner.bands.clone()
    }

    /// Arbitrary user metadata (returned as a Python dict / value).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "StoredEntry(id='{}', text='{}', text_simhash={}, bands={} entries)",
            self.inner.id,
            truncate_text(&self.inner.text, 40),
            self.inner.text_simhash,
            self.inner.bands.len(),
        )
    }
}

impl From<StoredEntry> for PyStoredEntry {
    fn from(inner: StoredEntry) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// Free functions: similarity helpers and SimHash hex conversion
// ---------------------------------------------------------------------------

/// Compute similarity between two 64-bit text `SimHashes`.
///
/// Uses Hamming distance normalized over 64 bits: returns a value in
/// ``[0.0, 1.0]`` where ``1.0`` means the hashes are identical.
///
/// Args:
///     a: First 64-bit `SimHash` value.
///     b: Second 64-bit `SimHash` value.
///
/// Returns:
///     Similarity score in ``[0.0, 1.0]``.
#[gen_stub_pyfunction]
#[pyfunction]
#[must_use]
pub fn compute_text_simhash_similarity(a: u64, b: u64) -> f64 {
    search::compute_text_simhash_similarity(a, b)
}

/// Compute similarity between two 128-bit embedding `SimHashes`.
///
/// Each input is provided as a 32-char hex string (matching the
/// ``simhash_hex`` field on `StoredEntry`). Uses Hamming distance
/// normalized over 128 bits.
///
/// Args:
///     a: First 128-bit `SimHash` as a hex string.
///     b: Second 128-bit `SimHash` as a hex string.
///
/// Returns:
///     Similarity score in ``[0.0, 1.0]``.
///
/// Raises:
///     ValueError: If either argument is not a valid 128-bit hex string.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_embedding_simhash_similarity(a: &str, b: &str) -> PyResult<f64> {
    let av = search::simhash_from_hex(a)
        .ok_or_else(|| PyValueError::new_err(format!("invalid 128-bit hex SimHash: {a:?}")))?;
    let bv = search::simhash_from_hex(b)
        .ok_or_else(|| PyValueError::new_err(format!("invalid 128-bit hex SimHash: {b:?}")))?;
    Ok(search::compute_embedding_simhash_similarity(av, bv))
}

/// Compute similarity between two ELID strings.
///
/// Decodes both ELIDs and computes the Hamming distance between their
/// Mini128 payloads, normalized into ``[0.0, 1.0]``.
///
/// Args:
///     a: First ELID string.
///     b: Second ELID string.
///
/// Returns:
///     Similarity score in ``[0.0, 1.0]``.
///
/// Raises:
///     MemoryError: If either string is not a valid Mini128 ELID.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_elid_similarity(a: &str, b: &str) -> PyResult<f64> {
    search::compute_elid_similarity(a, b).map_err(memory_err)
}

/// Parse a hex-encoded 128-bit `SimHash` back into an integer.
///
/// Python ints are arbitrary-precision so the full 128-bit value is
/// returned without truncation.
///
/// Args:
///     hex_str: A 32-char hex-encoded `SimHash`.
///
/// Returns:
///     The decoded `SimHash` as a Python int.
///
/// Raises:
///     ValueError: If `hex_str` is not a valid 128-bit hex string.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn simhash_from_hex(hex_str: &str) -> PyResult<u128> {
    search::simhash_from_hex(hex_str)
        .ok_or_else(|| PyValueError::new_err(format!("invalid 128-bit hex SimHash: {hex_str:?}")))
}

/// Encode a 128-bit `SimHash` integer as a zero-padded 32-char hex string.
///
/// Args:
///     value: A 128-bit `SimHash` as a Python int.
///
/// Returns:
///     A zero-padded 32-character lowercase hex string.
///
/// Raises:
///     OverflowError: If `value` does not fit in 128 bits.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn simhash_to_hex(value: u128) -> String {
    search::simhash_to_hex(value)
}

// ---------------------------------------------------------------------------
// PyMemoryResult
// ---------------------------------------------------------------------------

/// A single search result from a Memory search.
///
/// Example:
///     >>> results = await memory.search("France capital", limit=5)
///     >>> results[0].id
///     'doc1'
///     >>> results[0].text
///     'Paris is the capital of France'
///     >>> results[0].score
///     0.85
#[gen_stub_pyclass]
#[pyclass(name = "MemoryResult", frozen)]
pub struct PyMemoryResult {
    inner: MemoryResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMemoryResult {
    /// The entry's unique identifier.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// The original text content.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Similarity score in [0.0, 1.0], higher means more similar.
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    /// Arbitrary user metadata (returned as a Python dict/value).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryResult(id='{}', score={:.4}, text='{}')",
            self.inner.id,
            self.inner.score,
            truncate_text(&self.inner.text, 50),
        )
    }
}

// ---------------------------------------------------------------------------
// PyMemory
// ---------------------------------------------------------------------------

/// A memory store with optional embedding-based semantic search.
///
/// When created with an EmbeddingModel (``Memory(embedder, backend)``),
/// both ``search()`` (semantic) and ``search_local()`` (SimHash) are available.
///
/// When created in local mode (``Memory.local(backend)``), only
/// ``search_local()`` is available -- no embedding model is required.
///
/// Example:
///     >>> embedder = EmbeddingModel.openai(options=ProviderOptions(api_key=key))
///     >>> memory = Memory(embedder, InMemoryBackend())
///     >>> await memory.add("doc1", "Paris is the capital of France")
///     >>> results = await memory.search("France capital", limit=5)
///     >>> print(results[0].text)
#[gen_stub_pyclass]
#[pyclass(name = "Memory")]
pub struct PyMemory {
    inner: Arc<Memory>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMemory {
    /// Create a Memory with an embedding model for full semantic search.
    ///
    /// Args:
    ///     embedder: An EmbeddingModel instance (built-in or Python subclass).
    ///     backend: A backend instance (InMemoryBackend, JsonlBackend, or ValkeyBackend).
    #[new]
    fn new(embedder: Bound<'_, PyEmbeddingModel>, backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        let arc_backend = extract_backend(backend)?;
        let emb: Arc<dyn blazen_llm::EmbeddingModel> = {
            let embedder_borrow = embedder.borrow();
            if let Some(ref inner) = embedder_borrow.inner {
                inner.clone()
            } else {
                // Subclassed Python `EmbeddingModel`: wrap the Python object
                // in an adapter that dispatches `embed()` back into Python.
                let model_id = embedder_borrow
                    .config
                    .as_ref()
                    .and_then(|c| c.model_id.clone())
                    .unwrap_or_else(|| "subclass".to_owned());
                let dimensions = embedder_borrow.dimensions_override.unwrap_or(0);
                drop(embedder_borrow);
                let py_obj: Py<PyAny> = embedder.clone().into_any().unbind();
                Arc::new(crate::types::embedding::PySubclassEmbeddingModel::new(
                    py_obj, model_id, dimensions,
                ))
            }
        };
        let memory = Memory::new_arc(emb, arc_backend);
        Ok(Self {
            inner: Arc::new(memory),
        })
    }

    /// Create a Memory in local-only mode (no embedding model).
    ///
    /// Only ``search_local()`` is available; ``search()`` will raise an error.
    ///
    /// Args:
    ///     backend: A backend instance (InMemoryBackend, JsonlBackend, or ValkeyBackend).
    #[staticmethod]
    fn local(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        let arc_backend = extract_backend(backend)?;
        let memory = Memory::local_arc(arc_backend);
        Ok(Self {
            inner: Arc::new(memory),
        })
    }

    /// Add a document to the memory store.
    ///
    /// Args:
    ///     id: A unique identifier for the document. If empty, a UUID is generated.
    ///     text: The text content to store.
    ///     metadata: Optional metadata dict to attach.
    ///
    /// Returns:
    ///     The id of the stored document.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.str]", imports = ("typing", "builtins")))]
    #[pyo3(signature = (id, text, metadata=None))]
    fn add<'py>(
        &self,
        py: Python<'py>,
        id: String,
        text: String,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        let meta = match metadata {
            Some(obj) => crate::convert::py_to_json(py, obj.bind(py))?,
            None => serde_json::Value::Null,
        };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let entry = MemoryEntry {
                id,
                text,
                metadata: meta,
            };
            let ids = memory.add(vec![entry]).await.map_err(memory_err)?;
            Ok(ids.into_iter().next().unwrap_or_default())
        })
    }

    /// Add multiple documents to the memory store in a single batch.
    ///
    /// Args:
    ///     entries: A list of dicts, each with "id" (optional), "text", and "metadata" (optional) keys.
    ///
    /// Returns:
    ///     A list of ids for the stored documents.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, list[builtins.str]]", imports = ("typing", "builtins")))]
    fn add_many<'py>(
        &self,
        py: Python<'py>,
        entries: Vec<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        // Convert Python dicts to MemoryEntry objects.
        let mut rust_entries = Vec::with_capacity(entries.len());
        for obj in &entries {
            let bound = obj.bind(py);
            let dict: &Bound<'_, PyDict> = bound
                .cast()
                .map_err(|_| PyTypeError::new_err("each entry must be a dict"))?;

            let text: String = dict
                .get_item("text")?
                .ok_or_else(|| PyTypeError::new_err("each entry must have a 'text' key"))?
                .extract()?;

            let id: String = dict
                .get_item("id")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_default();

            let metadata = dict
                .get_item("metadata")?
                .map(|v| crate::convert::py_to_json(py, &v))
                .transpose()?
                .unwrap_or(serde_json::Value::Null);

            rust_entries.push(MemoryEntry { id, text, metadata });
        }
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ids = memory.add(rust_entries).await.map_err(memory_err)?;
            Ok(ids)
        })
    }

    /// Semantic search using the configured embedding model.
    ///
    /// Requires an embedding model to be configured (i.e. not local-only mode).
    ///
    /// Args:
    ///     query: The search query string.
    ///     limit: Maximum number of results to return (default: 5).
    ///     metadata_filter: Optional dict of key-value pairs to filter results.
    ///         Only entries whose metadata is a superset of the filter are
    ///         returned. For example, ``{"category": "news"}`` matches entries
    ///         that have at least ``category: "news"`` in their metadata.
    ///
    /// Returns:
    ///     A list of MemoryResult objects sorted by descending similarity.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, list[MemoryResult]]", imports = ("typing",)))]
    #[pyo3(signature = (query, limit=5, metadata_filter=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: usize,
        metadata_filter: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        let filter = match metadata_filter {
            Some(obj) => Some(crate::convert::py_to_json(py, obj.bind(py))?),
            None => None,
        };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = memory
                .search(&query, limit, filter.as_ref())
                .await
                .map_err(memory_err)?;
            let py_results: Vec<PyMemoryResult> = results
                .into_iter()
                .map(|r| PyMemoryResult { inner: r })
                .collect();
            Ok(py_results)
        })
    }

    /// Local SimHash-based search (no embedding model required).
    ///
    /// Uses string-level SimHash for lightweight similarity search.
    /// Works in both full and local-only modes.
    ///
    /// Args:
    ///     query: The search query string.
    ///     limit: Maximum number of results to return (default: 5).
    ///     metadata_filter: Optional dict of key-value pairs to filter results.
    ///         Only entries whose metadata is a superset of the filter are
    ///         returned. See ``search()`` for details.
    ///
    /// Returns:
    ///     A list of MemoryResult objects sorted by descending similarity.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, list[MemoryResult]]", imports = ("typing",)))]
    #[pyo3(signature = (query, limit=5, metadata_filter=None))]
    fn search_local<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: usize,
        metadata_filter: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        let filter = match metadata_filter {
            Some(obj) => Some(crate::convert::py_to_json(py, obj.bind(py))?),
            None => None,
        };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = memory
                .search_local(&query, limit, filter.as_ref())
                .await
                .map_err(memory_err)?;
            let py_results: Vec<PyMemoryResult> = results
                .into_iter()
                .map(|r| PyMemoryResult { inner: r })
                .collect();
            Ok(py_results)
        })
    }

    /// Retrieve a single entry by its id.
    ///
    /// Args:
    ///     id: The document id.
    ///
    /// Returns:
    ///     A dict with the entry data, or None if not found.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.Optional[dict[str, typing.Any]]]", imports = ("typing",)))]
    fn get<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let entry = memory.get(&id).await.map_err(memory_err)?;
            match entry {
                Some(e) => {
                    let py_dict: Py<PyAny> = Python::try_attach(|py| -> PyResult<Py<PyAny>> {
                        let dict = PyDict::new(py);
                        dict.set_item("id", &e.id)?;
                        dict.set_item("text", &e.text)?;
                        dict.set_item("metadata", crate::convert::json_to_py(py, &e.metadata)?)?;
                        Ok(dict.unbind().into_any())
                    })
                    .ok_or_else(|| {
                        PyErr::from(BlazenPyError::Llm(
                            "failed to acquire Python GIL".to_string(),
                        ))
                    })??;
                    Ok(Some(py_dict))
                }
                None => Ok(None),
            }
        })
    }

    /// Delete a document by its id.
    ///
    /// Args:
    ///     id: The document id to delete.
    ///
    /// Returns:
    ///     True if the document existed and was deleted, False otherwise.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.bool]", imports = ("typing", "builtins")))]
    fn delete<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let deleted = memory.delete(&id).await.map_err(memory_err)?;
            Ok(deleted)
        })
    }

    /// Return the number of documents in the store.
    ///
    /// Returns:
    ///     The document count.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.int]", imports = ("typing", "builtins")))]
    fn count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let memory = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let n = memory.len().await.map_err(memory_err)?;
            Ok(n)
        })
    }

    fn __repr__(&self) -> &'static str {
        "Memory(...)"
    }
}

/// Truncate text for display purposes.
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

/// Resolve the platform-appropriate default HTTP client. Mirrors the
/// `default_http_client` helper in `blazen-node`'s `peer_http` module.
#[cfg(target_os = "wasi")]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::http_napi_wasi::LazyHttpClient::new().into_arc()
}

#[cfg(not(target_os = "wasi"))]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::ReqwestHttpClient::new().into_arc()
}
