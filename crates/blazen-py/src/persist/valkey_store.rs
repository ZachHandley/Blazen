//! Python wrapper for the Valkey/Redis-backed checkpoint store.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_persist::CheckpointStore;
use blazen_persist::valkey::ValkeyCheckpointStore;

use crate::persist::checkpoint::PyWorkflowCheckpoint;
use crate::persist::error::persist_err;
use crate::persist::store::parse_run_id;

/// A Valkey/Redis-backed checkpoint store.
///
/// Constructed via :meth:`connect` (async) because the underlying connection
/// manager requires an active runtime. Optionally accepts a TTL in seconds
/// after which keys auto-expire.
///
/// Example:
///     >>> store = await ValkeyCheckpointStore.connect("redis://localhost:6379")
///     >>> # Or with a 24-hour TTL on saved checkpoints:
///     >>> store = await ValkeyCheckpointStore.connect("redis://localhost:6379", ttl_seconds=86400)
#[gen_stub_pyclass]
#[pyclass(name = "ValkeyCheckpointStore")]
pub struct PyValkeyCheckpointStore {
    pub(crate) inner: Arc<ValkeyCheckpointStore>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyValkeyCheckpointStore {
    /// Connect to a Valkey/Redis server at ``url``.
    ///
    /// Args:
    ///     url: A Redis/Valkey connection URL such as ``redis://host:port/db``
    ///         (or ``rediss://`` for TLS).
    ///     ttl_seconds: Optional TTL applied to every saved checkpoint.
    ///
    /// This is an async classmethod because the initial connection is established eagerly.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ValkeyCheckpointStore]", imports = ("typing",)))]
    #[staticmethod]
    #[pyo3(signature = (url, ttl_seconds=None))]
    fn connect(
        py: Python<'_>,
        url: String,
        ttl_seconds: Option<u64>,
    ) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = if let Some(ttl) = ttl_seconds {
                ValkeyCheckpointStore::with_ttl(&url, ttl)
                    .await
                    .map_err(persist_err)?
            } else {
                ValkeyCheckpointStore::new(&url)
                    .await
                    .map_err(persist_err)?
            };
            Ok(PyValkeyCheckpointStore {
                inner: Arc::new(store),
            })
        })
    }

    /// Persist a checkpoint, overwriting any existing entry with the same ``run_id``.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn save<'py>(
        &self,
        py: Python<'py>,
        checkpoint: PyRef<'_, PyWorkflowCheckpoint>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        let cp = checkpoint.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store.save(&cp).await.map_err(persist_err)?;
            Ok(())
        })
    }

    /// Load a checkpoint by its ``run_id`` (UUID string).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.Optional[WorkflowCheckpoint]]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>, run_id: String) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        let id = parse_run_id(&run_id)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = store.load(&id).await.map_err(persist_err)?;
            Ok(result.map(PyWorkflowCheckpoint::from_inner))
        })
    }

    /// List all stored checkpoints, ordered by timestamp descending.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, list[WorkflowCheckpoint]]", imports = ("typing",)))]
    fn list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let items = store.list().await.map_err(persist_err)?;
            let py_items: Vec<PyWorkflowCheckpoint> = items
                .into_iter()
                .map(PyWorkflowCheckpoint::from_inner)
                .collect();
            Ok(py_items)
        })
    }

    /// Delete the checkpoint for the given ``run_id`` (UUID string).
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn delete<'py>(&self, py: Python<'py>, run_id: String) -> PyResult<Bound<'py, PyAny>> {
        let store = self.inner.clone();
        let id = parse_run_id(&run_id)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store.delete(&id).await.map_err(persist_err)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> &'static str {
        "ValkeyCheckpointStore(...)"
    }
}
