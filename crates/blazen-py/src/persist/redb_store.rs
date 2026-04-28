//! Python wrapper for the redb-backed checkpoint store.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_persist::{CheckpointStore, RedbCheckpointStore};

use crate::persist::checkpoint::PyWorkflowCheckpoint;
use crate::persist::error::persist_err;
use crate::persist::store::parse_run_id;

/// A redb-backed checkpoint store.
///
/// Stores workflow checkpoints in an embedded ACID key-value file at the
/// given path. The file is created if it does not exist.
///
/// Example:
///     >>> store = RedbCheckpointStore("workflow.db")
///     >>> await store.save(checkpoint)
///     >>> loaded = await store.load(checkpoint.run_id)
#[gen_stub_pyclass]
#[pyclass(name = "RedbCheckpointStore")]
pub struct PyRedbCheckpointStore {
    pub(crate) inner: Arc<RedbCheckpointStore>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRedbCheckpointStore {
    /// Create or open a redb-backed store at ``path``.
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let store = RedbCheckpointStore::new(&path).map_err(persist_err)?;
        Ok(Self {
            inner: Arc::new(store),
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
        "RedbCheckpointStore(...)"
    }
}
