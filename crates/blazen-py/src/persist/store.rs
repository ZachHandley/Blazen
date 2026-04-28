//! Subclassable Python checkpoint store and host-side adapter.

use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use uuid::Uuid;

use blazen_persist::{CheckpointStore, PersistError, WorkflowCheckpoint};

use crate::persist::checkpoint::PyWorkflowCheckpoint;
use crate::persist::redb_store::PyRedbCheckpointStore;
use crate::persist::valkey_store::PyValkeyCheckpointStore;

/// Attempt to extract any registered checkpoint-store backend from a Python object.
///
/// Recognises built-in concrete stores plus any Python subclass of
/// [`PyCheckpointStore`] (which is wrapped in a host-dispatch adapter).
#[allow(dead_code)]
pub(crate) fn extract_store(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn CheckpointStore>> {
    if let Ok(s) = obj.extract::<PyRef<'_, PyRedbCheckpointStore>>() {
        return Ok(s.inner.clone());
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PyValkeyCheckpointStore>>() {
        return Ok(s.inner.clone());
    }
    if obj.is_instance_of::<PyCheckpointStore>() {
        return Ok(Arc::new(PyHostCheckpointStore::new(obj.clone().unbind())));
    }
    Err(PyTypeError::new_err(
        "expected RedbCheckpointStore, ValkeyCheckpointStore, or CheckpointStore subclass",
    ))
}

/// Base class for custom workflow checkpoint stores.
///
/// Subclass and override all four methods (``save``, ``load``, ``list``,
/// ``delete``) to implement a custom backend (e.g. Postgres, S3, DynamoDB).
/// Methods must be ``async``.
///
/// Example:
///     >>> class PostgresCheckpointStore(CheckpointStore):
///     ...     async def save(self, checkpoint: WorkflowCheckpoint) -> None: ...
///     ...     async def load(self, run_id: str) -> WorkflowCheckpoint | None: ...
///     ...     async def list(self) -> list[WorkflowCheckpoint]: ...
///     ...     async def delete(self, run_id: str) -> None: ...
#[gen_stub_pyclass]
#[pyclass(name = "CheckpointStore", subclass)]
pub struct PyCheckpointStore {}

#[gen_stub_pymethods]
#[pymethods]
impl PyCheckpointStore {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Persist a checkpoint, overwriting any existing entry with the same ``run_id``.
    fn save(&self, _checkpoint: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override save()",
        ))
    }

    /// Load a checkpoint by its ``run_id`` (UUID string). Return ``None`` if absent.
    fn load(&self, _run_id: String) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override load()",
        ))
    }

    /// List all stored checkpoints, ordered by timestamp descending (most recent first).
    fn list(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override list()",
        ))
    }

    /// Delete the checkpoint for the given ``run_id`` (UUID string).
    fn delete(&self, _run_id: String) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override delete()",
        ))
    }
}

/// Adapter that implements the Rust [`CheckpointStore`] trait by calling
/// back into a Python subclass instance.
pub struct PyHostCheckpointStore {
    py_obj: Py<PyAny>,
}

impl PyHostCheckpointStore {
    pub fn new(py_obj: Py<PyAny>) -> Self {
        Self { py_obj }
    }
}

impl std::fmt::Debug for PyHostCheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyHostCheckpointStore").finish()
    }
}

#[async_trait]
impl CheckpointStore for PyHostCheckpointStore {
    async fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<(), PersistError> {
        let cp = checkpoint.clone();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let py_cp = Py::new(py, PyWorkflowCheckpoint::from_inner(cp))?;
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("save", (py_cp,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| PersistError::Storage(format!("save dispatch failed: {e}")))?;

        pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| PersistError::Storage(format!("save raised: {e}")))?;

        Ok(())
    }

    async fn load(&self, run_id: &Uuid) -> Result<Option<WorkflowCheckpoint>, PersistError> {
        let run_id_str = run_id.to_string();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("load", (run_id_str,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| PersistError::Storage(format!("load dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| PersistError::Storage(format!("load raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<Option<WorkflowCheckpoint>, PersistError> {
                let bound = py_result.bind(py);
                if bound.is_none() {
                    return Ok(None);
                }
                let cp_ref: PyRef<'_, PyWorkflowCheckpoint> = bound.extract().map_err(|e| {
                    PersistError::Storage(format!(
                        "load() must return WorkflowCheckpoint or None: {e}"
                    ))
                })?;
                Ok(Some(cp_ref.inner.clone()))
            })
        })
    }

    async fn list(&self) -> Result<Vec<WorkflowCheckpoint>, PersistError> {
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method0("list")?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| PersistError::Storage(format!("list dispatch failed: {e}")))?;

        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| PersistError::Storage(format!("list raised: {e}")))?;

        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<Vec<WorkflowCheckpoint>, PersistError> {
                let bound = py_result.bind(py);
                let items = bound.try_iter().map_err(|e| {
                    PersistError::Storage(format!("list() must return an iterable: {e}"))
                })?;
                let mut out = Vec::new();
                for item in items {
                    let item = item.map_err(|e| {
                        PersistError::Storage(format!("list() iteration failed: {e}"))
                    })?;
                    let cp_ref: PyRef<'_, PyWorkflowCheckpoint> = item.extract().map_err(|e| {
                        PersistError::Storage(format!(
                            "list() entry must be WorkflowCheckpoint: {e}"
                        ))
                    })?;
                    out.push(cp_ref.inner.clone());
                }
                Ok(out)
            })
        })
    }

    async fn delete(&self, run_id: &Uuid) -> Result<(), PersistError> {
        let run_id_str = run_id.to_string();
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let host = self.py_obj.bind(py);
                let coro = host.call_method1("delete", (run_id_str,))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro)?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| PersistError::Storage(format!("delete dispatch failed: {e}")))?;

        pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| PersistError::Storage(format!("delete raised: {e}")))?;

        Ok(())
    }
}

/// Parse a UUID string to a [`Uuid`], returning a Python ``ValueError`` on failure.
pub(crate) fn parse_run_id(run_id: &str) -> PyResult<Uuid> {
    Uuid::parse_str(run_id)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid run_id UUID: {e}")))
}
