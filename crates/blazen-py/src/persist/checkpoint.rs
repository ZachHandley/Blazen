//! Python wrappers for `blazen-persist` data types.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_persist::{SerializedEvent, WorkflowCheckpoint};

use crate::convert::json_to_py;

/// A serialized representation of a queued event captured in a checkpoint.
///
/// Renamed in Python from the Rust ``SerializedEvent`` to avoid a naming
/// collision with the ``SerializedEvent`` type produced by ``blazen-core``
/// for workflow event streams.
///
/// Example:
///     >>> ev = PersistedEvent(event_type="blazen::StartEvent", data={"input": "hi"})
///     >>> ev.event_type
///     'blazen::StartEvent'
#[gen_stub_pyclass]
#[pyclass(name = "PersistedEvent", frozen)]
pub struct PyPersistedEvent {
    pub(crate) inner: SerializedEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPersistedEvent {
    /// Construct a new persisted event.
    #[new]
    #[pyo3(signature = (event_type, data=None))]
    fn new(py: Python<'_>, event_type: String, data: Option<Py<PyAny>>) -> PyResult<Self> {
        let data = match data {
            Some(obj) => crate::convert::py_to_json(py, obj.bind(py))?,
            None => serde_json::Value::Null,
        };
        Ok(Self {
            inner: SerializedEvent { event_type, data },
        })
    }

    /// The event type identifier (e.g. ``"blazen::StartEvent"``).
    #[getter]
    fn event_type(&self) -> &str {
        &self.inner.event_type
    }

    /// The event data as a Python value (decoded from JSON).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.data)
    }

    fn __repr__(&self) -> String {
        format!("PersistedEvent(event_type={:?})", self.inner.event_type)
    }
}

impl PyPersistedEvent {
    pub(crate) fn from_inner(inner: SerializedEvent) -> Self {
        Self { inner }
    }
}

/// A snapshot of a workflow's state at a point in time.
///
/// Use a [`CheckpointStore`] backend to ``save``/``load``/``list``/``delete``
/// these snapshots for crash recovery and pause/resume.
///
/// Example:
///     >>> import uuid
///     >>> from datetime import datetime, timezone
///     >>> cp = WorkflowCheckpoint(
///     ...     workflow_name="demo",
///     ...     run_id=str(uuid.uuid4()),
///     ...     timestamp=datetime.now(timezone.utc).isoformat(),
///     ...     state={"counter": 1},
///     ...     pending_events=[],
///     ...     metadata={},
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "WorkflowCheckpoint", frozen)]
pub struct PyWorkflowCheckpoint {
    pub(crate) inner: WorkflowCheckpoint,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWorkflowCheckpoint {
    /// Construct a new workflow checkpoint.
    ///
    /// Args:
    ///     workflow_name: The name of the workflow.
    ///     run_id: The unique run id (UUID string). If empty, a new UUID is generated.
    ///     timestamp: An ISO-8601 / RFC-3339 timestamp string. If empty, ``now`` (UTC) is used.
    ///     state: Optional dict mapping state keys to JSON-serialisable values.
    ///     pending_events: Optional list of ``PersistedEvent`` instances representing the queue.
    ///     metadata: Optional dict of arbitrary metadata.
    #[new]
    #[pyo3(signature = (
        workflow_name,
        run_id = String::new(),
        timestamp = String::new(),
        state = None,
        pending_events = None,
        metadata = None,
    ))]
    fn new(
        py: Python<'_>,
        workflow_name: String,
        run_id: String,
        timestamp: String,
        state: Option<Py<PyAny>>,
        pending_events: Option<Vec<Py<PyAny>>>,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let run_id = if run_id.is_empty() {
            uuid::Uuid::new_v4()
        } else {
            uuid::Uuid::parse_str(&run_id).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid run_id UUID: {e}"))
            })?
        };

        let timestamp = if timestamp.is_empty() {
            chrono::Utc::now()
        } else {
            chrono::DateTime::parse_from_rfc3339(&timestamp)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid timestamp (expected RFC-3339): {e}"
                    ))
                })?
                .with_timezone(&chrono::Utc)
        };

        let state_map: HashMap<String, serde_json::Value> = match state {
            Some(obj) => {
                let val = crate::convert::py_to_json(py, obj.bind(py))?;
                serde_json::from_value(val).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "state must be a dict[str, Any]: {e}"
                    ))
                })?
            }
            None => HashMap::new(),
        };

        let pending: Vec<SerializedEvent> = match pending_events {
            Some(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    let bound = item.bind(py);
                    if let Ok(ev) = bound.extract::<PyRef<'_, PyPersistedEvent>>() {
                        out.push(ev.inner.clone());
                    } else {
                        let dict: &Bound<'_, PyDict> = bound.cast().map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "pending_events entries must be PersistedEvent or dict",
                            )
                        })?;
                        let event_type: String = dict
                            .get_item("event_type")?
                            .ok_or_else(|| {
                                pyo3::exceptions::PyTypeError::new_err(
                                    "each pending_events entry must have 'event_type'",
                                )
                            })?
                            .extract()?;
                        let data = match dict.get_item("data")? {
                            Some(v) => crate::convert::py_to_json(py, &v)?,
                            None => serde_json::Value::Null,
                        };
                        out.push(SerializedEvent { event_type, data });
                    }
                }
                out
            }
            None => Vec::new(),
        };

        let metadata_map: HashMap<String, serde_json::Value> = match metadata {
            Some(obj) => {
                let val = crate::convert::py_to_json(py, obj.bind(py))?;
                serde_json::from_value(val).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "metadata must be a dict[str, Any]: {e}"
                    ))
                })?
            }
            None => HashMap::new(),
        };

        Ok(Self {
            inner: WorkflowCheckpoint {
                workflow_name,
                run_id,
                timestamp,
                state: state_map,
                pending_events: pending,
                metadata: metadata_map,
            },
        })
    }

    /// The name of the workflow that produced this checkpoint.
    #[getter]
    fn workflow_name(&self) -> &str {
        &self.inner.workflow_name
    }

    /// The unique run id, formatted as a UUID string.
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// The checkpoint timestamp as an RFC-3339 string.
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    /// The state map (key -> JSON-decoded Python value).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.state {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.unbind().into_any())
    }

    /// The list of pending events captured at checkpoint time.
    #[getter]
    fn pending_events(&self) -> Vec<PyPersistedEvent> {
        self.inner
            .pending_events
            .iter()
            .cloned()
            .map(PyPersistedEvent::from_inner)
            .collect()
    }

    /// The metadata map (key -> JSON-decoded Python value).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.metadata {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.unbind().into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkflowCheckpoint(workflow_name={:?}, run_id={:?}, timestamp={:?})",
            self.inner.workflow_name,
            self.inner.run_id.to_string(),
            self.inner.timestamp.to_rfc3339(),
        )
    }
}

impl PyWorkflowCheckpoint {
    pub(crate) fn from_inner(inner: WorkflowCheckpoint) -> Self {
        Self { inner }
    }
}
