//! Python wrappers for core control-plane value types
//! (`WorkerCapability`, `AdmissionMode`, `ResourceHint`, `RunStatus`)
//! and small helpers that convert them into the dict / list shapes
//! returned by [`super::client::PyControlPlaneClient`].
//!
//! The control plane wire types (`Assignment`, `RunEvent`,
//! `RunStateSnapshot`, `WorkerInfo`) are surfaced to Python as plain
//! dicts rather than dedicated `#[pyclass]` types. Dicts are the
//! idiomatic Python choice for "read-only data with a few fields you
//! might key into", they round-trip cleanly through JSON, and they
//! match how the rest of `blazen-py` already exposes streaming
//! payloads (e.g. workflow events).

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_core::distributed::{
    AdmissionMode, AdmissionSnapshot, ResourceHint, RunEvent, RunStateSnapshot, RunStatus,
    WorkerCapability, WorkerInfo,
};

use crate::convert::json_to_py;

// ===========================================================================
// PyControlPlaneWorkerCapability
// ===========================================================================

/// A `kind`/`version` pair advertised by a worker.
#[gen_stub_pyclass]
#[pyclass(name = "WorkerCapability", from_py_object)]
#[derive(Clone)]
pub struct PyControlPlaneWorkerCapability {
    pub(crate) inner: WorkerCapability,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyControlPlaneWorkerCapability {
    /// Build a capability advertisement.
    ///
    /// Args:
    ///     kind: Conventional capability tag, e.g. ``"workflow:summarize"``,
    ///         ``"step:fetch"``, ``"provider:openai"``, ``"tag:gpu=h100"``.
    ///     version: Schema version for this capability. The control plane
    ///         refuses to route work to a worker advertising a mismatched
    ///         version.
    #[new]
    #[pyo3(signature = (kind, version=1))]
    fn new(kind: String, version: u32) -> Self {
        Self {
            inner: WorkerCapability { kind, version },
        }
    }

    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkerCapability(kind={:?}, version={})",
            self.inner.kind, self.inner.version,
        )
    }
}

impl From<WorkerCapability> for PyControlPlaneWorkerCapability {
    fn from(inner: WorkerCapability) -> Self {
        Self { inner }
    }
}

// ===========================================================================
// PyAdmissionMode
// ===========================================================================

/// How a worker declares its capacity to the control plane. Mirrors
/// [`blazen_core::distributed::AdmissionMode`].
///
/// Construct via one of the classmethods rather than `__init__`:
///
/// ```python
/// from blazen import AdmissionMode
/// mode = AdmissionMode.fixed(max_in_flight=4)
/// mode = AdmissionMode.vram_budget(max_vram_mb=24_576)
/// mode = AdmissionMode.reactive()
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "AdmissionMode", from_py_object)]
#[derive(Clone)]
pub struct PyAdmissionMode {
    pub(crate) inner: AdmissionMode,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAdmissionMode {
    /// Hard count cap. Best for fungible CPU work where every job costs
    /// roughly the same.
    #[staticmethod]
    #[pyo3(signature = (max_in_flight=1))]
    fn fixed(max_in_flight: u32) -> Self {
        Self {
            inner: AdmissionMode::Fixed { max_in_flight },
        }
    }

    /// VRAM-sum cap. Every assignment routed to a `VramBudget` worker
    /// must carry a ``resource_hint.vram_mb`` estimate.
    #[staticmethod]
    fn vram_budget(max_vram_mb: u64) -> Self {
        Self {
            inner: AdmissionMode::VramBudget { max_vram_mb },
        }
    }

    /// Worker self-decides via offer/claim/decline negotiation. Best
    /// for multi-model GPUs, browser/WebGPU workers, and `CustomProvider`
    /// hosts with their own queueing.
    #[staticmethod]
    fn reactive() -> Self {
        Self {
            inner: AdmissionMode::Reactive,
        }
    }

    /// Identifier of the underlying admission variant — one of
    /// ``"Fixed"``, ``"VramBudget"``, ``"Reactive"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            AdmissionMode::Fixed { .. } => "Fixed",
            AdmissionMode::VramBudget { .. } => "VramBudget",
            AdmissionMode::Reactive => "Reactive",
        }
    }

    /// For ``Fixed`` admission, the in-flight cap. `None` otherwise.
    #[getter]
    fn max_in_flight(&self) -> Option<u32> {
        match &self.inner {
            AdmissionMode::Fixed { max_in_flight } => Some(*max_in_flight),
            _ => None,
        }
    }

    /// For ``VramBudget`` admission, the VRAM cap in MB. `None`
    /// otherwise.
    #[getter]
    fn max_vram_mb(&self) -> Option<u64> {
        match &self.inner {
            AdmissionMode::VramBudget { max_vram_mb } => Some(*max_vram_mb),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            AdmissionMode::Fixed { max_in_flight } => {
                format!("AdmissionMode.fixed(max_in_flight={max_in_flight})")
            }
            AdmissionMode::VramBudget { max_vram_mb } => {
                format!("AdmissionMode.vram_budget(max_vram_mb={max_vram_mb})")
            }
            AdmissionMode::Reactive => "AdmissionMode.reactive()".to_owned(),
        }
    }
}

impl From<AdmissionMode> for PyAdmissionMode {
    fn from(inner: AdmissionMode) -> Self {
        Self { inner }
    }
}

// ===========================================================================
// PyControlPlaneResourceHint
// ===========================================================================

/// Optional resource estimate attached to a workflow submission. Used
/// by `VramBudget` workers to track in-flight VRAM and by `Reactive`
/// workers as input to their decide-fn.
#[gen_stub_pyclass]
#[pyclass(name = "ResourceHint", from_py_object)]
#[derive(Clone, Default)]
pub struct PyControlPlaneResourceHint {
    pub(crate) inner: ResourceHint,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyControlPlaneResourceHint {
    #[new]
    #[pyo3(signature = (vram_mb=None, cpu_cores=None, expected_seconds=None))]
    fn new(vram_mb: Option<u64>, cpu_cores: Option<f32>, expected_seconds: Option<u32>) -> Self {
        Self {
            inner: ResourceHint {
                vram_mb,
                cpu_cores,
                expected_seconds,
            },
        }
    }

    #[getter]
    fn vram_mb(&self) -> Option<u64> {
        self.inner.vram_mb
    }

    #[getter]
    fn cpu_cores(&self) -> Option<f32> {
        self.inner.cpu_cores
    }

    #[getter]
    fn expected_seconds(&self) -> Option<u32> {
        self.inner.expected_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourceHint(vram_mb={:?}, cpu_cores={:?}, expected_seconds={:?})",
            self.inner.vram_mb, self.inner.cpu_cores, self.inner.expected_seconds,
        )
    }
}

// ===========================================================================
// PyControlPlaneRunStatus
// ===========================================================================

/// Lifecycle status of a workflow run on the control plane.
#[gen_stub_pyclass_enum]
#[pyclass(name = "RunStatus", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyControlPlaneRunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl PyControlPlaneRunStatus {
    pub(crate) fn from_core(status: RunStatus) -> Self {
        match status {
            RunStatus::Pending => Self::Pending,
            RunStatus::Running => Self::Running,
            RunStatus::Completed => Self::Completed,
            RunStatus::Failed => Self::Failed,
            RunStatus::Cancelled => Self::Cancelled,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }
}

// ===========================================================================
// Conversions: control-plane wire / core types → Python dicts
// ===========================================================================

/// Build a Python dict representation of a [`RunStateSnapshot`].
///
/// The keys mirror the core struct fields verbatim; ``output`` is
/// JSON-decoded back to a native Python value when present.
pub(crate) fn snapshot_to_pydict<'py>(
    py: Python<'py>,
    snapshot: &RunStateSnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("run_id", snapshot.run_id.to_string())?;
    dict.set_item(
        "status",
        PyControlPlaneRunStatus::from_core(snapshot.status).as_str(),
    )?;
    dict.set_item("started_at_ms", snapshot.started_at_ms)?;
    dict.set_item("completed_at_ms", snapshot.completed_at_ms)?;
    dict.set_item("assigned_to", snapshot.assigned_to.clone())?;
    dict.set_item("last_event_at_ms", snapshot.last_event_at_ms)?;
    let output_py = match &snapshot.output {
        Some(v) => json_to_py(py, v)?,
        None => py.None(),
    };
    dict.set_item("output", output_py)?;
    dict.set_item("error", snapshot.error.clone())?;
    Ok(dict)
}

/// Build a Python dict representation of a [`RunEvent`].
pub(crate) fn run_event_to_pydict<'py>(
    py: Python<'py>,
    event: &RunEvent,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("run_id", event.run_id.to_string())?;
    dict.set_item("event_type", event.event_type.clone())?;
    dict.set_item("data", json_to_py(py, &event.data)?)?;
    dict.set_item("timestamp_ms", event.timestamp_ms)?;
    Ok(dict)
}

/// Build a Python dict representation of a [`WorkerInfo`].
pub(crate) fn worker_info_to_pydict<'py>(
    py: Python<'py>,
    info: &WorkerInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("node_id", info.node_id.clone())?;
    let caps = PyList::empty(py);
    for cap in &info.capabilities {
        let cap_dict = PyDict::new(py);
        cap_dict.set_item("kind", cap.kind.clone())?;
        cap_dict.set_item("version", cap.version)?;
        caps.append(cap_dict)?;
    }
    dict.set_item("capabilities", caps)?;
    dict.set_item("tags", tags_to_pydict(py, &info.tags)?)?;
    dict.set_item("admission", PyAdmissionMode::from(info.admission.clone()))?;
    dict.set_item("in_flight", info.in_flight)?;
    if let Some(snap) = &info.admission_snapshot {
        dict.set_item(
            "admission_snapshot",
            admission_snapshot_to_pydict(py, snap)?,
        )?;
    } else {
        dict.set_item("admission_snapshot", py.None())?;
    }
    dict.set_item("connected_at_ms", info.connected_at_ms)?;
    Ok(dict)
}

fn tags_to_pydict<'py>(
    py: Python<'py>,
    tags: &BTreeMap<String, String>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in tags {
        dict.set_item(k, v)?;
    }
    Ok(dict)
}

fn admission_snapshot_to_pydict<'py>(
    py: Python<'py>,
    snap: &AdmissionSnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("capacity_score", snap.capacity_score)?;
    let models = PyList::empty(py);
    for m in &snap.model_residency {
        models.append(m.clone())?;
    }
    dict.set_item("model_residency", models)?;
    dict.set_item("vram_free_mb", snap.vram_free_mb)?;
    dict.set_item("in_flight_vram_mb", snap.in_flight_vram_mb)?;
    Ok(dict)
}
