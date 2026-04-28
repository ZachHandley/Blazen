//! Python bindings for `blazen_telemetry::history`.
//!
//! Exposes [`WorkflowHistory`], [`HistoryEvent`], [`HistoryEventKind`], and
//! [`PauseReason`] as `WorkflowHistory`, `HistoryEvent`, `HistoryEventKind`,
//! and `PauseReason` Python classes.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use pythonize::pythonize;
use uuid::Uuid;

use blazen_telemetry::{HistoryEvent, HistoryEventKind, PauseReason, WorkflowHistory};

use crate::error::BlazenPyError;

// ---------------------------------------------------------------------------
// PyPauseReason
// ---------------------------------------------------------------------------

/// Reason a workflow was paused.
#[gen_stub_pyclass_enum]
#[pyclass(name = "PauseReason", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPauseReason {
    Manual,
    InputRequired,
}

impl From<PyPauseReason> for PauseReason {
    fn from(v: PyPauseReason) -> Self {
        match v {
            PyPauseReason::Manual => Self::Manual,
            PyPauseReason::InputRequired => Self::InputRequired,
        }
    }
}

impl From<PauseReason> for PyPauseReason {
    fn from(v: PauseReason) -> Self {
        match v {
            PauseReason::Manual => Self::Manual,
            PauseReason::InputRequired => Self::InputRequired,
        }
    }
}

// ---------------------------------------------------------------------------
// PyHistoryEventKind
// ---------------------------------------------------------------------------

/// A workflow history event payload.
///
/// Construct via the static factory methods (`HistoryEventKind.workflow_started`,
/// `HistoryEventKind.step_completed`, etc.). Use `to_dict()` to inspect the
/// payload as a Python dict.
#[gen_stub_pyclass]
#[pyclass(name = "HistoryEventKind", from_py_object)]
#[derive(Clone)]
pub struct PyHistoryEventKind {
    pub(crate) inner: HistoryEventKind,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHistoryEventKind {
    /// Workflow started executing with the given JSON-serializable input.
    #[staticmethod]
    fn workflow_started(input: Bound<'_, PyAny>) -> PyResult<Self> {
        let value: serde_json::Value = pythonize::depythonize(&input)
            .map_err(|e| BlazenPyError::Serialization(e.to_string()))?;
        Ok(Self {
            inner: HistoryEventKind::WorkflowStarted { input: value },
        })
    }

    /// An event was received by the workflow engine.
    #[staticmethod]
    #[pyo3(signature = (event_type, source_step=None))]
    fn event_received(event_type: String, source_step: Option<String>) -> Self {
        Self {
            inner: HistoryEventKind::EventReceived {
                event_type,
                source_step,
            },
        }
    }

    /// A step was dispatched for execution.
    #[staticmethod]
    fn step_dispatched(step_name: String, event_type: String) -> Self {
        Self {
            inner: HistoryEventKind::StepDispatched {
                step_name,
                event_type,
            },
        }
    }

    /// A step completed successfully.
    #[staticmethod]
    fn step_completed(step_name: String, duration_ms: u64, output_type: String) -> Self {
        Self {
            inner: HistoryEventKind::StepCompleted {
                step_name,
                duration_ms,
                output_type,
            },
        }
    }

    /// A step failed with an error.
    #[staticmethod]
    fn step_failed(step_name: String, error: String, duration_ms: u64) -> Self {
        Self {
            inner: HistoryEventKind::StepFailed {
                step_name,
                error,
                duration_ms,
            },
        }
    }

    /// An LLM call was initiated.
    #[staticmethod]
    fn llm_call_started(provider: String, model: String) -> Self {
        Self {
            inner: HistoryEventKind::LlmCallStarted { provider, model },
        }
    }

    /// An LLM call completed successfully.
    #[staticmethod]
    fn llm_call_completed(
        provider: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        duration_ms: u64,
    ) -> Self {
        Self {
            inner: HistoryEventKind::LlmCallCompleted {
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                duration_ms,
            },
        }
    }

    /// An LLM call failed.
    #[staticmethod]
    fn llm_call_failed(provider: String, model: String, error: String, duration_ms: u64) -> Self {
        Self {
            inner: HistoryEventKind::LlmCallFailed {
                provider,
                model,
                error,
                duration_ms,
            },
        }
    }

    /// The workflow was paused.
    #[staticmethod]
    fn workflow_paused(reason: PyPauseReason, pending_count: usize) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowPaused {
                reason: reason.into(),
                pending_count,
            },
        }
    }

    /// The workflow resumed from a paused state.
    #[staticmethod]
    fn workflow_resumed() -> Self {
        Self {
            inner: HistoryEventKind::WorkflowResumed,
        }
    }

    /// The workflow is requesting human input.
    #[staticmethod]
    fn input_requested(request_id: String, prompt: String) -> Self {
        Self {
            inner: HistoryEventKind::InputRequested { request_id, prompt },
        }
    }

    /// Human input was received.
    #[staticmethod]
    fn input_received(request_id: String) -> Self {
        Self {
            inner: HistoryEventKind::InputReceived { request_id },
        }
    }

    /// The workflow completed successfully.
    #[staticmethod]
    fn workflow_completed(duration_ms: u64) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowCompleted { duration_ms },
        }
    }

    /// The workflow failed.
    #[staticmethod]
    fn workflow_failed(error: String, duration_ms: u64) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowFailed { error, duration_ms },
        }
    }

    /// The workflow timed out.
    #[staticmethod]
    fn workflow_timed_out(elapsed_ms: u64) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowTimedOut { elapsed_ms },
        }
    }

    /// Convert this event payload to a Python dict via the serde representation.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.inner).map_err(|e| BlazenPyError::Serialization(e.to_string()).into())
    }

    /// Return the variant tag (e.g. `"WorkflowStarted"`, `"StepCompleted"`).
    #[getter]
    fn type_name(&self) -> &'static str {
        match &self.inner {
            HistoryEventKind::WorkflowStarted { .. } => "WorkflowStarted",
            HistoryEventKind::EventReceived { .. } => "EventReceived",
            HistoryEventKind::StepDispatched { .. } => "StepDispatched",
            HistoryEventKind::StepCompleted { .. } => "StepCompleted",
            HistoryEventKind::StepFailed { .. } => "StepFailed",
            HistoryEventKind::LlmCallStarted { .. } => "LlmCallStarted",
            HistoryEventKind::LlmCallCompleted { .. } => "LlmCallCompleted",
            HistoryEventKind::LlmCallFailed { .. } => "LlmCallFailed",
            HistoryEventKind::WorkflowPaused { .. } => "WorkflowPaused",
            HistoryEventKind::WorkflowResumed => "WorkflowResumed",
            HistoryEventKind::InputRequested { .. } => "InputRequested",
            HistoryEventKind::InputReceived { .. } => "InputReceived",
            HistoryEventKind::WorkflowCompleted { .. } => "WorkflowCompleted",
            HistoryEventKind::WorkflowFailed { .. } => "WorkflowFailed",
            HistoryEventKind::WorkflowTimedOut { .. } => "WorkflowTimedOut",
        }
    }

    fn __repr__(&self) -> String {
        format!("HistoryEventKind({})", self.type_name())
    }
}

// ---------------------------------------------------------------------------
// PyHistoryEvent
// ---------------------------------------------------------------------------

/// A single timestamped event in a workflow's history.
#[gen_stub_pyclass]
#[pyclass(name = "HistoryEvent", from_py_object)]
#[derive(Clone)]
pub struct PyHistoryEvent {
    pub(crate) inner: HistoryEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHistoryEvent {
    /// ISO-8601 UTC timestamp of when this event occurred.
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    /// Monotonically increasing sequence number within the run.
    #[getter]
    fn sequence(&self) -> u64 {
        self.inner.sequence
    }

    /// The event payload.
    #[getter]
    fn kind(&self) -> PyHistoryEventKind {
        PyHistoryEventKind {
            inner: self.inner.kind.clone(),
        }
    }

    /// Convert this event to a Python dict via the serde representation.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.inner).map_err(|e| BlazenPyError::Serialization(e.to_string()).into())
    }

    fn __repr__(&self) -> String {
        format!(
            "HistoryEvent(seq={}, kind={})",
            self.inner.sequence,
            PyHistoryEventKind {
                inner: self.inner.kind.clone()
            }
            .type_name()
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflowHistory
// ---------------------------------------------------------------------------

/// Append-only history of events for a single workflow run.
///
/// Example:
///     >>> from blazen import WorkflowHistory, HistoryEventKind
///     >>> h = WorkflowHistory("00000000-0000-0000-0000-000000000000", "demo")
///     >>> h.push(HistoryEventKind.workflow_started({"q": "hi"}))
///     >>> h.push(HistoryEventKind.workflow_completed(duration_ms=42))
///     >>> assert len(h) == 2
#[gen_stub_pyclass]
#[pyclass(name = "WorkflowHistory", from_py_object)]
#[derive(Clone)]
pub struct PyWorkflowHistory {
    pub(crate) inner: WorkflowHistory,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWorkflowHistory {
    /// Create a new empty history for a workflow run.
    ///
    /// Args:
    ///     run_id: A UUID string (any RFC 4122 format) identifying the run.
    ///     name: The name of the workflow being executed.
    #[new]
    fn new(run_id: &str, name: String) -> PyResult<Self> {
        let parsed = Uuid::parse_str(run_id).map_err(|e| {
            BlazenPyError::InvalidArgument(format!("invalid run_id UUID '{run_id}': {e}"))
        })?;
        Ok(Self {
            inner: WorkflowHistory::new(parsed, name),
        })
    }

    /// The run UUID as a string.
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// The workflow name.
    #[getter]
    fn workflow_name(&self) -> String {
        self.inner.workflow_name.clone()
    }

    /// All events recorded so far.
    #[getter]
    fn events(&self) -> Vec<PyHistoryEvent> {
        self.inner
            .events
            .iter()
            .cloned()
            .map(|e| PyHistoryEvent { inner: e })
            .collect()
    }

    /// Append an event to the history.
    fn push(&mut self, kind: PyHistoryEventKind) {
        self.inner.push(kind.inner);
    }

    /// Number of events recorded.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of events recorded.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if no events have been recorded.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Serialize the full history to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| BlazenPyError::Serialization(e.to_string()).into())
    }

    /// Convert the full history to a Python dict via the serde representation.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.inner).map_err(|e| BlazenPyError::Serialization(e.to_string()).into())
    }

    /// Reconstruct a `WorkflowHistory` from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: WorkflowHistory =
            serde_json::from_str(json).map_err(|e| BlazenPyError::Serialization(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Reconstruct a `WorkflowHistory` from a Python dict (matching
    /// the `to_dict()` shape).
    #[staticmethod]
    fn from_dict(data: Bound<'_, PyDict>) -> PyResult<Self> {
        let inner: WorkflowHistory = pythonize::depythonize(&data.into_any())
            .map_err(|e| BlazenPyError::Serialization(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkflowHistory(run_id={}, name={}, events={})",
            self.inner.run_id,
            self.inner.workflow_name,
            self.inner.events.len()
        )
    }
}
