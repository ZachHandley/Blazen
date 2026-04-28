//! Node bindings for `blazen_telemetry::history`.
//!
//! Exposes [`JsWorkflowHistory`] as an opaque NAPI class wrapping
//! [`blazen_telemetry::WorkflowHistory`]. Event kinds are constructed via
//! factory methods on [`JsHistoryEventKind`] because the underlying Rust
//! enum carries struct-variant payloads that cannot be represented as a
//! NAPI string enum.

use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use uuid::Uuid;

use blazen_telemetry::{HistoryEvent, HistoryEventKind, PauseReason, WorkflowHistory};

use crate::error::to_napi_error;

/// Cast a JS-provided signed millisecond value into the unsigned form
/// required by the underlying Rust types. Negative values are clamped to
/// zero; this matches the convention used elsewhere in the Node binding
/// (see e.g. `pipeline::snapshot::JsStageResult::duration_ms`).
#[allow(clippy::cast_sign_loss)]
fn ms_to_u64(v: i64) -> u64 {
    if v < 0 { 0 } else { v as u64 }
}

// ---------------------------------------------------------------------------
// PauseReason
// ---------------------------------------------------------------------------

/// Why a workflow was paused.
///
/// - `Manual`: paused manually by user or API call.
/// - `InputRequired`: paused because human input is required.
#[napi(string_enum, js_name = "PauseReason")]
#[derive(Debug, Clone, Copy)]
pub enum JsPauseReason {
    Manual,
    InputRequired,
}

impl From<JsPauseReason> for PauseReason {
    fn from(p: JsPauseReason) -> Self {
        match p {
            JsPauseReason::Manual => Self::Manual,
            JsPauseReason::InputRequired => Self::InputRequired,
        }
    }
}

impl From<PauseReason> for JsPauseReason {
    fn from(p: PauseReason) -> Self {
        match p {
            PauseReason::Manual => Self::Manual,
            PauseReason::InputRequired => Self::InputRequired,
        }
    }
}

// ---------------------------------------------------------------------------
// HistoryEventKind
// ---------------------------------------------------------------------------

/// Discriminator strings for the variants of [`JsHistoryEventKind`].
///
/// Returned by [`JsHistoryEventKind::kind`] so JavaScript code can
/// pattern-match on the variant without inspecting the JSON payload.
#[napi(string_enum, js_name = "HistoryEventKindTag")]
#[derive(Debug, Clone, Copy)]
pub enum JsHistoryEventKindTag {
    WorkflowStarted,
    EventReceived,
    StepDispatched,
    StepCompleted,
    StepFailed,
    LlmCallStarted,
    LlmCallCompleted,
    LlmCallFailed,
    WorkflowPaused,
    WorkflowResumed,
    InputRequested,
    InputReceived,
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowTimedOut,
}

fn tag_for(kind: &HistoryEventKind) -> JsHistoryEventKindTag {
    match kind {
        HistoryEventKind::WorkflowStarted { .. } => JsHistoryEventKindTag::WorkflowStarted,
        HistoryEventKind::EventReceived { .. } => JsHistoryEventKindTag::EventReceived,
        HistoryEventKind::StepDispatched { .. } => JsHistoryEventKindTag::StepDispatched,
        HistoryEventKind::StepCompleted { .. } => JsHistoryEventKindTag::StepCompleted,
        HistoryEventKind::StepFailed { .. } => JsHistoryEventKindTag::StepFailed,
        HistoryEventKind::LlmCallStarted { .. } => JsHistoryEventKindTag::LlmCallStarted,
        HistoryEventKind::LlmCallCompleted { .. } => JsHistoryEventKindTag::LlmCallCompleted,
        HistoryEventKind::LlmCallFailed { .. } => JsHistoryEventKindTag::LlmCallFailed,
        HistoryEventKind::WorkflowPaused { .. } => JsHistoryEventKindTag::WorkflowPaused,
        HistoryEventKind::WorkflowResumed => JsHistoryEventKindTag::WorkflowResumed,
        HistoryEventKind::InputRequested { .. } => JsHistoryEventKindTag::InputRequested,
        HistoryEventKind::InputReceived { .. } => JsHistoryEventKindTag::InputReceived,
        HistoryEventKind::WorkflowCompleted { .. } => JsHistoryEventKindTag::WorkflowCompleted,
        HistoryEventKind::WorkflowFailed { .. } => JsHistoryEventKindTag::WorkflowFailed,
        HistoryEventKind::WorkflowTimedOut { .. } => JsHistoryEventKindTag::WorkflowTimedOut,
    }
}

/// A single workflow history event variant.
///
/// Construct via the factory methods (e.g. [`JsHistoryEventKind::workflowStarted`]).
/// Inspect the discriminator with [`JsHistoryEventKind::kind`] and pull
/// the payload out as JSON via [`JsHistoryEventKind::toJson`].
#[napi(js_name = "HistoryEventKind")]
pub struct JsHistoryEventKind {
    pub(crate) inner: HistoryEventKind,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsHistoryEventKind {
    /// Workflow started executing with the given input.
    #[napi(factory, js_name = "workflowStarted")]
    pub fn workflow_started(input: serde_json::Value) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowStarted { input },
        }
    }

    /// An event was received by the workflow engine.
    #[napi(factory, js_name = "eventReceived")]
    pub fn event_received(event_type: String, source_step: Option<String>) -> Self {
        Self {
            inner: HistoryEventKind::EventReceived {
                event_type,
                source_step,
            },
        }
    }

    /// A step was dispatched for execution.
    #[napi(factory, js_name = "stepDispatched")]
    pub fn step_dispatched(step_name: String, event_type: String) -> Self {
        Self {
            inner: HistoryEventKind::StepDispatched {
                step_name,
                event_type,
            },
        }
    }

    /// A step completed successfully.
    #[napi(factory, js_name = "stepCompleted")]
    pub fn step_completed(step_name: String, duration_ms: i64, output_type: String) -> Self {
        Self {
            inner: HistoryEventKind::StepCompleted {
                step_name,
                duration_ms: ms_to_u64(duration_ms),
                output_type,
            },
        }
    }

    /// A step failed.
    #[napi(factory, js_name = "stepFailed")]
    pub fn step_failed(step_name: String, error: String, duration_ms: i64) -> Self {
        Self {
            inner: HistoryEventKind::StepFailed {
                step_name,
                error,
                duration_ms: ms_to_u64(duration_ms),
            },
        }
    }

    /// An LLM call was initiated.
    #[napi(factory, js_name = "llmCallStarted")]
    pub fn llm_call_started(provider: String, model: String) -> Self {
        Self {
            inner: HistoryEventKind::LlmCallStarted { provider, model },
        }
    }

    /// An LLM call completed successfully.
    #[napi(factory, js_name = "llmCallCompleted")]
    pub fn llm_call_completed(
        provider: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        duration_ms: i64,
    ) -> Self {
        Self {
            inner: HistoryEventKind::LlmCallCompleted {
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                duration_ms: ms_to_u64(duration_ms),
            },
        }
    }

    /// An LLM call failed.
    #[napi(factory, js_name = "llmCallFailed")]
    pub fn llm_call_failed(
        provider: String,
        model: String,
        error: String,
        duration_ms: i64,
    ) -> Self {
        Self {
            inner: HistoryEventKind::LlmCallFailed {
                provider,
                model,
                error,
                duration_ms: ms_to_u64(duration_ms),
            },
        }
    }

    /// The workflow was paused.
    #[napi(factory, js_name = "workflowPaused")]
    pub fn workflow_paused(reason: JsPauseReason, pending_count: u32) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowPaused {
                reason: reason.into(),
                pending_count: pending_count as usize,
            },
        }
    }

    /// The workflow resumed from a paused state.
    #[napi(factory, js_name = "workflowResumed")]
    pub fn workflow_resumed() -> Self {
        Self {
            inner: HistoryEventKind::WorkflowResumed,
        }
    }

    /// The workflow is requesting human input.
    #[napi(factory, js_name = "inputRequested")]
    pub fn input_requested(request_id: String, prompt: String) -> Self {
        Self {
            inner: HistoryEventKind::InputRequested { request_id, prompt },
        }
    }

    /// Human input was received for a previously requested input.
    #[napi(factory, js_name = "inputReceived")]
    pub fn input_received(request_id: String) -> Self {
        Self {
            inner: HistoryEventKind::InputReceived { request_id },
        }
    }

    /// The workflow completed successfully.
    #[napi(factory, js_name = "workflowCompleted")]
    pub fn workflow_completed(duration_ms: i64) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowCompleted {
                duration_ms: ms_to_u64(duration_ms),
            },
        }
    }

    /// The workflow failed.
    #[napi(factory, js_name = "workflowFailed")]
    pub fn workflow_failed(error: String, duration_ms: i64) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowFailed {
                error,
                duration_ms: ms_to_u64(duration_ms),
            },
        }
    }

    /// The workflow timed out.
    #[napi(factory, js_name = "workflowTimedOut")]
    pub fn workflow_timed_out(elapsed_ms: i64) -> Self {
        Self {
            inner: HistoryEventKind::WorkflowTimedOut {
                elapsed_ms: ms_to_u64(elapsed_ms),
            },
        }
    }

    /// Discriminator tag identifying which variant this event holds.
    #[napi(getter, js_name = "kind")]
    pub fn kind(&self) -> JsHistoryEventKindTag {
        tag_for(&self.inner)
    }

    /// Serialize this event kind to a JSON value.
    ///
    /// The format mirrors `serde(tag = "type")` from the underlying Rust
    /// enum, e.g. `{"type":"StepCompleted","step_name":"a","duration_ms":42,"output_type":"E"}`.
    #[napi(js_name = "toJson")]
    pub fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(&self.inner).map_err(to_napi_error)
    }

    /// Construct a [`JsHistoryEventKind`] from its JSON representation.
    #[napi(factory, js_name = "fromJson")]
    pub fn from_json(value: serde_json::Value) -> Result<Self> {
        let inner: HistoryEventKind = serde_json::from_value(value).map_err(to_napi_error)?;
        Ok(Self { inner })
    }
}

// ---------------------------------------------------------------------------
// HistoryEvent
// ---------------------------------------------------------------------------

/// A single timestamped entry in a [`JsWorkflowHistory`].
///
/// `kind` is the JSON-serialized form of the event (matches
/// `serde(tag = "type")` from the underlying Rust enum); use
/// [`JsHistoryEventKind::fromJson`] to round-trip back into a typed
/// wrapper.
#[napi(object, js_name = "HistoryEvent")]
pub struct JsHistoryEvent {
    /// The UUID of the run this event belongs to (as a string).
    #[napi(js_name = "runId")]
    pub run_id: String,
    /// RFC3339 / ISO-8601 timestamp at which the event was recorded.
    pub timestamp: String,
    /// Monotonically increasing sequence number within the run.
    pub sequence: i64,
    /// The event payload as a tagged JSON value.
    pub kind: serde_json::Value,
}

fn event_to_js(run_id: &Uuid, event: &HistoryEvent) -> JsHistoryEvent {
    JsHistoryEvent {
        run_id: run_id.to_string(),
        timestamp: event.timestamp.to_rfc3339(),
        sequence: i64::try_from(event.sequence).unwrap_or(i64::MAX),
        kind: serde_json::to_value(&event.kind).unwrap_or(serde_json::Value::Null),
    }
}

// ---------------------------------------------------------------------------
// WorkflowHistory
// ---------------------------------------------------------------------------

/// Append-only history of events for a single workflow run.
///
/// ```javascript
/// const history = new WorkflowHistory(
///   "00000000-0000-0000-0000-000000000000",
///   "my-workflow",
/// );
/// history.push(HistoryEventKind.workflowStarted({ foo: "bar" }));
/// console.log(history.len); // 1
/// ```
#[napi(js_name = "WorkflowHistory")]
pub struct JsWorkflowHistory {
    pub(crate) inner: Mutex<WorkflowHistory>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsWorkflowHistory {
    /// Create a new empty history.
    ///
    /// `runId` must be a valid UUID string. `name` is the workflow name
    /// recorded alongside the events.
    #[napi(constructor)]
    pub fn new(run_id: String, name: String) -> Result<Self> {
        let uuid = Uuid::parse_str(&run_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid runId UUID: {e}")))?;
        Ok(Self {
            inner: Mutex::new(WorkflowHistory::new(uuid, name)),
        })
    }

    /// Append an event to the history. The timestamp and sequence number
    /// are assigned automatically.
    #[napi]
    pub fn push(&self, kind: &JsHistoryEventKind) {
        let mut guard = self.inner.lock().expect("poisoned");
        guard.push(kind.inner.clone());
    }

    /// Number of events recorded.
    #[napi(getter, js_name = "len")]
    pub fn len(&self) -> u32 {
        let guard = self.inner.lock().expect("poisoned");
        u32::try_from(guard.len()).unwrap_or(u32::MAX)
    }

    /// `true` if no events have been recorded.
    #[napi(getter, js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        let guard = self.inner.lock().expect("poisoned");
        guard.is_empty()
    }

    /// The UUID identifier of the run, as a string.
    #[napi(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.run_id.to_string()
    }

    /// The workflow name.
    #[napi(getter, js_name = "workflowName")]
    pub fn workflow_name(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.workflow_name.clone()
    }

    /// Snapshot the recorded events as a list of POJOs.
    #[napi(getter, js_name = "events")]
    pub fn events(&self) -> Vec<JsHistoryEvent> {
        let guard = self.inner.lock().expect("poisoned");
        let run_id = guard.run_id;
        guard
            .events
            .iter()
            .map(|e| event_to_js(&run_id, e))
            .collect()
    }

    /// Serialize the entire history to a JSON string.
    #[napi(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String> {
        let guard = self.inner.lock().expect("poisoned");
        serde_json::to_string(&*guard).map_err(to_napi_error)
    }

    /// Pretty-print the history as a JSON string.
    #[napi(js_name = "toJsonPretty")]
    pub fn to_json_pretty(&self) -> Result<String> {
        let guard = self.inner.lock().expect("poisoned");
        serde_json::to_string_pretty(&*guard).map_err(to_napi_error)
    }

    /// Parse a history from its JSON representation.
    #[napi(factory, js_name = "fromJson")]
    pub fn from_json(json: String) -> Result<Self> {
        let inner: WorkflowHistory = serde_json::from_str(&json).map_err(to_napi_error)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }
}
