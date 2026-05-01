//! `wasm-bindgen` wrappers for [`blazen_telemetry::history`].
//!
//! Exposes [`WorkflowHistory`] as a JS class along with TypeScript-friendly
//! plain-data mirrors of [`HistoryEvent`], [`HistoryEventKind`], and
//! [`PauseReason`] derived via `tsify::Tsify`.

use blazen_telemetry::history::{
    HistoryEvent as InnerHistoryEvent, HistoryEventKind as InnerHistoryEventKind,
    PauseReason as InnerPauseReason, WorkflowHistory as InnerWorkflowHistory,
};
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Plain-data mirrors (tsify)
// ---------------------------------------------------------------------------

/// The reason a workflow was paused.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum PauseReason {
    /// Paused manually by user or API call.
    Manual,
    /// Paused because human input is required.
    InputRequired,
}

impl From<InnerPauseReason> for PauseReason {
    fn from(value: InnerPauseReason) -> Self {
        match value {
            InnerPauseReason::Manual => Self::Manual,
            InnerPauseReason::InputRequired => Self::InputRequired,
        }
    }
}

impl From<PauseReason> for InnerPauseReason {
    fn from(value: PauseReason) -> Self {
        match value {
            PauseReason::Manual => Self::Manual,
            PauseReason::InputRequired => Self::InputRequired,
        }
    }
}

/// The different kinds of events that can be recorded in a workflow history.
///
/// Mirrors [`blazen_telemetry::history::HistoryEventKind`] and serializes
/// using the same `#[serde(tag = "type")]` discriminator.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(tag = "type")]
pub enum HistoryEventKind {
    WorkflowStarted {
        input: serde_json::Value,
    },
    EventReceived {
        event_type: String,
        source_step: Option<String>,
    },
    StepDispatched {
        step_name: String,
        event_type: String,
    },
    StepCompleted {
        step_name: String,
        duration_ms: u64,
        output_type: String,
    },
    StepFailed {
        step_name: String,
        error: String,
        duration_ms: u64,
    },
    LlmCallStarted {
        provider: String,
        model: String,
    },
    LlmCallCompleted {
        provider: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        duration_ms: u64,
    },
    LlmCallFailed {
        provider: String,
        model: String,
        error: String,
        duration_ms: u64,
    },
    WorkflowPaused {
        reason: PauseReason,
        pending_count: usize,
    },
    WorkflowResumed,
    InputRequested {
        request_id: String,
        prompt: String,
    },
    InputReceived {
        request_id: String,
    },
    WorkflowCompleted {
        duration_ms: u64,
    },
    WorkflowFailed {
        error: String,
        duration_ms: u64,
    },
    WorkflowTimedOut {
        elapsed_ms: u64,
    },
}

/// A single timestamped event in a workflow's history.
///
/// `timestamp` is serialized as an RFC3339 string (chrono's default).
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct HistoryEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub sequence: u64,
    pub kind: HistoryEventKind,
}

// ---------------------------------------------------------------------------
// WasmWorkflowHistory
// ---------------------------------------------------------------------------

/// Append-only history of events for a single workflow run.
///
/// ```js
/// import { WorkflowHistory } from '@blazen/sdk';
///
/// const history = new WorkflowHistory(crypto.randomUUID(), 'my-workflow');
/// history.pushEventKind('WorkflowResumed');
/// console.log(history.len);
/// console.log(history.toJSON());
/// ```
#[wasm_bindgen(js_name = "WorkflowHistory")]
pub struct WasmWorkflowHistory {
    inner: InnerWorkflowHistory,
}

#[wasm_bindgen(js_class = "WorkflowHistory")]
impl WasmWorkflowHistory {
    /// Create a new empty history for a workflow run.
    ///
    /// `run_id` must be a valid UUID string. If parsing fails the
    /// constructor returns a `JsValue` describing the error.
    ///
    /// # Errors
    ///
    /// Returns an error if `run_id` is not a valid UUID.
    #[wasm_bindgen(constructor)]
    pub fn new(run_id: String, name: String) -> Result<WasmWorkflowHistory, JsValue> {
        let parsed = Uuid::parse_str(&run_id)
            .map_err(|e| JsValue::from_str(&format!("invalid run_id UUID: {e}")))?;
        Ok(Self {
            inner: InnerWorkflowHistory::new(parsed, name),
        })
    }

    /// The unique run identifier as a UUID string.
    #[wasm_bindgen(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// The workflow name supplied at construction time.
    #[wasm_bindgen(getter, js_name = "workflowName")]
    pub fn workflow_name(&self) -> String {
        self.inner.workflow_name.clone()
    }

    /// Number of events recorded so far.
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` if no events have been recorded.
    #[wasm_bindgen(getter, js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Append an event by passing a structured [`HistoryEventKind`] value.
    ///
    /// # Errors
    ///
    /// Returns an error if `kind` cannot be deserialized into a
    /// [`HistoryEventKind`].
    #[wasm_bindgen(js_name = "pushEvent")]
    pub fn push_event(&mut self, kind: HistoryEventKind) -> Result<(), JsValue> {
        let inner_kind = history_event_kind_to_inner(kind);
        self.inner.push(inner_kind);
        Ok(())
    }

    /// Append an event using a JSON-serialized [`HistoryEventKind`].
    ///
    /// Accepts the same shape as the tagged-union TypeScript definition
    /// (e.g. `{"type":"WorkflowResumed"}` or
    /// `{"type":"StepCompleted","step_name":"...","duration_ms":42,"output_type":"..."}`).
    ///
    /// # Errors
    ///
    /// Returns an error if `kind_json` is not a valid JSON-encoded
    /// [`HistoryEventKind`].
    #[wasm_bindgen(js_name = "pushEventKind")]
    pub fn push_event_kind(&mut self, kind_json: &str) -> Result<(), JsValue> {
        let kind: InnerHistoryEventKind = serde_json::from_str(kind_json)
            .map_err(|e| JsValue::from_str(&format!("invalid HistoryEventKind JSON: {e}")))?;
        self.inner.push(kind);
        Ok(())
    }

    /// Return the recorded events as a plain JS array.
    ///
    /// # Errors
    ///
    /// Returns an error if any event fails to convert into a JS value.
    #[wasm_bindgen(js_name = "events")]
    pub fn events(&self) -> Result<JsValue, JsValue> {
        let mirrored: Vec<HistoryEvent> = self
            .inner
            .events
            .iter()
            .map(history_event_from_inner)
            .collect();
        serde_wasm_bindgen::to_value(&mirrored).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Serialize this history to a plain JS object.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[wasm_bindgen(js_name = "toJSON")]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Serialize this history to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[wasm_bindgen(js_name = "toJsonString")]
    pub fn to_json_string(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deserialize a [`WorkflowHistory`] from a plain JS object.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` does not match the expected shape.
    #[wasm_bindgen(js_name = "fromJSON")]
    pub fn from_json(value: JsValue) -> Result<WasmWorkflowHistory, JsValue> {
        let inner: InnerWorkflowHistory =
            serde_wasm_bindgen::from_value(value).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Deserialize a [`WorkflowHistory`] from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if `json` is not a valid serialized history.
    #[wasm_bindgen(js_name = "fromJsonString")]
    pub fn from_json_string(json: &str) -> Result<WasmWorkflowHistory, JsValue> {
        let inner: InnerWorkflowHistory =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

// ---------------------------------------------------------------------------
// Conversions between mirror and inner types
// ---------------------------------------------------------------------------

fn history_event_from_inner(event: &InnerHistoryEvent) -> HistoryEvent {
    HistoryEvent {
        timestamp: event.timestamp,
        sequence: event.sequence,
        kind: history_event_kind_from_inner(event.kind.clone()),
    }
}

fn history_event_kind_from_inner(kind: InnerHistoryEventKind) -> HistoryEventKind {
    match kind {
        InnerHistoryEventKind::WorkflowStarted { input } => {
            HistoryEventKind::WorkflowStarted { input }
        }
        InnerHistoryEventKind::EventReceived {
            event_type,
            source_step,
        } => HistoryEventKind::EventReceived {
            event_type,
            source_step,
        },
        InnerHistoryEventKind::StepDispatched {
            step_name,
            event_type,
        } => HistoryEventKind::StepDispatched {
            step_name,
            event_type,
        },
        InnerHistoryEventKind::StepCompleted {
            step_name,
            duration_ms,
            output_type,
        } => HistoryEventKind::StepCompleted {
            step_name,
            duration_ms,
            output_type,
        },
        InnerHistoryEventKind::StepFailed {
            step_name,
            error,
            duration_ms,
        } => HistoryEventKind::StepFailed {
            step_name,
            error,
            duration_ms,
        },
        InnerHistoryEventKind::LlmCallStarted { provider, model } => {
            HistoryEventKind::LlmCallStarted { provider, model }
        }
        InnerHistoryEventKind::LlmCallCompleted {
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            duration_ms,
        } => HistoryEventKind::LlmCallCompleted {
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            duration_ms,
        },
        InnerHistoryEventKind::LlmCallFailed {
            provider,
            model,
            error,
            duration_ms,
        } => HistoryEventKind::LlmCallFailed {
            provider,
            model,
            error,
            duration_ms,
        },
        InnerHistoryEventKind::WorkflowPaused {
            reason,
            pending_count,
        } => HistoryEventKind::WorkflowPaused {
            reason: reason.into(),
            pending_count,
        },
        InnerHistoryEventKind::WorkflowResumed => HistoryEventKind::WorkflowResumed,
        InnerHistoryEventKind::InputRequested { request_id, prompt } => {
            HistoryEventKind::InputRequested { request_id, prompt }
        }
        InnerHistoryEventKind::InputReceived { request_id } => {
            HistoryEventKind::InputReceived { request_id }
        }
        InnerHistoryEventKind::WorkflowCompleted { duration_ms } => {
            HistoryEventKind::WorkflowCompleted { duration_ms }
        }
        InnerHistoryEventKind::WorkflowFailed { error, duration_ms } => {
            HistoryEventKind::WorkflowFailed { error, duration_ms }
        }
        InnerHistoryEventKind::WorkflowTimedOut { elapsed_ms } => {
            HistoryEventKind::WorkflowTimedOut { elapsed_ms }
        }
    }
}

fn history_event_kind_to_inner(kind: HistoryEventKind) -> InnerHistoryEventKind {
    match kind {
        HistoryEventKind::WorkflowStarted { input } => {
            InnerHistoryEventKind::WorkflowStarted { input }
        }
        HistoryEventKind::EventReceived {
            event_type,
            source_step,
        } => InnerHistoryEventKind::EventReceived {
            event_type,
            source_step,
        },
        HistoryEventKind::StepDispatched {
            step_name,
            event_type,
        } => InnerHistoryEventKind::StepDispatched {
            step_name,
            event_type,
        },
        HistoryEventKind::StepCompleted {
            step_name,
            duration_ms,
            output_type,
        } => InnerHistoryEventKind::StepCompleted {
            step_name,
            duration_ms,
            output_type,
        },
        HistoryEventKind::StepFailed {
            step_name,
            error,
            duration_ms,
        } => InnerHistoryEventKind::StepFailed {
            step_name,
            error,
            duration_ms,
        },
        HistoryEventKind::LlmCallStarted { provider, model } => {
            InnerHistoryEventKind::LlmCallStarted { provider, model }
        }
        HistoryEventKind::LlmCallCompleted {
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            duration_ms,
        } => InnerHistoryEventKind::LlmCallCompleted {
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            duration_ms,
        },
        HistoryEventKind::LlmCallFailed {
            provider,
            model,
            error,
            duration_ms,
        } => InnerHistoryEventKind::LlmCallFailed {
            provider,
            model,
            error,
            duration_ms,
        },
        HistoryEventKind::WorkflowPaused {
            reason,
            pending_count,
        } => InnerHistoryEventKind::WorkflowPaused {
            reason: reason.into(),
            pending_count,
        },
        HistoryEventKind::WorkflowResumed => InnerHistoryEventKind::WorkflowResumed,
        HistoryEventKind::InputRequested { request_id, prompt } => {
            InnerHistoryEventKind::InputRequested { request_id, prompt }
        }
        HistoryEventKind::InputReceived { request_id } => {
            InnerHistoryEventKind::InputReceived { request_id }
        }
        HistoryEventKind::WorkflowCompleted { duration_ms } => {
            InnerHistoryEventKind::WorkflowCompleted { duration_ms }
        }
        HistoryEventKind::WorkflowFailed { error, duration_ms } => {
            InnerHistoryEventKind::WorkflowFailed { error, duration_ms }
        }
        HistoryEventKind::WorkflowTimedOut { elapsed_ms } => {
            InnerHistoryEventKind::WorkflowTimedOut { elapsed_ms }
        }
    }
}
