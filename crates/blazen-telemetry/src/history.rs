//! Append-only workflow event history for observability and replay.
//!
//! [`WorkflowHistory`] captures a chronological log of everything that happens
//! during a workflow run: step dispatches, LLM calls, pauses, completions, and
//! failures. This can be serialized for post-mortem debugging, billing
//! reconciliation, or replayed in a UI.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Append-only history of events for a single workflow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowHistory {
    /// Unique identifier for this workflow run.
    pub run_id: Uuid,
    /// The name of the workflow being executed.
    pub workflow_name: String,
    /// Chronologically ordered events.
    pub events: Vec<HistoryEvent>,
}

impl WorkflowHistory {
    /// Create a new empty history for a workflow run.
    #[must_use]
    pub fn new(run_id: Uuid, workflow_name: String) -> Self {
        Self {
            run_id,
            workflow_name,
            events: Vec::new(),
        }
    }

    /// Append an event to the history, auto-assigning timestamp and sequence.
    pub fn push(&mut self, kind: HistoryEventKind) {
        let sequence = self.events.len() as u64;
        self.events.push(HistoryEvent {
            timestamp: Utc::now(),
            sequence,
            kind,
        });
    }

    /// Return the number of events recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` if no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// A single timestamped event in a workflow's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEvent {
    /// When this event occurred.
    pub timestamp: DateTime<Utc>,
    /// Monotonically increasing sequence number within the run.
    pub sequence: u64,
    /// What happened.
    pub kind: HistoryEventKind,
}

/// The different kinds of events that can be recorded in a workflow history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HistoryEventKind {
    /// The workflow started execution.
    WorkflowStarted {
        /// The initial input provided to the workflow.
        input: serde_json::Value,
    },
    /// An event was received by the workflow engine.
    EventReceived {
        /// The type name of the event.
        event_type: String,
        /// The step that produced this event, if any.
        source_step: Option<String>,
    },
    /// A step was dispatched for execution.
    StepDispatched {
        /// The name of the step being dispatched.
        step_name: String,
        /// The type name of the triggering event.
        event_type: String,
    },
    /// A step completed successfully.
    StepCompleted {
        /// The name of the step that completed.
        step_name: String,
        /// How long the step took in milliseconds.
        duration_ms: u64,
        /// The type name of the output event.
        output_type: String,
    },
    /// A step failed with an error.
    StepFailed {
        /// The name of the step that failed.
        step_name: String,
        /// The error message.
        error: String,
        /// How long the step ran before failing, in milliseconds.
        duration_ms: u64,
    },
    /// An LLM call was initiated.
    LlmCallStarted {
        /// The LLM provider name.
        provider: String,
        /// The model identifier.
        model: String,
    },
    /// An LLM call completed successfully.
    LlmCallCompleted {
        /// The LLM provider name.
        provider: String,
        /// The model identifier.
        model: String,
        /// Number of tokens in the prompt.
        prompt_tokens: u32,
        /// Number of tokens in the completion.
        completion_tokens: u32,
        /// Total tokens consumed.
        total_tokens: u32,
        /// How long the call took in milliseconds.
        duration_ms: u64,
    },
    /// An LLM call failed.
    LlmCallFailed {
        /// The LLM provider name.
        provider: String,
        /// The model identifier.
        model: String,
        /// The error message.
        error: String,
        /// How long the call ran before failing, in milliseconds.
        duration_ms: u64,
    },
    /// The workflow was paused.
    WorkflowPaused {
        /// Why the workflow was paused.
        reason: PauseReason,
        /// Number of pending events at time of pause.
        pending_count: usize,
    },
    /// The workflow resumed from a paused state.
    WorkflowResumed,
    /// The workflow is requesting human input.
    InputRequested {
        /// Unique identifier for this input request.
        request_id: String,
        /// The prompt shown to the user.
        prompt: String,
    },
    /// Human input was received.
    InputReceived {
        /// The request identifier that was fulfilled.
        request_id: String,
    },
    /// The workflow completed successfully.
    WorkflowCompleted {
        /// Total wall-clock time in milliseconds.
        duration_ms: u64,
    },
    /// The workflow failed.
    WorkflowFailed {
        /// The error message.
        error: String,
        /// Total wall-clock time in milliseconds.
        duration_ms: u64,
    },
    /// The workflow timed out.
    WorkflowTimedOut {
        /// How long the workflow ran before timing out, in milliseconds.
        elapsed_ms: u64,
    },
}

/// The reason a workflow was paused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PauseReason {
    /// Paused manually by user or API call.
    Manual,
    /// Paused because human input is required.
    InputRequired,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_history_is_empty() {
        let h = WorkflowHistory::new(Uuid::new_v4(), "test-workflow".into());
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn push_increments_sequence() {
        let mut h = WorkflowHistory::new(Uuid::new_v4(), "test-workflow".into());
        h.push(HistoryEventKind::WorkflowStarted {
            input: serde_json::json!({"key": "value"}),
        });
        h.push(HistoryEventKind::WorkflowResumed);
        assert_eq!(h.len(), 2);
        assert_eq!(h.events[0].sequence, 0);
        assert_eq!(h.events[1].sequence, 1);
    }

    #[test]
    fn history_serialization_roundtrip() {
        let mut h = WorkflowHistory::new(Uuid::new_v4(), "my-workflow".into());
        h.push(HistoryEventKind::WorkflowStarted {
            input: serde_json::json!(null),
        });
        h.push(HistoryEventKind::StepDispatched {
            step_name: "step1".into(),
            event_type: "MyEvent".into(),
        });
        h.push(HistoryEventKind::StepCompleted {
            step_name: "step1".into(),
            duration_ms: 42,
            output_type: "OutputEvent".into(),
        });
        h.push(HistoryEventKind::WorkflowCompleted { duration_ms: 100 });

        let json = serde_json::to_string(&h).unwrap();
        let deserialized: WorkflowHistory = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.run_id, h.run_id);
        assert_eq!(deserialized.workflow_name, h.workflow_name);
        assert_eq!(deserialized.events.len(), h.events.len());
    }

    #[test]
    fn pause_reason_roundtrip() {
        let reason = PauseReason::InputRequired;
        let json = serde_json::to_string(&reason).unwrap();
        let deserialized: PauseReason = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, PauseReason::InputRequired));
    }
}
