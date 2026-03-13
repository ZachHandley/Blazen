//! Workflow snapshot for pause/resume support.
//!
//! A [`WorkflowSnapshot`] captures all the state needed to resume a paused
//! workflow: the context state, collected fan-in events, pending events that
//! were still in the routing channel, and arbitrary metadata.
//!
//! The snapshot is fully serializable via serde so it can be persisted to
//! disk, sent over the network, or stored in a checkpoint backend.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::WorkflowError;

/// A serialized representation of an event captured during a pause.
///
/// This is the core-crate version -- distinct from the one in
/// `blazen-persist` so that the core crate has no dependency on the
/// persistence layer. The two are structurally identical and can be
/// converted freely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEvent {
    /// The event type identifier (e.g. `"blazen::StartEvent"`).
    pub event_type: String,
    /// The event data as a JSON value.
    pub data: serde_json::Value,
    /// The name of the step that produced this event, if any.
    pub source_step: Option<String>,
}

/// Complete snapshot of a workflow's state at the moment it was paused.
///
/// Contains everything needed to reconstruct and resume the workflow:
///
/// - Context key/value state
/// - Fan-in collected events
/// - Events that were pending in the routing channel
/// - Workflow metadata (run ID, workflow name, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSnapshot {
    /// The name of the workflow that produced this snapshot.
    pub workflow_name: String,
    /// Unique identifier for the workflow run.
    pub run_id: Uuid,
    /// When the snapshot was captured.
    pub timestamp: DateTime<Utc>,
    /// The context's key/value state at snapshot time.
    pub context_state: HashMap<String, serde_json::Value>,
    /// The fan-in collected events at snapshot time.
    pub collected_events: HashMap<String, Vec<serde_json::Value>>,
    /// Events that were pending in the routing channel at snapshot time.
    pub pending_events: Vec<SerializedEvent>,
    /// Arbitrary metadata (includes `run_id`, `workflow_name`, and any
    /// user-defined metadata set via `Context::set_metadata`).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl WorkflowSnapshot {
    /// Serialize the snapshot to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Serialization`] if serialization fails.
    pub fn to_json(&self) -> Result<String, WorkflowError> {
        serde_json::to_string(self).map_err(WorkflowError::Serialization)
    }

    /// Serialize the snapshot to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Serialization`] if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, WorkflowError> {
        serde_json::to_string_pretty(self).map_err(WorkflowError::Serialization)
    }

    /// Deserialize a snapshot from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Serialization`] if the JSON is malformed or
    /// does not match the expected schema.
    pub fn from_json(json: &str) -> Result<Self, WorkflowError> {
        serde_json::from_str(json).map_err(WorkflowError::Serialization)
    }
}

// ---------------------------------------------------------------------------
// Conversions to/from WorkflowCheckpoint (persist feature)
// ---------------------------------------------------------------------------

/// Reserved metadata key used to stash the core-crate's `collected_events`
/// inside a [`WorkflowCheckpoint`] (which has no dedicated field for them).
#[cfg(feature = "persist")]
const COLLECTED_EVENTS_META_KEY: &str = "__blazen_collected_events";

/// Reserved metadata key used to stash `source_step` information for
/// pending events, since the persist crate's `SerializedEvent` does not
/// carry that field.
#[cfg(feature = "persist")]
const SOURCE_STEPS_META_KEY: &str = "__blazen_pending_source_steps";

#[cfg(feature = "persist")]
impl From<WorkflowSnapshot> for blazen_persist::WorkflowCheckpoint {
    fn from(snap: WorkflowSnapshot) -> Self {
        let mut metadata = snap.metadata;

        // Stash collected_events into metadata so we can restore them later.
        if !snap.collected_events.is_empty()
            && let Ok(val) = serde_json::to_value(&snap.collected_events)
        {
            metadata.insert(COLLECTED_EVENTS_META_KEY.to_owned(), val);
        }

        // Convert pending events, preserving source_step in metadata.
        let source_steps: Vec<Option<String>> = snap
            .pending_events
            .iter()
            .map(|e| e.source_step.clone())
            .collect();

        if source_steps.iter().any(Option::is_some)
            && let Ok(val) = serde_json::to_value(&source_steps)
        {
            metadata.insert(SOURCE_STEPS_META_KEY.to_owned(), val);
        }

        let pending_events = snap
            .pending_events
            .into_iter()
            .map(|e| blazen_persist::SerializedEvent {
                event_type: e.event_type,
                data: e.data,
            })
            .collect();

        blazen_persist::WorkflowCheckpoint {
            workflow_name: snap.workflow_name,
            run_id: snap.run_id,
            timestamp: snap.timestamp,
            state: snap.context_state,
            pending_events,
            metadata,
        }
    }
}

#[cfg(feature = "persist")]
impl From<blazen_persist::WorkflowCheckpoint> for WorkflowSnapshot {
    fn from(cp: blazen_persist::WorkflowCheckpoint) -> Self {
        let mut metadata = cp.metadata;

        // Restore collected_events from metadata.
        let collected_events = metadata
            .remove(COLLECTED_EVENTS_META_KEY)
            .and_then(|val| {
                serde_json::from_value::<HashMap<String, Vec<serde_json::Value>>>(val).ok()
            })
            .unwrap_or_default();

        // Restore source_step information from metadata.
        let source_steps: Vec<Option<String>> = metadata
            .remove(SOURCE_STEPS_META_KEY)
            .and_then(|val| serde_json::from_value(val).ok())
            .unwrap_or_default();

        let pending_events = cp
            .pending_events
            .into_iter()
            .enumerate()
            .map(|(i, e)| SerializedEvent {
                event_type: e.event_type,
                data: e.data,
                source_step: source_steps.get(i).and_then(Clone::clone),
            })
            .collect();

        WorkflowSnapshot {
            workflow_name: cp.workflow_name,
            run_id: cp.run_id,
            timestamp: cp.timestamp,
            context_state: cp.state,
            collected_events,
            pending_events,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> WorkflowSnapshot {
        let mut state = HashMap::new();
        state.insert("counter".to_owned(), serde_json::json!(42));
        state.insert("name".to_owned(), serde_json::json!("alice"));

        let mut collected = HashMap::new();
        collected.insert(
            "blazen::StartEvent".to_owned(),
            vec![serde_json::json!({"data": 1})],
        );

        let mut metadata = HashMap::new();
        let run_id = Uuid::new_v4();
        metadata.insert(
            "run_id".to_owned(),
            serde_json::Value::String(run_id.to_string()),
        );
        metadata.insert(
            "workflow_name".to_owned(),
            serde_json::Value::String("test_wf".to_owned()),
        );

        WorkflowSnapshot {
            workflow_name: "test_wf".to_owned(),
            run_id,
            timestamp: Utc::now(),
            context_state: state,
            collected_events: collected,
            pending_events: vec![SerializedEvent {
                event_type: "blazen::StartEvent".to_owned(),
                data: serde_json::json!({"data": "hello"}),
                source_step: Some("step_a".to_owned()),
            }],
            metadata,
        }
    }

    #[test]
    fn json_roundtrip() {
        let snap = sample_snapshot();
        let json = snap.to_json().unwrap();
        let restored = WorkflowSnapshot::from_json(&json).unwrap();
        assert_eq!(restored.workflow_name, snap.workflow_name);
        assert_eq!(restored.run_id, snap.run_id);
        assert_eq!(restored.context_state, snap.context_state);
        assert_eq!(restored.collected_events, snap.collected_events);
        assert_eq!(restored.pending_events.len(), snap.pending_events.len());
        assert_eq!(
            restored.pending_events[0].event_type,
            snap.pending_events[0].event_type
        );
    }

    #[test]
    fn pretty_json_roundtrip() {
        let snap = sample_snapshot();
        let json = snap.to_json_pretty().unwrap();
        let restored = WorkflowSnapshot::from_json(&json).unwrap();
        assert_eq!(restored.workflow_name, snap.workflow_name);
    }

    #[test]
    fn from_invalid_json_fails() {
        let result = WorkflowSnapshot::from_json("not valid json");
        assert!(result.is_err());
    }
}
