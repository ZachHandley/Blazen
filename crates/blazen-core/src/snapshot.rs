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

use blazen_events::InputRequestEvent;

use crate::error::WorkflowError;
use crate::value::StateValue;

/// Snapshot format version. Incremented whenever the on-disk or
/// on-the-wire representation changes in a way that older readers
/// cannot handle. Readers that see a snapshot with a higher version
/// MUST return [`WorkflowError::SnapshotVersionMismatch`].
pub const SNAPSHOT_VERSION: u32 = 1;

fn default_snapshot_version() -> u32 {
    SNAPSHOT_VERSION
}

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
/// - History events recorded up to the pause point (requires `telemetry` feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSnapshot {
    /// Snapshot format version.
    ///
    /// Defaults to [`SNAPSHOT_VERSION`] when missing, so snapshots
    /// written by pre-versioned (legacy) readers/writers can still
    /// be deserialized cleanly.
    #[serde(default = "default_snapshot_version")]
    pub version: u32,
    /// The name of the workflow that produced this snapshot.
    pub workflow_name: String,
    /// Unique identifier for the workflow run.
    pub run_id: Uuid,
    /// When the snapshot was captured.
    pub timestamp: DateTime<Utc>,
    /// The context's key/value state at snapshot time.
    pub context_state: HashMap<String, StateValue>,
    /// The fan-in collected events at snapshot time.
    pub collected_events: HashMap<String, Vec<serde_json::Value>>,
    /// Events that were pending in the routing channel at snapshot time.
    pub pending_events: Vec<SerializedEvent>,
    /// Arbitrary metadata (includes `run_id`, `workflow_name`, and any
    /// user-defined metadata set via `Context::set_metadata`).
    pub metadata: HashMap<String, serde_json::Value>,
    /// History events recorded up to this snapshot point.
    ///
    /// Only populated when the `telemetry` feature is enabled and history
    /// collection was turned on via [`WorkflowBuilder::with_history`].
    #[cfg(feature = "telemetry")]
    #[serde(default)]
    pub history: Vec<blazen_telemetry::HistoryEvent>,
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
    /// does not match the expected schema, or
    /// [`WorkflowError::SnapshotVersionMismatch`] if the snapshot was
    /// written by a newer version of `blazen-core` than this reader
    /// supports.
    pub fn from_json(json: &str) -> Result<Self, WorkflowError> {
        let snapshot: Self = serde_json::from_str(json).map_err(WorkflowError::Serialization)?;
        if snapshot.version > SNAPSHOT_VERSION {
            return Err(WorkflowError::SnapshotVersionMismatch {
                snapshot: snapshot.version,
                supported: SNAPSHOT_VERSION,
            });
        }
        Ok(snapshot)
    }

    /// Serialize the snapshot to `MessagePack` bytes.
    ///
    /// `MessagePack` is a compact binary format that is especially efficient
    /// for [`StateValue::Bytes`] data since `serde_bytes` avoids per-byte
    /// overhead.
    ///
    /// Uses the field-named (map) encoding so that snapshots remain
    /// forward-compatible: readers can skip unknown fields and apply
    /// `#[serde(default)]` for missing ones (e.g. legacy snapshots
    /// written before [`SNAPSHOT_VERSION`] existed).
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::BinarySerialization`] if serialization fails.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, WorkflowError> {
        rmp_serde::to_vec_named(self).map_err(|e| WorkflowError::BinarySerialization(e.to_string()))
    }

    /// Deserialize a snapshot from `MessagePack` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::BinarySerialization`] if the bytes are
    /// malformed or do not match the expected schema, or
    /// [`WorkflowError::SnapshotVersionMismatch`] if the snapshot was
    /// written by a newer version of `blazen-core` than this reader
    /// supports.
    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, WorkflowError> {
        let snapshot: Self = rmp_serde::from_slice(bytes)
            .map_err(|e| WorkflowError::BinarySerialization(e.to_string()))?;
        if snapshot.version > SNAPSHOT_VERSION {
            return Err(WorkflowError::SnapshotVersionMismatch {
                snapshot: snapshot.version,
                supported: SNAPSHOT_VERSION,
            });
        }
        Ok(snapshot)
    }

    /// Returns the pending input request, if the workflow paused for human input.
    ///
    /// When a workflow auto-pauses due to an [`InputRequestEvent`], the
    /// request is stored in the snapshot's metadata under the
    /// `"__input_request"` key. This method extracts and deserializes it.
    #[must_use]
    pub fn input_request(&self) -> Option<InputRequestEvent> {
        self.metadata
            .get("__input_request")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
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

        // Convert StateValue map to serde_json::Value map for the persist
        // layer. Binary values are serialized as JSON so they can be stored
        // in the checkpoint's JSON-based state field.
        let state = snap
            .context_state
            .into_iter()
            .map(|(k, v)| {
                let json = serde_json::to_value(&v).unwrap_or(serde_json::Value::Null);
                (k, json)
            })
            .collect();

        blazen_persist::WorkflowCheckpoint {
            workflow_name: snap.workflow_name,
            run_id: snap.run_id,
            timestamp: snap.timestamp,
            state,
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

        // Convert the checkpoint's serde_json::Value map back to StateValue.
        // Try to deserialize each value as a StateValue first (preserving
        // Bytes variants); fall back to wrapping as StateValue::Json.
        let context_state = cp
            .state
            .into_iter()
            .map(|(k, v)| {
                let sv =
                    serde_json::from_value::<StateValue>(v.clone()).unwrap_or(StateValue::Json(v));
                (k, sv)
            })
            .collect();

        WorkflowSnapshot {
            version: SNAPSHOT_VERSION,
            workflow_name: cp.workflow_name,
            run_id: cp.run_id,
            timestamp: cp.timestamp,
            context_state,
            collected_events,
            pending_events,
            metadata,
            #[cfg(feature = "telemetry")]
            history: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> WorkflowSnapshot {
        let mut state = HashMap::new();
        state.insert(
            "counter".to_owned(),
            StateValue::Json(serde_json::json!(42)),
        );
        state.insert(
            "name".to_owned(),
            StateValue::Json(serde_json::json!("alice")),
        );

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
            version: SNAPSHOT_VERSION,
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
            #[cfg(feature = "telemetry")]
            history: Vec::new(),
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

    #[test]
    fn msgpack_roundtrip() {
        let snap = sample_snapshot();
        let bytes = snap.to_msgpack().unwrap();
        let restored = WorkflowSnapshot::from_msgpack(&bytes).unwrap();
        assert_eq!(restored.workflow_name, snap.workflow_name);
        assert_eq!(restored.run_id, snap.run_id);
        assert_eq!(restored.context_state, snap.context_state);
        assert_eq!(restored.collected_events, snap.collected_events);
        assert_eq!(restored.pending_events.len(), snap.pending_events.len());
    }

    #[test]
    fn msgpack_with_bytes_roundtrip() {
        use crate::value::BytesWrapper;

        let mut state = HashMap::new();
        state.insert(
            "data".to_owned(),
            StateValue::Bytes(BytesWrapper(vec![0xDE, 0xAD, 0xBE, 0xEF])),
        );
        state.insert("count".to_owned(), StateValue::Json(serde_json::json!(42)));

        let snap = WorkflowSnapshot {
            version: SNAPSHOT_VERSION,
            workflow_name: "bytes_test".to_owned(),
            run_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            context_state: state,
            collected_events: HashMap::new(),
            pending_events: Vec::new(),
            metadata: HashMap::new(),
            #[cfg(feature = "telemetry")]
            history: Vec::new(),
        };

        let bytes = snap.to_msgpack().unwrap();
        let restored = WorkflowSnapshot::from_msgpack(&bytes).unwrap();
        assert_eq!(restored.context_state, snap.context_state);
        assert_eq!(
            restored
                .context_state
                .get("data")
                .unwrap()
                .as_bytes()
                .unwrap(),
            &[0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    fn from_invalid_msgpack_fails() {
        let result = WorkflowSnapshot::from_msgpack(&[0xFF, 0xFF]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // Version field / SnapshotVersionMismatch coverage
    // -----------------------------------------------------------------

    #[test]
    fn snapshot_default_version_is_one() {
        let snap = sample_snapshot();
        assert_eq!(snap.version, 1);
        assert_eq!(snap.version, SNAPSHOT_VERSION);
    }

    #[test]
    fn snapshot_write_includes_version_in_json() {
        let snap = sample_snapshot();
        let json = snap.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["version"], serde_json::json!(SNAPSHOT_VERSION));
    }

    #[test]
    fn snapshot_round_trip_via_json_preserves_version() {
        let snap = sample_snapshot();
        let json = snap.to_json().unwrap();
        let restored = WorkflowSnapshot::from_json(&json).unwrap();
        assert_eq!(restored.version, SNAPSHOT_VERSION);
    }

    #[test]
    fn snapshot_read_rejects_newer_json_version() {
        // Build a snapshot then re-encode with a higher version field.
        let snap = sample_snapshot();
        let json = snap.to_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["version"] = serde_json::json!(999u32);
        let bumped = serde_json::to_string(&value).unwrap();

        let err = WorkflowSnapshot::from_json(&bumped).unwrap_err();
        match err {
            WorkflowError::SnapshotVersionMismatch {
                snapshot,
                supported,
            } => {
                assert_eq!(snapshot, 999);
                assert_eq!(supported, SNAPSHOT_VERSION);
            }
            other => panic!("expected SnapshotVersionMismatch, got: {other:?}"),
        }
    }

    #[test]
    fn snapshot_read_accepts_missing_json_version_defaults_to_one() {
        // Build a snapshot then strip the version field, simulating a
        // pre-versioned (legacy) snapshot.
        let snap = sample_snapshot();
        let json = snap.to_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value.as_object_mut().unwrap().remove("version");
        let stripped = serde_json::to_string(&value).unwrap();

        let restored = WorkflowSnapshot::from_json(&stripped).unwrap();
        assert_eq!(restored.version, 1);
        assert_eq!(restored.workflow_name, snap.workflow_name);
    }

    #[test]
    fn snapshot_write_includes_version_in_msgpack() {
        // Decode just the `version` field by deserializing into a
        // partial shadow struct. `to_vec_named` writes a map keyed
        // by field name, and rmp-serde will skip unknown fields when
        // the target struct does not request them. This proves the
        // field is on the wire and is not just being synthesized at
        // read time by `#[serde(default)]`.
        #[derive(Deserialize)]
        struct VersionOnly {
            version: u32,
        }

        let snap = sample_snapshot();
        let bytes = snap.to_msgpack().unwrap();

        let probe: VersionOnly = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(probe.version, SNAPSHOT_VERSION);

        // And the standard reader still round-trips cleanly.
        let restored = WorkflowSnapshot::from_msgpack(&bytes).unwrap();
        assert_eq!(restored.version, SNAPSHOT_VERSION);
    }

    #[test]
    fn snapshot_read_rejects_newer_msgpack_version() {
        // Construct a snapshot with version=999, encode with msgpack,
        // and verify the reader rejects it.
        let mut snap = sample_snapshot();
        snap.version = 999;
        let bytes = snap.to_msgpack().unwrap();

        let err = WorkflowSnapshot::from_msgpack(&bytes).unwrap_err();
        match err {
            WorkflowError::SnapshotVersionMismatch {
                snapshot,
                supported,
            } => {
                assert_eq!(snapshot, 999);
                assert_eq!(supported, SNAPSHOT_VERSION);
            }
            other => panic!("expected SnapshotVersionMismatch, got: {other:?}"),
        }
    }

    #[test]
    fn snapshot_read_accepts_missing_msgpack_version_defaults_to_one() {
        // Build a struct WITHOUT the version field by serializing a
        // shadow type that omits it, then read it back through the
        // versioned reader. The serde default should kick in.
        #[derive(Serialize)]
        struct LegacySnapshot {
            workflow_name: String,
            run_id: Uuid,
            timestamp: DateTime<Utc>,
            context_state: HashMap<String, StateValue>,
            collected_events: HashMap<String, Vec<serde_json::Value>>,
            pending_events: Vec<SerializedEvent>,
            metadata: HashMap<String, serde_json::Value>,
            #[cfg(feature = "telemetry")]
            history: Vec<blazen_telemetry::HistoryEvent>,
        }

        let snap = sample_snapshot();
        let legacy = LegacySnapshot {
            workflow_name: snap.workflow_name.clone(),
            run_id: snap.run_id,
            timestamp: snap.timestamp,
            context_state: snap.context_state.clone(),
            collected_events: snap.collected_events.clone(),
            pending_events: snap.pending_events.clone(),
            metadata: snap.metadata.clone(),
            #[cfg(feature = "telemetry")]
            history: snap.history.clone(),
        };

        // Encode using the field-named map form, which is what the
        // versioned writer uses too. A legacy producer that wrote
        // positional msgpack with no `version` field cannot be
        // decoded by a versioned reader because positional encoding
        // requires a fixed field order; the on-wire format MUST be
        // map-encoded for forward compatibility.
        let bytes = rmp_serde::to_vec_named(&legacy).unwrap();
        let restored = WorkflowSnapshot::from_msgpack(&bytes).unwrap();
        assert_eq!(restored.version, 1);
        assert_eq!(restored.workflow_name, snap.workflow_name);
    }
}
