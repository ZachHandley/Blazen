//! Workflow checkpoint and persisted event bindings.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::Result;
use napi_derive::napi;
use uuid::Uuid;

use blazen_persist::{SerializedEvent, WorkflowCheckpoint};

use crate::error::to_napi_error;

/// A serialized representation of an event for persistence.
#[napi(object, js_name = "PersistedEvent")]
pub struct JsPersistedEvent {
    /// The event type identifier (e.g. `"blazen::StartEvent"`).
    #[napi(js_name = "eventType")]
    pub event_type: String,
    /// The event data as a JSON value.
    pub data: serde_json::Value,
}

impl From<SerializedEvent> for JsPersistedEvent {
    fn from(value: SerializedEvent) -> Self {
        Self {
            event_type: value.event_type,
            data: value.data,
        }
    }
}

impl From<JsPersistedEvent> for SerializedEvent {
    fn from(value: JsPersistedEvent) -> Self {
        Self {
            event_type: value.event_type,
            data: value.data,
        }
    }
}

/// Plain-object payload accepted by `WorkflowCheckpoint.create()` for
/// constructing a checkpoint from JS.
#[napi(object)]
pub struct JsWorkflowCheckpointInit {
    /// The name of the workflow that produced this checkpoint.
    #[napi(js_name = "workflowName")]
    pub workflow_name: String,
    /// Unique identifier for this workflow run (RFC 4122 UUID string). If
    /// omitted a new v4 UUID is generated.
    #[napi(js_name = "runId")]
    pub run_id: Option<String>,
    /// Optional ISO 8601 timestamp. Defaults to the current time.
    pub timestamp: Option<String>,
    /// Serialized context state as a JSON object.
    pub state: Option<serde_json::Value>,
    /// Events in the queue at checkpoint time.
    #[napi(js_name = "pendingEvents")]
    pub pending_events: Option<Vec<JsPersistedEvent>>,
    /// Arbitrary metadata attached to this checkpoint as a JSON object.
    pub metadata: Option<serde_json::Value>,
}

/// A snapshot of workflow state at a point in time.
#[napi(js_name = "WorkflowCheckpoint")]
pub struct JsWorkflowCheckpoint {
    pub(crate) inner: WorkflowCheckpoint,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsWorkflowCheckpoint {
    /// Construct a checkpoint from a plain options object.
    #[napi(factory)]
    pub fn create(init: JsWorkflowCheckpointInit) -> Result<Self> {
        let run_id = match init.run_id {
            Some(s) => Uuid::parse_str(&s).map_err(to_napi_error)?,
            None => Uuid::new_v4(),
        };
        let timestamp = match init.timestamp {
            Some(s) => DateTime::parse_from_rfc3339(&s)
                .map_err(to_napi_error)?
                .with_timezone(&Utc),
            None => Utc::now(),
        };
        let state = json_object_to_map(init.state)?;
        let metadata = json_object_to_map(init.metadata)?;
        let pending_events = init
            .pending_events
            .unwrap_or_default()
            .into_iter()
            .map(SerializedEvent::from)
            .collect();
        Ok(Self {
            inner: WorkflowCheckpoint {
                workflow_name: init.workflow_name,
                run_id,
                timestamp,
                state,
                pending_events,
                metadata,
            },
        })
    }

    /// Parse a checkpoint from its JSON representation.
    #[napi(factory, js_name = "fromJson")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_json(json: String) -> Result<Self> {
        let inner: WorkflowCheckpoint = serde_json::from_str(&json).map_err(to_napi_error)?;
        Ok(Self { inner })
    }

    /// Serialize this checkpoint to a JSON string.
    #[napi(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.inner).map_err(to_napi_error)
    }

    /// Serialize this checkpoint to a pretty-printed JSON string.
    #[napi(js_name = "toJsonPretty")]
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.inner).map_err(to_napi_error)
    }

    #[napi(getter, js_name = "workflowName")]
    pub fn workflow_name(&self) -> String {
        self.inner.workflow_name.clone()
    }

    #[napi(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    #[napi(getter)]
    pub fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    #[napi(getter)]
    pub fn state(&self) -> serde_json::Value {
        let map: HashMap<String, serde_json::Value> = self.inner.state.clone();
        serde_json::Value::Object(map.into_iter().collect())
    }

    #[napi(getter, js_name = "pendingEvents")]
    pub fn pending_events(&self) -> Vec<JsPersistedEvent> {
        self.inner
            .pending_events
            .iter()
            .cloned()
            .map(JsPersistedEvent::from)
            .collect()
    }

    #[napi(getter)]
    pub fn metadata(&self) -> serde_json::Value {
        let map: HashMap<String, serde_json::Value> = self.inner.metadata.clone();
        serde_json::Value::Object(map.into_iter().collect())
    }
}

impl JsWorkflowCheckpoint {
    pub(crate) fn from_inner(inner: WorkflowCheckpoint) -> Self {
        Self { inner }
    }

    pub(crate) fn inner_ref(&self) -> &WorkflowCheckpoint {
        &self.inner
    }
}

fn json_object_to_map(
    value: Option<serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(HashMap::new()),
        Some(serde_json::Value::Object(map)) => Ok(map.into_iter().collect()),
        Some(_) => Err(napi::Error::from_reason(
            "expected a JSON object for state/metadata",
        )),
    }
}
