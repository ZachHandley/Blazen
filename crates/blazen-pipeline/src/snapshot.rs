//! Pipeline snapshot for pause/resume support.
//!
//! A [`PipelineSnapshot`] captures all the state needed to resume a paused
//! pipeline: which stages have completed, the current stage index, any
//! active workflow snapshots, and the shared state.

use std::collections::HashMap;
use std::sync::Arc;

use blazen_core::SessionRefRegistry;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::PipelineError;

fn fresh_empty_registry() -> Arc<SessionRefRegistry> {
    Arc::new(SessionRefRegistry::new())
}

/// The outcome of a single completed stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    /// The name of the stage.
    pub name: String,
    /// The output value produced by the stage's workflow.
    pub output: serde_json::Value,
    /// Whether the stage was skipped due to its condition returning `false`.
    pub skipped: bool,
    /// How long the stage took to execute, in milliseconds.
    pub duration_ms: u64,
}

/// A snapshot of an in-progress workflow within a stage.
///
/// Captured when a pipeline is paused mid-stage so that the workflow
/// can be resumed later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveWorkflowSnapshot {
    /// The name of the stage that owns this workflow.
    pub stage_name: String,
    /// For parallel stages, the name of the specific branch. `None` for
    /// sequential stages.
    pub branch_name: Option<String>,
    /// The underlying workflow snapshot.
    pub workflow_snapshot: blazen_core::WorkflowSnapshot,
}

/// Complete snapshot of a pipeline's state at the moment it was paused.
///
/// Contains everything needed to reconstruct and resume the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    /// The name of the pipeline.
    pub pipeline_name: String,
    /// Unique identifier for this pipeline run.
    pub run_id: Uuid,
    /// When the snapshot was captured.
    pub timestamp: DateTime<Utc>,
    /// Index of the stage that was executing (or about to execute) when
    /// the pipeline was paused.
    pub current_stage_index: usize,
    /// Results from stages that completed before the pause.
    pub completed_stages: Vec<StageResult>,
    /// Snapshots of workflows that were actively executing when the pause
    /// signal arrived.
    pub active_snapshots: Vec<ActiveWorkflowSnapshot>,
    /// The shared key/value state at pause time.
    pub shared_state: HashMap<String, serde_json::Value>,
    /// The original pipeline input.
    pub input: serde_json::Value,
}

impl PipelineSnapshot {
    /// Serialize the snapshot to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Serialization`] if serialization fails.
    pub fn to_json(&self) -> Result<String, PipelineError> {
        serde_json::to_string(self).map_err(PipelineError::Serialization)
    }

    /// Serialize the snapshot to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Serialization`] if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, PipelineError> {
        serde_json::to_string_pretty(self).map_err(PipelineError::Serialization)
    }

    /// Deserialize a snapshot from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Serialization`] if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self, PipelineError> {
        serde_json::from_str(json).map_err(PipelineError::Serialization)
    }
}

/// The final output of a successfully completed pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// The name of the pipeline.
    pub pipeline_name: String,
    /// Unique identifier for this pipeline run.
    pub run_id: Uuid,
    /// The output of the last stage (the pipeline's final result).
    pub final_output: serde_json::Value,
    /// Results from every stage, in execution order.
    pub stage_results: Vec<StageResult>,
    /// The shared key/value state at completion time.
    pub shared_state: HashMap<String, serde_json::Value>,
    /// Shared session-ref registry for this pipeline run. Skipped during
    /// serialization; deserialized snapshots get a fresh empty registry
    /// since live refs can't survive cross-process resume.
    #[serde(skip, default = "fresh_empty_registry")]
    pub session_refs: Arc<SessionRefRegistry>,
}
