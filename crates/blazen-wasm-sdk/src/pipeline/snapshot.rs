//! WASM wrappers for `blazen_pipeline` snapshot and result types.
//!
//! Mirrors the napi binding's class-with-getters layout so JS callers see
//! the same surface across both runtimes:
//!
//! - [`WasmStageResult`] / [`WasmActiveWorkflowSnapshot`] — small read-only
//!   classes with field getters.
//! - [`WasmPipelineSnapshot`] — read/serialise/deserialise plus
//!   `toJson`/`fromJson` round-trip helpers, since callers persist
//!   snapshots.
//! - [`WasmPipelineResult`] — terminal result with the final output and
//!   per-stage outputs.

use std::collections::HashMap;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::pipeline::error::pipeline_err;

// ---------------------------------------------------------------------------
// Marshalling helper
// ---------------------------------------------------------------------------

/// Convert a `Serialize` value into a `JsValue` shaped as a plain JS object
/// (maps marshalled as objects so `JSON.stringify` round-trips cleanly).
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

// ---------------------------------------------------------------------------
// WasmStageResult
// ---------------------------------------------------------------------------

/// Outcome of a single completed pipeline stage.
///
/// Wraps [`blazen_pipeline::StageResult`] and exposes its fields as
/// JS-side getters. Skipped stages have `output === null` and `skipped`
/// set to `true`.
#[wasm_bindgen(js_name = "StageResult")]
pub struct WasmStageResult {
    pub(crate) inner: blazen_pipeline::StageResult,
}

impl WasmStageResult {
    /// Wrap a native [`blazen_pipeline::StageResult`].
    #[must_use]
    pub(crate) fn from_inner(inner: blazen_pipeline::StageResult) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = "StageResult")]
impl WasmStageResult {
    /// The stage's name.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Output value produced by the stage's workflow, or `null` if the
    /// stage was skipped.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling the JSON output to a JS
    /// value fails.
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Result<JsValue, JsValue> {
        marshal_to_js(&self.inner.output)
    }

    /// Whether the stage was skipped due to its condition returning `false`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn skipped(&self) -> bool {
        self.inner.skipped
    }

    /// How long the stage took to execute, in milliseconds.
    /// How long the stage took to execute, in milliseconds.
    ///
    /// Returned as `f64` rather than `u64` because wasm-bindgen surfaces
    /// `u64` as JS `BigInt`, and durations don't need that range. Stage
    /// durations of multiple millennia would overflow this; we accept the
    /// trade-off.
    #[wasm_bindgen(getter, js_name = "durationMs")]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_ms(&self) -> f64 {
        self.inner.duration_ms as f64
    }
}

// ---------------------------------------------------------------------------
// WasmActiveWorkflowSnapshot
// ---------------------------------------------------------------------------

/// Snapshot of an in-progress workflow within a stage.
///
/// Captured when a pipeline is paused mid-stage so the workflow can be
/// resumed later. Wraps [`blazen_pipeline::ActiveWorkflowSnapshot`].
#[wasm_bindgen(js_name = "ActiveWorkflowSnapshot")]
pub struct WasmActiveWorkflowSnapshot {
    pub(crate) inner: blazen_pipeline::ActiveWorkflowSnapshot,
}

impl WasmActiveWorkflowSnapshot {
    /// Wrap a native [`blazen_pipeline::ActiveWorkflowSnapshot`].
    #[must_use]
    pub(crate) fn from_inner(inner: blazen_pipeline::ActiveWorkflowSnapshot) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = "ActiveWorkflowSnapshot")]
impl WasmActiveWorkflowSnapshot {
    /// Name of the stage that owns this workflow.
    #[wasm_bindgen(getter, js_name = "stageName")]
    #[must_use]
    pub fn stage_name(&self) -> String {
        self.inner.stage_name.clone()
    }

    /// For parallel stages, the name of the specific branch. `null` for
    /// sequential stages.
    #[wasm_bindgen(getter, js_name = "branchName")]
    #[must_use]
    pub fn branch_name(&self) -> Option<String> {
        self.inner.branch_name.clone()
    }

    /// JSON serialisation of the underlying [`blazen_core::WorkflowSnapshot`].
    ///
    /// Use [`crate::core_types::snapshot::WasmWorkflowSnapshot::from_json`]
    /// to reconstruct a typed snapshot object on the JS side, or pass the
    /// string directly to `Workflow.resumeFromSnapshot`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the workflow snapshot's serializer
    /// rejects the structure (should not happen in practice; the snapshot
    /// is round-tripped from a previously valid snapshot).
    #[wasm_bindgen(js_name = "workflowSnapshotJson")]
    pub fn workflow_snapshot_json(&self) -> Result<String, JsValue> {
        self.inner
            .workflow_snapshot
            .to_json()
            .map_err(|e| JsValue::from_str(&format!("workflow snapshot serialize failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// WasmPipelineSnapshot
// ---------------------------------------------------------------------------

/// Complete snapshot of a pipeline's state at the moment it was paused.
///
/// Wraps [`blazen_pipeline::PipelineSnapshot`]. Used as both an output
/// (returned by [`WasmPipelineHandler::pause`](super::handler::WasmPipelineHandler::pause))
/// and an input (passed to [`WasmPipeline::resume`](super::pipeline::WasmPipeline::resume)).
///
/// Snapshots can be persisted via [`Self::to_json`] /
/// [`Self::to_json_pretty`] and restored via [`Self::from_json`].
#[wasm_bindgen(js_name = "PipelineSnapshot")]
pub struct WasmPipelineSnapshot {
    pub(crate) inner: blazen_pipeline::PipelineSnapshot,
}

impl WasmPipelineSnapshot {
    /// Wrap a native [`blazen_pipeline::PipelineSnapshot`].
    #[must_use]
    pub(crate) fn from_inner(inner: blazen_pipeline::PipelineSnapshot) -> Self {
        Self { inner }
    }

    /// Borrow the underlying [`blazen_pipeline::PipelineSnapshot`].
    #[must_use]
    pub(crate) fn inner(&self) -> &blazen_pipeline::PipelineSnapshot {
        &self.inner
    }
}

#[wasm_bindgen(js_class = "PipelineSnapshot")]
impl WasmPipelineSnapshot {
    /// Name of the pipeline that produced this snapshot.
    #[wasm_bindgen(getter, js_name = "pipelineName")]
    #[must_use]
    pub fn pipeline_name(&self) -> String {
        self.inner.pipeline_name.clone()
    }

    /// UUID (string) of the pipeline run.
    #[wasm_bindgen(getter, js_name = "runId")]
    #[must_use]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// Wall-clock timestamp at which the snapshot was captured (RFC 3339).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    /// Index of the stage that was executing (or about to execute) when
    /// the pipeline was paused.
    #[wasm_bindgen(getter, js_name = "currentStageIndex")]
    #[must_use]
    pub fn current_stage_index(&self) -> u32 {
        // `usize` -> `u32` truncation is safe in practice: pipelines with
        // billions of stages are not a realistic concern.
        u32::try_from(self.inner.current_stage_index).unwrap_or(u32::MAX)
    }

    /// Results from stages that completed before the pause.
    #[wasm_bindgen(getter, js_name = "completedStages")]
    #[must_use]
    pub fn completed_stages(&self) -> Vec<WasmStageResult> {
        self.inner
            .completed_stages
            .iter()
            .map(|s| WasmStageResult::from_inner(s.clone()))
            .collect()
    }

    /// Snapshots of workflows that were actively executing when the pause
    /// signal arrived.
    #[wasm_bindgen(getter, js_name = "activeSnapshots")]
    #[must_use]
    pub fn active_snapshots(&self) -> Vec<WasmActiveWorkflowSnapshot> {
        self.inner
            .active_snapshots
            .iter()
            .map(|s| WasmActiveWorkflowSnapshot::from_inner(s.clone()))
            .collect()
    }

    /// Original input passed to the pipeline run.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(getter)]
    pub fn input(&self) -> Result<JsValue, JsValue> {
        marshal_to_js(&self.inner.input)
    }

    /// Shared key/value state at the moment of the pause.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(getter, js_name = "sharedState")]
    pub fn shared_state(&self) -> Result<JsValue, JsValue> {
        let map: HashMap<String, serde_json::Value> = self.inner.shared_state.clone();
        let value = serde_json::Value::Object(map.into_iter().collect());
        marshal_to_js(&value)
    }

    /// Serialise the snapshot to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if serialisation fails.
    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.inner.to_json().map_err(pipeline_err)
    }

    /// Serialise the snapshot to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if serialisation fails.
    #[wasm_bindgen(js_name = "toJsonPretty")]
    pub fn to_json_pretty(&self) -> Result<String, JsValue> {
        self.inner.to_json_pretty().map_err(pipeline_err)
    }

    /// Deserialise a snapshot from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JSON is malformed or fails the
    /// snapshot version check.
    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(json: &str) -> Result<WasmPipelineSnapshot, JsValue> {
        blazen_pipeline::PipelineSnapshot::from_json(json)
            .map(|inner| Self { inner })
            .map_err(pipeline_err)
    }
}

// ---------------------------------------------------------------------------
// WasmPipelineResult
// ---------------------------------------------------------------------------

/// Final output of a successfully completed pipeline run.
///
/// Wraps [`blazen_pipeline::PipelineResult`].
#[wasm_bindgen(js_name = "PipelineResult")]
pub struct WasmPipelineResult {
    pub(crate) inner: blazen_pipeline::PipelineResult,
}

impl WasmPipelineResult {
    /// Wrap a native [`blazen_pipeline::PipelineResult`].
    #[must_use]
    pub(crate) fn from_inner(inner: blazen_pipeline::PipelineResult) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = "PipelineResult")]
impl WasmPipelineResult {
    /// Name of the pipeline that produced this result.
    #[wasm_bindgen(getter, js_name = "pipelineName")]
    #[must_use]
    pub fn pipeline_name(&self) -> String {
        self.inner.pipeline_name.clone()
    }

    /// UUID (string) of the pipeline run.
    #[wasm_bindgen(getter, js_name = "runId")]
    #[must_use]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// Output of the last stage (the pipeline's final result).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(getter, js_name = "finalOutput")]
    pub fn final_output(&self) -> Result<JsValue, JsValue> {
        marshal_to_js(&self.inner.final_output)
    }

    /// Per-stage outcomes, in execution order.
    #[wasm_bindgen(getter, js_name = "stageResults")]
    #[must_use]
    pub fn stage_results(&self) -> Vec<WasmStageResult> {
        self.inner
            .stage_results
            .iter()
            .map(|s| WasmStageResult::from_inner(s.clone()))
            .collect()
    }

    /// Shared key/value state at completion time.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(getter, js_name = "sharedState")]
    pub fn shared_state(&self) -> Result<JsValue, JsValue> {
        let map: HashMap<String, serde_json::Value> = self.inner.shared_state.clone();
        let value = serde_json::Value::Object(map.into_iter().collect());
        marshal_to_js(&value)
    }
}
