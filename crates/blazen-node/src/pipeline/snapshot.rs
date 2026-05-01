//! napi-rs wrappers for `blazen_pipeline` snapshot and result types.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::{pipeline_error_to_napi, workflow_error_to_napi};

#[napi(js_name = "StageResult")]
pub struct JsStageResult {
    pub(crate) inner: blazen_pipeline::StageResult,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_wrap
)]
impl JsStageResult {
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[napi(getter)]
    pub fn output(&self) -> serde_json::Value {
        self.inner.output.clone()
    }

    #[napi(getter)]
    pub fn skipped(&self) -> bool {
        self.inner.skipped
    }

    #[napi(getter, js_name = "durationMs")]
    pub fn duration_ms(&self) -> i64 {
        self.inner.duration_ms as i64
    }

    /// Token usage for this stage, if any LLM calls inside the stage
    /// emitted [`UsageEvent`](blazen_events::UsageEvent)s. Mirrors
    /// [`blazen_pipeline::StageResult::usage`] (Wave 3).
    #[napi(getter)]
    pub fn usage(&self) -> Option<crate::types::JsTokenUsageClass> {
        self.inner
            .usage
            .as_ref()
            .map(crate::types::JsTokenUsageClass::from)
    }

    /// Cost in USD for this stage, if known. Mirrors
    /// [`blazen_pipeline::StageResult::cost_usd`] (Wave 3).
    #[napi(getter, js_name = "costUsd")]
    pub fn cost_usd(&self) -> Option<f64> {
        self.inner.cost_usd
    }
}

impl JsStageResult {
    pub(crate) fn from_inner(inner: blazen_pipeline::StageResult) -> Self {
        Self { inner }
    }
}

#[napi(js_name = "ActiveWorkflowSnapshot")]
pub struct JsActiveWorkflowSnapshot {
    pub(crate) inner: blazen_pipeline::ActiveWorkflowSnapshot,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_wrap
)]
impl JsActiveWorkflowSnapshot {
    #[napi(getter, js_name = "stageName")]
    pub fn stage_name(&self) -> String {
        self.inner.stage_name.clone()
    }

    #[napi(getter, js_name = "branchName")]
    pub fn branch_name(&self) -> Option<String> {
        self.inner.branch_name.clone()
    }

    #[napi(js_name = "workflowSnapshotJson")]
    pub fn workflow_snapshot_json(&self) -> Result<String> {
        self.inner
            .workflow_snapshot
            .to_json()
            .map_err(workflow_error_to_napi)
    }
}

impl JsActiveWorkflowSnapshot {
    pub(crate) fn from_inner(inner: blazen_pipeline::ActiveWorkflowSnapshot) -> Self {
        Self { inner }
    }
}

#[napi(js_name = "PipelineSnapshot")]
pub struct JsPipelineSnapshot {
    pub(crate) inner: blazen_pipeline::PipelineSnapshot,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_wrap
)]
impl JsPipelineSnapshot {
    #[napi(getter, js_name = "pipelineName")]
    pub fn pipeline_name(&self) -> String {
        self.inner.pipeline_name.clone()
    }

    #[napi(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    #[napi(getter)]
    pub fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    #[napi(getter, js_name = "currentStageIndex")]
    pub fn current_stage_index(&self) -> i64 {
        self.inner.current_stage_index as i64
    }

    #[napi(getter, js_name = "completedStages")]
    pub fn completed_stages(&self) -> Vec<JsStageResult> {
        self.inner
            .completed_stages
            .iter()
            .map(|s| JsStageResult::from_inner(s.clone()))
            .collect()
    }

    #[napi(getter, js_name = "activeSnapshots")]
    pub fn active_snapshots(&self) -> Vec<JsActiveWorkflowSnapshot> {
        self.inner
            .active_snapshots
            .iter()
            .map(|s| JsActiveWorkflowSnapshot::from_inner(s.clone()))
            .collect()
    }

    #[napi(getter)]
    pub fn input(&self) -> serde_json::Value {
        self.inner.input.clone()
    }

    #[napi(getter, js_name = "sharedState")]
    pub fn shared_state(&self) -> serde_json::Value {
        let map: HashMap<String, serde_json::Value> = self.inner.shared_state.clone();
        serde_json::Value::Object(map.into_iter().collect())
    }

    #[napi(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String> {
        self.inner.to_json().map_err(pipeline_error_to_napi)
    }

    #[napi(js_name = "toJsonPretty")]
    pub fn to_json_pretty(&self) -> Result<String> {
        self.inner.to_json_pretty().map_err(pipeline_error_to_napi)
    }

    #[napi(factory, js_name = "fromJson")]
    pub fn from_json(json: String) -> Result<Self> {
        blazen_pipeline::PipelineSnapshot::from_json(&json)
            .map(|inner| Self { inner })
            .map_err(pipeline_error_to_napi)
    }
}

impl JsPipelineSnapshot {
    pub(crate) fn from_inner(inner: blazen_pipeline::PipelineSnapshot) -> Self {
        Self { inner }
    }
}

#[napi(js_name = "PipelineResult")]
pub struct JsPipelineResult {
    pub(crate) inner: blazen_pipeline::PipelineResult,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_wrap
)]
impl JsPipelineResult {
    #[napi(getter, js_name = "pipelineName")]
    pub fn pipeline_name(&self) -> String {
        self.inner.pipeline_name.clone()
    }

    #[napi(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    #[napi(getter, js_name = "finalOutput")]
    pub fn final_output(&self) -> serde_json::Value {
        self.inner.final_output.clone()
    }

    #[napi(getter, js_name = "stageResults")]
    pub fn stage_results(&self) -> Vec<JsStageResult> {
        self.inner
            .stage_results
            .iter()
            .map(|s| JsStageResult::from_inner(s.clone()))
            .collect()
    }

    #[napi(getter, js_name = "sharedState")]
    pub fn shared_state(&self) -> serde_json::Value {
        let map: HashMap<String, serde_json::Value> = self.inner.shared_state.clone();
        serde_json::Value::Object(map.into_iter().collect())
    }

    /// Aggregated token usage across the pipeline run. Mirrors
    /// [`blazen_pipeline::PipelineResult::usage_total`] (Wave 3).
    #[napi(getter, js_name = "usageTotal")]
    pub fn usage_total(&self) -> crate::types::JsTokenUsageClass {
        crate::types::JsTokenUsageClass::from(&self.inner.usage_total)
    }

    /// Aggregated cost in USD across the pipeline run. Mirrors
    /// [`blazen_pipeline::PipelineResult::cost_total_usd`] (Wave 3).
    #[napi(getter, js_name = "costTotalUsd")]
    pub fn cost_total_usd(&self) -> f64 {
        self.inner.cost_total_usd
    }
}

impl JsPipelineResult {
    pub(crate) fn from_inner(inner: blazen_pipeline::PipelineResult) -> Self {
        Self { inner }
    }
}
