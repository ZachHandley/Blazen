//! TS-facing wrapper around [`blazen_pipeline::PipelineEvent`].
//!
//! The native [`PipelineEvent`](blazen_pipeline::PipelineEvent) holds a
//! `Box<dyn AnyEvent>` which can't cross the WASM ABI directly. This
//! mirrors the Node binding's flattening: every streamed event becomes a
//! plain `{ stageName, branchName, workflowRunId, event }` JS object where
//! `event` is the underlying event's `to_json()` representation.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

/// Stage-tagged pipeline event, flattened for JS.
///
/// Emitted by [`WasmPipelineHandler::stream_events`](super::handler::WasmPipelineHandler::stream_events)
/// for every event published by a stage's workflow during pipeline
/// execution. The `event` payload is the workflow event in JSON form
/// (the same shape as `event.toJson()` on a native `Event` trait impl).
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmPipelineEvent {
    /// Name of the stage that produced this event.
    pub stage_name: String,
    /// For parallel stages, the name of the specific branch. `None` for
    /// sequential stages.
    pub branch_name: Option<String>,
    /// UUID (string) of the workflow run that produced this event.
    pub workflow_run_id: String,
    /// JSON payload of the wrapped event, with a `type` discriminant under
    /// `event.type` (mirrors `blazen_events::AnyEvent::to_json`).
    pub event: serde_json::Value,
}

impl WasmPipelineEvent {
    /// Build a flattened TS-facing view from a native
    /// [`blazen_pipeline::PipelineEvent`].
    #[must_use]
    pub fn from_native(event: &blazen_pipeline::PipelineEvent) -> Self {
        Self {
            stage_name: event.stage_name.clone(),
            branch_name: event.branch_name.clone(),
            workflow_run_id: event.workflow_run_id.to_string(),
            event: event.event.to_json(),
        }
    }
}
