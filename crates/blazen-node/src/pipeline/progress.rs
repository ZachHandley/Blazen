//! Plain-object napi mirror of [`blazen_pipeline::ProgressSnapshot`].
//!
//! Surfaced via [`super::handler::JsPipelineHandler::progress`] so JS
//! callers can poll a running pipeline's stage cursor without subscribing
//! to the broadcast event stream.

use napi_derive::napi;

use blazen_pipeline::ProgressSnapshot;

/// Lightweight, polled view of a running pipeline's progress.
///
/// Mirrors [`ProgressSnapshot`]. Reads are best-effort and may briefly be
/// one stage behind the actual position because they do not synchronise
/// with the executor task.
#[napi(object, js_name = "ProgressSnapshot")]
pub struct JsProgressSnapshot {
    /// 1-based index of the stage currently executing (or just completed).
    /// `0` before the first stage starts.
    #[napi(js_name = "currentStageIndex")]
    pub current_stage_index: u32,
    /// Total number of stages declared on the pipeline.
    #[napi(js_name = "totalStages")]
    pub total_stages: u32,
    /// Progress as a percentage in `0.0..=100.0`.
    pub percent: f64,
    /// Name of the current stage, when available. Always `null` from the
    /// current atomic-index implementation; reserved for future use.
    #[napi(js_name = "currentStageName")]
    pub current_stage_name: Option<String>,
}

#[allow(clippy::cast_lossless)]
impl From<ProgressSnapshot> for JsProgressSnapshot {
    fn from(s: ProgressSnapshot) -> Self {
        Self {
            current_stage_index: s.current_stage_index,
            total_stages: s.total_stages,
            percent: f64::from(s.percent),
            current_stage_name: s.current_stage_name,
        }
    }
}
