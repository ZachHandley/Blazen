//! Low-level compute job types: request, handle, status, result, timing.

use napi_derive::napi;

/// Input for a generic compute job.
#[napi(object)]
pub struct JsComputeRequest {
    /// The model/endpoint to run (e.g., "fal-ai/flux/dev").
    pub model: String,
    /// Input parameters as JSON (model-specific).
    pub input: serde_json::Value,
    /// Optional webhook URL for async completion notification.
    pub webhook: Option<String>,
}

/// A handle to a submitted compute job.
#[napi(object)]
pub struct JsJobHandle {
    /// Provider-assigned job/request identifier.
    pub id: String,
    /// Provider name (e.g., "fal", "replicate", "runpod").
    pub provider: String,
    /// The model/endpoint that was invoked.
    pub model: String,
    /// When the job was submitted (ISO 8601).
    #[napi(js_name = "submittedAt")]
    pub submitted_at: String,
}

/// Status of a compute job.
#[napi(string_enum)]
pub enum JsJobStatus {
    /// Job is waiting in the provider's queue.
    #[napi(value = "queued")]
    Queued,
    /// Job is currently executing.
    #[napi(value = "running")]
    Running,
    /// Job completed successfully.
    #[napi(value = "completed")]
    Completed,
    /// Job failed with an error.
    #[napi(value = "failed")]
    Failed,
    /// Job was cancelled.
    #[napi(value = "cancelled")]
    Cancelled,
}

/// Result of a completed compute job.
#[napi(object)]
pub struct JsComputeResult {
    /// The job handle that produced this result, if available.
    pub job: Option<JsJobHandle>,
    /// Output data (model-specific JSON).
    pub output: serde_json::Value,
    /// Request timing breakdown.
    pub timing: Option<JsComputeTiming>,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Raw provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// Timing breakdown for a compute request.
#[napi(object)]
pub struct JsComputeTiming {
    /// Time spent waiting in queue, in milliseconds.
    #[napi(js_name = "queueMs")]
    pub queue_ms: Option<i64>,
    /// Time spent executing, in milliseconds.
    #[napi(js_name = "executionMs")]
    pub execution_ms: Option<i64>,
    /// Total wall-clock time, in milliseconds.
    #[napi(js_name = "totalMs")]
    pub total_ms: Option<i64>,
}
