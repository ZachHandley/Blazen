//! Low-level compute job types: request, handle, status, result, timing.

use napi_derive::napi;

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
