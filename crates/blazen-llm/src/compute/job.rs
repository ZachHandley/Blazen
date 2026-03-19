//! Core compute job types: handles, status, requests, and results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::RequestTiming;

// ---------------------------------------------------------------------------
// Job lifecycle types
// ---------------------------------------------------------------------------

/// A handle to a submitted compute job.
///
/// Returned by [`super::ComputeProvider::submit`] and used to poll status,
/// retrieve results, or cancel the job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobHandle {
    /// Provider-assigned job/request identifier.
    pub id: String,
    /// Provider name (e.g., "fal", "replicate", "runpod").
    pub provider: String,
    /// The model/endpoint that was invoked.
    pub model: String,
    /// When the job was submitted.
    pub submitted_at: DateTime<Utc>,
}

/// Status of a compute job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobStatus {
    /// Job is waiting in the provider's queue.
    Queued,
    /// Job is currently executing.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job failed with an error message.
    Failed {
        /// Human-readable description of the failure.
        error: String,
    },
    /// Job was cancelled.
    Cancelled,
}

/// Input for a compute job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequest {
    /// The model/endpoint to run (e.g., "fal-ai/flux/dev", "stability-ai/sdxl").
    pub model: String,
    /// Input parameters as JSON (model-specific).
    pub input: serde_json::Value,
    /// Optional webhook URL for async completion notification.
    pub webhook: Option<String>,
}

/// Result of a completed compute job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeResult {
    /// The job handle that produced this result, if available.
    pub job: Option<JobHandle>,
    /// Output data (model-specific JSON).
    pub output: serde_json::Value,
    /// Request timing breakdown.
    pub timing: RequestTiming,
    /// Cost in USD, if reported by the provider.
    pub cost: Option<f64>,
    /// Raw provider-specific metadata.
    pub metadata: serde_json::Value,
}
