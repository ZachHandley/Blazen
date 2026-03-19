//! Token usage statistics and request timing metadata.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Token usage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Number of tokens in the prompt / input.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion / output.
    pub completion_tokens: u32,
    /// Total tokens consumed (prompt + completion).
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Request timing
// ---------------------------------------------------------------------------

/// Timing metadata for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTiming {
    /// Time spent waiting in queue (ms), if applicable.
    pub queue_ms: Option<u64>,
    /// Time spent executing the request (ms).
    pub execution_ms: Option<u64>,
    /// Total wall-clock time from submit to response (ms).
    pub total_ms: Option<u64>,
}
