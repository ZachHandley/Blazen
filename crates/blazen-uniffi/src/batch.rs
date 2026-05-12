//! Batch completion surface for the UniFFI bindings.
//!
//! Thin wrapper around upstream [`blazen_llm::batch::complete_batch`], which
//! runs N [`CompletionRequest`]s through a single [`CompletionModel`] with
//! bounded concurrency and aggregates usage / cost across the successful
//! responses.
//!
//! ## Result ordering and partial failure
//!
//! Responses are returned in the same order as the input requests. Individual
//! requests may fail without short-circuiting the rest of the batch — each
//! slot in [`BatchResult::responses`] is a [`BatchItem::Success`] or
//! [`BatchItem::Failure`] independent of its siblings. This matches the
//! upstream contract (`Vec<Result<CompletionResponse, _>>` with per-slot
//! errors).
//!
//! ## Reused upstream API
//!
//! No reimplementation of the fan-out happens here; the parallelism, the
//! concurrency semaphore, and the usage / cost accumulators all live in
//! upstream `complete_batch`.

use std::sync::Arc;

use blazen_llm::batch::{
    BatchConfig as CoreBatchConfig, BatchResult as CoreBatchResult,
    complete_batch as core_complete_batch,
};
use blazen_llm::types::CompletionRequest as CoreCompletionRequest;

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::{CompletionModel, CompletionRequest, CompletionResponse, TokenUsage};
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// Per-request outcome within a [`BatchResult`].
///
/// Slot `i` of [`BatchResult::responses`] corresponds to input request `i`.
/// Successful slots carry the [`CompletionResponse`]; failed slots carry an
/// `error_message` only (the structured `BlazenError` variant doesn't survive
/// nesting inside a `uniffi::Enum` cleanly across all four target languages,
/// so the message is flattened to a string here — foreign callers wanting
/// typed errors should run requests individually).
#[derive(Debug, Clone, uniffi::Enum)]
pub enum BatchItem {
    /// The request completed and the model returned a response.
    Success { response: CompletionResponse },
    /// The request failed. The message mirrors the `Display` form of the
    /// underlying [`BlazenError`].
    Failure { error_message: String },
}

/// Outcome of a [`complete_batch`] call.
///
/// `total_usage` and `total_cost_usd` aggregate only the successful responses
/// — failed slots contribute zero. When no provider reports cost data the
/// total is `0.0` (the wire format does not distinguish "zero" from "unknown").
#[derive(Debug, Clone, uniffi::Record)]
pub struct BatchResult {
    /// One slot per input request, in the same order.
    pub responses: Vec<BatchItem>,
    /// Aggregated token usage across successful responses.
    pub total_usage: TokenUsage,
    /// Aggregated USD cost across successful responses.
    pub total_cost_usd: f64,
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Run a batch of completion requests with bounded concurrency.
///
/// - `model`: the model to drive (one provider, one model id; for cross-model
///   batches dispatch from foreign code instead).
/// - `requests`: the requests to send, in order. Each is converted to the
///   upstream wire format before dispatch; conversion errors short-circuit
///   the entire batch (the request list is rejected as a whole, since a bad
///   schema means the batch was misconfigured).
/// - `max_concurrency`: hard cap on in-flight requests. `0` means unlimited
///   (all dispatched in parallel).
///
/// Returns a [`BatchResult`] with per-request outcomes and aggregated
/// usage / cost. Individual request failures appear in
/// [`BatchResult::responses`] as [`BatchItem::Failure`] — they do not cause
/// this function itself to return `Err`.
///
/// # Errors
///
/// Returns [`BlazenError::Validation`] if any input request fails to convert
/// to the upstream wire format (typically a malformed `parameters_json` or
/// `response_format_json` payload).
#[uniffi::export(async_runtime = "tokio")]
pub async fn complete_batch(
    model: Arc<CompletionModel>,
    requests: Vec<CompletionRequest>,
    max_concurrency: u32,
) -> BlazenResult<BatchResult> {
    let core_requests = build_core_requests(requests)?;
    let config = CoreBatchConfig::new(max_concurrency as usize);
    let core_result = core_complete_batch(model.inner.as_ref(), core_requests, config).await;
    Ok(batch_result_from_core(core_result))
}

/// Synchronous variant of [`complete_batch`] — blocks the current thread on
/// the shared Tokio runtime.
///
/// # Errors
///
/// Same as [`complete_batch`].
#[uniffi::export]
pub fn complete_batch_blocking(
    model: Arc<CompletionModel>,
    requests: Vec<CompletionRequest>,
    max_concurrency: u32,
) -> BlazenResult<BatchResult> {
    runtime().block_on(async move { complete_batch(model, requests, max_concurrency).await })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert wire [`CompletionRequest`]s into upstream [`CoreCompletionRequest`]s.
///
/// Errors out as soon as any request fails to convert — a malformed request
/// in the middle of a batch usually means the entire batch is misconfigured.
fn build_core_requests(
    requests: Vec<CompletionRequest>,
) -> BlazenResult<Vec<CoreCompletionRequest>> {
    requests
        .into_iter()
        .map(CoreCompletionRequest::try_from)
        .collect()
}

/// Convert an upstream [`CoreBatchResult`] into the wire [`BatchResult`].
fn batch_result_from_core(core: CoreBatchResult) -> BatchResult {
    let responses = core
        .responses
        .into_iter()
        .map(|slot| match slot {
            Ok(resp) => BatchItem::Success {
                response: CompletionResponse::from(resp),
            },
            Err(err) => BatchItem::Failure {
                error_message: BlazenError::from(err).to_string(),
            },
        })
        .collect();
    BatchResult {
        responses,
        total_usage: core.total_usage.map(TokenUsage::from).unwrap_or_default(),
        total_cost_usd: core.total_cost.unwrap_or(0.0),
    }
}
