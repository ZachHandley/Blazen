//! Batch completion execution with bounded concurrency.
//!
//! Provides [`complete_batch`] for running multiple [`CompletionRequest`]s
//! through a [`CompletionModel`] with an optional concurrency limit.

use std::sync::Arc;

use crate::error::BlazenError;
use crate::traits::CompletionModel;
use crate::types::{CompletionRequest, CompletionResponse, TokenUsage};

/// Configuration for a batch completion run.
pub struct BatchConfig {
    /// Maximum number of concurrent requests. `0` means unlimited.
    pub concurrency: usize,
}

impl BatchConfig {
    /// Create a new batch config with the given concurrency limit.
    #[must_use]
    pub fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    /// Create a batch config with unlimited concurrency.
    #[must_use]
    pub fn unlimited() -> Self {
        Self { concurrency: 0 }
    }
}

impl Default for BatchConfig {
    /// Defaults to unlimited concurrency.
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Result of a batch completion run.
pub struct BatchResult {
    /// One result per input request, in the same order. Individual requests
    /// may fail independently.
    pub responses: Vec<Result<CompletionResponse, BlazenError>>,
    /// Aggregated token usage across all successful responses.
    pub total_usage: Option<TokenUsage>,
    /// Aggregated cost across all successful responses.
    pub total_cost: Option<f64>,
}

/// Execute multiple completion requests with bounded concurrency.
///
/// Results are returned in the same order as the input requests. Individual
/// requests may fail without affecting others — check each element of
/// [`BatchResult::responses`].
///
/// # Panics
///
/// Panics if the internal concurrency semaphore is closed, which should never
/// happen under normal usage since the semaphore is owned by this function.
///
/// # Example
///
/// ```rust,ignore
/// use blazen_llm::{CompletionRequest, ChatMessage};
/// use blazen_llm::batch::{complete_batch, BatchConfig};
///
/// let requests = vec![
///     CompletionRequest::new(vec![ChatMessage::user("Question 1")]),
///     CompletionRequest::new(vec![ChatMessage::user("Question 2")]),
///     CompletionRequest::new(vec![ChatMessage::user("Question 3")]),
/// ];
///
/// let result = complete_batch(&model, requests, BatchConfig::new(2)).await;
/// for (i, response) in result.responses.iter().enumerate() {
///     match response {
///         Ok(r) => println!("Response {i}: {}", r.content.as_deref().unwrap_or("")),
///         Err(e) => eprintln!("Request {i} failed: {e}"),
///     }
/// }
/// ```
pub async fn complete_batch(
    model: &dyn CompletionModel,
    requests: Vec<CompletionRequest>,
    config: BatchConfig,
) -> BatchResult {
    if requests.is_empty() {
        return BatchResult {
            responses: vec![],
            total_usage: None,
            total_cost: None,
        };
    }

    // Build a semaphore for bounded concurrency (None = unlimited).
    let semaphore = if config.concurrency > 0 {
        Some(Arc::new(tokio::sync::Semaphore::new(config.concurrency)))
    } else {
        None
    };

    // Execute all requests concurrently, preserving order via `join_all`.
    let futures: Vec<_> = requests
        .into_iter()
        .map(|request| {
            let sem = semaphore.clone();
            async move {
                let _permit = match sem {
                    Some(ref s) => Some(s.acquire().await.expect("semaphore closed")),
                    None => None,
                };
                model.complete(request).await
            }
        })
        .collect();

    let results = futures_util::future::join_all(futures).await;

    // Aggregate usage and cost across successful responses.
    let mut total_usage: Option<TokenUsage> = None;
    let mut total_cost: Option<f64> = None;

    for response in results.iter().flatten() {
        if let Some(usage) = &response.usage {
            if let Some(ref mut existing) = total_usage {
                existing.prompt_tokens += usage.prompt_tokens;
                existing.completion_tokens += usage.completion_tokens;
                existing.total_tokens += usage.total_tokens;
                existing.reasoning_tokens += usage.reasoning_tokens;
                existing.cached_input_tokens += usage.cached_input_tokens;
                existing.audio_input_tokens += usage.audio_input_tokens;
                existing.audio_output_tokens += usage.audio_output_tokens;
            } else {
                total_usage = Some(usage.clone());
            }
        }
        if let Some(cost) = response.cost {
            *total_cost.get_or_insert(0.0) += cost;
        }
    }

    BatchResult {
        responses: results,
        total_usage,
        total_cost,
    }
}
