//! JavaScript bindings for batch completion execution.
//!
//! Exposes [`complete_batch`] which wraps the Rust
//! [`blazen_llm::batch::complete_batch`] function, allowing Node.js /
//! TypeScript users to run multiple completion requests concurrently with
//! optional concurrency limits.

use napi_derive::napi;

use blazen_llm::batch::{BatchConfig, complete_batch as rust_complete_batch};
use blazen_llm::types::CompletionRequest;

use crate::error::llm_error_to_napi;
use crate::generated::JsTokenUsage;
use crate::providers::JsCompletionModel;
use crate::types::{JsChatMessage, JsCompletionResponse, build_response};

// ---------------------------------------------------------------------------
// JsBatchOptions
// ---------------------------------------------------------------------------

/// Options for configuring a batch completion run.
#[derive(Default)]
#[napi(object)]
pub struct JsBatchOptions {
    /// Maximum number of concurrent requests. `0` or omitted means unlimited.
    pub concurrency: Option<u32>,
}

// ---------------------------------------------------------------------------
// JsBatchResult
// ---------------------------------------------------------------------------

/// The result of a batch completion run.
///
/// Each index corresponds to the input request at the same position.
/// On success the `responses` entry is populated and the `errors` entry is
/// `null`; on failure the `responses` entry is `null` and `errors` contains
/// the error message.
#[napi(object)]
pub struct JsBatchResult {
    /// One response per input request. `null` for failed requests.
    pub responses: Vec<Option<JsCompletionResponse>>,
    /// One error message per input request. `null` for successful requests.
    pub errors: Vec<Option<String>>,
    /// Aggregated token usage across all successful responses.
    #[napi(js_name = "totalUsage")]
    pub total_usage: Option<JsTokenUsage>,
    /// Aggregated cost in USD across all successful responses.
    #[napi(js_name = "totalCost")]
    pub total_cost: Option<f64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run multiple chat completions concurrently with optional concurrency limits.
///
/// Each element of `messageSets` is an array of `ChatMessage` instances that
/// forms one independent completion request. Results are returned in the same
/// order as the input.
///
/// ```typescript
/// import { CompletionModel, ChatMessage, completeBatch } from 'blazen';
///
/// const model = CompletionModel.openai({ apiKey: "sk-..." });
///
/// const result = await completeBatch(
///   model,
///   [
///     [ChatMessage.user("What is 2 + 2?")],
///     [ChatMessage.user("What is 3 + 3?")],
///     [ChatMessage.user("What is 4 + 4?")],
///   ],
///   { concurrency: 2 }
/// );
///
/// for (let i = 0; i < result.responses.length; i++) {
///   if (result.responses[i]) {
///     console.log(`Response ${i}:`, result.responses[i].content);
///   } else {
///     console.error(`Request ${i} failed:`, result.errors[i]);
///   }
/// }
/// ```
#[napi(js_name = "completeBatch")]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub async fn complete_batch(
    model: &JsCompletionModel,
    message_sets: Vec<Vec<&JsChatMessage>>,
    options: Option<JsBatchOptions>,
) -> napi::Result<JsBatchResult> {
    let requests: Vec<CompletionRequest> = message_sets
        .into_iter()
        .map(|msgs| {
            let chat_messages = msgs.iter().map(|m| m.inner.clone()).collect();
            CompletionRequest::new(chat_messages)
        })
        .collect();

    let config = match options {
        Some(opts) => match opts.concurrency {
            Some(c) if c > 0 => BatchConfig::new(c as usize),
            _ => BatchConfig::unlimited(),
        },
        None => BatchConfig::default(),
    };

    let inner = model.inner.as_ref().ok_or_else(|| {
        napi::Error::from_reason(
            "completeBatch() is not supported on subclassed CompletionModel instances",
        )
    })?;
    let result = rust_complete_batch(inner.as_ref(), requests, config).await;

    let mut responses = Vec::with_capacity(result.responses.len());
    let mut errors = Vec::with_capacity(result.responses.len());

    for res in result.responses {
        match res {
            Ok(response) => {
                responses.push(Some(build_response(response)));
                errors.push(None);
            }
            Err(e) => {
                responses.push(None);
                errors.push(Some(llm_error_to_napi(e).to_string()));
            }
        }
    }

    Ok(JsBatchResult {
        responses,
        errors,
        total_usage: result.total_usage.map(Into::into),
        total_cost: result.total_cost,
    })
}
