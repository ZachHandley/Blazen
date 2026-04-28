//! JavaScript bindings for batch completion execution.
//!
//! Exposes [`complete_batch`] which wraps the Rust
//! [`blazen_llm::batch::complete_batch`] function, allowing Node.js /
//! TypeScript users to run multiple completion requests concurrently with
//! optional concurrency limits.

use napi_derive::napi;

use blazen_llm::batch::{
    BatchConfig, BatchResult as RustBatchResult, complete_batch as rust_complete_batch,
};
use blazen_llm::types::CompletionRequest;

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
// JsBatchConfig (typed class)
// ---------------------------------------------------------------------------

/// Typed configuration for a batch completion run.
///
/// Mirrors [`blazen_llm::batch::BatchConfig`] and exposes the static
/// `unlimited()` factory.
///
/// ```typescript
/// import { BatchConfig, completeBatchConfig } from 'blazen';
///
/// const cfg = new BatchConfig(4);             // up to 4 concurrent requests
/// const cfg2 = BatchConfig.unlimited();        // no limit
/// ```
#[napi(js_name = "BatchConfig")]
pub struct JsBatchConfig {
    pub(crate) inner: BatchConfig,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsBatchConfig {
    /// Create a batch config with the given concurrency limit.
    ///
    /// Pass `0` (or call `BatchConfig.unlimited()`) for unlimited concurrency.
    #[napi(constructor)]
    pub fn new(concurrency: u32) -> Self {
        Self {
            inner: BatchConfig::new(concurrency as usize),
        }
    }

    /// Create a batch config with unlimited concurrency.
    #[napi(factory)]
    pub fn unlimited() -> Self {
        Self {
            inner: BatchConfig::unlimited(),
        }
    }

    /// Maximum number of concurrent requests (`0` means unlimited).
    #[napi(getter)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn concurrency(&self) -> u32 {
        self.inner.concurrency as u32
    }
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
#[napi(js_name = "BatchResult")]
pub struct JsBatchResult {
    inner: RustBatchResult,
    // Cached error strings, generated once at construction.
    error_strings: Vec<Option<String>>,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsBatchResult {
    /// One response per input request. `null` for failed requests.
    #[napi(getter)]
    pub fn responses(&self) -> Vec<Option<JsCompletionResponse>> {
        self.inner
            .responses
            .iter()
            .map(|r| r.as_ref().ok().cloned().map(build_response))
            .collect()
    }

    /// One error message per input request. `null` for successful requests.
    #[napi(getter)]
    pub fn errors(&self) -> Vec<Option<String>> {
        self.error_strings.clone()
    }

    /// Aggregated token usage across all successful responses.
    #[napi(getter, js_name = "totalUsage")]
    pub fn total_usage(&self) -> Option<JsTokenUsage> {
        self.inner.total_usage.as_ref().map(|u| u.clone().into())
    }

    /// Aggregated cost in USD across all successful responses.
    #[napi(getter, js_name = "totalCost")]
    pub fn total_cost(&self) -> Option<f64> {
        self.inner.total_cost
    }

    /// Number of successful requests in the batch.
    #[napi(getter, js_name = "successCount")]
    pub fn success_count(&self) -> u32 {
        self.inner
            .responses
            .iter()
            .filter(|r| r.is_ok())
            .count()
            .try_into()
            .unwrap_or(u32::MAX)
    }

    /// Number of failed requests in the batch.
    #[napi(getter, js_name = "failureCount")]
    pub fn failure_count(&self) -> u32 {
        self.inner
            .responses
            .iter()
            .filter(|r| r.is_err())
            .count()
            .try_into()
            .unwrap_or(u32::MAX)
    }

    /// Total number of requests in the batch.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.responses.len().try_into().unwrap_or(u32::MAX)
    }

    /// Human-readable summary of the batch result.
    #[napi(js_name = "toString")]
    pub fn display_string(&self) -> String {
        format!(
            "BatchResult(length={}, successCount={}, totalCost={:?})",
            self.length(),
            self.success_count(),
            self.inner.total_cost
        )
    }
}

impl JsBatchResult {
    pub(crate) fn from_rust(inner: RustBatchResult) -> Self {
        let error_strings = inner
            .responses
            .iter()
            .map(|r| r.as_ref().err().map(ToString::to_string))
            .collect();
        Self {
            inner,
            error_strings,
        }
    }
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

    Ok(JsBatchResult::from_rust(result))
}

/// Run a batch using a typed [`JsBatchConfig`] instance instead of an options
/// dict.
///
/// Equivalent to [`complete_batch`] but takes a `BatchConfig` class instance.
/// Useful when constructing the config programmatically or sharing it across
/// multiple calls.
///
/// ```typescript
/// import { CompletionModel, ChatMessage, BatchConfig, completeBatchConfig } from 'blazen';
///
/// const cfg = new BatchConfig(4);
/// const result = await completeBatchConfig(model, messageSets, cfg);
/// ```
#[napi(js_name = "completeBatchConfig")]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub async fn complete_batch_config(
    model: &JsCompletionModel,
    message_sets: Vec<Vec<&JsChatMessage>>,
    config: &JsBatchConfig,
) -> napi::Result<JsBatchResult> {
    let requests: Vec<CompletionRequest> = message_sets
        .into_iter()
        .map(|msgs| {
            let chat_messages = msgs.iter().map(|m| m.inner.clone()).collect();
            CompletionRequest::new(chat_messages)
        })
        .collect();

    let cfg = BatchConfig::new(config.inner.concurrency);

    let inner = model.inner.as_ref().ok_or_else(|| {
        napi::Error::from_reason(
            "completeBatchConfig() is not supported on subclassed CompletionModel instances",
        )
    })?;
    let result = rust_complete_batch(inner.as_ref(), requests, cfg).await;

    Ok(JsBatchResult::from_rust(result))
}
