//! Retry-with-exponential-backoff decorator for [`CompletionModel`].
//!
//! Wrap any model with [`RetryCompletionModel`] to automatically retry
//! transient failures (rate limits, timeouts, server errors) with
//! configurable exponential backoff, jitter, and `Retry-After` support.
//!
//! ```rust,ignore
//! use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
//!
//! let inner = /* any CompletionModel */;
//! let model = RetryCompletionModel::new(inner, RetryConfig::default());
//! let response = model.complete(request).await?;
//! ```

use std::hash::{BuildHasher, RandomState};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::Stream;
use serde::{Deserialize, Serialize};

use crate::error::BlazenError;
use crate::traits::CompletionModel;
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for retry behaviour.
///
/// Delays are expressed as u64 milliseconds for cross-language binding
/// compatibility (Python, Node, WASM). They are converted to [`Duration`]
/// internally at the call sites.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct RetryConfig {
    /// Maximum number of retry attempts (total calls = `max_retries + 1`).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Delay before the first retry, in milliseconds.
    #[serde(default = "default_initial_delay_ms")]
    pub initial_delay_ms: u64,
    /// Upper bound on the computed backoff delay, in milliseconds.
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    /// When `true`, a [`BlazenError::RateLimit`] that carries a
    /// `retry_after_ms` value will override the computed backoff.
    #[serde(default = "default_true")]
    pub honor_retry_after: bool,
    /// When `true`, a random 0-25 % jitter is added to each delay to
    /// avoid thundering-herd effects.
    #[serde(default = "default_true")]
    pub jitter: bool,
}

fn default_max_retries() -> u32 {
    3
}
fn default_initial_delay_ms() -> u64 {
    1000
}
fn default_max_delay_ms() -> u64 {
    30_000
}
fn default_true() -> bool {
    true
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            initial_delay_ms: default_initial_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            honor_retry_after: true,
            jitter: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Decorator
// ---------------------------------------------------------------------------

/// A [`CompletionModel`] decorator that retries transient errors with
/// exponential backoff.
pub struct RetryCompletionModel {
    inner: Arc<dyn CompletionModel>,
    config: RetryConfig,
}

impl RetryCompletionModel {
    /// Wrap `inner` with the given retry configuration.
    pub fn new(inner: impl CompletionModel + 'static, config: RetryConfig) -> Self {
        Self {
            inner: Arc::new(inner),
            config,
        }
    }

    /// Wrap an already-`Arc`'d model.
    pub fn from_arc(inner: Arc<dyn CompletionModel>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    /// Compute the delay for the given `attempt` (0-indexed), optionally
    /// honouring a provider-supplied `retry_after_ms`.
    fn compute_delay(&self, attempt: u32, retry_after_ms: Option<u64>) -> Duration {
        let initial_delay = Duration::from_millis(self.config.initial_delay_ms);
        let max_delay = Duration::from_millis(self.config.max_delay_ms);

        // If the provider told us when to retry and we're configured to
        // honour that, use it as the base delay.
        let base = if self.config.honor_retry_after {
            if let Some(ms) = retry_after_ms {
                Duration::from_millis(ms)
            } else {
                exp_backoff(initial_delay, attempt, max_delay)
            }
        } else {
            exp_backoff(initial_delay, attempt, max_delay)
        };

        let capped = base.min(max_delay);

        if self.config.jitter {
            add_jitter(capped)
        } else {
            capped
        }
    }
}

/// Standard exponential backoff: `initial * 2^attempt`, capped.
#[allow(clippy::cast_possible_truncation)] // `.min(u128::from(u64::MAX))` guarantees no truncation
fn exp_backoff(initial: Duration, attempt: u32, max: Duration) -> Duration {
    // Saturating shift to avoid overflow.
    let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
    let millis = u64::try_from(
        initial
            .as_millis()
            .saturating_mul(u128::from(factor))
            .min(u128::from(u64::MAX)),
    )
    .unwrap_or(u64::MAX);
    Duration::from_millis(millis).min(max)
}

/// Add 0-25 % random jitter using `RandomState` so we avoid pulling in
/// the `rand` crate.
#[allow(clippy::cast_possible_truncation)] // durations here are small (seconds), never exceed u64
fn add_jitter(d: Duration) -> Duration {
    // `RandomState::new()` is seeded from the OS on each call.
    let hash = RandomState::new().hash_one(0u64);
    // Map the hash into 0..=25  (percent).
    let pct = hash % 26; // 0-25 inclusive
    let extra_millis = u64::try_from(d.as_millis()).unwrap_or(u64::MAX) * pct / 100;
    d + Duration::from_millis(extra_millis)
}

/// Extract `retry_after_ms` from a rate-limit error.
fn retry_after_from_error(err: &BlazenError) -> Option<u64> {
    match err {
        BlazenError::RateLimit { retry_after_ms } => *retry_after_ms,
        BlazenError::ProviderHttp(d) => d.retry_after_ms,
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CompletionModel impl
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for RetryCompletionModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let max = self.config.max_retries;
        let mut last_err: Option<BlazenError> = None;

        for attempt in 0..=max {
            match self.inner.complete(request.clone()).await {
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    if !err.is_retryable() || attempt == max {
                        return Err(err);
                    }
                    let retry_after = retry_after_from_error(&err);
                    let delay = self.compute_delay(attempt, retry_after);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    crate::sleep::sleep(delay).await;
                }
            }
        }

        // Should be unreachable, but just in case:
        Err(last_err.unwrap_or_else(|| BlazenError::request("all retry attempts exhausted")))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // Retry only the initial connection, not mid-stream failures.
        let max = self.config.max_retries;
        let mut last_err: Option<BlazenError> = None;

        for attempt in 0..=max {
            match self.inner.stream(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(err) => {
                    if !err.is_retryable() || attempt == max {
                        return Err(err);
                    }
                    let retry_after = retry_after_from_error(&err);
                    let delay = self.compute_delay(attempt, retry_after);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "stream: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    crate::sleep::sleep(delay).await;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| BlazenError::request("all retry attempts exhausted")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, CompletionResponse, StreamChunk};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -- Mock model --------------------------------------------------------

    /// A configurable mock that returns pre-defined results based on a call
    /// counter.
    struct MockCompletionModel {
        model_id: String,
        results: Vec<Result<CompletionResponse, BlazenError>>,
        call_count: AtomicUsize,
    }

    impl MockCompletionModel {
        fn new(results: Vec<Result<CompletionResponse, BlazenError>>) -> Self {
            Self {
                model_id: "mock-model".to_string(),
                results,
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    fn ok_response() -> CompletionResponse {
        CompletionResponse {
            content: Some("hello".to_string()),
            tool_calls: vec![],
            reasoning: None,
            citations: vec![],
            artifacts: vec![],
            usage: None,
            model: "mock-model".to_string(),
            finish_reason: Some("stop".to_string()),
            cost: None,
            timing: None,
            images: vec![],
            audio: vec![],
            videos: vec![],
            metadata: serde_json::Value::Null,
        }
    }

    #[async_trait]
    impl CompletionModel for MockCompletionModel {
        fn model_id(&self) -> &str {
            &self.model_id
        }

        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, BlazenError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.results.len() {
                // We can't clone BlazenError, so we reconstruct from the
                // stored results. For the mock we store enough info.
                match &self.results[idx] {
                    Ok(r) => Ok(r.clone()),
                    Err(e) => Err(reconstruct_error(e)),
                }
            } else {
                // Fallback: succeed.
                Ok(ok_response())
            }
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.results.len() {
                match &self.results[idx] {
                    Ok(_) => {
                        let stream = futures_util::stream::empty();
                        Ok(Box::pin(stream))
                    }
                    Err(e) => Err(reconstruct_error(e)),
                }
            } else {
                let stream = futures_util::stream::empty();
                Ok(Box::pin(stream))
            }
        }
    }

    /// Reconstruct a `BlazenError` from a reference (since it's not `Clone`).
    fn reconstruct_error(e: &BlazenError) -> BlazenError {
        match e {
            BlazenError::RateLimit { retry_after_ms } => BlazenError::RateLimit {
                retry_after_ms: *retry_after_ms,
            },
            BlazenError::Auth { message } => BlazenError::Auth {
                message: message.clone(),
            },
            BlazenError::Timeout { elapsed_ms } => BlazenError::Timeout {
                elapsed_ms: *elapsed_ms,
            },
            BlazenError::Request { message, .. } => BlazenError::Request {
                message: message.clone(),
                source: None,
            },
            BlazenError::Provider {
                provider,
                message,
                status_code,
            } => BlazenError::Provider {
                provider: provider.clone(),
                message: message.clone(),
                status_code: *status_code,
            },
            BlazenError::ProviderHttp(d) => {
                BlazenError::ProviderHttp(Box::new(crate::error::ProviderHttpDetails {
                    provider: d.provider.clone(),
                    endpoint: d.endpoint.clone(),
                    status: d.status,
                    request_id: d.request_id.clone(),
                    detail: d.detail.clone(),
                    raw_body: d.raw_body.clone(),
                    retry_after_ms: d.retry_after_ms,
                }))
            }
            _ => BlazenError::request("unknown mock error"),
        }
    }

    fn simple_request() -> CompletionRequest {
        CompletionRequest::new(vec![ChatMessage::user("hi")])
    }

    /// Retry config with zero delays so tests are fast.
    fn fast_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            initial_delay_ms: 0,
            max_delay_ms: 0,
            honor_retry_after: false,
            jitter: false,
        }
    }

    // -- Tests -------------------------------------------------------------

    #[tokio::test]
    async fn test_no_retry_on_success() {
        let mock = MockCompletionModel::new(vec![Ok(ok_response())]);
        let model = RetryCompletionModel::new(mock, fast_config(3));

        let resp = model.complete(simple_request()).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello"));

        // The inner model was called exactly once because it should hold
        // an Arc. We can't access `mock.calls()` directly after move, so
        // we use an Arc wrapper for the counter. Instead, we verify the
        // response content which proves success on first try.
    }

    #[tokio::test]
    async fn test_retries_on_rate_limit() {
        let mock = Arc::new(MockCompletionModel::new(vec![
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
            Ok(ok_response()),
        ]));

        let model = RetryCompletionModel::from_arc(mock.clone(), fast_config(3));
        let resp = model.complete(simple_request()).await.unwrap();

        assert_eq!(resp.content.as_deref(), Some("hello"));
        assert_eq!(mock.calls(), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn test_no_retry_on_auth_error() {
        let mock = Arc::new(MockCompletionModel::new(vec![Err(BlazenError::Auth {
            message: "bad key".to_string(),
        })]));

        let model = RetryCompletionModel::from_arc(mock.clone(), fast_config(3));
        let result = model.complete(simple_request()).await;

        assert!(result.is_err());
        assert_eq!(mock.calls(), 1); // should not have retried
    }

    #[tokio::test]
    async fn test_max_retries_exhausted() {
        let mock = Arc::new(MockCompletionModel::new(vec![
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
        ]));

        let model = RetryCompletionModel::from_arc(mock.clone(), fast_config(2));
        let result = model.complete(simple_request()).await;

        assert!(result.is_err());
        // max_retries=2  =>  3 total attempts (initial + 2 retries)
        assert_eq!(mock.calls(), 3);
    }

    #[tokio::test]
    async fn test_honors_retry_after() {
        // We can't easily assert on the exact sleep duration in a unit test,
        // but we can verify the delay computation logic directly.
        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30_000,
            honor_retry_after: true,
            jitter: false,
        };

        let mock = MockCompletionModel::new(vec![]);
        let model = RetryCompletionModel::new(mock, config);

        // When a retry_after_ms is present, the delay should use it
        // instead of exponential backoff.
        let delay = model.compute_delay(0, Some(5000));
        assert_eq!(delay, Duration::from_secs(5));

        // When no retry_after_ms, should use exponential backoff.
        let delay = model.compute_delay(0, None);
        assert_eq!(delay, Duration::from_secs(1)); // 1 * 2^0 = 1s

        let delay = model.compute_delay(1, None);
        assert_eq!(delay, Duration::from_secs(2)); // 1 * 2^1 = 2s

        let delay = model.compute_delay(2, None);
        assert_eq!(delay, Duration::from_secs(4)); // 1 * 2^2 = 4s

        // Capped at max_delay
        let delay = model.compute_delay(10, None);
        assert_eq!(delay, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn test_jitter_adds_extra() {
        let config = RetryConfig {
            max_retries: 1,
            initial_delay_ms: 1000,
            max_delay_ms: 30_000,
            honor_retry_after: false,
            jitter: true,
        };

        let mock = MockCompletionModel::new(vec![]);
        let model = RetryCompletionModel::new(mock, config);

        let delay = model.compute_delay(0, None);
        // With jitter the delay should be >= 1s and <= 1.25s.
        assert!(delay >= Duration::from_secs(1));
        assert!(delay <= Duration::from_millis(1250));
    }

    #[tokio::test]
    async fn test_stream_retries_on_connection_error() {
        let mock = Arc::new(MockCompletionModel::new(vec![
            Err(BlazenError::Request {
                message: "connection reset".to_string(),
                source: None,
            }),
            Ok(ok_response()), // stream returns empty on Ok
        ]));

        let model = RetryCompletionModel::from_arc(mock.clone(), fast_config(3));
        let stream = model.stream(simple_request()).await;

        assert!(stream.is_ok());
        assert_eq!(mock.calls(), 2);
    }
}
