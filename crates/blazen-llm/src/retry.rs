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
// RetryEmbeddingModel
// ---------------------------------------------------------------------------

/// An [`EmbeddingModel`] decorator that retries transient errors with
/// exponential backoff.
pub struct RetryEmbeddingModel {
    inner: Arc<dyn crate::traits::EmbeddingModel>,
    config: RetryConfig,
}

impl RetryEmbeddingModel {
    /// Wrap `inner` with the given retry configuration.
    pub fn new(inner: impl crate::traits::EmbeddingModel + 'static, config: RetryConfig) -> Self {
        Self {
            inner: Arc::new(inner),
            config,
        }
    }

    /// Wrap an already-`Arc`'d embedding model.
    #[must_use]
    pub fn from_arc(inner: Arc<dyn crate::traits::EmbeddingModel>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    /// Compute the delay for a given attempt.
    fn compute_delay(&self, attempt: u32, retry_after_ms: Option<u64>) -> Duration {
        let initial_delay = Duration::from_millis(self.config.initial_delay_ms);
        let max_delay = Duration::from_millis(self.config.max_delay_ms);
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

#[async_trait]
impl crate::traits::EmbeddingModel for RetryEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed(
        &self,
        texts: &[String],
    ) -> Result<crate::types::EmbeddingResponse, BlazenError> {
        let max = self.config.max_retries;
        let mut last_err: Option<BlazenError> = None;
        for attempt in 0..=max {
            match self.inner.embed(texts).await {
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
                        "embed retrying after {}ms, attempt {}/{}",
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

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        // The decorator's own config is held by-value; we don't expose an Arc
        // here so we return None — callers reading retry_config from a Retry-
        // wrapped model are usually trying to layer another retry on top,
        // which is a no-op anyway.
        None
    }
}

// ---------------------------------------------------------------------------
// Layered resolver
// ---------------------------------------------------------------------------

/// Resolve the effective retry configuration from a stack of scopes.
///
/// Scope precedence (most specific wins): `call > step > workflow > pipeline > provider`.
/// When all scopes are `None`, returns `Arc::new(RetryConfig::default())`.
///
/// The returned value is `Arc`-shared so callers can pass it through to a
/// [`RetryCompletionModel`] / `RetryEmbeddingModel` decorator without
/// cloning the underlying config.
#[must_use]
pub fn resolve_retry(
    call: Option<&Arc<RetryConfig>>,
    step: Option<&Arc<RetryConfig>>,
    workflow: Option<&Arc<RetryConfig>>,
    pipeline: Option<&Arc<RetryConfig>>,
    provider: Option<&Arc<RetryConfig>>,
) -> Arc<RetryConfig> {
    if let Some(c) = call {
        return Arc::clone(c);
    }
    if let Some(c) = step {
        return Arc::clone(c);
    }
    if let Some(c) = workflow {
        return Arc::clone(c);
    }
    if let Some(c) = pipeline {
        return Arc::clone(c);
    }
    if let Some(c) = provider {
        return Arc::clone(c);
    }
    Arc::new(RetryConfig::default())
}

/// A snapshot of every scope's retry configuration, for callers that want
/// to capture the stack before resolving against a per-call override.
#[derive(Debug, Clone, Default)]
pub struct RetryStack {
    /// Provider-level default (lowest priority).
    pub provider: Option<Arc<RetryConfig>>,
    /// Pipeline-level default.
    pub pipeline: Option<Arc<RetryConfig>>,
    /// Workflow-level override.
    pub workflow: Option<Arc<RetryConfig>>,
    /// Step-level override (highest priority before the per-call override).
    pub step: Option<Arc<RetryConfig>>,
}

impl RetryStack {
    /// Return the effective retry config given an optional per-call override.
    #[must_use]
    pub fn resolve(&self, call: Option<&Arc<RetryConfig>>) -> Arc<RetryConfig> {
        resolve_retry(
            call,
            self.step.as_ref(),
            self.workflow.as_ref(),
            self.pipeline.as_ref(),
            self.provider.as_ref(),
        )
    }
}

// ---------------------------------------------------------------------------
// RetryHttpClient
// ---------------------------------------------------------------------------

/// An [`HttpClient`] decorator that retries transient errors (rate limits,
/// timeouts, 5xx responses, network errors) with exponential backoff.
///
/// Use this anywhere you'd otherwise hold an `Arc<dyn HttpClient>` and want
/// retries on raw HTTP — including non-LLM use cases like
/// [`http_fetch.rs`](crate::http_fetch) bridge code, memory backends, or
/// custom data fetchers.
///
/// Streaming requests are not retried mid-stream; only the initial connection
/// (the call to `send_streaming`) is retried.
#[derive(Debug)]
pub struct RetryHttpClient {
    inner: Arc<dyn crate::http::HttpClient>,
    config: RetryConfig,
}

impl RetryHttpClient {
    /// Wrap an existing client with the given retry config.
    #[must_use]
    pub fn new(inner: Arc<dyn crate::http::HttpClient>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    /// Wrap `Self` in an `Arc` for use as `Arc<dyn HttpClient>`.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn crate::http::HttpClient> {
        Arc::new(self)
    }

    fn compute_delay(&self, attempt: u32, retry_after_ms: Option<u64>) -> Duration {
        let initial_delay = Duration::from_millis(self.config.initial_delay_ms);
        let max_delay = Duration::from_millis(self.config.max_delay_ms);
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

#[async_trait]
impl crate::http::HttpClient for RetryHttpClient {
    async fn send(
        &self,
        request: crate::http::HttpRequest,
    ) -> Result<crate::http::HttpResponse, BlazenError> {
        let max = self.config.max_retries;
        let mut last_err: Option<BlazenError> = None;
        for attempt in 0..=max {
            match self.inner.send(request.clone()).await {
                Ok(resp) => {
                    // Surface 4xx/5xx as a `Provider`-like retryable signal
                    // when the status code indicates a server / rate-limit
                    // failure. Below 400 is success.
                    let status = resp.status;
                    if status < 400 || status == 401 || status == 403 || status == 404 {
                        return Ok(resp);
                    }
                    let retryable = matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504);
                    if !retryable || attempt == max {
                        return Ok(resp);
                    }
                    let retry_after = parse_retry_after_header(&resp);
                    let delay = self.compute_delay(attempt, retry_after);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        status,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        "http retrying after {}ms, status {}, attempt {}/{}",
                        delay.as_millis(),
                        status,
                        attempt + 1,
                        max,
                    );
                    last_err = Some(BlazenError::Provider {
                        provider: "http".into(),
                        message: format!("status {status}"),
                        status_code: Some(status),
                    });
                    crate::sleep::sleep(delay).await;
                }
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
                        "http retrying after {}ms, attempt {}/{}",
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

    async fn send_streaming(
        &self,
        request: crate::http::HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, crate::http::ByteStream), BlazenError> {
        // Retry only the initial connect.
        let max = self.config.max_retries;
        let mut last_err: Option<BlazenError> = None;
        for attempt in 0..=max {
            match self.inner.send_streaming(request.clone()).await {
                Ok(t) => return Ok(t),
                Err(err) => {
                    if !err.is_retryable() || attempt == max {
                        return Err(err);
                    }
                    let retry_after = retry_after_from_error(&err);
                    let delay = self.compute_delay(attempt, retry_after);
                    last_err = Some(err);
                    crate::sleep::sleep(delay).await;
                    let _ = attempt; // suppress unused-var clippy
                }
            }
        }
        Err(last_err.unwrap_or_else(|| BlazenError::request("all retry attempts exhausted")))
    }

    fn config(&self) -> &crate::http::HttpClientConfig {
        self.inner.config()
    }
}

/// Parse a `Retry-After` header (in seconds) into milliseconds.
fn parse_retry_after_header(resp: &crate::http::HttpResponse) -> Option<u64> {
    let v = resp.header("Retry-After")?;
    // Numeric: seconds.
    if let Ok(secs) = v.trim().parse::<u64>() {
        return Some(secs.saturating_mul(1000));
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::EmbeddingModel as _;
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

    #[test]
    fn resolver_prefers_call_over_all_others() {
        let call = Arc::new(RetryConfig {
            max_retries: 1,
            ..RetryConfig::default()
        });
        let step = Arc::new(RetryConfig {
            max_retries: 2,
            ..RetryConfig::default()
        });
        let workflow = Arc::new(RetryConfig {
            max_retries: 3,
            ..RetryConfig::default()
        });
        let pipeline = Arc::new(RetryConfig {
            max_retries: 4,
            ..RetryConfig::default()
        });
        let provider = Arc::new(RetryConfig {
            max_retries: 5,
            ..RetryConfig::default()
        });
        let r = resolve_retry(
            Some(&call),
            Some(&step),
            Some(&workflow),
            Some(&pipeline),
            Some(&provider),
        );
        assert_eq!(r.max_retries, 1);
    }

    #[test]
    fn resolver_falls_through_in_order() {
        let provider = Arc::new(RetryConfig {
            max_retries: 5,
            ..RetryConfig::default()
        });
        let r = resolve_retry(None, None, None, None, Some(&provider));
        assert_eq!(r.max_retries, 5);

        let pipeline = Arc::new(RetryConfig {
            max_retries: 4,
            ..RetryConfig::default()
        });
        let r = resolve_retry(None, None, None, Some(&pipeline), Some(&provider));
        assert_eq!(r.max_retries, 4);
    }

    #[test]
    fn resolver_default_when_all_none() {
        let r = resolve_retry(None, None, None, None, None);
        let d = RetryConfig::default();
        assert_eq!(r.max_retries, d.max_retries);
        assert_eq!(r.initial_delay_ms, d.initial_delay_ms);
    }

    #[test]
    fn retry_stack_resolve_with_call_override() {
        let stack = RetryStack {
            provider: Some(Arc::new(RetryConfig {
                max_retries: 5,
                ..RetryConfig::default()
            })),
            pipeline: Some(Arc::new(RetryConfig {
                max_retries: 4,
                ..RetryConfig::default()
            })),
            workflow: None,
            step: None,
        };
        let call = Arc::new(RetryConfig {
            max_retries: 1,
            ..RetryConfig::default()
        });
        assert_eq!(stack.resolve(Some(&call)).max_retries, 1);
        assert_eq!(stack.resolve(None).max_retries, 4);
    }

    // -- RetryEmbeddingModel tests ----------------------------------------

    struct MockEmbeddingModel {
        model_id: String,
        results: Vec<Result<crate::types::EmbeddingResponse, BlazenError>>,
        call_count: AtomicUsize,
    }

    impl MockEmbeddingModel {
        fn new(results: Vec<Result<crate::types::EmbeddingResponse, BlazenError>>) -> Self {
            Self {
                model_id: "mock-embed".to_string(),
                results,
                call_count: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    fn ok_embedding() -> crate::types::EmbeddingResponse {
        crate::types::EmbeddingResponse {
            embeddings: vec![vec![0.1, 0.2, 0.3]],
            model: "mock-embed".to_string(),
            usage: None,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        }
    }

    #[async_trait]
    impl crate::traits::EmbeddingModel for MockEmbeddingModel {
        fn model_id(&self) -> &str {
            &self.model_id
        }
        fn dimensions(&self) -> usize {
            3
        }

        async fn embed(
            &self,
            _texts: &[String],
        ) -> Result<crate::types::EmbeddingResponse, BlazenError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.results.len() {
                match &self.results[idx] {
                    Ok(r) => Ok(r.clone()),
                    Err(e) => Err(reconstruct_error(e)),
                }
            } else {
                Ok(ok_embedding())
            }
        }
    }

    #[tokio::test]
    async fn test_retry_embedding_no_retry_on_success() {
        let mock = Arc::new(MockEmbeddingModel::new(vec![Ok(ok_embedding())]));
        let model = RetryEmbeddingModel::from_arc(mock.clone(), fast_config(3));
        let resp = model.embed(&["hi".to_string()]).await.unwrap();
        assert_eq!(resp.embeddings.len(), 1);
        assert_eq!(mock.calls(), 1);
    }

    #[tokio::test]
    async fn test_retry_embedding_retries_on_rate_limit() {
        let mock = Arc::new(MockEmbeddingModel::new(vec![
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
            Err(BlazenError::RateLimit {
                retry_after_ms: None,
            }),
            Ok(ok_embedding()),
        ]));
        let model = RetryEmbeddingModel::from_arc(mock.clone(), fast_config(3));
        let resp = model.embed(&["hi".to_string()]).await.unwrap();
        assert_eq!(resp.embeddings.len(), 1);
        assert_eq!(mock.calls(), 3);
    }

    #[tokio::test]
    async fn test_retry_embedding_max_retries_exhausted() {
        let mock = Arc::new(MockEmbeddingModel::new(vec![
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
        let model = RetryEmbeddingModel::from_arc(mock.clone(), fast_config(2));
        let result = model.embed(&["hi".to_string()]).await;
        assert!(result.is_err());
        assert_eq!(mock.calls(), 3);
    }

    // -- RetryHttpClient tests --------------------------------------------

    use crate::http::{ByteStream, HttpClient, HttpRequest, HttpResponse};

    #[derive(Debug)]
    struct MockHttpClient {
        responses: std::sync::Mutex<Vec<Result<HttpResponse, BlazenError>>>,
        call_count: AtomicUsize,
    }

    impl MockHttpClient {
        fn new(responses: Vec<Result<HttpResponse, BlazenError>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl HttpClient for MockHttpClient {
        async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, BlazenError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                return Ok(HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: vec![],
                });
            }
            match q.remove(0) {
                Ok(r) => Ok(r),
                Err(e) => Err(reconstruct_error(&e)),
            }
        }
        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                let stream: ByteStream = Box::pin(futures_util::stream::empty());
                return Ok((200, vec![], stream));
            }
            match q.remove(0) {
                Ok(r) => {
                    let _ = r;
                    let stream: ByteStream = Box::pin(futures_util::stream::empty());
                    Ok((200, vec![], stream))
                }
                Err(e) => Err(reconstruct_error(&e)),
            }
        }
    }

    fn ok_http_response(status: u16) -> HttpResponse {
        HttpResponse {
            status,
            headers: vec![],
            body: vec![],
        }
    }

    #[tokio::test]
    async fn test_retry_http_503_then_success() {
        let mock = Arc::new(MockHttpClient::new(vec![
            Ok(ok_http_response(503)),
            Ok(ok_http_response(200)),
        ]));
        let client = RetryHttpClient::new(mock.clone() as Arc<dyn HttpClient>, fast_config(3));
        let req = HttpRequest::get("https://example.com");
        let resp = client.send(req).await.unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(mock.calls(), 2);
    }

    #[tokio::test]
    async fn test_retry_http_404_does_not_retry() {
        let mock = Arc::new(MockHttpClient::new(vec![Ok(ok_http_response(404))]));
        let client = RetryHttpClient::new(mock.clone() as Arc<dyn HttpClient>, fast_config(3));
        let req = HttpRequest::get("https://example.com");
        let resp = client.send(req).await.unwrap();
        assert_eq!(resp.status, 404);
        assert_eq!(mock.calls(), 1);
    }

    #[tokio::test]
    async fn test_retry_http_max_retries_exhausted_returns_last_response() {
        let mock = Arc::new(MockHttpClient::new(vec![
            Ok(ok_http_response(503)),
            Ok(ok_http_response(503)),
            Ok(ok_http_response(503)),
        ]));
        let client = RetryHttpClient::new(mock.clone() as Arc<dyn HttpClient>, fast_config(2));
        let req = HttpRequest::get("https://example.com");
        let resp = client.send(req).await.unwrap();
        // After exhausting retries, last 503 is returned (not an error).
        assert_eq!(resp.status, 503);
        assert_eq!(mock.calls(), 3);
    }
}
