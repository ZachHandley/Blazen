//! Retry decorator for [`MemoryBackend`].
//!
//! [`RetryMemoryBackend`] wraps any `Arc<dyn MemoryBackend>` with retry-on-
//! transient-error behavior using [`blazen_llm::retry::RetryConfig`]. Backend
//! errors do not currently distinguish transient vs. permanent failures, so
//! every [`MemoryError`] is treated as retryable up to `max_retries`.
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use blazen_llm::retry::RetryConfig;
//! use blazen_memory::{InMemoryBackend, MemoryBackend, RetryMemoryBackend};
//!
//! let inner: Arc<dyn MemoryBackend> = Arc::new(InMemoryBackend::new());
//! let backend = RetryMemoryBackend::new(inner, RetryConfig::default()).into_arc();
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use blazen_llm::retry::RetryConfig;

use crate::error::{MemoryError, Result};
use crate::store::MemoryBackend;
use crate::types::StoredEntry;

// ---------------------------------------------------------------------------
// Decorator
// ---------------------------------------------------------------------------

/// A [`MemoryBackend`] decorator that retries transient errors with
/// exponential backoff.
///
/// Wraps any `Arc<dyn MemoryBackend>`. Because [`MemoryError`] does not
/// currently expose retryability information, every error is retried up to
/// `config.max_retries` times. Delays follow the same exponential-backoff
/// formula as [`blazen_llm::retry::RetryCompletionModel`] but without the
/// jitter / `Retry-After` plumbing — backend errors don't carry retry-after
/// metadata.
pub struct RetryMemoryBackend {
    inner: Arc<dyn MemoryBackend>,
    config: RetryConfig,
}

impl RetryMemoryBackend {
    /// Wrap `inner` with the given retry configuration.
    #[must_use]
    pub fn new(inner: Arc<dyn MemoryBackend>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    /// Consume `self` and return the decorated backend as an `Arc<dyn MemoryBackend>`.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn MemoryBackend> {
        Arc::new(self)
    }

    /// Compute the delay for the given `attempt` (0-indexed) using
    /// exponential backoff capped at `config.max_delay_ms`.
    #[allow(clippy::cast_possible_truncation)]
    // `.min(u128::from(u64::MAX))` guarantees no truncation
    fn compute_delay(&self, attempt: u32) -> Duration {
        let initial = Duration::from_millis(self.config.initial_delay_ms);
        let max = Duration::from_millis(self.config.max_delay_ms);
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
}

/// Whether a [`MemoryError`] should trigger a retry.
///
/// [`MemoryError`] does not currently distinguish transient vs. permanent
/// failures, so we retry everything. This is a conservative default — most
/// backend transient failures (network blips, lock contention, file-system
/// races) benefit from retry, and permanent errors will simply exhaust the
/// retry budget quickly.
fn is_retryable(_err: &MemoryError) -> bool {
    true
}

// ---------------------------------------------------------------------------
// MemoryBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl MemoryBackend for RetryMemoryBackend {
    async fn put(&self, entry: StoredEntry) -> Result<()> {
        let max = self.config.max_retries;
        let mut last_err: Option<MemoryError> = None;
        for attempt in 0..=max {
            match self.inner.put(entry.clone()).await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !is_retryable(&err) || attempt == max {
                        return Err(err);
                    }
                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "memory backend put: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| MemoryError::Backend("all retry attempts exhausted".to_string())))
    }

    async fn get(&self, id: &str) -> Result<Option<StoredEntry>> {
        let max = self.config.max_retries;
        let mut last_err: Option<MemoryError> = None;
        for attempt in 0..=max {
            match self.inner.get(id).await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !is_retryable(&err) || attempt == max {
                        return Err(err);
                    }
                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "memory backend get: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| MemoryError::Backend("all retry attempts exhausted".to_string())))
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let max = self.config.max_retries;
        let mut last_err: Option<MemoryError> = None;
        for attempt in 0..=max {
            match self.inner.delete(id).await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !is_retryable(&err) || attempt == max {
                        return Err(err);
                    }
                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "memory backend delete: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| MemoryError::Backend("all retry attempts exhausted".to_string())))
    }

    async fn list(&self) -> Result<Vec<StoredEntry>> {
        let max = self.config.max_retries;
        let mut last_err: Option<MemoryError> = None;
        for attempt in 0..=max {
            match self.inner.list().await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !is_retryable(&err) || attempt == max {
                        return Err(err);
                    }
                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "memory backend list: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| MemoryError::Backend("all retry attempts exhausted".to_string())))
    }

    async fn len(&self) -> Result<usize> {
        let max = self.config.max_retries;
        let mut last_err: Option<MemoryError> = None;
        for attempt in 0..=max {
            match self.inner.len().await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !is_retryable(&err) || attempt == max {
                        return Err(err);
                    }
                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "memory backend len: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| MemoryError::Backend("all retry attempts exhausted".to_string())))
    }

    async fn search_by_bands(&self, bands: &[String], limit: usize) -> Result<Vec<StoredEntry>> {
        let max = self.config.max_retries;
        let mut last_err: Option<MemoryError> = None;
        for attempt in 0..=max {
            match self.inner.search_by_bands(bands, limit).await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !is_retryable(&err) || attempt == max {
                        return Err(err);
                    }
                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        error = %err,
                        "memory backend search_by_bands: retrying after {}ms, attempt {}/{}",
                        delay.as_millis(),
                        attempt + 1,
                        max,
                    );
                    last_err = Some(err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| MemoryError::Backend("all retry attempts exhausted".to_string())))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A configurable mock backend that returns pre-defined results based on
    /// a per-method call counter.
    struct MockBackend {
        put_results: Mutex<Vec<Result<()>>>,
        put_calls: AtomicUsize,
    }

    impl MockBackend {
        fn new(put_results: Vec<Result<()>>) -> Self {
            Self {
                put_results: Mutex::new(put_results),
                put_calls: AtomicUsize::new(0),
            }
        }

        fn put_call_count(&self) -> usize {
            self.put_calls.load(Ordering::SeqCst)
        }

        fn next_put_result(&self) -> Result<()> {
            let idx = self.put_calls.fetch_add(1, Ordering::SeqCst);
            let mut results = self.put_results.lock().unwrap();
            if idx < results.len() {
                // Replace with an Ok placeholder so we can move the original
                // out (Result<()> is not Clone in general — MemoryError isn't
                // Clone — so we use mem::replace).
                std::mem::replace(&mut results[idx], Ok(()))
            } else {
                Ok(())
            }
        }
    }

    #[async_trait]
    impl MemoryBackend for MockBackend {
        async fn put(&self, _entry: StoredEntry) -> Result<()> {
            self.next_put_result()
        }

        async fn get(&self, _id: &str) -> Result<Option<StoredEntry>> {
            Ok(None)
        }

        async fn delete(&self, _id: &str) -> Result<bool> {
            Ok(false)
        }

        async fn list(&self) -> Result<Vec<StoredEntry>> {
            Ok(vec![])
        }

        async fn len(&self) -> Result<usize> {
            Ok(0)
        }

        async fn search_by_bands(
            &self,
            _bands: &[String],
            _limit: usize,
        ) -> Result<Vec<StoredEntry>> {
            Ok(vec![])
        }
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

    fn sample_entry() -> StoredEntry {
        StoredEntry {
            id: "e1".to_string(),
            text: "hello".to_string(),
            elid: None,
            simhash_hex: None,
            text_simhash: 0,
            bands: vec!["band0".into()],
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn retry_memory_backend_no_retry_on_success() {
        let mock = Arc::new(MockBackend::new(vec![Ok(())]));
        let backend =
            RetryMemoryBackend::new(mock.clone() as Arc<dyn MemoryBackend>, fast_config(3));

        backend.put(sample_entry()).await.unwrap();
        assert_eq!(mock.put_call_count(), 1);
    }

    #[tokio::test]
    async fn retry_memory_backend_retries_then_succeeds() {
        let mock = Arc::new(MockBackend::new(vec![
            Err(MemoryError::Backend("boom 1".to_string())),
            Err(MemoryError::Backend("boom 2".to_string())),
            Ok(()),
        ]));
        let backend =
            RetryMemoryBackend::new(mock.clone() as Arc<dyn MemoryBackend>, fast_config(3));

        backend.put(sample_entry()).await.unwrap();
        assert_eq!(mock.put_call_count(), 3);
    }

    #[tokio::test]
    async fn retry_memory_backend_max_retries_exhausted() {
        let mock = Arc::new(MockBackend::new(vec![
            Err(MemoryError::Backend("boom 1".to_string())),
            Err(MemoryError::Backend("boom 2".to_string())),
            Err(MemoryError::Backend("boom 3".to_string())),
        ]));
        let backend =
            RetryMemoryBackend::new(mock.clone() as Arc<dyn MemoryBackend>, fast_config(2));

        let result = backend.put(sample_entry()).await;
        assert!(result.is_err());
        // max_retries=2 => 3 total attempts (initial + 2 retries)
        assert_eq!(mock.put_call_count(), 3);
    }

    #[tokio::test]
    async fn retry_memory_backend_into_arc_returns_dyn_backend() {
        let mock: Arc<dyn MemoryBackend> = Arc::new(MockBackend::new(vec![Ok(())]));
        let arc = RetryMemoryBackend::new(mock, fast_config(0)).into_arc();

        // Should be usable as a `dyn MemoryBackend`.
        arc.put(sample_entry()).await.unwrap();
    }
}
