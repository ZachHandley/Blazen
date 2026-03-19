//! Provider fallback wrapper that tries multiple [`CompletionModel`] providers
//! in order.
//!
//! When the primary provider fails with a *retryable* error (rate limit,
//! timeout, transient server failure) the [`FallbackModel`] automatically
//! forwards the request to the next provider in line. Non-retryable errors
//! (authentication, validation, content policy) short-circuit immediately so
//! that broken credentials are not masked by a fallback attempt.
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use blazen_llm::fallback::FallbackModel;
//!
//! let model = FallbackModel::new(vec![
//!     Arc::new(primary_provider),
//!     Arc::new(backup_provider),
//! ]);
//! let response = model.complete(request).await?;
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use crate::error::BlazenError;
use crate::traits::CompletionModel;
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

/// A [`CompletionModel`] that tries multiple providers in order, falling back
/// on retryable failures.
pub struct FallbackModel {
    providers: Vec<Arc<dyn CompletionModel>>,
}

impl FallbackModel {
    /// Create a new `FallbackModel` from one or more providers.
    ///
    /// # Panics
    ///
    /// Panics if `providers` is empty.
    #[must_use]
    pub fn new(providers: Vec<Arc<dyn CompletionModel>>) -> Self {
        assert!(
            !providers.is_empty(),
            "FallbackModel requires at least one provider"
        );
        Self { providers }
    }
}

#[async_trait]
impl CompletionModel for FallbackModel {
    fn model_id(&self) -> &str {
        // Always report the primary provider's model id.
        self.providers[0].model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let last_idx = self.providers.len() - 1;
        let mut last_error: Option<BlazenError> = None;

        for (i, provider) in self.providers.iter().enumerate() {
            let req = request.clone();
            match provider.complete(req).await {
                Ok(response) => return Ok(response),
                Err(err) => {
                    if !err.is_retryable() {
                        return Err(err);
                    }

                    if i < last_idx {
                        let next = &self.providers[i + 1];
                        tracing::info!(
                            "fallback: provider {} failed, trying {}",
                            provider.model_id(),
                            next.model_id(),
                        );
                    }

                    last_error = Some(err);
                }
            }
        }

        // All providers exhausted -- return the last error.
        Err(last_error.expect("at least one provider must be present"))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let last_idx = self.providers.len() - 1;
        let mut last_error: Option<BlazenError> = None;

        for (i, provider) in self.providers.iter().enumerate() {
            let req = request.clone();
            match provider.stream(req).await {
                Ok(stream) => return Ok(stream),
                Err(err) => {
                    if !err.is_retryable() {
                        return Err(err);
                    }

                    if i < last_idx {
                        let next = &self.providers[i + 1];
                        tracing::info!(
                            "fallback: provider {} failed, trying {}",
                            provider.model_id(),
                            next.model_id(),
                        );
                    }

                    last_error = Some(err);
                }
            }
        }

        Err(last_error.expect("at least one provider must be present"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::error::BlazenError;
    use crate::types::{CompletionRequest, CompletionResponse};

    // -- Mock provider -------------------------------------------------------

    /// A mock [`CompletionModel`] with a configurable sequence of results and
    /// an invocation counter.
    struct MockCompletionModel {
        id: String,
        results: Vec<Result<CompletionResponse, BlazenError>>,
        call_count: AtomicUsize,
    }

    impl MockCompletionModel {
        fn new(id: &str, results: Vec<Result<CompletionResponse, BlazenError>>) -> Self {
            Self {
                id: id.to_owned(),
                results,
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl CompletionModel for MockCompletionModel {
        fn model_id(&self) -> &str {
            &self.id
        }

        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, BlazenError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let idx = idx.min(self.results.len() - 1);

            // We need to produce owned values; manually reconstruct.
            match &self.results[idx] {
                Ok(resp) => Ok(resp.clone()),
                Err(e) => Err(clone_error(e)),
            }
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let idx = idx.min(self.results.len() - 1);

            match &self.results[idx] {
                Ok(_) => {
                    let stream = futures_util::stream::empty();
                    Ok(Box::pin(stream))
                }
                Err(e) => Err(clone_error(e)),
            }
        }
    }

    // -- Helpers --------------------------------------------------------------

    /// Produce a new `BlazenError` mimicking the shape of `err`.
    fn clone_error(err: &BlazenError) -> BlazenError {
        match err {
            BlazenError::RateLimit { retry_after_ms } => BlazenError::RateLimit {
                retry_after_ms: *retry_after_ms,
            },
            BlazenError::Auth { message } => BlazenError::Auth {
                message: message.clone(),
            },
            BlazenError::Timeout { elapsed_ms } => BlazenError::Timeout {
                elapsed_ms: *elapsed_ms,
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
            BlazenError::Validation { field, message } => BlazenError::Validation {
                field: field.clone(),
                message: message.clone(),
            },
            BlazenError::Request { message, .. } => BlazenError::Request {
                message: message.clone(),
                source: None,
            },
            other => BlazenError::Provider {
                provider: "mock".into(),
                message: format!("{other}"),
                status_code: None,
            },
        }
    }

    fn ok_response(content: &str) -> CompletionResponse {
        CompletionResponse {
            content: Some(content.to_owned()),
            tool_calls: vec![],
            usage: None,
            model: "mock".to_owned(),
            finish_reason: Some("stop".to_owned()),
            cost: None,
            timing: None,
            images: vec![],
            audio: vec![],
            videos: vec![],
            metadata: serde_json::Value::Null,
        }
    }

    fn simple_request() -> CompletionRequest {
        CompletionRequest::new(vec![crate::types::ChatMessage::user("hello")])
    }

    // -- Tests ----------------------------------------------------------------

    #[tokio::test]
    async fn test_uses_first_provider_on_success() {
        let primary = Arc::new(MockCompletionModel::new(
            "primary",
            vec![Ok(ok_response("primary-answer"))],
        ));
        let secondary = Arc::new(MockCompletionModel::new(
            "secondary",
            vec![Ok(ok_response("secondary-answer"))],
        ));

        let fallback = FallbackModel::new(vec![primary.clone(), secondary.clone()]);
        let resp = fallback.complete(simple_request()).await.unwrap();

        assert_eq!(resp.content.as_deref(), Some("primary-answer"));
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 0);
    }

    #[tokio::test]
    async fn test_falls_back_on_retryable_error() {
        let primary = Arc::new(MockCompletionModel::new(
            "primary",
            vec![Err(BlazenError::RateLimit {
                retry_after_ms: Some(1000),
            })],
        ));
        let secondary = Arc::new(MockCompletionModel::new(
            "secondary",
            vec![Ok(ok_response("secondary-answer"))],
        ));

        let fallback = FallbackModel::new(vec![primary.clone(), secondary.clone()]);
        let resp = fallback.complete(simple_request()).await.unwrap();

        assert_eq!(resp.content.as_deref(), Some("secondary-answer"));
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 1);
    }

    #[tokio::test]
    async fn test_no_fallback_on_auth_error() {
        let primary = Arc::new(MockCompletionModel::new(
            "primary",
            vec![Err(BlazenError::auth("bad api key"))],
        ));
        let secondary = Arc::new(MockCompletionModel::new(
            "secondary",
            vec![Ok(ok_response("secondary-answer"))],
        ));

        let fallback = FallbackModel::new(vec![primary.clone(), secondary.clone()]);
        let result = fallback.complete(simple_request()).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BlazenError::Auth { .. }));
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 0); // never reached
    }

    #[tokio::test]
    async fn test_all_providers_fail() {
        let primary = Arc::new(MockCompletionModel::new(
            "primary",
            vec![Err(BlazenError::RateLimit {
                retry_after_ms: None,
            })],
        ));
        let secondary = Arc::new(MockCompletionModel::new(
            "secondary",
            vec![Err(BlazenError::Timeout { elapsed_ms: 5000 })],
        ));

        let fallback = FallbackModel::new(vec![primary.clone(), secondary.clone()]);
        let result = fallback.complete(simple_request()).await;

        assert!(result.is_err());
        // Should return the *last* error (timeout from secondary).
        assert!(matches!(result.unwrap_err(), BlazenError::Timeout { .. }));
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 1);
    }

    #[tokio::test]
    async fn test_model_id_returns_primary() {
        let primary = Arc::new(MockCompletionModel::new(
            "gpt-4o",
            vec![Ok(ok_response("x"))],
        ));
        let secondary = Arc::new(MockCompletionModel::new(
            "claude-opus",
            vec![Ok(ok_response("y"))],
        ));

        let fallback = FallbackModel::new(vec![primary, secondary]);
        assert_eq!(fallback.model_id(), "gpt-4o");
    }
}
