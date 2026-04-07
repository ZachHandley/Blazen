//! Composable middleware system for [`CompletionModel`].
//!
//! The [`Middleware`] trait and [`MiddlewareStack`] builder formalize the
//! existing decorator pattern (retry, cache, fallback) into a composable
//! chain. Each middleware wraps an inner model with additional behaviour
//! (logging, retries, caching, etc.) and the stack applies them in a
//! predictable order.
//!
//! # Ordering
//!
//! Layers are added via [`MiddlewareStack::layer`] and applied so that the
//! **first layer added becomes the outermost wrapper**. This means it
//! executes first on the request path and last on the response path.
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use blazen_llm::middleware::{MiddlewareStack, RetryMiddleware, CacheMiddleware};
//! use blazen_llm::retry::RetryConfig;
//! use blazen_llm::cache::CacheConfig;
//!
//! // Retry wraps cache wraps the raw provider:
//! //   request -> retry -> cache -> provider -> cache -> retry -> response
//! let model = MiddlewareStack::new()
//!     .with_retry(RetryConfig::default())
//!     .with_cache(CacheConfig::default())
//!     .apply(Arc::new(provider));
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use crate::cache::{CacheConfig, CachedCompletionModel};
use crate::retry::{RetryCompletionModel, RetryConfig};
use crate::traits::CompletionModel;

// ---------------------------------------------------------------------------
// Middleware trait
// ---------------------------------------------------------------------------

/// A middleware that wraps a [`CompletionModel`], adding behaviour
/// before/after completion calls.
///
/// Implementors produce a new `Arc<dyn CompletionModel>` that decorates the
/// provided inner model. This is intentionally identical to the decorator
/// pattern already used by [`RetryCompletionModel`] and
/// [`CachedCompletionModel`], but expressed as a composable unit.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Wrap the inner model with this middleware's behaviour.
    fn wrap(&self, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel>;
}

// ---------------------------------------------------------------------------
// MiddlewareStack builder
// ---------------------------------------------------------------------------

/// A builder for composing multiple [`Middleware`] layers around a
/// [`CompletionModel`].
///
/// Layers are applied in reverse insertion order so that the first layer
/// added ends up as the outermost wrapper (executed first).
pub struct MiddlewareStack {
    layers: Vec<Box<dyn Middleware>>,
}

impl Default for MiddlewareStack {
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewareStack {
    /// Create an empty middleware stack.
    #[must_use]
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a middleware layer.
    ///
    /// The first layer added will be the **outermost** wrapper -- it
    /// executes first on the request path and last on the response path.
    #[must_use]
    pub fn layer(mut self, middleware: impl Middleware + 'static) -> Self {
        self.layers.push(Box::new(middleware));
        self
    }

    /// Apply all middleware layers to `model`, returning the fully wrapped
    /// model.
    ///
    /// Layers are applied in reverse order so that the first layer added
    /// becomes the outermost wrapper.
    pub fn apply(self, model: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
        let mut wrapped = model;
        for layer in self.layers.into_iter().rev() {
            wrapped = layer.wrap(wrapped);
        }
        wrapped
    }

    /// Convenience: add a [`RetryMiddleware`] layer.
    #[must_use]
    pub fn with_retry(self, config: RetryConfig) -> Self {
        self.layer(RetryMiddleware { config })
    }

    /// Convenience: add a [`CacheMiddleware`] layer.
    #[must_use]
    pub fn with_cache(self, config: CacheConfig) -> Self {
        self.layer(CacheMiddleware { config })
    }
}

// ---------------------------------------------------------------------------
// Built-in middleware: Retry
// ---------------------------------------------------------------------------

/// Middleware that wraps a model with [`RetryCompletionModel`] for automatic
/// retry-with-exponential-backoff on transient failures.
pub struct RetryMiddleware {
    /// The retry configuration to apply.
    pub config: RetryConfig,
}

impl Middleware for RetryMiddleware {
    fn wrap(&self, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
        Arc::new(RetryCompletionModel::from_arc(inner, self.config.clone()))
    }
}

// ---------------------------------------------------------------------------
// Built-in middleware: Cache
// ---------------------------------------------------------------------------

/// Middleware that wraps a model with [`CachedCompletionModel`] for
/// content-hash-based response caching.
pub struct CacheMiddleware {
    /// The cache configuration to apply.
    pub config: CacheConfig,
}

impl Middleware for CacheMiddleware {
    fn wrap(&self, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
        Arc::new(CachedCompletionModel::from_arc(inner, self.config.clone()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures_util::Stream;

    use super::*;
    use crate::error::BlazenError;
    use crate::types::{ChatMessage, CompletionRequest, CompletionResponse, StreamChunk};

    // -- Mock model ----------------------------------------------------------

    /// A mock [`CompletionModel`] that counts invocations and returns a
    /// configurable model id.
    struct MockCompletionModel {
        id: String,
        call_count: Arc<AtomicUsize>,
    }

    impl MockCompletionModel {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_owned(),
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn call_count(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.call_count)
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
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ok_response())
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(Box::pin(futures_util::stream::empty()))
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
            model: "mock".to_string(),
            finish_reason: Some("stop".to_string()),
            cost: None,
            timing: None,
            images: vec![],
            audio: vec![],
            videos: vec![],
            metadata: serde_json::Value::Null,
        }
    }

    fn simple_request() -> CompletionRequest {
        CompletionRequest::new(vec![ChatMessage::user("hi")])
    }

    // -- Tracking middleware for ordering tests ------------------------------

    /// A test middleware that records its position when `wrap` is called and
    /// when the resulting model's `complete` is invoked, so we can verify
    /// execution ordering.
    struct TrackingMiddleware {
        label: String,
        wrap_order: Arc<std::sync::Mutex<Vec<String>>>,
    }

    /// The model produced by [`TrackingMiddleware`], delegates to inner but
    /// records calls.
    struct TrackingModel {
        label: String,
        inner: Arc<dyn CompletionModel>,
        call_order: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl TrackingMiddleware {
        fn new(label: &str, wrap_order: Arc<std::sync::Mutex<Vec<String>>>) -> Self {
            Self {
                label: label.to_owned(),
                wrap_order,
            }
        }
    }

    impl Middleware for TrackingMiddleware {
        fn wrap(&self, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
            self.wrap_order.lock().unwrap().push(self.label.clone());
            Arc::new(TrackingModel {
                label: self.label.clone(),
                inner,
                call_order: Arc::clone(&self.wrap_order),
            })
        }
    }

    #[async_trait]
    impl CompletionModel for TrackingModel {
        fn model_id(&self) -> &str {
            self.inner.model_id()
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, BlazenError> {
            self.call_order
                .lock()
                .unwrap()
                .push(format!("{}-call", self.label));
            self.inner.complete(request).await
        }

        async fn stream(
            &self,
            request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            self.call_order
                .lock()
                .unwrap()
                .push(format!("{}-stream", self.label));
            self.inner.stream(request).await
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[tokio::test]
    async fn test_empty_stack() {
        let mock = MockCompletionModel::new("base-model");
        let counter = mock.call_count();
        let model: Arc<dyn CompletionModel> = Arc::new(mock);

        let wrapped = MiddlewareStack::new().apply(model);

        // Model id should be the original.
        assert_eq!(wrapped.model_id(), "base-model");

        // A call should go straight through.
        let resp = wrapped.complete(simple_request()).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello"));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_single_layer() {
        let mock = MockCompletionModel::new("inner");
        let counter = mock.call_count();
        let model: Arc<dyn CompletionModel> = Arc::new(mock);

        let wrapped = MiddlewareStack::new()
            .with_retry(RetryConfig::default())
            .apply(model);

        // The retry wrapper should delegate to the inner model's id.
        assert_eq!(wrapped.model_id(), "inner");

        let resp = wrapped.complete(simple_request()).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello"));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_multiple_layers_order() {
        // Verify that the first middleware added is the outermost wrapper
        // (executes first on the call path).
        let order = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

        let stack = MiddlewareStack::new()
            .layer(TrackingMiddleware::new("outer", Arc::clone(&order)))
            .layer(TrackingMiddleware::new("inner", Arc::clone(&order)));

        let model: Arc<dyn CompletionModel> = Arc::new(MockCompletionModel::new("base"));
        let wrapped = stack.apply(model);

        // During apply, layers are iterated in reverse (inner first, then
        // outer), so wrap_order should be ["inner", "outer"].
        {
            let wrap_log = order.lock().unwrap();
            assert_eq!(wrap_log.as_slice(), &["inner", "outer"]);
        }

        // Clear the log for the call-order test.
        order.lock().unwrap().clear();

        // When we call complete, outer executes first, then inner, then base.
        let resp = wrapped.complete(simple_request()).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello"));

        let call_log = order.lock().unwrap();
        assert_eq!(call_log.as_slice(), &["outer-call", "inner-call"]);
    }

    #[tokio::test]
    async fn test_convenience_methods() {
        let mock = MockCompletionModel::new("test-model");
        let counter = mock.call_count();
        let model: Arc<dyn CompletionModel> = Arc::new(mock);

        let wrapped = MiddlewareStack::new()
            .with_retry(RetryConfig::default())
            .with_cache(CacheConfig::default())
            .apply(model);

        assert_eq!(wrapped.model_id(), "test-model");

        // First call goes through.
        let r1 = wrapped.complete(simple_request()).await.unwrap();
        assert_eq!(r1.content.as_deref(), Some("hello"));

        // Second identical call should be served from cache (inner model
        // called only once).
        let r2 = wrapped.complete(simple_request()).await.unwrap();
        assert_eq!(r2.content.as_deref(), Some("hello"));

        // The inner model was only called once thanks to the cache layer.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
