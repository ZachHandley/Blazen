//! Smart caching layer for [`CompletionModel`] with content-based hashing.
//!
//! [`CachedCompletionModel`] is a decorator that wraps any [`CompletionModel`]
//! and caches non-streaming responses keyed on a hash of the full request
//! (messages, parameters, tools, and model). Repeated identical requests are
//! served from memory without hitting the underlying provider.
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use blazen_llm::cache::{CachedCompletionModel, CacheConfig, CacheStrategy};
//!
//! let inner = /* any CompletionModel */;
//! let model = CachedCompletionModel::new(inner, CacheConfig::default());
//! // First call hits the provider:
//! let r1 = model.complete(request.clone()).await?;
//! // Second identical call is served from cache:
//! let r2 = model.complete(request).await?;
//! ```
//!
//! Streaming requests are **never** cached and always delegate directly to the
//! inner model.
//!
//! # Eviction
//!
//! When the cache exceeds [`CacheConfig::max_entries`], the oldest entry (by
//! insertion time) is evicted before inserting the new one. Entries that have
//! exceeded their TTL are treated as misses and lazily removed on the next
//! lookup.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::Stream;

use crate::error::BlazenError;
use crate::traits::CompletionModel;
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Determines how cache keys are computed and whether provider-specific
/// optimisations are applied.
#[derive(Debug, Clone, Default)]
pub enum CacheStrategy {
    /// No caching -- all requests pass through to the inner model.
    None,
    /// Hash the full request (messages + parameters) for exact-match
    /// response caching.  This is the default strategy.
    #[default]
    ContentHash,
    /// Anthropic-specific: attach `cache_control` metadata to system and
    /// large context blocks so the provider can cache prefixes server-side.
    ///
    /// **Not yet implemented** -- falls back to `ContentHash` for now.
    AnthropicEphemeral,
    /// Auto-detect: uses `ContentHash` for most providers and will use
    /// `AnthropicEphemeral` for Anthropic once implemented.
    ///
    /// **Currently equivalent to `ContentHash`.**
    Auto,
}

/// Configuration for the response cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// The caching strategy to use.
    pub strategy: CacheStrategy,
    /// How long a cached response remains valid.
    pub ttl: Duration,
    /// Maximum number of entries to keep in the cache. When exceeded, the
    /// oldest entry is evicted.
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            strategy: CacheStrategy::default(),
            ttl: Duration::from_secs(5 * 60),
            max_entries: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache internals
// ---------------------------------------------------------------------------

/// A cached response together with its insertion timestamp.
struct CacheEntry {
    response: CompletionResponse,
    created_at: Instant,
}

// ---------------------------------------------------------------------------
// CachedCompletionModel
// ---------------------------------------------------------------------------

/// A [`CompletionModel`] decorator that caches non-streaming responses.
///
/// See the [module-level documentation](self) for usage examples.
pub struct CachedCompletionModel {
    inner: Arc<dyn CompletionModel>,
    config: CacheConfig,
    cache: Arc<RwLock<HashMap<u64, CacheEntry>>>,
}

impl CachedCompletionModel {
    /// Wrap `inner` with the given cache configuration.
    pub fn new(inner: impl CompletionModel + 'static, config: CacheConfig) -> Self {
        Self {
            inner: Arc::new(inner),
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Wrap an already-`Arc`'d model.
    pub fn from_arc(inner: Arc<dyn CompletionModel>, config: CacheConfig) -> Self {
        Self {
            inner,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the number of entries currently in the cache.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.read().expect("cache lock poisoned").len()
    }

    /// Returns `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries from the cache.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    pub fn clear(&self) {
        self.cache.write().expect("cache lock poisoned").clear();
    }

    /// Whether the configured strategy actually caches responses.
    fn caching_enabled(&self) -> bool {
        !matches!(self.config.strategy, CacheStrategy::None)
    }

    /// Compute a deterministic cache key from a [`CompletionRequest`].
    ///
    /// The key is a 64-bit hash of the model id, serialised messages,
    /// temperature, max tokens, top-p, tools, and response format.  This is
    /// *not* cryptographic -- it is only used as a cache key where collisions
    /// are acceptable (they would simply cause a cache miss/overwrite).
    fn cache_key(&self, request: &CompletionRequest) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Model identifier (from the provider, or the request-level override).
        if let Some(ref model) = request.model {
            model.hash(&mut hasher);
        } else {
            self.inner.model_id().hash(&mut hasher);
        }

        // Messages -- serialise to canonical JSON for a stable representation.
        if let Ok(json) = serde_json::to_string(&request.messages) {
            json.hash(&mut hasher);
        }

        // Sampling parameters.
        if let Some(t) = request.temperature {
            t.to_bits().hash(&mut hasher);
        }
        if let Some(m) = request.max_tokens {
            m.hash(&mut hasher);
        }
        if let Some(p) = request.top_p {
            p.to_bits().hash(&mut hasher);
        }

        // Tools -- serialise definitions so different tool sets produce
        // different keys.
        if !request.tools.is_empty() {
            if let Ok(json) = serde_json::to_string(&request.tools) {
                json.hash(&mut hasher);
            }
        }

        // Response format / JSON schema constraint.
        if let Some(ref fmt) = request.response_format {
            if let Ok(json) = serde_json::to_string(fmt) {
                json.hash(&mut hasher);
            }
        }

        // Modalities.
        if let Some(ref mods) = request.modalities {
            mods.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Look up a non-expired entry under `key`.
    fn get_cached(&self, key: u64) -> Option<CompletionResponse> {
        let cache = self.cache.read().expect("cache lock poisoned");
        let entry = cache.get(&key)?;

        if entry.created_at.elapsed() > self.config.ttl {
            // Expired -- the caller will remove it after acquiring a write lock.
            return None;
        }

        Some(entry.response.clone())
    }

    /// Insert `response` under `key`, evicting the oldest entry if the cache
    /// is at capacity.
    fn insert(&self, key: u64, response: CompletionResponse) {
        let mut cache = self.cache.write().expect("cache lock poisoned");

        // Evict oldest if at capacity.
        if cache.len() >= self.config.max_entries && !cache.contains_key(&key) {
            if let Some((&oldest_key, _)) = cache.iter().min_by_key(|(_, e)| e.created_at) {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(
            key,
            CacheEntry {
                response,
                created_at: Instant::now(),
            },
        );
    }
}

#[async_trait]
impl CompletionModel for CachedCompletionModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        // If caching is disabled, pass through immediately.
        if !self.caching_enabled() {
            return self.inner.complete(request).await;
        }

        let key = self.cache_key(&request);

        // Check cache (read lock).
        if let Some(cached) = self.get_cached(key) {
            tracing::debug!(key, "cache hit");
            return Ok(cached);
        }

        // Cache miss -- call the inner model.
        let response = self.inner.complete(request).await?;

        // Store on success.
        self.insert(key, response.clone());

        Ok(response)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // Streaming is not cacheable; always delegate.
        self.inner.stream(request).await
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
    use crate::types::ChatMessage;

    // -- Mock model ----------------------------------------------------------

    /// A mock [`CompletionModel`] that counts invocations and returns
    /// pre-configured responses.
    struct MockCompletionModel {
        id: String,
        responses: Vec<CompletionResponse>,
        call_count: AtomicUsize,
    }

    impl MockCompletionModel {
        fn new(id: &str, responses: Vec<CompletionResponse>) -> Self {
            Self {
                id: id.to_owned(),
                responses,
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
            let idx = idx.min(self.responses.len().saturating_sub(1));
            Ok(self.responses[idx].clone())
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let stream = futures_util::stream::empty();
            Ok(Box::pin(stream))
        }
    }

    // -- Helpers -------------------------------------------------------------

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
        CompletionRequest::new(vec![ChatMessage::user("hello")])
    }

    // -- Tests ---------------------------------------------------------------

    #[tokio::test]
    async fn test_cache_hit() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("first"), ok_response("second")],
        ));
        let model = CachedCompletionModel::from_arc(mock.clone(), CacheConfig::default());

        let r1 = model.complete(simple_request()).await.unwrap();
        let r2 = model.complete(simple_request()).await.unwrap();

        // Both should return the same cached response.
        assert_eq!(r1.content.as_deref(), Some("first"));
        assert_eq!(r2.content.as_deref(), Some("first"));
        // The inner model should only have been called once.
        assert_eq!(mock.calls(), 1);
    }

    #[tokio::test]
    async fn test_cache_miss_different_params() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("cold"), ok_response("hot")],
        ));
        let model = CachedCompletionModel::from_arc(mock.clone(), CacheConfig::default());

        let req_cold =
            CompletionRequest::new(vec![ChatMessage::user("hello")]).with_temperature(0.0);
        let req_hot =
            CompletionRequest::new(vec![ChatMessage::user("hello")]).with_temperature(1.5);

        let r1 = model.complete(req_cold).await.unwrap();
        let r2 = model.complete(req_hot).await.unwrap();

        assert_eq!(r1.content.as_deref(), Some("cold"));
        assert_eq!(r2.content.as_deref(), Some("hot"));
        // Different temperatures produce different cache keys.
        assert_eq!(mock.calls(), 2);
    }

    #[tokio::test]
    async fn test_cache_expiry() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("first"), ok_response("second")],
        ));
        let model = CachedCompletionModel::from_arc(
            mock.clone(),
            CacheConfig {
                strategy: CacheStrategy::ContentHash,
                ttl: Duration::from_secs(0), // immediate expiry
                max_entries: 1000,
            },
        );

        let r1 = model.complete(simple_request()).await.unwrap();
        assert_eq!(r1.content.as_deref(), Some("first"));

        // The entry was inserted but with TTL=0 it should be expired
        // by the time the next call checks.
        let r2 = model.complete(simple_request()).await.unwrap();
        assert_eq!(r2.content.as_deref(), Some("second"));
        // Both calls hit the inner model.
        assert_eq!(mock.calls(), 2);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("a"), ok_response("b"), ok_response("a-again")],
        ));
        let model = CachedCompletionModel::from_arc(
            mock.clone(),
            CacheConfig {
                strategy: CacheStrategy::ContentHash,
                ttl: Duration::from_secs(300),
                max_entries: 1, // only one entry allowed
            },
        );

        // First request caches "a".
        let req_a = CompletionRequest::new(vec![ChatMessage::user("alpha")]);
        let r1 = model.complete(req_a.clone()).await.unwrap();
        assert_eq!(r1.content.as_deref(), Some("a"));
        assert_eq!(model.len(), 1);

        // Second request with different content evicts "a", caches "b".
        let req_b = CompletionRequest::new(vec![ChatMessage::user("bravo")]);
        let r2 = model.complete(req_b).await.unwrap();
        assert_eq!(r2.content.as_deref(), Some("b"));
        assert_eq!(model.len(), 1);

        // Re-request "a" -- it was evicted so the inner model is called again.
        let r3 = model.complete(req_a).await.unwrap();
        assert_eq!(r3.content.as_deref(), Some("a-again"));
        assert_eq!(mock.calls(), 3);
    }

    #[tokio::test]
    async fn test_stream_not_cached() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("x")],
        ));
        let model = CachedCompletionModel::from_arc(mock.clone(), CacheConfig::default());

        // Streaming always delegates, even if called twice.
        let _s1 = model.stream(simple_request()).await.unwrap();
        let _s2 = model.stream(simple_request()).await.unwrap();

        assert_eq!(mock.calls(), 2);
        // And the cache should remain empty.
        assert!(model.is_empty());
    }

    #[tokio::test]
    async fn test_strategy_none_disables_caching() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("first"), ok_response("second")],
        ));
        let model = CachedCompletionModel::from_arc(
            mock.clone(),
            CacheConfig {
                strategy: CacheStrategy::None,
                ttl: Duration::from_secs(300),
                max_entries: 1000,
            },
        );

        let r1 = model.complete(simple_request()).await.unwrap();
        let r2 = model.complete(simple_request()).await.unwrap();

        assert_eq!(r1.content.as_deref(), Some("first"));
        assert_eq!(r2.content.as_deref(), Some("second"));
        assert_eq!(mock.calls(), 2);
    }

    #[tokio::test]
    async fn test_different_messages_produce_different_keys() {
        let mock = Arc::new(MockCompletionModel::new(
            "test-model",
            vec![ok_response("hello-resp"), ok_response("goodbye-resp")],
        ));
        let model = CachedCompletionModel::from_arc(mock.clone(), CacheConfig::default());

        let req1 = CompletionRequest::new(vec![ChatMessage::user("hello")]);
        let req2 = CompletionRequest::new(vec![ChatMessage::user("goodbye")]);

        let r1 = model.complete(req1).await.unwrap();
        let r2 = model.complete(req2).await.unwrap();

        assert_eq!(r1.content.as_deref(), Some("hello-resp"));
        assert_eq!(r2.content.as_deref(), Some("goodbye-resp"));
        assert_eq!(mock.calls(), 2);
    }
}
