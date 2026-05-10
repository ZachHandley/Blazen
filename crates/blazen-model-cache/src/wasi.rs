//! Wasi-compatible model cache.
//!
//! Wasi targets (Cloudflare Workers, wasmtime, etc.) have no persistent
//! filesystem, so the native [`super::ModelCache`] -- which depends on
//! `hf-hub`, `dirs`, and `tokio::fs` -- is unavailable. This module provides
//! [`WasiModelCache`], an in-memory cache that fetches model bytes through
//! a small caller-provided HTTP fetch trait ([`WasiHttpFetch`]) and stores
//! them in a process-wide `RwLock<HashMap<...>>` shared across the lifetime
//! of the worker isolate.
//!
//! # Why not [`blazen_llm::http::HttpClient`] directly?
//!
//! `blazen-llm` transitively depends on `blazen-model-cache` (via
//! `blazen-audio-piper`), so `blazen-model-cache` cannot depend on
//! `blazen-llm` without creating a Cargo path cycle. Callers (e.g.
//! `blazen-node`, `blazen-tract-embed`) wrap their
//! `Arc<dyn blazen_llm::HttpClient>` in a thin adapter that implements
//! [`WasiHttpFetch`] and pass it to [`WasiModelCache::new`].
//!
//! Models are identified by `(repo_id, filename)` -- the same key used by
//! the native cache. The download path constructs an HTTPS URL against a
//! configurable base (defaults to
//! `https://huggingface.co/<repo_id>/resolve/main/<filename>`) and pulls
//! bytes via the supplied [`WasiHttpFetch`].

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::CacheError;

/// Minimal HTTP-fetch abstraction used by [`WasiModelCache`].
///
/// Callers adapt their preferred HTTP client (typically
/// `Arc<dyn blazen_llm::HttpClient>`) to this trait. The cache only needs a
/// single capability: GET a URL and return the response body bytes plus
/// the HTTP status code.
#[async_trait::async_trait]
pub trait WasiHttpFetch: Send + Sync + std::fmt::Debug {
    /// Fetch `url` with HTTP GET and return `(status_code, body_bytes)`.
    ///
    /// Implementors should not raise on non-2xx statuses -- return the
    /// status to the caller and let it decide. Errors are reserved for
    /// transport failures (DNS, TLS, connection drop, etc.).
    async fn get(&self, url: &str) -> Result<(u16, Vec<u8>), CacheError>;
}

/// In-memory model cache for wasi targets.
///
/// Cloned [`WasiModelCache`] handles share the same underlying storage map,
/// so a download performed through one handle is visible to all clones.
#[derive(Debug, Clone)]
pub struct WasiModelCache {
    base_url: String,
    http: Arc<dyn WasiHttpFetch>,
    inner: Arc<RwLock<HashMap<String, Arc<[u8]>>>>,
}

impl WasiModelCache {
    /// Create a new in-memory cache that fetches files through the given
    /// [`WasiHttpFetch`] implementation. The base URL defaults to
    /// `https://huggingface.co`.
    #[must_use]
    pub fn new(http: Arc<dyn WasiHttpFetch>) -> Self {
        Self {
            base_url: "https://huggingface.co".to_owned(),
            http,
            inner: Arc::default(),
        }
    }

    /// Override the base URL used to resolve `(repo_id, filename)` pairs.
    /// Trailing slashes are stripped.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    /// Replace the HTTP fetcher used for downloads.
    #[must_use]
    pub fn with_http_client(mut self, http: Arc<dyn WasiHttpFetch>) -> Self {
        self.http = http;
        self
    }

    fn cache_key(repo_id: &str, filename: &str) -> String {
        format!("{repo_id}/{filename}")
    }

    /// Check whether `(repo_id, filename)` has already been pulled into the
    /// in-memory cache. Returns `false` if the lock is poisoned.
    #[must_use]
    pub fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        self.inner
            .read()
            .is_ok_and(|m| m.contains_key(&Self::cache_key(repo_id, filename)))
    }

    /// Download a file from the configured base URL if it is not already in
    /// the in-memory cache, and return its bytes.
    ///
    /// On a cache hit the returned `Arc<[u8]>` shares storage with whatever
    /// is held in the map; on a miss the response body is inserted under
    /// `(repo_id, filename)` for later hits.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::Download`] if the underlying HTTP request fails
    /// or the response status is >= 400.
    pub async fn download(&self, repo_id: &str, filename: &str) -> Result<Arc<[u8]>, CacheError> {
        let key = Self::cache_key(repo_id, filename);

        // Cache hit -- return the shared bytes immediately.
        if let Some(bytes) = self.inner.read().ok().and_then(|m| m.get(&key).cloned()) {
            return Ok(bytes);
        }

        let url = format!("{}/{}/resolve/main/{}", self.base_url, repo_id, filename);
        let (status, body) = self.http.get(&url).await?;

        if status >= 400 {
            return Err(CacheError::Download(format!(
                "HTTP {status} fetching {key}"
            )));
        }

        let bytes: Arc<[u8]> = Arc::from(body);
        if let Ok(mut m) = self.inner.write() {
            m.insert(key, Arc::clone(&bytes));
        }
        Ok(bytes)
    }
}
