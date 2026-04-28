//! Shared model download and cache layer for Blazen local-inference backends.
//!
//! Provides [`ModelCache`] for downloading and caching ML models from
//! [`HuggingFace` Hub](https://huggingface.co). Designed to be shared by all
//! local-inference backends (fastembed, mistral.rs, whisper.cpp, etc.).
//!
//! ## wasm32 support
//!
//! On `wasm32-*` targets the underlying download stack (`hf-hub`, `dirs`,
//! `tokio::fs`) is not available, so [`ModelCache`] is a stub that always
//! returns [`CacheError::Unsupported`]. Browser/Worker callers should obtain
//! model bytes through a different mechanism (e.g. `fetch()` on the JS side,
//! pre-bundled assets, or a manually populated cache directory).

use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::LazyLock;

/// Per-destination-path mutexes that serialize concurrent [`ModelCache::download`]
/// calls for the same file. Different files download in parallel; same-file
/// callers wait so hf-hub's internal blob lock isn't raced.
#[cfg(not(target_arch = "wasm32"))]
static DOWNLOAD_LOCKS: LazyLock<dashmap::DashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>> =
    LazyLock::new(dashmap::DashMap::new);

/// Errors that can occur during model cache operations.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// A model download failed.
    #[error("failed to download model: {0}")]
    Download(String),

    /// The cache directory could not be resolved or created.
    #[error("cache directory error: {0}")]
    CacheDir(String),

    /// An underlying I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The requested operation is not supported on this target (e.g. WASM,
    /// where the underlying `HuggingFace` Hub download stack is unavailable).
    #[error("model cache operation not supported on this target: {0}")]
    Unsupported(String),
}

/// Callback trait for receiving download progress updates.
///
/// Implement this on your own type to get notified as bytes are downloaded.
pub trait ProgressCallback: Send + Sync {
    /// Called periodically during download.
    ///
    /// * `downloaded_bytes` - Total bytes downloaded so far.
    /// * `total_bytes` - Total file size if known by the server.
    fn on_progress(&self, downloaded_bytes: u64, total_bytes: Option<u64>);
}

/// Adapter that bridges our [`ProgressCallback`] trait to `hf_hub`'s
/// [`Progress`](hf_hub::api::tokio::Progress) trait.
///
/// Uses `Arc` so the adapter is `Clone + Send + Sync + 'static` as
/// required by [`hf_hub::api::tokio::ApiRepo::download_with_progress`].
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct HfProgressAdapter {
    callback: Arc<dyn ProgressCallback>,
    downloaded: u64,
    total: Option<u64>,
}

#[cfg(not(target_arch = "wasm32"))]
impl HfProgressAdapter {
    fn new(callback: Arc<dyn ProgressCallback>) -> Self {
        Self {
            callback,
            downloaded: 0,
            total: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl hf_hub::api::tokio::Progress for HfProgressAdapter {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.total = Some(size as u64);
        self.downloaded = 0;
        self.callback.on_progress(0, self.total);
    }

    async fn update(&mut self, size: usize) {
        self.downloaded += size as u64;
        self.callback.on_progress(self.downloaded, self.total);
    }

    async fn finish(&mut self) {
        self.callback
            .on_progress(self.downloaded.max(1), self.total);
    }
}

/// A no-op progress implementation used when no callback is provided.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct NoProgress;

#[cfg(not(target_arch = "wasm32"))]
impl hf_hub::api::tokio::Progress for NoProgress {
    async fn init(&mut self, _size: usize, _filename: &str) {}
    async fn update(&mut self, _size: usize) {}
    async fn finish(&mut self) {}
}

/// Local cache for ML models downloaded from `HuggingFace` Hub.
///
/// Models are stored under `{cache_dir}/{repo_id}/{filename}`. Files are
/// downloaded only once; subsequent calls return the cached path immediately.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> Result<(), blazen_model_cache::CacheError> {
/// use blazen_model_cache::ModelCache;
///
/// let cache = ModelCache::new()?;
/// let path = cache.download("bert-base-uncased", "config.json", None).await?;
/// println!("model config at: {}", path.display());
/// # Ok(())
/// # }
/// ```
pub struct ModelCache {
    cache_dir: PathBuf,
}

impl ModelCache {
    /// Create a cache rooted at a specific directory.
    ///
    /// The directory does not need to exist yet; it will be created on the
    /// first download.
    ///
    /// Available on every target — only [`Self::new`] and [`Self::download`]
    /// require the native `HuggingFace` Hub download stack.
    #[must_use]
    pub fn with_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// The root cache directory path.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if a file is already present in the cache (without downloading).
    ///
    /// On wasm32 this always returns `false` — there is no filesystem to
    /// inspect, so callers should not rely on cache hits in the browser.
    #[must_use]
    pub fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.cached_path(repo_id, filename).is_file()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (repo_id, filename);
            false
        }
    }

    /// Compute the expected cache path for a repo/file pair.
    fn cached_path(&self, repo_id: &str, filename: &str) -> PathBuf {
        self.cache_dir.join(repo_id).join(filename)
    }
}

// -- Native-only methods (filesystem + HuggingFace Hub) -----------------------

#[cfg(not(target_arch = "wasm32"))]
impl ModelCache {
    /// Create a cache in the default location.
    ///
    /// Uses `$BLAZEN_CACHE_DIR/models/` if the `BLAZEN_CACHE_DIR` environment
    /// variable is set, otherwise falls back to `~/.cache/blazen/models/`.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::CacheDir`] if the home directory cannot be
    /// determined and `BLAZEN_CACHE_DIR` is not set.
    pub fn new() -> Result<Self, CacheError> {
        let cache_dir = if let Ok(dir) = std::env::var("BLAZEN_CACHE_DIR") {
            PathBuf::from(dir).join("models")
        } else {
            dirs::cache_dir()
                .ok_or_else(|| {
                    CacheError::CacheDir(
                        "could not determine home cache directory; \
                         set BLAZEN_CACHE_DIR to override"
                            .to_string(),
                    )
                })?
                .join("blazen")
                .join("models")
        };

        Ok(Self { cache_dir })
    }

    /// Download a file from `HuggingFace` Hub if it is not already cached.
    ///
    /// Returns the local filesystem path to the cached file.
    ///
    /// The file is first downloaded via `hf-hub` into its own managed cache,
    /// then hard-linked (or copied as fallback) into our
    /// `{cache_dir}/{repo_id}/{filename}` layout so that callers get a stable,
    /// predictable path.
    ///
    /// # Progress
    ///
    /// Pass an `Arc<dyn ProgressCallback>` to receive byte-level progress
    /// updates during the download. Pass `None` to download silently.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::Download`] if the `HuggingFace` API request fails,
    /// or [`CacheError::Io`] if filesystem operations fail.
    pub async fn download(
        &self,
        repo_id: &str,
        filename: &str,
        progress: Option<Arc<dyn ProgressCallback>>,
    ) -> Result<PathBuf, CacheError> {
        let dest = self.cached_path(repo_id, filename);

        // Serialize concurrent callers for the same destination path. Different
        // files download in parallel; same-file callers wait so hf-hub's internal
        // blob lock isn't raced.
        let lock = DOWNLOAD_LOCKS
            .entry(dest.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .value()
            .clone();
        let _guard = lock.lock().await;

        // Already cached -- return immediately (re-check inside the lock so
        // subsequent callers observe the file created by whoever went first).
        if dest.is_file() {
            return Ok(dest);
        }

        // Ensure the target directory exists.
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Build the hf-hub async API.
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(false) // We handle progress ourselves.
            .build()
            .map_err(|e| CacheError::Download(e.to_string()))?;

        let repo = api.model(repo_id.to_string());

        // Download through hf-hub (it manages its own cache under ~/.cache/huggingface).
        let hf_path = if let Some(cb) = progress {
            let adapter = HfProgressAdapter::new(Arc::clone(&cb));
            repo.download_with_progress(filename, adapter)
                .await
                .map_err(|e| CacheError::Download(e.to_string()))?
        } else {
            let noop = NoProgress;
            repo.download_with_progress(filename, noop)
                .await
                .map_err(|e| CacheError::Download(e.to_string()))?
        };

        // hf-hub returns `snapshots/main/<filename>` as a symlink into
        // `blobs/<hash>`. Resolve to the real blob so hard-linking targets the
        // actual file; otherwise on some filesystems we would hard-link the
        // symlink itself, which can behave unexpectedly if the snapshot is
        // pruned later. If canonicalization fails (e.g. broken chain), fall
        // back to the original path and let the copy fallback handle it.
        let hf_path_resolved = tokio::fs::canonicalize(&hf_path)
            .await
            .unwrap_or_else(|_| hf_path.clone());

        // Link or copy the file into our own cache layout.
        if dest != hf_path_resolved {
            // Try hard link first (instant, no extra disk space).
            if tokio::fs::hard_link(&hf_path_resolved, &dest)
                .await
                .is_err()
            {
                // Cross-device or unsupported -- fall back to copy.
                tokio::fs::copy(&hf_path_resolved, &dest).await?;
            }
        }

        // Postcondition: dest must exist after a successful download.
        if !dest.is_file() {
            return Err(CacheError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "download completed but cache path is missing: {}",
                    dest.display()
                ),
            )));
        }

        Ok(dest)
    }
}

// -- wasm32 stubs -------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
impl ModelCache {
    /// Stub that returns [`CacheError::Unsupported`] on wasm32.
    ///
    /// On wasm32 there is no `dirs::cache_dir()` and no `BLAZEN_CACHE_DIR`
    /// environment lookup that would produce a usable path. Use
    /// [`Self::with_dir`] with an explicit virtual path instead.
    ///
    /// # Errors
    ///
    /// Always returns [`CacheError::Unsupported`].
    pub fn new() -> Result<Self, CacheError> {
        Err(CacheError::Unsupported(
            "ModelCache::new() is not supported on wasm32; use ModelCache::with_dir() instead"
                .to_string(),
        ))
    }

    /// Stub that always returns [`CacheError::Unsupported`] on wasm32.
    ///
    /// The HuggingFace Hub client (`hf-hub`) and `tokio::fs` are not
    /// compatible with the `wasm32-*` targets, so model files cannot be
    /// downloaded from this side. Browser/Worker callers should fetch model
    /// bytes via the JavaScript `fetch()` API and pass them in directly.
    ///
    /// # Errors
    ///
    /// Always returns [`CacheError::Unsupported`].
    pub async fn download(
        &self,
        repo_id: &str,
        filename: &str,
        progress: Option<Arc<dyn ProgressCallback>>,
    ) -> Result<PathBuf, CacheError> {
        let _ = (repo_id, filename, progress);
        Err(CacheError::Unsupported(format!(
            "ModelCache::download() is not supported on wasm32 \
             (cache_dir={})",
            self.cache_dir.display()
        )))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(unsafe_code)] // env::set_var / env::remove_var require unsafe in edition 2024
mod tests {
    use super::*;

    #[test]
    fn test_default_cache_dir() {
        // When BLAZEN_CACHE_DIR is not set, the cache should live under the
        // platform cache directory (e.g. ~/.cache/blazen/models/ on Linux).
        // We temporarily remove the env var to test the default path.
        let had_var = std::env::var("BLAZEN_CACHE_DIR").ok();

        // SAFETY: This test runs single-threaded and restores the variable
        // immediately after the assertion. No other thread reads this var
        // concurrently in the test suite (env-var tests are inherently racy
        // but acceptable in `#[test]` which defaults to `--test-threads=1`
        // per binary).
        unsafe {
            std::env::remove_var("BLAZEN_CACHE_DIR");
        }

        let cache = ModelCache::new().expect("default cache should succeed");
        let path = cache.cache_dir();

        // Should end with blazen/models
        assert!(
            path.ends_with("blazen/models"),
            "expected path ending with blazen/models, got: {}",
            path.display()
        );

        // Restore env var if it was set.
        if let Some(val) = had_var {
            // SAFETY: restoring the original value.
            unsafe {
                std::env::set_var("BLAZEN_CACHE_DIR", val);
            }
        }
    }

    #[test]
    fn test_default_cache_dir_from_env() {
        let prev = std::env::var("BLAZEN_CACHE_DIR").ok();

        // SAFETY: see `test_default_cache_dir`.
        unsafe {
            std::env::set_var("BLAZEN_CACHE_DIR", "/tmp/blazen-test-cache");
        }

        let cache = ModelCache::new().expect("env-based cache should succeed");
        assert_eq!(
            cache.cache_dir(),
            Path::new("/tmp/blazen-test-cache/models")
        );

        // Restore.
        // SAFETY: see `test_default_cache_dir`.
        unsafe {
            match prev {
                Some(val) => std::env::set_var("BLAZEN_CACHE_DIR", val),
                None => std::env::remove_var("BLAZEN_CACHE_DIR"),
            }
        }
    }

    #[test]
    fn test_with_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = ModelCache::with_dir(dir.path());
        assert_eq!(cache.cache_dir(), dir.path());
    }

    #[test]
    fn test_is_cached_false_initially() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = ModelCache::with_dir(dir.path());
        assert!(!cache.is_cached("foo/bar", "model.gguf"));
    }

    #[test]
    fn test_is_cached_true_after_manual_placement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = ModelCache::with_dir(dir.path());

        // Manually create the file to simulate a cached download.
        let file_dir = dir.path().join("my-org/my-model");
        std::fs::create_dir_all(&file_dir).unwrap();
        std::fs::write(file_dir.join("config.json"), b"{}").unwrap();

        assert!(cache.is_cached("my-org/my-model", "config.json"));
    }

    #[test]
    fn test_cached_path_layout() {
        let cache = ModelCache::with_dir("/fake/cache");
        let path = cache.cached_path("org/model", "weights.bin");
        assert_eq!(path, PathBuf::from("/fake/cache/org/model/weights.bin"));
    }

    /// Verifies that the per-path lock actually serializes concurrent callers
    /// targeting the same destination. We can't easily mock hf-hub inside
    /// `download()`, so this test exercises the serialization primitive
    /// directly: if the lock map misbehaves (e.g. hands out independent
    /// mutexes for the same path), more than one task will sit inside the
    /// critical section at the same time and the counter will exceed zero
    /// when observed by another task.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_downloads_serialize_same_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache = ModelCache::with_dir(tmp.path().to_path_buf());
        let dest = cache.cached_path("test/repo", "file.bin");

        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let dest_clone = dest.clone();
            let counter_clone = Arc::clone(&counter);
            handles.push(tokio::spawn(async move {
                let lock = DOWNLOAD_LOCKS
                    .entry(dest_clone)
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                    .value()
                    .clone();
                let _guard = lock.lock().await;
                // If another task already holds the lock, it would have
                // incremented the counter before us; the assertion below
                // would then catch the violation.
                let prev = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Simulate in-flight work to widen the race window.
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                counter_clone.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                prev
            }));
        }

        let results = futures_util::future::join_all(handles).await;
        for r in results {
            let prev = r.expect("task panicked");
            assert_eq!(prev, 0, "another task held the lock concurrently");
        }
    }

    /// Integration test that actually downloads from `HuggingFace` Hub.
    ///
    /// Ignored by default because it requires network access. Run with:
    /// ```sh
    /// cargo test -p blazen-model-cache -- --ignored
    /// ```
    #[tokio::test]
    #[ignore = "requires network access to HuggingFace Hub"]
    async fn test_download_and_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = ModelCache::with_dir(dir.path());

        // Download a tiny file (~600 bytes).
        let path = cache
            .download("hf-internal-testing/tiny-random-gpt2", "config.json", None)
            .await
            .expect("download should succeed");

        // File should exist and have non-zero size.
        assert!(path.is_file(), "downloaded file should exist");
        let meta = std::fs::metadata(&path).expect("metadata");
        assert!(meta.len() > 0, "downloaded file should be non-empty");

        // Second call should return the cached path instantly.
        let path2 = cache
            .download("hf-internal-testing/tiny-random-gpt2", "config.json", None)
            .await
            .expect("cached download should succeed");
        assert_eq!(path, path2);
    }

    /// Integration test verifying progress callback fires.
    #[tokio::test]
    #[ignore = "requires network access to HuggingFace Hub"]
    async fn test_download_with_progress() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct TestProgress {
            calls: AtomicU64,
        }

        impl ProgressCallback for TestProgress {
            fn on_progress(&self, _downloaded_bytes: u64, _total_bytes: Option<u64>) {
                self.calls.fetch_add(1, Ordering::Relaxed);
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let cache = ModelCache::with_dir(dir.path());
        let progress = Arc::new(TestProgress {
            calls: AtomicU64::new(0),
        });

        // Clone the Arc so we retain a handle for assertions.
        let cb: Arc<dyn ProgressCallback> = Arc::clone(&progress) as Arc<dyn ProgressCallback>;

        cache
            .download(
                "hf-internal-testing/tiny-random-gpt2",
                "config.json",
                Some(cb),
            )
            .await
            .expect("download should succeed");

        assert!(
            progress.calls.load(Ordering::Relaxed) > 0,
            "progress callback should have been called at least once"
        );
    }
}
