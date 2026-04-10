//! Shared model download and cache layer for Blazen local-inference backends.
//!
//! Provides [`ModelCache`] for downloading and caching ML models from
//! [`HuggingFace` Hub](https://huggingface.co). Designed to be shared by all
//! local-inference backends (fastembed, mistral.rs, whisper.cpp, etc.).

use std::path::{Path, PathBuf};
use std::sync::Arc;

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
#[derive(Clone)]
struct HfProgressAdapter {
    callback: Arc<dyn ProgressCallback>,
    downloaded: u64,
    total: Option<u64>,
}

impl HfProgressAdapter {
    fn new(callback: Arc<dyn ProgressCallback>) -> Self {
        Self {
            callback,
            downloaded: 0,
            total: None,
        }
    }
}

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
#[derive(Clone)]
struct NoProgress;

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

    /// Create a cache rooted at a specific directory.
    ///
    /// The directory does not need to exist yet; it will be created on the
    /// first download.
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
    #[must_use]
    pub fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        self.cached_path(repo_id, filename).is_file()
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

        // Already cached -- return immediately.
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

        // Link or copy the file into our own cache layout.
        if dest != hf_path {
            // Try hard link first (instant, no extra disk space).
            if tokio::fs::hard_link(&hf_path, &dest).await.is_err() {
                // Cross-device or unsupported -- fall back to copy.
                tokio::fs::copy(&hf_path, &dest).await?;
            }
        }

        Ok(dest)
    }

    /// Compute the expected cache path for a repo/file pair.
    fn cached_path(&self, repo_id: &str, filename: &str) -> PathBuf {
        self.cache_dir.join(repo_id).join(filename)
    }
}

#[cfg(test)]
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

    /// Integration test that actually downloads from `HuggingFace` Hub.
    ///
    /// Ignored by default because it requires network access. Run with:
    /// ```sh
    /// cargo test -p blazen-model-cache -- --ignored
    /// ```
    #[tokio::test]
    #[ignore]
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
    #[ignore]
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
