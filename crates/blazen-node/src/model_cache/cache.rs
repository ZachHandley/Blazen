//! JavaScript wrapper for [`blazen_model_cache::ModelCache`].
//!
//! Exposes the cache as a napi class so TypeScript users can download models
//! from `HuggingFace` Hub with optional progress callbacks.

use std::path::PathBuf;
use std::sync::Arc;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

use blazen_model_cache::{ModelCache, ProgressCallback};

use super::error::cache_error_to_napi;

/// Progress callback: takes `(downloaded, total)` and returns nothing.
///
/// The JS callback signature is `(downloaded: number, total: number | null) => void`.
/// `FnArgs<(f64, Option<f64>)>` spreads the tuple into two separate JS arguments
/// rather than packing them into an Array.
///
/// `CalleeHandled = false` disables the error-first callback convention so the
/// JS handler receives `(downloaded, total)` directly.
///
/// `Weak = true` unrefs the TSFN so it does not prevent Node.js from exiting.
type ProgressTsfn = ThreadsafeFunction<
    FnArgs<(f64, Option<f64>)>,
    Unknown<'static>,
    FnArgs<(f64, Option<f64>)>,
    Status,
    false,
    true,
>;

/// Adapter that bridges the JS progress TSFN to the Rust [`ProgressCallback`]
/// trait expected by [`ModelCache::download`].
struct JsProgressAdapter {
    tsfn: Arc<ProgressTsfn>,
}

impl ProgressCallback for JsProgressAdapter {
    #[allow(clippy::cast_precision_loss)]
    fn on_progress(&self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        let downloaded = downloaded_bytes as f64;
        let total = total_bytes.map(|n| n as f64);
        let _ = self.tsfn.call(
            FnArgs::from((downloaded, total)),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

/// Local cache for ML models downloaded from `HuggingFace` Hub.
///
/// Models are stored under `{cacheDir}/{repoId}/{filename}`. Files are
/// downloaded only once; subsequent calls return the cached path immediately.
///
/// ```javascript
/// import { ModelCache } from 'blazen';
///
/// const cache = ModelCache.create();
/// if (!cache.isCached('bert-base-uncased', 'config.json')) {
///   await cache.download('bert-base-uncased', 'config.json', (downloaded, total) => {
///     if (total !== null) {
///       console.log(`${(downloaded / total * 100).toFixed(1)}%`);
///     }
///   });
/// }
/// ```
#[napi(js_name = "ModelCache")]
pub struct JsModelCache {
    inner: Arc<ModelCache>,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsModelCache {
    /// Create a cache in the default location.
    ///
    /// Uses `$BLAZEN_CACHE_DIR/models/` if set, otherwise falls back to
    /// `~/.cache/blazen/models/`.
    #[napi(factory)]
    pub fn create() -> Result<Self> {
        let inner = ModelCache::new().map_err(cache_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Create a cache rooted at a specific directory.
    ///
    /// The directory does not need to exist yet; it will be created on the
    /// first download.
    #[napi(factory, js_name = "withDir")]
    #[must_use]
    pub fn with_dir(path: String) -> Self {
        let inner = ModelCache::with_dir(PathBuf::from(path));
        Self {
            inner: Arc::new(inner),
        }
    }

    /// The root cache directory path as a string.
    #[napi(getter, js_name = "cacheDir")]
    #[must_use]
    pub fn cache_dir(&self) -> String {
        self.inner.cache_dir().to_string_lossy().into_owned()
    }

    /// Check if a file is already present in the cache (without downloading).
    #[napi(js_name = "isCached")]
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn is_cached(&self, repo: String, file: String) -> bool {
        self.inner.is_cached(&repo, &file)
    }

    /// Download a file from `HuggingFace` Hub if it is not already cached.
    ///
    /// Returns the local filesystem path to the cached file.
    ///
    /// The optional `onProgress` callback receives `(downloaded, total)` where
    /// `total` is `null` when the server does not report the file size up
    /// front.
    #[napi(js_name = "download")]
    pub async fn download(
        &self,
        repo: String,
        file: String,
        on_progress: Option<ProgressTsfn>,
    ) -> Result<String> {
        let progress: Option<Arc<dyn ProgressCallback>> = on_progress.map(|tsfn| {
            let adapter = JsProgressAdapter {
                tsfn: Arc::new(tsfn),
            };
            Arc::new(adapter) as Arc<dyn ProgressCallback>
        });

        let path = self
            .inner
            .download(&repo, &file, progress)
            .await
            .map_err(cache_error_to_napi)?;

        Ok(path.to_string_lossy().into_owned())
    }
}
