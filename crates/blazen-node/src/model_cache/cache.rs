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

/// TSFN built from a bound `onProgress` method on a JS `ProgressCallback`
/// subclass instance. Uses `BigInt` for byte counts so values larger than
/// `Number.MAX_SAFE_INTEGER` (~9 PB) survive the Rust -> JS hop without loss.
///
/// The JS handler signature is
/// `(downloaded: bigint, total: bigint | null) => void`.
type ProgressMethodTsfn = ThreadsafeFunction<
    FnArgs<(BigInt, Option<BigInt>)>,
    (),
    FnArgs<(BigInt, Option<BigInt>)>,
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

/// Adapter that bridges a bound JS `onProgress` method (extracted from a
/// [`JsProgressCallback`] subclass instance) to the Rust [`ProgressCallback`]
/// trait. Byte counts are passed as `BigInt` to preserve full `u64` precision.
struct JsProgressMethodAdapter {
    tsfn: Arc<ProgressMethodTsfn>,
}

impl ProgressCallback for JsProgressMethodAdapter {
    fn on_progress(&self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        let downloaded = BigInt::from(downloaded_bytes);
        let total = total_bytes.map(BigInt::from);
        let _ = self.tsfn.call(
            FnArgs::from((downloaded, total)),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

/// Subclassable base for download progress callbacks.
///
/// Extend this class and override `onProgress(downloaded, total)` to receive
/// progress updates from [`JsModelCache::download`]. Byte counts are `bigint`
/// values so large downloads (multi-gigabyte model files) keep full precision.
///
/// `total` is `null` when the server does not report `Content-Length` up
/// front (e.g. streaming responses without a known size).
///
/// ```javascript
/// import { ModelCache, ProgressCallback } from 'blazen';
///
/// class LoggingProgress extends ProgressCallback {
///     onProgress(downloaded, total) {
///         if (total !== null) {
///             const pct = Number(downloaded * 100n / total);
///             console.log(`${pct}%`);
///         } else {
///             console.log(`${downloaded} bytes`);
///         }
///     }
/// }
///
/// const cache = ModelCache.create();
/// await cache.download('bert-base-uncased', 'config.json', new LoggingProgress());
/// ```
#[napi(js_name = "ProgressCallback")]
pub struct JsProgressCallback;

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::unused_self
)]
impl JsProgressCallback {
    /// Create a new `ProgressCallback` base instance.
    ///
    /// Subclasses should call `super()` and override `onProgress`.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Receive a progress update. Subclasses **must** override this method.
    ///
    /// Calling the base implementation always throws so that forgetting to
    /// override is caught loudly rather than silently swallowing progress
    /// events.
    #[napi(js_name = "onProgress")]
    pub fn on_progress(&self, _downloaded: BigInt, _total: Option<BigInt>) -> napi::Result<()> {
        Err(napi::Error::new(
            napi::Status::GenericFailure,
            "ProgressCallback subclass must override onProgress(downloaded, total)",
        ))
    }
}

/// Build a [`ProgressCallback`] adapter from a JS `ProgressCallback` subclass
/// instance.
///
/// Extracts the `onProgress` property from the instance, binds `this` to the
/// instance via `Function.prototype.bind` (so `this.foo` works inside the
/// override), and wraps the bound function in a TSFN that the Rust download
/// code can call from any thread.
///
/// Errors if the instance has no `onProgress` property, the property is not a
/// function, or building the TSFN fails.
fn build_progress_adapter_from_instance(
    instance: &Object<'_>,
) -> Result<Arc<dyn ProgressCallback>> {
    if !instance.has_named_property("onProgress").unwrap_or(false) {
        return Err(napi::Error::from_reason(
            "ProgressCallback instance is missing onProgress(downloaded, total)",
        ));
    }

    // Extract as a typed Function so we can bind `this` and build a TSFN with
    // matching argument types. `()` return reflects that the JS handler's
    // return value is discarded.
    let js_function: Function<'_, FnArgs<(BigInt, Option<BigInt>)>, ()> =
        instance.get_named_property("onProgress").map_err(|e| {
            napi::Error::from_reason(format!(
                "ProgressCallback.onProgress is not a function: {e}"
            ))
        })?;

    // Bind `this` to the subclass instance so user overrides can use `this.*`.
    let bound = js_function.bind(instance).map_err(|e| {
        napi::Error::from_reason(format!(
            "failed to bind `this` on ProgressCallback.onProgress: {e}"
        ))
    })?;

    let tsfn: ProgressMethodTsfn = bound
        .build_threadsafe_function::<FnArgs<(BigInt, Option<BigInt>)>>()
        .weak::<true>()
        .build()
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "failed to build threadsafe function for ProgressCallback.onProgress: {e}"
            ))
        })?;

    Ok(Arc::new(JsProgressMethodAdapter {
        tsfn: Arc::new(tsfn),
    }))
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
    /// The optional `onProgress` argument accepts either:
    /// - A raw callback `(downloaded: number, total: number | null) => void`
    ///   for a quick inline progress hook, or
    /// - A [`JsProgressCallback`] subclass instance (recommended for stateful
    ///   reporters), whose `onProgress(downloaded, total)` method receives
    ///   byte counts as `bigint` values.
    ///
    /// `total` is `null` when the server does not report the file size up
    /// front.
    //
    // Hand-rolled `AsyncBlock` rather than `async fn` because the `Object<'_>`
    // arm of the `on_progress` parameter is `!Send` and would poison the
    // generated future. We extract the JS instance to a `Send`-able adapter
    // synchronously, then move only the adapter across the await.
    #[napi(js_name = "download")]
    pub fn download(
        &self,
        env: &Env,
        repo: String,
        file: String,
        on_progress: Option<Either<ProgressTsfn, Object<'_>>>,
    ) -> Result<AsyncBlock<String>> {
        let progress: Option<Arc<dyn ProgressCallback>> = match on_progress {
            None => None,
            Some(Either::A(tsfn)) => {
                let adapter = JsProgressAdapter {
                    tsfn: Arc::new(tsfn),
                };
                Some(Arc::new(adapter) as Arc<dyn ProgressCallback>)
            }
            Some(Either::B(instance)) => Some(build_progress_adapter_from_instance(&instance)?),
        };

        let inner = Arc::clone(&self.inner);
        let fut = async move {
            let path = inner
                .download(&repo, &file, progress)
                .await
                .map_err(cache_error_to_napi)?;
            Ok(path.to_string_lossy().into_owned())
        };

        AsyncBlockBuilder::new(fut).build(env)
    }
}
