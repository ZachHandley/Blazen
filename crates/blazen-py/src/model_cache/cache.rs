//! Python wrapper for [`blazen_model_cache::ModelCache`].

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_model_cache::{ModelCache, ProgressCallback};

use super::error::cache_err;

// ---------------------------------------------------------------------------
// PyHostProgressCallback -- Rust trait adapter over a Python callable
// ---------------------------------------------------------------------------

/// Adapter that implements the Rust [`ProgressCallback`] trait by calling
/// back into a Python callable.
///
/// The Python callable is invoked synchronously as
/// ``callback(downloaded: int, total: int | None)`` from inside the
/// hf-hub progress driver.
struct PyHostProgressCallback {
    py_obj: Py<PyAny>,
}

impl PyHostProgressCallback {
    fn new(py_obj: Py<PyAny>) -> Self {
        Self { py_obj }
    }
}

impl ProgressCallback for PyHostProgressCallback {
    fn on_progress(&self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        Python::attach(|py| {
            let cb = self.py_obj.bind(py);
            let _ = cb.call1((downloaded_bytes, total_bytes));
        });
    }
}

// ---------------------------------------------------------------------------
// PyModelCache
// ---------------------------------------------------------------------------

/// Local cache for ML models downloaded from `HuggingFace` Hub.
///
/// Models are stored under ``{cache_dir}/{repo_id}/{filename}``. Files are
/// downloaded only once; subsequent calls return the cached path immediately.
///
/// Example:
///     >>> cache = ModelCache()
///     >>> path = await cache.download("bert-base-uncased", "config.json")
///     >>> print(path)
#[gen_stub_pyclass]
#[pyclass(name = "ModelCache")]
pub struct PyModelCache {
    inner: Arc<ModelCache>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelCache {
    /// Create a cache in the default location.
    ///
    /// Uses ``$BLAZEN_CACHE_DIR/models/`` if the ``BLAZEN_CACHE_DIR``
    /// environment variable is set, otherwise falls back to the platform
    /// cache directory (e.g. ``~/.cache/blazen/models/`` on Linux).
    #[new]
    fn new() -> PyResult<Self> {
        let inner = ModelCache::new().map_err(cache_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Create a cache rooted at a specific directory.
    ///
    /// The directory does not need to exist yet; it will be created on the
    /// first download.
    ///
    /// Args:
    ///     path: Filesystem path to use as the cache root.
    #[staticmethod]
    fn with_dir(path: PathBuf) -> Self {
        Self {
            inner: Arc::new(ModelCache::with_dir(path)),
        }
    }

    /// The root cache directory path as a string.
    fn cache_dir(&self) -> String {
        self.inner.cache_dir().display().to_string()
    }

    /// Check if a file is already present in the cache (without downloading).
    ///
    /// Args:
    ///     repo: HuggingFace repo id (e.g. "bert-base-uncased").
    ///     file: Filename within the repo (e.g. "config.json").
    fn is_cached(&self, repo: &str, file: &str) -> bool {
        self.inner.is_cached(repo, file)
    }

    /// Download a file from HuggingFace Hub if it is not already cached.
    ///
    /// Returns the local filesystem path to the cached file as a string.
    ///
    /// Args:
    ///     repo: HuggingFace repo id (e.g. "bert-base-uncased").
    ///     file: Filename within the repo (e.g. "config.json").
    ///     progress: Optional callable invoked as
    ///         ``progress(downloaded_bytes: int, total_bytes: int | None)``
    ///         during the download.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.str]", imports = ("typing", "builtins")))]
    #[pyo3(signature = (repo, file, progress=None))]
    fn download<'py>(
        &self,
        py: Python<'py>,
        repo: String,
        file: String,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cache = self.inner.clone();
        let cb: Option<Arc<dyn ProgressCallback>> = progress
            .map(|obj| -> Arc<dyn ProgressCallback> { Arc::new(PyHostProgressCallback::new(obj)) });
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let path = cache.download(&repo, &file, cb).await.map_err(cache_err)?;
            Ok(path.display().to_string())
        })
    }

    fn __repr__(&self) -> String {
        format!("ModelCache(cache_dir={:?})", self.cache_dir())
    }
}
