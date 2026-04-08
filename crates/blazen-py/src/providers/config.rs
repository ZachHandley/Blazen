//! Typed config wrappers for retry and cache decorators.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::cache::{CacheConfig, CacheStrategy};
use blazen_llm::retry::RetryConfig;

// ---------------------------------------------------------------------------
// CacheStrategy enum
// ---------------------------------------------------------------------------

#[gen_stub_pyclass_enum]
#[pyclass(name = "CacheStrategy", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyCacheStrategy {
    None,
    ContentHash,
    AnthropicEphemeral,
    Auto,
}

impl From<PyCacheStrategy> for CacheStrategy {
    fn from(s: PyCacheStrategy) -> Self {
        match s {
            PyCacheStrategy::None => Self::None,
            PyCacheStrategy::ContentHash => Self::ContentHash,
            PyCacheStrategy::AnthropicEphemeral => Self::AnthropicEphemeral,
            PyCacheStrategy::Auto => Self::Auto,
        }
    }
}

impl From<CacheStrategy> for PyCacheStrategy {
    fn from(s: CacheStrategy) -> Self {
        match s {
            CacheStrategy::None => Self::None,
            CacheStrategy::ContentHash => Self::ContentHash,
            CacheStrategy::AnthropicEphemeral => Self::AnthropicEphemeral,
            CacheStrategy::Auto => Self::Auto,
        }
    }
}

// ---------------------------------------------------------------------------
// RetryConfig wrapper
// ---------------------------------------------------------------------------

#[gen_stub_pyclass]
#[pyclass(name = "RetryConfig", from_py_object)]
#[derive(Clone, Default)]
pub struct PyRetryConfig {
    pub(crate) inner: RetryConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryConfig {
    /// Create a new RetryConfig.
    ///
    /// Args:
    ///     max_retries: Maximum retry attempts (default: 3).
    ///     initial_delay_ms: Delay before first retry (default: 1000).
    ///     max_delay_ms: Maximum backoff delay (default: 30000).
    ///     honor_retry_after: Honor server retry_after (default: True).
    ///     jitter: Add random jitter (default: True).
    #[new]
    #[pyo3(signature = (*, max_retries=3, initial_delay_ms=1000, max_delay_ms=30_000, honor_retry_after=true, jitter=true))]
    fn new(
        max_retries: u32,
        initial_delay_ms: u64,
        max_delay_ms: u64,
        honor_retry_after: bool,
        jitter: bool,
    ) -> Self {
        Self {
            inner: RetryConfig {
                max_retries,
                initial_delay_ms,
                max_delay_ms,
                honor_retry_after,
                jitter,
            },
        }
    }

    #[getter]
    fn max_retries(&self) -> u32 {
        self.inner.max_retries
    }
    #[setter]
    fn set_max_retries(&mut self, v: u32) {
        self.inner.max_retries = v;
    }

    #[getter]
    fn initial_delay_ms(&self) -> u64 {
        self.inner.initial_delay_ms
    }
    #[setter]
    fn set_initial_delay_ms(&mut self, v: u64) {
        self.inner.initial_delay_ms = v;
    }

    #[getter]
    fn max_delay_ms(&self) -> u64 {
        self.inner.max_delay_ms
    }
    #[setter]
    fn set_max_delay_ms(&mut self, v: u64) {
        self.inner.max_delay_ms = v;
    }

    #[getter]
    fn honor_retry_after(&self) -> bool {
        self.inner.honor_retry_after
    }
    #[setter]
    fn set_honor_retry_after(&mut self, v: bool) {
        self.inner.honor_retry_after = v;
    }

    #[getter]
    fn jitter(&self) -> bool {
        self.inner.jitter
    }
    #[setter]
    fn set_jitter(&mut self, v: bool) {
        self.inner.jitter = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryConfig(max_retries={}, initial_delay_ms={}, max_delay_ms={})",
            self.inner.max_retries, self.inner.initial_delay_ms, self.inner.max_delay_ms
        )
    }
}

// ---------------------------------------------------------------------------
// CacheConfig wrapper
// ---------------------------------------------------------------------------

#[gen_stub_pyclass]
#[pyclass(name = "CacheConfig", from_py_object)]
#[derive(Clone, Default)]
pub struct PyCacheConfig {
    pub(crate) inner: CacheConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCacheConfig {
    /// Create a new CacheConfig.
    ///
    /// Args:
    ///     strategy: Caching strategy (default: CacheStrategy.ContentHash).
    ///     ttl_seconds: How long a cached response remains valid (default: 300).
    ///     max_entries: Maximum number of cache entries (default: 1000).
    #[new]
    #[pyo3(signature = (*, strategy=None, ttl_seconds=300, max_entries=1000))]
    fn new(strategy: Option<PyCacheStrategy>, ttl_seconds: u64, max_entries: usize) -> Self {
        Self {
            inner: CacheConfig {
                strategy: strategy.unwrap_or(PyCacheStrategy::ContentHash).into(),
                ttl_seconds,
                max_entries,
            },
        }
    }

    #[getter]
    fn strategy(&self) -> PyCacheStrategy {
        self.inner.strategy.clone().into()
    }
    #[setter]
    fn set_strategy(&mut self, v: PyCacheStrategy) {
        self.inner.strategy = v.into();
    }

    #[getter]
    fn ttl_seconds(&self) -> u64 {
        self.inner.ttl_seconds
    }
    #[setter]
    fn set_ttl_seconds(&mut self, v: u64) {
        self.inner.ttl_seconds = v;
    }

    #[getter]
    fn max_entries(&self) -> usize {
        self.inner.max_entries
    }
    #[setter]
    fn set_max_entries(&mut self, v: usize) {
        self.inner.max_entries = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheConfig(strategy={:?}, ttl_seconds={}, max_entries={})",
            self.inner.strategy, self.inner.ttl_seconds, self.inner.max_entries
        )
    }
}
