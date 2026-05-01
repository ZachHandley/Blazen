//! Opaque Python handle around `Arc<dyn HttpClient>` for the provider
//! `http_client()` escape hatch. Cannot be constructed from Python; only
//! returned by provider accessors.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::http::HttpClient;

/// Opaque handle around an `Arc<dyn HttpClient>` returned by
/// `*Provider.http_client()`. Use it to pass the underlying client to
/// other providers or to inspect its config.
///
/// This class cannot be constructed directly from Python — instances are
/// produced by the `http_client()` accessor on each provider class.
#[gen_stub_pyclass]
#[pyclass(name = "HttpClientHandle", module = "blazen", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyHttpClientHandle {
    pub(crate) inner: Arc<dyn HttpClient>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHttpClientHandle {
    fn __repr__(&self) -> String {
        format!("HttpClientHandle({:?})", self.inner.config())
    }
}
