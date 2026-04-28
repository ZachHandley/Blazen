//! Python binding for `blazen_telemetry::OtlpConfig` and `init_otlp`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_telemetry::OtlpConfig;

use crate::error::BlazenPyError;

// ---------------------------------------------------------------------------
// PyOtlpConfig
// ---------------------------------------------------------------------------

/// Configuration for the OpenTelemetry OTLP exporter.
///
/// Example:
///     >>> from blazen import OtlpConfig, init_otlp
///     >>> cfg = OtlpConfig(endpoint="http://localhost:4317", service_name="my-app")
///     >>> init_otlp(cfg)
#[gen_stub_pyclass]
#[pyclass(name = "OtlpConfig", from_py_object)]
#[derive(Clone)]
pub struct PyOtlpConfig {
    pub(crate) inner: OtlpConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOtlpConfig {
    /// Create a new OTLP configuration.
    ///
    /// Args:
    ///     endpoint: The OTLP gRPC endpoint URL (e.g. ``"http://localhost:4317"``).
    ///     service_name: The service name reported to the backend.
    #[new]
    #[pyo3(signature = (*, endpoint, service_name))]
    fn new(endpoint: String, service_name: String) -> Self {
        Self {
            inner: OtlpConfig {
                endpoint,
                service_name,
            },
        }
    }

    #[getter]
    fn endpoint(&self) -> String {
        self.inner.endpoint.clone()
    }
    #[setter]
    fn set_endpoint(&mut self, v: String) {
        self.inner.endpoint = v;
    }

    #[getter]
    fn service_name(&self) -> String {
        self.inner.service_name.clone()
    }
    #[setter]
    fn set_service_name(&mut self, v: String) {
        self.inner.service_name = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "OtlpConfig(endpoint={}, service_name={})",
            self.inner.endpoint, self.inner.service_name
        )
    }
}

// ---------------------------------------------------------------------------
// init_otlp
// ---------------------------------------------------------------------------

/// Initialize the OTLP trace exporter and install it as the global tracing
/// subscriber layer.
///
/// Sets up an OTLP gRPC span exporter pointed at ``config.endpoint`` and
/// installs a combined ``tracing`` subscriber (env-filter + OTel layer +
/// fmt layer). Call this once at process startup, before any traced work.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn init_otlp(config: &PyOtlpConfig) -> PyResult<()> {
    blazen_telemetry::init_otlp(config.inner.clone())
        .map_err(|e| BlazenPyError::Workflow(format!("init_otlp failed: {e}")).into())
}
