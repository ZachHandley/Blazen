//! Python binding for `blazen_telemetry::OtlpConfig` and `init_otlp`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{
    gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pyfunction, gen_stub_pymethods,
};

use blazen_telemetry::{OtlpConfig, OtlpProtocol};

use crate::error::BlazenPyError;

// ---------------------------------------------------------------------------
// PyOtlpProtocol
// ---------------------------------------------------------------------------

/// OTLP wire-level transport.
///
/// ``HTTP_PROTO`` (HTTP / binary-protobuf) is the default — it traverses
/// public HTTPS / CDN infrastructure cleanly and is what every managed OTLP
/// collector exposes. ``GRPC`` (tonic) is preferred for mesh-bound collectors
/// but requires h2c upstream config on most reverse proxies.
#[gen_stub_pyclass_enum]
#[pyclass(name = "OtlpProtocol", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyOtlpProtocol {
    Grpc,
    HttpProto,
}

impl From<PyOtlpProtocol> for OtlpProtocol {
    fn from(p: PyOtlpProtocol) -> Self {
        match p {
            PyOtlpProtocol::Grpc => OtlpProtocol::Grpc,
            PyOtlpProtocol::HttpProto => OtlpProtocol::HttpProto,
        }
    }
}

impl From<OtlpProtocol> for PyOtlpProtocol {
    fn from(p: OtlpProtocol) -> Self {
        match p {
            OtlpProtocol::Grpc => PyOtlpProtocol::Grpc,
            OtlpProtocol::HttpProto => PyOtlpProtocol::HttpProto,
        }
    }
}

// ---------------------------------------------------------------------------
// PyOtlpConfig
// ---------------------------------------------------------------------------

/// Configuration for the OpenTelemetry OTLP exporter.
///
/// Example:
/// ```text
///  >>> from blazen import OtlpConfig, OtlpProtocol, init_otlp
///  >>> cfg = OtlpConfig(
///  ...     endpoint="https://otel.example.com/v1/traces",
///  ...     service_name="my-app",
///  ...     protocol=OtlpProtocol.HttpProto,
///  ...     headers={"Authorization": "Bearer xxx"},
///  ... )
///  >>> init_otlp(cfg)
/// ```
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
    ///     endpoint: The OTLP endpoint URL. For HTTP/protobuf use
    ///         ``"https://collector/v1/traces"``; for gRPC use
    ///         ``"http://collector:4317"``.
    ///     service_name: The service name reported to the backend.
    ///     protocol: Wire-level transport. Defaults to
    ///         ``OtlpProtocol.HttpProto``.
    ///     headers: Optional auth / routing headers. Honored on HTTP;
    ///         currently ignored on gRPC (use HTTP for header-based auth).
    #[new]
    #[pyo3(signature = (*, endpoint, service_name, protocol=None, headers=None))]
    fn new(
        endpoint: String,
        service_name: String,
        protocol: Option<PyOtlpProtocol>,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            inner: OtlpConfig {
                endpoint,
                service_name,
                protocol: protocol.map(Into::into).unwrap_or_default(),
                headers,
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

    #[getter]
    fn protocol(&self) -> PyOtlpProtocol {
        self.inner.protocol.into()
    }
    #[setter]
    fn set_protocol(&mut self, v: PyOtlpProtocol) {
        self.inner.protocol = v.into();
    }

    #[getter]
    fn headers(&self) -> Option<HashMap<String, String>> {
        self.inner.headers.clone()
    }
    #[setter]
    fn set_headers(&mut self, v: Option<HashMap<String, String>>) {
        self.inner.headers = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "OtlpConfig(endpoint={}, service_name={}, protocol={:?})",
            self.inner.endpoint, self.inner.service_name, self.inner.protocol
        )
    }
}

// ---------------------------------------------------------------------------
// init_otlp
// ---------------------------------------------------------------------------

/// Initialize the OTLP trace exporter and install it as the global tracing
/// subscriber layer.
///
/// Dispatches on ``config.protocol``: ``HTTP_PROTO`` uses the HTTP/binary-
/// protobuf exporter (requires the ``otlp-http`` Cargo feature); ``GRPC``
/// uses tonic (requires the ``otlp`` Cargo feature). Call this once at
/// process startup, before any traced work.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn init_otlp(config: &PyOtlpConfig) -> PyResult<()> {
    blazen_telemetry::init_otlp(config.inner.clone())
        .map_err(|e| BlazenPyError::Workflow(format!("init_otlp failed: {e}")).into())
}
