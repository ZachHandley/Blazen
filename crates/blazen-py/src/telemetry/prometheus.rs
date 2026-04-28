//! Python binding for `blazen_telemetry::init_prometheus`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::error::BlazenPyError;

/// Initialize the Prometheus metrics exporter and start the HTTP listener.
///
/// Installs a global ``metrics`` recorder backed by Prometheus and starts an
/// HTTP server on ``0.0.0.0:{port}`` serving the ``/metrics`` endpoint. After
/// calling this, any code using the ``metrics`` macros (``counter!``,
/// ``histogram!``, ``gauge!``) will be exposed at the Prometheus endpoint.
///
/// Args:
///     port: TCP port to bind the metrics HTTP listener on.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn init_prometheus(port: u16) -> PyResult<()> {
    blazen_telemetry::init_prometheus(port)
        .map_err(|e| BlazenPyError::Workflow(format!("init_prometheus failed: {e}")).into())
}
