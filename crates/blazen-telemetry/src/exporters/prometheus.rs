//! Prometheus metrics exporter.
//!
//! Exposes a `/metrics` HTTP endpoint for Prometheus scraping with counters,
//! histograms, and gauges for workflow execution telemetry.

use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;

/// Initialize the Prometheus metrics exporter and start the HTTP listener.
///
/// This installs a global `metrics` recorder backed by Prometheus and starts
/// an HTTP server on `0.0.0.0:{port}` that serves the `/metrics` endpoint.
///
/// After calling this function, any code using the `metrics` crate macros
/// (`counter!`, `histogram!`, `gauge!`) will have its data collected and
/// exposed at the Prometheus endpoint.
///
/// # Errors
///
/// Returns an error if the HTTP listener cannot be bound or the recorder
/// cannot be installed (e.g. a global recorder is already set).
pub fn init_prometheus(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();

    PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()?;

    Ok(())
}
