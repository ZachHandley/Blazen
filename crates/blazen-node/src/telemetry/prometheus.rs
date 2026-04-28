//! Node binding for the Prometheus metrics exporter.
//!
//! Wraps [`blazen_telemetry::init_prometheus`] which installs a global
//! `metrics` recorder backed by Prometheus and starts an HTTP listener
//! serving `/metrics` on `0.0.0.0:{port}`.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::to_napi_error;

/// Initialize the Prometheus metrics exporter and start its HTTP listener
/// on `0.0.0.0:{port}`.
///
/// After calling this, every workflow / step / LLM span emitted via the
/// `tracing` infrastructure feeds into counters and histograms exposed at
/// `http://0.0.0.0:{port}/metrics` for Prometheus to scrape.
///
/// ```javascript
/// initPrometheus(9100);
/// ```
///
/// Calling this more than once in a single process will fail because the
/// global recorder can only be installed once.
#[napi(js_name = "initPrometheus")]
#[allow(clippy::missing_errors_doc)]
pub fn init_prometheus(port: u16) -> Result<()> {
    blazen_telemetry::init_prometheus(port).map_err(to_napi_error)
}
