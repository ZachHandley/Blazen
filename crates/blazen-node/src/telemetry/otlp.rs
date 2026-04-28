//! Node binding for the OpenTelemetry OTLP exporter.
//!
//! The underlying [`blazen_telemetry::init_otlp`] takes a config struct
//! with an endpoint and a service name, and installs a global tracing
//! subscriber that exports spans over OTLP/gRPC. The service version and
//! header fields surfaced here are accepted for forward compatibility
//! and may be wired through once the upstream config gains them; today
//! they are recorded but not forwarded.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

use blazen_telemetry::OtlpConfig;

use crate::error::to_napi_error;

/// Configuration for the OTLP exporter.
///
/// ```javascript
/// initOtlp({
///   endpoint: "http://localhost:4317",
///   serviceName: "my-service",
///   serviceVersion: "1.0.0",
///   headers: { "x-api-key": "secret" },
/// });
/// ```
#[napi(object, js_name = "OtlpConfig")]
pub struct JsOtlpConfig {
    /// The OTLP endpoint URL (e.g. `"http://localhost:4317"`).
    pub endpoint: String,
    /// The service name reported to the backend.
    #[napi(js_name = "serviceName")]
    pub service_name: String,
    /// Service version reported to the backend (recorded for forward
    /// compatibility; not yet forwarded by the underlying exporter).
    #[napi(js_name = "serviceVersion")]
    pub service_version: Option<String>,
    /// Additional headers to attach to OTLP requests (recorded for forward
    /// compatibility; not yet forwarded by the underlying exporter).
    pub headers: Option<HashMap<String, String>>,
}

impl From<JsOtlpConfig> for OtlpConfig {
    fn from(c: JsOtlpConfig) -> Self {
        Self {
            endpoint: c.endpoint,
            service_name: c.service_name,
        }
    }
}

/// Initialize the OTLP trace exporter and install the global tracing
/// subscriber.
///
/// Calling this more than once in a single process will fail because the
/// global subscriber can only be installed once.
#[napi(js_name = "initOtlp")]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::missing_errors_doc)]
pub fn init_otlp(config: JsOtlpConfig) -> Result<()> {
    blazen_telemetry::init_otlp(config.into()).map_err(to_napi_error)
}
