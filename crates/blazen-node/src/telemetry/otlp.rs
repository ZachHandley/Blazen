//! Node binding for the OpenTelemetry OTLP exporter.
//!
//! Mirrors the Python and `UniFFI` surfaces: an `OtlpConfig` record selects
//! between gRPC (`Grpc`) and HTTP/protobuf (`HttpProto`, default) transports,
//! and forwards optional auth headers. Headers are honored on HTTP; gRPC
//! exports drop them with a `tracing::warn!` because Blazen does not carry a
//! direct `tonic` dep for metadata-map construction.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

use blazen_telemetry::{OtlpConfig, OtlpProtocol};

use crate::error::to_napi_error;

/// OTLP wire-level transport.
#[napi(string_enum, js_name = "OtlpProtocol")]
pub enum JsOtlpProtocol {
    /// gRPC over tonic. Requires the `otlp` Cargo feature.
    Grpc,
    /// HTTP with binary protobuf payload. Requires the `otlp-http` Cargo
    /// feature.
    HttpProto,
}

impl From<JsOtlpProtocol> for OtlpProtocol {
    fn from(p: JsOtlpProtocol) -> Self {
        match p {
            JsOtlpProtocol::Grpc => OtlpProtocol::Grpc,
            JsOtlpProtocol::HttpProto => OtlpProtocol::HttpProto,
        }
    }
}

/// Configuration for the OTLP exporter.
///
/// ```javascript
/// initOtlp({
///   endpoint: "https://otel.example.com/v1/traces",
///   serviceName: "my-service",
///   protocol: "HttpProto",
///   headers: { Authorization: "Bearer xxx" },
/// });
/// ```
#[napi(object, js_name = "OtlpConfig")]
pub struct JsOtlpConfig {
    /// The OTLP endpoint URL.
    ///
    /// For gRPC: `"http://localhost:4317"`.
    /// For HTTP: `"https://collector/v1/traces"`.
    pub endpoint: String,
    /// The service name reported to the backend.
    #[napi(js_name = "serviceName")]
    pub service_name: String,
    /// Wire-level transport. Defaults to `HttpProto`.
    pub protocol: Option<JsOtlpProtocol>,
    /// Service version (recorded for forward compatibility; not yet attached
    /// as a resource attribute by the underlying exporter).
    #[napi(js_name = "serviceVersion")]
    pub service_version: Option<String>,
    /// Auth / routing headers attached to OTLP requests. Honored on HTTP;
    /// dropped with a warning on gRPC.
    pub headers: Option<HashMap<String, String>>,
}

impl From<JsOtlpConfig> for OtlpConfig {
    fn from(c: JsOtlpConfig) -> Self {
        Self {
            endpoint: c.endpoint,
            service_name: c.service_name,
            protocol: c.protocol.map(Into::into).unwrap_or_default(),
            headers: c.headers,
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
