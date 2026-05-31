//! `wasm-bindgen` wrappers for [`blazen_telemetry::OtlpConfig`] and
//! [`blazen_telemetry::init_otlp_http`].
//!
//! Exposes the OTLP HTTP/protobuf trace exporter to JS/TS callers. The gRPC
//! variant (`init_otlp`) is intentionally not bound — `tonic` does not
//! compile to `wasm32`.
//!
//! ```js
//! import { OtlpConfig, initOtlp } from '@blazen-dev/wasm';
//!
//! const cfg = new OtlpConfig(
//!   'https://otel-collector.example.com/v1/traces',
//!   'my-service',
//! );
//! initOtlp(cfg);
//! ```

use blazen_telemetry::{OtlpConfig, OtlpProtocol};
use wasm_bindgen::prelude::*;

/// Wire-level OTLP transport protocol.
///
/// Mirrors [`blazen_telemetry::OtlpProtocol`]. On `wasm32` only
/// [`WasmOtlpProtocol::HttpProto`] is functional — the `Grpc` variant is
/// surfaced for parity with the native bindings but selecting it has no
/// effect because `tonic` does not compile to `wasm32`; the exporter always
/// uses HTTP/protobuf transport.
#[wasm_bindgen(js_name = "OtlpProtocol")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmOtlpProtocol {
    /// gRPC over tonic. Native-only; not functional on `wasm32`.
    Grpc,
    /// HTTP with binary protobuf payload. The `wasm32` default.
    HttpProto,
}

impl From<WasmOtlpProtocol> for OtlpProtocol {
    fn from(value: WasmOtlpProtocol) -> Self {
        match value {
            WasmOtlpProtocol::Grpc => OtlpProtocol::Grpc,
            WasmOtlpProtocol::HttpProto => OtlpProtocol::HttpProto,
        }
    }
}

impl From<OtlpProtocol> for WasmOtlpProtocol {
    fn from(value: OtlpProtocol) -> Self {
        match value {
            OtlpProtocol::Grpc => WasmOtlpProtocol::Grpc,
            OtlpProtocol::HttpProto => WasmOtlpProtocol::HttpProto,
        }
    }
}

/// Configuration for the OTLP trace exporter.
///
/// Wraps [`blazen_telemetry::OtlpConfig`] for `wasm-bindgen` interop.
#[wasm_bindgen(js_name = "OtlpConfig")]
pub struct WasmOtlpConfig {
    inner: OtlpConfig,
}

#[wasm_bindgen(js_class = "OtlpConfig")]
impl WasmOtlpConfig {
    /// Create a new OTLP exporter configuration.
    ///
    /// `endpoint` should be the full HTTP/protobuf traces endpoint, e.g.
    /// `"http://localhost:4318/v1/traces"`.
    ///
    /// `service_name` is reported to the backend as the
    /// `service.name` resource attribute.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(endpoint: String, service_name: String) -> Self {
        // WASM always exports over HTTP/protobuf (tonic does not compile to
        // wasm32), so the protocol field is left at its `HttpProto` default.
        Self {
            inner: OtlpConfig::new(endpoint, service_name),
        }
    }

    /// The configured OTLP endpoint URL.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.inner.endpoint.clone()
    }

    /// The configured service name.
    #[wasm_bindgen(getter, js_name = "serviceName")]
    #[must_use]
    pub fn service_name(&self) -> String {
        self.inner.service_name.clone()
    }

    /// The configured transport protocol.
    ///
    /// On `wasm32` this is always [`WasmOtlpProtocol::HttpProto`] in practice
    /// (tonic gRPC does not compile to `wasm32`), but the field is honoured so
    /// JS callers can inspect / set it for parity with the native bindings.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn protocol(&self) -> WasmOtlpProtocol {
        self.inner.protocol.into()
    }

    /// Set the transport protocol.
    ///
    /// Selecting [`WasmOtlpProtocol::Grpc`] has no functional effect on
    /// `wasm32` — the exporter always uses HTTP/protobuf — but it is accepted
    /// for parity with the native bindings.
    #[wasm_bindgen(js_name = "withProtocol")]
    pub fn with_protocol(&mut self, protocol: WasmOtlpProtocol) {
        self.inner.protocol = protocol.into();
    }

    /// Set auth / routing headers attached to every OTLP/HTTP request.
    ///
    /// Pass a plain JS object of string → string pairs, e.g.
    /// `cfg.setHeaders({ Authorization: "Bearer xxx" })`. Pass `undefined`
    /// or `null` to clear.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `headers` is not a string → string object.
    #[wasm_bindgen(js_name = "setHeaders")]
    pub fn set_headers(&mut self, headers: JsValue) -> Result<(), JsValue> {
        if headers.is_undefined() || headers.is_null() {
            self.inner.headers = None;
            return Ok(());
        }
        let map: std::collections::HashMap<String, String> =
            serde_wasm_bindgen::from_value(headers)
                .map_err(|e| JsValue::from_str(&format!("[BlazenError] setHeaders: {e}")))?;
        self.inner.headers = Some(map);
        Ok(())
    }
}

/// Initialise the global OTLP HTTP/protobuf trace exporter.
///
/// Installs a `tracing-subscriber` stack with an OpenTelemetry layer that
/// exports spans over HTTP to the configured OTLP collector. Must be called
/// once at startup; subsequent calls will fail because the global subscriber
/// can only be installed a single time.
///
/// # Errors
///
/// Returns a JS error if the exporter or tracer provider cannot be built,
/// or if a global subscriber has already been installed.
#[wasm_bindgen(js_name = "initOtlp")]
pub fn init_otlp(config: &WasmOtlpConfig) -> Result<(), JsValue> {
    blazen_telemetry::init_otlp_http(config.inner.clone())
        .map_err(|e| JsValue::from_str(&format!("[BlazenError] init_otlp_http: {e}")))
}
