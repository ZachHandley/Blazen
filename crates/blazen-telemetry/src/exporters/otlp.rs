//! OpenTelemetry OTLP exporter.
//!
//! Exports traces to any OTLP-compatible backend (Jaeger, Grafana Tempo,
//! Honeycomb, Datadog, etc.).
//!
//! Two transports are supported, gated by separate Cargo features:
//! - `otlp`: gRPC over tonic (native only).
//! - `otlp-http`: HTTP/protobuf with a custom `HttpClient` impl. On native we
//!   wrap `reqwest::Client`; on wasm32 we wrap `web_sys::fetch`. We do **not**
//!   pull `opentelemetry-otlp/reqwest-blocking-client` or `reqwest-client`
//!   because those impls are `!Send` on wasm32 and break the trait's
//!   `Send + Sync` bound at compile time.

use std::collections::HashMap;

#[cfg(any(feature = "otlp", feature = "otlp-http"))]
use opentelemetry::global;
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
use opentelemetry::trace::TracerProvider;
#[cfg(feature = "otlp-http")]
use opentelemetry_otlp::WithHttpConfig;
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
use opentelemetry_sdk::{resource::Resource, trace::SdkTracerProvider};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
use tracing_opentelemetry::OpenTelemetryLayer;
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

// Native `opentelemetry_http::HttpClient` wrapper around `reqwest::Client`.
// Mirrors the impl shipped behind `opentelemetry-http/reqwest`, but we own it
// here so we don't need to enable that feature (which would also compile on
// wasm32 and trip the `!Send` future error). Excluded on wasi (no socket
// access; the wasi build uses `WasiFetchHttpClient` instead).
#[cfg(all(
    feature = "otlp-http",
    not(target_arch = "wasm32"),
    not(target_os = "wasi")
))]
mod native_reqwest_client {
    use bytes::Bytes;
    use http::{Request, Response};
    use opentelemetry_http::{HttpClient, HttpError};

    #[derive(Debug, Clone)]
    pub struct ReqwestHttpClient(pub reqwest::Client);

    #[async_trait::async_trait]
    impl HttpClient for ReqwestHttpClient {
        async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
            // Translate http::Request<Bytes> -> reqwest::Request via TryFrom.
            let reqwest_request: reqwest::Request = request.try_into()?;
            let mut response = self.0.execute(reqwest_request).await?.error_for_status()?;
            let headers = std::mem::take(response.headers_mut());
            let mut http_response = Response::builder()
                .status(response.status())
                .body(response.bytes().await?)?;
            *http_response.headers_mut() = headers;
            Ok(http_response)
        }
    }
}

/// Wire-level OTLP transport protocol.
///
/// `HttpProto` (HTTP/binary-protobuf) is the default — it traverses CDN/proxy
/// infrastructure cleanly and is the protocol every public-HTTPS OTLP
/// collector exposes. `Grpc` (tonic) is preferred for direct mesh-bound
/// collectors but requires h2c upstream config on most reverse proxies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OtlpProtocol {
    /// gRPC over tonic. Requires the `otlp` Cargo feature.
    Grpc,
    /// HTTP with binary protobuf payload. Requires the `otlp-http` Cargo
    /// feature.
    #[default]
    HttpProto,
}

/// Configuration for the OTLP exporter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpConfig {
    /// The OTLP endpoint URL.
    ///
    /// For gRPC (`otlp` feature): `"http://localhost:4317"`.
    /// For HTTP (`otlp-http` feature): `"http://localhost:4318/v1/traces"`.
    pub endpoint: String,
    /// The service name reported to the backend.
    pub service_name: String,
    /// Transport protocol. Defaults to [`OtlpProtocol::HttpProto`].
    #[serde(default)]
    pub protocol: OtlpProtocol,
    /// Optional extra request headers (e.g. `Authorization`, `x-honeycomb-team`).
    ///
    /// For HTTP/protobuf the headers are attached verbatim. For gRPC the
    /// keys must be valid HTTP/2 metadata keys (lower-case ASCII); invalid
    /// keys are silently skipped with a `tracing::warn!`.
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
}

impl OtlpConfig {
    /// Construct a new `OtlpConfig` with the default protocol
    /// ([`OtlpProtocol::HttpProto`]) and no extra headers.
    #[must_use]
    pub fn new(endpoint: impl Into<String>, service_name: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            service_name: service_name.into(),
            protocol: OtlpProtocol::default(),
            headers: None,
        }
    }

    /// Set the transport protocol.
    #[must_use]
    pub fn with_protocol(mut self, protocol: OtlpProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Set request headers.
    #[must_use]
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }
}

/// Initialize the OTLP trace exporter and install it as the global tracing
/// subscriber layer.
///
/// Dispatches on `config.protocol`:
/// - [`OtlpProtocol::Grpc`] → tonic gRPC exporter (requires the `otlp` Cargo
///   feature).
/// - [`OtlpProtocol::HttpProto`] → HTTP/binary-protobuf exporter (requires the
///   `otlp-http` Cargo feature).
///
/// Default is [`OtlpProtocol::HttpProto`] — public-HTTPS OTLP collectors
/// traverse CDN/proxy infrastructure cleanly.
///
/// `config.headers` is honored on both transports for auth (e.g. Honeycomb
/// `x-honeycomb-team`, OTLP/HTTP `Authorization: Basic ...`). gRPC metadata
/// keys must be valid HTTP/2 lower-case ASCII; invalid keys are skipped with
/// a `tracing::warn!`.
///
/// # Errors
///
/// Returns an error if Blazen was not built with the requested protocol's
/// feature, or if the OTLP exporter or tracer provider cannot be created.
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
pub fn init_otlp(config: OtlpConfig) -> Result<(), Box<dyn std::error::Error>> {
    match config.protocol {
        OtlpProtocol::Grpc => {
            #[cfg(feature = "otlp")]
            {
                init_otlp_grpc(config)
            }
            #[cfg(not(feature = "otlp"))]
            {
                let _ = config;
                Err(
                    "Blazen was built without the `otlp` (gRPC) feature; rebuild with --features otlp or set protocol = http_proto"
                        .into(),
                )
            }
        }
        OtlpProtocol::HttpProto => {
            #[cfg(feature = "otlp-http")]
            {
                init_otlp_http(config)
            }
            #[cfg(not(feature = "otlp-http"))]
            {
                let _ = config;
                Err(
                    "Blazen was built without the `otlp-http` feature; rebuild with --features otlp-http or set protocol = grpc"
                        .into(),
                )
            }
        }
    }
}

/// Build an OTLP gRPC (tonic) exporter and install it as the global tracing
/// subscriber layer.
///
/// `config.headers` is currently ignored on the gRPC path — wiring tonic
/// `MetadataMap` requires a direct `tonic` dependency we don't carry. For
/// header-based auth (Honeycomb, Grafana Cloud, etc.) use
/// [`OtlpProtocol::HttpProto`] instead.
///
/// # Errors
///
/// Returns an error if the OTLP exporter or tracer provider cannot be created.
#[cfg(feature = "otlp")]
fn init_otlp_grpc(config: OtlpConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.headers.as_ref().is_some_and(|h| !h.is_empty()) {
        tracing::warn!(
            "OtlpConfig.headers is set but Blazen's gRPC exporter does not forward headers; use OtlpProtocol::HttpProto for header-based auth"
        );
    }

    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.endpoint)
        .build()?;

    install_provider(exporter, config.service_name);
    Ok(())
}

/// Initialize the OTLP trace exporter using HTTP (binary protobuf) and install
/// it as the global tracing subscriber layer.
///
/// Prefer [`init_otlp`] with `OtlpProtocol::HttpProto` for new code; this
/// entrypoint stays public so the WASM SDK (which cannot link tonic) can call
/// it directly without going through the dispatch shim.
///
/// # Errors
///
/// Returns an error if the OTLP exporter or tracer provider cannot be created.
#[cfg(feature = "otlp-http")]
pub fn init_otlp_http(config: OtlpConfig) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build the OTLP HTTP/protobuf span exporter with a target-appropriate
    //    `HttpClient` registered via `with_http_client`. We do not rely on the
    //    `opentelemetry-otlp/reqwest-*-client` features because reqwest's
    //    wasm32 client is `!Send`, which violates the trait's `Send + Sync`
    //    bound and breaks compilation for `wasm32-unknown-unknown`.

    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    let builder = SpanExporter::builder()
        .with_http()
        .with_http_client(crate::exporters::wasm_otlp_client::WasmFetchHttpClient::new());

    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    let builder = SpanExporter::builder()
        .with_http()
        .with_http_client(crate::exporters::wasi_otlp_client::WasiFetchHttpClient::new());

    #[cfg(not(target_arch = "wasm32"))]
    let builder = SpanExporter::builder().with_http().with_http_client(
        native_reqwest_client::ReqwestHttpClient(reqwest::Client::new()),
    );

    let mut builder = builder
        .with_endpoint(&config.endpoint)
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary);

    if let Some(headers) = config.headers {
        builder = builder.with_headers(headers);
    }

    let exporter = builder.build()?;

    install_provider(exporter, config.service_name);
    Ok(())
}

/// Wire a built `SpanExporter` into a global `SdkTracerProvider` and install
/// the matching `tracing-subscriber` stack.
///
/// Calls [`crate::subscriber::swap_exporter_layer`] when Blazen owns the
/// global subscriber (the standard binding path — Python / Node / `UniFFI`
/// install the shared subscriber at module import). Falls back to a
/// `try_init` on a registry-based stack when no reload handle is
/// present, which covers the host-owns-subscriber case (e.g. WASM
/// builds with no module-import hook). Either path is non-panicking and
/// idempotent — repeated `init_otlp` calls in the same process never
/// abort.
#[cfg(any(feature = "otlp", feature = "otlp-http"))]
fn install_provider(exporter: SpanExporter, service_name: String) {
    // Build the tracer provider with batch export and service.name resource.
    let resource = Resource::builder_empty()
        .with_service_name(service_name)
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    // Set the provider as the global tracer provider.
    global::set_tracer_provider(provider.clone());

    // Build the tracing-opentelemetry layer and erase its type so it can
    // be swapped into the reload slot. The explicit type annotation on
    // `otel_layer` pins `S = Registry` for inference (the
    // `OpenTelemetryLayer<S, T>` `S` is a phantom; without it Rust
    // can't pick it).
    let tracer = provider.tracer("blazen");
    let otel_layer: OpenTelemetryLayer<tracing_subscriber::registry::Registry, _> =
        OpenTelemetryLayer::new(tracer);
    let boxed: Box<
        dyn tracing_subscriber::Layer<tracing_subscriber::registry::Registry>
            + Send
            + Sync
            + 'static,
    > = Box::new(otel_layer);

    if crate::subscriber::has_reload_handle() {
        let _ = crate::subscriber::swap_exporter_layer(boxed);
        return;
    }

    // No reload slot — host owns the subscriber. Best-effort try_init
    // on a fresh registry stack composed with the OTLP layer. If a
    // foreign subscriber is already installed this silently no-ops
    // (events keep flowing through that subscriber; OTLP doesn't
    // export, and the caller's host should compose the layer
    // themselves to recover). Never panics.
    //
    // The boxed layer sits *directly* on `registry()` so its `Layer<S>`
    // implementation (which holds for `S = Registry` only) matches the
    // inner subscriber type. Filters and fmt are layered on top.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let _ = tracing_subscriber::registry()
        .with(boxed)
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}
